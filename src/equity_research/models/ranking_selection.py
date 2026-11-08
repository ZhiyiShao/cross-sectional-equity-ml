"""Round 3: fixed-budget LambdaRank using preserved round-2 inputs and states."""
import json
from datetime import datetime, timezone
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from equity_research.backtest.market import ROOT
from equity_research.models.context_selection import period_mask, portfolio_compare
from equity_research.models.common import sha, daily_rank_ic, save_lightgbm, circular_block_interval
from equity_research.models.preprocessing import require, cross_sectional_rank, transform_features, frame_digest

SOURCE=ROOT/'outputs/context_selection_v2'
PREVIOUS=ROOT/'results/context_selection_v2'
OUT=ROOT/'outputs/ranking_selection_v2'
PUBLIC=ROOT/'results/ranking_selection_v2'
CONFIG=ROOT/'config/ranking_v2.json'


def ranking_labels(frame):
    """Integer relevance and contiguous query lengths; input already label-qualified."""
    require(frame.signal_date.is_monotonic_increasing,'Queries must be sorted')
    require(np.isfinite(frame.y_excess_5d).all(),'Nonfinite ranking target')
    grouped=frame.groupby('signal_date',sort=False).y_excess_5d
    rank=grouped.rank(method='average');n=grouped.transform('size')
    y=np.minimum(9,np.floor(10*(rank-.5)/n)).astype('int32')
    sizes=grouped.size().to_numpy(dtype='int32')
    require(int(sizes.sum())==len(frame),'Invalid query lengths')
    return y,sizes


def matrix(frame,ranked,state):
    x=transform_features(ranked,state['base'])
    if 'context' in state:
        x=pd.concat([x,transform_features(frame[state['context']['features']],state['context'])],axis=1)
    require(x.columns.tolist()==state['columns'],'Feature order changed')
    return x


def ndcg_top(labels,scores,ids):
    k=int(np.ceil(len(labels)/10));discount=1/np.log2(np.arange(k)+2)
    order=np.lexsort((np.asarray(ids),-np.asarray(scores)))[:k]
    ideal=np.sort(np.asarray(labels))[::-1][:k]@discount
    return float(np.asarray(labels)[order]@discount/ideal) if ideal>0 else np.nan


def main():
    require(not OUT.exists() and not PUBLIC.exists(),'Preserve existing experiment')
    cfg=json.loads(CONFIG.read_text());parent=json.loads((ROOT/cfg['parent']).read_text())
    context=json.loads((ROOT/cfg['context_config']).read_text())
    prior=json.loads((SOURCE/'verification.json').read_text())
    require(prior['fits']==32 and prior['remaining_candidate_budget']==16,'Reserved budget mismatch')
    metadata={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
        for folder in ['outputs','results'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    paths=[CONFIG,ROOT/cfg['parent'],ROOT/cfg['context_config'],SOURCE/'development_features_local.parquet',
        ROOT/'src/equity_research/models/ranking_selection.py',ROOT/'src/equity_research/models/context_selection.py',
        ROOT/'src/equity_research/backtest/risk_comparison.py',ROOT/'src/equity_research/backtest/engine.py']
    paths+=list((SOURCE/'preprocessing').glob('*lightgbm*.json'))
    paths+=list(PREVIOUS.glob('*.csv'))
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    OUT.mkdir();(OUT/'models').mkdir();(OUT/'preprocessing').mkdir()
    write=lambda p,x:p.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
    write(OUT/'frozen_plan.json',dict(config=cfg,parent=parent,context=context,sha256=hashes,
        started_utc=datetime.now(timezone.utc).isoformat(),lightgbm_version=lgb.__version__))
    f=pd.read_parquet(SOURCE/'development_features_local.parquet',filters=[('signal_date','<',pd.Timestamp(cfg['data_end_exclusive']))])
    require(frame_digest(f)==prior['feature_digest'],'Round2 feature snapshot changed')
    require(f.signal_date.min()>=pd.Timestamp('2015-01-01') and f.signal_date.max()<pd.Timestamp('2022-01-01'),'Date breach')
    features=json.loads((ROOT/'config/features_labels.json').read_text())['features']
    ranked=cross_sectional_rank(f,features)
    ledger=[];metrics=[];daily=[];predictions={}
    for wi,window in enumerate(parent['development_windows'],1):
        train=period_mask(f,window['train']);evaluate=period_mask(f,window['evaluate'])
        require(f.loc[train,'label_end_date'].max()<pd.Timestamp(window['evaluate'][0]),'Label boundary')
        y,groups=ranking_labels(f.loc[train]);require((groups>1).all(),'Singleton training query')
        ec=f.signal_date.between(*window['evaluate'])
        pred=f.loc[ec,['PERMNO','signal_date','signal_eligible']].copy()
        f.loc[train,['PERMNO','signal_date','label_end_date','y_excess_5d']].assign(relevance=y).to_parquet(OUT/f'w{wi}_training_labels_local.parquet',index=False)
        for variant in ['original','enhanced']:
            design='ranker_'+variant
            statepath=SOURCE/'preprocessing'/f'w{wi}_lightgbm_{variant}.json'
            state=json.loads(statepath.read_text());X=matrix(f,ranked,state)
            (OUT/'preprocessing'/f'w{wi}_{design}.json').write_bytes(statepath.read_bytes())
            require(state['base']['fitted_rows']==int(train.sum()),'Preprocessing population mismatch')
            for ci,leaves in enumerate(cfg['num_leaves']):
                settings={**context['lightgbm_fixed'],**cfg['ranking_fixed'],'num_leaves':leaves}
                model=lgb.LGBMRanker(**settings);require(len(ledger)<16,'Fit budget exceeded')
                with threadpool_limits(limits=4):
                    model.fit(X.loc[train],y,group=groups)
                    pred[f'{design}_{ci}']=model.predict(X.loc[ec])
                save_lightgbm(model,OUT/'models'/f'w{wi}_{design}_{ci}.txt')
                ic=daily_rank_ic(f.loc[evaluate,'signal_date'],pred.loc[evaluate,f'{design}_{ci}'],f.loc[evaluate,'y_excess_5d'])
                require(np.isfinite(ic).all(),'Undefined IC')
                metrics.append(dict(window=wi,design=design,candidate=ci,parameter=leaves,rank_ic=ic.mean(),dates=len(ic),rows=int(evaluate.sum())))
                daily.append(ic.rename('rank_ic').rename_axis('signal_date').reset_index().assign(window=wi,design=design,candidate=ci))
                ledger.append(dict(fit=len(ledger)+1,window=wi,design=design,candidate=ci,num_leaves=leaves,
                    train_rows=int(train.sum()),queries=len(groups),query_min=int(groups.min()),query_max=int(groups.max()),
                    max_train_signal=str(f.loc[train,'signal_date'].max()),max_train_label=str(f.loc[train,'label_end_date'].max()),
                    eval_start=window['evaluate'][0],preprocessing_sha256=sha(statepath)))
                write(OUT/'fit_ledger.json',ledger);print(f'Fit {len(ledger)}/16: w{wi} {design} leaves={leaves}',flush=True)
        pred.to_parquet(OUT/f'w{wi}_predictions_local.parquet',index=False);predictions[wi]=pred
    require(len(ledger)==16,'Incomplete fit budget')
    PUBLIC.mkdir();metrics=pd.DataFrame(metrics);daily=pd.concat(daily,ignore_index=True)
    ranking=metrics.groupby(['design','candidate','parameter'],as_index=False).rank_ic.mean()
    selected=ranking.sort_values(['design','rank_ic','candidate'],ascending=[True,False,True]).groupby('design').head(1)
    for name,t in [('candidate_metrics',metrics),('candidate_ranking',ranking),('selection',selected)]:t.to_csv(PUBLIC/(name+'.csv'),index=False)
    daily.to_csv(OUT/'candidate_daily_ic.csv',index=False);write(OUT/'selected.json',selected.to_dict('records'))
    chosen=pd.concat([daily[daily.design.eq(r.design)&daily.candidate.eq(r.candidate)] for r in selected.itertuples()])
    chosen.to_csv(PUBLIC/'selected_daily_ic.csv',index=False)
    # Selection is final before any portfolio return is calculated.
    portfolio_compare(predictions[2],selected,f,parent,output_dir=OUT,public_dir=PUBLIC)
    compare(f,selected,chosen,metrics,parent)
    for p,info in metadata.items():require([(ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns]==info,'Previous artifact changed: '+p)
    for p,digest in hashes.items():require(sha(ROOT/p)==digest,'Frozen input changed: '+p)
    audit=dict(status='completed',fits=16,total_v2_candidate_fits=48,remaining_candidate_budget=0,
        final_refits=0,test_rows_read=0,test_predictions=0,test_metrics_used=False,
        previous_artifacts_unchanged=True,source_hashes=hashes,lightgbm_version=lgb.__version__)
    write(OUT/'verification.json',audit);write(PUBLIC/'verification.json',audit)
    write(PUBLIC/'frozen_plan.json',dict(config=cfg,parent=parent,context=context))
    print(selected.to_string(index=False),flush=True)


def compare(f,selected,chosen,metrics,parent):
    """Saved regression controls, matched-leaf deltas, IC uncertainty and top-tail diagnostics."""
    old_selected=pd.read_csv(PREVIOUS/'selection.csv');old_daily=pd.read_csv(PREVIOUS/'selected_daily_ic.csv',parse_dates=['signal_date'])
    old_metrics=pd.read_csv(PREVIOUS/'candidate_metrics.csv')
    all_daily=pd.concat([old_daily,chosen],ignore_index=True)
    all_daily['year']=all_daily.signal_date.dt.year
    all_daily.groupby(['design','year']).rank_ic.agg(['mean','count']).to_csv(PUBLIC/'annual_ic.csv')
    all_daily.groupby(['design','window']).rank_ic.mean().to_csv(PUBLIC/'window_ic.csv')
    all_selected=pd.concat([old_selected,selected]);all_selected.to_csv(PUBLIC/'all_selection.csv',index=False)
    for name in ['portfolio_metrics','annual_returns']:
        pd.concat([pd.read_csv(PREVIOUS/(name+'.csv')),pd.read_csv(PUBLIC/(name+'.csv'))]).to_csv(PUBLIC/('comparison_'+name+'.csv'),index=False)
    matched=[];pairs=[];equal=[]
    for variant in ['original','enhanced']:
        a=old_metrics[old_metrics.design.eq('lightgbm_'+variant)]
        b=metrics[metrics.design.eq('ranker_'+variant)]
        z=a.merge(b,on=['window','candidate','parameter'],suffixes=('_regression','_ranking'))
        z['delta']=z.rank_ic_ranking-z.rank_ic_regression;z['variant']=variant;matched.append(z)
        draws=[];means=[];rng=np.random.default_rng(20261107)
        for wi in [1,2]:
            wide=all_daily[all_daily.window.eq(wi)].pivot(index='signal_date',columns='design',values='rank_ic')
            delta=(wide['ranker_'+variant]-wide['lightgbm_'+variant]).to_numpy()
            pairs.append(dict(variant=variant,window=wi,**circular_block_interval(delta,10,2000,20261107)))
            n=len(delta);starts=rng.integers(n,size=(2000,int(np.ceil(n/10))))
            ix=(starts[:,:,None]+np.arange(10))%n;draws.append(delta[ix.reshape(2000,-1)[:,:n]].mean(axis=1));means.append(delta.mean())
        lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975]);equal.append(dict(variant=variant,delta=np.mean(means),ci_low=lo,ci_high=hi))
    for name,rows in [('paired_ic',pairs),('paired_equal_window_ic',equal)]:pd.DataFrame(rows).to_csv(PUBLIC/(name+'.csv'),index=False)
    pd.concat(matched).to_csv(PUBLIC/'matched_leaf_deltas.csv',index=False)
    diagnostics=[]
    for wi,window in enumerate(parent['development_windows'],1):
        pred=pd.read_parquet(OUT/f'w{wi}_predictions_local.parquet').merge(
            pd.read_parquet(SOURCE/f'w{wi}_predictions_local.parquet'),on=['PERMNO','signal_date','signal_eligible'],validate='one_to_one')
        g=f.merge(pred,on=['PERMNO','signal_date','signal_eligible'],validate='one_to_one')
        qualified=period_mask(g,window['evaluate'])
        for r in all_selected.itertuples():
            name=f'{r.design}_{r.candidate}'
            ranks=g[name].where(g.signal_eligible).groupby(g.signal_date).rank(method='first',ascending=False)
            count=g.signal_eligible.groupby(g.signal_date).transform('sum')
            top=ranks.le(np.ceil(count/10))
            for date,part in g[qualified].groupby('signal_date'):
                relevance,_=ranking_labels(part)
                diagnostics.append(dict(window=wi,design=r.design,signal_date=date,
                    ndcg_top10=ndcg_top(relevance,part[name],part.PERMNO),
                    top10_mean_label=part.loc[top.loc[part.index],'y_excess_5d'].mean(),
                    top10_valid_count=int(top.loc[part.index].sum()),valid_population=len(part)))
    diag=pd.DataFrame(diagnostics);diag.to_csv(PUBLIC/'daily_ranking_diagnostics.csv',index=False)
    diag.groupby(['design','window'])[['ndcg_top10','top10_mean_label','top10_valid_count','valid_population']].mean().to_csv(PUBLIC/'ranking_diagnostics.csv')
