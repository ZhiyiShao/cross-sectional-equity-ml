"""Frozen V2 context experiment: 32 fits, two past-only windows, no test reads."""
import json
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from equity_research.backtest import market as base, validation
from equity_research.backtest.risk_comparison import risk_inputs, optimize, exposure_row
from equity_research.models.common import sha, daily_rank_ic, circular_block_interval, save_lightgbm
from equity_research.models.preprocessing import (cross_sectional_rank, fit_preprocessing,
    transform_features, fit_state, require, frame_digest)

ROOT=base.ROOT
CONFIG=ROOT/'config/context_features_v2.json'
OUT=ROOT/'outputs/context_selection_v2'
PUBLIC=ROOT/'results/context_selection_v2'


def contexts(frame,minimum=5):
    """Contemporaneous aggregates never condition on future label availability."""
    f=frame.copy(); good=f.signal_eligible
    source=f[['momentum_20','momentum_60','volatility_20']].replace([np.inf,-np.inf],np.nan).where(good,axis=0)
    dates=source.groupby(f.signal_date)
    group=[f.signal_date,f.industry]
    for window in [20,60]:
        s=source[f'momentum_{window}'];groups=s.groupby(group)
        mean=groups.transform('mean');count=groups.transform('count')
        fallback=s.groupby(f.signal_date).transform('mean')
        f[f'industry_mom{window}']=mean.where(count.ge(minimum)&f.industry.ne('Unknown'),fallback)
        f[f'relative_mom{window}']=s-f[f'industry_mom{window}']
    s=source.momentum_20;groups=s.groupby(group);n=groups.transform('count')
    ranks=(groups.rank(method='average')-.5)/n-.5
    allgroups=s.groupby(f.signal_date)
    fallback=(allgroups.rank(method='average')-.5)/allgroups.transform('count')-.5
    f['industry_rank20']=ranks.where(n.ge(minimum)&f.industry.ne('Unknown'),fallback)
    f['universe_mean_mom20']=allgroups.transform('mean')
    f['universe_breadth20']=s.gt(0).where(s.notna()).groupby(f.signal_date).transform('mean').astype(float)
    f['universe_dispersion20']=allgroups.transform('std')
    f['universe_median_vol20']=dates.volatility_20.transform('median')
    f['momentum_vol_interaction']=f.momentum_20*f.universe_median_vol20
    f['reversal_breadth_interaction']=f.reversal_1*(f.universe_breadth20-.5)
    f['momentum_dispersion_interaction']=f.momentum_60*f.universe_dispersion20
    return f


def period_mask(f,period):
    start,end=map(pd.Timestamp,period)
    return (f.signal_date.between(start,end)&f.signal_eligible&f.label_end_date.le(end)
            &f.label_end_date.gt(f.signal_date)&f.label_status.isin(['ok','ok_bridged'])
            &np.isfinite(f.y_excess_5d))


def load_frame(features):
    f=pd.read_parquet(ROOT/'data/processed/research_v1/features_labels.parquet',
        columns=['PERMNO','signal_date','signal_eligible','label_end_date','label_status','y_excess_5d']+features,
        filters=[('signal_date','>=',pd.Timestamp('2015-01-01')),('signal_date','<',pd.Timestamp('2022-01-01'))])
    require(f.signal_date.max()<pd.Timestamp('2022-01-01'),'Date isolation failed')
    require(not f.duplicated(['PERMNO','signal_date']).any(),'Duplicate samples')
    f=f.sort_values(['signal_date','PERMNO']).reset_index(drop=True)
    keys=f[['PERMNO','signal_date','signal_eligible']]
    risk=risk_inputs(keys,pd.DataFrame({'signal_date':sorted(f.signal_date.unique())}))
    return f.merge(risk[['PERMNO','signal_date','industry']],on=['PERMNO','signal_date'],validate='one_to_one')


def main():
    require(not OUT.exists(),'Output exists; no silent reruns')
    cfg=json.loads(CONFIG.read_text());parent=json.loads((ROOT/cfg['parent']).read_text())
    features=json.loads((ROOT/'config/features_labels.json').read_text())['features']
    added=list(cfg['context_features']);require(len(added)==12,'Feature budget')
    # Protect previous research without opening test-period records or metrics.
    metadata={str(p.relative_to(ROOT)): [p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'outputs').rglob('*') if p.is_file()}
    OUT.mkdir();(OUT/'models').mkdir();(OUT/'preprocessing').mkdir()
    write=lambda p,x:p.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
    input_paths=[CONFIG,ROOT/cfg['parent'],ROOT/'src/equity_research/models/context_selection.py',
        ROOT/'src/equity_research/backtest/risk_comparison.py',ROOT/'src/equity_research/models/preprocessing.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in input_paths}
    write(OUT/'frozen_plan.json',dict(config=cfg,parent=parent,sha256=hashes,started_utc=datetime.now(timezone.utc).isoformat()))
    print('Loading 2015-2021 only',flush=True)
    f=contexts(load_frame(features),cfg['industry_min_count'])
    f.to_parquet(OUT/'development_features_local.parquet',index=False)
    ranked=cross_sectional_rank(f,features)
    ledger=[];scores=[];daily=[];predictions={};states={}
    for wi,window in enumerate(parent['development_windows'],1):
        train=period_mask(f,window['train']);evaluate=period_mask(f,window['evaluate'])
        require(f.loc[train,'label_end_date'].max()<pd.Timestamp(window['evaluate'][0]),'Label boundary violation')
        eval_candidates=f.signal_date.between(*window['evaluate'])
        pred=f.loc[eval_candidates,['PERMNO','signal_date','signal_eligible']].copy()
        for family in ['ridge','lightgbm']:
            for enhanced in [False,True]:
                design=family+('_enhanced' if enhanced else '_original')
                x=f[features] if family=='ridge' else ranked
                state=fit_state(x.loc[train],'raw' if family=='ridge' else 'ranked',
                    {'raw_clip_quantiles':[.005,.995],'rank_clip_bounds':[-.5,.5]})
                X=transform_features(x,state)
                state={'base':state}
                if enhanced:
                    extra=fit_preprocessing(f.loc[train,added],[.005,.995])
                    X=pd.concat([X,transform_features(f[added],extra)],axis=1);state['context']=extra
                state.update(train_start=window['train'][0],train_end=window['train'][1],
                    max_label_end=str(f.loc[train,'label_end_date'].max()),columns=X.columns.tolist())
                write(OUT/'preprocessing'/f'w{wi}_{design}.json',state)
                states[(wi,design)]=state
                grid=cfg['ridge_alphas'] if family=='ridge' else cfg['lightgbm_num_leaves']
                for ci,value in enumerate(grid):
                    name=f'{design}_{ci}'
                    model=Ridge(alpha=value,solver='cholesky') if family=='ridge' else lgb.LGBMRegressor(**cfg['lightgbm_fixed'],num_leaves=value)
                    require(len(ledger)<cfg['max_fits'],'Fit budget exceeded')
                    with threadpool_limits(limits=4):
                        model.fit(X.loc[train],f.loc[train,'y_excess_5d'])
                        p=model.predict(X.loc[eval_candidates])
                    pred[name]=p
                    modelpath=OUT/'models'/f'w{wi}_{name}'
                    if family=='ridge':joblib.dump(model,modelpath.with_suffix('.joblib'))
                    else:save_lightgbm(model,modelpath.with_suffix('.txt'))
                    ic=daily_rank_ic(f.loc[evaluate,'signal_date'],pred.loc[evaluate,name],f.loc[evaluate,'y_excess_5d'])
                    require(np.isfinite(ic).all(),'Undefined evaluation IC')
                    scores.append(dict(window=wi,design=design,candidate=ci,parameter=value,rank_ic=ic.mean(),dates=len(ic),rows=int(evaluate.sum())))
                    daily.append(ic.rename('rank_ic').rename_axis('signal_date').reset_index().assign(window=wi,design=design,candidate=ci))
                    ledger.append(dict(fit=len(ledger)+1,window=wi,design=design,candidate=ci,parameter=value,
                        train_rows=int(train.sum()),max_train_signal=str(f.loc[train,'signal_date'].max()),
                        max_train_label=str(f.loc[train,'label_end_date'].max()),eval_start=window['evaluate'][0],eval_end=window['evaluate'][1]))
                    write(OUT/'fit_ledger.json',ledger)
                    print(f'Fit {len(ledger)}/32: window {wi} {name}',flush=True)
        pred.to_parquet(OUT/f'w{wi}_predictions_local.parquet',index=False);predictions[wi]=pred
    require(len(ledger)==32,'Incomplete budget')
    PUBLIC.mkdir(exist_ok=False)
    scores=pd.DataFrame(scores);daily=pd.concat(daily,ignore_index=True)
    ranking=scores.groupby(['design','candidate','parameter'],as_index=False).rank_ic.mean()
    selected=ranking.sort_values(['design','rank_ic','candidate'],ascending=[True,False,True]).groupby('design').head(1)
    for name,table in [('candidate_metrics',scores),('selection',selected),('candidate_ranking',ranking)]:
        table.to_csv(PUBLIC/(name+'.csv'),index=False)
    daily.to_csv(OUT/'candidate_daily_ic.csv',index=False)
    write(OUT/'selected.json',selected.to_dict('records'))
    chosen=[]
    for r in selected.itertuples():chosen.append(daily[daily.design.eq(r.design)&daily.candidate.eq(r.candidate)])
    chosen=pd.concat(chosen);chosen.to_csv(PUBLIC/'selected_daily_ic.csv',index=False)
    pairs=[]
    for family in ['ridge','lightgbm']:
        for wi,g in chosen[chosen.design.str.startswith(family)].groupby('window'):
            wide=g.pivot(index='signal_date',columns='design',values='rank_ic')
            delta=wide[family+'_enhanced']-wide[family+'_original']
            ci=circular_block_interval(delta.to_numpy(),10,2000,20261107)
            pairs.append(dict(family=family,window=wi,**ci))
    pd.DataFrame(pairs).to_csv(PUBLIC/'paired_ic.csv',index=False)
    portfolio_compare(predictions[2],selected,f,parent)
    for p,info in metadata.items():require([ (ROOT/p).stat().st_size,(ROOT/p).stat().st_mtime_ns]==info,'Prior artifact changed')
    for p,digest in hashes.items():require(sha(ROOT/p)==digest,'Frozen code/config changed during run')
    audit=dict(status='completed',fits=32,remaining_candidate_budget=16,final_refits=0,test_rows_read=0,
        test_predictions=0,test_metrics_used=False,original_artifact_metadata_unchanged=True,
        feature_digest=frame_digest(f),windows=parent['development_windows'],sha256=hashes)
    write(OUT/'verification.json',audit);write(PUBLIC/'verification.json',audit)
    write(PUBLIC/'frozen_plan.json',dict(config=cfg,parent=parent))
    print(selected.to_string(index=False),flush=True)


def portfolio_compare(pred,selected,f,parent,*,output_dir=None,public_dir=None):
    """Replay selected window-2 fits; portfolio results cannot change selection."""
    out=OUT if output_dir is None else output_dir
    public=PUBLIC if public_dir is None else public_dir
    _,terms,_,_,dates,m,_=validation.prepare_inputs()
    old=pd.read_csv(ROOT/'outputs/final_funding_backtest_v1/daily_portfolios.csv',parse_dates=['date'])
    ew=old[old.model.eq('equal_weight')&old.scenario.eq('lag_2')&old.cost_bps.eq(10)].sort_values('date')
    schedule=pd.DataFrame({'entry_date':ew.loc[ew.rebalance,'date']})
    schedule['signal_date']=[dates[dates.get_loc(d)-1] for d in schedule.entry_date]
    inputs=f[f.signal_date.isin(schedule.signal_date)].copy();inputs['annual_vol']=inputs.volatility_20
    metrics=[];annual=[];exposures=[];daily=[];checks=0
    for row in selected.itertuples():
        name=f'{row.design}_{row.candidate}'
        original=base.make_targets(pred,schedule,models=[name])[name]
        joined=inputs.merge(pred[['PERMNO','signal_date',name]],on=['PERMNO','signal_date'],validate='one_to_one')
        constrained={};snapshots={}
        for s in schedule.itertuples():
            g=joined[joined.signal_date.eq(s.signal_date)&joined.signal_eligible]
            weights,g=optimize(g,name,parent['constraints']);constrained[s.entry_date]=weights;snapshots[s.entry_date]=g
            exposures.append(dict(design=row.design,variant='constrained',stage='target',date=s.entry_date,**exposure_row(g,weights)))
        for variant,targets in [('original',original),('constrained',constrained)]:
            d,o,h,e=validation.simulate(m,targets,terms,10,2,False,True);checks+=validation.verify(d,o,h)
            active=d.return_net.iloc[1:].to_numpy()-ew.return_net.iloc[1:].to_numpy()
            te=active.std(ddof=1)*np.sqrt(252)
            metrics.append(dict(design=row.design,variant=variant,**base.performance(d),annual_mean_active=active.mean()*252,
                tracking_error=te,information_ratio=active.mean()*252/te,annual_fee_drag=float((d.fees.iloc[1:]/d.pretrade_nav.iloc[1:]).mean()*252)))
            for year,g in d.iloc[1:].groupby(d.date.iloc[1:].dt.year):annual.append(dict(design=row.design,variant=variant,year=year,net_return=float((1+g.return_net).prod()-1)))
            for date,g in snapshots.items():
                held=h[h.date.eq(date)&h.kind.isin(['equity','inherited','otc_frozen'])].groupby('PERMNO').value.sum()
                nav=d.loc[d.date.eq(date),'nav'].iloc[0]
                exposures.append(dict(design=row.design,variant=variant,stage='executed',date=date,**exposure_row(g,(held/nav).to_dict())))
            for kind,table in [('holdings',h),('trades',o),('events',e)]:table.to_parquet(out/f'{row.design}_{variant}_{kind}_local.parquet',index=False)
            daily.append(d.assign(design=row.design,variant=variant))
            print('Portfolio',row.design,variant,'verified',flush=True)
    for name,table in [('portfolio_metrics',pd.DataFrame(metrics)),('annual_returns',pd.DataFrame(annual)),
                       ('exposures',pd.DataFrame(exposures)),('daily_portfolios',pd.concat(daily))]:table.to_csv(public/(name+'.csv'),index=False)
    (out/'portfolio_checks.json').write_text(json.dumps(dict(ledger_days=checks,scope='2020-2021',selected_before_backtest=True)),encoding='utf-8')
