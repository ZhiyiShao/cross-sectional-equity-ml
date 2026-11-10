"""Frozen development-only diagnostics for the completed 128-task robustness run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from equity_research.models import parameter_robustness as execution
from equity_research.models.robustness_ledger import sha, atomic_json
from equity_research.models.preprocessing import require

ROOT = execution.ROOT


def validate_run(root, plan):
    out = root/plan['outputs']['local']; ledger = execution.read_json(out/'fit_ledger.json')
    schedule = execution.tasks(plan)
    require(ledger['tasks'] == {t['key']:t for t in schedule}, 'Execution task schedule mismatch')
    require(set(ledger['entries']) == set(ledger['tasks']), 'Incomplete execution ledger')
    require(ledger['budget'] == 80 and sum(e['kind']=='fit' for e in ledger['entries'].values()) == 80, 'Execution budget mismatch')
    for key, entry in ledger['entries'].items():
        require(entry['status']=='completed' and entry['kind']==ledger['tasks'][key]['kind'], 'Failed or changed task')
        require(len(entry['artifacts'])>0, 'Missing task artifacts')
        for name, digest in entry['artifacts'].items():
            path = (out/name).resolve()
            require(path.is_relative_to(out.resolve()), 'Artifact path escape')
            require(sha(path)==digest, 'Task artifact changed: '+name)
    for name,digest in ledger['manifest']['inputs_and_code'].items():
        require(sha(root/name)==digest, 'Execution input changed: '+name)
    require(ledger['manifest']['plan_sha256']==sha(root/execution.PLAN), 'Execution plan changed')
    verification = execution.read_json(out/'execution_verification.json')
    require(verification['status']=='candidate_execution_complete' and verification['new_fits']==80,
            'No completed execution verification')
    return out, schedule, ledger


def prediction_diagnostics(frame, prediction, qualified):
    """Select top stocks before masking labels; NDCG uses the strict label population."""
    require(len(frame)==len(prediction) and frame.index.equals(qualified.index), 'Diagnostic alignment')
    require(np.isfinite(prediction).all(), 'Nonfinite candidate predictions')
    f=frame.copy(); f['prediction']=np.asarray(prediction); f['qualified']=qualified.to_numpy()
    rows=[]
    for date,group in f.groupby('signal_date',sort=True):
        eligible=group[group.signal_eligible]
        k=int(np.ceil(len(eligible)/10))
        require(k>0,'Empty signal population')
        top=eligible.sort_values(['prediction','PERMNO'],ascending=[False,True]).head(k)
        labeled=group[group.qualified]
        ic=ndcg=np.nan
        if len(labeled):
            require(len(labeled)>1 and np.isfinite(labeled.y_excess_5d).all(),'Insufficient strict labels')
            ic=float(spearmanr(labeled.prediction,labeled.y_excess_5d).statistic)
            require(np.isfinite(ic),'Undefined IC')
            relevance=np.minimum(9,np.floor(10*(labeled.y_excess_5d.rank(method='average')-.5)/len(labeled)))
            ranked=labeled.assign(relevance=relevance).sort_values(['prediction','PERMNO'],ascending=[False,True])
            q=int(np.ceil(len(labeled)/10)); discount=1/np.log2(np.arange(q)+2)
            ideal=float(np.sort(relevance.to_numpy())[::-1][:q]@discount)
            ndcg=float(ranked.relevance.iloc[:q].to_numpy()@discount/ideal) if ideal>0 else np.nan
        valid_top=top[top.qualified]
        rows.append(dict(signal_date=date,rank_ic=ic,ndcg_top10=ndcg,
                         top10_mean_label=float(valid_top.y_excess_5d.mean()),selected=k,
                         top10_valid_count=len(valid_top),eligible_population=len(eligible),
                         strict_population=len(labeled),top10_coverage=len(valid_top)/k))
    return pd.DataFrame(rows)


def candidate_tables(plan, window_metrics):
    rows=[]; selected=[]
    for design in plan['designs']:
        name=design['id']; grid=plan[design['grid']]['candidates']
        group=window_metrics[window_metrics.design.eq(name)]
        wide=group.pivot(index='candidate',columns='window',values='rank_ic')
        require(set(wide.index)=={c['id'] for c in grid} and set(wide.columns)=={1,2} and wide.notna().all().all(), 'Candidate coverage mismatch')
        wide['mean']=(wide[1]+wide[2])/2
        order=wide.sort_values('mean',ascending=False,kind='stable').sort_index()
        order=order.sort_values('mean',ascending=False,kind='stable')
        best=order.index[0]; runner=order.index[1]
        old_ids=[c['id'] for c in grid if c['reuse_candidate'] is not None]
        old=order[order.index.isin(old_ids)].index[0]
        rho=float(spearmanr(wide[1],wide[2]).statistic) if wide[1].nunique()>1 and wide[2].nunique()>1 else np.nan
        for c in grid:
            ci=c['id']; row=dict(design=name,candidate=ci,kind='reuse' if c['reuse_candidate'] is not None else 'fit',
                ic_2019=float(wide.loc[ci,1]),ic_2020_2021=float(wide.loc[ci,2]),equal_window_ic=float(wide.loc[ci,'mean']),
                candidate_rank=int(order.index.get_loc(ci))+1)
            for parameter in (['alpha'] if design['grid']=='ridge' else ['num_leaves','n_estimators','min_child_samples']):
                values=[x[parameter] for x in grid];v=c[parameter]
                row[parameter]=v
                row[parameter+'_position']='lower' if v==min(values) else 'upper' if v==max(values) else 'interior'
            rows.append(row)
        d1=float(wide.loc[best,1]-wide.loc[old,1]);d2=float(wide.loc[best,2]-wide.loc[old,2])
        flag='unchanged' if best==old else 'both_windows_improved' if d1>0 and d2>0 else 'window_tradeoff' if min(d1,d2)<0 else 'nonnegative_one_window'
        selected.append(dict(design=name,candidate=best,old_candidate=old,
            ic_2019=float(wide.loc[best,1]),ic_2020_2021=float(wide.loc[best,2]),equal_window_ic=float(wide.loc[best,'mean']),
            old_ic_2019=float(wide.loc[old,1]),old_ic_2020_2021=float(wide.loc[old,2]),old_equal_window_ic=float(wide.loc[old,'mean']),
            delta_2019=d1,delta_2020_2021=d2,equal_window_delta=(d1+d2)/2,robustness_flag=flag,
            runner_up=runner,winner_margin=float(wide.loc[best,'mean']-wide.loc[runner,'mean']),
            best_2019=wide.sort_index().sort_values(1,ascending=False,kind='stable').index[0],
            best_2020_2021=wide.sort_index().sort_values(2,ascending=False,kind='stable').index[0],
            window_candidate_rank_correlation=rho))
    candidates=pd.DataFrame(rows); selection=pd.DataFrame(selected)
    parameters=[c for c in candidates.columns if c not in ['kind','ic_2019','ic_2020_2021','equal_window_ic','candidate_rank']]
    selection=selection.merge(candidates[parameters],on=['design','candidate'],validate='one_to_one')
    return candidates,selection


def contrasts(plan, selection):
    pairs=[]
    for family,grid in [('ridge','ridge'),('lightgbm','lightgbm'),('ranker','lightgbm')]:
        for c in plan[grid]['candidates']:
            pairs.append(dict(comparison=f'features_{family}_{c["id"]}',kind='feature',
                baseline_design=family+'_original',baseline_candidate=c['id'],
                treatment_design=family+'_enhanced',treatment_candidate=c['id']))
    for variant in ['original','enhanced']:
        for c in plan['lightgbm']['candidates']:
            pairs.append(dict(comparison=f'objective_{variant}_{c["id"]}',kind='objective',
                baseline_design='lightgbm_'+variant,baseline_candidate=c['id'],
                treatment_design='ranker_'+variant,treatment_candidate=c['id']))
    require(len(pairs)==56,'Matched contrast count')
    for row in selection.itertuples():
        pairs.append(dict(comparison='selected_'+row.design,kind='selection',baseline_design=row.design,
            baseline_candidate=row.old_candidate,treatment_design=row.design,treatment_candidate=row.candidate))
    return pairs


def paired_intervals(daily, pairs, config):
    daily=daily[daily.rank_ic.notna()]
    rng=np.random.default_rng(config['seed']);windows={}
    for wi in [1,2]:
        wide=daily[daily.window.eq(wi)].pivot(index='signal_date',columns=['design','candidate'],values='rank_ic').sort_index()
        require(wide.notna().all().all() and len(wide)>=config['block_signals'],'Bootstrap paired coverage')
        n=len(wide);block=config['block_signals'];reps=config['replications']
        starts=rng.integers(n,size=(reps,int(np.ceil(n/block))))
        indices=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
        # One common resampling matrix per window for every candidate and contrast.
        draws=wide.to_numpy()[indices].mean(axis=1)
        windows[wi]=(wide,draws)
    rows=[]
    for pair in pairs:
        deltas=[];draws=[]
        for wi in [1,2]:
            wide,bootstrap=windows[wi]
            left=(pair['baseline_design'],pair['baseline_candidate']);right=(pair['treatment_design'],pair['treatment_candidate'])
            delta=wide[right].to_numpy()-wide[left].to_numpy()
            sample=bootstrap[:,wide.columns.get_loc(right)]-bootstrap[:,wide.columns.get_loc(left)]
            deltas.append(float(delta.mean()));draws.append(sample)
            lo,hi=np.quantile(sample,[.025,.975])
            rows.append(dict(**pair,window=str(wi),delta=deltas[-1],ci_low=lo,ci_high=hi))
        lo,hi=np.quantile((draws[0]+draws[1])/2,[.025,.975])
        rows.append(dict(**pair,window='equal',delta=np.mean(deltas),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)


def factorial_effects(candidate_metrics):
    rows=[]
    for design,group in candidate_metrics[~candidate_metrics.design.str.startswith('ridge')].groupby('design'):
        for scope,col in [('1','ic_2019'),('2','ic_2020_2021'),('equal','equal_window_ic')]:
            lookup=group.set_index(['num_leaves','n_estimators','min_child_samples'])[col]
            for leaves in [7,63]:
                a=lookup.loc[(leaves,100,500)];b=lookup.loc[(leaves,300,500)]
                c=lookup.loc[(leaves,100,100)];d=lookup.loc[(leaves,300,100)]
                rows.append(dict(design=design,window=scope,num_leaves=leaves,
                    trees_at_min500=b-a,trees_at_min100=d-c,min_samples_at_100trees=c-a,
                    min_samples_at_300trees=d-b,interaction=d-c-b+a))
    boundaries=[]
    for design,group in candidate_metrics[~candidate_metrics.design.str.startswith('ridge')].groupby('design'):
        lookup=group.set_index(['num_leaves','n_estimators','min_child_samples'])
        for left,right in [(7,3),(63,127)]:
            for scope,col in [('1','ic_2019'),('2','ic_2020_2021'),('equal','equal_window_ic')]:
                boundaries.append(dict(design=design,window=scope,baseline_leaves=left,new_leaves=right,
                                       delta=lookup.loc[(right,100,500),col]-lookup.loc[(left,100,500),col]))
    return pd.DataFrame(rows),pd.DataFrame(boundaries)


def main(root=ROOT):
    plan=execution.load_plan(root);public=root/plan['outputs']['public']
    require(not public.exists(),'Evaluation output exists; do not overwrite')
    out,schedule,ledger=validate_run(root,plan)
    ledger_hash=sha(out/'fit_ledger.json');snapshot=root/plan['reuse']['development_snapshot']
    windows={};access={};daily=[];summaries=[]
    for wi,window in enumerate(plan['development_windows'],1):
        frame,masks,log=execution.read_window(snapshot,window,[])
        ec=frame.signal_date.between(*window['evaluate'])
        windows[wi]=(frame.loc[ec].reset_index(drop=True),masks['evaluate'].loc[ec].reset_index(drop=True))
        access[str(wi)]=log
    for i,task in enumerate(schedule,1):
        frame,qualified=windows[task['window']];folder=out/task['key']
        window=plan['development_windows'][task['window']-1]
        pred=pd.read_parquet(folder/'predictions.parquet',columns=execution.KEYS+['signal_eligible','prediction'],
            filters=[('signal_date','>=',pd.Timestamp(window['evaluate'][0])),('signal_date','<=',pd.Timestamp(window['evaluate'][1]))])
        pd.testing.assert_frame_equal(frame[execution.KEYS+['signal_eligible']],pred[execution.KEYS+['signal_eligible']])
        table=prediction_diagnostics(frame,pred.prediction.to_numpy(),qualified)
        saved=pd.read_csv(folder/'daily_ic.csv',parse_dates=['signal_date'])
        ic_table=table[table.rank_ic.notna()].reset_index(drop=True)
        pd.testing.assert_series_equal(ic_table.signal_date,saved.signal_date,check_names=False)
        np.testing.assert_allclose(ic_table.rank_ic,saved.rank_ic,atol=1e-12,rtol=1e-12)
        summary=execution.read_json(folder/'summary.json')
        np.testing.assert_allclose(table.rank_ic.mean(),summary['rank_ic'],atol=1e-12,rtol=1e-12)
        table=table.assign(window=task['window'],design=task['design'],candidate=task['candidate']['id'])
        daily.append(table)
        summary={k:v for k,v in summary.items() if k!='parameters'}
        summary.update({name:float(table[name].mean()) for name in ['ndcg_top10','top10_mean_label','top10_coverage','top10_valid_count']})
        summaries.append(summary)
        if i%16==0:print(f'Independent candidate diagnostics: {i}/128',flush=True)
    daily=pd.concat(daily,ignore_index=True);metrics=pd.DataFrame(summaries)
    candidates,selection=candidate_tables(plan,metrics)
    # Freeze the computed selection before calculating any secondary comparisons.
    selection_records=selection[['design','candidate','old_candidate']].to_dict('records')
    intervals=paired_intervals(daily,contrasts(plan,selection),plan['evaluation']['bootstrap'])
    effects,boundaries=factorial_effects(candidates)
    annual=daily.assign(year=daily.signal_date.dt.year).groupby(['design','candidate','year'],as_index=False).agg(
        rank_ic=('rank_ic','mean'),dates=('rank_ic','count'),total_signal_dates=('signal_date','size'),ndcg_top10=('ndcg_top10','mean'),
        top10_mean_label=('top10_mean_label','mean'),top10_coverage=('top10_coverage','mean'))
    old_sources=pd.concat([pd.read_csv(root/'results'/name/'candidate_ranking.csv') for name in ['context_selection_v2','ranking_selection_v2']])
    anchors=candidates[candidates.kind.eq('reuse')].copy();anchors['old_id']=anchors.candidate.str[1:].astype(int)
    check=anchors.merge(old_sources,left_on=['design','old_id'],right_on=['design','candidate'],validate='one_to_one')
    require(len(check)==24,'Old mean coverage')
    np.testing.assert_allclose(check.equal_window_ic,check.rank_ic,atol=1e-12,rtol=1e-12)
    require(sha(out/'fit_ledger.json')==ledger_hash,'Ledger changed during evaluation')
    for name,digest in ledger['manifest']['inputs_and_code'].items():require(sha(root/name)==digest,'Protected input changed')
    public.mkdir()
    for name,table in [('candidate_window_metrics',metrics),('candidate_daily_metrics',daily),('candidate_annual_metrics',annual),
                       ('candidate_ranking',candidates),('selection',selection),('paired_ic_intervals',intervals),
                       ('factorial_effects',effects),('boundary_effects',boundaries)]:
        table.to_csv(public/f'{name}.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharey=True)
    for ax,design in zip(axes.flat,plan['designs']):
        data=candidates[candidates.design.eq(design['id'])].sort_values('alpha' if design['grid']=='ridge' else 'candidate')
        labels=[f'{a:g}' for a in data.alpha] if design['grid']=='ridge' else data.candidate.tolist()
        for col,label in [('ic_2019','2019'),('ic_2020_2021','2020-2021'),('equal_window_ic','Equal-window mean')]:
            ax.plot(range(len(data)),data[col],marker='o',label=label,markersize=4)
        ax.set_xticks(range(len(data)),labels,rotation=45);ax.axhline(0,color='gray',lw=.7)
        ax.set_title(design['id']);ax.set_ylabel('Mean Rank IC');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('V2 parameter robustness | development windows only | old + frozen new candidates')
    fig.tight_layout();fig.savefig(public/'candidate_curves.png',dpi=150);plt.close(fig)
    atomic_json(out/'evaluation_access_audit.json',access)
    audit=dict(status='completed',candidate_window_results=len(metrics),design_configurations=len(candidates),
        new_fits=80,reused=48,evaluation_fits=0,final_refits=0,backtests=0,holdout_performance_reads=0,
        daily_diagnostic_rows=len(daily),annual_rows=len(annual),matched_contrasts=56,selected_comparisons=6,
        interval_rows=len(intervals),old_scores_reproduced=True,selection=selection_records,
        plan_sha256=sha(root/execution.PLAN),execution_ledger_sha256=ledger_hash,evaluation_code_sha256=sha(Path(__file__)),
        bootstrap=plan['evaluation']['bootstrap'],interpretation='Development diagnostics; not independent or selection-adjusted evidence')
    atomic_json(public/'verification.json',audit)
    atomic_json(public/'manifest.json',{p.name:sha(p) for p in public.iterdir() if p.is_file()})
    print(selection[['design','candidate','old_candidate','ic_2019','ic_2020_2021','equal_window_ic','equal_window_delta','robustness_flag']].to_string(index=False),flush=True)
    return audit


if __name__=='__main__':
    main()
