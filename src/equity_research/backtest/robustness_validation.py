"""Saved window-2 robustness predictions, unchanged V2 validation execution rules."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.backtest import market as base
from equity_research.backtest.risk_comparison import exposure_row
from equity_research.models.context_selection import portfolio_compare
from equity_research.models.common import circular_block_interval
from equity_research.models.parameter_robustness import load_plan, read_json
from equity_research.models.robustness_ledger import sha, atomic_json

ROOT=base.ROOT
OUT=ROOT/'outputs/robustness_validation_backtest_v2'
PUBLIC=ROOT/'results/robustness_validation_backtest_v2'
START=pd.Timestamp('2020-01-01');END=pd.Timestamp('2021-12-31')
KEYS=['PERMNO','signal_date','signal_eligible']
REFERENCE='lightgbm_original_old100'


def assert_dates(frame,column='signal_date'):
    base.require(frame[column].between(START,END).all(),'Validation-only date boundary')


def verify_selection(selection,metrics,designs):
    base.require(set(selection.design)==set(designs) and len(selection)==6,'Six selected designs required')
    base.require(not metrics.duplicated(['design','candidate','window']).any(),'Duplicate candidate metric')
    for row in selection.itertuples():
        group=metrics[metrics.design.eq(row.design)]
        wide=group.pivot(index='candidate',columns='window',values='rank_ic').sort_index()
        base.require(set(wide.columns)=={1,2} and wide.notna().all().all(),'Incomplete development windows')
        mean=(wide[1]+wide[2])/2
        base.require(row.candidate==mean.idxmax(),'Selection changed after development evaluation')
        old=[c for c in mean.index if int(c[1:])<4]
        base.require(row.old_candidate==mean.loc[old].idxmax(),'Old comparator changed')
    return selection


def read_scores(path,source_name,reader=pd.read_parquet):
    frame=reader(path,columns=KEYS+['prediction'],filters=[('signal_date','>=',START),('signal_date','<=',END)])
    assert_dates(frame)
    base.require(not frame.duplicated(['PERMNO','signal_date']).any(),'Duplicate prediction keys')
    base.require(np.isfinite(frame.loc[frame.signal_eligible,'prediction']).all(),'Missing eligible prediction')
    return frame.sort_values(['signal_date','PERMNO']).reset_index(drop=True).rename(columns={'prediction':source_name})


def load_predictions(root,selection,ledger):
    rows=selection[['design','candidate']].to_dict('records')
    old=selection.loc[selection.design.eq('lightgbm_original'),'old_candidate'].item()
    rows.append(dict(design=REFERENCE,candidate=old))
    merged=None;hashes={}
    for row in rows:
        design='lightgbm_original' if row['design']==REFERENCE else row['design']
        key=f"w2_{design}_{row['candidate']}"
        entry=ledger['entries'][key]
        base.require(entry['status']=='completed','Prediction task incomplete')
        relative=key+'/predictions.parquet'
        path=root/'outputs/parameter_robustness_v2'/relative
        base.require(sha(path)==entry['artifacts'][relative],'Saved prediction changed')
        hashes[path.relative_to(root).as_posix()]=sha(path)
        name=f"{row['design']}_{row['candidate']}"
        frame=read_scores(path,name)
        if merged is None:merged=frame
        else:
            pd.testing.assert_frame_equal(merged[KEYS],frame[KEYS])
            merged[name]=frame[name]
    return merged,pd.DataFrame(rows),hashes


def paired_comparison(new_daily,old_daily):
    """Paired net daily return differences; no model selection from this table."""
    keys=['design','variant','date']
    for frame in [new_daily,old_daily]:
        assert_dates(frame,'date')
        base.require(not frame.duplicated(keys).any(),'Duplicate daily portfolio')
    rows=[]
    for (design,variant),current in new_daily.groupby(['design','variant']):
        if design==REFERENCE:continue
        reference=old_daily[old_daily.design.eq(design)&old_daily.variant.eq(variant)]
        current=current.sort_values('date');reference=reference.sort_values('date')
        np.testing.assert_array_equal(current.date,reference.date)
        difference=current.return_net.iloc[1:].to_numpy()-reference.return_net.iloc[1:].to_numpy()
        ci=circular_block_interval(difference,20,2000,20261107)
        rows.append(dict(design=design,variant=variant,annual_mean_net_difference=ci['mean_difference']*252,
                         ci_low=ci['ci_low']*252,ci_high=ci['ci_high']*252,return_days=len(difference)))
    return pd.DataFrame(rows)


def exposure_summary(exposures,limits):
    x=exposures.copy();x['variant']=x.variant.fillna('constrained')
    x['beta_active']=x.beta-x.benchmark_beta;x['vol_active']=x.annual_stock_vol-x.benchmark_vol
    for name,breach in [('weight_breach',x.max_weight>limits['max_weight']+1e-8),
                        ('industry_breach',x.max_industry_active>limits['industry_active']+1e-8),
                        ('beta_breach',x.beta_active.abs()>limits['beta_active']+1e-8),
                        ('vol_breach',x.vol_active>limits['annual_stock_vol_active_upper']+1e-8)]:x[name]=breach
    target=x[x.stage.eq('target')]
    base.require(not target[['weight_breach','industry_breach','beta_breach','vol_breach']].any().any(),'Independent target constraint violation')
    summary=x.groupby(['design','variant','stage'],as_index=False).agg(
        observations=('date','size'),beta=('beta','mean'),benchmark_beta=('benchmark_beta','mean'),
        annual_stock_vol=('annual_stock_vol','mean'),benchmark_vol=('benchmark_vol','mean'),
        max_weight=('max_weight','max'),max_industry_active=('max_industry_active','max'),
        max_beta_active=('beta_active',lambda s:s.abs().max()),min_coverage=('coverage','min'),
        weight_breaches=('weight_breach','sum'),industry_breaches=('industry_breach','sum'),
        beta_breaches=('beta_breach','sum'),vol_breaches=('vol_breach','sum'))
    return summary,len(target)


def main(root=ROOT):
    out=root/'outputs/robustness_validation_backtest_v2';public=root/'results/robustness_validation_backtest_v2'
    base.require(not out.exists() and not public.exists(),'Preserve existing backtest')
    plan=load_plan(root);parent=read_json(root/'config/research_v2.json')
    base.require(parent['execution']['cost_bps_per_side']==10 and parent['execution']['company_action_lag']==2
                 and parent['execution']['ordinary_sales_reusable'],'Frozen execution contract changed')
    original_metadata={p.relative_to(root).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
                       for folder in ['outputs','results'] for p in (root/folder).rglob('*') if p.is_file()}
    result=root/'results/parameter_robustness_v2'
    source_paths=[result/name for name in ['selection.csv','candidate_window_metrics.csv','manifest.json']]
    source_paths += [root/'config'/name for name in ['research_v2.json','parameter_robustness_v2.json',
                                                  'validation_corporate_actions.json','settlement_calendar.json']]
    source_paths += [root/'src/equity_research'/name for name in ['backtest/robustness_validation.py','backtest/engine.py',
        'backtest/market.py','backtest/validation.py','backtest/risk_comparison.py','models/context_selection.py']]
    manifest=read_json(result/'manifest.json')
    for name in ['selection.csv','candidate_window_metrics.csv']:
        base.require(sha(result/name)==manifest[name],'Robustness result changed')
    selection=verify_selection(pd.read_csv(result/'selection.csv'),pd.read_csv(result/'candidate_window_metrics.csv'),
                               [d['id'] for d in plan['designs']])
    ledger_path=root/'outputs/parameter_robustness_v2/fit_ledger.json';source_paths.append(ledger_path)
    ledger=read_json(ledger_path)
    base.require(sum(e['kind']=='fit' for e in ledger['entries'].values())==80 and
                 len(ledger['entries'])==128 and all(e['status']=='completed' for e in ledger['entries'].values()),'Incomplete training ledger')
    pred,selected,prediction_hashes=load_predictions(root,selection,ledger)
    risk_path=root/plan['reuse']['development_snapshot'];source_paths.append(risk_path)
    base.require(sha(risk_path)==ledger['manifest']['inputs_and_code'][risk_path.relative_to(root).as_posix()], 'Risk input snapshot changed')
    risk_columns=KEYS+['industry','market_beta_60','volatility_20']
    risk=pd.read_parquet(risk_path,columns=risk_columns,filters=[('signal_date','>=',START),('signal_date','<=',END)])
    assert_dates(risk)
    pd.testing.assert_frame_equal(pred[KEYS],risk[KEYS].sort_values(['signal_date','PERMNO']).reset_index(drop=True))
    old_daily=[];old_metrics=[];old_annual=[]
    for directory in ['context_selection_v2','ranking_selection_v2']:
        for name,collection in [('daily_portfolios',old_daily),('portfolio_metrics',old_metrics),('annual_returns',old_annual)]:
            path=root/'results'/directory/(name+'.csv');source_paths.append(path)
            collection.append(pd.read_csv(path,parse_dates=['date'] if name=='daily_portfolios' else None))
    old_daily=pd.concat(old_daily,ignore_index=True);assert_dates(old_daily,'date')
    old_metrics=pd.concat(old_metrics,ignore_index=True);old_annual=pd.concat(old_annual,ignore_index=True)
    # Validate reference tables before any comparison, including full date coverage.
    base.require(set(old_daily.design)==set(selection.design),'Old V2 design coverage mismatch')
    hashes={p.relative_to(root).as_posix():sha(p) for p in source_paths};hashes.update(prediction_hashes)
    out.mkdir();public.mkdir()
    atomic_json(out/'protected_metadata.json',original_metadata)
    contract=dict(scope='2020-2021 validation; user-authorized backtest after robustness selection',
        selections=selected.to_dict('records'),paths=14,cost_bps_per_side=10,company_action_lag=2,
        ordinary_sales_reusable=True,constraints=parent['constraints'],new_fits=0,final_fits=0,
        prediction_calls=0,test_performance_reads=0,selection_changed=False,
        bootstrap=dict(block_days=20,replications=2000,seed=20261107))
    atomic_json(out/'frozen_backtest.json',contract);atomic_json(out/'source_hashes.json',hashes)
    print('Replaying 7 prediction sets x 2 variants under unchanged V2 validation rules',flush=True)
    portfolio_compare(pred,selected,risk,parent,output_dir=out,public_dir=public)
    daily=pd.read_csv(public/'daily_portfolios.csv',parse_dates=['date']);assert_dates(daily,'date')
    metrics=pd.read_csv(public/'portfolio_metrics.csv');annual=pd.read_csv(public/'annual_returns.csv')
    # Five unchanged designs plus explicit old-100-tree reference: 12 old paths.
    reproduced=[]
    for design in selected.design:
        if design=='lightgbm_original':continue
        original='lightgbm_original' if design==REFERENCE else design
        for variant in ['original','constrained']:
            a=daily[daily.design.eq(design)&daily.variant.eq(variant)].sort_values('date')
            b=old_daily[old_daily.design.eq(original)&old_daily.variant.eq(variant)].sort_values('date')
            np.testing.assert_array_equal(a.date,b.date)
            for col in ['nav','return_net','fees','turnover','cash','equity','receivable','payable']:
                np.testing.assert_allclose(a[col],b[col],atol=1e-10,rtol=1e-10)
            reproduced.append(dict(design=design,variant=variant,max_nav_error=float(np.abs(a.nav.to_numpy()-b.nav.to_numpy()).max())))
    current=metrics[metrics.design.ne(REFERENCE)]
    comparison=current.merge(old_metrics,on=['design','variant'],suffixes=('_new','_old'),validate='one_to_one')
    for metric in ['cagr','annual_volatility','sharpe','max_drawdown','mean_rebalance_turnover','annual_fee_drag','annual_mean_active','information_ratio']:
        comparison['delta_'+metric]=comparison[metric+'_new']-comparison[metric+'_old']
    year_comparison=annual[annual.design.ne(REFERENCE)].merge(old_annual,on=['design','variant','year'],suffixes=('_new','_old'),validate='one_to_one')
    year_comparison['delta_net_return']=year_comparison.net_return_new-year_comparison.net_return_old
    intervals=paired_comparison(daily,old_daily)
    exposures,target_count=exposure_summary(pd.read_csv(public/'exposures.csv'),parent['constraints'])
    focus=daily[daily.design.isin(['lightgbm_original',REFERENCE])].copy()
    focus=focus[focus.date.ne(focus.date.min())];focus['month']=focus.date.dt.to_period('M').astype(str)
    focus['fee_rate']=focus.fees/focus.pretrade_nav
    monthly=focus.groupby(['design','variant','month'],as_index=False).agg(net_return=('return_net',lambda s:(1+s).prod()-1),
        fee_rate_sum=('fee_rate','sum'),mean_nav=('nav','mean'),return_days=('date','size'))
    for name,table in [('comparison',comparison),('annual_comparison',year_comparison),('paired_intervals',intervals),
                       ('exposure_summary',exposures),('lightgbm_monthly',monthly),('reproduced_paths',pd.DataFrame(reproduced))]:
        table.to_csv(public/(name+'.csv'),index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,variant in zip(axes,['original','constrained']):
        for name,label in [(REFERENCE,'100 trees (original V2)'),('lightgbm_original','300 trees (robustness selection)')]:
            g=daily[daily.design.eq(name)&daily.variant.eq(variant)]
            ax.plot(g.date,g.nav,label=label)
        ax.set_title(variant+' | 10 bps per side');ax.set_ylabel('NAV');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(public/'lightgbm_comparison.png',dpi=160);plt.close(fig)
    for name,digest in hashes.items():base.require(sha(root/name)==digest,'Frozen backtest input changed')
    for name,metadata in original_metadata.items():
        p=root/name;base.require([p.stat().st_size,p.stat().st_mtime_ns]==metadata,'Old artifact modified')
    checks=read_json(out/'portfolio_checks.json')
    audit=dict(status='completed',model_fits=0,final_refits=0,new_prediction_calls=0,holdout_performance_reads=0,
        portfolio_paths=14,cost_bps_per_side=10,company_action_lag=2,target_checks=target_count,
        independently_verified_ledger_days=checks['ledger_days'],reproduced_old_paths=len(reproduced),
        first_date=str(daily.date.min().date()),last_date=str(daily.date.max().date()),
        protected_artifacts=len(original_metadata),old_artifact_metadata_unchanged=True,input_hashes_unchanged=True,
        selection_changed=False,bootstrap=contract['bootstrap'])
    atomic_json(out/'verification.json',audit);atomic_json(public/'verification.json',audit)
    atomic_json(public/'frozen_backtest.json',contract)
    atomic_json(public/'manifest.json',{p.name:sha(p) for p in public.iterdir() if p.is_file()})
    print(comparison[['design','variant','cagr_old','cagr_new','delta_cagr','delta_mean_rebalance_turnover','delta_annual_fee_drag']].to_string(index=False),flush=True)
    return audit


if __name__=='__main__':
    main()
