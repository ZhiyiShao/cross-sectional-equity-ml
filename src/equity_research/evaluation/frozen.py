"""Predeclared test metrics and paired circular block intervals; no selection."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from equity_research.models.common import daily_rank_ic, sha
from equity_research.backtest.frozen import OUT, ROOT
from equity_research.backtest.market import MODELS, require

def paired_interval(a,b,block,kind):
    a=np.asarray(a,float);b=np.asarray(b,float);n=len(a)
    require(n==len(b) and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid paired bootstrap')
    rng=np.random.default_rng(20261014);values=[]
    for _ in range(50):
        ix=(rng.integers(0,n,size=(100,int(np.ceil(n/block)),1))+np.arange(block)).reshape(100,-1)[:,:n]%n
        if kind=='ic':v=(a[ix]-b[ix]).mean(axis=1)
        else:v=np.expm1(np.log1p(a[ix]).mean(axis=1)*252)-np.expm1(np.log1p(b[ix]).mean(axis=1)*252)
        values.extend(v)
    point=float(np.mean(a-b)) if kind=='ic' else float(np.expm1(np.log1p(a).mean()*252)-np.expm1(np.log1p(b).mean()*252))
    return dict(estimate=point,lower95=float(np.quantile(values,.025)),upper95=float(np.quantile(values,.975)),
        lower97_5=float(np.quantile(values,.0125)),upper97_5=float(np.quantile(values,.9875)),replications=5000,block=block)

def make_figure(primary,stats,ics):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    names=dict(zip(MODELS,['Raw Ridge','Rank Ridge','Raw LightGBM','Rank LightGBM','Momentum20','Universe EW']))
    for model in MODELS:
        g=primary[primary.model.eq(model)];axes[0,0].plot(g.date,g.nav,label=names[model],linewidth=1.3)
        g=stats[stats.model.eq(model)&stats.scenario.eq('lag_2')].sort_values('cost_bps')
        axes[1,1].plot(g.cost_bps,g.cagr*100,marker='o',label=names[model])
    axes[0,0].set_title('Test NAV | 10 bps per side | lag 2');axes[0,0].legend(fontsize=7)
    axes[0,1].bar([names[m] for m in MODELS[:-1]],[ics[m].mean() for m in MODELS[:-1]])
    axes[0,1].axhline(0,color='black',linewidth=.7);axes[0,1].set_title('Mean 5-day Rank IC')
    axes[0,1].tick_params(axis='x',labelrotation=25)
    annual=pd.read_csv(OUT/'annual_metrics.csv');a=annual[annual.scenario.eq('lag_2')&annual.cost_bps.eq(10)]
    a.pivot(index='year',columns='model',values='total_return').reindex(columns=MODELS).rename(columns=names).mul(100).plot.bar(ax=axes[1,0],legend=False)
    axes[1,0].set_title('Annual net return (%)');axes[1,1].set_title('CAGR (%) by one-way cost (bps)')
    fig.suptitle('Frozen 2022–2025 test | OTC quote proxies; CVRs excluded from investable NAV',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'test_overview.png',dpi=160);plt.close(fig)

def main():
    meta=json.loads((OUT/'verification.json').read_text())
    require(meta['status']=='backtest_verified_predictive_evaluation_pending','Unexpected evaluation state')
    pred=pd.read_parquet(OUT/'predictions_local.parquet')
    labels=pd.read_parquet(ROOT/'data/processed/research_v1/features_labels.parquet',
        columns=['PERMNO','signal_date','label_end_date','label_status','y_excess_5d'],filters=[
            ('signal_date','>=',pred.signal_date.min()),('signal_date','<=',pred.signal_date.max()),
            ('label_end_date','<=',pd.Timestamp('2025-12-31'))])
    frame=pred.merge(labels,on=['PERMNO','signal_date'],how='left',validate='one_to_one')
    valid=frame.signal_eligible & frame.label_status.isin(['ok','ok_bridged']) & np.isfinite(frame.y_excess_5d)
    require(frame.loc[valid,'label_end_date'].le('2025-12-31').all(),'Label boundary violation')
    coverage=frame.assign(valid_label=valid).groupby('signal_date').agg(candidates=('PERMNO','size'),
        eligible=('signal_eligible','sum'),valid_labels=('valid_label','sum'))
    coverage['label_coverage']=coverage.valid_labels/coverage.eligible
    coverage.to_csv(OUT/'label_coverage.csv')
    x=frame[valid];ics={};metrics=[];yearly=[];groups=[]
    for model in MODELS[:-1]:
        ic=daily_rank_ic(x.signal_date,x[model],x.y_excess_5d)
        independent=x.groupby('signal_date').apply(lambda g:spearmanr(g[model],g.y_excess_5d).statistic,include_groups=False)
        np.testing.assert_allclose(ic,independent,atol=1e-12)
        ics[model]=ic
        metrics.append(dict(model=model,rank_ic=float(ic.mean()),ic_std=float(ic.std()),
            positive_ic_fraction=float(ic.gt(0).mean()),mse=float(np.mean((x[model]-x.y_excess_5d)**2)),
            label_rows=len(x),eligible_rows=int(frame.signal_eligible.sum()),coverage=float(valid.sum()/frame.signal_eligible.sum())))
        for year,g in ic.groupby(ic.index.year):yearly.append(dict(model=model,year=int(year),rank_ic=float(g.mean()),dates=len(g)))
        # Top decile membership formed on all eligible predictions, not valid-label subset.
        for date,g in frame[frame.signal_eligible].groupby('signal_date'):
            top=g.sort_values([model,'PERMNO'],ascending=[False,True]).head(int(np.ceil(len(g)/10)))
            ok=top.label_status.isin(['ok','ok_bridged'])&np.isfinite(top.y_excess_5d)
            groups.append(dict(model=model,signal_date=date,selected=len(top),valid_labels=int(ok.sum()),
                strict_label_mean=float(top.loc[ok,'y_excess_5d'].mean())))
    icframe=pd.DataFrame(ics);icframe.to_csv(OUT/'daily_rank_ic.csv')
    pd.DataFrame(metrics).to_csv(OUT/'predictive_metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'annual_rank_ic.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'top_decile_label_coverage.csv',index=False)
    daily=pd.read_csv(OUT/'daily_portfolios.csv',parse_dates=['date'])
    primary=daily[daily.scenario.eq('lag_2')&daily.cost_bps.eq(10)]
    ret=primary.pivot(index='date',columns='model',values='return_net').iloc[1:]
    intervals=[]
    for kind,a,b,blocks in [('ic','cross_sectional_rank__lightgbm_10','cross_sectional_rank__ridge_13',[10,5,20]),
                           ('cagr','cross_sectional_rank__lightgbm_10','equal_weight',[20,10,60])]:
        f=icframe if kind=='ic' else ret
        for block in blocks:intervals.append(dict(metric=kind,model=a,benchmark=b,primary=True,
                                                  **paired_interval(f[a],f[b],block,kind)))
    for model in MODELS[:-2]:
        intervals.append(dict(metric='ic',model=model,benchmark='momentum20',primary=False,
                              **paired_interval(icframe[model],icframe.momentum20,10,'ic')))
        intervals.append(dict(metric='cagr',model=model,benchmark='momentum20',primary=False,
                              **paired_interval(ret[model],ret.momentum20,20,'cagr')))
    pd.DataFrame(intervals).to_csv(OUT/'paired_confidence_intervals.csv',index=False)
    stats=pd.read_csv(OUT/'metrics.csv');mainstats=stats[stats.scenario.eq('lag_2')&stats.cost_bps.eq(10)]
    summary=mainstats.merge(pd.DataFrame(metrics),on='model',how='left')
    summary.to_csv(OUT/'summary.csv',index=False)
    make_figure(primary,stats,ics)
    meta.update(status='completed',strict_label_rows=int(valid.sum()),eligible_rows=int(frame.signal_eligible.sum()),
        test_prediction_selection_or_retuning=False,bootstrap_replications=5000,
        authorization_source='outputs/test_event_audit_v1/execution_scope.json',
        implementation_hashes={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'src/equity_research/backtest/frozen.py',ROOT/'src/equity_research/backtest/engine.py',
            ROOT/'src/equity_research/data/test_events.py',ROOT/'src/equity_research/evaluation/frozen.py',
            ROOT/'src/equity_research/backtest/calendar.py']})
    (OUT/'verification.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(summary[['model','cagr','sharpe','max_drawdown','rank_ic','coverage','cvr_contractual_maximum']].to_string(index=False))
    print(pd.DataFrame(intervals).query('primary').to_string(index=False))
