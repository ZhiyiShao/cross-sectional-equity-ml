"""V2 round one: saved predictions, fixed risk constraints, original execution ledger."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from equity_research.backtest import market as base, engine, frozen, validation
from equity_research.evaluation.exposures import sic_group
from equity_research.models.common import circular_block_interval

ROOT = base.ROOT
CONFIG = ROOT / 'config/research_v2.json'
OUT = ROOT / 'outputs/risk_comparison_v2'


def optimize(g, model, limits):
    """Use signal-date exposures only; the universe EW is a feasible witness."""
    g = g.sort_values('PERMNO').copy()
    n = len(g)
    base.require(n * limits['max_weight'] >= 1, 'Universe too small for weight cap')
    for c in ['market_beta_60', 'annual_vol']:
        g[c] = g[c].replace([np.inf, -np.inf], np.nan)
        base.require(g[c].notna().any(), 'Entirely missing exposure')
        g[c] = g[c].fillna(g[c].median())
    b = np.full(n, 1 / n)
    industry = pd.get_dummies(g.industry, dtype=float).to_numpy().T
    exposures = np.vstack([industry, g.market_beta_60.to_numpy()])
    bands = np.r_[np.full(len(industry), limits['industry_active']), limits['beta_active']]
    A = np.vstack([exposures, -exposures, g.annual_vol.to_numpy()])
    rhs = np.r_[exposures @ b + bands, -exposures @ b + bands,
                g.annual_vol.mean() + limits['annual_stock_vol_active_upper']]
    score = g[model].rank(method='average', pct=True).to_numpy()
    base.require(np.isfinite(score).all(), 'Missing prediction')
    result = linprog(-score, A_ub=A, b_ub=rhs, A_eq=np.ones((1, n)), b_eq=[1.],
                     bounds=(0, limits['max_weight']), method='highs')
    base.require(result.success, 'Risk optimization failed: ' + result.message)
    w = result.x
    base.require(np.max(A @ w - rhs) < 1e-8 and abs(w.sum()-1) < 1e-8, 'Target constraint violation')
    return dict(zip(g.PERMNO.astype(int), w)), g


def risk_inputs(pred, schedule):
    dates = list(schedule.signal_date)
    f = pd.read_parquet(ROOT/'data/processed/research_v1/features_only.parquet',
        columns=['PERMNO','signal_date','market_beta_60','volatility_20'], filters=[('signal_date','in',dates)])
    source = ROOT/'data/interim/crsp_audit_v1'
    parts = json.loads((source/'ingest.json').read_text())['parts']
    frames = []
    for part in parts:
        x = pd.read_parquet(source/part, columns=['PERMNO','DlyCalDt','SICCD','SecInfoStartDt','SecInfoEndDt'],
            filters=[('DlyCalDt','in',dates), ('PERMNO','in',list(pred.PERMNO.unique()))])
        frames.append(x)
    raw = pd.concat(frames, ignore_index=True)
    raw['industry'] = [sic_group(c,s,e,d) for c,s,e,d in zip(raw.SICCD,raw.SecInfoStartDt,raw.SecInfoEndDt,raw.DlyCalDt)]
    raw = raw.rename(columns={'DlyCalDt':'signal_date'})
    joined = pred.merge(f,on=['PERMNO','signal_date'],validate='one_to_one').merge(
        raw[['PERMNO','signal_date','industry']],on=['PERMNO','signal_date'],how='left',validate='one_to_one')
    joined['industry'] = joined.industry.fillna('Unknown')
    # features.make_features already annualizes volatility_20.
    joined['annual_vol'] = joined.volatility_20
    return joined


def exposure_row(g, weights):
    """NAV weights; missing/out-of-universe exposure is visible through coverage."""
    w = g.PERMNO.map(weights).fillna(0).to_numpy()
    b = np.full(len(g), 1/len(g))
    sectors = pd.get_dummies(g.industry,dtype=float).to_numpy().T
    return dict(coverage=float(w.sum()), beta=float(w @ g.market_beta_60),
        benchmark_beta=float(g.market_beta_60.mean()), annual_stock_vol=float(w @ g.annual_vol),
        benchmark_vol=float(g.annual_vol.mean()), max_industry_active=float(np.max(np.abs(sectors @ (w-b)))),
        max_weight=float(max(weights.values(),default=0)))


def main():
    base.require(not OUT.exists(), 'Output exists; preserve previous run')
    cfg = json.loads(CONFIG.read_text())
    # Hash all saved predictions, results and final models before reading performance.
    protected = {}
    for folder in ['frozen_test_v1','final_funding_backtest_v1','full_train_selection_v1','final_models_v1']:
        for p in (ROOT/'outputs'/folder).rglob('*'):
            if p.is_file(): protected[p.relative_to(ROOT).as_posix()] = base.sha(p)
    OUT.mkdir()
    (OUT/'frozen_plan.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    stats, years, exposures, intervals, daily_all = [], [], [], [], []
    checks = 0
    for period in ['validation','seen_test']:
        print('Loading',period,flush=True)
        if period == 'validation':
            _,terms,_,_,_,m,_ = validation.prepare_inputs()
            pred = base.read_predictions(ROOT/'outputs/full_train_selection_v1/selected_predictions.parquet')
            oldfolder = ROOT/'outputs/final_funding_backtest_v1'
            old = pd.read_csv(oldfolder/'daily_portfolios.csv',parse_dates=['date'])
            old = old[old.scenario.eq('lag_2') & old.cost_bps.eq(10)]
            entry = old.loc[old.model.eq('equal_weight') & old.rebalance,'date']
            dates = m['dates']
            schedule = pd.DataFrame({'entry_date':entry.to_numpy()})
            schedule['signal_date'] = [dates[dates.get_loc(d)-1] for d in schedule.entry_date]
            simulate, verify = validation.simulate, validation.verify
        else:
            _,terms,schedule,m = frozen.inputs()
            pred = pd.read_parquet(frozen.OUT/'predictions_local.parquet')
            old = pd.read_csv(frozen.OUT/'daily_portfolios.csv',parse_dates=['date'])
            old = old[old.scenario.eq('lag_2') & old.cost_bps.eq(10)]
            simulate, verify = engine.simulate, engine.verify
        joined = risk_inputs(pred,schedule)
        joined.to_parquet(OUT/(period+'_risk_inputs_local.parquet'),index=False)
        original = base.make_targets(pred,schedule,models=cfg['models']+['equal_weight'])
        ew = old[old.model.eq('equal_weight')].sort_values('date').reset_index(drop=True)
        pd.testing.assert_index_equal(pd.DatetimeIndex(ew.date),m['dates'],check_names=False)
        for model in cfg['models']:
            constrained, snapshots = {}, {}
            for row in schedule.itertuples():
                g = joined[joined.signal_date.eq(row.signal_date) & joined.signal_eligible]
                weights,g = optimize(g,model,cfg['constraints'])
                constrained[row.entry_date] = weights
                snapshots[row.entry_date] = g
                exposures.append(dict(period=period,model=model,variant='constrained',stage='target',date=row.entry_date,**exposure_row(g,weights)))
            paths = {}
            for variant,targets in [('original',original[model]),('constrained',constrained)]:
                print('Simulating',period,model,variant,flush=True)
                d,o,h,e = simulate(m,targets,terms,10,2,False,True)
                checks += verify(d,o,h)
                if variant == 'original':
                    reference = old[old.model.eq(model)].sort_values('date')
                    np.testing.assert_allclose(d.nav,reference.nav,atol=1e-10,rtol=1e-10)
                name = period+'__'+model+'__'+variant
                h.to_parquet(OUT/(name+'_holdings_local.parquet'),index=False)
                o.to_parquet(OUT/(name+'_trades_local.parquet'),index=False)
                e.to_parquet(OUT/(name+'_events_local.parquet'),index=False)
                paths[variant] = d
                daily_all.append(d.assign(period=period,model=model,variant=variant))
                active = d.return_net.iloc[1:].to_numpy()-ew.return_net.iloc[1:].to_numpy()
                te = active.std(ddof=1)*np.sqrt(252)
                stats.append(dict(period=period,model=model,variant=variant,**base.performance(d),
                    annual_mean_active=active.mean()*252, tracking_error=te, information_ratio=active.mean()*252/te,
                    annual_fee_drag=float((d.fees.iloc[1:]/d.pretrade_nav.iloc[1:]).mean()*252)))
                for year,g in d.iloc[1:].groupby(d.date.iloc[1:].dt.year):
                    years.append(dict(period=period,model=model,variant=variant,year=year,net_return=float((1+g.return_net).prod()-1)))
                for date,g in snapshots.items():
                    held = h[h.date.eq(date) & h.kind.isin(['equity','inherited','otc_frozen'])].groupby('PERMNO').value.sum()
                    nav = d.loc[d.date.eq(date),'nav'].iloc[0]
                    exposures.append(dict(period=period,model=model,variant=variant,stage='executed',date=date,
                        **exposure_row(g,(held/nav).to_dict())))
            difference = paths['constrained'].return_net.iloc[1:].to_numpy()-paths['original'].return_net.iloc[1:].to_numpy()
            bc = cfg['evaluation']['bootstrap']
            ci = circular_block_interval(difference,bc['block_days'],bc['replications'],bc['seed'])
            intervals.append(dict(period=period,model=model,annual_mean_delta=ci['mean_difference']*252,
                ci_low=ci['ci_low']*252,ci_high=ci['ci_high']*252,block_days=bc['block_days']))
        # Save progress without touching original artifacts.
        for name,rows in [('metrics',stats),('annual_returns',years),('exposures',exposures),('paired_intervals',intervals)]:
            pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
        pd.concat(daily_all).to_csv(OUT/'daily_portfolios.csv',index=False)
    for p,digest in protected.items():base.require(base.sha(ROOT/p)==digest,'Original artifact changed: '+p)
    verification = dict(model_fits=0,new_predictions=0,ledger_days_checked=checks,original_paths_reproduced=4,
        original_artifacts_unchanged=True,plan_sha256=base.sha(CONFIG),status='verified')
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    summarize()
    print(verification,flush=True)


def summarize():
    """Publish aggregate diagnostics only; licensed security-level files stay local."""
    public=ROOT/'results/risk_comparison_v2'
    public.mkdir(exist_ok=True)
    tables={name:pd.read_csv(OUT/(name+'.csv')) for name in ['metrics','annual_returns','exposures','paired_intervals']}
    x=tables['exposures'].copy()
    x['beta_active']=x.beta-x.benchmark_beta
    x['vol_active']=x.annual_stock_vol-x.benchmark_vol
    for name,condition in [('beta_breach',x.beta_active.abs()>.10000001),
        ('industry_breach',x.max_industry_active>.02000001),('weight_breach',x.max_weight>.02000001),
        ('vol_breach',x.vol_active>.10000001)]:x[name]=condition
    target=x[x.stage.eq('target')]
    base.require(not target[['beta_breach','industry_breach','weight_breach','vol_breach']].any().any(),'Independent target audit failed')
    summary=x.groupby(['period','model','variant','stage']).agg(
        dates=('date','size'),beta=('beta','mean'),benchmark_beta=('benchmark_beta','mean'),
        annual_stock_vol=('annual_stock_vol','mean'),benchmark_vol=('benchmark_vol','mean'),
        max_industry_active=('max_industry_active','max'),max_weight=('max_weight','max'),
        min_covered_nav=('coverage','min'),beta_breaches=('beta_breach','sum'),
        industry_breaches=('industry_breach','sum'),weight_breaches=('weight_breach','sum'),vol_breaches=('vol_breach','sum')).reset_index()
    summary.to_csv(public/'exposure_summary.csv',index=False)
    for name,t in tables.items():
        if name!='exposures':t.to_csv(public/(name+'.csv'),index=False)
    daily=pd.read_csv(OUT/'daily_portfolios.csv',parse_dates=['date'])
    pairs=[]
    for period,g in daily.groupby('period'):
        for variant,q in g.groupby('variant'):
            wide=q.pivot(index='date',columns='model',values='return_net').iloc[1:]
            diff=wide['raw__ridge_03']-wide['cross_sectional_rank__lightgbm_10']
            ci=circular_block_interval(diff.to_numpy(),20,2000,20261107)
            pairs.append(dict(period=period,variant=variant,ridge_minus_lgb_annual_mean=ci['mean_difference']*252,
                              ci_low=ci['ci_low']*252,ci_high=ci['ci_high']*252))
    pd.DataFrame(pairs).to_csv(public/'model_gap_intervals.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,period in zip(axes,['validation','seen_test']):
        for (model,variant),g in daily[daily.period.eq(period)].groupby(['model','variant']):
            label=('Ridge' if model.startswith('raw') else 'Rank LGBM')+' '+variant
            ax.plot(g.date,g.nav,label=label,linestyle='--' if variant=='original' else '-')
        ax.set_title(period+' | 10 bps/side, lag 2');ax.set_ylabel('NAV');ax.legend(fontsize=8)
    fig.savefig(public/'comparison.png',dpi=150);plt.close(fig)
    # Execution coverage: off-universe inherited assets are not silently counted as zero risk.
    rows=[]
    for period in ['validation','seen_test']:
        f=pd.read_parquet(OUT/(period+'_risk_inputs_local.parquet'))
        f=f[f.signal_eligible]
        rows.append(dict(period=period,rows=len(f),missing_beta=int(f.market_beta_60.isna().sum()),
            missing_vol=int(f.annual_vol.isna().sum()),unknown_industry=int(f.industry.eq('Unknown').sum())))
    pd.DataFrame(rows).to_csv(public/'input_coverage.csv',index=False)
    for name in ['frozen_plan.json','verification.json']:
        (public/name).write_bytes((OUT/name).read_bytes())
