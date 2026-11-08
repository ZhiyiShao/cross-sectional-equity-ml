"""Frozen V2 historical replay (2022–2025). No fitting or unseen-data route."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from equity_research.paths import ROOT
from equity_research.data.ingest import write_json
from equity_research.data.date_gate import signal_schedule
from equity_research.models.common import sha
from equity_research.models.preprocessing import require, frame_digest
from equity_research.models.final_fit_v2 import artifact_metadata, check_metadata, read_json, IDS, PLAN
from equity_research.evaluation.v2 import locked_bundle, predict_frame, run_evaluation
from equity_research.backtest import market, engine
from equity_research.backtest.calendar import CONFIG as CALENDAR_CONFIG, is_exchange_day, settlement_date
from equity_research.backtest.risk_comparison import optimize

OUT = ROOT/'outputs/historical_replay_v2'
PUBLIC = ROOT/'results/historical_replay_v2'
AUDIT = ROOT/'outputs/test_event_audit_v1'
START, END = pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31')
BENCHMARKS = ['momentum20', 'equal_weight']


class ReplayReader:
    """Explicit source/date allowlist; every parquet call has storage-level bounds."""
    SOURCES = {
        'data/interim/crsp_audit_v1/calendar.parquet': ('date', '2015-01-02'),
        'data/processed/research_v1/features_only.parquet': ('signal_date', '2022-01-01'),
        'data/processed/research_v1/features_labels.parquet': ('signal_date', '2022-01-01'),
        'outputs/risk_comparison_v2/seen_test_risk_inputs_local.parquet': ('signal_date', '2022-01-01'),
        'outputs/test_event_audit_v1/execution_market_local.parquet': ('DlyCalDt', '2022-01-01'),
        'outputs/test_event_audit_v1/candidates_local.parquet': ('signal_date', '2022-01-01'),
    }

    def __init__(self):
        self.log = []

    def read(self, source, columns, start=START, end=END):
        require(source in self.SOURCES, 'Unregistered historical observation source')
        date, minimum = self.SOURCES[source]
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        require(pd.Timestamp(minimum) <= start <= end <= END, 'Unseen or out-of-scope date request')
        require(date in columns, 'Date projection required')
        if date == 'date':
            require(columns == ['date'], 'Calendar warmup may not read returns')
            lo, hi = str(start.date()), str(end.date())
        else:
            lo, hi = start, end
        filters = [(date, '>=', lo), (date, '<=', hi)]
        if 'y_excess_5d' in columns:
            require('label_end_date' in columns, 'Target horizon projection required')
            filters += [('label_end_date', '<=', end)]
        f = pd.read_parquet(ROOT/source, columns=columns, filters=filters)
        f[date] = pd.to_datetime(f[date])
        require(len(f)>0 and f[date].between(start, end).all(), 'Returned date breach')
        if 'y_excess_5d' in columns:
            require(f.label_end_date.le(end).all(), 'Returned target horizon breach')
        self.log.append(dict(source=source, columns=columns, filters=repr(filters), rows=len(f),
            minimum=str(f[date].min()), maximum=str(f[date].max()), digest=frame_digest(f)))
        return f


def inputs(reader, bundle):
    calendar = reader.read('data/interim/crsp_audit_v1/calendar.parquet', ['date'], '2015-01-02').date
    calendar = pd.DatetimeIndex(calendar).sort_values()
    expected_exchange = pd.DatetimeIndex([d for d in pd.bdate_range(START, END) if is_exchange_day(d)])
    pd.testing.assert_index_equal(calendar[(calendar>=START)&(calendar<=END)], expected_exchange, check_names=False)
    schedule = signal_schedule(calendar, START, END)
    previous = pd.read_csv(AUDIT/'schedule.csv', parse_dates=['signal_date', 'entry_date', 'end_date'])
    pd.testing.assert_frame_equal(schedule, previous)
    dates = calendar[(calendar>=schedule.signal_date.min())&(calendar<=schedule.end_date.max())]
    for date in dates:
        settlement_date(date)  # Verify every session, including future settlement obligations.
    columns = ['PERMNO', 'signal_date', 'signal_eligible']+bundle['original_features']
    features = reader.read('data/processed/research_v1/features_only.parquet', columns,
                           schedule.signal_date.min(), schedule.signal_date.max())
    features = features[features.signal_date.isin(schedule.signal_date)].sort_values(['signal_date', 'PERMNO']).reset_index(drop=True)
    candidates = reader.read('outputs/test_event_audit_v1/candidates_local.parquet', columns[:3])
    pd.testing.assert_frame_equal(features[columns[:3]], candidates.sort_values(['signal_date','PERMNO']).reset_index(drop=True))
    require(features.groupby('signal_date').size().eq(500).all(), 'Historical candidate count changed')
    source = 'outputs/risk_comparison_v2/seen_test_risk_inputs_local.parquet'
    hashes = read_json(ROOT/'outputs/risk_comparison_v2/implementation_and_inputs.json')
    require(sha(ROOT/source)==hashes[source], 'Previously audited industry snapshot changed')
    risk = reader.read(source, ['PERMNO','signal_date','industry','market_beta_60','volatility_20'])
    joined = features.merge(risk, on=['PERMNO','signal_date'], how='left', validate='one_to_one', suffixes=('', '_cached'))
    require(joined.industry.notna().all(), 'Industry snapshot coverage mismatch')
    for col in ['market_beta_60','volatility_20']:
        np.testing.assert_allclose(joined[col], joined[col+'_cached'], equal_nan=True)
    features = joined.drop(columns=['market_beta_60_cached','volatility_20_cached'])
    registry = read_json(AUDIT/'action_registry_draft.json')
    require(registry['unresolved_count']==0 and registry['exception_authorization']=='采用简化口径并明确披露', 'Unresolved corporate action policy')
    for name, digest in registry['inputs'].items():
        require(sha(ROOT/name)==digest, 'Event source changed')
    terms = {e['permno']:e for e in registry['events']}
    raw_columns = ['PERMNO','DlyCalDt','DlyDelFlg','DlyPrc','DlyPrcFlg','DlyRet','DlyRetx','DlyCumFacPr',
                   'DlyRetMissFlg','DlyRetDurFlg','sellable','buyable','ordinary_return','known_terminal']
    raw = reader.read('outputs/test_event_audit_v1/execution_market_local.parquet', raw_columns, dates.min(), dates.max())
    require(not raw.duplicated(['PERMNO','DlyCalDt']).any(), 'Duplicate market observations')
    pd.testing.assert_index_equal(pd.DatetimeIndex(sorted(raw.DlyCalDt.unique())), dates, check_names=False)
    m = market.build_market(raw, dates, set(raw.PERMNO))
    m['price'] = raw.pivot(index='DlyCalDt', columns='PERMNO', values='DlyPrc').reindex(index=dates, columns=m['ids']).to_numpy()
    for permno, event in terms.items():
        date = pd.Timestamp(event['event_date'])
        require(dates.min()<=date<=dates.max(), 'Event outside execution scope')
        i, j = dates.get_loc(date), np.flatnonzero(m['ids']==permno)[0]
        require(i>0 and m['price'][i-1,j]>0, 'Missing pre-event price')
        m['terminal'][i,j], m['ordinary'][i,j] = True, False
    return features, schedule, m, registry, terms


def target_sets(predictions, features, schedule, constraints):
    models = IDS+BENCHMARKS
    original = market.make_targets(predictions, schedule, models=models)
    portfolios = {(name, 'original' if name in IDS else 'benchmark'): value for name, value in original.items()}
    snapshots, target_rows = {}, []
    joined = features.merge(predictions[['PERMNO','signal_date']+IDS+['momentum20']], on=['PERMNO','signal_date'], validate='one_to_one')
    for row in schedule.itertuples():
        g = joined[joined.signal_date.eq(row.signal_date)&joined.signal_eligible].copy()
        g['annual_vol'] = g.volatility_20
        for col in ['market_beta_60','annual_vol']:
            g[col] = g[col].replace([np.inf,-np.inf], np.nan)
            require(g[col].notna().any(), 'Entirely missing exposure')
            g[col] = g[col].fillna(g[col].median())
        snapshots[row.entry_date] = g
        for model in IDS:
            weights, _ = optimize(g, model, constraints)
            portfolios.setdefault((model,'constrained'), {})[row.entry_date] = weights
        for (model, variant), targets in portfolios.items():
            if row.entry_date not in targets:
                continue
            specification = targets[row.entry_date]
            weights = specification if isinstance(specification,dict) else dict.fromkeys(specification,1/len(specification))
            target_rows.append(dict(model=model, variant=variant, date=row.entry_date, stage='target', **exposure(g,weights)))
    return portfolios, snapshots, pd.DataFrame(target_rows)


def exposure(g, weights):
    """Current-signal universe coverage; outside-universe inherited holdings stay visible."""
    w = g.PERMNO.map(weights).fillna(0).to_numpy()
    sectors = pd.get_dummies(g.industry, dtype=float).to_numpy().T
    b = np.full(len(g),1/len(g))
    return dict(coverage=float(w.sum()), total_equity_weight=float(sum(weights.values())),
        outside_universe_weight=float(sum(weights.values())-w.sum()),
        beta=float(w@g.market_beta_60), benchmark_beta=float(g.market_beta_60.mean()),
        annual_stock_vol=float(w@g.annual_vol), benchmark_vol=float(g.annual_vol.mean()),
        log_market_cap=float(w@g.log_market_cap.fillna(0)), benchmark_log_market_cap=float(g.log_market_cap.mean()),
        max_industry_active=float(np.max(np.abs(sectors@(w-b)))), max_weight=float(max(weights.values(),default=0)))


def check_targets(table, limits):
    target = table[table.variant.eq('constrained')&table.stage.eq('target')]
    require(target.coverage.sub(1).abs().lt(1e-8).all(), 'Target weight sum')
    require(target.max_weight.le(limits['max_weight']+1e-8).all(), 'Target concentration cap')
    require(target.max_industry_active.le(limits['industry_active']+1e-8).all(), 'Target industry band')
    require((target.beta-target.benchmark_beta).abs().le(limits['beta_active']+1e-8).all(), 'Target beta band')
    require((target.annual_stock_vol-target.benchmark_vol).le(limits['annual_stock_vol_active_upper']+1e-8).all(), 'Target vol band')


def main():
    require(not OUT.exists() and not PUBLIC.exists() and not (ROOT/'outputs/evaluation_v2_historical_2022').exists(), 'Preserve prior V2 replay; no implicit rerun')
    plan, bundle, model_manifest = locked_bundle()
    require(plan['historical_evaluation']['dates']==['2022-01-01','2025-12-31'], 'Historical period changed')
    protected = artifact_metadata()
    reader = ReplayReader()
    features, schedule, m, registry, terms = inputs(reader, bundle)
    print(f'Inputs audited: {len(schedule)} signals; {len(m["dates"])} market dates; {len(terms)} registered events', flush=True)
    OUT.mkdir()
    write_json(OUT/'protected_metadata.json', protected)
    write_json(OUT/'model_manifest.json', model_manifest)
    write_json(OUT/'frozen_plan.json', plan)
    paths = [PLAN, CALENDAR_CONFIG, AUDIT/'schedule.csv', AUDIT/'action_registry_draft.json',
             ROOT/'src/equity_research/backtest/replay_v2.py', ROOT/'src/equity_research/evaluation/replay_v2.py',
             ROOT/'src/equity_research/backtest/engine.py', ROOT/'src/equity_research/backtest/risk_comparison.py',
             ROOT/'src/equity_research/evaluation/v2.py']
    source_hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    write_json(OUT/'source_hashes.json', source_hashes)
    write_json(OUT/'event_registry.json', registry)
    schedule.to_csv(OUT/'schedule.csv',index=False)
    features.to_parquet(OUT/'features_local.parquet',index=False)
    # Exercise the full-universe event/ledger path before generating model scores.
    ew = market.make_targets(features,schedule,models=['equal_weight'])['equal_weight']
    d,o,h,e = engine.simulate(m,ew,terms,10,2,False,True)
    checks = engine.verify(d,o,h)
    old = pd.read_csv(ROOT/'outputs/frozen_test_v1/daily_portfolios.csv',parse_dates=['date'])
    reference = old[old.model.eq('equal_weight')&old.scenario.eq('lag_2')&old.cost_bps.eq(10)].sort_values('date')
    pd.testing.assert_index_equal(pd.DatetimeIndex(d.date),pd.DatetimeIndex(reference.date),check_names=False)
    np.testing.assert_allclose(d.nav,reference.nav,atol=1e-10,rtol=1e-10)
    print('Equal-weight event exercise verified and original NAV reproduced',flush=True)
    with threadpool_limits(limits=4):
        predictions = predict_frame(features,bundle,(START,END))
    predictions.to_parquet(OUT/'predictions_local.parquet',index=False)
    portfolios,snapshots,exposures = target_sets(predictions,features,schedule,plan['portfolio']['constraints'])
    check_targets(exposures,plan['portfolio']['constraints'])
    exposures.to_csv(OUT/'target_exposures.csv',index=False)
    print('Six predictions and 1,194 constrained targets frozen before performance calculation',flush=True)
    all_daily, metrics, annual, realized, ledger = [], [], [], [], []
    cached_ew = (d,o,h,e)
    for cost in [10]+plan['portfolio']['costs']['diagnostic_bps_per_side']:
        for (model,variant),targets in portfolios.items():
            primary = cost==10
            name = f'{model}__{variant}__{cost}'
            ledger.append(dict(path=name,status='started'))
            write_json(OUT/'path_ledger.json',ledger)
            if primary and model=='equal_weight':
                d,o,h,e = cached_ew
            else:
                d,o,h,e = engine.simulate(m,targets,terms,cost,2,False,primary)
                if primary:
                    checks += engine.verify(d,o,h)
            require(d.date.equals(cached_ew[0].date), 'Portfolio dates differ')
            require(d.nav.iloc[0]==1 and d.equity.iloc[0]==0, 'Portfolio must start cash-only at NAV 1')
            if primary:
                for kind,table in [('holdings',h),('trades',o),('events',e)]:
                    table.to_parquet(OUT/f'{name}_{kind}_local.parquet',index=False)
                for date,g in snapshots.items():
                    held = h[h.date.eq(date)&h.kind.isin(['equity','inherited','otc_frozen'])].groupby('PERMNO').value.sum()
                    nav = d.loc[d.date.eq(date),'nav'].iloc[0]
                    realized.append(dict(model=model,variant=variant,date=date,stage='executed',**exposure(g,(held/nav).to_dict())))
            metrics.append(dict(model=model,variant=variant,cost_bps=cost,**market.performance(d),
                annual_fee_rate=float((d.fees.iloc[1:]/d.pretrade_nav.iloc[1:]).mean()*252),
                cvr_contractual_maximum=float(d.cvr_contractual_maximum.iloc[-1])))
            for year,g in d.iloc[1:].groupby(d.date.iloc[1:].dt.year):
                annual.append(dict(model=model,variant=variant,cost_bps=cost,year=int(year),net_return=float((1+g.return_net).prod()-1)))
            all_daily.append(d.assign(model=model,variant=variant,cost_bps=cost))
            # Persist each completed path, allowing audit of an interrupted run without refitting.
            d.to_parquet(OUT/f'{name}_daily.parquet',index=False)
            ledger[-1]['status']='completed'
            write_json(OUT/'path_ledger.json',ledger)
            print(f'Completed {len(ledger)}/56: {name}',flush=True)
    daily = pd.concat(all_daily,ignore_index=True)
    daily.to_parquet(OUT/'daily_portfolios.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'portfolio_metrics.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_returns.csv',index=False)
    pd.concat([exposures,pd.DataFrame(realized)],ignore_index=True).to_csv(OUT/'exposures.csv',index=False)
    labels = reader.read('data/processed/research_v1/features_labels.parquet',
        ['PERMNO','signal_date','label_end_date','label_status','y_excess_5d'],schedule.signal_date.min(),schedule.signal_date.max()+pd.Timedelta(days=14))
    full = predictions.merge(labels,on=['PERMNO','signal_date'],how='left',validate='one_to_one')
    qualified = (full.signal_eligible & full.label_status.isin(['ok','ok_bridged']) &
        np.isfinite(full.y_excess_5d) & full.label_end_date.gt(full.signal_date) & full.label_end_date.le(END))
    strict = full.loc[qualified].copy()
    full.to_parquet(OUT/'scored_population_local.parquet',index=False)
    strict.to_parquet(OUT/'strict_predictions_local.parquet',index=False)
    primary_daily = daily[daily.variant.eq('constrained')&daily.cost_bps.eq(10)&daily.date.gt(m['dates'][0])]
    primary_daily[['date','model','return_net']].to_parquet(OUT/'primary_daily_local.parquet',index=False)
    write_json(OUT/'observation_access_log.json',reader.log)
    check_metadata(protected)
    for name,digest in source_hashes.items():
        require(sha(ROOT/name)==digest,'Source changed during replay')
    locked_bundle()
    sources = {key:dict(path=(OUT/name).relative_to(ROOT).as_posix(),sha256=sha(OUT/name),rows=rows)
        for key,name,rows in [('daily','primary_daily_local.parquet',len(primary_daily)),('predictions','strict_predictions_local.parquet',len(strict))]}
    audit = dict(status='passed',model_manifest=model_manifest,period=plan['historical_evaluation']['dates'],
        sources=sources,cost_bps_per_side=10,variant='constrained',nav_anchor_excluded=True,
        ledger_days=checks,paths=len(ledger),signals=len(schedule),market_dates=len(m['dates']),
        candidate_rows=len(features),strict_label_rows=len(strict),registered_events=len(terms),
        equal_weight_original_reproduced=True,model_fits=0,unseen_data_read=False,
        prior_artifacts_unchanged=True,protected_prior_files=len(protected))
    write_json(OUT/'replay_audit.json',audit)
    release = dict(authorized=True,authorization_record='User explicitly authorized frozen V2 2022-2025 historical replay; no fit or unseen access',
        plan_sha256=sha(PLAN),model_manifest=model_manifest,mode='historical',period=plan['historical_evaluation']['dates'],
        full_period_coverage=True,calendar_verified=True,events_resolved=True,ledger_verified=True,same_frozen_models=True,
        audit_record=dict(path=(OUT/'replay_audit.json').relative_to(ROOT).as_posix(),sha256=sha(OUT/'replay_audit.json')),sources=sources)
    write_json(OUT/'release.json',release)
    run_evaluation(OUT/'release.json')
    from equity_research.evaluation.replay_v2 import main as summarize
    summarize()
