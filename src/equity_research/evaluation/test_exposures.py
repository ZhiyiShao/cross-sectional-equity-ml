"""Post-test holdings attribution. Reads saved scores/positions; never fits or trades."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from equity_research.paths import ROOT
from equity_research.models.common import sha, daily_rank_ic
from equity_research.models.preprocessing import require, frame_digest
from equity_research.evaluation.exposures import (
    FEATURES, GROUPS, sic_group, division, style_snapshots, attach_features,
    security_ledger, summarize, contribution_tables,
)

BT = ROOT / 'outputs/frozen_test_v1'
OUT = ROOT / 'outputs/test_exposures_v1'


def check_hashes(hashes):
    for name, digest in hashes.items():
        path = ROOT / Path(name.replace('\\', '/'))
        require(sha(path) == digest, 'Frozen source changed: ' + name)


def load_inputs():
    hashes = json.loads((BT / 'manifest.json').read_text(encoding='utf-8'))
    # Also protect the saved final models and preprocessing, without loading estimators.
    for path in (ROOT / 'outputs/final_models_v1').rglob('*'):
        if path.is_file():
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    check_hashes(hashes)
    d = pd.read_csv(BT / 'daily_portfolios.csv', parse_dates=['date'])
    d = d.loc[d.scenario.eq('lag_2') & d.cost_bps.eq(10)].sort_values(['model', 'date']).copy()
    require(d.date.between('2022-01-01', '2025-12-31').all(), 'Test date breach')
    d['prev_nav'] = d.groupby('model').nav.shift().fillna(1.)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    h = pd.read_parquet(BT / 'holdings_local.parquet')
    t = pd.read_parquet(BT / 'trades_local.parquet')
    e = pd.read_parquet(BT / 'events_local.parquet')
    e = e.loc[e.scenario.eq('lag_2') & e.cost_bps.eq(10)]
    require(h.date.isin(dates).all() and t.date.isin(dates).all(), 'Holdings date mismatch')
    registry = pd.DataFrame(json.loads((BT / 'event_registry.json').read_text(encoding='utf-8'))['events']).set_index('permno')
    feature_path = ROOT / 'data/processed/research_v1/features_only.parquet'
    hashes[feature_path.relative_to(ROOT).as_posix()] = sha(feature_path)
    f = pd.read_parquet(feature_path, columns=['PERMNO', 'signal_date', 'signal_eligible'] + FEATURES,
                        filters=[('signal_date', '>=', dates.min()), ('signal_date', '<=', dates.max())])
    f = style_snapshots(f)
    pred = pd.read_parquet(BT / 'predictions_local.parquet')
    require(pred.signal_date.isin(dates).all(), 'Prediction date mismatch')
    source = ROOT / 'data/interim/crsp_audit_v1'
    parts = json.loads((source / 'ingest.json').read_text(encoding='utf-8'))['parts']
    ids = set(h.PERMNO) | set(t.PERMNO)
    industry, metadata = [], []
    for name in parts:
        path = source / name
        md = pq.read_metadata(path)
        low = min(md.row_group(i).column(0).statistics.min for i in range(md.num_row_groups))
        high = max(md.row_group(i).column(0).statistics.max for i in range(md.num_row_groups))
        wanted = sorted(p for p in ids if low <= p <= high)
        if wanted:
            industry.append(pd.read_parquet(path, columns=['PERMNO', 'DlyCalDt', 'SICCD', 'SecInfoStartDt', 'SecInfoEndDt', 'Ticker'],
                filters=[('PERMNO', 'in', wanted), ('DlyCalDt', '>=', dates.min()), ('DlyCalDt', '<=', dates.max())]))
            metadata.append(dict(path=path.relative_to(ROOT).as_posix(), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    raw = pd.concat(industry, ignore_index=True)
    require(not raw.duplicated(['PERMNO', 'DlyCalDt']).any(), 'Duplicate industry records')
    raw['industry'] = [sic_group(c, a, b, date) for c, a, b, date in
                       zip(raw.SICCD, raw.SecInfoStartDt, raw.SecInfoEndDt, raw.DlyCalDt)]
    raw['division'] = raw.industry.map(division)
    audit = dict(source_hashes=hashes, market_parts=metadata, features_digest=frame_digest(f),
                 industry_digest=frame_digest(raw), model_fits=0, new_predictions=0, backtest_runs=0,
                 scenario='lag_2', cost_bps=10, interpretation='post-test descriptive attribution, not causal factor attribution')
    return d, h, t, e, registry, dates, f, pred, raw, audit


def period_contributions(p, d):
    """PnL divided by period opening NAV reconciles exactly, including fees.

    Monthly-start contributions sum to that month's return. Annual-start monthly
    contributions sum to the full year's return; monthly returns cannot be added.
    """
    q = p.assign(year=p.date.dt.year, month=p.date.dt.strftime('%Y-%m'))
    daily = d.loc[d.date.gt(d.date.min())].copy()
    daily['year'] = daily.date.dt.year
    daily['month'] = daily.date.dt.strftime('%Y-%m')
    starts = daily.groupby(['model', 'year']).prev_nav.first().rename('year_start_nav').reset_index()
    monthly = daily.groupby(['model', 'year', 'month']).agg(
        month_start_nav=('prev_nav', 'first'), end_nav=('nav', 'last'),
        net_pnl=('pnl', 'sum'), costs=('fees', 'sum'), days=('date', 'size'),
    ).reset_index().merge(starts, on=['model', 'year'], validate='many_to_one')
    monthly['monthly_return'] = monthly.end_nav / monthly.month_start_nav - 1
    monthly['annual_contribution'] = (monthly.end_nav - monthly.month_start_nav) / monthly.year_start_nav
    monthly['annual_fee_contribution'] = -monthly.costs / monthly.year_start_nav
    q = q.merge(monthly[['model', 'month', 'month_start_nav', 'year_start_nav', 'days']],
                on=['model', 'month'], validate='many_to_one')
    q['monthly_contribution'] = q.net_pnl / q.month_start_nav
    q['annual_contribution'] = q.net_pnl / q.year_start_nav
    q['annual_fee_contribution'] = -q.cost / q.year_start_nav
    results = []
    for dimension in ['industry', 'PERMNO']:
        table = q.groupby(['model', 'year', 'month', dimension]).agg(
            monthly_contribution=('monthly_contribution', 'sum'),
            annual_contribution=('annual_contribution', 'sum'),
            annual_fee_contribution=('annual_fee_contribution', 'sum'),
            average_weight=('equity_weight', 'sum'), days=('days', 'first'),
        ).reset_index()
        table['average_weight'] /= table.days
        total = table.groupby(['model', 'month']).monthly_contribution.sum().sort_index()
        expected = monthly.set_index(['model', 'month']).monthly_return.sort_index()
        np.testing.assert_allclose(total, expected, atol=1e-10, rtol=1e-10)
        results.append(table)
    return monthly, *results


def score_diagnostics(pred, f):
    joined = pred.merge(f[['PERMNO', 'signal_date'] + FEATURES], on=['PERMNO', 'signal_date'], validate='one_to_one')
    joined = joined.loc[joined.signal_eligible]
    rows = []
    for model in [c for c in pred if c not in ['PERMNO', 'signal_date', 'signal_eligible']]:
        for feature in FEATURES:
            ic = daily_rank_ic(joined.signal_date, joined[model], joined[feature])
            rows.append(ic.rename('rank_correlation').rename_axis('signal_date').reset_index().assign(model=model, feature=feature))
    result = pd.concat(rows, ignore_index=True)
    result['year'] = result.signal_date.dt.year
    return result.groupby(['model', 'year', 'feature']).agg(
        mean_rank_correlation=('rank_correlation', 'mean'), dates=('rank_correlation', 'count')).reset_index()


def plot_diagnostics(monthly, monthly_styles, contributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = {'raw__ridge_03': 'Raw Ridge', 'cross_sectional_rank__ridge_13': 'Rank Ridge',
             'cross_sectional_rank__lightgbm_10': 'Rank LightGBM', 'equal_weight': 'Universe EW'}
    ridge, lgb = 'raw__ridge_03', 'cross_sectional_rank__lightgbm_10'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    returns = monthly.loc[monthly.year.eq(2022)].pivot(index='month', columns='model', values='annual_contribution')
    gap = (returns[lgb] - returns[ridge]) * 100
    axes[0, 0].bar(gap.index.str[-2:], gap, color=np.where(gap < 0, '#c44e52', '#4c72b0'))
    axes[0, 0].set(title='Monthly contributions to 2022 return gap', ylabel='Percentage points')
    industries = contributions.loc[contributions.year.eq(2022) & contributions.dimension.eq('industry')]
    wide = industries.pivot(index='category', columns='model', values='contribution').fillna(0.)
    diff = (wide[lgb] - wide[ridge]) * 100
    top = diff.loc[diff.abs().nlargest(8).index].sort_values()
    axes[0, 1].barh(['SIC ' + str(s) for s in top.index], top, color=np.where(top < 0, '#c44e52', '#4c72b0'))
    axes[0, 1].set(title='Industry contributions to 2022 return gap', xlabel='Percentage points')
    for model, name in names.items():
        rows = monthly_styles.loc[monthly_styles.model.eq(model) & monthly_styles.month.str.startswith('2022')]
        axes[1, 0].plot(rows.month.str[-2:], rows.market_beta_60, label=name)
        axes[1, 1].plot(rows.month.str[-2:], rows.volatility_20 * 100, label=name)
    axes[1, 0].set(title='Holdings-weighted trailing 60-day beta', xlabel='Month')
    axes[1, 1].set(title='Holdings-weighted stock volatility (20-day)', ylabel='Annualized %', xlabel='Month')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('2022 frozen-test diagnostics | Rank LightGBM minus Raw Ridge | 10 bps/side, lag_2\nDescriptive holdings attribution; no retraining or parameter changes', fontsize=12)
    fig.savefig(OUT / '2022_diagnostics.png', dpi=150)
    plt.close(fig)


def main():
    require(not OUT.exists() or not any(OUT.iterdir()), 'Output already exists; preserve completed analysis')
    print('Read frozen portfolios and feature/industry snapshots; no fitting or new trades.', flush=True)
    d, h, t, events, registry, dates, features, pred, raw, audit = load_inputs()
    print('Sources verified; reconcile security PnL including corporate actions.', flush=True)
    p = security_ledger(d, h, t, dates, events, registry)
    matched = attach_features(p[['asof_date', 'PERMNO']].drop_duplicates(), features)
    industry = raw.rename(columns={'DlyCalDt': 'asof_date'})[['asof_date', 'PERMNO', 'industry', 'division', 'Ticker']]
    matched = matched.merge(industry, on=['asof_date', 'PERMNO'], how='left', validate='one_to_one')
    p = p.merge(matched, on=['asof_date', 'PERMNO'], how='left', validate='many_to_one')
    p[['industry', 'division']] = p[['industry', 'division']].fillna('Unknown')
    print('Calculate holdings exposures and period contributions.', flush=True)
    daily, yearly, monthly_styles = summarize(p, d)
    contributions, annual = contribution_tables(p, d)
    monthly, monthly_industry, monthly_stock = period_contributions(p, d)
    annual_source = pd.read_csv(BT / 'annual_metrics.csv')
    annual_source = annual_source.loc[annual_source.scenario.eq('lag_2') & annual_source.cost_bps.eq(10)]
    check = annual.merge(annual_source[['model', 'year', 'total_return']], on=['model', 'year'], validate='one_to_one')
    np.testing.assert_allclose(check.net_return, check.total_return, atol=1e-10)
    # Names are historical labels, not present-day classifications.
    names = p.dropna(subset=['Ticker']).groupby('PERMNO').Ticker.agg(lambda x: '/'.join(sorted(set(x)))).rename('historical_tickers')
    monthly_stock = monthly_stock.join(names, on='PERMNO')
    score_styles = score_diagnostics(pred, features)
    ic = pd.read_csv(BT / 'daily_rank_ic.csv', parse_dates=['date']).rename(columns={'date': 'signal_date'})
    ic = ic.melt(id_vars='signal_date', var_name='model', value_name='rank_ic')
    ic['month'] = ic.signal_date.dt.strftime('%Y-%m')
    monthly_ic = ic.groupby(['model', 'month']).agg(mean_rank_ic=('rank_ic', 'mean'), signals=('rank_ic', 'count')).reset_index()
    check_hashes(audit['source_hashes'])
    for part in audit['market_parts']:
        stat = (ROOT / part['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (part['size'], part['mtime_ns']), 'Industry source changed')
    audit.update(daily_reconciliations=len(d), annual_reconciliations=len(check), monthly_reconciliations=len(monthly) * 2,
                 ledger_rows=len(p), original_artifacts_unchanged=True,
                 minimum_annual_style_coverage=float(yearly[[c + '_coverage' for c in FEATURES]].min().min()),
                 maximum_annual_unknown_industry_weight=float(yearly.industry_unknown_weight.max()),
                 source_code_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                     'src/equity_research/evaluation/exposures.py': sha(ROOT / 'src/equity_research/evaluation/exposures.py')})
    OUT.mkdir(parents=True, exist_ok=True)
    tables = dict(daily_exposures=daily, yearly_exposures=yearly, monthly_exposures=monthly_styles,
                  contributions=contributions, annual_reconciliation=annual, monthly_returns=monthly,
                  monthly_industry_contributions=monthly_industry, monthly_stock_contributions_local=monthly_stock,
                  score_style_correlations=score_styles, monthly_rank_ic=monthly_ic)
    for name, table in tables.items():
        table.to_csv(OUT / (name + '.csv'), index=False)
    p.to_parquet(OUT / 'exposure_ledger_local.parquet', index=False)
    plot_diagnostics(monthly, monthly_styles, contributions)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print('All daily, monthly and annual contributions reconcile. Original artifacts unchanged.', flush=True)
    print(yearly.loc[yearly.year.eq(2022), ['model', 'geometric_market_cap_bn'] + FEATURES].to_string(index=False))


if __name__ == '__main__':
    main()
