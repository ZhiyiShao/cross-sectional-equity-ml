"""Audit CIZ semantics and construct historical monthly universes locally.

No model is trained and no strategy performance is inspected by this script.
Missing returns and delisting rows remain in the normalized source parts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.data.ingest import ROOT, write_json

SINGLE_DAY = ['D1', 'D2', 'D3', 'D4', 'DU']


def filter_codes(d, filters):
    result = pd.Series(True, index=d.index)
    for col, values in filters.items():
        result &= d[col].isin(values)
    return result


def regular_trade(d, cfg):
    return (filter_codes(d, cfg['regular_trading_filters']) &
            d.DlyClose.gt(0) & d.DlyVol.ge(0) &
            np.isfinite(d.DlyClose) & np.isfinite(d.DlyVol))


def monthly_snapshots(g, calendar, formation_dates, cfg):
    """Backward-only statistics. Reindex on MARKET days, not observed row count."""
    g = g.set_index('DlyCalDt').sort_index()
    idx = calendar[(calendar >= g.index.min()) & (calendar <= g.index.max())]
    x = g.reindex(idx)
    valid = regular_trade(x, cfg)
    dollars = (x.DlyClose * x.DlyVol).where(valid)
    w = cfg['liquidity_window_trading_days']
    x['valid_liquidity_days'] = dollars.rolling(w, min_periods=1).count()
    x['adv60_usd'] = dollars.rolling(w, min_periods=cfg['min_valid_liquidity_days']).mean()
    observed = (x.DlyDelFlg.eq('N') & x.ConditionalType.eq('RW') &
                x.TradingStatusFlg.eq('A') & x.DlyPrc.gt(0) & x.DlyPrcFlg.isin(['TR', 'BA']))
    x['history_days'] = observed.cumsum()
    x['eligible_security'] = filter_codes(x, cfg['security_filters'])
    x['eligible_trading'] = regular_trade(x, cfg)
    x['eligible_history'] = x.history_days.ge(cfg['min_observed_history_days'])
    x['eligible_price'] = x.DlyClose.ge(cfg['min_close_usd'])
    x['eligible_liquidity'] = (x.valid_liquidity_days.ge(cfg['min_valid_liquidity_days']) & x.adv60_usd.gt(0))
    cols = ['PERMNO','PERMCO','Ticker','PrimaryExch','DlyClose','DlyCap','history_days',
            'valid_liquidity_days','adv60_usd','eligible_security','eligible_trading',
            'eligible_history','eligible_price','eligible_liquidity']
    # No stale last-observed value is carried to a market month end.
    result = x.loc[x.index.isin(formation_dates) & x.PERMNO.notna(), cols].copy()
    result.index.name = 'formation_date'
    return result.reset_index()


def select_month(s, cfg):
    mask = pd.Series(True, index=s.index)
    funnel = {'observed': len(s)}
    for name in ['security','trading','history','price','liquidity']:
        mask &= s['eligible_' + name]
        funnel[name] = int(mask.sum())
    eligible = s.loc[mask].sort_values(['adv60_usd','PERMNO'], ascending=[False,True])
    one_share = eligible.drop_duplicates('PERMCO', keep='first')
    funnel['unique_companies'] = len(one_share)
    result = one_share.head(cfg['max_securities']).copy()
    result['liquidity_rank'] = np.arange(1, len(result) + 1)
    funnel['selected'] = len(result)
    return result, funnel


def semantic_audit(d, calendar_positions, stats, samples):
    group = d.groupby('PERMNO', sort=False)
    prev = group[['DlyPrc','DlyCumFacPr','DlyCumFacShr','ShrOut','DlyDelFlg','DlyCalDt']].shift()
    normal = d.DlyDelFlg.eq('N')
    common = (d.SecurityType.eq('EQTY') & d.SecuritySubType.eq('COM') &
              d.ShareType.eq('NS') & d.USIncFlg.eq('Y') & d.IssuerType.isin(['CORP','ACOR']))
    unit_ok = normal & d.DlyPrc.gt(0) & d.ShrOut.gt(0) & d.DlyCap.notna()
    ratio = d.DlyCap / (d.DlyPrc * d.ShrOut)
    stats['cap_unit_checked'] += int(unit_ok.sum())
    stats['cap_unit_relative_error_gt_1e_4'] += int((unit_ok & ((ratio-1).abs() > 1e-4)).sum())
    # Deterministic sparse samples only for aggregate quantiles, not output observations.
    samples['cap_ratio'].extend(ratio[unit_ok].iloc[::500].tolist())
    adjacent = (d.DlyCalDt.map(calendar_positions) - prev.DlyCalDt.map(calendar_positions)).eq(1)
    single = d.DlyRetDurFlg.isin(SINGLE_DAY)
    comparable = (normal & prev.DlyDelFlg.eq('N') & single & adjacent & d.DlyRetMissFlg.eq('NA') & d.DlyPrc.gt(0) &
                  prev.DlyPrc.gt(0) & d.DlyCumFacPr.gt(0) & prev.DlyCumFacPr.gt(0) & d.DlyRetx.notna())
    factor_ratio = prev.DlyCumFacPr / d.DlyCumFacPr
    divide_return = d.DlyPrc / prev.DlyPrc * factor_ratio - 1
    multiply_return = d.DlyPrc / prev.DlyPrc / factor_ratio - 1
    changed = comparable & ~np.isclose(factor_ratio, 1, rtol=1e-8, atol=1e-10)
    split_like = changed & np.isclose(factor_ratio, prev.DlyCumFacShr / d.DlyCumFacShr, rtol=1e-6, atol=1e-8)
    for name, mask in [('all_comparable',comparable), ('factor_change',changed),('split_like',split_like)]:
        stats[name + '_rows'] += int(mask.sum())
        stats[name + '_divide_matches_1e_5'] += int((mask & ((divide_return-d.DlyRetx).abs() <= 1e-5)).sum())
        stats[name + '_multiply_matches_1e_5'] += int((mask & ((multiply_return-d.DlyRetx).abs() <= 1e-5)).sum())
    share_event = split_like & d.ShrOut.gt(0) & prev.ShrOut.gt(0)
    share_ratio = prev.DlyCumFacShr / d.DlyCumFacShr
    stats['share_split_checked'] += int(share_event.sum())
    stats['share_split_ratio_matches_1pct'] += int((share_event & np.isclose(d.ShrOut / prev.ShrOut, share_ratio, rtol=.01, atol=1e-8)).sum())
    # DlyFacPrc is a daily gross factor, not the legacy factor-minus-one.
    event = comparable & d.DlyFacPrc.ne(1) & d.DlyFacPrc.notna()
    stats['daily_factor_event_rows'] += int(event.sum())
    stats['daily_factor_cumulative_ratio_matches'] += int((event & np.isclose(d.DlyFacPrc, factor_ratio, rtol=1e-6, atol=1e-8)).sum())
    tr = d.DlyPrcFlg.eq('TR') & normal
    stats['trade_price_close_mismatches'] += int((tr & ((d.DlyPrc-d.DlyClose).abs() > 1e-6)).sum())
    stats['trade_price_close_missing'] += int((tr & d.DlyClose.isna()).sum())
    ohlc = tr & d[['DlyHigh','DlyLow','DlyClose']].notna().all(axis=1)
    stats['ohlc_invalid_bounds'] += int((ohlc & ((d.DlyHigh < d.DlyLow) | (d.DlyClose > d.DlyHigh+1e-6) | (d.DlyClose < d.DlyLow-1e-6))).sum())
    stats['numeric_return_with_non_NA_flag'] += int((d.DlyRet.notna() & d.DlyRetMissFlg.ne('NA')).sum())
    stats['missing_return_with_NA_flag'] += int((d.DlyRet.isna() & d.DlyRetMissFlg.eq('NA')).sum())
    ordinary_bad = normal & (~single | d.DlyRet.isna() | d.DlyRetMissFlg.ne('NA'))
    stats['ordinary_non_single_or_missing_return'] += int(ordinary_bad.sum())
    stats['common_non_single_or_missing_return'] += int((ordinary_bad & common).sum())
    delist = d.DlyDelFlg.eq('Y')
    stats['delist_rows'] += int(delist.sum())
    stats['delist_missing_return_rows'] += int((delist & d.DlyRet.isna()).sum())
    stats['delist_numeric_but_flagged_rows'] += int((delist & d.DlyRet.notna() & d.DlyRetMissFlg.ne('NA')).sum())
    stats['delist_common_previous_row'] += int((delist & common.groupby(d.PERMNO).shift(fill_value=False)).sum())
    stats['delist_not_next_market_date'] += int((delist & prev.DlyCalDt.notna() & ~adjacent).sum())
    stats['delist_with_return_not_minus_100pct'] += int((delist & d.DlyRet.notna() & d.DlyRet.ne(-1)).sum())
    stats['one_day_flag_but_nonadjacent_rows'] += int((normal & single & prev.DlyCalDt.notna() & ~adjacent).sum())
    # Retain event-level exceptions locally; never print their observations.
    exceptions = d.loc[delist | (common & ordinary_bad) | (split_like & ((divide_return-d.DlyRetx).abs()>1e-5))].copy()
    exceptions['audit_delist'] = delist.loc[exceptions.index]
    exceptions['audit_return_gap'] = (common & ordinary_bad).loc[exceptions.index]
    exceptions['audit_adjustment_mismatch'] = (split_like & ((divide_return-d.DlyRetx).abs()>1e-5)).loc[exceptions.index]
    return exceptions


def build(source, out):
    cfg = json.loads((ROOT / 'config/universe.json').read_text(encoding='utf-8'))
    ing = json.loads((source / 'ingest.json').read_text(encoding='utf-8'))
    if (out / 'semantic_audit.json').exists():
        raise RuntimeError('Completed build exists. Use a new output directory to avoid mixing versions.')
    for k in ['outside_classification_interval','negative_price_rows','negative_volume_rows',
              'return_below_minus_one','market_conflicting_dates','market_missing_rows']:
        if ing[k]:
            raise ValueError(f'Unresolved ingestion failure: {k}={ing[k]}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'exceptions').mkdir(exist_ok=True)
    cal = pd.read_parquet(source / 'calendar.parquet')
    calendar = pd.DatetimeIndex(pd.to_datetime(cal.date))
    positions = dict(zip(calendar, range(len(calendar))))
    month_ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period('M')).max()
    month_starts = pd.Series(calendar, index=calendar).groupby(calendar.to_period('M')).min()
    first = pd.Period(cfg['start_month'], 'M')
    last = pd.Period(cfg['end_month'], 'M')
    formations = month_ends.loc[(month_ends.index >= first-1) & (month_ends.index < last)]
    stats, samples = Counter(), {'cap_ratio':[]}
    snapshots = []
    for i, name in enumerate(ing['parts']):
        d = pd.read_parquet(source / name)
        exceptions = semantic_audit(d, positions, stats, samples)
        if not exceptions.empty:
            exceptions.to_parquet(out / 'exceptions' / name, index=False, compression='zstd')
        for _, g in d.groupby('PERMNO', sort=False):
            snapshots.append(monthly_snapshots(g, calendar, formations.values, cfg))
        if (i+1) % 10 == 0:
            print(f'Audited {i+1}/{len(ing["parts"])} parts; monthly historical statistics built.', flush=True)
    all_snapshots = pd.concat(snapshots, ignore_index=True)
    all_snapshots.to_parquet(out / 'monthly_candidates.parquet', index=False, compression='zstd')
    selections, funnels = [], []
    for formation, s in all_snapshots.groupby('formation_date', sort=True):
        selected, funnel = select_month(s, cfg)
        month = formation.to_period('M') + 1
        selected['effective_month'] = str(month)
        selected['effective_from'] = month_starts.loc[month]
        selected['effective_to'] = month_ends.loc[month]
        selected['formation_date'] = formation
        for c in ['PERMNO','PERMCO','history_days','valid_liquidity_days']:
            selected[c] = selected[c].astype('int64')
        selections.append(selected)
        funnels.append({'effective_month':str(month), 'formation_date':str(formation.date()), **funnel})
    universe = pd.concat(selections, ignore_index=True)
    assert not universe.duplicated(['effective_month','PERMNO']).any()
    assert not universe.duplicated(['effective_month','PERMCO']).any()
    assert (universe.formation_date < universe.effective_from).all()
    assert universe.effective_month.nunique() == len(pd.period_range(first,last,freq='M'))
    universe.to_parquet(out / 'monthly_universe.parquet', index=False, compression='zstd')
    universe.to_csv(out / 'monthly_universe.csv', index=False)
    pd.DataFrame(funnels).to_csv(out / 'universe_funnel.csv', index=False)
    report = dict(stats)
    report.update(universe_months=int(universe.effective_month.nunique()),
                  universe_rows=len(universe), unique_selected_permnos=int(universe.PERMNO.nunique()),
                  min_month_size=int(universe.groupby('effective_month').size().min()),
                  max_month_size=int(universe.groupby('effective_month').size().max()),
                  cap_ratio_sample_quantiles={str(q):float(np.quantile(samples['cap_ratio'],q)) for q in [.01,.5,.99]},
                  source_sha256=ing['sha256'], config=cfg,
                  status='universe_generated_return_exceptions_require_review')
    write_json(out / 'semantic_audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2), flush=True)
