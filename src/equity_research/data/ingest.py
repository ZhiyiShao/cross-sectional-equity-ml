"""Local, streaming CRSP audit. Never print licensed observations.

Stage ingest preserves flags literally (notably NA), removes only identical
export rows, and writes typed Parquet parts with complete PERMNO histories.
Conflicting keys are fatal; the original gzip is never modified.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.paths import ROOT
KEY = ['PERMNO', 'DlyCalDt']
NUMERIC = ['PERMNO', 'PERMCO', 'SICCD', 'DlyPrc', 'DlyCap', 'DlyRet',
           'DlyRetx', 'DlyFacPrc', 'DlyVol', 'DlyClose', 'DlyLow',
           'DlyHigh', 'DlyOpen', 'ShrOut', 'DlyCumFacPr', 'DlyCumFacShr', 'vwretd']
FLAGS = ['PrimaryExch', 'ConditionalType', 'TradingStatusFlg', 'USIncFlg',
         'IssuerType', 'SecurityType', 'SecuritySubType', 'ShareType',
         'DlyDelFlg', 'DlyPrcFlg', 'DlyRetMissFlg', 'DlyRetDurFlg']


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def deduplicate(frame):
    """Exact row duplicates only. Reject conflicting security/date keys."""
    unique = frame.drop_duplicates()
    conflicts = unique.duplicated(KEY, keep=False)
    return unique, int(len(frame) - len(unique)), unique.loc[conflicts]


def ingest(out, chunksize):
    source = ROOT / 'data/raw/crsp/crsp_ciz_daily_2014_2025.csv.gz'
    cfg = json.loads((ROOT / 'config/crsp_query.json').read_text(encoding='utf-8'))
    if (out / 'ingest.json').exists():
        raise RuntimeError('Completed ingestion already exists. Use its parts or a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != cfg['sha256']:
        raise ValueError('Raw file hash differs from the verified download.')
    counts = Counter()
    flags = {c: Counter() for c in FLAGS}
    missing, bad_numeric = Counter(), Counter()
    date_counts, market_min, market_max = Counter(), {}, {}
    files, carry, last_key = [], pd.DataFrame(), None
    securities = set()

    def save(frame):
        if frame.empty:
            return
        unique, removed, conflicts = deduplicate(frame)
        counts['exact_duplicate_rows_removed'] += removed
        if not conflicts.empty:
            conflicts.to_parquet(out / 'conflicting_keys.parquet', index=False)
            raise ValueError('Conflicting keys quarantined; universe generation stopped.')
        d = unique.copy()
        counts['clean_rows'] += len(d)
        securities.update(d.PERMNO.unique().tolist())
        for c in FLAGS:
            flags[c].update(d[c].value_counts(dropna=False).to_dict())
        for c in d.columns:
            missing[c] += int(d[c].eq('').sum())
        for c in NUMERIC:
            v = pd.to_numeric(d[c], errors='coerce')
            bad_numeric[c] += int((d[c].ne('') & v.isna()).sum())
            d[c] = v
        for c in ['PERMNO', 'PERMCO']:
            if d[c].isna().any() or (d[c] % 1 != 0).any():
                raise ValueError('Invalid security/company identifier.')
            d[c] = d[c].astype('int32')
        for c in ['DlyCalDt', 'SecInfoStartDt', 'SecInfoEndDt']:
            d[c] = pd.to_datetime(d[c], errors='raise')
        valid_interval = d.DlyCalDt.between(d.SecInfoStartDt, d.SecInfoEndDt)
        counts['outside_classification_interval'] += int((~valid_interval).sum())
        counts['negative_price_rows'] += int((d.DlyPrc < 0).sum())
        counts['negative_volume_rows'] += int((d.DlyVol < 0).sum())
        counts['return_below_minus_one'] += int((d.DlyRet < -1).sum())
        counts['return_flag_inconsistent'] += int((d.DlyRet.notna() != d.DlyRetMissFlg.eq('NA')).sum())
        for c in ['DlyCumFacPr', 'DlyCumFacShr']:
            counts[c + '_nonpositive'] += int((d[c] <= 0).sum())
        for dt, n in d.DlyCalDt.value_counts().items():
            date_counts[dt.strftime('%Y-%m-%d')] += int(n)
        for dt, g in d.groupby('DlyCalDt', sort=False)['vwretd']:
            key = dt.strftime('%Y-%m-%d')
            if g.notna().any():
                market_min[key] = min(market_min.get(key, np.inf), float(g.min()))
                market_max[key] = max(market_max.get(key, -np.inf), float(g.max()))
        counts['market_missing_rows'] += int(d.vwretd.isna().sum())
        name = f'part-{len(files):04d}.parquet'
        d.to_parquet(out / name, index=False, compression='zstd')
        files.append(name)

    # keep_default_na=False is essential: NA is a valid CRSP flag.
    for i, raw in enumerate(pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=chunksize)):
        counts['raw_rows'] += len(raw)
        idx = pd.MultiIndex.from_arrays([raw.PERMNO.astype('int64'), raw.DlyCalDt])
        if not idx.is_monotonic_increasing or (last_key is not None and idx[0] < last_key):
            raise ValueError('Source is not ordered by PERMNO/date; external sort required.')
        last_key = idx[-1]
        frame = pd.concat([carry, raw], ignore_index=True)
        final_id = frame.PERMNO.iloc[-1]
        carry = frame.loc[frame.PERMNO.eq(final_id)].copy()
        save(frame.loc[frame.PERMNO.ne(final_id)])
        if (i + 1) % 5 == 0:
            print(f'Ingested {counts["raw_rows"]:,} rows; aggregates only.', flush=True)
    save(carry)
    if counts['raw_rows'] != cfg['server_result_rows'] or any(bad_numeric.values()):
        raise ValueError('Row count or numeric parsing failed.')
    dates = sorted(date_counts)
    market_conflicts = [d for d in dates if market_min.get(d) != market_max.get(d)]
    pd.DataFrame({'date': dates, 'row_count': [date_counts[d] for d in dates],
                  'vwretd': [market_min.get(d, np.nan) for d in dates]}).to_parquet(out / 'calendar.parquet', index=False)
    report = dict(counts)
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  sha256=digest.hexdigest(), unique_permnos=len(securities),
                  first_date=dates[0], last_date=dates[-1], trading_dates=len(dates),
                  columns=raw.columns.tolist(), missing=dict(missing), bad_numeric=dict(bad_numeric),
                  flags={k: dict(v) for k,v in flags.items()}, market_conflicting_dates=len(market_conflicts),
                  parts=files, status='ingested_pending_semantic_audit')
    write_json(out / 'ingest.json', report)
    print(json.dumps({k: v for k,v in report.items() if k not in ['missing','bad_numeric','flags','parts','columns']}, indent=2), flush=True)
