"""V2 evaluation gate and paired statistics. Default CLI is observation-free.

Actual evaluation consumes a separately authorized, audited replay bundle; it
does not fetch data, fit models, or silently invoke the V1 test runner.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from equity_research.data.ingest import write_json
from equity_research.models.common import sha, load_lightgbm, circular_block_interval, daily_rank_ic
from equity_research.models.preprocessing import require, cross_sectional_rank
from equity_research.models.context_selection import contexts
from equity_research.models.ranking_selection import ranking_labels, ndcg_top
from equity_research.models.final_fit_v2 import (
    ROOT, OUT, PLAN, IDS, load_plan, read_json, model_matrix,
)


def locked_bundle():
    """Read only frozen configuration, models and verification metadata."""
    plan, _, _ = load_plan()
    require((OUT/'manifest.json').exists(), 'Final models not independently verified yet')
    manifest = read_json(OUT/'manifest.json')
    for path, digest in manifest.items():
        require(sha(ROOT/path) == digest, 'Locked model artifact changed: ' + path)
    bundle = read_json(OUT/'model_bundle.json')
    require(bundle['plan_sha256'] == sha(PLAN), 'Plan hash mismatch')
    require(read_json(OUT/'verification.json')['status'] == 'passed', 'Verification required')
    require([m['id'] for m in bundle['models']] == IDS, 'Incomplete six-model bundle')
    for saved, candidate in zip(bundle['models'], plan['models']):
        for key in ['id', 'family', 'features', 'parameters', 'role']:
            require(saved[key] == candidate[key], 'Bundle differs from frozen design')
    return plan, bundle, manifest


def preflight():
    plan, bundle, manifest = locked_bundle()
    return dict(status='model_bundle_ready_evaluation_still_closed', models=len(bundle['models']),
                verified_artifacts=len(manifest), observations_read=0, historical_replay_executed=False,
                historical_period=plan['historical_evaluation']['dates'],
                unseen_period=plan['unseen_evaluation']['planned_period'],
                unseen_eligibility=plan['unseen_evaluation']['eligibility_status'],
                requires='Separate authorization and audited full-period replay bundle; no automatic market IO')


def validate_release(release, plan, model_manifest):
    """Fail before opening observation sources; booleans alone cannot unlock IO."""
    require(release is not None, 'Evaluation sealed: explicit audited release required')
    require(release.get('authorized') is True and bool(release.get('authorization_record')), 'Separate evaluation authorization required')
    require(release.get('plan_sha256') == sha(PLAN) and release.get('model_manifest') == model_manifest, 'Release must bind exact plan and six models')
    mode = release.get('mode')
    if mode == 'historical':
        period = plan['historical_evaluation']['dates']
    elif mode == 'unseen':
        require(release.get('access_audit_status') == 'eligible' and bool(release.get('access_audit_record')), 'Unseen eligibility pending')
        fallback = release.get('fallback_trigger')
        require(fallback in [None, '2026_used_in_research', '2026_early_performance_requested'], 'Invalid fallback trigger')
        period = (plan['unseen_evaluation']['fallback_if_already_used']['period'] if fallback
                  else plan['unseen_evaluation']['planned_period'])
        require(pd.Timestamp(release['as_of']) > pd.Timestamp(period[1]), 'Incomplete calendar year')
    else:
        raise ValueError('Unknown evaluation mode')
    require(release.get('period') == period, 'Frozen evaluation dates changed')
    for key in ['full_period_coverage', 'calendar_verified', 'events_resolved', 'ledger_verified', 'same_frozen_models']:
        require(release.get(key) is True, 'Missing evaluation gate: ' + key)
    require(bool(release.get('audit_record')), 'Audited replay provenance required')
    return tuple(map(pd.Timestamp, period))


def check_dates(frame, period, date_column='signal_date'):
    dates = pd.to_datetime(frame[date_column])
    require(len(frame)>0 and dates.notna().all() and dates.between(*period).all(), 'Evaluation date breach')


def predict_frame(frame, bundle, period):
    """Predict an explicitly supplied eligible-date population without target access.

    The authorized replay adapter must read bounded point-in-time features first;
    a caller cannot slip target columns into this interface.
    """
    check_dates(frame, period)
    require(not any(c.startswith('y_') or c.startswith('label_') for c in frame), 'Prediction input must be feature-only')
    require(not frame.duplicated(['signal_date', 'PERMNO']).any(), 'Duplicate prediction keys')
    f = contexts(frame.sort_values(['signal_date', 'PERMNO']).reset_index(drop=True), minimum=5)
    ranked = cross_sectional_rank(f, bundle['original_features'])
    states = {}
    for kind, record in bundle['preprocessing'].items():
        require(sha(ROOT/record['path']) == record['sha256'], 'Preprocessing hash changed')
        states[kind] = read_json(ROOT/record['path'])
    predictions = f[['PERMNO', 'signal_date', 'signal_eligible']].copy()
    for candidate in bundle['models']:
        require(sha(ROOT/candidate['path']) == candidate['sha256'], 'Model hash changed')
        model = joblib.load(ROOT/candidate['path']) if candidate['family']=='ridge' else load_lightgbm(ROOT/candidate['path'])
        X = model_matrix(f, ranked, states, candidate)
        predictions[candidate['id']] = model.predict(X)
    base = bundle['baseline']
    require(sha(ROOT/base['path']) == base['sha256'], 'Baseline state changed')
    predictions['momentum20'] = f.momentum_20.replace([np.inf, -np.inf], np.nan).fillna(read_json(ROOT/base['path'])['momentum_missing'])
    require(np.isfinite(predictions[IDS+['momentum20']]).all().all(), 'Nonfinite frozen predictions')
    return predictions


def primary_comparison(daily, plan):
    """Exactly one primary contrast; input excludes initial zero-return NAV anchor."""
    require(not daily.duplicated(['date', 'model']).any(), 'Duplicate daily model keys')
    require(set(daily.model) == set(IDS), 'All six designs required')
    wide = daily.pivot(index='date', columns='model', values='return_net').sort_index()
    require(np.isfinite(wide.to_numpy()).all(), 'All six must cover identical finite dates')
    require((wide.to_numpy() >= -1).all(), 'Invalid portfolio return')
    a, b = plan['evaluation']['primary_contrast']
    cfg = plan['evaluation']['primary_interval']
    delta = (wide[a]-wide[b]).to_numpy()*252
    adequate = len(wide) >= plan['evaluation']['adequacy']['minimum_daily_returns']
    if len(wide) < cfg['block_days']:
        return dict(annual_mean_net_difference=float(delta.mean()), ci_low=None, ci_high=None,
                    paired_days=len(wide), adequate_daily_coverage=False, conclusion='descriptive_only')
    interval = circular_block_interval(delta, cfg['block_days'], cfg['replications'], cfg['seed'])
    return dict(annual_mean_net_difference=interval['mean_difference'], ci_low=interval['ci_low'],
                ci_high=interval['ci_high'], paired_days=len(wide), adequate_daily_coverage=adequate,
                conclusion=('supports_predeclared_improvement' if adequate and interval['ci_low']>0
                            else 'insufficient_evidence_or_deterioration' if adequate else 'descriptive_only'))


def prediction_metrics(frame, plan, period):
    """Continuous IC and frozen relevance NDCG on strict, shared eligible rows."""
    check_dates(frame, period)
    require(not frame.duplicated(['signal_date', 'PERMNO']).any(), 'Duplicate scored keys')
    require(frame.label_end_date.le(period[1]).all() and frame.label_end_date.gt(frame.signal_date).all(), 'Target horizon breach')
    require(frame.signal_eligible.all() and frame.label_status.isin(['ok', 'ok_bridged']).all(), 'Non-strict evaluation target')
    require(np.isfinite(frame[IDS+['y_excess_5d']]).all().all(), 'Missing score/target')
    rows = []
    for name in IDS:
        ic = daily_rank_ic(frame.signal_date, frame[name], frame.y_excess_5d)
        require(np.isfinite(ic).all(), 'Undefined IC; do not silently drop dates')
        for date, group in frame.sort_values('signal_date').groupby('signal_date'):
            relevance, _ = ranking_labels(group)
            rows.append(dict(model=name, signal_date=date, rank_ic=float(ic.loc[date]),
                             ndcg_top10=ndcg_top(relevance, group[name], group.PERMNO)))
    return pd.DataFrame(rows)


def run_evaluation(release_path):
    """Evaluate already audited replay outputs; never starts a backtest or fit.

    Release.sources must contain two date-bounded parquet datasets: `daily`
    (six constrained, 10 bps net paths, NAV anchor excluded) and `predictions`
    (strict scored targets). Their provenance is bound to a separate replay audit.
    """
    plan, bundle, manifest = locked_bundle()
    release = read_json(Path(release_path))
    period = validate_release(release, plan, manifest)
    audit_path = ROOT/release['audit_record']['path']
    require(sha(audit_path) == release['audit_record']['sha256'], 'Replay audit hash mismatch')
    audit = read_json(audit_path)
    require(audit['status'] == 'passed' and audit['model_manifest'] == manifest and
            audit['period'] == release['period'] and audit['sources'] == release['sources'], 'Replay audit binding mismatch')
    require(audit['cost_bps_per_side'] == 10 and audit['variant'] == 'constrained' and
            audit['nav_anchor_excluded'] is True, 'Primary portfolio protocol mismatch')
    data = {}
    specs = {'daily': ['date', 'model', 'return_net'], 'predictions':
             ['PERMNO', 'signal_date', 'signal_eligible', 'label_status', 'label_end_date', 'y_excess_5d']+IDS}
    for kind, columns in specs.items():
        record = release['sources'][kind]
        path = (ROOT/record['path']).resolve()
        require(path.is_relative_to(ROOT/'outputs') and path.suffix == '.parquet', 'Unregistered evaluation output')
        require(sha(path) == record['sha256'], 'Replay source hash mismatch')
        date = 'date' if kind=='daily' else 'signal_date'
        filters = [(date, '>=', period[0]), (date, '<=', period[1])]
        if kind=='predictions':
            filters.append(('label_end_date', '<=', period[1]))
        data[kind] = pd.read_parquet(path, columns=columns, filters=filters)
        check_dates(data[kind], period, date)
        require(len(data[kind]) == record['rows'], 'Filtered replay coverage changed')
    metrics = prediction_metrics(data['predictions'], plan, period)
    primary = primary_comparison(data['daily'], plan)
    signals = metrics.signal_date.nunique()
    primary['valid_signal_dates'] = signals
    if signals < plan['evaluation']['adequacy']['minimum_signal_dates']:
        primary['conclusion'] = 'descriptive_only'
    destination = ROOT/'outputs'/f"evaluation_v2_{release['mode']}_{period[0].year}"
    require(not destination.exists(), 'Preserve existing evaluation; no implicit rerun')
    destination.mkdir()
    metrics.to_csv(destination/'daily_prediction_metrics.csv', index=False)
    write_json(destination/'primary_comparison.json', primary)
    write_json(destination/'release.json', release)
    write_json(destination/'verification.json', dict(status='completed', fits=0,
        label=plan['historical_evaluation']['label'] if release['mode']=='historical' else 'audited new holdout',
        model_manifest=manifest, period=release['period']))
    return primary
