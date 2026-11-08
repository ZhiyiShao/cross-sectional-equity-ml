"""Six one-shot final fits, bounded to 2015–2021; never opens evaluation data."""
import json
import platform
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from equity_research.paths import ROOT
from equity_research.data.ingest import write_json
from equity_research.models.common import sha, save_lightgbm, load_lightgbm
from equity_research.models.final_fit import read_final_inputs, START, END
from equity_research.models.context_selection import contexts
from equity_research.models.ranking_selection import ranking_labels
from equity_research.models.preprocessing import (
    require, cross_sectional_rank, fit_state, transform_features, frame_digest,
)

PLAN = ROOT / 'config/final_evaluation_v2.json'
OUT = ROOT / 'outputs/final_models_v2'
PUBLIC = ROOT / 'results/final_models_v2'
INDUSTRY = ROOT / 'outputs/context_selection_v2/development_features_local.parquet'
RECIPE = {'raw_clip_quantiles': [.005, .995], 'rank_clip_bounds': [-.5, .5]}
IDS = ['ridge_original', 'ridge_enhanced', 'lightgbm_original',
       'lightgbm_enhanced', 'ranker_original', 'ranker_enhanced']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_plan():
    """Check the preserved freeze copy and its source hashes before any data IO."""
    plan = read_json(PLAN)
    require(plan == read_json(ROOT/'results/v2_comparison/frozen_plan.json'), 'Frozen plan changed')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT/name) == digest, 'Frozen source changed: ' + name)
    require([c['id'] for c in plan['models']] == IDS, 'Six model identities changed')
    require(plan['final_fit']['dates'] == ['2015-01-01', '2021-12-31'], 'Fit dates changed')
    require(plan['budget']['candidate_fits_remaining'] == 0 and
            plan['budget']['future_final_fits_maximum'] == 6 and
            plan['budget']['fits_per_model'] == 1, 'Fit budget changed')
    original = read_json(ROOT/'config/features_labels.json')['features']
    context = list(read_json(ROOT/'config/context_features_v2.json')['context_features'])
    for c in plan['models']:
        expected = original + (context if c['id'].endswith('enhanced') else [])
        require(c['features'] == expected, 'Frozen feature order changed')
    return plan, original, context


def load_training(original):
    """Read targets with horizon pushdown; cached industries contain no new data."""
    frame, mask, access = read_final_inputs(original)
    columns = ['PERMNO', 'signal_date', 'industry']
    filters = [('signal_date', '>=', START), ('signal_date', '<=', END)]
    industry = pd.read_parquet(INDUSTRY, columns=columns, filters=filters)
    require(industry.signal_date.between(START, END).all(), 'Industry date breach')
    joined = frame.merge(industry, on=['PERMNO', 'signal_date'], how='left',
                         validate='one_to_one', indicator=True, sort=False)
    require(joined['_merge'].eq('both').all(), 'Industry coverage mismatch')
    require(joined.industry.notna().all(), 'Missing cached industry')
    joined = contexts(joined.drop(columns='_merge'), minimum=5)
    pd.testing.assert_frame_equal(joined[['PERMNO', 'signal_date']], frame[['PERMNO', 'signal_date']])
    require(np.isfinite(joined.loc[mask, 'y_excess_5d']).all(), 'Nonfinite qualified label')
    access.append(dict(source=str(INDUSTRY.relative_to(ROOT)), columns=columns,
                       filters=repr(filters), rows=len(industry),
                       min_signal=str(industry.signal_date.min()), max_signal=str(industry.signal_date.max()),
                       kind='frozen_development_industry', digest=frame_digest(industry)))
    return joined, mask, access


def fit_components(frame, mask, original, added):
    """Three shared states, learned only from qualified final-training rows."""
    ranked = cross_sectional_rank(frame, original)
    return {
        'raw': fit_state(frame.loc[mask, original], 'raw', RECIPE),
        'rank': fit_state(ranked.loc[mask], 'cross_sectional_rank', RECIPE),
        'context': fit_state(frame.loc[mask, added], 'raw', RECIPE),
    }


def model_matrix(frame, ranked, components, candidate):
    """Identical immutable transforms for fitting and future prediction."""
    kind = 'raw' if candidate['family'] == 'ridge' else 'rank'
    state = components[kind]
    source = frame[state['features']] if kind == 'raw' else ranked
    matrix = transform_features(source, state)
    if candidate['id'].endswith('enhanced'):
        state = components['context']
        matrix = pd.concat([matrix, transform_features(frame[state['features']], state)], axis=1)
    require(matrix.columns.tolist() == candidate['features'], 'Model feature order changed')
    return matrix


def artifact_metadata():
    """Protect old outputs by stat only, without reading sealed observations."""
    return {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
            for folder in ['outputs', 'results'] for p in (ROOT/folder).rglob('*') if p.is_file()}


def check_metadata(metadata):
    for name, expected in metadata.items():
        p = ROOT/name
        require(p.exists() and [p.stat().st_size, p.stat().st_mtime_ns] == expected,
                'Previous artifact changed: ' + name)


def main():
    require(not OUT.exists() and not PUBLIC.exists(), 'Final V2 output exists; no implicit refit or resume')
    plan, original, added = load_plan()
    protected = artifact_metadata()
    frame, mask, access = load_training(original)
    training = frame.loc[mask]
    require(len(training) > 0, 'Empty final training')
    restored = mask & ~frame.eligible_main_sample
    require(frame.loc[restored, 'split'].eq('train').all() and
            frame.loc[restored, 'purged_boundary'].all(), 'Unexpected restored boundary')
    boundary = dict(candidate_rows=len(frame), fit_rows=len(training),
                    signal_dates=training.signal_date.nunique(),
                    signal_start=str(training.signal_date.min().date()),
                    signal_end=str(training.signal_date.max().date()),
                    max_label_end=str(training.label_end_date.max().date()),
                    restored_internal_boundary_rows=int(restored.sum()),
                    excluded_cross_2022_rows=int(frame.label_end_date.gt(END).sum()),
                    original_train_rows=int((mask & frame.split.eq('train')).sum()),
                    original_validation_rows=int((mask & frame.split.eq('validation')).sum()))
    # Reserve the entire run before any preprocessing or estimator fit. Failures
    # leave a visible ledger; rerunning cannot silently spend the budget again.
    OUT.mkdir()
    (OUT/'models').mkdir()
    (OUT/'preprocessing').mkdir()
    sources = [PLAN, ROOT/'config/features_labels.json', ROOT/'config/context_features_v2.json']
    sources += [ROOT/'src/equity_research'/p for p in [
        'models/final_fit_v2.py', 'models/verification_v2.py', 'models/final_fit.py',
        'models/context_selection.py', 'models/ranking_selection.py',
        'models/preprocessing.py', 'models/common.py', 'evaluation/v2.py']]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    write_json(OUT/'run_started.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        authorization='Six frozen 2015-2021 final fits only; no historical replay or unseen access',
        plan_sha256=sha(PLAN), boundary=boundary, input_hashes=inputs,
        filtered_candidate_digest=frame_digest(frame), fit_digest=frame_digest(training),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, lightgbm=lgb.__version__)))
    write_json(OUT/'protected_artifact_metadata.json', protected)
    write_json(OUT/'observation_access_log.json', access)
    write_json(OUT/'frozen_plan.json', plan)
    training[['PERMNO', 'signal_date', 'label_end_date', 'label_status', 'split']].to_parquet(
        OUT/'fit_rows_local.parquet', index=False)
    training.groupby(training.signal_date.dt.year).agg(
        rows=('PERMNO', 'size'), signal_dates=('signal_date', 'nunique'),
        max_label_end=('label_end_date', 'max')).to_csv(OUT/'annual_fit_coverage.csv')
    write_json(OUT/'baseline_state.json', dict(momentum_direction='positive',
        momentum_missing=float(training.momentum_20.replace([np.inf, -np.inf], np.nan).median())))
    components = fit_components(frame, mask, original, added)
    for kind, state in components.items():
        write_json(OUT/'preprocessing'/f'{kind}.json', state)
    relevance, groups = ranking_labels(training)
    require((groups > 1).all(), 'Singleton ranking query')
    training[['PERMNO', 'signal_date']].assign(relevance=relevance).to_parquet(
        OUT/'ranking_labels_local.parquet', index=False)
    ranked = cross_sectional_rank(frame, original)
    ledger, checks = [], []
    reference = training[['PERMNO', 'signal_date']].copy()
    print('V2 final boundary:', boundary, flush=True)
    with threadpool_limits(limits=4):
        for candidate in plan['models']:
            require(len(ledger) < 6, 'Final fit budget exhausted')
            name, family = candidate['id'], candidate['family']
            X = model_matrix(frame, ranked, components, candidate).loc[mask]
            model = {'ridge': Ridge, 'lightgbm': lgb.LGBMRegressor,
                     'ranker': lgb.LGBMRanker}[family](**candidate['parameters'])
            rec = dict(id=name, status='fit_started', parameters=candidate['parameters'],
                       fit_rows=len(X), transformed_fit_digest=frame_digest(X),
                       features=candidate['features'],
                       queries=len(groups) if family=='ranker' else None,
                       query_min=int(groups.min()) if family=='ranker' else None,
                       query_max=int(groups.max()) if family=='ranker' else None)
            ledger.append(rec)
            write_json(OUT/'fit_ledger.json', ledger)
            if family == 'ranker':
                model.fit(X, relevance, group=groups)
            else:
                model.fit(X, training.y_excess_5d)
            path = OUT/'models'/(name + ('.joblib' if family == 'ridge' else '.txt'))
            if family == 'ridge':
                joblib.dump(model, path)
                loaded = joblib.load(path)
            else:
                save_lightgbm(model, path)
                loaded = load_lightgbm(path)
                require(loaded.num_trees() == 100, 'Unexpected tree count')
            p, q = model.predict(X), loaded.predict(X)
            require(np.isfinite(q).all(), 'Nonfinite serialization prediction')
            np.testing.assert_allclose(p, q, atol=1e-12, rtol=1e-12)
            reference[name] = q
            rec.update(status='completed', model_path=path.relative_to(ROOT).as_posix(),
                       model_sha256=sha(path), effective_parameters=model.get_params(),
                       trees=None if family == 'ridge' else loaded.num_trees())
            write_json(OUT/'fit_ledger.json', ledger)
            checks.append(dict(id=name, max_serialization_error=float(np.max(np.abs(p-q)))))
            print(f'Final fit {len(ledger)}/6 completed: {name}', flush=True)
    reference.to_parquet(OUT/'reload_reference_local.parquet', index=False)
    check_metadata(protected)
    for name, digest in inputs.items():
        require(sha(ROOT/name) == digest, 'Source changed during fitting: ' + name)
    write_json(OUT/'fit_summary.json', dict(status='six_fits_completed_pending_independent_verification',
        fits=len(ledger), preprocessing_fits=3, boundary=boundary, roundtrip_checks=checks,
        historical_replay_executed=False, unseen_data_read=False, performance_evaluated=False))
    from equity_research.models.verification_v2 import main as verify
    verify()
