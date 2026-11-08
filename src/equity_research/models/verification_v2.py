"""Independent final-fit checks: NumPy transforms, SciPy ranks, no estimator fit."""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from equity_research.data.ingest import write_json
from equity_research.models.common import sha, load_lightgbm
from equity_research.models.preprocessing import require, frame_digest
from equity_research.models.final_fit_v2 import (
    ROOT, OUT, PUBLIC, PLAN, START, END, read_json, load_plan, load_training, check_metadata,
)


def independent_components(frame, mask, original, added, states):
    """Recompute fitted statistics and transformed matrices without fit helpers."""
    ranked = pd.DataFrame(np.nan, index=frame.index, columns=original)
    for _, group in frame.groupby('signal_date'):
        for name in original:
            ix = group.index[group.signal_eligible & np.isfinite(group[name])]
            if len(ix):
                ranked.loc[ix, name] = (rankdata(frame.loc[ix, name], method='average')-.5)/len(ix)-.5
    matrices = {}
    for kind, values in [('raw', frame[original]), ('rank', ranked), ('context', frame[added])]:
        a = values.loc[mask].to_numpy(float)
        a[~np.isfinite(a)] = np.nan
        median = np.nanmedian(a, axis=0)
        lo, hi = (np.full(a.shape[1], -.5), np.full(a.shape[1], .5)) if kind == 'rank' else np.nanquantile(a, [.005, .995], axis=0)
        clean = np.where(np.isnan(a), median, np.clip(a, lo, hi))
        mean, scale = clean.mean(axis=0), clean.std(axis=0)
        scale[scale == 0] = 1
        state = states[kind]
        require(state['features'] == values.columns.tolist() and state['fitted_rows'] == int(mask.sum()), 'State row/order mismatch')
        for key, expected in [('lower', lo), ('upper', hi), ('median', median), ('mean', mean), ('scale', scale)]:
            np.testing.assert_allclose(state[key], expected, atol=1e-12, rtol=1e-12)
        matrices[kind] = pd.DataFrame((clean-state['mean'])/state['scale'], columns=values.columns)
    return matrices


def verify_contexts(frame):
    """Independent per-date/per-industry arithmetic, no labels in the population."""
    for _, g in frame.groupby('signal_date'):
        good = g.signal_eligible
        m20 = g.momentum_20.where(good).replace([np.inf, -np.inf], np.nan)
        m60 = g.momentum_60.where(good).replace([np.inf, -np.inf], np.nan)
        vol = g.volatility_20.where(good).replace([np.inf, -np.inf], np.nan)
        mean, breadth, dispersion, medvol = m20.mean(), (m20.dropna()>0).mean(), m20.std(), vol.median()
        expected = {'universe_mean_mom20': mean, 'universe_breadth20': breadth,
                    'universe_dispersion20': dispersion, 'universe_median_vol20': medvol,
                    'momentum_vol_interaction': g.momentum_20*medvol,
                    'reversal_breadth_interaction': g.reversal_1*(breadth-.5),
                    'momentum_dispersion_interaction': g.momentum_60*dispersion}
        for key, value in expected.items():
            np.testing.assert_allclose(g[key], value, rtol=1e-11, atol=1e-12, equal_nan=True)
        for industry, part in g.groupby('industry'):
            for window, values in [(20, m20), (60, m60)]:
                own = values.loc[part.index].dropna()
                source = own if len(own)>=5 and industry!='Unknown' else values.dropna()
                avg = source.mean()
                np.testing.assert_allclose(part[f'industry_mom{window}'], avg, rtol=1e-11, atol=1e-12, equal_nan=True)
                np.testing.assert_allclose(part[f'relative_mom{window}'], values.loc[part.index]-avg, rtol=1e-11, atol=1e-12, equal_nan=True)
                if window == 20:
                    ranks = pd.Series((rankdata(source)-.5)/len(source)-.5, index=source.index)
                    np.testing.assert_allclose(part.industry_rank20, ranks.reindex(part.index), rtol=1e-11, atol=1e-12, equal_nan=True)


def main():
    plan, original, added = load_plan()
    started = read_json(OUT/'run_started.json')
    for path, digest in started['input_hashes'].items():
        require(sha(ROOT/path) == digest, 'Final-fit source drift: ' + path)
    check_metadata(read_json(OUT/'protected_artifact_metadata.json'))
    frame, _, access = load_training(original)
    # Independently recompute eligibility, including finite target and both bounds.
    mask = (frame.signal_date.between(START, END) & frame.signal_eligible.eq(True) &
            frame.label_status.isin(['ok', 'ok_bridged']) & frame.label_end_date.gt(frame.signal_date) &
            frame.label_end_date.le(END) & np.isfinite(frame.y_excess_5d))
    training = frame.loc[mask]
    require(frame_digest(frame) == started['filtered_candidate_digest'] and
            frame_digest(training) == started['fit_digest'], 'Final training digest changed')
    saved = pd.read_parquet(OUT/'fit_rows_local.parquet')
    pd.testing.assert_frame_equal(saved, training[saved.columns].reset_index(drop=True))
    verify_contexts(frame)
    states = {k: read_json(OUT/'preprocessing'/f'{k}.json') for k in ['raw', 'rank', 'context']}
    matrices = independent_components(frame, mask, original, added, states)
    labels = pd.read_parquet(OUT/'ranking_labels_local.parquet')
    pd.testing.assert_frame_equal(labels[['PERMNO', 'signal_date']], training[['PERMNO', 'signal_date']].reset_index(drop=True))
    for _, g in training.groupby('signal_date'):
        expected = np.minimum(9, np.floor(10*(rankdata(g.y_excess_5d)-.5)/len(g))).astype(int)
        actual = labels.loc[labels.signal_date.eq(g.signal_date.iloc[0]), 'relevance']
        np.testing.assert_array_equal(actual, expected)
    ledger = read_json(OUT/'fit_ledger.json')
    require(len(ledger) == 6 and [r['id'] for r in ledger] == [c['id'] for c in plan['models']], 'Fit budget/identity mismatch')
    reference = pd.read_parquet(OUT/'reload_reference_local.parquet')
    pd.testing.assert_frame_equal(reference[['PERMNO', 'signal_date']], saved[['PERMNO', 'signal_date']])
    checks, bundle_models = [], []
    with threadpool_limits(limits=4):
        for candidate, rec in zip(plan['models'], ledger):
            require(rec['status'] == 'completed' and rec['parameters'] == candidate['parameters'] and rec['fit_rows'] == len(training), 'Fit ledger mismatch')
            for key, value in candidate['parameters'].items():
                require(rec['effective_parameters'][key] == value, 'Effective parameter mismatch')
            require(sha(ROOT/rec['model_path']) == rec['model_sha256'], 'Model hash changed')
            model = joblib.load(ROOT/rec['model_path']) if candidate['family']=='ridge' else load_lightgbm(ROOT/rec['model_path'])
            kind = 'raw' if candidate['family']=='ridge' else 'rank'
            X = matrices[kind]
            components = [kind]
            if candidate['id'].endswith('enhanced'):
                X = pd.concat([X, matrices['context']], axis=1)
                components.append('context')
            names = model.feature_names_in_.tolist() if candidate['family']=='ridge' else model.feature_name()
            require(names == candidate['features'] == X.columns.tolist(), 'Serialized feature order changed')
            if candidate['family'] == 'ridge':
                for key, value in candidate['parameters'].items():
                    require(model.get_params()[key] == value, 'Serialized Ridge parameter changed')
            if candidate['family'] != 'ridge':
                require(model.num_trees() == 100, 'Serialized tree count changed')
                expected_objective = 'lambdarank' if candidate['family']=='ranker' else 'regression'
                require(expected_objective in model.dump_model()['objective'], 'Serialized objective changed')
                for key, value in candidate['parameters'].items():
                    if key in model.params:
                        require(model.params[key] == value, 'Serialized LightGBM parameter changed: ' + key)
            pred = model.predict(X)
            expected = reference[candidate['id']].to_numpy()
            np.testing.assert_allclose(pred, expected, atol=1e-10, rtol=1e-10)
            checks.append(dict(id=candidate['id'], rows=len(training), independent_prediction_max_error=float(np.max(np.abs(pred-expected)))))
            bundle_models.append(dict(**candidate, path=rec['model_path'], sha256=rec['model_sha256'], components=components))
    baseline = read_json(OUT/'baseline_state.json')
    require(baseline['momentum_missing'] == float(training.momentum_20.median()), 'Baseline median changed')
    write_json(OUT/'verification_access_log.json', access)
    result = dict(status='passed', fits=6, candidate_fits_added=0, verification_fits=0,
                  boundary=started['boundary'], independent_context_verified=True,
                  independent_preprocessing_verified=True, independent_relevance_verified=True,
                  checks=checks, historical_replay_executed=False, unseen_data_read=False,
                  performance_evaluated=False, prior_artifacts_unchanged=True)
    write_json(OUT/'verification.json', result)
    write_json(OUT/'model_bundle.json', dict(version='2.0', plan_sha256=sha(PLAN), models=bundle_models,
        preprocessing={k: dict(path=(OUT/'preprocessing'/f'{k}.json').relative_to(ROOT).as_posix(),
                               sha256=sha(OUT/'preprocessing'/f'{k}.json')) for k in states},
        baseline=dict(path=(OUT/'baseline_state.json').relative_to(ROOT).as_posix(), sha256=sha(OUT/'baseline_state.json')),
        original_features=original, context_features=added, fit_dates=plan['final_fit']['dates']))
    # Public manifest includes only model/config/state hashes, never licensed rows.
    paths = list((OUT/'models').glob('*')) + list((OUT/'preprocessing').glob('*.json'))
    paths += [OUT/name for name in ['model_bundle.json', 'baseline_state.json', 'verification.json', 'fit_ledger.json']]
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    write_json(OUT/'manifest.json', manifest)
    PUBLIC.mkdir(exist_ok=True)
    write_json(PUBLIC/'verification.json', result)
    write_json(PUBLIC/'manifest.json', manifest)
    (PUBLIC/'annual_fit_coverage.csv').write_bytes((OUT/'annual_fit_coverage.csv').read_bytes())
    print('Six V2 models independently verified; no evaluation observations read.', flush=True)
