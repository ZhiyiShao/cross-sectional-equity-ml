"""Frozen V2 development runner. Default entry is a public-config-only preflight."""
import importlib.metadata
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.models.robustness_ledger import FitLedger, atomic_json, sha
from equity_research.models.preprocessing import cross_sectional_rank, transform_features, require

ROOT = Path(__file__).resolve().parents[3]
PLAN = 'config/parameter_robustness_v2.json'
PLAN_SHA = '80cb1d87d2f540ea814f612548200b652b883922a0c3c52c7a968718f80b330e'
KEYS = ['PERMNO', 'signal_date']
META = KEYS+['signal_eligible', 'label_end_date', 'label_status']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_plan(root=ROOT):
    require(sha(root/PLAN) == PLAN_SHA, 'Frozen robustness plan changed')
    plan = read_json(root/PLAN)
    for name, digest in plan['source_sha256'].items():
        require(sha(root/name) == digest, 'Frozen source changed: '+name)
    return plan


def tasks(plan):
    result = []
    for wi, window in enumerate(plan['development_windows'], 1):
        for design in plan['designs']:
            for candidate in plan[design['grid']]['candidates']:
                result.append(dict(key=f"w{wi}_{design['id']}_{candidate['id']}", window=wi,
                                   design=design['id'], grid=design['grid'], candidate=candidate,
                                   kind='reuse' if candidate['reuse_candidate'] is not None else 'fit'))
    require(len({t['key'] for t in result}) == len(result), 'Duplicate schedule')
    require(sum(t['kind'] == 'fit' for t in result) == plan['budget']['max_new_fit_calls'], 'New fit budget mismatch')
    require(sum(t['kind'] == 'reuse' for t in result) == plan['budget']['reused_candidate_window_results'], 'Reuse budget mismatch')
    require(len(result) == plan['budget']['total_candidate_window_results'], 'Total budget mismatch')
    return result


def preflight(root=ROOT):
    plan = load_plan(root); schedule = tasks(plan)
    return dict(status='preflight_passed', plan_sha256=sha(root/PLAN), tasks=len(schedule),
                new_fits_planned=plan['budget']['max_new_fit_calls'], old_candidates_to_verify=48,
                formal_fits_executed=0, private_artifacts_read=False,
                execution='Requires explicit --execute; this command only reads frozen public configuration')


def parameters(plan, task):
    candidate = task['candidate']
    if task['grid'] == 'ridge':
        return {**plan['ridge']['fixed'], 'alpha': candidate['alpha']}
    objective = plan['lightgbm']['ranking_fixed'] if task['design'].startswith('ranker') else {'objective': 'regression'}
    return {**plan['lightgbm']['fixed'], **objective,
            **{k: candidate[k] for k in ['num_leaves', 'n_estimators', 'min_child_samples']}}


def read_window(path, window, features, reader=pd.read_parquet):
    """Read metadata/features first; Arrow filters exclude crossing labels before IO."""
    start, end = pd.Timestamp(window['train'][0]), pd.Timestamp(window['evaluate'][1])
    require(end < pd.Timestamp('2022-01-01'), 'Development date boundary')
    require(pd.Timestamp(window['train'][1]) < pd.Timestamp(window['evaluate'][0]), 'Overlapping windows')
    filters = [('signal_date', '>=', start), ('signal_date', '<=', end)]
    frame = reader(path, columns=META+features, filters=filters).sort_values(KEYS[::-1]).reset_index(drop=True)
    require(len(frame) > 0 and frame.signal_date.between(start, end).all(), 'Feature date boundary')
    require(not frame.duplicated(KEYS).any(), 'Duplicate feature keys')
    audit = [dict(columns=META+features, period=[str(start), str(end)], targets=False)]
    masks = {}
    frame['y_excess_5d'] = np.nan
    for role in ['train', 'evaluate']:
        lo, hi = map(pd.Timestamp, window[role])
        target_filters = [('signal_date', '>=', lo), ('signal_date', '<=', hi),
                          ('label_end_date', '<=', hi), ('label_end_date', '>', lo),
                          ('signal_eligible', '==', True), ('label_status', 'in', ['ok', 'ok_bridged'])]
        target = reader(path, columns=META+['y_excess_5d'], filters=target_filters)
        require(not target.duplicated(KEYS).any(), 'Duplicate target keys')
        require(target.signal_date.between(lo, hi).all() and target.label_end_date.le(hi).all(), 'Target date boundary')
        require(target.signal_eligible.all() and target.label_status.isin(['ok', 'ok_bridged']).all(), 'Target eligibility breach')
        aligned = frame[META].merge(target, on=KEYS, how='left', validate='one_to_one', suffixes=('', '_target'))
        require(target.set_index(KEYS).index.isin(frame.set_index(KEYS).index).all(), 'Unexpected target keys')
        present = aligned.label_end_date_target.notna()
        for col in ['signal_eligible', 'label_end_date', 'label_status']:
            require(aligned.loc[present, col].eq(aligned.loc[present, col+'_target']).all(), 'Target metadata mismatch')
        expected = (frame.signal_date.between(lo, hi) & frame.signal_eligible & frame.label_status.isin(['ok', 'ok_bridged'])
                    & frame.label_end_date.gt(frame.signal_date) & frame.label_end_date.le(hi))
        require((~expected | present).all(), 'Missing eligible target rows')
        valid = expected & np.isfinite(aligned.y_excess_5d)
        frame.loc[valid, 'y_excess_5d'] = aligned.loc[valid, 'y_excess_5d']
        masks[role] = valid
        audit.append(dict(columns=META+['y_excess_5d'], period=[str(lo), str(hi)], targets=True,
                          rows=len(target), max_label_end=str(target.label_end_date.max())))
    require(not (masks['train'] & masks['evaluate']).any(), 'Train/evaluation overlap')
    return frame, masks, audit


def check_state(state, frame, train, original, added, design, window):
    """Independently check stored statistics, without replacing the saved state."""
    raw = design.startswith('ridge')
    base = frame[original] if raw else cross_sectional_rank(frame, original)
    expected_columns = original+(added if design.endswith('enhanced') else [])
    require(state['columns'] == expected_columns, 'Feature order mismatch')
    require(state['train_start'] == window['train'][0] and state['train_end'] == window['train'][1], 'State training window mismatch')
    require(pd.Timestamp(state['max_label_end']) == frame.loc[train, 'label_end_date'].max(), 'State label boundary mismatch')
    parts = [(base, state['base'], raw)]
    if design.endswith('enhanced'):
        parts.append((frame[added], state['context'], True))
    elif 'context' in state:
        raise ValueError('Unexpected context state')
    matrices = []
    for values, part, clip in parts:
        require(part['features'] == values.columns.tolist() and part['fitted_rows'] == int(train.sum()), 'State training population mismatch')
        a = values.loc[train].to_numpy(float).copy(); a[~np.isfinite(a)] = np.nan
        median = np.nanmedian(a, axis=0)
        lo, hi = np.nanquantile(a, [.005, .995], axis=0) if clip else (np.full(a.shape[1], -.5), np.full(a.shape[1], .5))
        clean = np.where(np.isnan(a), median, np.clip(a, lo, hi))
        scale = clean.std(axis=0); scale[scale == 0] = 1
        for name, expected in [('lower', lo), ('upper', hi), ('median', median), ('mean', clean.mean(axis=0)), ('scale', scale)]:
            np.testing.assert_allclose(part[name], expected, atol=1e-12, rtol=1e-12)
        matrices.append(transform_features(values, part))
    return pd.concat(matrices, axis=1)


def daily_ic(frame, prediction, mask):
    # scipy provides an independent check against the historical vectorized IC.
    from scipy.stats import spearmanr
    selected = frame.loc[mask, ['signal_date', 'y_excess_5d']].assign(prediction=np.asarray(prediction)[mask.to_numpy()])
    result = selected.groupby('signal_date').apply(
        lambda g: spearmanr(g.prediction, g.y_excess_5d).statistic, include_groups=False)
    require(len(result) > 0 and np.isfinite(result).all(), 'Undefined daily IC')
    return result.rename('rank_ic')


def ranking_target(frame):
    require(frame.signal_date.is_monotonic_increasing, 'Noncontiguous ranking queries')
    groups = frame.groupby('signal_date', sort=False).y_excess_5d
    require(np.isfinite(frame.y_excess_5d).all(), 'Invalid ranking labels')
    levels = np.minimum(9, np.floor(10*(groups.rank(method='average')-.5)/groups.transform('size'))).astype('int32')
    sizes = groups.size().to_numpy('int32')
    require((sizes > 1).all(), 'Singleton ranking query')
    return levels, sizes


def new_estimator(plan, task):
    if task['grid'] == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**parameters(plan, task))
    import lightgbm as lgb
    cls = lgb.LGBMRanker if task['design'].startswith('ranker') else lgb.LGBMRegressor
    return cls(**parameters(plan, task))


def fit_candidate(model, task, matrix, frame, mask, cutoff):
    from threadpoolctl import threadpool_limits
    train = frame.loc[mask]
    require(len(train) > 0 and train.label_end_date.le(pd.Timestamp(cutoff)).all()
            and train.signal_date.lt(train.label_end_date).all(), 'Fit label boundary breach')
    require(train.signal_eligible.all() and train.label_status.isin(['ok', 'ok_bridged']).all()
            and np.isfinite(train.y_excess_5d).all(), 'Unqualified training row')
    require(matrix.index.equals(frame.index), 'Training matrix misalignment')
    with threadpool_limits(limits=4):
        if task['design'].startswith('ranker'):
            labels, sizes = ranking_target(train)
            model.fit(matrix.loc[mask], labels, group=sizes)
        else:
            model.fit(matrix.loc[mask], train.y_excess_5d)


def verify_reuse_predictions(task, frame, matrix, masks, model, stored_prediction, stored_daily, metric, ledger, window):
    """Check all candidate rows, labels, historical fit ledger, predictions and IC."""
    ec = frame.signal_date.between(*map(pd.Timestamp, window['evaluate']))
    expected = frame.loc[ec, KEYS+['signal_eligible']].reset_index(drop=True)
    pd.testing.assert_frame_equal(stored_prediction[expected.columns].reset_index(drop=True), expected)
    train = frame.loc[masks['train']]
    require(ledger['train_rows'] == len(train), 'Old fit training rows mismatch')
    require(pd.Timestamp(ledger['max_train_signal']) == train.signal_date.max()
            and pd.Timestamp(ledger['max_train_label']) == train.label_end_date.max(), 'Old fit label boundary mismatch')
    require(ledger['eval_start'] == window['evaluate'][0], 'Old fit evaluation window mismatch')
    pred = np.asarray(model.predict(matrix.loc[ec]), float)
    require(np.isfinite(pred).all(), 'Nonfinite reused predictions')
    np.testing.assert_allclose(pred, stored_prediction.prediction, atol=1e-12, rtol=1e-12)
    full = np.full(len(frame), np.nan); full[ec.to_numpy()] = pred
    ic = daily_ic(frame, full, masks['evaluate'])
    expected_daily = stored_daily.set_index('signal_date').rank_ic.sort_index()
    pd.testing.assert_index_equal(ic.index, expected_daily.index)
    np.testing.assert_allclose(ic, expected_daily, atol=1e-12, rtol=1e-12)
    require(metric['dates'] == len(ic) and metric['rows'] == int(masks['evaluate'].sum()), 'Old evaluation coverage mismatch')
    np.testing.assert_allclose(ic.mean(), metric['rank_ic'], atol=1e-12, rtol=1e-12)
    return full, ic


def check_model_parameters(model, expected, ridge):
    if ridge:
        actual = model.get_params()
    else:
        actual = dict(model.params)
        aliases = {'n_estimators': 'num_iterations', 'min_child_samples': 'min_data_in_leaf',
                   'reg_lambda': 'lambda_l2', 'subsample': 'bagging_fraction',
                   'subsample_freq': 'bagging_freq', 'colsample_bytree': 'feature_fraction',
                   'random_state': 'seed', 'n_jobs': 'num_threads'}
        actual.update({k: actual[v] for k, v in aliases.items() if v in actual})
    for name, value in expected.items():
        require(name in actual, 'Missing saved estimator parameter: '+name)
        observed = actual[name]
        if isinstance(value, list):
            if isinstance(observed, str): observed = [float(v) for v in observed.split(',')]
            require(list(observed) == value, 'Saved estimator parameter mismatch: '+name)
        elif isinstance(value, str):
            require(observed == value, 'Saved estimator parameter mismatch: '+name)
        else:
            require(float(observed) == float(value), 'Saved estimator parameter mismatch: '+name)


def legacy_files(root, plan, schedule):
    """An explicit allowlist excludes all historical replay/holdout artifacts."""
    context = root/plan['reuse']['context_artifacts']; ranking = root/plan['reuse']['ranking_artifacts']
    files = {root/PLAN, root/'config/features_labels.json', root/plan['reuse']['development_snapshot']}
    for directory in [context, ranking]:
        files.update(directory/name for name in ['verification.json', 'frozen_plan.json', 'fit_ledger.json', 'candidate_daily_ic.csv'])
        files.update(directory/f'w{wi}_predictions_local.parquet' for wi in [1, 2])
    for task in schedule:
        if task['kind'] != 'reuse': continue
        directory = ranking if task['design'].startswith('ranker') else context
        name = f"w{task['window']}_{task['design']}"
        files.add(directory/'preprocessing'/f'{name}.json')
        suffix = '.joblib' if task['grid'] == 'ridge' else '.txt'
        files.add(directory/'models'/f"{name}_{task['candidate']['reuse_candidate']}{suffix}")
    files.update(root/name for name in plan['source_sha256'])
    files.update(root/name for name in ['src/equity_research/models/parameter_robustness.py',
                  'src/equity_research/models/robustness_ledger.py',
                  'src/equity_research/models/preprocessing.py', 'scripts/train_models.py'])
    return sorted(files)


def prepare_legacy(root, plan, schedule):
    """Validate all 48 old candidates before permitting a single new fit."""
    import joblib
    import lightgbm as lgb
    from sklearn.exceptions import InconsistentVersionWarning
    context = root/plan['reuse']['context_artifacts']; ranking = root/plan['reuse']['ranking_artifacts']
    prior = read_json(context/'verification.json'); rank_prior = read_json(ranking/'verification.json')
    require(prior['fits'] == 32 and rank_prior['fits'] == 16, 'Old fit budget mismatch')
    require(rank_prior['lightgbm_version'] == lgb.__version__, 'LightGBM version mismatch')
    context_plan = read_json(context/'frozen_plan.json')
    ranking_plan = read_json(ranking/'frozen_plan.json')
    require(context_plan['config'] == read_json(root/'config/context_features_v2.json'), 'Old context configuration mismatch')
    require(ranking_plan['config'] == read_json(root/'config/ranking_v2.json'), 'Old ranking configuration mismatch')
    require(context_plan['parent'] == ranking_plan['parent'] == read_json(root/'config/research_v2.json'), 'Old window configuration mismatch')
    snapshot = root/plan['reuse']['development_snapshot']
    require(sha(snapshot) == rank_prior['source_hashes'][snapshot.relative_to(root).as_posix()], 'Development snapshot changed')
    original = read_json(root/'config/features_labels.json')['features']
    added = list(read_json(root/'config/context_features_v2.json')['context_features'])
    windows = {}; matrices = {}; reused = {}
    for wi, window in enumerate(plan['development_windows'], 1):
        print(f'Checking development window {wi}: bounded targets and saved preprocessing', flush=True)
        frame, masks, access = read_window(snapshot, window, original+added)
        coverage = plan['expected_evaluation_coverage'][wi-1]
        require(int(masks['evaluate'].sum()) == coverage['strict_rows'] and
                frame.loc[masks['evaluate'], 'signal_date'].nunique() == coverage['dates'], 'Frozen evaluation coverage changed')
        windows[wi] = (frame, masks, access)
        for design in plan['designs']:
            name = design['id']; directory = ranking if name.startswith('ranker') else context
            statepath = directory/'preprocessing'/f'w{wi}_{name}.json'
            state = read_json(statepath)
            if name.startswith('ranker'):
                original_state = context/'preprocessing'/f"w{wi}_{name.replace('ranker', 'lightgbm')}.json"
                require(sha(statepath) == sha(original_state), 'Ranking preprocessing differs')
                require(sha(original_state) == rank_prior['source_hashes'][original_state.relative_to(root).as_posix()], 'Historical preprocessing hash mismatch')
            matrices[(wi, name)] = check_state(state, frame, masks['train'], original, added, name, window)
    for directory, families in [(context, ['ridge', 'lightgbm']), (ranking, ['ranker'])]:
        old_ledger = read_json(directory/'fit_ledger.json')
        daily = pd.read_csv(directory/'candidate_daily_ic.csv', parse_dates=['signal_date'])
        require(daily.signal_date.between('2019-01-01', '2021-12-31').all(), 'Old IC outside development')
        require(not daily.duplicated(['window', 'design', 'candidate', 'signal_date']).any(), 'Duplicate old daily IC')
        metrics = pd.read_csv(root/'results'/directory.name/'candidate_metrics.csv')
        expected_keys = {(t['window'], t['design'], t['candidate']['reuse_candidate']) for t in schedule
                         if t['kind']=='reuse' and t['design'].split('_')[0] in families}
        ledger_keys = [(v['window'], v['design'], v['candidate']) for v in old_ledger]
        require(len(ledger_keys) == len(expected_keys) and set(ledger_keys) == expected_keys, 'Old ledger candidate coverage mismatch')
        require(len(metrics) == len(expected_keys) and set(metrics[['window','design','candidate']].itertuples(index=False,name=None)) == expected_keys,
                'Old metrics candidate coverage mismatch')
        require(set(daily[['window','design','candidate']].itertuples(index=False,name=None)) == expected_keys, 'Old daily candidate coverage mismatch')
        for wi, window in enumerate(plan['development_windows'], 1):
            frame, masks, _ = windows[wi]
            relevant = [t for t in schedule if t['kind'] == 'reuse' and t['window'] == wi and t['design'].split('_')[0] in families]
            cols = KEYS+['signal_eligible']+[f"{t['design']}_{t['candidate']['reuse_candidate']}" for t in relevant]
            saved = pd.read_parquet(directory/f'w{wi}_predictions_local.parquet', columns=cols,
                                    filters=[('signal_date', '>=', pd.Timestamp(window['evaluate'][0])),
                                             ('signal_date', '<=', pd.Timestamp(window['evaluate'][1]))])
            for task in relevant:
                design = task['design']; ci = task['candidate']['reuse_candidate']; column = f'{design}_{ci}'
                stem = directory/'models'/f'w{wi}_{column}'
                if task['grid'] == 'ridge':
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', InconsistentVersionWarning)
                        model = joblib.load(stem.with_suffix('.joblib'))
                    require(list(model.feature_names_in_) == matrices[(wi, design)].columns.tolist(), 'Ridge model feature order mismatch')
                else:
                    model = lgb.Booster(model_str=stem.with_suffix('.txt').read_text(encoding='utf-8'))
                    require(model.feature_name() == matrices[(wi, design)].columns.tolist(), 'Tree model feature order mismatch')
                check_model_parameters(model, parameters(plan, task), task['grid'] == 'ridge')
                old = [v for v in old_ledger if v['window'] == wi and v['design'] == design and v['candidate'] == ci]
                metric = metrics[metrics.window.eq(wi)&metrics.design.eq(design)&metrics.candidate.eq(ci)]
                require(len(old) == len(metric) == 1, 'Missing or duplicate legacy candidate')
                expected_parameter = task['candidate']['alpha'] if task['grid'] == 'ridge' else task['candidate']['num_leaves']
                require(float(metric.iloc[0].parameter) == expected_parameter, 'Old metric parameter mismatch')
                d = daily[daily.window.eq(wi)&daily.design.eq(design)&daily.candidate.eq(ci)]
                reused[task['key']] = verify_reuse_predictions(task, frame, matrices[(wi, design)], masks, model,
                    saved[KEYS+['signal_eligible', column]].rename(columns={column: 'prediction'}), d,
                    metric.iloc[0].to_dict(), old[0], window)
                if len(reused) % 8 == 0:
                    print(f'Verified old candidates: {len(reused)}/48', flush=True)
    require(len(reused) == plan['budget']['reused_candidate_window_results'], 'Incomplete legacy verification')
    return windows, matrices, reused


def run_tasks(schedule, ledger, operation):
    # All reuse validation happens before this function, not interleaved with fits.
    for task in sorted(schedule, key=lambda t: (t['kind'] != 'reuse', t['key'])):
        ledger.perform(task, operation)


def execute(root=ROOT, *, resume=False):
    plan = load_plan(root); schedule = tasks(plan)
    out = root/plan['outputs']['local']
    require(resume or not out.exists(), 'Existing experiment; no implicit rerun')
    require(not resume or (out/'fit_ledger.json').exists(), 'No ledger to resume')
    files = legacy_files(root, plan, schedule)
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in files}
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'pyarrow', 'joblib']}
    manifest = dict(plan_sha256=sha(root/PLAN), inputs_and_code=hashes, versions=versions)
    if resume:
        require(read_json(out/'fit_ledger.json')['manifest'] == manifest, 'Resume input or environment drift')
    windows, matrices, reused = prepare_legacy(root, plan, schedule)
    print('All 48 old candidates verified; frozen new-fit schedule may proceed', flush=True)
    require(all(sha(root/name) == digest for name, digest in hashes.items()), 'Inputs changed during preflight')
    with FitLedger(out, manifest, schedule, plan['budget']['max_new_fit_calls'], resume=resume) as ledger:
        atomic_json(out/'access_audit.json', {str(w): data[2] for w, data in windows.items()})
        def operation(task):
            print(f"Starting {task['kind']}: {task['key']} (reserved fits {ledger.used}/80)", flush=True)
            frame, masks, _ = windows[task['window']]; matrix = matrices[(task['window'], task['design'])]
            window = plan['development_windows'][task['window']-1]
            folder = out/task['key']; folder.mkdir(exist_ok=False)
            artifact_names = []
            if task['kind'] == 'reuse':
                pred, ic = reused[task['key']]
            else:
                model = new_estimator(plan, task)
                fit_candidate(model, task, matrix, frame, masks['train'], window['train'][1])
                ec = frame.signal_date.between(*window['evaluate'])
                pred = np.full(len(frame), np.nan); pred[ec] = model.predict(matrix.loc[ec])
                require(np.isfinite(pred[ec]).all(), 'Nonfinite new prediction')
                ic = daily_ic(frame, pred, masks['evaluate'])
                if task['grid'] == 'ridge':
                    import joblib
                    path = folder/'model.joblib'; joblib.dump(model, path)
                else:
                    path = folder/'model.txt'; path.write_text(model.booster_.model_to_string(), encoding='utf-8')
                artifact_names.append(path.relative_to(out).as_posix())
            ec = frame.signal_date.between(*window['evaluate'])
            frame.loc[ec, KEYS+['signal_eligible']].assign(prediction=pred[ec]).to_parquet(folder/'predictions.parquet', index=False)
            ic.rename_axis('signal_date').to_csv(folder/'daily_ic.csv', lineterminator='\n')
            atomic_json(folder/'summary.json', dict(window=task['window'], design=task['design'],
                        candidate=task['candidate']['id'], kind=task['kind'], rank_ic=float(ic.mean()),
                        dates=len(ic), rows=int(masks['evaluate'].sum()), parameters=parameters(plan, task)))
            return artifact_names+[f"{task['key']}/{name}" for name in ['predictions.parquet', 'daily_ic.csv', 'summary.json']]
        run_tasks(schedule, ledger, operation)
        ledger.verify()
        require(ledger.used == 80 and len(ledger.state['entries']) == 128, 'Incomplete frozen budget')
        require(all(sha(root/name) == digest for name, digest in hashes.items()), 'Protected input changed')
        atomic_json(out/'execution_verification.json', dict(status='candidate_execution_complete', new_fits=ledger.used,
                    reused=48, final_refits=0, backtests=0, holdout_reads=0,
                    next_step='Frozen aggregate diagnostics; no automatic final refit or model promotion'))
    return dict(status='candidate_execution_complete', new_fits=80, reused=48)


def main(*, execute_requested=False, resume=False):
    require(not resume or execute_requested, '--resume requires --execute')
    result = execute(resume=resume) if execute_requested else preflight()
    print(json.dumps(result, indent=2))
