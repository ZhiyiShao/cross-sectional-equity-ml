"""V3 stage 0a: the six V2 selected designs on four yearly walk-forward windows.

The default entry is a public-config-only preflight. W2 and W3 share their
training sets with the two V2 windows, so their verified development
predictions are reused; only W1 and W4 are fitted. Nothing here selects a model.
"""
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.models import parameter_robustness as v2
from equity_research.models.preprocessing import cross_sectional_rank, fit_preprocessing, fit_state, require
from equity_research.models.robustness_ledger import FitLedger, atomic_json, sha

ROOT = v2.ROOT
PLAN = 'config/baseline_v3.json'
PLAN_SHA = '5ffec87837ab2bcddbe35bf54d4afdfb61bad8310f314807c1ec9a34ab1741de'
KEYS = v2.KEYS
CODE = ['src/equity_research/models/baseline_v3.py', 'src/equity_research/models/parameter_robustness.py',
        'src/equity_research/models/robustness_ledger.py', 'src/equity_research/models/preprocessing.py',
        'scripts/baseline_v3.py']
SOURCE_FILES = ['predictions.parquet', 'daily_ic.csv', 'summary.json']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def period(window, role):
    return tuple(pd.Timestamp(value) for value in window[role])


def load_plan(root=ROOT):
    """Read frozen public configuration only; no private artifact is opened."""
    require(sha(root/PLAN) == PLAN_SHA, 'Frozen V3 baseline plan changed')
    plan = read_json(root/PLAN)
    for name, digest in plan['source_sha256'].items():
        require(sha(root/name) == digest, 'Frozen source changed: '+name)
    parent = read_json(root/plan['parent']); old = read_json(root/plan['v2_plan'])
    require(parent['status'] == 'frozen', 'V3 protocol is not frozen')
    windows = {window['id']: window for window in parent['development_windows']}
    require(len(windows) == len(parent['development_windows']), 'Duplicate window id')
    require(set(plan['fit_windows']).isdisjoint(plan['reuse'])
            and set(plan['fit_windows']) | set(plan['reuse']) == set(windows), 'Window roles do not cover the protocol')
    development_end = pd.Timestamp(parent['partitions']['development'][1])
    require(development_end < pd.Timestamp(plan['isolation']['data_end_exclusive']), 'Development boundary mismatch')
    for window in windows.values():
        require(period(window, 'train')[1] < period(window, 'evaluate')[0]
                and period(window, 'evaluate')[1] <= development_end, 'Window outside development period')
    for wid, spec in plan['reuse'].items():
        source = old['development_windows'][spec['v2_window']-1]
        require(source['train'] == windows[wid]['train'], 'Reused window has a different training interval: '+wid)
        require(source['evaluate'][0] == windows[wid]['evaluate'][0]
                and period(windows[wid], 'evaluate')[1] <= period(source, 'evaluate')[1],
                'Reused window evaluation is not contained in its source: '+wid)
    with (root/plan['selection_source']).open(encoding='utf-8', newline='') as stream:
        selection = {row['design']: row['candidate'] for row in csv.DictReader(stream)}
    require(selection == plan['selected'], 'Baseline differs from the recorded V2 development selection')
    require([design['id'] for design in old['designs']] == list(plan['selected']), 'Design list mismatch')
    stage = next(item for item in parent['stages'] if item['id'] == plan['stage'])
    require(plan['budget']['max_new_fit_calls'] <= stage['max_fits'] == plan['budget']['protocol_stage_cap'],
            'Planned fits exceed the protocol cap')
    return dict(plan=plan, windows=windows, v2=old)


def tasks(bundle):
    plan, old = bundle['plan'], bundle['v2']
    candidates = {grid: {c['id']: c for c in old[grid]['candidates']} for grid in ['ridge', 'lightgbm']}
    result = []
    for wid in bundle['windows']:
        reuse = plan['reuse'].get(wid)
        for design in old['designs']:
            cid = plan['selected'][design['id']]
            result.append(dict(key=f"{wid}_{design['id']}_{cid}", window=wid, design=design['id'],
                               grid=design['grid'], candidate=candidates[design['grid']][cid],
                               kind='reuse' if reuse else 'fit',
                               source=f"w{reuse['v2_window']}_{design['id']}_{cid}" if reuse else None))
    budget = plan['budget']
    require(len({t['key'] for t in result}) == len(result), 'Duplicate schedule')
    require(sum(t['kind'] == 'fit' for t in result) == budget['max_new_fit_calls'], 'New fit budget mismatch')
    require(sum(t['kind'] == 'reuse' for t in result) == budget['reused_window_results'], 'Reuse budget mismatch')
    require(len(result) == budget['total_window_results'], 'Total budget mismatch')
    return result


def preflight(root=ROOT):
    bundle = load_plan(root); schedule = tasks(bundle)
    return dict(status='preflight_passed', plan_sha256=sha(root/PLAN), tasks=len(schedule),
                new_fits_planned=sum(t['kind'] == 'fit' for t in schedule),
                reused_results_to_verify=sum(t['kind'] == 'reuse' for t in schedule),
                formal_fits_executed=0, private_artifacts_read=False,
                execution='Requires explicit --execute; this command only reads frozen public configuration')


def restrict_evaluation(frame, mask, window):
    """Keep whole signal dates whose labels end inside the V3 evaluation window."""
    lo, hi = period(window, 'evaluate')
    inside = mask & frame.signal_date.between(lo, hi)
    kept = inside & frame.label_end_date.le(hi)
    before = inside.groupby(frame.signal_date).sum(); after = kept.groupby(frame.signal_date).sum()
    require(((after == 0) | (after == before)).all(), 'Partially bounded evaluation date')
    require(kept.any(), 'Empty evaluation window')
    return kept


def build_state(frame, train, original, added, design, window, clip):
    """Fit the V2 preprocessing recipe on this window's training rows only."""
    raw = design.startswith('ridge')
    base = frame[original] if raw else cross_sectional_rank(frame, original)
    state = {'base': fit_state(base.loc[train], 'raw' if raw else 'ranked', clip)}
    columns = list(original)
    if design.endswith('enhanced'):
        state['context'] = fit_preprocessing(frame.loc[train, added], clip['raw_clip_quantiles'])
        columns += list(added)
    state.update(train_start=window['train'][0], train_end=window['train'][1],
                 max_label_end=str(frame.loc[train, 'label_end_date'].max()), columns=columns)
    return state


def feature_lists(root):
    original = read_json(root/'config/features_labels.json')['features']
    added = list(read_json(root/'config/context_features_v2.json')['context_features'])
    return original, added


def input_files(root, bundle, schedule):
    """Explicit allowlist: development snapshot, twelve reused V2 results, code."""
    plan = bundle['plan']; run_dir = root/bundle['v2']['outputs']['local']
    files = {root/PLAN, root/plan['snapshot'], run_dir/'fit_ledger.json', run_dir/'execution_verification.json'}
    files.update(root/name for name in plan['source_sha256'])
    files.update(root/name for name in CODE)
    for task in schedule:
        if task['kind'] == 'reuse':
            files.update(run_dir/task['source']/name for name in SOURCE_FILES)
    return sorted(files)


def prepare_reuse(root, bundle, schedule):
    """Verify every reused V2 result before any output directory or fit exists."""
    plan, old = bundle['plan'], bundle['v2']
    run_dir = root/old['outputs']['local']; snapshot = root/plan['snapshot']
    ledger = read_json(run_dir/'fit_ledger.json')
    require(ledger['manifest']['plan_sha256'] == sha(root/plan['v2_plan']), 'V2 run used a different plan')
    require(ledger['manifest']['inputs_and_code'][snapshot.relative_to(root).as_posix()] == sha(snapshot),
            'Development snapshot differs from the V2 run')
    require(read_json(run_dir/'execution_verification.json')['status'] == 'candidate_execution_complete',
            'V2 run is not complete')
    original, added = feature_lists(root)
    results = {}; access = {}
    for wid, spec in plan['reuse'].items():
        source_window = old['development_windows'][spec['v2_window']-1]; window = bundle['windows'][wid]
        print(f'Checking reused window {wid} against V2 window {spec["v2_window"]}', flush=True)
        frame, masks, access[wid] = v2.read_window(snapshot, source_window, original+added)
        require(frame.loc[masks['train'], 'label_end_date'].max() <= period(window, 'train')[1], 'Reused training boundary')
        candidates = frame.signal_date.between(*period(source_window, 'evaluate'))
        expected = frame.loc[candidates, KEYS+['signal_eligible']].reset_index(drop=True)
        evaluate = restrict_evaluation(frame, masks['evaluate'], window)
        kept = frame.signal_date.between(*period(window, 'evaluate'))
        for task in [t for t in schedule if t['window'] == wid]:
            entry = ledger['entries'].get(task['source'])
            require(entry is not None and entry['status'] == 'completed', 'Source task not completed: '+task['source'])
            folder = run_dir/task['source']
            for name in SOURCE_FILES:
                relative = f"{task['source']}/{name}"
                require(entry['artifacts'].get(relative) == sha(folder/name), 'Source artifact changed: '+relative)
            summary = read_json(folder/'summary.json')
            require(summary['window'] == spec['v2_window'] and summary['design'] == task['design']
                    and summary['candidate'] == task['candidate']['id']
                    and summary['parameters'] == v2.parameters(old, task), 'Source summary mismatch: '+task['source'])
            stored = pd.read_parquet(folder/'predictions.parquet')
            pd.testing.assert_frame_equal(stored[expected.columns].reset_index(drop=True), expected)
            prediction = stored.prediction.to_numpy(float)
            require(np.isfinite(prediction).all(), 'Nonfinite reused prediction')
            full = np.full(len(frame), np.nan); full[candidates.to_numpy()] = prediction
            source_ic = v2.daily_ic(frame, full, masks['evaluate'])
            saved = pd.read_csv(folder/'daily_ic.csv', parse_dates=['signal_date']).set_index('signal_date').rank_ic.sort_index()
            pd.testing.assert_index_equal(source_ic.index, saved.index)
            np.testing.assert_allclose(source_ic, saved, atol=1e-12, rtol=1e-12)
            require(summary['dates'] == len(source_ic) and summary['rows'] == int(masks['evaluate'].sum()),
                    'Source evaluation coverage mismatch')
            np.testing.assert_allclose(source_ic.mean(), summary['rank_ic'], atol=1e-12, rtol=1e-12)
            ic = v2.daily_ic(frame, full, evaluate)
            np.testing.assert_allclose(ic, source_ic.loc[ic.index], atol=1e-12, rtol=1e-12)
            results[task['key']] = dict(
                predictions=frame.loc[kept, KEYS+['signal_eligible']].assign(prediction=full[kept.to_numpy()]),
                ic=ic, rows=int(evaluate.sum()), candidate_dates=int(frame.loc[kept, 'signal_date'].nunique()))
    require(len(results) == plan['budget']['reused_window_results'], 'Incomplete reuse verification')
    return results, access


def prepare_fits(root, bundle, schedule):
    """Bounded reads and training-only preprocessing for the windows to be fitted."""
    plan = bundle['plan']; snapshot = root/plan['snapshot']
    original, added = feature_lists(root)
    clip = {name: plan['preprocessing'][name] for name in ['raw_clip_quantiles', 'rank_clip_bounds']}
    data = {}; access = {}
    for wid in plan['fit_windows']:
        window = bundle['windows'][wid]
        print(f'Preparing fit window {wid}: bounded targets and training-only preprocessing', flush=True)
        frame, masks, access[wid] = v2.read_window(snapshot, window, original+added)
        require(frame.loc[masks['train'], 'label_end_date'].max() <= period(window, 'train')[1]
                and frame.loc[masks['train'], 'signal_date'].max() < period(window, 'evaluate')[0], 'Training boundary')
        require(frame.loc[masks['evaluate'], 'label_end_date'].max() <= period(window, 'evaluate')[1], 'Evaluation boundary')
        states = {}; matrices = {}
        for task in [t for t in schedule if t['window'] == wid]:
            # Regression and ranking trees share one recipe; build each recipe once.
            recipe = (task['design'].startswith('ridge'), task['design'].endswith('enhanced'))
            if recipe not in matrices:
                state = build_state(frame, masks['train'], original, added, task['design'], window, clip)
                matrices[recipe] = v2.check_state(state, frame, masks['train'], original, added, task['design'], window)
                states[recipe] = state
        data[wid] = dict(frame=frame, masks=masks, states=states, matrices=matrices)
    return data, access


def aggregate(out, public, bundle, schedule):
    """Public aggregate tables from the completed, ledger-verified task folders."""
    rows = []; daily = []
    for task in schedule:
        summary = read_json(out/task['key']/'summary.json')
        ic = pd.read_csv(out/task['key']/'daily_ic.csv', parse_dates=['signal_date'])
        lo, hi = period(bundle['windows'][task['window']], 'evaluate')
        require(len(ic) == summary['dates'] and ic.signal_date.between(lo, hi).all()
                and np.isfinite(ic.rank_ic).all(), 'Task IC outside its window')
        np.testing.assert_allclose(ic.rank_ic.mean(), summary['rank_ic'], atol=1e-12, rtol=1e-12)
        rows.append({name: summary[name] for name in ['window', 'design', 'candidate', 'kind', 'rank_ic', 'dates', 'rows', 'candidate_dates']})
        daily.append(ic.assign(window=task['window'], design=task['design']))
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index='design', columns='window', values='rank_ic').loc[list(bundle['plan']['selected']), list(bundle['windows'])]
    require(np.isfinite(wide.to_numpy()).all(), 'Missing window result')
    summary = wide.assign(mean_rank_ic=wide.mean(axis=1), positive_windows=wide.gt(0).sum(axis=1),
                          min_window_ic=wide.min(axis=1), max_window_ic=wide.max(axis=1))
    summary.insert(0, 'candidate', [bundle['plan']['selected'][design] for design in summary.index])
    public.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(public/'window_metrics.csv', index=False, lineterminator='\n')
    summary.reset_index().to_csv(public/'baseline_summary.csv', index=False, lineterminator='\n')
    pd.concat(daily, ignore_index=True)[['window', 'design', 'signal_date', 'rank_ic']].to_csv(
        public/'daily_ic.csv', index=False, lineterminator='\n')
    return summary


def execute(root=ROOT, *, resume=False):
    bundle = load_plan(root); schedule = tasks(bundle); plan, old = bundle['plan'], bundle['v2']
    out = root/plan['outputs']['local']; public = root/plan['outputs']['public']
    require(resume or not out.exists(), 'Existing experiment; no implicit rerun')
    require(not resume or (out/'fit_ledger.json').exists(), 'No ledger to resume')
    require(not public.exists(), 'Public baseline already exists')
    files = input_files(root, bundle, schedule)
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in files}
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'pyarrow', 'joblib']}
    manifest = dict(plan_sha256=sha(root/PLAN), inputs_and_code=hashes, versions=versions)
    if resume:
        require(read_json(out/'fit_ledger.json')['manifest'] == manifest, 'Resume input or environment drift')
    reused, reuse_access = prepare_reuse(root, bundle, schedule)
    data, fit_access = prepare_fits(root, bundle, schedule)
    budget = plan['budget']['max_new_fit_calls']
    print('All reused results verified; frozen fit schedule may proceed', flush=True)
    require(all(sha(root/name) == digest for name, digest in hashes.items()), 'Inputs changed during preflight')
    with FitLedger(out, manifest, schedule, budget, resume=resume) as ledger:
        atomic_json(out/'access_audit.json', {**reuse_access, **fit_access})

        def operation(task):
            print(f"Starting {task['kind']}: {task['key']} (reserved fits {ledger.used}/{budget})", flush=True)
            window = bundle['windows'][task['window']]
            folder = out/task['key']; folder.mkdir(exist_ok=False)
            names = []
            if task['kind'] == 'reuse':
                result = reused[task['key']]
                predictions, ic = result['predictions'], result['ic']
                rows, candidate_dates = result['rows'], result['candidate_dates']
            else:
                item = data[task['window']]; frame, masks = item['frame'], item['masks']
                recipe = (task['design'].startswith('ridge'), task['design'].endswith('enhanced'))
                matrix = item['matrices'][recipe]
                model = v2.new_estimator(old, task)
                v2.fit_candidate(model, task, matrix, frame, masks['train'], window['train'][1])
                candidates = frame.signal_date.between(*period(window, 'evaluate'))
                full = np.full(len(frame), np.nan); full[candidates.to_numpy()] = model.predict(matrix.loc[candidates])
                require(np.isfinite(full[candidates.to_numpy()]).all(), 'Nonfinite new prediction')
                ic = v2.daily_ic(frame, full, masks['evaluate'])
                predictions = frame.loc[candidates, KEYS+['signal_eligible']].assign(prediction=full[candidates.to_numpy()])
                rows = int(masks['evaluate'].sum()); candidate_dates = int(frame.loc[candidates, 'signal_date'].nunique())
                if task['grid'] == 'ridge':
                    import joblib
                    joblib.dump(model, folder/'model.joblib'); names.append('model.joblib')
                else:
                    (folder/'model.txt').write_text(model.booster_.model_to_string(), encoding='utf-8'); names.append('model.txt')
                atomic_json(folder/'preprocessing.json', item['states'][recipe]); names.append('preprocessing.json')
            predictions.to_parquet(folder/'predictions.parquet', index=False)
            ic.rename_axis('signal_date').to_csv(folder/'daily_ic.csv', lineterminator='\n')
            atomic_json(folder/'summary.json', dict(
                window=task['window'], design=task['design'], candidate=task['candidate']['id'], kind=task['kind'],
                source=task['source'], train=window['train'], evaluate=window['evaluate'], rank_ic=float(ic.mean()),
                dates=len(ic), rows=rows, candidate_dates=candidate_dates, parameters=v2.parameters(old, task)))
            return [f"{task['key']}/{name}" for name in names+SOURCE_FILES]

        v2.run_tasks(schedule, ledger, operation)
        ledger.verify()
        require(ledger.used == budget and len(ledger.state['entries']) == len(schedule), 'Incomplete frozen budget')
        require(all(sha(root/name) == digest for name, digest in hashes.items()), 'Protected input changed')
        aggregate(out, public, bundle, schedule)
        verification = dict(status='baseline_complete', plan_sha256=sha(root/PLAN), new_fits=ledger.used,
                            reused=len(reused), final_refits=0, backtests=0, seen_period_reads=0, unseen_period_reads=0,
                            selection='none; fixed V2 development selections',
                            next_step='Stage 0b and stage A need their own frozen plans')
        atomic_json(out/'execution_verification.json', verification)
        atomic_json(public/'verification.json', verification)
    return dict(status='baseline_complete', new_fits=budget, reused=len(reused))


def main(*, execute_requested=False, resume=False):
    require(not resume or execute_requested, '--resume requires --execute')
    result = execute(resume=resume) if execute_requested else preflight()
    print(json.dumps(result, indent=2))
