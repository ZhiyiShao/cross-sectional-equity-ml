"""Synthetic-only contracts for the V3 stage 0a baseline; never uses private artifacts.

Fixtures copy frozen public configuration into a temporary root and invent every
observation, prediction and V2 source result there.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from equity_research.models import baseline_v3 as run
from equity_research.models import parameter_robustness as v2
from equity_research.models.robustness_ledger import atomic_json, sha

ORIGINAL = ['a', 'b', 'c']
ADDED = ['d', 'e']
CLIP = dict(raw_clip_quantiles=[.005, .995], rank_clip_bounds=[-.5, .5])
MONTH_DAYS = ['02-02', '05-04', '08-03', '11-02', '12-28']  # the last label crosses each year end


def synthetic_frame(stocks, seed=5):
    dates = pd.to_datetime([f'{year}-{day}' for year in range(2015, 2022) for day in MONTH_DAYS]+['2022-01-04'])
    rng = np.random.default_rng(seed); n = len(dates)*stocks
    frame = pd.DataFrame({'PERMNO': np.tile(np.arange(stocks), len(dates)), 'signal_date': np.repeat(dates, stocks),
                          'signal_eligible': True, 'label_status': 'ok'})
    for column in ORIGINAL+ADDED:
        frame[column] = rng.normal(size=n)
    frame['label_end_date'] = frame.signal_date+pd.Timedelta(days=8)
    frame['y_excess_5d'] = .01*frame.a+.004*frame.d+.02*rng.normal(size=n)
    frame.loc[0, 'signal_eligible'] = False
    frame.loc[1, 'label_status'] = 'missing'
    frame.loc[2, 'y_excess_5d'] = np.nan
    frame.loc[3, 'a'] = np.nan
    return frame


def linear(matrix):
    return (matrix.iloc[:, 0]+.2*matrix.iloc[:, 1]+.1*matrix.iloc[:, -1]).to_numpy()


class SyntheticModel:
    """Deterministic stand-in; records each fit so tests can count and inspect them."""
    def __init__(self, key, log, fail=None):
        self.key = key; self.log = log; self.fail = fail; self.booster_ = self

    def fit(self, matrix, labels, **kwargs):
        self.log.append(dict(key=self.key, rows=len(matrix), arguments=sorted(kwargs)))
        if self.key == self.fail:
            raise RuntimeError('synthetic fit failure')
        return self

    def predict(self, matrix):
        return linear(matrix)

    def model_to_string(self):
        return json.dumps({'key': self.key})


def pin(root, plan, monkeypatch):
    atomic_json(root/run.PLAN, plan)
    monkeypatch.setattr(run, 'PLAN_SHA', sha(root/run.PLAN))


def rehash_source(root, old, key):
    """Make the invented V2 ledger agree with an edited source folder."""
    run_dir = root/old['outputs']['local']; ledger = run.read_json(run_dir/'fit_ledger.json')
    ledger['entries'][key]['artifacts'] = {f'{key}/{name}': sha(run_dir/key/name) for name in run.SOURCE_FILES}
    atomic_json(run_dir/'fit_ledger.json', ledger)


def build_root(root, monkeypatch, stocks=40):
    """Temporary project root with invented data and an invented completed V2 run."""
    for directory in ['config', 'scripts', 'src/equity_research/models', 'results/parameter_robustness_v2',
                      'outputs/context_selection_v2']:
        (root/directory).mkdir(parents=True)
    for name in run.CODE:
        (root/name).write_text('# synthetic implementation manifest fixture\n')
    for name in ['config/research_v3.json', 'config/parameter_robustness_v2.json',
                 'results/parameter_robustness_v2/selection.csv']:
        shutil.copyfile(run.ROOT/name, root/name)
    atomic_json(root/'config/features_labels.json', {'features': ORIGINAL})
    atomic_json(root/'config/context_features_v2.json', {'context_features': dict.fromkeys(ADDED, 'synthetic')})
    plan = run.read_json(run.ROOT/run.PLAN); old = run.read_json(root/'config/parameter_robustness_v2.json')
    snapshot = root/plan['snapshot']; synthetic_frame(stocks).to_parquet(snapshot, index=False)
    plan['source_sha256'] = {name: sha(root/name) for name in plan['source_sha256']}
    pin(root, plan, monkeypatch)
    candidates = {grid: {c['id']: c for c in old[grid]['candidates']} for grid in ['ridge', 'lightgbm']}
    run_dir = root/old['outputs']['local']; entries = {}
    for wi, window in enumerate(old['development_windows'], 1):
        frame, masks, _ = v2.read_window(snapshot, window, ORIGINAL+ADDED)
        ec = frame.signal_date.between(*map(pd.Timestamp, window['evaluate']))
        for design in old['designs']:
            cid = plan['selected'][design['id']]; key = f"w{wi}_{design['id']}_{cid}"
            task = dict(design=design['id'], grid=design['grid'], candidate=candidates[design['grid']][cid])
            state = run.build_state(frame, masks['train'], ORIGINAL, ADDED, design['id'], window, CLIP)
            matrix = v2.check_state(state, frame, masks['train'], ORIGINAL, ADDED, design['id'], window)
            full = np.full(len(frame), np.nan); full[ec.to_numpy()] = linear(matrix.loc[ec])
            ic = v2.daily_ic(frame, full, masks['evaluate'])
            folder = run_dir/key; folder.mkdir(parents=True)
            frame.loc[ec, v2.KEYS+['signal_eligible']].assign(prediction=full[ec.to_numpy()]).to_parquet(
                folder/'predictions.parquet', index=False)
            ic.rename_axis('signal_date').to_csv(folder/'daily_ic.csv', lineterminator='\n')
            atomic_json(folder/'summary.json', dict(window=wi, design=design['id'], candidate=cid, kind='reuse',
                        rank_ic=float(ic.mean()), dates=len(ic), rows=int(masks['evaluate'].sum()),
                        parameters=v2.parameters(old, task)))
            entries[key] = dict(kind='reuse', status='completed',
                                artifacts={f'{key}/{name}': sha(folder/name) for name in run.SOURCE_FILES})
    atomic_json(run_dir/'fit_ledger.json', dict(
        manifest=dict(plan_sha256=sha(root/'config/parameter_robustness_v2.json'),
                      inputs_and_code={plan['snapshot']: sha(snapshot)}), entries=entries))
    atomic_json(run_dir/'execution_verification.json', {'status': 'candidate_execution_complete'})
    return plan, old


def use_synthetic_models(monkeypatch, log, fail=None):
    monkeypatch.setattr(v2, 'new_estimator', lambda plan, task: SyntheticModel(task['key'], log, fail))


def test_real_plan_preflight_reads_no_data(monkeypatch, capsys):
    monkeypatch.setattr(run, 'execute', lambda **kw: pytest.fail('No execution'))
    monkeypatch.setattr(run, 'prepare_reuse', lambda *args: pytest.fail('No private reads'))
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kw: pytest.fail('No data reads'))
    run.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'preflight_passed' and report['formal_fits_executed'] == 0
    assert (report['tasks'], report['new_fits_planned'], report['reused_results_to_verify']) == (24, 12, 12)
    with pytest.raises(ValueError, match='requires'):
        run.main(resume=True)


def test_real_schedule_fits_only_new_training_cutoffs():
    bundle = run.load_plan(); schedule = run.tasks(bundle)
    assert {t['window'] for t in schedule if t['kind'] == 'fit'} == {'W1', 'W4'}
    assert {t['source'][:2] for t in schedule if t['window'] == 'W2'} == {'w1'}
    assert {t['source'][:2] for t in schedule if t['window'] == 'W3'} == {'w2'}
    assert all(t['source'] is None for t in schedule if t['kind'] == 'fit')
    assert json.loads(json.dumps(schedule)) == schedule  # the ledger compares tasks after a JSON round trip
    windows = bundle['windows']; old = bundle['v2']['development_windows']
    assert windows['W2']['train'] == old[0]['train'] and windows['W3']['train'] == old[1]['train']
    assert all(pd.Timestamp(w['evaluate'][1]) < pd.Timestamp('2022-01-01') for w in windows.values())
    selected = {t['design']: t['candidate']['id'] for t in schedule}
    assert selected == bundle['plan']['selected'] and selected['lightgbm_original'] == 'L06'


@pytest.mark.parametrize('fault,message', [('selection', 'differs'), ('reuse_window', 'training interval'),
                                           ('fit_window', 'cover'), ('budget', 'budget'), ('source', 'Frozen source'),
                                           ('unpinned', 'plan changed')])
def test_plan_changes_are_rejected(tmp_path, monkeypatch, fault, message):
    plan, _ = build_root(tmp_path, monkeypatch)
    if fault == 'selection': plan['selected']['ridge_enhanced'] = 'R03'
    if fault == 'reuse_window': plan['reuse']['W3']['v2_window'] = 1
    if fault == 'fit_window': plan['fit_windows'] = ['W1']
    if fault == 'budget': plan['budget']['max_new_fit_calls'] = 11
    if fault == 'unpinned':
        atomic_json(tmp_path/run.PLAN, {**plan, 'status': 'edited'})
    else:
        pin(tmp_path, plan, monkeypatch)
    if fault == 'source':
        (tmp_path/'config/features_labels.json').write_text('{"features": ["a"]}')
    with pytest.raises(ValueError, match=message):
        run.tasks(run.load_plan(tmp_path))


def test_reused_evaluation_keeps_whole_dates_inside_the_new_window(tmp_path, monkeypatch):
    plan, old = build_root(tmp_path, monkeypatch)
    window = run.load_plan(tmp_path)['windows']['W3']
    frame, masks, _ = v2.read_window(tmp_path/plan['snapshot'], old['development_windows'][1], ORIGINAL+ADDED)
    kept = run.restrict_evaluation(frame, masks['evaluate'], window)
    dates = frame.loc[kept, 'signal_date'].drop_duplicates()
    assert dates.dt.strftime('%m-%d').tolist() == MONTH_DAYS[:4] and dates.dt.year.eq(2020).all()
    assert frame.loc[kept, 'label_end_date'].max() <= pd.Timestamp('2020-12-31')
    assert masks['evaluate'][frame.signal_date.eq('2020-12-28')].any()  # evaluable in V2, crossing in V3
    broken = frame.copy()
    broken.loc[kept[kept].index[0], 'label_end_date'] = pd.Timestamp('2021-01-04')
    with pytest.raises(ValueError, match='Partially'):
        run.restrict_evaluation(broken, masks['evaluate'], window)


@pytest.mark.parametrize('design', ['ridge_enhanced', 'lightgbm_original', 'ranker_enhanced'])
def test_preprocessing_uses_training_rows_only(tmp_path, monkeypatch, design):
    plan, _ = build_root(tmp_path, monkeypatch)
    window = run.load_plan(tmp_path)['windows']['W1']
    frame, masks, _ = v2.read_window(tmp_path/plan['snapshot'], window, ORIGINAL+ADDED)
    state = run.build_state(frame, masks['train'], ORIGINAL, ADDED, design, window, CLIP)
    matrix = v2.check_state(state, frame, masks['train'], ORIGINAL, ADDED, design, window)
    assert state['base']['fitted_rows'] == int(masks['train'].sum()) and matrix.columns.tolist() == state['columns']
    assert state['train_end'] == '2017-12-31' and pd.Timestamp(state['max_label_end']) <= pd.Timestamp('2017-12-31')
    assert ('context' in state) == design.endswith('enhanced')
    later = frame.copy(); outside = later.signal_date.gt('2017-12-31')
    later.loc[outside, ORIGINAL+ADDED] = later.loc[outside, ORIGINAL+ADDED]*50+7
    assert run.build_state(later, masks['train'], ORIGINAL, ADDED, design, window, CLIP) == state


def test_full_synthetic_execution_reuses_two_windows_and_fits_two(tmp_path, monkeypatch):
    plan, old = build_root(tmp_path, monkeypatch); log = []
    use_synthetic_models(monkeypatch, log)
    result = run.execute(tmp_path)
    assert result == dict(status='baseline_complete', new_fits=12, reused=12)
    assert len(log) == 12 and {entry['key'][:2] for entry in log} == {'W1', 'W4'}
    assert all(('group' in entry['arguments']) == ('ranker' in entry['key']) for entry in log)
    out = tmp_path/plan['outputs']['local']; public = tmp_path/plan['outputs']['public']
    ledger = run.read_json(out/'fit_ledger.json')
    assert len(ledger['entries']) == 24 and sum(e['kind'] == 'fit' for e in ledger['entries'].values()) == 12
    assert all(e['status'] == 'completed' for e in ledger['entries'].values())

    data = synthetic_frame(40)
    qualified = data.signal_eligible & data.label_status.eq('ok') & data.y_excess_5d.notna()
    for wid, cutoff in [('W1', '2017-12-31'), ('W4', '2020-12-31')]:
        expected = int((qualified & data.signal_date.le(cutoff) & data.label_end_date.le(cutoff)).sum())
        assert {entry['rows'] for entry in log if entry['key'].startswith(wid)} == {expected}

    metrics = pd.read_csv(public/'window_metrics.csv'); daily = pd.read_csv(public/'daily_ic.csv', parse_dates=['signal_date'])
    assert len(metrics) == 24 and metrics.groupby('window').kind.agg(set).to_dict() == {
        'W1': {'fit'}, 'W2': {'reuse'}, 'W3': {'reuse'}, 'W4': {'fit'}}
    assert metrics.dates.eq(4).all() and metrics.candidate_dates.eq(5).all()
    assert daily.signal_date.max() < pd.Timestamp('2021-12-28') and not daily.signal_date.dt.strftime('%m-%d').eq('12-28').any()
    assert (daily.signal_date.dt.year == daily.window.map({'W1': 2018, 'W2': 2019, 'W3': 2020, 'W4': 2021})).all()
    run_dir = tmp_path/old['outputs']['local']
    for design, cid in plan['selected'].items():
        row = lambda wid: metrics[metrics.window.eq(wid) & metrics.design.eq(design)].iloc[0]
        first = run.read_json(run_dir/f'w1_{design}_{cid}'/'summary.json')
        assert row('W2').rank_ic == pytest.approx(first['rank_ic'], abs=1e-12) and row('W2')['rows'] == first['rows']
        second = pd.read_csv(run_dir/f'w2_{design}_{cid}'/'daily_ic.csv', parse_dates=['signal_date'])
        assert len(second) == 9  # four dates in 2020, the 2020 year-end date, four dates in 2021
        kept = second[second.signal_date.le('2020-12-27')]
        assert row('W3').rank_ic == pytest.approx(kept.rank_ic.mean(), abs=1e-12)
        predictions = pd.read_parquet(out/f'W3_{design}_{cid}'/'predictions.parquet')
        assert predictions.signal_date.dt.year.eq(2020).all() and predictions.signal_date.nunique() == 5
        assert (out/f'W1_{design}_{cid}'/'preprocessing.json').exists()
        assert not (out/f'W2_{design}_{cid}'/'preprocessing.json').exists()
    summary = pd.read_csv(public/'baseline_summary.csv')
    assert summary.design.tolist() == list(plan['selected']) and summary.candidate.tolist() == list(plan['selected'].values())
    np.testing.assert_allclose(summary.mean_rank_ic, summary[['W1', 'W2', 'W3', 'W4']].mean(axis=1), atol=1e-15)
    verification = run.read_json(public/'verification.json')
    assert verification['status'] == 'baseline_complete' and verification['new_fits'] == 12
    assert verification['seen_period_reads'] == 0 and verification['final_refits'] == 0

    with pytest.raises(ValueError, match='implicit rerun'):
        run.execute(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        run.execute(tmp_path, resume=True)
    assert len(log) == 12


@pytest.mark.parametrize('fault', ['artifact_hash', 'prediction', 'daily_ic', 'summary', 'rows', 'unfinished',
                                   'snapshot', 'other_plan', 'incomplete_run'])
def test_source_mismatch_blocks_outputs_and_fits(tmp_path, monkeypatch, fault):
    plan, old = build_root(tmp_path, monkeypatch)
    monkeypatch.setattr(v2, 'new_estimator', lambda *args: pytest.fail('No fit before every reused result is verified'))
    run_dir = tmp_path/old['outputs']['local']; key = 'w2_ranker_enhanced_L00'; folder = run_dir/key
    ledger_path = run_dir/'fit_ledger.json'; ledger = run.read_json(ledger_path)
    stored = pd.read_parquet(folder/'predictions.parquet')
    if fault == 'artifact_hash':
        stored.loc[0, 'prediction'] += 1; stored.to_parquet(folder/'predictions.parquet', index=False)
    if fault == 'prediction':
        stored.loc[0, 'prediction'] += 5; stored.to_parquet(folder/'predictions.parquet', index=False)
        rehash_source(tmp_path, old, key)
    if fault == 'rows':
        stored.iloc[1:].to_parquet(folder/'predictions.parquet', index=False); rehash_source(tmp_path, old, key)
    if fault == 'daily_ic':
        saved = pd.read_csv(folder/'daily_ic.csv'); saved.loc[0, 'rank_ic'] -= .1
        saved.to_csv(folder/'daily_ic.csv', index=False, lineterminator='\n'); rehash_source(tmp_path, old, key)
    if fault == 'summary':
        summary = run.read_json(folder/'summary.json'); summary['parameters']['num_leaves'] = 63
        atomic_json(folder/'summary.json', summary); rehash_source(tmp_path, old, key)
    if fault == 'unfinished':
        ledger['entries'][key]['status'] = 'running'; atomic_json(ledger_path, ledger)
    if fault == 'snapshot':
        ledger['manifest']['inputs_and_code'][plan['snapshot']] = '0'*64; atomic_json(ledger_path, ledger)
    if fault == 'other_plan':
        ledger['manifest']['plan_sha256'] = '0'*64; atomic_json(ledger_path, ledger)
    if fault == 'incomplete_run':
        atomic_json(run_dir/'execution_verification.json', {'status': 'running'})
    with pytest.raises((AssertionError, ValueError)):
        run.execute(tmp_path)
    assert not (tmp_path/plan['outputs']['local']).exists() and not (tmp_path/plan['outputs']['public']).exists()


def test_fit_failure_consumes_budget_and_blocks_resume(tmp_path, monkeypatch):
    plan, _ = build_root(tmp_path, monkeypatch); log = []
    use_synthetic_models(monkeypatch, log, fail='W4_ridge_original_R02')
    with pytest.raises(RuntimeError, match='synthetic fit failure'):
        run.execute(tmp_path)
    out = tmp_path/plan['outputs']['local']
    entries = run.read_json(out/'fit_ledger.json')['entries']
    assert entries['W4_ridge_original_R02']['status'] == 'failed'
    assert sum(e['kind'] == 'fit' for e in entries.values()) == len(log)
    assert not (tmp_path/plan['outputs']['public']).exists() and not (out/'.writer.lock').exists()
    with pytest.raises(ValueError, match='review'):
        run.execute(tmp_path, resume=True)
    with pytest.raises(ValueError, match='implicit rerun'):
        run.execute(tmp_path)
    assert sum(entry['key'] == 'W4_ridge_original_R02' for entry in log) == 1


def test_real_estimators_end_to_end_on_invented_observations(tmp_path, monkeypatch):
    """Twelve actual fits on invented data; reloads saved models and reproduces stored predictions."""
    import joblib
    import lightgbm as lgb
    plan, _ = build_root(tmp_path, monkeypatch, stocks=600)
    assert run.execute(tmp_path)['new_fits'] == 12
    out = tmp_path/plan['outputs']['local']
    bundle = run.load_plan(tmp_path); schedule = run.tasks(bundle)
    data, _ = run.prepare_fits(tmp_path, bundle, schedule)
    for task in [t for t in schedule if t['kind'] == 'fit']:
        folder = out/task['key']; item = data[task['window']]
        matrix = item['matrices'][(task['design'].startswith('ridge'), task['design'].endswith('enhanced'))]
        stored = pd.read_parquet(folder/'predictions.parquet')
        rows = item['frame'].signal_date.between(*run.period(bundle['windows'][task['window']], 'evaluate'))
        if task['grid'] == 'ridge':
            model = joblib.load(folder/'model.joblib')
        else:
            model = lgb.Booster(model_str=(folder/'model.txt').read_text(encoding='utf-8'))
        v2.check_model_parameters(model, v2.parameters(bundle['v2'], task), task['grid'] == 'ridge')
        np.testing.assert_allclose(model.predict(matrix.loc[rows]), stored.prediction, atol=1e-12)
        assert run.read_json(folder/'preprocessing.json')['columns'] == matrix.columns.tolist()
    summary = pd.read_csv(tmp_path/plan['outputs']['public']/'baseline_summary.csv').set_index('design')
    assert np.isfinite(summary[['W1', 'W2', 'W3', 'W4']].to_numpy()).all()
    # The invented target depends on feature a and context feature d, so the linear model must find it.
    assert summary.loc['ridge_enhanced', ['W1', 'W4']].gt(.2).all()
