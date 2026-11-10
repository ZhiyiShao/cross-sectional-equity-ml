"""Synthetic-only contracts for V2 robustness execution; never use private artifacts."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.models import parameter_robustness as run
from equity_research.models.preprocessing import fit_state, fit_preprocessing, cross_sectional_rank
from equity_research.models.robustness_ledger import FitLedger, atomic_json


@pytest.fixture
def plan():
    return run.load_plan()


def frame_fixture():
    dates = pd.to_datetime(['2018-06-01', '2018-12-28', '2019-06-03', '2019-12-30', '2020-06-01', '2021-12-30', '2022-01-04'])
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({'PERMNO': np.tile(np.arange(12), len(dates)),
                          'signal_date': np.repeat(dates, 12), 'signal_eligible': True,
                          'label_status': 'ok', 'a': rng.normal(size=84), 'b': rng.normal(size=84)})
    frame['label_end_date'] = frame.signal_date + pd.Timedelta(days=7)
    frame['y_excess_5d'] = frame.a*.01 + frame.b*.002
    frame.loc[0, 'signal_eligible'] = False
    frame.loc[1, 'label_status'] = 'missing'
    frame.loc[25, 'y_excess_5d'] = np.nan
    return frame


def filtered_reader(frame, calls):
    def read(path, columns, filters):
        mask = pd.Series(True, index=frame.index)
        for col, op, value in filters:
            if op == '>=': mask &= frame[col].ge(value)
            elif op == '<=': mask &= frame[col].le(value)
            elif op == '>': mask &= frame[col].gt(value)
            elif op == '==': mask &= frame[col].eq(value)
            elif op == 'in': mask &= frame[col].isin(value)
            else: raise AssertionError(op)
        result = frame.loc[mask, columns].copy()
        calls.append((columns, filters, result.copy()))
        return result
    return read


def window_fixture(plan):
    frame, masks, _ = run.read_window('synthetic', plan['development_windows'][0], ['a','b'], filtered_reader(frame_fixture(), []))
    return frame, masks


def test_default_entry_is_public_config_only(plan, monkeypatch, capsys):
    monkeypatch.setattr(run, 'execute', lambda **kw: pytest.fail('No execution'))
    monkeypatch.setattr(run, 'prepare_legacy', lambda *args: pytest.fail('No private reads'))
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kw: pytest.fail('No data reads'))
    run.main()
    assert json.loads(capsys.readouterr().out)['formal_fits_executed'] == 0
    schedule = run.tasks(plan)
    assert len(schedule) == 128 and sum(t['kind']=='fit' for t in schedule) == 80
    assert len({t['key'] for t in schedule}) == 128
    with pytest.raises(ValueError, match='requires'):
        run.main(resume=True)


def test_read_window_projects_features_before_bounded_targets(plan):
    calls = []
    frame, masks, _ = run.read_window('synthetic', plan['development_windows'][1], ['a','b'], filtered_reader(frame_fixture(), calls))
    assert 'y_excess_5d' not in calls[0][0]
    assert frame.signal_date.max() < pd.Timestamp('2022-01-01')
    for _, filters, result in calls[1:]:
        upper = next(v for c,o,v in filters if c=='label_end_date' and o=='<=')
        assert result.label_end_date.le(upper).all() and upper < pd.Timestamp('2022-01-01')
    assert frame.loc[frame.label_end_date.ge('2022-01-01'), 'y_excess_5d'].isna().all()
    assert frame.loc[masks['train'], 'label_end_date'].le('2019-12-31').all()
    assert not (masks['train'] & masks['evaluate']).any()
    # All feature candidates remain present, including rows without evaluable labels.
    assert frame.signal_date.eq('2021-12-30').sum() == 12
    assert frame.signal_date.eq('2019-06-03').sum() == 12


@pytest.mark.parametrize('fault', ['future_feature', 'future_target', 'missing_target', 'duplicate_target', 'changed_eligibility'])
def test_reader_rejects_broken_date_and_population_contract(plan, fault):
    base = filtered_reader(frame_fixture(), [])
    def read(path, columns, filters):
        result = base(path, columns, filters)
        if fault == 'future_feature' and 'y_excess_5d' not in columns:
            result.loc[result.index[0], 'signal_date'] = pd.Timestamp('2022-01-04')
        if 'y_excess_5d' in columns:
            if fault == 'future_target': result.loc[result.index[0], 'label_end_date'] = pd.Timestamp('2022-01-04')
            if fault == 'missing_target': result = result.iloc[1:]
            if fault == 'duplicate_target': result = pd.concat([result, result.iloc[:1]])
            if fault == 'changed_eligibility': result.loc[result.index[0], 'signal_eligible'] = False
        return result
    with pytest.raises(ValueError):
        run.read_window('synthetic', plan['development_windows'][0], ['a','b'], read)


def state_fixture(frame, train, raw=True):
    original = ['a']; added = ['b']
    base = frame[original] if raw else cross_sectional_rank(frame, original)
    state = dict(base=fit_state(base.loc[train], 'raw' if raw else 'ranked',
                               dict(raw_clip_quantiles=[.005,.995], rank_clip_bounds=[-.5,.5])),
                 context=fit_preprocessing(frame.loc[train, added], [.005,.995]),
                 columns=original+added, train_start='2015-01-01', train_end='2018-12-31',
                 max_label_end=str(frame.loc[train, 'label_end_date'].max()))
    return state


@pytest.mark.parametrize('design', ['ridge_enhanced','lightgbm_enhanced','ranker_enhanced'])
def test_saved_preprocessing_reused_and_validated(plan, design):
    frame, masks = window_fixture(plan)
    state = state_fixture(frame, masks['train'], design.startswith('ridge'))
    before = copy.deepcopy(state)
    matrix = run.check_state(state, frame, masks['train'], ['a'], ['b'], design, plan['development_windows'][0])
    assert matrix.shape == (len(frame), 2) and state == before
    changed = copy.deepcopy(state); changed['base']['mean'][0] += .1
    with pytest.raises(AssertionError):
        run.check_state(changed, frame, masks['train'], ['a'], ['b'], design, plan['development_windows'][0])
    changed = copy.deepcopy(state); changed['columns'] = ['b','a']
    with pytest.raises(ValueError, match='order'):
        run.check_state(changed, frame, masks['train'], ['a'], ['b'], design, plan['development_windows'][0])


class LinearPredictor:
    def predict(self, matrix):
        return matrix.a.to_numpy() + .2*matrix.b.to_numpy()


def reuse_fixture(plan):
    frame, masks = window_fixture(plan); matrix = frame[['a','b']]
    model = LinearPredictor(); window = plan['development_windows'][0]
    ec = frame.signal_date.between(*window['evaluate'])
    saved = frame.loc[ec, run.KEYS+['signal_eligible']].assign(prediction=model.predict(matrix.loc[ec])).reset_index(drop=True)
    full = np.full(len(frame), np.nan); full[ec] = saved.prediction
    daily = run.daily_ic(frame, full, masks['evaluate']).rename_axis('signal_date').reset_index()
    train = frame.loc[masks['train']]
    ledger = dict(train_rows=len(train), max_train_signal=str(train.signal_date.max()),
                  max_train_label=str(train.label_end_date.max()), eval_start=window['evaluate'][0])
    metric = dict(dates=len(daily), rows=int(masks['evaluate'].sum()), rank_ic=daily.rank_ic.mean())
    return dict(task={}, frame=frame, matrix=matrix, masks=masks, model=model, stored_prediction=saved,
                stored_daily=daily, metric=metric, ledger=ledger, window=window)


def test_reused_candidate_reproduces_saved_prediction_and_independent_ic(plan):
    args = reuse_fixture(plan)
    prediction, ic = run.verify_reuse_predictions(**args)
    assert np.isfinite(ic).all() and len(prediction)==len(args['frame'])


@pytest.mark.parametrize('fault', ['prediction','rows','daily_ic','coverage','train_boundary'])
def test_reuse_mismatch_stops_without_fit(plan, fault):
    args = reuse_fixture(plan)
    if fault == 'prediction': args['stored_prediction'].loc[0, 'prediction'] += 1
    if fault == 'rows': args['stored_prediction'] = args['stored_prediction'].iloc[1:]
    if fault == 'daily_ic': args['stored_daily'].loc[0, 'rank_ic'] -= .1
    if fault == 'coverage': args['metric']['rows'] += 1
    if fault == 'train_boundary': args['ledger']['max_train_label'] = '2022-01-01'
    with pytest.raises((AssertionError, ValueError)):
        run.verify_reuse_predictions(**args)


def test_128_task_schedule_consumes_exactly_80_and_resume_skips_all(plan, tmp_path):
    schedule = run.tasks(plan); directory = tmp_path/'run'; calls = []
    def operation(task):
        calls.append(task['kind'])
        name = task['key']+'.json'; atomic_json(directory/name, {'key':task['key']})
        return [name]
    with FitLedger(directory, {'synthetic':True}, schedule, 80) as ledger:
        run.run_tasks(schedule, ledger, operation)
        assert ledger.used == 80 and len(ledger.state['entries']) == 128
        ledger.verify()
    assert calls == ['reuse']*48 + ['fit']*80
    with FitLedger(directory, {'synthetic':True}, schedule, 80, resume=True) as ledger:
        run.run_tasks(schedule, ledger, lambda task: pytest.fail('Must not repeat completed task'))
        assert ledger.used == 80


def test_failure_reserved_before_call_consumes_budget_and_blocks_retry(tmp_path):
    task = {'key':'a','kind':'fit'}; directory=tmp_path/'run'
    with FitLedger(directory, {'synthetic':True}, [task], 1) as ledger:
        def fail(task):
            disk = json.loads(ledger.path.read_text())
            assert disk['entries']['a']['status'] == 'running'
            assert ledger.used == 1
            raise RuntimeError('synthetic fit failure')
        with pytest.raises(RuntimeError): ledger.perform(task, fail)
        assert ledger.used == 1
        with pytest.raises(ValueError, match='retry'): ledger.perform(task, fail)
    with pytest.raises(ValueError, match='review'):
        FitLedger(directory, {'synthetic':True}, [task], 1, resume=True)


def test_budget_lock_and_artifact_integrity(tmp_path):
    schedule = [{'key':key, 'kind':'fit'} for key in ['a','b']]; directory=tmp_path/'run'
    manifest = {'synthetic':True}
    with FitLedger(directory, manifest, schedule, 1) as ledger:
        with pytest.raises(FileExistsError): FitLedger(directory, manifest, schedule, 1, resume=True)
        def operation(task):
            atomic_json(directory/'a.json', {'done':True}); return ['a.json']
        ledger.perform(schedule[0], operation)
        with pytest.raises(ValueError, match='exhausted'): ledger.perform(schedule[1], operation)
    with pytest.raises(ValueError, match='manifest'):
        FitLedger(directory, {'synthetic':False}, schedule, 1, resume=True)
    (directory/'a.json').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        FitLedger(directory, manifest, schedule, 1, resume=True)


def test_running_reservation_and_external_artifacts_are_rejected(tmp_path):
    task = {'key':'a','kind':'fit'}; directory=tmp_path/'run'
    with FitLedger(directory, {}, [task], 1) as ledger:
        with pytest.raises(ValueError, match='outside'):
            ledger.artifact_path('../external.json')
        ledger.state['entries']['a'] = dict(kind='fit', status='running'); ledger.save()
    with pytest.raises(ValueError, match='review'):
        FitLedger(directory, {}, [task], 1, resume=True)


def test_reuse_failure_prevents_output_and_fit(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_plan', lambda root: plan)
    marker = tmp_path/run.PLAN; marker.parent.mkdir(); marker.write_text('synthetic')
    monkeypatch.setattr(run, 'legacy_files', lambda *args: [marker])
    monkeypatch.setattr(run, 'prepare_legacy', lambda *args: (_ for _ in ()).throw(ValueError('Legacy mismatch')))
    monkeypatch.setattr(run, 'new_estimator', lambda *args: pytest.fail('No fitting allowed'))
    with pytest.raises(ValueError, match='Legacy mismatch'): run.execute(tmp_path)
    assert not (tmp_path/plan['outputs']['local']).exists()


class FitSpy:
    def fit(self, x, y, **kwargs):
        self.seen = (x.copy(), np.asarray(y), kwargs)


def test_fit_boundary_and_query_groups(plan):
    frame, masks = window_fixture(plan); model = FitSpy()
    task = {'design':'ranker_original'}
    run.fit_candidate(model, task, frame[['a','b']], frame, masks['train'], '2018-12-31')
    x,y,kw = model.seen
    assert len(y) == masks['train'].sum() == sum(kw['group'])
    assert np.issubdtype(y.dtype, np.integer) and y.min() >= 0 and y.max() <= 9
    with pytest.raises(ValueError, match='boundary'):
        run.fit_candidate(FitSpy(), task, frame[['a','b']], frame, masks['evaluate'], '2018-12-31')


@pytest.mark.parametrize('family', ['ridge','lightgbm','ranker'])
def test_real_estimators_synthetic_fit_and_serialization(plan, tmp_path, family):
    """One actual fit per family on invented observations; never production inputs."""
    import joblib
    import lightgbm as lgb
    rng = np.random.default_rng(13)
    matrix = pd.DataFrame(rng.normal(size=(1200, 2)), columns=['a','b'])
    frame = pd.DataFrame({'signal_date':np.repeat(pd.to_datetime(['2017-01-03','2017-02-01']),600),
                          'label_end_date':np.repeat(pd.to_datetime(['2017-01-10','2017-02-08']),600),
                          'signal_eligible':True,'label_status':'ok','y_excess_5d':matrix.a*.01+matrix.b*.002})
    task = next(t for t in run.tasks(plan) if t['design']==family+'_original' and t['kind']=='fit')
    model = run.new_estimator(plan, task)
    run.fit_candidate(model, task, matrix, frame, pd.Series(True,index=frame.index), '2018-12-31')
    if family == 'ridge':
        joblib.dump(model,tmp_path/'model.joblib'); loaded=joblib.load(tmp_path/'model.joblib')
    else:
        loaded=lgb.Booster(model_str=model.booster_.model_to_string())
    run.check_model_parameters(loaded, run.parameters(plan,task), family=='ridge')
    np.testing.assert_allclose(model.predict(matrix),loaded.predict(matrix),atol=1e-12)
    bad=run.parameters(plan,task);bad['alpha' if family=='ridge' else 'num_leaves'] += 1
    with pytest.raises(ValueError, match='parameter mismatch'):
        run.check_model_parameters(loaded,bad,family=='ridge')


class SyntheticModel:
    """Deterministic test double for the complete 128-task orchestration."""
    def __init__(self, params, columns):
        self.params = params
        self.feature_names_in_ = np.asarray(columns)
        self.booster_ = self

    def get_params(self):
        return self.params

    def feature_name(self):
        return list(self.feature_names_in_)

    def predict(self, matrix):
        return matrix.iloc[:,0].to_numpy()+.2*matrix.iloc[:,1].to_numpy()

    def fit(self, matrix, labels, **kwargs):
        self.feature_names_in_ = np.asarray(matrix.columns)
        return self

    def model_to_string(self):
        return json.dumps({'params':self.params,'columns':self.feature_name()})


def make_legacy_fixture(root, frozen):
    """Write all old artifacts from invented observations, without fitting models."""
    import joblib
    import lightgbm as lgb
    plan = copy.deepcopy(frozen)
    plan['source_sha256'] = {}
    for directory in ['config', 'outputs/context_selection_v2/models', 'outputs/context_selection_v2/preprocessing',
                      'outputs/ranking_selection_v2/models', 'outputs/ranking_selection_v2/preprocessing',
                      'results/context_selection_v2', 'results/ranking_selection_v2',
                      'src/equity_research/models', 'scripts']:
        (root/directory).mkdir(parents=True, exist_ok=True)
    for name in ['src/equity_research/models/parameter_robustness.py','src/equity_research/models/robustness_ledger.py',
                 'src/equity_research/models/preprocessing.py','scripts/train_models.py']:
        (root/name).write_text('# synthetic implementation manifest fixture\n')
    original = ['a','b']+[f'x{i}' for i in range(19)]; added = [f'c{i}' for i in range(12)]
    frame = frame_fixture()
    rng = np.random.default_rng(21)
    for col in original[2:]+added: frame[col]=rng.normal(size=len(frame))
    # Real Parquet contains future and crossing labels to test IO filters end to end.
    snapshot=root/plan['reuse']['development_snapshot'];frame.to_parquet(snapshot,index=False)
    context_cfg={'context_features':dict.fromkeys(added,'synthetic')}
    parent={'development_windows':plan['development_windows']}; ranking_cfg={'synthetic':True}
    for name,value in [('features_labels',{'features':original}),('context_features_v2',context_cfg),
                       ('research_v2',parent),('ranking_v2',ranking_cfg)]:
        atomic_json(root/'config'/f'{name}.json',value)
    schedule=run.tasks(plan); metadata={name:dict(ledger=[],daily=[],metrics=[]) for name in ['context_selection_v2','ranking_selection_v2']}
    rank_hashes={snapshot.relative_to(root).as_posix():run.sha(snapshot)}
    for wi,window in enumerate(plan['development_windows'],1):
        data,masks,_=run.read_window(snapshot,window,original+added)
        plan['expected_evaluation_coverage'][wi-1]=dict(window=wi,dates=data.loc[masks['evaluate'],'signal_date'].nunique(),strict_rows=int(masks['evaluate'].sum()))
        ec=data.signal_date.between(*window['evaluate']);predictions={}
        for directory in metadata:
            predictions[directory]=data.loc[ec,run.KEYS+['signal_eligible']].copy()
        for design in plan['designs']:
            name=design['id']; directory='ranking_selection_v2' if name.startswith('ranker') else 'context_selection_v2'
            raw=name.startswith('ridge'); enhanced=name.endswith('enhanced')
            values=data[original] if raw else cross_sectional_rank(data,original)
            state=dict(base=fit_state(values.loc[masks['train']], 'raw' if raw else 'ranked',
                                     dict(raw_clip_quantiles=[.005,.995],rank_clip_bounds=[-.5,.5])),
                       columns=original+(added if enhanced else []),train_start=window['train'][0],train_end=window['train'][1],
                       max_label_end=str(data.loc[masks['train'],'label_end_date'].max()))
            if enhanced:state['context']=fit_preprocessing(data.loc[masks['train'],added],[.005,.995])
            statepath=root/'outputs'/directory/'preprocessing'/f'w{wi}_{name}.json';atomic_json(statepath,state)
            if name.startswith('lightgbm'):rank_hashes[statepath.relative_to(root).as_posix()]=run.sha(statepath)
            matrix=run.check_state(state,data,masks['train'],original,added,name,window)
            for task in [t for t in schedule if t['kind']=='reuse' and t['window']==wi and t['design']==name]:
                ci=task['candidate']['reuse_candidate'];model=SyntheticModel(run.parameters(plan,task),matrix.columns)
                stem=root/'outputs'/directory/'models'/f'w{wi}_{name}_{ci}'
                if raw:joblib.dump(model,stem.with_suffix('.joblib'))
                else:stem.with_suffix('.txt').write_text(model.model_to_string(),encoding='utf-8')
                full=np.full(len(data),np.nan);full[ec]=model.predict(matrix.loc[ec])
                predictions[directory][f'{name}_{ci}']=full[ec]
                daily=run.daily_ic(data,full,masks['evaluate']).rename_axis('signal_date').reset_index()
                daily=daily.assign(window=wi,design=name,candidate=ci)
                metadata[directory]['daily'].append(daily)
                metadata[directory]['metrics'].append(dict(window=wi,design=name,candidate=ci,
                    parameter=task['candidate'].get('alpha',task['candidate'].get('num_leaves')),
                    rank_ic=daily.rank_ic.mean(),dates=len(daily),rows=int(masks['evaluate'].sum())))
                training=data.loc[masks['train']]
                metadata[directory]['ledger'].append(dict(window=wi,design=name,candidate=ci,train_rows=len(training),
                    max_train_signal=str(training.signal_date.max()),max_train_label=str(training.label_end_date.max()),eval_start=window['evaluate'][0]))
        for directory,prediction in predictions.items():prediction.to_parquet(root/'outputs'/directory/f'w{wi}_predictions_local.parquet',index=False)
    for directory,content in metadata.items():
        folder=root/'outputs'/directory
        atomic_json(folder/'fit_ledger.json',content['ledger'])
        pd.concat(content['daily']).to_csv(folder/'candidate_daily_ic.csv',index=False)
        pd.DataFrame(content['metrics']).to_csv(root/'results'/directory/'candidate_metrics.csv',index=False)
        atomic_json(folder/'frozen_plan.json',dict(config=context_cfg if directory.startswith('context') else ranking_cfg,parent=parent))
        atomic_json(folder/'verification.json',dict(fits=32 if directory.startswith('context') else 16,
                    lightgbm_version=lgb.__version__,source_hashes=rank_hashes))
    atomic_json(root/run.PLAN,plan)
    return plan


def fake_booster(model_str):
    value=json.loads(model_str)
    return SyntheticModel(value['params'],value['columns'])


def test_full_synthetic_legacy_adapter_execution_and_resume(plan,tmp_path,monkeypatch):
    import lightgbm as lgb
    synthetic=make_legacy_fixture(tmp_path,plan)
    monkeypatch.setattr(run,'load_plan',lambda root:synthetic)
    monkeypatch.setattr(lgb,'Booster',fake_booster)
    calls=[]
    def estimator(plan,task):
        calls.append(task['key'])
        return SyntheticModel(run.parameters(plan,task),[])
    monkeypatch.setattr(run,'new_estimator',estimator)
    result=run.execute(tmp_path)
    assert result['new_fits']==len(calls)==80 and len(set(calls))==80
    ledger=run.read_json(tmp_path/synthetic['outputs']['local']/'fit_ledger.json')
    assert len(ledger['entries'])==128 and sum(e['kind']=='reuse' for e in ledger['entries'].values())==48
    run.execute(tmp_path,resume=True)
    assert len(calls)==80
    with pytest.raises(ValueError,match='implicit rerun'):run.execute(tmp_path)
    # Drift in a completed prediction prevents resumed use before another estimator exists.
    artifact=next((tmp_path/synthetic['outputs']['local']).glob('*/predictions.parquet'))
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifact changed'):run.execute(tmp_path,resume=True)
    assert len(calls)==80


def test_last_old_candidate_mismatch_blocks_all_new_fits(plan,tmp_path,monkeypatch):
    import lightgbm as lgb
    synthetic=make_legacy_fixture(tmp_path,plan)
    monkeypatch.setattr(run,'load_plan',lambda root:synthetic)
    monkeypatch.setattr(lgb,'Booster',fake_booster)
    monkeypatch.setattr(run,'new_estimator',lambda *args:pytest.fail('No new fit before all reuse checks'))
    path=tmp_path/'outputs/ranking_selection_v2/w2_predictions_local.parquet'
    data=pd.read_parquet(path);data.loc[0,'ranker_enhanced_3']+=1;data.to_parquet(path,index=False)
    with pytest.raises(AssertionError):run.execute(tmp_path)
    assert not (tmp_path/synthetic['outputs']['local']).exists()
