"""Read-only checks of completed local development artifacts; never refit."""
import warnings

import numpy as np
import pandas as pd
import pytest

from equity_research.models import parameter_robustness as execution
from equity_research.evaluation.parameter_robustness import validate_run

ROOT=execution.ROOT
OUT=ROOT/'outputs/parameter_robustness_v2'
pytestmark=pytest.mark.skipif(not (OUT/'execution_verification.json').exists(),reason='Completed local robustness run absent')


def test_all_frozen_tasks_complete_and_protected_inputs_unchanged():
    plan=execution.load_plan()
    _,schedule,ledger=validate_run(ROOT,plan)
    assert len(schedule)==128
    assert sum(v['kind']=='fit' for v in ledger['entries'].values())==80
    assert sum(v['kind']=='reuse' for v in ledger['entries'].values())==48
    assert all(v['status']=='completed' for v in ledger['entries'].values())
    log=execution.read_json(OUT/'access_audit.json')
    for entries in log.values():
        assert 'y_excess_5d' not in entries[0]['columns']
        for entry in entries:
            assert pd.Timestamp(entry['period'][1])<pd.Timestamp('2022-01-01')
            if entry['targets']:assert pd.Timestamp(entry['max_label_end'])<=pd.Timestamp(entry['period'][1])


def test_all_80_saved_new_models_reproduce_sampled_evaluation_predictions():
    import joblib
    import lightgbm as lgb
    from sklearn.exceptions import InconsistentVersionWarning
    plan=execution.load_plan();schedule=execution.tasks(plan)
    original=execution.read_json(ROOT/'config/features_labels.json')['features']
    added=list(execution.read_json(ROOT/'config/context_features_v2.json')['context_features'])
    checked=0
    for wi,window in enumerate(plan['development_windows'],1):
        frame,masks,_=execution.read_window(ROOT/plan['reuse']['development_snapshot'],window,original+added)
        ec=frame.signal_date.between(*window['evaluate']);locations=np.flatnonzero(ec)
        positions=np.unique(np.linspace(0,len(locations)-1,101).astype(int));selected=locations[positions]
        for design in plan['designs']:
            name=design['id'];folder='ranking_selection_v2' if name.startswith('ranker') else 'context_selection_v2'
            state=execution.read_json(ROOT/'outputs'/folder/'preprocessing'/f'w{wi}_{name}.json')
            matrix=execution.check_state(state,frame,masks['train'],original,added,name,window)
            for task in [t for t in schedule if t['window']==wi and t['design']==name and t['kind']=='fit']:
                folder=OUT/task['key']
                if task['grid']=='ridge':
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',InconsistentVersionWarning)
                        model=joblib.load(folder/'model.joblib')
                else:model=lgb.Booster(model_str=(folder/'model.txt').read_text(encoding='utf-8'))
                execution.check_model_parameters(model,execution.parameters(plan,task),task['grid']=='ridge')
                pred=pd.read_parquet(folder/'predictions.parquet',columns=execution.KEYS+['prediction'],
                    filters=[('signal_date','>=',pd.Timestamp(window['evaluate'][0])),('signal_date','<=',pd.Timestamp(window['evaluate'][1]))])
                pd.testing.assert_frame_equal(pred[execution.KEYS].reset_index(drop=True),frame.loc[ec,execution.KEYS].reset_index(drop=True))
                np.testing.assert_allclose(model.predict(matrix.iloc[selected]),pred.prediction.iloc[positions],atol=1e-12,rtol=1e-12)
                checked+=1
    assert checked==80


def test_published_selection_follows_equal_windows_and_prior_anchor_order():
    public=ROOT/'results/parameter_robustness_v2'
    if not (public/'verification.json').exists():pytest.skip('Unified evaluation absent')
    metrics=pd.read_csv(public/'candidate_window_metrics.csv')
    ranking=metrics.groupby(['design','candidate']).rank_ic.mean()
    selected=pd.read_csv(public/'selection.csv')
    plan=execution.load_plan()
    for row in selected.itertuples():
        scores=ranking.loc[row.design].sort_index()
        assert row.candidate==scores.idxmax()
        grid=plan['ridge' if row.design.startswith('ridge') else 'lightgbm']['candidates']
        old=[c['id'] for c in grid if c['reuse_candidate'] is not None]
        assert row.old_candidate==scores.loc[old].idxmax()
        assert row.equal_window_ic==pytest.approx(scores.max())
    pairs=pd.read_csv(public/'paired_ic_intervals.csv')
    assert len(pairs)==186 and pairs.comparison.nunique()==62
    assert set(pairs.window.astype(str))=={'1','2','equal'}
