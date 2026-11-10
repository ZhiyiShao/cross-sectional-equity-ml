"""Independent synthetic checks of frozen selection, paired bootstrap and tail metrics."""
import numpy as np
import pandas as pd
import pytest

from equity_research.evaluation import parameter_robustness as evaluation
from equity_research.models import parameter_robustness as execution


def metrics_fixture():
    plan=execution.load_plan();rows=[]
    for task in execution.tasks(plan):
        # Old R00/L00 favors 2019. New candidate 04 wins equal-window mean,
        # but deliberately sacrifices 2020-2021. Date weighting would prefer 00.
        ci=int(task['candidate']['id'][1:]);wi=task['window']
        value=([.01,.02] if ci==0 else [.04,0.] if ci==4 else [.005,.005])[wi-1]
        rows.append(dict(design=task['design'],candidate=task['candidate']['id'],window=wi,rank_ic=value))
    return plan,pd.DataFrame(rows)


def test_selection_equal_windows_and_old_anchor_tie_priority():
    plan,metrics=metrics_fixture()
    candidates,selected=evaluation.candidate_tables(plan,metrics)
    assert len(candidates)==64 and selected.candidate.str.endswith('04').all()
    assert selected.old_candidate.str.endswith('00').all()
    assert selected.robustness_flag.eq('window_tradeoff').all()
    assert selected.equal_window_delta.tolist()==pytest.approx([.005]*6)
    assert len(evaluation.contrasts(plan,selected))==62
    metrics['rank_ic']=.01
    _,selected=evaluation.candidate_tables(plan,metrics)
    assert selected.candidate.str.endswith('00').all()
    assert selected.robustness_flag.eq('unchanged').all()


def test_bootstrap_matches_independent_paired_loop_and_equal_weighting():
    rows=[]
    for wi,n in [(1,12),(2,21)]:
        for i,date in enumerate(pd.date_range(f'{2018+wi}-01-01',periods=n)):
            for candidate in ['a','b']:
                rows.append(dict(window=wi,signal_date=date,design='toy',candidate=candidate,
                                 rank_ic=.001*i+(np.cos(i)*.01+wi*.002 if candidate=='b' else 0)))
    daily=pd.DataFrame(rows)
    pairs=[dict(comparison='paired',kind='selection',baseline_design='toy',baseline_candidate='a',treatment_design='toy',treatment_candidate='b')]
    cfg=dict(seed=20261107,block_signals=10,replications=2000)
    actual=evaluation.paired_intervals(daily,pairs,cfg).set_index('window')
    rng=np.random.default_rng(cfg['seed']);window_draws=[];means=[]
    for wi,n in [(1,12),(2,21)]:
        d=np.cos(np.arange(n))*.01+wi*.002;means.append(d.mean())
        starts=rng.integers(n,size=(2000,int(np.ceil(n/10))))
        samples=[]
        for row in starts:
            indices=np.concatenate([(start+np.arange(10))%n for start in row])[:n]
            samples.append(d[indices].mean())
        window_draws.append(samples)
        np.testing.assert_allclose(actual.loc[str(wi),['ci_low','ci_high']].astype(float),np.quantile(samples,[.025,.975]),atol=1e-14)
    np.testing.assert_allclose(actual.loc['equal',['ci_low','ci_high']].astype(float),np.quantile(np.mean(window_draws,axis=0),[.025,.975]),atol=1e-14)
    assert actual.loc['equal','delta']==pytest.approx(np.mean(means))
    same=[dict(pairs[0],treatment_candidate='a')]
    zero=evaluation.paired_intervals(daily,same,cfg)
    assert zero[['delta','ci_low','ci_high']].eq(0).all().all()


def test_top_decile_determined_before_label_mask_and_ties_use_permno():
    f=pd.DataFrame(dict(PERMNO=np.arange(20),signal_date=pd.Timestamp('2019-01-02'),
                        signal_eligible=True,y_excess_5d=np.arange(20)*.001))
    prediction=np.arange(20,dtype=float)
    prediction[0]=100;prediction[1]=100
    valid=pd.Series(True,index=f.index);valid.iloc[0]=False;f.loc[0,'y_excess_5d']=np.nan
    result=evaluation.prediction_diagnostics(f,prediction,valid).iloc[0]
    assert result.selected==2 and result.top10_valid_count==1 and result.top10_coverage==.5
    assert result.top10_mean_label==pytest.approx(.001)
    assert 0<=result.ndcg_top10<=1
    # Increasing score of an ineligible security cannot alter selection or NDCG.
    f.loc[19,'signal_eligible']=False;valid.iloc[19]=False
    expected=evaluation.prediction_diagnostics(f,prediction,valid)
    prediction[19]=1e10
    pd.testing.assert_frame_equal(evaluation.prediction_diagnostics(f,prediction,valid),expected)


def test_factorial_interaction_and_boundary_signs():
    rows=[]
    for leaf,tree,minimum in [(7,100,500),(7,300,500),(7,100,100),(7,300,100),
                              (63,100,500),(63,300,500),(63,100,100),(63,300,100),(3,100,500),(127,100,500)]:
        value=leaf*.0001+(tree==300)*.01+(minimum==100)*.02+(tree==300 and minimum==100)*.03
        rows.append(dict(design='lightgbm_original',num_leaves=leaf,n_estimators=tree,min_child_samples=minimum,
                         ic_2019=value,ic_2020_2021=value*2,equal_window_ic=value*1.5))
    effect,boundary=evaluation.factorial_effects(pd.DataFrame(rows))
    first=effect[effect.window.eq('1')]
    assert first.interaction.tolist()==pytest.approx([.03,.03])
    assert first.trees_at_min500.tolist()==pytest.approx([.01,.01])
    assert first.trees_at_min100.tolist()==pytest.approx([.04,.04])
    assert boundary.loc[boundary.window.eq('1'),'delta'].tolist()==pytest.approx([-.0004,.0064])


def test_missing_window_is_not_silently_averaged():
    plan,metrics=metrics_fixture()
    with pytest.raises(ValueError,match='coverage'):
        evaluation.candidate_tables(plan,metrics.iloc[:-1])


def test_boundary_signal_without_strict_labels_keeps_coverage_not_fake_ic():
    frame=pd.DataFrame(dict(PERMNO=np.arange(20),signal_date=pd.Timestamp('2021-12-30'),
                            signal_eligible=True,y_excess_5d=np.nan))
    result=evaluation.prediction_diagnostics(frame,np.arange(20),pd.Series(False,index=frame.index)).iloc[0]
    assert result.selected==2 and result.top10_valid_count==0 and result.top10_coverage==0
    assert np.isnan(result.rank_ic) and np.isnan(result.top10_mean_label)
