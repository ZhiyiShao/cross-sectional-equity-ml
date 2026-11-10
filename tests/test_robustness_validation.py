import numpy as np
import pandas as pd
import pytest

from equity_research.backtest import robustness_validation as rv


def selection_fixture():
    designs=[f'design_{i}' for i in range(6)]
    selection=pd.DataFrame({'design':designs,'candidate':'L04','old_candidate':'L00'})
    metrics=pd.DataFrame([dict(design=d,candidate=c,window=w,rank_ic=ic)
                         for d in designs for c,ic in [('L00',.01),('L01',0),('L04',.02)] for w in [1,2]])
    return selection,metrics,designs


def test_selection_remains_ic_based_and_requires_both_windows():
    s,m,d=selection_fixture()
    rv.verify_selection(s,m,d)
    with pytest.raises(ValueError,match='Selection changed'):
        rv.verify_selection(s.assign(candidate='L00'),m,d)
    with pytest.raises(ValueError,match='Incomplete'):
        rv.verify_selection(s,m.drop(index=0),d)


def test_saved_score_projection_and_date_guard():
    data=pd.DataFrame({'PERMNO':[1],'signal_date':[pd.Timestamp('2020-01-02')],
                       'signal_eligible':[True],'prediction':[.1]})
    def reader(path,**kwargs):
        assert kwargs['columns']==rv.KEYS+['prediction']
        assert kwargs['filters']==[('signal_date','>=',rv.START),('signal_date','<=',rv.END)]
        return data.copy()
    assert rv.read_scores('dummy','score',reader).score.item()==.1
    data['signal_date']=pd.Timestamp('2022-01-03')
    with pytest.raises(ValueError,match='Validation-only'):
        rv.read_scores('dummy','score',reader)


def test_saved_scores_reject_duplicates_and_missing_eligible():
    data=pd.DataFrame({'PERMNO':[1],'signal_date':[pd.Timestamp('2020-01-02')],
                       'signal_eligible':[True],'prediction':[np.nan]})
    with pytest.raises(ValueError,match='Missing eligible'):
        rv.read_scores('dummy','score',lambda *a,**k:data)
    data['prediction']=.1
    with pytest.raises(ValueError,match='Duplicate'):
        rv.read_scores('dummy','score',lambda *a,**k:pd.concat([data,data]))


def test_paired_net_difference_drops_anchor_and_preserves_dates():
    old=pd.DataFrame({'date':pd.bdate_range('2020-01-02',periods=31),
        'design':'lightgbm_original','variant':'constrained','return_net':0.})
    new=old.copy();new['return_net']=.001;new.loc[0,'return_net']=999
    result=rv.paired_comparison(new,old).iloc[0]
    assert result.return_days==30
    np.testing.assert_allclose([result.annual_mean_net_difference,result.ci_low,result.ci_high],.252)
    with pytest.raises(AssertionError):rv.paired_comparison(new.iloc[1:],old)


def test_target_limits_are_required_but_execution_drift_reported():
    limits=dict(max_weight=.02,industry_active=.02,beta_active=.1,annual_stock_vol_active_upper=.1)
    data=pd.DataFrame([dict(design='x',variant='constrained',stage=stage,date='2020-01-02',
        beta=1.,benchmark_beta=1.,annual_stock_vol=.3,benchmark_vol=.3,
        max_weight=weight,max_industry_active=.01,coverage=1.) for stage,weight in [('target',.02),('executed',.03)]])
    summary,count=rv.exposure_summary(data,limits)
    assert count==1 and summary.weight_breaches.sum()==1
    data.loc[0,'beta']=1.2
    with pytest.raises(ValueError,match='constraint violation'):rv.exposure_summary(data,limits)
