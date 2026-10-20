"""No-fit diagnostics tests: ex-ante groups, censoring and paired uncertainty."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.evaluation.groups import (load_validation, assign_groups, boundary_dates,
    group_statistics, summarize_groups, circular_interval)


def test_saved_prediction_loader_excludes_train_and_test_but_keeps_missing_labels(tmp_path):
    frame = pd.DataFrame({'PERMNO': [1,1,2,1], 'signal_date': pd.to_datetime(['2019-01-01','2020-01-01','2020-01-01','2022-01-01']),
                          'split': ['train','validation','validation','test'], 'score': [1.,2.,3.,1e99],
                          'y': [0.,.1,np.nan,1e99]})
    path = tmp_path/'predictions.parquet'; frame.to_parquet(path,index=False)
    result = load_validation(path,list(frame.columns))
    assert len(result)==2 and result.y.isna().sum()==1 and result.score.max()==3


def sample():
    return pd.DataFrame({'PERMNO': np.arange(1,12), 'signal_date': pd.to_datetime(['2020-01-02']*11),
        'signal_eligible': [True]*11, 'score': [1.]*11, 'y_excess_5d': [np.nan]+[.01]*10,
        'y_excess_5d_event_proxy': [-.2]+[np.nan]*10, 'label_status': ['delist']+['ok']*10})


def test_top_decile_uses_ceiling_ties_and_does_not_depend_on_future_labels():
    frame = sample().sample(frac=1,random_state=7)
    assigned = assign_groups(frame,'score')
    assert set(assigned.loc[assigned.group.eq(10),'PERMNO'])=={1,2}
    changed=frame.copy();changed.y_excess_5d=99.;changed.label_status='ok'
    pd.testing.assert_series_equal(assigned.set_index('PERMNO').group.sort_index(),
                                  assign_groups(changed,'score').set_index('PERMNO').group.sort_index())
    assert assigned.groupby('group').size().sum()==11


def test_missing_top_member_is_not_replaced_or_zero_filled_and_proxy_is_separate():
    frame=sample();assigned=assign_groups(frame,'score')
    boundary=pd.Series(False,index=pd.to_datetime(['2020-01-02']))
    rows,_=group_statistics(assigned,boundary)
    top=rows.loc[rows.group.eq(10)].iloc[0]
    assert top.selected==2 and top.observed==1 and top.coverage==.5
    assert top.mean_excess==pytest.approx(.01)
    assert top.augmented_mean_excess==pytest.approx(-.095)
    assert top.proxy_added==1 and top.proxy_remaining_missing==0
    assert pd.isna(assigned.loc[assigned.PERMNO.eq(1),'y_excess_5d'].item())


def test_missing_entire_day_not_confused_with_administrative_boundary():
    a=sample();a['label_end_date']=pd.Timestamp('2020-01-10');a['purged_boundary']=False
    a.y_excess_5d=np.nan;a.y_excess_5d_event_proxy=np.nan
    b=a.copy();b.signal_date=pd.Timestamp('2021-12-30');b.label_end_date=pd.Timestamp('2022-01-10');b.purged_boundary=True
    frame=pd.concat([a,b],ignore_index=True);boundary=boundary_dates(frame)
    assert boundary.tolist()==[False,True]
    daily,_=group_statistics(assign_groups(frame,'score'),boundary);daily['model']='synthetic'
    summary=summarize_groups(daily)
    top=summary.loc[summary.period.eq('all') & summary.group.eq(10)].iloc[0]
    assert top.dates==1 and top.no_observed_return_dates==1 and top.coverage==0
    assert np.isnan(top.mean_excess)


def test_paired_identical_predictions_have_zero_interval():
    result=circular_interval(np.zeros(100),10,100,42)
    assert result['mean_difference']==result['ci_low']==result['ci_high']==0
