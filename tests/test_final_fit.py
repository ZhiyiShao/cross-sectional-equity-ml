import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from equity_research.models.final_fit import eligibility, read_final_inputs, END
from equity_research.models.preprocessing import cross_sectional_rank

def test_restored_boundary_and_test_purge():
    f=pd.DataFrame(dict(signal_date=pd.to_datetime(['2019-12-30','2021-12-30','2022-01-03']),
        label_end_date=pd.to_datetime(['2020-01-08','2022-01-10','2022-01-11']),signal_eligible=[True]*3,label_status=['ok']*3))
    assert eligibility(f).tolist()==[True,False,False]

def test_ranks_ignore_future_labels_and_ineligible_stocks():
    f=pd.DataFrame(dict(signal_date=pd.to_datetime(['2021-01-04']*4),signal_eligible=[True,True,True,False],x=[1,2,3,100],label_status=['ok','missing','ok','ok']))
    r=cross_sectional_rank(f,['x'])
    np.testing.assert_allclose(r.x[:3],[-1/3,0,1/3])
    f.label_status='anything'
    pd.testing.assert_frame_equal(r,cross_sectional_rank(f,['x']))

def test_loader_filters_targets_before_read_and_keeps_rank_population(monkeypatch):
    calls=[]
    candidates=pd.DataFrame(dict(PERMNO=[1,2],signal_date=pd.to_datetime(['2019-12-30','2021-12-30']),split=['train','validation'],
        signal_eligible=[True,True],label_status=['ok','ok'],label_end_date=pd.to_datetime(['2020-01-08','2022-01-10']),
        eligible_refit_through_2021=[True,False],eligible_main_sample=[False,False],purged_boundary=[True,True],x=[1.,2.]))
    def read(path,columns,filters):
        calls.append(filters)
        assert ('signal_date','<=',END) in filters
        if 'y_excess_5d' in columns:
            assert ('label_end_date','<=',END) in filters
            return candidates.iloc[:1][['PERMNO','signal_date','label_end_date']].assign(y_excess_5d=.1)
        return candidates.copy()
    monkeypatch.setattr(pd,'read_parquet',read)
    frame,mask,_=read_final_inputs(['x'])
    assert len(calls)==2 and len(frame)==2 and mask.tolist()==[True,False]
    assert pd.isna(frame.y_excess_5d.iloc[1])

def test_loader_rejects_returned_test_candidate(monkeypatch):
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pd.DataFrame({'signal_date':pd.to_datetime(['2022-01-03'])}))
    with pytest.raises(ValueError,match='outside development'):read_final_inputs(['x'])
