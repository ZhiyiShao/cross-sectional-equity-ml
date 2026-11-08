import numpy as np
import pandas as pd
import pytest
from equity_research.models.ranking_selection import ranking_labels,ndcg_top


def test_labels_queries_ties_and_future_isolation():
    f=pd.DataFrame({'signal_date':pd.to_datetime(['2018-01-02']*10),'y_excess_5d':np.arange(10)})
    y,sizes=ranking_labels(f)
    np.testing.assert_array_equal(y,np.arange(10));np.testing.assert_array_equal(sizes,[10])
    tied=f.assign(y_excess_5d=1.)
    y,_=ranking_labels(tied);assert y.nunique()==1
    future=f.assign(signal_date=pd.Timestamp('2021-01-02'),y_excess_5d=1e6)
    combined=pd.concat([f,future],ignore_index=True)
    y,sizes=ranking_labels(combined)
    np.testing.assert_array_equal(y.iloc[:10],np.arange(10));np.testing.assert_array_equal(sizes,[10,10])


def test_labels_refuse_unsorted_queries_and_nonfinite_targets():
    f=pd.DataFrame({'signal_date':pd.to_datetime(['2020-01-03','2020-01-02']),'y_excess_5d':[1.,2.]})
    with pytest.raises(ValueError,match='sorted'):ranking_labels(f)
    f=f.sort_values('signal_date');f.iloc[0,1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):ranking_labels(f)


def test_ndcg_cutoff_and_tie_break():
    y=np.arange(10);ids=np.arange(10)
    assert ndcg_top(y,y,ids)==1
    assert ndcg_top(y,-y,ids)==0
    assert ndcg_top(y,np.zeros(10),ids)==0
    assert ndcg_top(y,np.zeros(10),ids[::-1])==1
