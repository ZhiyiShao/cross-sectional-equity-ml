import numpy as np
import pandas as pd
from equity_research.models.context_selection import contexts,period_mask
from equity_research.models.preprocessing import fit_preprocessing,transform_features


def example():
    return pd.DataFrame({'signal_date':pd.to_datetime(['2019-01-02']*6),'PERMNO':range(6),
        'signal_eligible':[True]*5+[False],'industry':['A','A','B','B','B','A'],
        'momentum_20':[.1,.3,-.1,.2,.5,100.], 'momentum_60':[.2,.4,-.2,.3,.6,100.],
        'volatility_20':[.2,.4,.3,.5,.6,100.],'reversal_1':[.01]*6})


def test_contexts_eligible_only_and_no_future_dependence():
    f=example();a=contexts(f,2)
    np.testing.assert_allclose(a.industry_mom20.iloc[:2],[.2,.2])
    assert np.isclose(a.universe_breadth20.iloc[0],.8)
    assert a.universe_median_vol20.iloc[0]==.4
    np.testing.assert_allclose(a.industry_rank20.iloc[:2],[-.25,.25])
    assert np.isclose(a.momentum_vol_interaction.iloc[0],.04)
    future=f.assign(signal_date=pd.Timestamp('2025-01-01'),momentum_20=999.)
    b=contexts(pd.concat([f,future],ignore_index=True),2).iloc[:len(f)]
    pd.testing.assert_frame_equal(a,b)
    changed=f.assign(y_excess_5d=np.arange(6)*1e6)
    pd.testing.assert_frame_equal(a,contexts(changed,2).drop(columns='y_excess_5d'))


def test_small_industry_fallback_and_finite_population():
    f=example();f.loc[4,'momentum_20']=np.nan
    a=contexts(f,5)
    np.testing.assert_allclose(a.industry_mom20,.125)
    assert a.universe_breadth20.iloc[0]==.75


def test_labels_purged_at_each_window_boundary():
    f=pd.DataFrame({'signal_date':pd.to_datetime(['2018-12-20','2018-12-28','2019-01-02']),
        'label_end_date':pd.to_datetime(['2018-12-28','2019-01-07','2019-01-10']),
        'signal_eligible':True,'label_status':'ok','y_excess_5d':.1})
    assert period_mask(f,['2015-01-01','2018-12-31']).tolist()==[True,False,False]


def test_market_context_survives_training_transform():
    x=pd.DataFrame({'context':[.2,.2,.4,.4],'interaction':[.01,-.02,.03,.05]})
    state=fit_preprocessing(x,[.005,.995]);before=dict(state)
    result=transform_features(x,state)
    assert result.context.iloc[0]!=result.context.iloc[2]
    transform_features(pd.DataFrame({'context':[100.],'interaction':[999.]}),state)
    assert state==before
