import sys
from pathlib import Path
import numpy as np
import pandas as pd
from test_action_portfolios import market
from equity_research.backtest.engine import simulate, verify
from reference.preflight import simulate as frozen_simulate, verify as reference_verify
import pytest

def test_unchanged_normal_trading():
    m=market(); targets={m['dates'][1]:(1,),m['dates'][3]:(2,)}
    d,o,h,e=simulate(m,targets,{},10,2,False,True)
    ref=frozen_simulate(m,targets,{},10,2,False,True)
    pd.testing.assert_frame_equal(d[ref[0].columns],ref[0])
    for a,b in zip([o,h,e],ref[1:]):pd.testing.assert_frame_equal(a,b)
    verify(d,o,h)
    reference_verify(d,o,h)

def event_market():
    m=market();m['ordinary'][2,0]=False;m['terminal'][2,0]=True
    m['buyable'][2:,0]=False;m['sellable'][2:,0]=False
    return m

def test_otc_quote_is_not_backdated_or_spendable():
    m=event_market();t={1:dict(event_date=str(m['dates'][2].date()),
       election_policy='first_post_delisting_quote_forced_sale_proxy',
       otc_first_observation_date=str(m['dates'][5].date()),otc_first_observation_price=.4)}
    d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][5]:(2,)},t,10,2,False,True)
    assert d.iloc[3].buy==0 and d.iloc[4].otc_frozen_value>0
    assert d.iloc[4].nav==d.iloc[1].nav
    sale=o[o.origin.eq('otc_quote_proxy')].iloc[0]
    assert sale.date==m['dates'][5]
    np.testing.assert_allclose(sale.amount, (1/1.001)/10*.4)
    assert sale.cost>0 and d.iloc[5].buy>0
    verify(d,o,h)
    reference_verify(d,o,h)

def test_cvr_entitlement_survives_without_buying_power():
    m=event_market();t={1:dict(event_date=str(m['dates'][2].date()),cash_per_share=8.,
        special_per_share=0.,special_expected_date=None,ratio=0.,successor_permno=0,cvr_max_per_share=3.)}
    d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][4]:(2,)},t,0,2,False,True)
    assert d.iloc[2].nav==.8 and d.iloc[3].buy==0
    np.testing.assert_allclose(d.iloc[-1].cvr_contractual_maximum,.3)
    assert h[h.kind.eq('cvr_excluded')].value.eq(0).all()
    verify(d,o,h)
    reference_verify(d,o,h)

def test_grouped_verifier_detects_corrupt_orders():
    m=market();d,o,h,_=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,)},{},10,2,False,True)
    o.loc[0,'amount']+=.01
    with pytest.raises(AssertionError):verify(d,o,h)
    with pytest.raises(AssertionError):reference_verify(d,o,h)
