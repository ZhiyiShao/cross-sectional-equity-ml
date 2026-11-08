import numpy as np
import pandas as pd
import pytest
from test_action_portfolios import market
from equity_research.backtest.engine import simulate, verify
from equity_research.backtest.risk_comparison import optimize


def test_volatility_source_is_already_annualized(tmp_path,monkeypatch):
    from equity_research.backtest import risk_comparison as module
    source=tmp_path/'data/interim/crsp_audit_v1';source.mkdir(parents=True)
    (source/'ingest.json').write_text('{"parts":["part.parquet"]}')
    date=pd.Timestamp('2020-01-02')
    pred=pd.DataFrame({'PERMNO':[1],'signal_date':[date],'signal_eligible':[True]})
    features=pred[['PERMNO','signal_date']].assign(market_beta_60=1.2,volatility_20=.4)
    industry=pd.DataFrame({'PERMNO':[1],'DlyCalDt':[date],'SICCD':[2000],
        'SecInfoStartDt':[pd.Timestamp('2010-01-01')],'SecInfoEndDt':[pd.Timestamp('2030-01-01')]})
    monkeypatch.setattr(module,'ROOT',tmp_path)
    def read(path,**kwargs):
        assert kwargs['filters'][0][2]==[date]
        return features.copy() if path.name=='features_only.parquet' else industry.copy()
    monkeypatch.setattr(pd,'read_parquet',read)
    result=module.risk_inputs(pred,pd.DataFrame({'signal_date':[date]}))
    assert result.annual_vol.iloc[0]==.4


def test_weighted_execution_and_funding():
    m=market()
    d,o,h,_=simulate(m,{m['dates'][1]:{1:.7,2:.3}}, {},10,2,False,True)
    first=h[h.date.eq(m['dates'][1]) & h.kind.eq('equity')].set_index('PERMNO').value
    np.testing.assert_allclose(first.loc[1]/first.loc[2],7/3)
    verify(d,o,h)


def test_equal_weights_match_legacy_and_invalid_weights_fail():
    m=market(); date=m['dates'][1]
    a=simulate(m,{date:(1,2)},{},10,2,False,True)
    b=simulate(m,{date:{1:.5,2:.5}},{},10,2,False,True)
    for x,y in zip(a,b):pd.testing.assert_frame_equal(x,y)
    with pytest.raises(ValueError,match='Invalid target weights'):
        simulate(m,{date:{1:.8,2:.8}},{})


def test_optimizer_caps_industry_beta_and_volatility():
    n=100
    g=pd.DataFrame({'PERMNO':range(n),'industry':['A']*50+['B']*50,
        'market_beta_60':np.linspace(.5,2.5,n),'annual_vol':np.linspace(.1,1,n),'score':range(n)})
    c=dict(max_weight=.02,industry_active=.02,beta_active=.1,annual_stock_vol_active_upper=.1)
    weights,_=optimize(g,'score',c); w=np.array(list(weights.values()))
    assert abs(w.sum()-1)<1e-8 and w.max()<=.02000001
    assert abs(w[:50].sum()-.5)<=.02000001
    assert abs(w@g.market_beta_60-g.market_beta_60.mean())<=.10000001
    assert w@g.annual_vol<=g.annual_vol.mean()+.10000001
    # Merely changing input row order must not change the solution.
    other,_=optimize(g.sample(frac=1,random_state=7),'score',c)
    np.testing.assert_allclose(list(weights.values()),list(other.values()))


def test_optimizer_refuses_infeasible_cap_and_all_missing_risk():
    g=pd.DataFrame({'PERMNO':[1,2],'industry':['A','B'],'market_beta_60':[np.nan,np.nan],
                    'annual_vol':[.2,.3],'score':[0.,1.]})
    c=dict(max_weight=.02,industry_active=.02,beta_active=.1,annual_stock_vol_active_upper=.1)
    with pytest.raises(ValueError,match='too small'):optimize(g,'score',c)
    c['max_weight']=1
    with pytest.raises(ValueError,match='Entirely missing'):optimize(g,'score',c)
