import sys
from pathlib import Path
import pandas as pd
import numpy as np
from equity_research.backtest.validation import simulate
from equity_research.evaluation.exposures import security_ledger,attach_features,style_snapshots,FEATURES
from test_action_portfolios import market

def test_swap_pnl_original_then_successor_and_no_settlement_profit():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True
 m['price'][2,1]=20;m['ret'][2,1]=.5;m['ret'][3,1]=.1
 terms={1:dict(event_date=str(date.date()),cash_per_share=2,special_per_share=1,special_expected_date=None,ratio=.5,successor_permno=2,previous_price=10)}
 d,t,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][4]:(2,)},terms,0,2,False,True)
 for f in [d,t,h,e]:f['model']='m'
 d['prev_nav']=d.nav.shift().fillna(1.)
 p=security_ledger(d,h,t,m['dates'],e,pd.DataFrame.from_dict(terms,orient='index'))
 np.testing.assert_allclose(p.loc[p.date.eq(date)&p.PERMNO.eq(1),'pnl'].sum(),.3)
 np.testing.assert_allclose(p.loc[p.date.eq(date)&p.PERMNO.eq(2),'pnl'].sum(),0)
 np.testing.assert_allclose(p.loc[p.date.eq(m['dates'][3])&p.PERMNO.eq(2),'pnl'].sum(),.1)
 assert p.loc[p.date.eq(m['dates'][3])&p.PERMNO.eq(2),'prev_equity'].iloc[0]==1
 np.testing.assert_allclose(p.loc[p.date.eq(m['dates'][4]),'pnl'].sum(),0,atol=1e-12)

def test_cash_acquisition_release_not_new_profit():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True
 terms={1:dict(event_date=str(date.date()),cash_per_share=11,special_per_share=0,special_expected_date=None,ratio=0,successor_permno=0,previous_price=10)}
 d,t,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][6]:(2,)},terms,10,2,False,True)
 for f in [d,t,h,e]:f['model']='m'
 d['prev_nav']=d.nav.shift().fillna(1.)
 p=security_ledger(d,h,t,m['dates'],e,pd.DataFrame.from_dict(terms,orient='index'))
 assert not p.loc[p.date.eq(m['dates'][4]),'pnl'].sum()
 np.testing.assert_allclose(p.net_pnl.sum(),d.nav.iloc[-1]-1,atol=1e-10)

def test_unknown_successor_features_not_filled_from_future():
 f=pd.DataFrame({'PERMNO':[1,2],'signal_date':pd.to_datetime(['2020-01-02','2020-01-09']),'signal_eligible':[True,True]})
 for c in FEATURES:f[c]=[1.,999.]
 keys=pd.DataFrame({'PERMNO':[2,1],'asof_date':pd.to_datetime(['2020-01-08','2020-02-01'])})
 x=attach_features(keys,style_snapshots(f))
 assert x[FEATURES].isna().all().all()
