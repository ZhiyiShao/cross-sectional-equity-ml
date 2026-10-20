import sys
from pathlib import Path
import numpy as np
import pandas as pd
from reference.corporate_actions import simulate,verify

def market():
 dates=pd.bdate_range('2020-01-01',periods=9);shape=(9,2)
 return dict(dates=dates,ids=np.array([1,2]),ret=np.zeros(shape),ordinary=np.ones(shape,bool),terminal=np.zeros(shape,bool),buyable=np.ones(shape,bool),sellable=np.ones(shape,bool),flag=np.full(shape,'NA',object),span_start=np.full(shape,-1),price=np.ones(shape)*10)

def test_settled_cash_cannot_fund_same_day_replacement():
 m=market();d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][6]:(2,)},{},0,2,2,True)
 assert d.iloc[3].buy==0 and d.iloc[3].sale_receivable==1
 assert d.iloc[5].cash==1 and d.iloc[6].buy==1
 verify(d,o,h,2)

def test_conversion_marks_once_and_release_is_not_profit():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True
 m['ret'][2,0]=-.9;m['ret'][2,1]=.5;m['price'][2,1]=20
 terms={1:dict(event_date=str(date.date()),cash_per_share=2,special_per_share=1,special_expected_date=str(m['dates'][5].date()),ratio=.5,successor_permno=2)}
 d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][6]:(2,)},terms,0,2,2,True)
 assert np.isclose(d.iloc[2]['nav'],1.3) # .1 old shares => 1 stock + .3 claims
 assert np.isclose(d.iloc[3].inherited,1) # unavailable lot cannot sell
 assert np.isclose(d.iloc[4].pnl,0) and np.isclose(d.iloc[5].pnl,0)
 assert np.isclose(o[o.origin.eq('inherited')].amount.sum(),1)
 verify(d,o,h,2)

def test_inherited_sale_failure_retains_lot_and_final_claim():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True;m['sellable'][4,1]=False
 t={1:dict(event_date=str(date.date()),cash_per_share=0,special_per_share=0,special_expected_date=None,ratio=1,successor_permno=2)}
 d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][4]:()},t,10,2,2,True)
 assert d.iloc[4].inherited>0 and d.iloc[-1].inherited==0
 assert d.iloc[-1].sale_receivable>0
 verify(d,o,h,2)
