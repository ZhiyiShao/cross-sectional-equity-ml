import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from equity_research.backtest.validation_calendar import Ledger,settlement_date,is_settlement_day
from equity_research.backtest.validation import simulate,verify
from test_action_portfolios import market

@pytest.mark.parametrize('trade,due',[
 ('2020-10-08','2020-10-13'),('2020-10-09','2020-10-14'),('2020-10-12','2020-10-14'),
 ('2020-11-10','2020-11-13'),('2021-03-31','2021-04-05'),('2021-04-01','2021-04-06'),
 ('2021-06-17','2021-06-21'),('2021-12-23','2021-12-28'),('2021-12-30','2022-01-03'),
 ('2017-09-01','2017-09-07'),('2017-09-05','2017-09-07'),
 ('2024-05-23','2024-05-28'),('2024-05-24','2024-05-29'),('2024-05-28','2024-05-29')])
def test_historical_settlement(trade,due):
 assert settlement_date(trade)==pd.Timestamp(due)

def test_unverified_calendar_fails_closed():
 with pytest.raises(ValueError):settlement_date('2023-01-03')
 assert is_settlement_day('2021-06-18')
 assert not is_settlement_day('2021-11-11')

def test_no_double_spend_and_mismatched_dates():
 l=Ledger(0);date=pd.Timestamp('2020-01-06');early=pd.Timestamp('2020-01-07');due=pd.Timestamp('2020-01-08')
 l.post(date,due,100,1)
 assert l.budget(early)==0 and l.budget(due)==100
 assert l.budget(due,True)==0
 l.post(date,due,-70,2)
 assert l.budget(due)==30
 l.post(date,due,-30,3);l.check()
 assert l.budget(due)==0
 assert l.settle(due)==0 and l.cash==0

def test_future_obligation_reserves_cash():
 l=Ledger(100);date=pd.Timestamp('2020-01-06');due=pd.Timestamp('2020-01-08')
 l.post(date,pd.Timestamp('2020-01-09'),-80,2)
 assert l.budget(due)==20

def test_same_day_replacement_with_costs_and_end_receivable():
 m=market();targets={m['dates'][1]:(1,),m['dates'][3]:(2,)}
 d,o,h,_=simulate(m,targets,{},10,2,False,True)
 assert d.iloc[3].buy>0 and d.iloc[3].payable>0 and d.iloc[3].sale_receivable>0
 np.testing.assert_allclose(d.iloc[3].buy*(1.001),d.iloc[3].sell*.999,atol=1e-10)
 assert d.iloc[-1].sale_receivable>0 and d.iloc[-1].equity==0
 verify(d,o,h)
 restricted,*_=simulate(m,targets,{},10,2,True)
 assert restricted.iloc[3].buy==0

def test_failed_sale_cannot_fund_buy():
 m=market();m['sellable'][3,0]=False
 d,*_=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,)},{},0)
 assert d.iloc[3].buy==0 and d.iloc[3].failed_sell==1

def test_unavailable_action_cash_cannot_fund_buy():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True
 t={1:dict(event_date=str(date.date()),cash_per_share=10,special_per_share=0,special_expected_date=None,ratio=0,successor_permno=0)}
 d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][6]:(2,)},t,0,2,False,True)
 assert d.iloc[3].buy==0 and d.iloc[3].action_receivable==1
 assert d.iloc[4].cash==1 and d.iloc[4].buy==0 and d.iloc[6].buy==1
 verify(d,o,h)

def test_inherited_sale_funds_same_rebalance_and_marks_once():
 m=market();date=m['dates'][2];m['ordinary'][2,0]=False;m['terminal'][2,0]=True
 m['ret'][2,0]=-.9;m['ret'][2,1]=.5;m['price'][2,1]=20
 t={1:dict(event_date=str(date.date()),cash_per_share=0,special_per_share=0,special_expected_date=None,ratio=.5,successor_permno=2)}
 d,o,h,e=simulate(m,{m['dates'][1]:(1,),m['dates'][3]:(2,),m['dates'][4]:(2,)},t,0,2,False,True)
 assert d.iloc[2]['nav']==1 and d.iloc[3].buy==0
 assert d.iloc[4].buy==1 and d.iloc[4].inherited==0
 assert o[o.origin.eq('inherited')].due_date.iloc[0]==settlement_date(m['dates'][4])
 verify(d,o,h)

def test_unsettled_liability_survives_early_final_sale():
 m=market();m={k:(v[:4] if k!='ids' else v) for k,v in m.items()}
 d,o,h,e=simulate(m,{m['dates'][2]:(1,)},{},10,2,False,True)
 assert d.iloc[-1].payable>0 and d.iloc[-1].sale_receivable>0
 verify(d,o,h)
