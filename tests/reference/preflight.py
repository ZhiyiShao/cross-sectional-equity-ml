"""Validation portfolios with explicit trade receivables/payables and zero financing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import equity_research.backtest.market as base
from equity_research.backtest.calendar import settlement_date, CONFIG as CALENDAR_CONFIG
from reference.ledger import Ledger
ROOT=base.ROOT
OUT=ROOT/'outputs/test_execution_preflight_v1'
CONFIG=ROOT/'config/validation_corporate_actions.json'


def simulate(m,targets,terms,cost_bps=10,availability=2,settled_only=False,record=False):
 dates,ids=m['dates'],m['ids'];n=len(ids);cost=cost_bps/10000
 lookup={int(p):j for j,p in enumerate(ids)}
 indexed={d:np.array([lookup[p] for p in pp],int) for d,pp in targets.items()}
 v=np.zeros(n);last=np.full(n,-1,int);ledger=Ledger();last_nav=1.
 # Each inherited security lot is separate from model-selected holdings.
 lots=[];claims=[];daily=[];orders=[];holdings=[];events=[]
 def total():return float(v.sum()+sum(x['value'] for x in lots)+sum(x['value'] for x in claims)+ledger.cash+sum(x['value'] for x in ledger.entries))
 def claim(value,due,kind,owner):
  if value>0:claims.append(dict(value=float(value),due=int(due),kind=kind,owner=int(owner)))
 for i,date in enumerate(dates):
  before=total();held=v>1e-14;fees=0.;buy=0.;sell=0.;released=0.;failed_buy=0.;failed_sell=0.
  ordinary=held&m['ordinary'][i]
  if ordinary.any():
   base.require(np.all(last[ordinary]==i-1),'Unresolved ordinary mark')
   v[ordinary]*=1+m['ret'][i,ordinary];last[ordinary]=i
  multi=held&(m['span_start'][i]>=0)&~ordinary&~m['terminal'][i]
  if multi.any():
   base.require(np.all(last[multi]==m['span_start'][i,multi]),'Unresolved span')
   v[multi]*=1+m['ret'][i,multi];last[multi]=i
  # Existing inherited lots accrue returns before new conversion lots are created.
  for lot in lots:
   j=lot['j']
   base.require(m['ordinary'][i,j] and lot['last']==i-1,'Unresolved successor return or chained action')
   lot['value']*=1+m['ret'][i,j];lot['last']=i
  terminal=held&m['terminal'][i]
  for j in np.flatnonzero(terminal):
   p=int(ids[j]);base.require(p in terms,'Unregistered held delisting '+str(p));t=terms[p]
   base.require(str(date.date())==t['event_date'] and last[j]==i-1,'Wrong event date')
   base.require(m['price'][i-1,j]>0,'No prior price')
   qty=v[j]/m['price'][i-1,j];old=v[j];v[j]=0.;last[j]=i
   due=i+availability
   claim(qty*t['cash_per_share'],due,'action_cash',p)
   special_due=due
   if t['special_expected_date']:special_due=max(due,int(dates.searchsorted(pd.Timestamp(t['special_expected_date']))))
   claim(qty*t['special_per_share'],special_due,'special_cash',p)
   stock=0.
   if t['ratio']:
    sj=lookup[t['successor_permno']];price=m['price'][i,sj]
    base.require(np.isfinite(price) and price>0,'No conversion price')
    stock=qty*t['ratio']*price
    lots.append(dict(j=sj,value=stock,due=due,last=i,owner=p))
   events.append(dict(date=date,PERMNO=p,kind='conversion',old_value=old,
    value=stock+qty*(t['cash_per_share']+t['special_per_share']),due_index=due))
  unknown=held&~ordinary&~multi&~terminal
  stale=unknown&(m['flag'][i]=='MP')
  base.require(not (unknown&~stale).any(),'Unresolved ordinary holding')
  pending=[]
  for c in claims:
   if c['due']<=i:ledger.cash+=c['value'];released+=c['value']
   else:pending.append(c)
  claims=pending
  trade_settlement=ledger.settle(date)
  trade_due=settlement_date(date)
  pnl=total()-before
  rebalance=date in indexed;liquidation=i==len(dates)-1;pretrade=total()
  if rebalance or liquidation:
   keep=[]
   for lot in lots:
    j=lot['j']
    if i>=lot['due'] and m['sellable'][i,j] and lot['last']==i:
     amount=lot['value'];fee=amount*cost;fees+=fee;sell+=amount
     ledger.post(date,trade_due,amount-fee,int(ids[j]))
     orders.append(dict(date=date,PERMNO=int(ids[j]),side='sells',amount=amount,cost=fee,origin='inherited',due_date=trade_due))
    else:keep.append(lot)
   lots=keep
   desired=np.zeros(n)
   wanted=np.array([],int) if liquidation else indexed[date]
   if len(wanted):desired[wanted]=pretrade/len(wanted)
   sell_wish=np.maximum(v-desired,0);sells=sell_wish*(m['sellable'][i]&(last==i))
   v-=sells;fee=float(sells.sum()*cost);fees+=fee;sell+=float(sells.sum())
   failed_sell=float((sell_wish-sells).sum())
   for j in np.flatnonzero(sells):
    ledger.post(date,trade_due,sells[j]*(1-cost),ids[j])
    orders.append(dict(date=date,PERMNO=int(ids[j]),side='sells',amount=sells[j],cost=sells[j]*cost,origin='strategy',due_date=trade_due))
   wish=np.maximum(desired-v,0);need=float(wish.sum())
   scale=min(1.,ledger.budget(trade_due,settled_only)/(need*(1+cost))) if need else 0.
   allocation=wish*scale;buys=allocation*m['buyable'][i]
   buy=float(buys.sum());fee=buy*cost;fees+=fee;v+=buys;last[buys>0]=i
   failed_buy=float((allocation-buys).sum())
   for j in np.flatnonzero(buys):
    ledger.post(date,trade_due,-buys[j]*(1+cost),ids[j])
    orders.append(dict(date=date,PERMNO=int(ids[j]),side='buys',amount=buys[j],cost=buys[j]*cost,origin='strategy',due_date=trade_due))
  nav=total();claim_value=sum(c['value'] for c in claims);inherited=sum(x['value'] for x in lots)
  ledger.check()
  sale_ar,buy_ap=ledger.balances()
  base.require(float(v.sum())+inherited<=nav+1e-10,'Equity leverage')
  base.require(ledger.cash>=-1e-10 and nav>0 and (v>=-1e-12).all(),'Invalid balance')
  np.testing.assert_allclose(nav,last_nav+pnl-fees,atol=1e-10)
  daily.append(dict(date=date,nav=nav,return_net=nav/last_nav-1,pnl=pnl,fees=fees,cash=ledger.cash,
    equity=float(v.sum())+inherited,inherited=inherited,receivable=claim_value+sale_ar,payable=buy_ap,
    min_projected_cash=min([ledger.cash]+[b for _,b in ledger.path()]),
    net_funding=ledger.cash+sale_ar-buy_ap,trade_settlement=trade_settlement,
    action_receivable=sum(c['value'] for c in claims if c['kind']!='sale_cash'),
    sale_receivable=sale_ar,
    released_cash=released,buy=buy,sell=sell,failed_buy=failed_buy,failed_sell=failed_sell,
    pretrade_nav=pretrade,turnover=(buy+sell)/2/pretrade,rebalance=rebalance and not liquidation,
    liquidation=liquidation,cash_weight=ledger.cash/nav,receivable_weight=(claim_value+sale_ar)/nav,payable_weight=buy_ap/nav,
    valuation_estimated=bool(stale.any())))
  if record:
   for j in np.flatnonzero(v>1e-12):holdings.append(dict(date=date,PERMNO=int(ids[j]),kind='equity',value=v[j]))
   for x in lots:holdings.append(dict(date=date,PERMNO=int(ids[x['j']]),kind='inherited',value=x['value']))
   for x in claims:holdings.append(dict(date=date,PERMNO=x['owner'],kind=x['kind'],value=x['value']))
   for x in ledger.entries:holdings.append(dict(date=date,PERMNO=x['PERMNO'],kind='trade_receivable' if x['value']>0 else 'trade_payable',value=x['value'],due_date=x['due_date']))
  last_nav=nav
 return pd.DataFrame(daily),pd.DataFrame(orders),pd.DataFrame(holdings),pd.DataFrame(events)


def verify(d,o,h):
 assets=h.groupby('date').value.sum().reindex(d.date,fill_value=0).to_numpy()
 np.testing.assert_allclose(d.nav,d.cash+assets,atol=1e-10)
 signed=np.where(o.side.eq('sells'),o.amount-o.cost,-o.amount-o.cost)
 flows=pd.DataFrame(dict(due=o.due_date,amount=signed)).groupby('due').amount.sum()
 expected_cash=1.;prior=None
 for row in d.itertuples():
  due=flows.loc[(flows.index<=row.date)&((flows.index>prior) if prior is not None else True)].sum()
  expected_cash+=float(due)+row.released_cash
  np.testing.assert_allclose(row.cash,expected_cash,atol=1e-10)
  outstanding=(o.date<=row.date)&(o.due_date>row.date)
  ar=float(signed[outstanding & o.side.eq('sells')].sum())
  ap=-float(signed[outstanding & o.side.eq('buys')].sum())
  np.testing.assert_allclose([row.sale_receivable,row.payable],[ar,ap],atol=1e-10)
  # Independent future-date solvency check from executed orders, excluding action receivables.
  balance=row.cash
  for due_date in sorted(o.loc[outstanding,'due_date'].unique()):
   balance+=float(signed[outstanding & o.due_date.eq(due_date)].sum())
   base.require(balance>=-1e-10,'Independent future funding shortfall')
  prior=row.date
 np.testing.assert_allclose(d.nav,d.cash+d.equity+d.receivable-d.payable,atol=1e-10)
 np.testing.assert_allclose(d.fees,o.groupby('date').cost.sum().reindex(d.date,fill_value=0),atol=1e-10)
 base.require(d.equity.le(d.nav+1e-10).all(),'Independent equity leverage')
 for r in o.itertuples():base.require(r.due_date==settlement_date(r.date),'Order settlement date mismatch')
 return len(d)
