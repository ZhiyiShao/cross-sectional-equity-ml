"""Validation-only portfolios with inherited lots and settled-cash constraints."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import equity_research.backtest.market as base
ROOT=base.ROOT
OUT=ROOT/'outputs/corporate_action_backtest_v1'
CONFIG=ROOT/'config/validation_corporate_actions.json'


def simulate(m,targets,terms,cost_bps=10,availability=2,settlement=2,record=False):
 dates,ids=m['dates'],m['ids'];n=len(ids);cost=cost_bps/10000
 lookup={int(p):j for j,p in enumerate(ids)}
 indexed={d:np.array([lookup[p] for p in pp],int) for d,pp in targets.items()}
 v=np.zeros(n);last=np.full(n,-1,int);cash=1.;last_nav=1.
 # Each inherited security lot is separate from model-selected holdings.
 lots=[];claims=[];daily=[];orders=[];holdings=[];events=[]
 def total():return float(v.sum()+sum(x['value'] for x in lots)+sum(x['value'] for x in claims)+cash)
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
   if c['due']<=i:cash+=c['value'];released+=c['value']
   else:pending.append(c)
  claims=pending
  pnl=total()-before
  rebalance=date in indexed;liquidation=i==len(dates)-1;pretrade=total()
  if rebalance or liquidation:
   keep=[]
   for lot in lots:
    j=lot['j']
    if i>=lot['due'] and m['sellable'][i,j] and lot['last']==i:
     amount=lot['value'];fee=amount*cost;fees+=fee;sell+=amount
     # Inherited sale follows T+2 in every scenario, including ordinary-trade ablation.
     claim(amount-fee,i+2,'sale_cash',int(ids[j]))
     orders.append(dict(date=date,PERMNO=int(ids[j]),side='sells',amount=amount,cost=fee,origin='inherited',due_index=i+2))
    else:keep.append(lot)
   lots=keep
   desired=np.zeros(n)
   wanted=np.array([],int) if liquidation else indexed[date]
   if len(wanted):desired[wanted]=pretrade/len(wanted)
   sell_wish=np.maximum(v-desired,0);sells=sell_wish*(m['sellable'][i]&(last==i))
   v-=sells;fee=float(sells.sum()*cost);fees+=fee;sell+=float(sells.sum())
   failed_sell=float((sell_wish-sells).sum())
   if settlement==0:cash+=float(sells.sum())-fee
   else:
    for j in np.flatnonzero(sells):claim(sells[j]*(1-cost),i+settlement,'sale_cash',ids[j])
   for j in np.flatnonzero(sells):orders.append(dict(date=date,PERMNO=int(ids[j]),side='sells',amount=sells[j],cost=sells[j]*cost,origin='strategy',due_index=i+settlement))
   wish=np.maximum(desired-v,0);need=float(wish.sum())
   scale=min(1.,max(0.,cash)/(need*(1+cost))) if need else 0.
   allocation=wish*scale;buys=allocation*m['buyable'][i]
   buy=float(buys.sum());fee=buy*cost;fees+=fee;cash-=buy+fee;v+=buys;last[buys>0]=i
   failed_buy=float((allocation-buys).sum())
   for j in np.flatnonzero(buys):orders.append(dict(date=date,PERMNO=int(ids[j]),side='buys',amount=buys[j],cost=buys[j]*cost,origin='strategy',due_index=i))
  nav=total();claim_value=sum(c['value'] for c in claims);inherited=sum(x['value'] for x in lots)
  base.require(cash>=-1e-10 and nav>0 and (v>=-1e-12).all(),'Invalid balance')
  np.testing.assert_allclose(nav,last_nav+pnl-fees,atol=1e-10)
  daily.append(dict(date=date,nav=nav,return_net=nav/last_nav-1,pnl=pnl,fees=fees,cash=cash,
    equity=float(v.sum())+inherited,inherited=inherited,receivable=claim_value,
    action_receivable=sum(c['value'] for c in claims if c['kind']!='sale_cash'),
    sale_receivable=sum(c['value'] for c in claims if c['kind']=='sale_cash'),
    released_cash=released,buy=buy,sell=sell,failed_buy=failed_buy,failed_sell=failed_sell,
    pretrade_nav=pretrade,turnover=(buy+sell)/2/pretrade,rebalance=rebalance and not liquidation,
    liquidation=liquidation,cash_weight=cash/nav,receivable_weight=claim_value/nav,
    valuation_estimated=bool(stale.any())))
  if record:
   for j in np.flatnonzero(v>1e-12):holdings.append(dict(date=date,PERMNO=int(ids[j]),kind='equity',value=v[j]))
   for x in lots:holdings.append(dict(date=date,PERMNO=int(ids[x['j']]),kind='inherited',value=x['value']))
   for x in claims:holdings.append(dict(date=date,PERMNO=x['owner'],kind=x['kind'],value=x['value']))
  last_nav=nav
 return pd.DataFrame(daily),pd.DataFrame(orders),pd.DataFrame(holdings),pd.DataFrame(events)


def verify(d,o,h,settlement):
 assets=h.groupby('date').value.sum().reindex(d.date,fill_value=0).to_numpy()
 np.testing.assert_allclose(d.nav,d.cash+assets,atol=1e-10)
 immediate=o.loc[o.side.eq('sells')&o.origin.eq('strategy')] if settlement==0 else o.iloc[:0]
 inflow=(immediate.amount-immediate.cost).groupby(immediate.date).sum().reindex(d.date,fill_value=0).to_numpy()
 bought=o.loc[o.side.eq('buys')];spent=(bought.amount+bought.cost).groupby(bought.date).sum().reindex(d.date,fill_value=0).to_numpy()
 np.testing.assert_allclose(d.cash,1+np.cumsum(d.released_cash.to_numpy()+inflow-spent),atol=1e-10)
 np.testing.assert_allclose(d.fees,o.groupby('date').cost.sum().reindex(d.date,fill_value=0),atol=1e-10)
 return len(d)
