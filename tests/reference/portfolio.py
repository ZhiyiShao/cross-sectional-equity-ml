"""Frozen validation prediction portfolio simulation. Never fit or read test observations."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from equity_research.backtest.market import ROOT, NAMES, require, sha, performance, make_targets, read_predictions
from equity_research.data.universe import regular_trade, filter_codes, SINGLE_DAY
from equity_research.data.features import return_segments

CONFIG = ROOT/'config/full_selection_backtest.json'
OUT = ROOT/'outputs/full_selection_backtest_v1'
MODELS = ['raw__ridge_03','cross_sectional_rank__ridge_13','raw__lightgbm_17','cross_sectional_rank__lightgbm_10','momentum20','equal_weight']
NAMES = dict(zip(MODELS, ['原始 Ridge','排名 Ridge','原始 LightGBM','排名 LightGBM','20日动量','股票池等权']))


def execute_rebalance(values,cash,claims,target_ids,buyable,sellable,cost):
    """Self-financing close rebalance. Failed-buy budgets are never redistributed."""
    before=float(cash+values.sum()+claims.sum())
    desired=np.zeros_like(values)
    if len(target_ids):desired[np.asarray(target_ids,int)]=before/len(target_ids)
    sell_wish=np.maximum(values-desired,0)
    sells=sell_wish*sellable
    positions=values-sells
    cash_after_sells=cash+sells.sum()*(1-cost)
    buy_wish=np.maximum(desired-positions,0)
    total=buy_wish.sum()
    scale=min(1.,max(0.,cash_after_sells)/(total*(1+cost))) if total>0 else 0.
    allocated=buy_wish*scale
    buys=allocated*buyable
    positions+=buys
    cash_final=float(cash_after_sells-buys.sum()*(1+cost))
    fees=float((buys.sum()+sells.sum())*cost)
    require(cash_final>=-1e-10 and (positions>=-1e-10).all(),'Leverage or short position')
    np.testing.assert_allclose(cash_final+positions.sum()+claims.sum(),before-fees,atol=1e-10,rtol=1e-10)
    return positions,max(0.,cash_final),dict(buys=buys,sells=sells,fees=fees,
        failed_buy=allocated*(~buyable),failed_sell=sell_wish*(~sellable),
        turnover=.5*(buys.sum()+sells.sum())/before if before else 0.,pretrade_nav=before)


def simulate(m,targets,cost_bps,release_lag,record=False):
    dates,ids=m['dates'],m['ids'];n=len(ids);cost=cost_bps/10000
    v=np.zeros(n);claims=np.zeros(n);cash=1.;last_mark=np.full(n,-1,int)
    releases=np.full(n,len(dates)+1,int)
    lookup={p:j for j,p in enumerate(ids)}
    indexed={date:np.array([lookup[p] for p in members],dtype=int) for date,members in targets.items()}
    daily=[];orders=[];holdings=[];events=[];last_nav=1.
    for i,date in enumerate(dates):
        before_assets=float(v.sum()+claims.sum()+cash)
        held=v>1e-14
        ordinary=held&m['ordinary'][i]
        if ordinary.any():
            require(i>0 and np.all(last_mark[ordinary]==i-1) and m['valid_price'][i-1,ordinary].all(), 'Single-day return crosses unresolved valuation')
            v[ordinary]*=1+m['ret'][i,ordinary];last_mark[ordinary]=i
        terminal=held&m['terminal'][i]
        if terminal.any():
            require(i>0 and np.all(last_mark[terminal]==i-1),'Delisting after unresolved value')
            for j in np.flatnonzero(terminal):
                amount=v[j]*(1+m['ret'][i,j]);claims[j]+=amount;v[j]=0
                releases[j]=i+release_lag if release_lag is not None else len(dates)+1
                events.append(dict(date=date,PERMNO=int(ids[j]),kind='delist_frozen_receivable',amount=float(amount)))
                last_mark[j]=i
        multi=held&(m['span_start'][i]>=0)&~ordinary&~terminal
        if multi.any():
            require(np.all(last_mark[multi]==m['span_start'][i,multi]),'Multi-period return starts outside unchanged holding')
            v[multi]*=1+m['ret'][i,multi];last_mark[multi]=i
            for j in np.flatnonzero(multi):events.append(dict(date=date,PERMNO=int(ids[j]),kind='verified_span_resolved',amount=float(v[j])))
        unknown=held&~ordinary&~terminal&~multi
        stale=unknown&(m['flag'][i]=='MP')
        if (unknown&~stale).any():
            raise ValueError(f'Unresolved held return on {date.date()}: {int((unknown&~stale).sum())} positions; no performance may be published')
        for j in np.flatnonzero(stale):events.append(dict(date=date,PERMNO=int(ids[j]),kind='stale_MP_estimate',amount=float(v[j])))
        release=(claims>0)&(releases<=i)
        for j in np.flatnonzero(release):
            events.append(dict(date=date,PERMNO=int(ids[j]),kind='assumed_cash_release',amount=float(claims[j])))
        released=float(claims[release].sum());cash+=released;claims[release]=0
        pnl=float(v.sum()+claims.sum()+cash-before_assets)
        trade_date=date in indexed
        liquidation=i==len(dates)-1
        pretrade=float(v.sum()+claims.sum()+cash)
        buy=sell=fees=turnover=failed_buy=failed_sell=0.
        if trade_date or liquidation:
            wanted=np.array([],int) if liquidation else indexed[date]
            buyable=m['buyable'][i]
            sellable=m['sellable'][i]&(last_mark==i)
            v,cash,trade=execute_rebalance(v,cash,claims,wanted,buyable,sellable,cost)
            buy=float(trade['buys'].sum());sell=float(trade['sells'].sum());fees=trade['fees'];turnover=trade['turnover']
            failed_buy=float(trade['failed_buy'].sum());failed_sell=float(trade['failed_sell'].sum())
            last_mark[trade['buys']>0]=i
            for side in ['failed_buy','failed_sell']:
                for j in np.flatnonzero(trade[side]>1e-12):
                    events.append(dict(date=date,PERMNO=int(ids[j]),kind=side,amount=float(trade[side][j])))
            if record:
                for side in ['buys','sells']:
                    for j in np.flatnonzero(trade[side]>1e-12):
                        amount=float(trade[side][j]);orders.append(dict(date=date,PERMNO=int(ids[j]),side=side,amount=amount,cost=amount*cost,liquidation=liquidation))
        nav=float(cash+v.sum()+claims.sum())
        require(nav>0 and cash>=-1e-10 and np.all(v>=-1e-10) and np.all(claims>=-1e-10),'Invalid balance sheet')
        np.testing.assert_allclose(nav,last_nav+pnl-fees,atol=1e-10,rtol=1e-10)
        daily.append(dict(date=date,nav=nav,return_net=nav/last_nav-1,pnl=pnl,fees=fees,cash=cash,
            equity=float(v.sum()),receivable=float(claims.sum()),released_cash=released,buy=buy,sell=sell,
            failed_buy=failed_buy,failed_sell=failed_sell,pretrade_nav=pretrade,turnover=turnover,
            rebalance=trade_date and not liquidation,liquidation=liquidation,cash_weight=cash/nav,
            receivable_weight=float(claims.sum()/nav),stale_positions=int(stale.sum()),stale_value=float(v[stale].sum()),
            valuation_estimated=bool(stale.any()),position_count=int((v>1e-12).sum())))
        if record:
            for kind,values in [('equity',v),('receivable',claims)]:
                for j in np.flatnonzero(values>1e-12):holdings.append(dict(date=date,PERMNO=int(ids[j]),kind=kind,value=float(values[j])))
        last_nav=nav
    require(not ((v>1e-12)&(last_mark<len(dates)-1)).any(),'Unresolved ending valuation')
    return pd.DataFrame(daily),pd.DataFrame(orders),pd.DataFrame(holdings),pd.DataFrame(events,columns=['date','PERMNO','kind','amount'])


def verify_primary(daily,orders,holdings):
    """Reconstruct balance sheets, cash and metrics from independently aggregated ledgers."""
    assets=holdings.groupby('date').value.sum().reindex(daily.date,fill_value=0).to_numpy() if len(holdings) else np.zeros(len(daily))
    np.testing.assert_allclose(daily.nav,daily.cash+assets,atol=1e-10)
    trades=orders.groupby(['date','side']).amount.sum().unstack(fill_value=0).reindex(daily.date,fill_value=0)
    fees=orders.groupby('date').cost.sum().reindex(daily.date,fill_value=0).to_numpy()
    buys=trades.get('buys',pd.Series(0,index=trades.index)).to_numpy();sells=trades.get('sells',pd.Series(0,index=trades.index)).to_numpy()
    reconstructed_cash=1+np.cumsum(sells-buys-fees+daily.released_cash.to_numpy())
    np.testing.assert_allclose(daily.cash,reconstructed_cash,atol=1e-10)
    np.testing.assert_allclose(daily.fees,fees,atol=1e-10)
    np.testing.assert_allclose(daily.nav.iloc[1:].to_numpy()/daily.nav.iloc[:-1].to_numpy()-1,daily.return_net.iloc[1:],atol=1e-12)
    stats=performance(daily)
    np.testing.assert_allclose(stats['total_return'],daily.nav.iloc[-1]/daily.nav.iloc[0]-1,atol=1e-10)
    np.testing.assert_allclose(stats['max_drawdown'],(daily.nav/daily.nav.cummax()-1).min(),atol=1e-10)
    return len(daily)
