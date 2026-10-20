"""Synthetic regression checks for financially consequential data handling."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from equity_research.data.ingest import ROOT, deduplicate
from equity_research.data.universe import monthly_snapshots, select_month, regular_trade, semantic_audit
from collections import Counter


def config():
    cfg = json.loads((ROOT / 'config/universe.json').read_text(encoding='utf-8'))
    cfg.update(min_observed_history_days=3, liquidity_window_trading_days=3, min_valid_liquidity_days=2, max_securities=2)
    return cfg


def history(dates):
    n = len(dates)
    return pd.DataFrame(dict(PERMNO=[1]*n, PERMCO=[10]*n, Ticker=['SYNTH']*n,
        DlyCalDt=dates, PrimaryExch=['N']*n, SecurityType=['EQTY']*n,
        SecuritySubType=['COM']*n, ShareType=['NS']*n, USIncFlg=['Y']*n,
        IssuerType=['CORP']*n, ConditionalType=['RW']*n,
        TradingStatusFlg=['A']*n, DlyDelFlg=['N']*n, DlyPrcFlg=['TR']*n,
        DlyClose=[10.]*n, DlyPrc=[10.]*n, DlyVol=[100.]*n, DlyCap=[1000.]*n))


def test_exact_duplicates_only_and_conflicts_detected():
    d = pd.DataFrame({'PERMNO':[1,1,1], 'DlyCalDt':['2020-01-02']*3, 'DlyRet':[.1,.1,.2]})
    clean, removed, conflicts = deduplicate(d)
    assert removed == 1 and len(clean) == 2 and len(conflicts) == 2


def test_literal_na_preserved_on_csv_read():
    from io import StringIO
    d = pd.read_csv(StringIO('DlyRetMissFlg,DlyRet\nNA,0.02\nMP,\n'),dtype=str,keep_default_na=False)
    assert d.DlyRetMissFlg.iloc[0] == 'NA'
    assert d.DlyRet.iloc[1] == ''


def test_rolling_uses_market_days_and_never_backfills():
    cal = pd.bdate_range('2020-01-01',periods=6)
    d = history(cal[[0,1,5]])
    row = monthly_snapshots(d,cal,[cal[-1]],config()).iloc[0]
    assert row.history_days == 3
    assert row.valid_liquidity_days == 1
    assert pd.isna(row.adv60_usd) and not row.eligible_liquidity


def test_future_prices_classification_and_volume_cannot_change_prior_snapshot():
    cal = pd.bdate_range('2020-01-01',periods=10)
    d = history(cal)
    original = monthly_snapshots(d,cal,[cal[4]],config())
    d.loc[5:,'DlyVol'] = 1e12
    d.loc[5:,'SecurityType'] = 'FUND'
    d.loc[5:,'DlyClose'] = .01
    changed = monthly_snapshots(d,cal,[cal[4]],config())
    pd.testing.assert_frame_equal(original,changed)


def test_stale_month_end_not_carried_forward():
    cal = pd.bdate_range('2020-01-01',periods=6)
    assert monthly_snapshots(history(cal[:-1]),cal,[cal[-1]],config()).empty


def test_bid_ask_and_delisting_are_not_tradable_closes():
    d = history(pd.bdate_range('2020-01-01',periods=3))
    d.loc[1,'DlyPrcFlg']='BA'
    d.loc[2,'DlyDelFlg']='Y'
    assert regular_trade(d,config()).tolist() == [True,False,False]


def test_one_share_per_company_then_global_rank_and_stable_tie():
    d = pd.DataFrame({'PERMNO':[3,2,1,4], 'PERMCO':[10,10,20,30], 'adv60_usd':[300.,400.,300.,200.]})
    for flag in ['security','trading','history','price','liquidity']:
        d['eligible_'+flag]=True
    selected, funnel = select_month(d,config())
    assert selected.PERMNO.tolist() == [2,1]
    assert funnel['unique_companies']==3


def test_split_adjustment_and_delist_event_preserved():
    cal=pd.DatetimeIndex(['2020-01-03','2020-01-06','2020-01-07'])
    d=history(cal)
    d['DlyPrc']=[100.,51.,25.5]
    d['DlyClose']=[100.,51.,np.nan]
    d['ShrOut']=[10.,20.,20.]
    d['DlyCap']=d.DlyPrc*d.ShrOut
    d['DlyHigh']=d.DlyClose
    d['DlyLow']=d.DlyClose
    d['DlyCumFacPr']=[2.,1.,1.]
    d['DlyCumFacShr']=[2.,1.,1.]
    d['DlyFacPrc']=[1.,2.,1.]
    d['DlyRet']=[np.nan,.02,-.5]
    d['DlyRetx']=d.DlyRet
    d['DlyRetDurFlg']=['MR','D3','DD']
    d['DlyRetMissFlg']=['NS','NA','NA']
    d.loc[2,'DlyDelFlg']='Y'
    stats=Counter()
    exceptions=semantic_audit(d,dict(zip(cal,range(3))),stats,{'cap_ratio':[]})
    assert stats['split_like_divide_matches_1e_5']==1
    assert stats['split_like_multiply_matches_1e_5']==0
    assert stats['delist_rows']==1
    assert exceptions.loc[exceptions.audit_delist,'DlyRet'].iloc[0]==-.5


def test_numeric_mv_return_is_quarantined_without_removing_universe_member():
    cal=pd.bdate_range('2020-01-01',periods=4)
    d=history(cal)
    for col in ['DlyCumFacPr','DlyCumFacShr','DlyFacPrc']:
        d[col]=1.
    d['ShrOut']=100.
    d['DlyHigh']=10.
    d['DlyLow']=10.
    d['DlyRet']=0.
    d['DlyRetx']=0.
    d['DlyRetDurFlg']='D1'
    d['DlyRetMissFlg']='NA'
    d.loc[3,'DlyRetMissFlg']='MV'
    stats=Counter()
    exceptions=semantic_audit(d,dict(zip(cal,range(4))),stats,{'cap_ratio':[]})
    assert stats['numeric_return_with_non_NA_flag']==1
    assert exceptions.loc[exceptions.DlyRetMissFlg.eq('MV'),'audit_return_gap'].all()
    snapshot=monthly_snapshots(d,cal,[cal[-1]],config())
    selected,_=select_month(snapshot,config())
    assert selected.PERMNO.tolist()==[1]
