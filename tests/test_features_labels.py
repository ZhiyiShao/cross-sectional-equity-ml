"""Temporal, missing-data and corporate-action regression tests (synthetic)."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from equity_research.data.ingest import ROOT
from equity_research.data.features import make_features, return_segments, label_one, split_masks, compound_past
from equity_research.data.universe import regular_trade


def configs():
    return (json.loads((ROOT/'config/universe.json').read_text(encoding='utf-8')),
            json.loads((ROOT/'config/features_labels.json').read_text(encoding='utf-8')))


def fixture(n=160):
    idx=pd.bdate_range('2018-01-01',periods=n)
    d=pd.DataFrame(index=idx)
    for k,v in dict(DlyDelFlg='N',DlyPrcFlg='TR',DlyRetMissFlg='NA',DlyRetDurFlg='D1',
        ConditionalType='RW',TradingStatusFlg='A',PrimaryExch='N',SecurityType='EQTY',
        SecuritySubType='COM',ShareType='NS',USIncFlg='Y',IssuerType='CORP').items():
        d[k]=v
    d['DlyRet']=np.sin(np.arange(n))*.01
    d['DlyPrc']=100*np.cumprod(1+d.DlyRet)
    d['DlyClose']=d.DlyPrc
    d['DlyHigh']=d.DlyClose*1.01; d['DlyLow']=d.DlyClose*.99
    d['DlyVol']=np.arange(n)+1000.
    d['ShrOut']=100.
    d['DlyCap']=d.DlyPrc*d.ShrOut
    d['DlyCumFacShr']=1.
    market=pd.Series(np.cos(np.arange(n))*.002,index=idx)
    return d,market


def target(d,market,t=0):
    trade=regular_trade(d,configs()[0]).to_numpy()
    return label_one(t,d,market,trade,trade,return_segments(d))


def test_label_uses_t_plus_2_through_t_plus_6_and_subtracts_market():
    d,m=fixture(12)
    d.loc[d.index[1],'DlyRet']=.9
    r=target(d,m)
    expected=np.prod(1+d.DlyRet.iloc[2:7])-1-(np.prod(1+m.iloc[2:7])-1)
    assert r['label_status']=='ok'
    assert np.isclose(r['y_excess_5d'],expected)
    assert r['entry_date']==d.index[1] and r['label_end_date']==d.index[6]


def gap(d,missing=3):
    d=d.copy()
    both=(1+d.DlyRet.iloc[missing])*(1+d.DlyRet.iloc[missing+1])-1
    for col in ['DlyPrc','DlyClose','DlyHigh','DlyLow']:
        d.loc[d.index[missing],col]=np.nan
    d.loc[d.index[missing],'DlyRet']=np.nan
    d.loc[d.index[missing],'DlyRetMissFlg']='MP'
    d.loc[d.index[missing],'DlyRetDurFlg']='MR'
    d.loc[d.index[missing],'DlyPrcFlg']='MP'
    d.loc[d.index[missing+1],'DlyRet']=both
    d.loc[d.index[missing+1],'DlyRetDurFlg']='P1'
    return d


def test_whole_two_day_span_recovers_horizon_without_daily_imputation():
    d,m=fixture(12)
    baseline=target(d,m)
    changed=gap(d)
    r=target(changed,m)
    assert r['label_status']=='ok_bridged' and r['bridged_spans']==1
    assert np.isclose(r['y_excess_5d'],baseline['y_excess_5d'])
    assert pd.isna(changed.DlyRet.iloc[3])


def test_span_cannot_extend_past_label_end():
    d,m=fixture(12)
    r=target(gap(d,6),m)
    assert r['label_status']=='span_crosses_exit'
    assert np.isnan(r['y_excess_5d'])


def test_span_cannot_be_split_at_entry_price():
    d,m=fixture(12)
    d=gap(d,1)
    assert target(d,m)['label_status']=='entry_not_tradable'
    # Starting at the later available close does not use the earlier two-day return.
    r=target(d,m,t=1)
    expected=np.prod(1+d.DlyRet.iloc[3:8])-1-(np.prod(1+m.iloc[3:8])-1)
    assert np.isclose(r['y_excess_5d'],expected)


def test_unverified_multiperiod_return_is_not_accepted():
    d,m=fixture(12)
    d.loc[d.index[3],'DlyRetDurFlg']='P1'  # previous day was priced; span does not match
    assert np.isnan(target(d,m)['y_excess_5d'])


def test_mv_numeric_return_does_not_become_valid_label():
    d,m=fixture(12)
    d.loc[d.index[4],'DlyRetMissFlg']='MV'
    r=target(d,m)
    assert r['label_status']=='unresolved_return_MV' and np.isnan(r['y_excess_5d'])


def test_delisting_proxy_is_separate_and_preserves_full_loss():
    d,m=fixture(12)
    d.loc[d.index[4],['DlyDelFlg','DlyRetDurFlg','DlyPrcFlg']]=['Y','DD','DA']
    d.loc[d.index[4],'DlyRet']=-1.
    r=target(d,m)
    assert r['label_status']=='delist_settlement_unverified'
    assert np.isnan(r['y_excess_5d']) and r['stock_return_5d_event_proxy']==-1.
    assert np.isclose(r['y_excess_5d_event_proxy'],-1-(np.prod(1+m.iloc[2:7])-1))
    d.loc[d.index[4],'DlyRetMissFlg']='MV'
    assert np.isnan(target(d,m)['y_excess_5d_event_proxy'])


def test_multiperiod_delist_does_not_double_count_preceding_daily_returns():
    d,m=fixture(12)
    d.loc[d.index[5],['DlyDelFlg','DlyRetDurFlg','DlyPrcFlg']]=['Y','P2','DP']
    r=target(d,m)
    assert r['label_status']=='unresolved_delist_NA'
    assert np.isnan(r['y_excess_5d'])
    assert np.isnan(r['y_excess_5d_event_proxy'])


def test_right_censoring_does_not_shorten_horizon():
    d,m=fixture(6)
    assert target(d,m)['label_status']=='right_censored'


def test_future_mutation_cannot_change_features_or_prior_label():
    d,m=fixture()
    before=make_features(d,m,configs()[0])
    changed=d.copy(); changed.loc[changed.index[131]:,'DlyRet']=.5
    changed.loc[changed.index[131]:,'DlyClose']=1e5
    changed.loc[changed.index[131]:,'DlyVol']=1e12
    changed.loc[changed.index[131]:,'DlyCumFacShr']=100.
    after=make_features(changed,m,configs()[0])
    pd.testing.assert_frame_equal(before.iloc[:131],after.iloc[:131])
    assert target(d,m,120)==target(changed,m,120)


def test_share_factor_global_rebase_cancels_in_volume_ratio():
    d,m=fixture()
    before=make_features(d,m,configs()[0])
    d.DlyCumFacShr*=3.7
    after=make_features(d,m,configs()[0])
    np.testing.assert_allclose(before,after,atol=1e-12,equal_nan=True)


def test_missing_daily_return_not_filled_from_future_span_for_features():
    d,m=fixture()
    d=gap(d,130)
    f=make_features(d,m,configs()[0])
    assert np.isnan(f.volatility_5.iloc[130]) and np.isnan(f.volatility_5.iloc[131])
    assert np.isnan(f.momentum_5.iloc[131])


def test_compound_handles_zero_wealth_and_missing_separately():
    s=pd.Series([.1,-1.,.2,np.nan])
    r=compound_past(s,3)
    assert r.iloc[2]==-1. and np.isnan(r.iloc[3])


def test_boundary_purge_retains_candidate_and_refit_can_use_it_later():
    p=pd.DataFrame({'signal_date':pd.to_datetime(['2019-12-27','2021-12-29']),
        'label_end_date':pd.to_datetime(['2020-01-07','2022-01-07']),
        'label_status':['ok','ok'],'signal_eligible':[True,True]})
    result=split_masks(p,configs()[1])
    assert len(result)==2 and result.purged_boundary.all()
    assert not result.eligible_main_sample.any()
    assert result.eligible_refit_through_2021.tolist()==[True,False]
