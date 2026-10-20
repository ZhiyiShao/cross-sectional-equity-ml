"""Causal price/volume features and auditable close(t+1)->close(t+6) targets.

No fitting, imputation, standardization, model selection or performance analysis.
All historical universe candidates survive, including rows without a label.
"""
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from equity_research.data.ingest import ROOT, write_json
from equity_research.data.universe import SINGLE_DAY, filter_codes, regular_trade
from equity_research.data.returns import selected_parts


def compound_past(returns, window):
    gross=(1+returns).where(returns.ge(-1))
    logg=np.log(gross.where(gross.gt(0),1.)).where(gross.notna())
    count=gross.rolling(window,min_periods=window).count()
    zero=gross.eq(0).rolling(window,min_periods=window).sum()
    result=np.expm1(logg.rolling(window,min_periods=window).sum())
    return result.mask(zero.gt(0),-1.).where(count.eq(window))


def make_features(x, market, universe_cfg):
    ordinary=(x.DlyDelFlg.eq('N') & x.DlyRetMissFlg.eq('NA') &
              x.DlyRetDurFlg.isin(SINGLE_DAY) & x.DlyRet.ge(-1) & np.isfinite(x.DlyRet))
    r=x.DlyRet.where(ordinary)
    trade=regular_trade(x,universe_cfg)
    dollars=(x.DlyClose*x.DlyVol).where(trade)
    f=pd.DataFrame(index=x.index)
    for w in [5,10,20,60,120]:
        f[f'momentum_{w}']=compound_past(r,w)
    f['momentum_60_skip5']=compound_past(r.shift(5),55)
    f['reversal_1']=-r
    for w in [5,20,60]:
        f[f'volatility_{w}']=r.rolling(w,min_periods=w).std(ddof=1)*np.sqrt(252)
    f['downside_volatility_20']=np.sqrt(r.clip(upper=0).pow(2).rolling(20,min_periods=20).mean()*252)
    mean20=dollars.rolling(20,min_periods=20).mean()
    for w in [20,60]:
        f[f'log_dollar_volume_{w}']=np.log1p(dollars.rolling(w,min_periods=w).mean())
    f['dollar_volume_ratio_5_20']=dollars.rolling(5,min_periods=5).mean()/mean20.where(mean20.gt(0))-1
    # The terminal basis cancels between numerator and denominator. Never export
    # this absolute adjusted volume or factor as a model feature.
    v=(x.DlyVol*x.DlyCumFacShr).where(trade & x.DlyCumFacShr.gt(0))
    v20=v.rolling(20,min_periods=20).mean()
    f['share_volume_ratio_5_20']=v.rolling(5,min_periods=5).mean()/v20.where(v20.gt(0))-1
    turn=(x.DlyVol/(1000*x.ShrOut.where(x.ShrOut.gt(0)))).where(trade)
    f['share_turnover_20']=turn.rolling(20,min_periods=20).mean()
    f['amihud_20']=(r.abs()/dollars.where(dollars.gt(0))*1e6).rolling(20,min_periods=20).mean()
    spread=x.DlyHigh-x.DlyLow
    good_range=(trade & spread.ge(0) & x.DlyClose.between(x.DlyLow,x.DlyHigh))
    f['intraday_range_20']=(spread/x.DlyClose).where(good_range).rolling(20,min_periods=20).mean()
    clv=((x.DlyClose-x.DlyLow)/spread.where(spread.gt(0))).mask(spread.eq(0),.5).where(good_range)
    f['close_location_5']=clv.rolling(5,min_periods=5).mean()
    f['log_market_cap']=np.log((x.DlyCap*1000).where(x.DlyCap.gt(0)))
    var=market.rolling(60,min_periods=60).var(ddof=1)
    f['market_beta_60']=r.rolling(60,min_periods=60).cov(market)/var.where(var.gt(0))
    return f.replace([np.inf,-np.inf],np.nan)


def return_segments(x):
    """Map a starting CLOSE to a verified cumulative return endpoint.

    A span Pk covers k+1 market periods. It may replace intervening MP rows
    only when prices, flags and missing-return coverage match exactly.
    """
    n=len(x)
    ends=np.full(n,-1,dtype=np.int32)
    gross=np.full(n,np.nan)
    delist=x.DlyDelFlg.to_numpy()
    flag=x.DlyRetMissFlg.to_numpy()
    duration=x.DlyRetDurFlg.to_numpy()
    ret=x.DlyRet.to_numpy(dtype=float)
    price=x.DlyPrc.to_numpy(dtype=float)
    pflag=x.DlyPrcFlg.to_numpy()
    valid_price=(np.isfinite(price)&(price>0)&np.isin(pflag,['TR','BA'])&(delist=='N'))
    previous_good=-1
    for j in range(n):
        if delist[j]=='N' and flag[j]=='NA' and np.isfinite(ret[j]) and ret[j]>=-1:
            dur=duration[j]
            if dur in SINGLE_DAY:
                if j>0 and valid_price[j-1]:
                    ends[j-1]=j; gross[j-1]=1+ret[j]
            elif isinstance(dur,str) and dur.startswith('P') and dur[1:].isdigit():
                span=int(dur[1:])+1
                start=j-span
                if (1<=int(dur[1:])<=9 and start>=0 and previous_good==start and
                    valid_price[j] and np.all(flag[start+1:j]=='MP') and
                    np.all(np.isnan(ret[start+1:j])) and np.all(delist[start+1:j]=='N')):
                    ends[start]=j; gross[start]=1+ret[j]
        if valid_price[j]:
            previous_good=j
    terminal=(delist=='Y')&(flag=='NA')&np.isfinite(ret)&(ret>=-1)&np.isin(duration,SINGLE_DAY+['DD'])
    terminal[0]=False
    terminal[1:] &= valid_price[:-1]
    return {'ends':ends,'gross':gross,'terminal':terminal,'returns':ret,'flags':flag,'delist':delist}


def label_one(t, x, market, entry_trade, exit_trade, segments, lag=1, horizon=5):
    n=len(x); entry=t+lag; end=entry+horizon
    result={'entry_date':x.index[entry] if entry<n else pd.NaT,
            'label_end_date':x.index[end] if end<n else pd.NaT,
            'label_status':'right_censored','stock_return_5d':np.nan,
            'market_return_5d':np.nan,'y_excess_5d':np.nan,
            'stock_return_5d_event_proxy':np.nan,'y_excess_5d_event_proxy':np.nan,
            'bridged_spans':0,'label_proxy_assumption':False}
    if end>=n:
        return result
    mr=market.iloc[entry+1:end+1].to_numpy(dtype=float)
    if not np.isfinite(mr).all() or (mr < -1).any():
        result['label_status']='market_return_missing'; return result
    market_return=float(np.prod(1+mr)-1)
    result['market_return_5d']=market_return
    if not entry_trade[entry]:
        result['label_status']='entry_not_tradable'; return result
    boundary=entry; wealth=1.
    while boundary<end:
        next_day=boundary+1
        if segments['delist'][next_day]=='Y':
            if segments['terminal'][next_day]:
                result['label_status']='delist_settlement_unverified'
                proxy=wealth*(1+segments['returns'][next_day])-1
                result.update(stock_return_5d_event_proxy=float(proxy),
                              y_excess_5d_event_proxy=float(proxy-market_return),label_proxy_assumption=True)
            else:
                result['label_status']='unresolved_delist_'+str(segments['flags'][next_day])
            return result
        finish=segments['ends'][boundary]
        if finish<0:
            flag=segments['flags'][next_day]
            result['label_status']='unresolved_return_'+(str(flag) if pd.notna(flag) else 'absent_row')
            return result
        if finish>end:
            result['label_status']='span_crosses_exit'; return result
        wealth*=segments['gross'][boundary]
        result['bridged_spans']+=int(finish-boundary>1)
        boundary=int(finish)
    if not exit_trade[end]:
        result['label_status']='exit_not_tradable'; return result
    result.update(label_status='ok_bridged' if result['bridged_spans'] else 'ok',
                  stock_return_5d=float(wealth-1),y_excess_5d=float(wealth-1-market_return))
    return result


def split_masks(panel,cfg):
    panel=panel.copy()
    panel['split']='outside'
    panel['purged_boundary']=False
    for name,(start,end) in cfg['splits'].items():
        s=panel.signal_date.between(pd.Timestamp(start),pd.Timestamp(end))
        panel.loc[s,'split']=name
        panel.loc[s,'purged_boundary']=panel.loc[s,'label_end_date'].gt(pd.Timestamp(end))
    valid=panel.label_status.isin(['ok','ok_bridged'])
    panel['eligible_main_sample']=panel.signal_eligible&valid&~panel.purged_boundary&panel.split.ne('outside')
    panel['eligible_refit_through_2021']=panel.signal_eligible&valid&panel.signal_date.between('2015-01-01','2021-12-31')&panel.label_end_date.le(pd.Timestamp('2021-12-31'))
    return panel


def main():
    source=ROOT/'data/interim/crsp_audit_v1'
    out=ROOT/'data/processed/research_v1'
    cfg=json.loads((ROOT/'config/features_labels.json').read_text(encoding='utf-8'))
    uc=json.loads((ROOT/'config/universe.json').read_text(encoding='utf-8'))
    review=json.loads((out/'return_review_summary.json').read_text(encoding='utf-8'))
    assert review['reviewed_records']==16 and review['all_multi_period_cases_verified']
    if (out/'dataset_summary.json').exists():
        raise RuntimeError('Completed research dataset exists; archive/version it before rebuilding.')
    universe=pd.read_parquet(ROOT/'data/processed/crsp_v1/monthly_universe.parquet')
    calframe=pd.read_parquet(source/'calendar.parquet')
    cal=pd.DatetimeIndex(pd.to_datetime(calframe.date))
    market=pd.Series(calframe.vwretd.to_numpy(),index=cal)
    first=cal.get_loc(pd.Timestamp(cfg['signal_start']))
    signal_pos=np.arange(first,len(cal),cfg['signal_step_trading_days'])
    signals=cal[signal_pos]
    signal_lookup={d:i for d,i in zip(signals,signal_pos)}
    membership={int(p):g for p,g in universe.groupby('PERMNO')}
    panels=[]; done=0
    columns=['PERMNO','DlyCalDt','DlyDelFlg','DlyPrc','DlyPrcFlg','DlyClose','DlyHigh','DlyLow',
             'DlyVol','ShrOut','DlyCap','DlyCumFacShr','DlyRet','DlyRetMissFlg','DlyRetDurFlg',
             'PrimaryExch','SecurityType','SecuritySubType','ShareType','USIncFlg','IssuerType',
             'ConditionalType','TradingStatusFlg']
    for _,d in selected_parts(source,set(membership),columns):
        for permno,g in d.groupby('PERMNO',sort=False):
            x=g.set_index('DlyCalDt').reindex(cal)
            f=make_features(x,market,uc)
            trade=regular_trade(x,uc).to_numpy()
            entry_trade=trade&filter_codes(x,uc['security_filters']).to_numpy()
            segments=return_segments(x)
            m=membership[int(permno)].set_index('effective_month')
            selected_signals=signals[signals.to_period('M').astype(str).isin(m.index)]
            f=f.loc[selected_signals].copy()
            f.insert(0,'signal_date',selected_signals)
            f.insert(0,'PERMNO',int(permno))
            f['effective_month']=selected_signals.to_period('M').astype(str)
            f['formation_date']=f.effective_month.map(m.formation_date)
            f['PERMCO']=f.effective_month.map(m.PERMCO)
            f['signal_eligible']=[bool(entry_trade[signal_lookup[t]]) for t in selected_signals]
            f['feature_missing_count']=f[cfg['features']].isna().sum(axis=1)
            labels=[label_one(signal_lookup[t],x,market,entry_trade,trade,segments,
                              cfg['entry_lag_trading_days'],cfg['holding_period_trading_days']) for t in selected_signals]
            panel=pd.concat([f.reset_index(drop=True),pd.DataFrame(labels)],axis=1)
            panels.append(panel)
            done+=1
            if done%200==0:
                print(f'Built causal features/targets for {done}/{len(membership)} securities; no observations displayed.',flush=True)
    panel=pd.concat(panels,ignore_index=True).sort_values(['signal_date','PERMNO']).reset_index(drop=True)
    panel=split_masks(panel,cfg)
    assert panel[cfg['features']].shape[1]==21
    assert not panel.duplicated(['signal_date','PERMNO']).any()
    assert panel.groupby('signal_date').size().eq(500).all()
    assert len(panel)==len(signals)*500
    assert (panel.formation_date<panel.signal_date).all()
    assert not np.isinf(panel[cfg['features']].to_numpy(dtype=float)).any()
    panel.to_parquet(out/'features_labels.parquet',index=False,compression='zstd')
    # Explicit feature-only view prevents accidentally feeding execution/target fields to a model.
    panel[['PERMNO','signal_date','signal_eligible']+cfg['features']].to_parquet(out/'features_only.parquet',index=False,compression='zstd')
    quality=panel.groupby(['split','label_status','purged_boundary'],dropna=False).size().reset_index(name='rows')
    quality.to_csv(out/'label_quality_counts.csv',index=False)
    missing=panel.groupby('split')[cfg['features']].agg(lambda s: int(s.isna().sum()))
    missing.to_csv(out/'feature_missing_counts.csv')
    coverage=panel.groupby('split').agg(candidates=('PERMNO','size'),signal_dates=('signal_date','nunique'),
        signal_eligible=('signal_eligible','sum'),main_samples=('eligible_main_sample','sum'),
        proxy_labels=('label_proxy_assumption','sum'),purged=('purged_boundary','sum'))
    coverage.to_csv(out/'split_coverage.csv')
    cases=pd.read_csv(out/'return_review_decisions.csv',keep_default_na=False)
    impacts=[]
    for _,case in cases.iterrows():
        event=pd.Timestamp(case.event_date)
        p=panel.loc[panel.PERMNO.eq(case.PERMNO)]
        touch=p.loc[p.entry_date.lt(event)&p.label_end_date.ge(event)]
        entry=p.entry_date.eq(event)
        impacts.append({'case_id':case.case_id,'holding_windows':len(touch),
                        'entry_on_event':int(entry.sum()),'label_status_counts':json.dumps(touch.label_status.value_counts().to_dict())})
    pd.DataFrame(impacts).to_csv(out/'review_case_label_impacts.csv',index=False)
    files=[ROOT/'config/features_labels.json',ROOT/'src/equity_research/data/features.py',ROOT/'src/equity_research/data/returns.py',
           ROOT/'docs/RETURN_POLICY.md',ROOT/'data/processed/crsp_v1/monthly_universe.parquet',
           out/'return_review_decisions.csv',out/'features_labels.parquet',out/'features_only.parquet']
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(panel),
        'signal_dates':len(signals),'securities':done,'features':cfg['features'],
        'label_status_counts':panel.label_status.value_counts().to_dict(),
        'eligible_main_by_split':panel.groupby('split').eligible_main_sample.sum().astype(int).to_dict(),
        'signal_eligible_rows':int(panel.signal_eligible.sum()),
        'purged_rows':int(panel.purged_boundary.sum()),
        'feature_complete_rows':int(panel.feature_missing_count.eq(0).sum()),
        'no_model_fitted':True,'strategy_backtest_ready':False,
        'configuration':cfg,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    write_json(out/'dataset_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sha256','configuration','features']},indent=2),flush=True)
