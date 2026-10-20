"""Market-array construction, target selection and portfolio statistics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from equity_research.paths import ROOT
from equity_research.models.common import sha
from equity_research.models.preprocessing import require
from equity_research.data.universe import regular_trade, filter_codes, SINGLE_DAY
from equity_research.data.features import return_segments

CONFIG = ROOT/'config/full_selection_backtest.json'
OUT = ROOT/'outputs/full_selection_backtest_v1'
MODELS = ['raw__ridge_03','cross_sectional_rank__ridge_13','raw__lightgbm_17','cross_sectional_rank__lightgbm_10','momentum20','equal_weight']
NAMES = dict(zip(MODELS, ['原始 Ridge','排名 Ridge','原始 LightGBM','排名 LightGBM','20日动量','股票池等权']))


def read_predictions(path):
    columns=['PERMNO','signal_date','split','signal_eligible','label_end_date','purged_boundary']+MODELS[:-1]
    d=pd.read_parquet(path,columns=columns,filters=[('split','==','validation'),
        ('signal_date','>=',pd.Timestamp('2020-01-01')),('signal_date','<',pd.Timestamp('2022-01-01'))])
    require(d.signal_date.between('2020-01-01','2021-12-31').all(),'Prediction date breach')
    require(not d.duplicated(['signal_date','PERMNO']).any(),'Duplicate predictions')
    return d.sort_values(['signal_date','PERMNO']).reset_index(drop=True)


def load_inputs():
    source=ROOT/'data/interim/crsp_audit_v1'
    pred_path=ROOT/'outputs/full_train_selection_v1/selected_predictions.parquet'
    pred=read_predictions(pred_path)
    cf=pd.read_parquet(source/'calendar.parquet',columns=['date','vwretd'],
        filters=[('date','>=','2020-01-01'),('date','<=','2021-12-31')])
    cf['date']=pd.to_datetime(cf.date);cf=cf.sort_values('date')
    cal=pd.DatetimeIndex(cf.date)
    schedules=[]
    for date,g in pred.groupby('signal_date',sort=True):
        pos=cal.get_loc(date);complete=pos+6<len(cal)
        require(g.purged_boundary.eq(not complete).all(),'Boundary mismatch')
        if complete:
            require(g.label_end_date.eq(cal[pos+6]).all(),'Horizon mismatch')
            schedules.append(dict(signal_date=date,entry_date=cal[pos+1],end_date=cal[pos+6]))
    schedule=pd.DataFrame(schedules)
    dates=cal[(cal>=schedule.signal_date.min())&(cal<=schedule.end_date.max())]
    market=cf.set_index('date').vwretd.reindex(dates)
    require(np.isfinite(market).all() and market.ge(-1).all(),'Invalid market returns')
    pred=pred.loc[pred.signal_date.isin(schedule.signal_date)].copy()
    require(np.isfinite(pred.loc[pred.signal_eligible,MODELS[:-1]]).all().all(),'Invalid eligible score')
    ids=set(pred.PERMNO.astype(int))
    columns=['PERMNO','DlyCalDt','DlyDelFlg','DlyPrc','DlyPrcFlg','DlyClose','DlyVol','DlyRet','DlyRetx','DlyCumFacPr',
        'DlyRetMissFlg','DlyRetDurFlg','ConditionalType','TradingStatusFlg','SecurityType','SecuritySubType',
        'ShareType','USIncFlg','IssuerType','PrimaryExch']
    parts=json.loads((source/'ingest.json').read_text(encoding='utf-8'))['parts']
    frames=[];read_parts=[]
    for name in parts:
        p=source/name;md=pq.read_metadata(p)
        lo=min(md.row_group(i).column(0).statistics.min for i in range(md.num_row_groups))
        hi=max(md.row_group(i).column(0).statistics.max for i in range(md.num_row_groups))
        wanted=sorted(i for i in ids if lo<=i<=hi)
        if wanted:
            d=pd.read_parquet(p,columns=columns,filters=[('PERMNO','in',wanted),
                ('DlyCalDt','>=',dates.min()),('DlyCalDt','<=',dates.max())])
            require(d.DlyCalDt.between(dates.min(),dates.max()).all(),'Market date breach')
            frames.append(d);read_parts.append(dict(name=name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    raw=pd.concat(frames,ignore_index=True).sort_values(['DlyCalDt','PERMNO']).reset_index(drop=True)
    require(not raw.duplicated(['DlyCalDt','PERMNO']).any(),'Duplicate market rows')
    uc=json.loads((ROOT/'config/universe.json').read_text(encoding='utf-8'))
    raw['sellable']=regular_trade(raw,uc)
    raw['buyable']=raw.sellable & filter_codes(raw,uc['security_filters'])
    raw['ordinary_return']=raw.DlyDelFlg.eq('N')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(SINGLE_DAY)&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
    raw['known_terminal']=raw.DlyDelFlg.eq('Y')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(SINGLE_DAY+['DD'])&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
    provenance=dict(prediction_sha256=sha(pred_path),market_parts_metadata=read_parts,
        filtered_market_sha256=hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest(),
        calendar_sha256=hashlib.sha256(pd.util.hash_pandas_object(cf,index=False).values.tobytes()).hexdigest(),
        market_rows=len(raw),securities=len(ids),test_rows_returned=0,
        first_signal=str(dates.min().date()),first_trade=str(schedule.entry_date.min().date()),
        final_liquidation=str(dates.max().date()),signal_dates=len(schedule))
    return pred,schedule,dates,market,raw,provenance


def make_targets(pred, schedule):
    result={m:{} for m in MODELS}
    for s in schedule.itertuples(index=False):
        g=pred.loc[pred.signal_date.eq(s.signal_date)&pred.signal_eligible]
        require(len(g)>0,'Empty eligible universe')
        for m in MODELS:
            chosen=g if m=='equal_weight' else g.sort_values([m,'PERMNO'],ascending=[False,True]).head(int(np.ceil(len(g)/10)))
            result[m][s.entry_date]=tuple(chosen.PERMNO.astype(int))
    return result


def audit_inputs(pred,schedule,dates,raw):
    # Potential full-universe holding paths, before portfolio performance is calculated.
    targets=make_targets(pred,schedule)['equal_weight']
    lookup=raw.set_index(['DlyCalDt','PERMNO']); rows=[]
    for s in schedule.itertuples(index=False):
        for p in targets[s.entry_date]:
            entry=lookup.loc[(s.entry_date,p)] if (s.entry_date,p) in lookup.index else None
            if entry is None or not entry.buyable:
                rows.append(dict(signal_date=s.signal_date,date=s.entry_date,PERMNO=p,kind='buy_not_executable',flag=None,duration=None))
                continue
            for date in dates[(dates>s.entry_date)&(dates<=s.end_date)]:
                r=lookup.loc[(date,p)] if (date,p) in lookup.index else None
                if r is None or not r.ordinary_return:
                    kind='absent_row' if r is None else 'known_delist' if r.known_terminal else 'other_return'
                    rows.append(dict(signal_date=s.signal_date,date=date,PERMNO=p,kind=kind,
                        flag=None if r is None else r.DlyRetMissFlg,duration=None if r is None else r.DlyRetDurFlg))
                    if r is not None and r.DlyDelFlg=='Y':break
    audit=pd.DataFrame(rows,columns=['signal_date','date','PERMNO','kind','flag','duration'])
    OUT.mkdir(parents=True,exist_ok=True)
    audit.to_csv(OUT/'potential_events_local.csv',index=False)
    return dict(potential_events=len(audit),counts={str(k):int(v) for k,v in audit.groupby(['kind','flag','duration'],dropna=False).size().items()})


def build_market(raw,dates,ids):
    ids=np.array(sorted(ids),dtype=int)
    def matrix(col,boolean=False):
        x=raw.pivot(index='DlyCalDt',columns='PERMNO',values=col).reindex(index=dates,columns=ids)
        return x.eq(True).to_numpy() if boolean else x.to_numpy()
    m=dict(ids=ids,dates=dates,ret=matrix('DlyRet').astype(float),
        ordinary=matrix('ordinary_return',True),terminal=matrix('known_terminal',True),
        buyable=matrix('buyable',True),sellable=matrix('sellable',True),flag=matrix('DlyRetMissFlg'),
        span_start=np.full((len(dates),len(ids)),-1,dtype=int),
        valid_price=np.zeros((len(dates),len(ids)),dtype=bool))
    lookup={p:j for j,p in enumerate(ids)}
    for p,g in raw.groupby('PERMNO'):
        j=lookup[p];x=g.set_index('DlyCalDt').reindex(dates)
        valid=x.DlyPrc.gt(0)&np.isfinite(x.DlyPrc)&x.DlyPrcFlg.isin(['TR','BA'])&x.DlyDelFlg.eq('N')
        m['valid_price'][:,j]=valid.to_numpy()
        segments=return_segments(x)
        # Terminal observations must have a valid immediately preceding price.
        m['terminal'][:,j] &= segments['terminal']
        for start,end in enumerate(segments['ends']):
            if end>start+1:
                expected=x.DlyPrc.iloc[end]/x.DlyPrc.iloc[start]*x.DlyCumFacPr.iloc[start]/x.DlyCumFacPr.iloc[end]-1
                require(np.isclose(expected,x.DlyRetx.iloc[end],atol=1e-5,rtol=0),'Unverified multi-period adjusted price bridge')
                m['span_start'][end,j]=start
    return m


def performance(daily):
    # Opening date is the initial capital anchor, not an extra return observation.
    d=daily.iloc[1:];r=d.return_net.to_numpy();n=len(r)
    total=float(np.prod(1+r)-1);std=float(np.std(r,ddof=1));path=np.r_[1.,np.cumprod(1+r)]
    return dict(days=n,total_return=total,cagr=(1+total)**(252/n)-1,
        annual_volatility=std*np.sqrt(252),sharpe=float(np.mean(r)/std*np.sqrt(252)) if std>0 else np.nan,
        max_drawdown=float(np.min(path/np.maximum.accumulate(path)-1)),
        mean_rebalance_turnover=float(d.loc[d.rebalance,'turnover'].mean()),
        total_fees=float(d.fees.sum()),average_cash_weight=float(d.cash_weight.mean()),
        average_receivable_weight=float(d.receivable_weight.mean()),ending_receivable_weight=float(d.receivable_weight.iloc[-1]),
        ending_cash_weight=float(d.cash_weight.iloc[-1]),stale_valuation_days=int(d.valuation_estimated.sum()),
        failed_buy_amount=float(d.failed_buy.sum()),failed_sell_amount=float(d.failed_sell.sum()))
