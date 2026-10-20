"""Chronological test candidate audit before predictions or performance publication."""
from equity_research.provenance import hash_matches
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from equity_research.data.ingest import ROOT,write_json
from equity_research.models.common import sha
from equity_research.data.date_gate import signal_schedule
from equity_research.backtest.calendar import is_exchange_day
from equity_research.backtest.market import regular_trade,filter_codes,SINGLE_DAY,require,build_market

OUT=ROOT/'outputs/test_event_audit_v1'

def main():
    require(not (OUT/'audit.json').exists(),'Preserve finished audit')
    frozen=json.loads((ROOT/'outputs/final_models_v1/manifest.json').read_text(encoding='utf-8'))
    for p,h in frozen.items():require(hash_matches(ROOT/p, h),'Final model artifact changed')
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'execution_scope.json',dict(user_instruction='直接告诉我接下来应该做什么，并且去执行',
        scope='test data and chronological event audit followed by frozen evaluation; no new fitting or tuning',
        performance_gate='unresolved events must be zero before publication',frozen_model_hashes=frozen))
    source=ROOT/'data/interim/crsp_audit_v1'
    cf=pd.read_parquet(source/'calendar.parquet',columns=['date'],filters=[('date','>=','2015-01-01'),('date','<=','2025-12-31')])
    cal=pd.DatetimeIndex(pd.to_datetime(cf.date)).sort_values()
    public=pd.DatetimeIndex([d for d in pd.date_range('2022-01-01','2025-12-31') if is_exchange_day(d)])
    require(cal[cal>='2022-01-01'].equals(public),'Public and vendor trading calendars disagree')
    schedule=signal_schedule(cal,'2022-01-01','2025-12-31')
    schedule.to_csv(OUT/'schedule.csv',index=False)
    frame=pd.read_parquet(ROOT/'data/processed/research_v1/features_only.parquet',columns=['PERMNO','signal_date','signal_eligible'],
        filters=[('signal_date','>=',pd.Timestamp('2022-01-01')),('signal_date','<=',pd.Timestamp('2025-12-31'))])
    require(not frame.duplicated(['PERMNO','signal_date']).any(),'Duplicate test candidate')
    require(set(schedule.signal_date)<=set(frame.signal_date),'Missing signal date')
    frame=frame[frame.signal_date.isin(schedule.signal_date)]
    require(frame.groupby('signal_date').size().eq(500).all(),'Candidate count drift')
    ids=set(frame.PERMNO.astype(int));dates=cal[(cal>=schedule.signal_date.min())&(cal<=schedule.end_date.max())]
    columns=['PERMNO','Ticker','PERMCO','SICCD','DlyCalDt','DlyDelFlg','DlyPrc','DlyPrcFlg','DlyClose','DlyVol','DlyRet','DlyRetx','DlyCumFacPr',
        'DlyRetMissFlg','DlyRetDurFlg','ConditionalType','TradingStatusFlg','SecurityType','SecuritySubType','ShareType','USIncFlg','IssuerType','PrimaryExch']
    frames=[];log=[]
    for name in json.loads((source/'ingest.json').read_text())['parts']:
        p=source/name;md=pq.read_metadata(p)
        lo=min(md.row_group(i).column(0).statistics.min for i in range(md.num_row_groups));hi=max(md.row_group(i).column(0).statistics.max for i in range(md.num_row_groups))
        wanted=sorted(i for i in ids if lo<=i<=hi)
        if not wanted:continue
        filters=[('PERMNO','in',wanted),('DlyCalDt','>=',dates.min()),('DlyCalDt','<=',dates.max())]
        d=pd.read_parquet(p,columns=columns,filters=filters)
        require(d.DlyCalDt.between(dates.min(),dates.max()).all(),'Date breach')
        frames.append(d);log.append(dict(part=name,rows=len(d),min_date=str(d.DlyCalDt.min()),max_date=str(d.DlyCalDt.max())))
    raw=pd.concat(frames,ignore_index=True).sort_values(['DlyCalDt','PERMNO']).reset_index(drop=True)
    require(not raw.duplicated(['DlyCalDt','PERMNO']).any(),'Duplicate test market row')
    uc=json.loads((ROOT/'config/universe.json').read_text())
    raw['sellable']=regular_trade(raw,uc);raw['buyable']=raw.sellable&filter_codes(raw,uc['security_filters'])
    raw['ordinary_return']=raw.DlyDelFlg.eq('N')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(SINGLE_DAY)&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
    raw['known_terminal']=raw.DlyDelFlg.eq('Y')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(SINGLE_DAY+['DD'])&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
    raw.to_parquet(OUT/'candidate_market_local.parquet',index=False)
    frame.to_parquet(OUT/'candidates_local.parquet',index=False)
    events=raw[raw.DlyDelFlg.eq('Y')].copy().sort_values(['DlyCalDt','PERMNO'])
    events.to_csv(OUT/'delistings_local.csv',index=False)
    (OUT/'wrds_permnos.txt').write_text(' '.join(map(str,sorted(events.PERMNO.unique()))),encoding='utf-8')
    anomalous=raw[~raw.ordinary_return&~raw.DlyDelFlg.eq('Y')]
    anomalous.to_parquet(OUT/'nonordinary_rows_local.parquet',index=False)
    # A full candidate audit is intentionally independent of scores and realized P&L.
    error=None
    try:build_market(raw,dates,ids)
    except (ValueError,AssertionError) as e:error=str(e)
    write_json(OUT/'observation_access_log.json',log)
    write_json(OUT/'audit.json',dict(status='awaiting_event_terms',candidate_rows=len(frame),securities=len(ids),market_rows=len(raw),
        signals=len(schedule),first_signal=str(schedule.signal_date.min().date()),first_entry=str(schedule.entry_date.min().date()),
        terminal_date=str(schedule.end_date.max().date()),delisting_rows=len(events),nonordinary_rows=len(anomalous),market_bridge_error=error,
        model_fits=0,predictions_generated=0,performance_evaluated=False))
    print(json.dumps(json.loads((OUT/'audit.json').read_text()),indent=2))
    print(events[['DlyCalDt','PERMNO','Ticker']].to_string(index=False))
