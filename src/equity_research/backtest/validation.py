"""Validation portfolios with explicit trade receivables/payables and zero financing."""
from equity_research.provenance import hash_matches
import json
from pathlib import Path
import numpy as np
import pandas as pd
import equity_research.backtest.market as base
from equity_research.backtest.validation_calendar import Ledger, settlement_date, CONFIG as CALENDAR_CONFIG
ROOT=base.ROOT
OUT=ROOT/'outputs/final_funding_backtest_v1'
CONFIG=ROOT/'config/validation_corporate_actions.json'


from equity_research.backtest import engine


def simulate(m, targets, terms, cost_bps=10, availability=2, settled_only=False, record=False):
    return engine.simulate(m, targets, terms, cost_bps, availability, settled_only, record,
                           ledger_factory=Ledger, settlement_function=settlement_date)


def verify(d, o, h):
    return engine.verify(d, o, h, settlement_function=settlement_date)


def main():
 base.require(not OUT.exists(),'Preserve finished output')
 cfg=json.loads(CONFIG.read_text(encoding='utf-8'));terms={x['permno']:x for x in cfg['events']}
 protected={}
 for folder in ['full_selection_backtest_v1','full_train_selection_v1','corporate_actions_v1','lvgo_evidence_v1','corporate_action_backtest_v1']:
  mapping=json.loads((ROOT/'outputs'/folder/'manifest.json').read_text(encoding='utf-8'))
  for p,h in mapping.items():base.require(hash_matches(ROOT/p, h),'Protected artifact changed '+p)
  protected.update(mapping)
 pred,schedule,dates,market,raw,provenance=base.load_inputs();targets=base.make_targets(pred,schedule)
 # Extra successor observations restricted to the same validation trading interval.
 extra=sorted({x['successor_permno'] for x in terms.values() if x['ratio']}-set(raw.PERMNO))
 source=ROOT/'data/interim/crsp_audit_v1';frames=[raw]
 for part in json.loads((source/'ingest.json').read_text())['parts']:
  if not extra:break
  f=pd.read_parquet(source/part,columns=[c for c in raw if c not in ['sellable','buyable','ordinary_return','known_terminal']],filters=[('PERMNO','in',extra),('DlyCalDt','>=',dates.min()),('DlyCalDt','<=',dates.max())])
  if len(f):frames.append(f)
 raw=pd.concat(frames,ignore_index=True)
 uc=json.loads((ROOT/'config/universe.json').read_text());raw['sellable']=base.regular_trade(raw,uc)
 raw['buyable']=raw.sellable&base.filter_codes(raw,uc['security_filters'])
 raw['ordinary_return']=raw.DlyDelFlg.eq('N')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(base.SINGLE_DAY)&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
 raw['known_terminal']=raw.DlyDelFlg.eq('Y')&raw.DlyRetMissFlg.eq('NA')&raw.DlyRetDurFlg.isin(base.SINGLE_DAY+['DD'])&np.isfinite(raw.DlyRet)&raw.DlyRet.ge(-1)
 m=base.build_market(raw,dates,set(raw.PERMNO));m['price']=raw.pivot(index='DlyCalDt',columns='PERMNO',values='DlyPrc').reindex(index=dates,columns=m['ids']).to_numpy()
 base.require(dates.min()>=pd.Timestamp('2020-01-01') and dates.max()<pd.Timestamp('2022-01-01'),'Date breach')
 OUT.mkdir();pd.DataFrame(cfg['events']).to_csv(OUT/'event_registry_local.csv',index=False)
 stats=[];annual=[];ds=[];os=[];hs=[];es=[];checks=0
 scenarios=[(f'lag_{lag}',lag,False,cost) for lag in cfg['availability_lags'] for cost in cfg['costs_bps']]
 scenarios.append(('settled_only',2,True,10))
 for name,lag,settled_only,cost in scenarios:
  for model in base.MODELS:
   primary=name=='lag_2' and cost==10
   d,o,h,e=simulate(m,targets[model],terms,cost,lag,settled_only,primary)
   if primary:checks+=verify(d,o,h);os.append(o.assign(model=model));hs.append(h.assign(model=model))
   ds.append(d.assign(model=model,scenario=name,cost_bps=cost));es.append(e.assign(model=model,scenario=name,cost_bps=cost))
   stats.append(dict(model=model,scenario=name,cost_bps=cost,**base.performance(d)))
   for year,g in d.iloc[1:].groupby(d.date.iloc[1:].dt.year):annual.append(dict(model=model,scenario=name,cost_bps=cost,year=int(year),total_return=float((1+g.return_net).prod()-1)))
  print('Completed',name,cost,flush=True)
 for fn,t in [('metrics',pd.DataFrame(stats)),('annual_metrics',pd.DataFrame(annual)),('daily_portfolios',pd.concat(ds))]:t.to_csv(OUT/(fn+'.csv'),index=False)
 pd.concat(os).to_parquet(OUT/'trades_local.parquet',index=False);pd.concat(hs).to_parquet(OUT/'holdings_local.parquet',index=False);pd.concat(es).to_parquet(OUT/'events_local.parquet',index=False)
 for p,h in protected.items():base.require(hash_matches(ROOT/p, h),'Protected output changed')
 meta=dict(provenance=provenance,simulations=len(stats),daily_checks=sum(len(d) for d in ds),independent_ledger_days=checks,new_fit_calls=0,new_prediction_calls=0,test_rows=0,protected_hashes=protected,config_sha256=base.sha(CONFIG),calendar_sha256=base.sha(CALENDAR_CONFIG))
 (OUT/'verification.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 pd.DataFrame({'trade_date':dates,'settlement_date':[settlement_date(d) for d in dates]}).to_csv(OUT/'settlement_dates.csv',index=False)
 print('Verified',len(stats),'simulations',checks,'ledger days')
