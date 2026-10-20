"""Authorized one-time frozen test: no fit, tuning, or annual model updates."""
from equity_research.provenance import hash_matches
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import equity_research.backtest.market as base
from equity_research.backtest.engine import simulate, verify
from equity_research.models.common import sha, transform_features, load_lightgbm
from equity_research.models.preprocessing import cross_sectional_rank
from equity_research.backtest.calendar import CONFIG as CALENDAR_CONFIG

ROOT=base.ROOT
AUDIT=ROOT/'outputs/test_event_audit_v1'
OUT=ROOT/'outputs/frozen_test_v1'

def inputs():
    registry=json.loads((AUDIT/'action_registry_draft.json').read_text())
    base.require(registry['unresolved_count']==0,'Unresolved event policy')
    for p,h in registry['inputs'].items():base.require(hash_matches(ROOT/p, h),'Event input drift')
    base.require(registry['exception_authorization']=='采用简化口径并明确披露','Missing exception authorization')
    terms={e['permno']:e for e in registry['events']}
    schedule=pd.read_csv(AUDIT/'schedule.csv',parse_dates=['signal_date','entry_date','end_date'])
    raw=pd.read_parquet(AUDIT/'execution_market_local.parquet')
    dates=pd.DatetimeIndex(sorted(raw.DlyCalDt.unique()))
    base.require(dates.min()==schedule.signal_date.min() and dates.max()==schedule.end_date.max(),'Market date drift')
    m=base.build_market(raw,dates,set(raw.PERMNO))
    m['price']=raw.pivot(index='DlyCalDt',columns='PERMNO',values='DlyPrc').reindex(index=dates,columns=m['ids']).to_numpy()
    for p,t in terms.items():
        i=dates.get_loc(pd.Timestamp(t['event_date']));j=np.flatnonzero(m['ids']==p)[0]
        base.require(i>0 and m['price'][i-1,j]>0,'No pre-action price')
        # Legal event accounting replaces vendor terminal return only in NAV, never in ML labels.
        m['terminal'][i,j]=True;m['ordinary'][i,j]=False
    return registry,terms,schedule,m

def predictions(bundle,schedule):
    features=bundle['features']
    frame=pd.read_parquet(ROOT/'data/processed/research_v1/features_only.parquet',
        columns=['PERMNO','signal_date','signal_eligible']+features,
        filters=[('signal_date','>=',schedule.signal_date.min()),('signal_date','<=',schedule.signal_date.max())])
    frame=frame[frame.signal_date.isin(schedule.signal_date)].sort_values(['signal_date','PERMNO']).reset_index(drop=True)
    base.require(not frame.duplicated(['signal_date','PERMNO']).any(),'Duplicate prediction candidates')
    base.require(frame.groupby('signal_date').size().eq(500).all(),'Candidate count drift')
    ranked=cross_sectional_rank(frame,features)
    # Independent scipy ranks for every same-date eligible population, before reading labels.
    for _,g in frame.groupby('signal_date'):
        for f in features:
            ix=g.index[g.signal_eligible&np.isfinite(g[f])]
            np.testing.assert_allclose(ranked.loc[ix,f],(rankdata(g.loc[ix,f])-.5)/len(ix)-.5,atol=1e-12)
    pred=frame[['PERMNO','signal_date','signal_eligible']].copy()
    with threadpool_limits(limits=4):
        for c in bundle['models']:
            base.require(sha(ROOT/c['path'])==c['sha256'],'Model hash mismatch')
            base.require(sha(ROOT/c['preprocessing'])==c['preprocessing_sha256'],'Preprocessing hash mismatch')
            state=json.loads((ROOT/c['preprocessing']).read_text())
            a=frame[features] if c['representation']=='raw' else ranked
            X=transform_features(a,state)
            ar=a.to_numpy(float);ar[~np.isfinite(ar)]=np.nan
            clean=np.where(np.isnan(ar),state['median'],np.clip(ar,state['lower'],state['upper']))
            np.testing.assert_allclose(X,(clean-state['mean'])/state['scale'],atol=1e-12)
            model=joblib.load(ROOT/c['path']) if c['model']=='ridge' else load_lightgbm(ROOT/c['path'])
            pred[c['id']]=model.predict(X)
    baseline=json.loads((ROOT/bundle['baseline_state']).read_text())
    pred['momentum20']=frame.momentum_20.replace([np.inf,-np.inf],np.nan).fillna(baseline['momentum_missing'])
    base.require(np.isfinite(pred.loc[pred.signal_eligible,base.MODELS[:-1]]).all().all(),'Invalid prediction')
    return pred

def main():
    print('Checking frozen artifacts',flush=True)
    base.require(not OUT.exists(),'Preserve finished or interrupted test; inspect before resuming')
    manifest=json.loads((ROOT/'outputs/final_models_v1/manifest.json').read_text())
    for p,h in manifest.items():base.require(hash_matches(ROOT/p, h),'Frozen model artifact drift')
    registry,terms,schedule,m=inputs()
    print('Market and event terms loaded',flush=True)
    # Chronological equal-weight event exercise precedes model predictions. No metric ranking.
    candidates=pd.read_parquet(AUDIT/'candidates_local.parquet')
    ew={r.entry_date:tuple(candidates.loc[candidates.signal_date.eq(r.signal_date)&candidates.signal_eligible,'PERMNO']) for r in schedule.itertuples()}
    for lag in [0,2,5,20]:
        d,o,h,e=simulate(m,ew,terms,10,lag,False,True)
        print('Event exercise simulated, lag',lag,flush=True)
        verify(d,o,h)
        print('Event exercise verified, lag',lag,flush=True)
    OUT.mkdir()
    registry['status']='locked_before_model_predictions'
    (OUT/'event_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    bundle=json.loads((ROOT/'outputs/final_models_v1/model_bundle.json').read_text())
    pred=predictions(bundle,schedule)
    pred.to_parquet(OUT/'predictions_local.parquet',index=False)
    targets=base.make_targets(pred,schedule)
    stats=[];annual=[];ds=[];os=[];hs=[];es=[];checks=0
    scenarios=[(f'lag_{lag}',lag,False,cost) for lag in [0,2,5,20] for cost in [0,5,10,20]]
    scenarios.append(('settled_only',2,True,10))
    for name,lag,settled,cost in scenarios:
        for model in base.MODELS:
            primary=name=='lag_2' and cost==10
            d,o,h,e=simulate(m,targets[model],terms,cost,lag,settled,primary)
            if primary:
                checks+=verify(d,o,h);os.append(o.assign(model=model));hs.append(h.assign(model=model))
            ds.append(d.assign(model=model,scenario=name,cost_bps=cost));es.append(e.assign(model=model,scenario=name,cost_bps=cost))
            stats.append(dict(model=model,scenario=name,cost_bps=cost,**base.performance(d),
                cvr_contractual_maximum=float(d.iloc[-1].cvr_contractual_maximum),
                estimated_valuation_days=int(d.valuation_estimated.sum())))
            for year,g in d.iloc[1:].groupby(d.date.iloc[1:].dt.year):
                annual.append(dict(model=model,scenario=name,cost_bps=cost,year=int(year),total_return=float((1+g.return_net).prod()-1)))
        print('Completed',name,cost,flush=True)
    for fn,t in [('metrics',pd.DataFrame(stats)),('annual_metrics',pd.DataFrame(annual)),('daily_portfolios',pd.concat(ds))]:
        t.to_csv(OUT/(fn+'.csv'),index=False)
    pd.concat(os).to_parquet(OUT/'trades_local.parquet',index=False)
    pd.concat(hs).to_parquet(OUT/'holdings_local.parquet',index=False)
    pd.concat(es).to_parquet(OUT/'events_local.parquet',index=False)
    for p,h in manifest.items():base.require(hash_matches(ROOT/p, h),'Frozen model changed')
    meta=dict(status='backtest_verified_predictive_evaluation_pending',model_fits=0,model_predictions=4,
        candidate_rows=len(pred),signal_dates=len(schedule),portfolio_paths=len(stats),independent_ledger_days=checks,
        calendar_sha256=sha(CALENDAR_CONFIG),frozen_model_hashes=manifest,
        exceptions='User approved OTC quote execution proxy and excluded CVR rights; see event_registry.json',
        terminal_date=str(m['dates'].max().date()))
    (OUT/'verification.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('All 102 paths and six primary ledgers verified.',flush=True)
