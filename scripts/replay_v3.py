"""One sealed historical replay; old snapshots supply the four fixed comparator scores."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import final_v3 as final
from equity_research.backtest import engine,market,replay_v2 as replay,portfolio_v3
from equity_research.models.common import circular_block_interval
from equity_research.models.robustness_ledger import atomic_json,sha
from equity_research.models.context_selection import contexts
from equity_research.models.preprocessing import require

ROOT=final.ROOT
OUT=ROOT/"outputs/v3_seen_replay"
PUBLIC=ROOT/"results/v3_seen_replay"
KEY=["signal_date","PERMNO"]
V2="v2_enhanced_ridge_alpha_1000"


def align(frame,cached,cols):
    frame=frame.sort_values(KEY).reset_index(drop=True)
    cached=cached.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[KEY+["signal_eligible"]],cached[KEY+["signal_eligible"]],check_exact=True)
    for col in cols:
        frame[col]=cached[col]
    require(np.isfinite(frame.loc[frame.signal_eligible,cols]).all().all(),"Missing reference predictions")
    return frame


def evaluate():
    spec=final.locked()
    require(not OUT.exists() and not PUBLIC.exists(),"Historical replay already reserved")
    fit=ROOT/"outputs/v3_final_2021"
    verified=final.b.base.read_json(fit/"verification.json")
    require(verified["cutoff"]=="2021-12-31" if "cutoff" in verified else verified["training_end"]=="2021-12-31","Fit cutoff")
    require(sha(fit/"fit_ledger.json")==verified["ledger_sha256"] and sha(fit/"risk_loadings_local.parquet")==verified["pca_sha256"],"Changed final fit")
    ledger=final.b.base.read_json(fit/"fit_ledger.json")
    for rel,digest in ledger["entries"]["v3_final"]["artifacts"].items(): require(sha(fit/rel)==digest,"Final model changed")
    OUT.mkdir()
    atomic_json(OUT/"run.json",dict(status="started",protocol_sha256=sha(final.e.FINAL),period=spec["seen"]["period"]))
    cfg,_,_=final.b.plan(ROOT)
    reader=replay.ReplayReader()
    f,schedule,m,registry,terms=replay.inputs(reader,dict(original_features=cfg["original"]))
    frame=contexts(f)
    folder=fit/"v3_final"
    model=joblib.load(folder/"model.joblib")
    state=final.b.base.read_json(folder/"state.json")
    frame["v3_final"]=final.apply(frame,model,state,spec["signal"])
    bounds=[("signal_date",">=",schedule.signal_date.min()),("signal_date","<=",schedule.signal_date.max())]
    sources={}
    for dirname,cols in [("frozen_test_v1",final.V1+["momentum20"]),("historical_replay_v2",["ridge_enhanced"])]:
        file=ROOT/"outputs"/dirname/"predictions_local.parquet"
        x=pd.read_parquet(file,columns=KEY+["signal_eligible"]+cols,filters=bounds)
        frame=align(frame,x,cols)
        sources[file.relative_to(ROOT).as_posix()]=sha(file)
    frame=frame.rename(columns={"ridge_enhanced":V2})
    frame["annual_vol"]=frame.volatility_20
    models=["v3_final",*final.V1,V2,"momentum20"]
    frame[KEY+["signal_eligible"]+models].to_parquet(OUT/"predictions_local.parquet",index=False)
    limits=final.b.base.read_json(ROOT/"config/research_v2.json")["constraints"]
    pc=pd.read_parquet(fit/"risk_loadings_local.parquet")
    pcs={year:pc for year in range(2022,2026)}
    original=market.make_targets(frame,schedule,models=[*final.V1,"momentum20","equal_weight"])
    paths={"equal_weight":original["equal_weight"]}
    for name in models:
        if name=="v3_final" or name==V2:
            policy=spec["portfolio"] if name=="v3_final" else dict(risk="vol",penalty=0.)
            paths[name]=portfolio_v3.targets(frame,schedule,name,limits,policy,pcs)
        else: paths[name]=original[name]
        if name!="v3_final":
            paths[name+"__same_rule"]=portfolio_v3.targets(frame,schedule,name,limits,spec["portfolio"],pcs)
    results={};rows=[];years=[];checks=0
    for name,weights in paths.items():
        daily,orders,holdings,events=engine.simulate(m,weights,terms,10,2,False,True)
        checks+=engine.verify(daily,orders,holdings)
        if name=="equal_weight":
            old=pd.read_csv(ROOT/"outputs/frozen_test_v1/daily_portfolios.csv",parse_dates=["date"])
            ref=old.loc[old.model.eq(name)&old.scenario.eq("lag_2")&old.cost_bps.eq(10)].sort_values("date")
            np.testing.assert_array_equal(daily.date,ref.date)
            np.testing.assert_allclose(daily.nav,ref.nav,atol=1e-10,rtol=1e-10)
        daily.to_parquet(OUT/(name+"_daily.parquet"),index=False)
        results[name]=daily.set_index("date").return_net.iloc[1:]
        active=results[name]-results["equal_weight"]
        rows.append(dict(strategy=name,annual_active=active.mean()*252,**market.performance(daily)))
        for y,g in results[name].groupby(results[name].index.year):
            years.append(dict(year=y,strategy=name,net_return=(1+g).prod()-1,annualized_active=active.loc[g.index].mean()*252))
    labels=reader.read("data/processed/research_v1/features_labels.parquet",
                      KEY+["label_end_date","label_status","y_excess_5d"],schedule.signal_date.min(),replay.END)
    score=frame.merge(labels,on=KEY,validate="one_to_one")
    score=score.loc[score.signal_eligible & score.label_status.isin(["ok","ok_bridged"]) &
                    score.label_end_date.gt(score.signal_date)&np.isfinite(score.y_excess_5d)]
    mask=pd.Series(True,index=score.index)
    ics=pd.DataFrame({name:final.b.v2.daily_ic(score,score[name],mask) for name in models})
    require(ics.index.tolist()==schedule.signal_date.tolist(),"IC did not cover full signal calendar")
    best=ics[models[1:]].mean().idxmax()
    pair=circular_block_interval((ics.v3_final-ics[best]).to_numpy(),10,2000,20261110)
    PUBLIC.mkdir()
    pd.DataFrame(rows).to_csv(PUBLIC/"portfolios.csv",index=False)
    pd.DataFrame(years).to_csv(PUBLIC/"annual.csv",index=False)
    ics.groupby(ics.index.year).mean().T.assign(full_period=ics.mean()).to_csv(PUBLIC/"rank_ic.csv")
    atomic_json(PUBLIC/"verification.json",dict(status="completed_previously_seen_replay",signals=len(ics),qualified_rows=len(score),
        ledger_checks=int(checks),model_fit_end="2021-12-31",funding_reference="equal weight NAV reproduced",
        best_fixed_ic=best,ic_paired_difference=pair,protocol_sha256=sha(final.e.FINAL),
        events_policy="Frozen OTC quote proxies, nontradable CVRs excluded from investable NAV"))
    atomic_json(OUT/"run.json",dict(status="completed",access=reader.log,stored_prediction_sha256=sources,protocol_sha256=sha(final.e.FINAL)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    if p.parse_args().execute: evaluate()
