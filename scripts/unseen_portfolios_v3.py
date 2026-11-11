"""Portfolio route for a future independently verified 2026 market/settlement/event package."""
import argparse
from functools import partial
from pathlib import Path
import unseen_guard_v3 as guard
if __name__=="__main__":
    guard.seal()

import numpy as np
import pandas as pd

import unseen_v3 as u
from equity_research.backtest import engine,market,portfolio_v3
from equity_research.backtest.ledger import Ledger
from equity_research.backtest.settlement import SettlementCalendar
from equity_research.models.preprocessing import require
from equity_research.models.robustness_ledger import sha,atomic_json

ROOT=u.ROOT


def portfolio_set(frame,schedule,spec,pcs):
    limits=u.final.b.base.read_json(ROOT/"config/research_v2.json")["constraints"]
    paths=market.make_targets(frame,schedule,models=["equal_weight",*u.final.V1,"momentum20"])
    paths["v3_final"]=portfolio_v3.targets(frame,schedule,"v3_final",limits,spec["portfolio"],
                                            {y:pcs for y in schedule.signal_date.dt.year.unique()})
    paths[u.V2]=portfolio_v3.targets(frame,schedule,u.V2,limits,dict(risk="vol",penalty=0.),{})
    return paths


def portfolio_work(path,journal):
    spec=u.final.locked()
    inputs=u.final.b.base.read_json(path)
    pred=ROOT/"outputs/v3_unseen_2026_prediction"
    run=guard.Journal(ROOT,"prediction").record
    require(run.get("status")=="completed" and run["seal_sha256"]==journal.record["seal_sha256"],"Prediction run required")
    vintage=u.verified(path.parent,inputs["prediction_manifest"])
    require(run["manifest_sha256"]==sha(vintage),"Different prediction vintage")
    for name,digest in run["saved_files"].items():
        require(sha(pred/name)==digest,"Changed prediction artifact: "+name)
    audit=u.final.b.base.read_json(u.verified(path.parent,inputs["execution_audit"]))
    require(audit["unresolved_events"]==0 and audit["settlement_verified"] is True and audit["market_verified"] is True,
            "Event, market and settlement audit required")
    paths={k:u.verified(path.parent,audit[k]) for k in ["execution_market","settlement_calendar","event_registry"]}
    calendar=SettlementCalendar(paths["settlement_calendar"])
    dates=pd.DatetimeIndex(pd.read_parquet(pred/"calendar.parquet",columns=["date"]).date)
    dates=pd.DatetimeIndex(pd.to_datetime(dates))
    exchange=pd.DatetimeIndex([d for d in pd.bdate_range(u.START,u.END) if calendar.is_exchange_day(d)])
    pd.testing.assert_index_equal(dates[(dates>=u.START)&(dates<=u.END)],exchange,check_names=False)
    sched=u.signal_schedule(dates,u.START,u.END)
    dates=exchange[(exchange>=sched.signal_date.min())&(exchange<=sched.end_date.max())]
    for date in dates: calendar.settlement_date(date)
    cols=["PERMNO","DlyCalDt","DlyDelFlg","DlyPrc","DlyPrcFlg","DlyRet","DlyRetx","DlyCumFacPr",
          "DlyRetMissFlg","DlyRetDurFlg","sellable","buyable","ordinary_return","known_terminal"]
    raw=pd.read_parquet(paths["execution_market"],columns=cols,filters=[("DlyCalDt",">=",dates.min()),("DlyCalDt","<=",dates.max())])
    pd.testing.assert_index_equal(pd.DatetimeIndex(sorted(raw.DlyCalDt.unique())),dates,check_names=False)
    m=market.build_market(raw,dates,set(raw.PERMNO))
    m["price"]=raw.pivot(index="DlyCalDt",columns="PERMNO",values="DlyPrc").reindex(index=dates,columns=m["ids"]).to_numpy()
    events=u.final.b.base.read_json(paths["event_registry"])["events"]
    terms={r["permno"]:r for r in events}
    require(len(terms)==len(events),"Duplicate corporate action")
    for permno,r in terms.items():
        i=dates.get_loc(pd.Timestamp(r["event_date"]))
        j=np.flatnonzero(m["ids"]==permno)[0]
        require(i>0 and m["price"][i-1,j]>0,"Missing pre-event price")
        m["terminal"][i,j],m["ordinary"][i,j]=True,False
    features=pd.read_parquet(pred/"features_local.parquet",
        columns=u.KEY+["signal_eligible","industry","market_beta_60","volatility_20"],
        filters=[("signal_date","in",sched.signal_date.tolist())]).rename(columns={"volatility_20":"annual_vol"})
    out=ROOT/"outputs/v3_unseen_2026_portfolios"
    out.mkdir(exist_ok=journal.record["attempt"]>1)
    saved=dict(journal.record.get("paths",{}))
    rows=[]
    comparisons={}
    for year in [2025,2021]:
        forecast=pd.read_parquet(pred/(str(year)+"_predictions_local.parquet"))
        frame=forecast.merge(features,on=u.KEY+["signal_eligible"],how="outer",indicator=True,validate="one_to_one")
        require(frame._merge.eq("both").all(),"Forecast population mismatch")
        model_dir=ROOT/f"outputs/v3_final_{year}"
        pcfile=model_dir/"risk_loadings_local.parquet"
        require(sha(pcfile)==u.final.b.base.read_json(model_dir/"verification.json")["pca_sha256"],"Risk matrix changed")
        paths_by_model=portfolio_set(frame,sched,spec,pd.read_parquet(pcfile))
        ew=None
        primary_daily={}
        for model,weights in paths_by_model.items():
            name=str(year)+"_"+model+"_daily.parquet"
            if name in saved:
                require(sha(out/name)==saved[name],"Changed portfolio checkpoint")
                d=pd.read_parquet(out/name)
            else:
                d,o,h,_=engine.simulate(m,weights,terms,10,2,False,True,
                    ledger_factory=partial(Ledger,is_settlement_day=calendar.is_settlement_day),settlement_function=calendar.settlement_date)
                engine.verify(d,o,h,settlement_function=calendar.settlement_date)
                d.to_parquet(out/name,index=False)
                saved[name]=sha(out/name)
                journal.event("path_saved",paths=saved)
            ret=d.set_index("date").return_net.iloc[1:]
            if model=="equal_weight": ew=ret
            require(ew is not None and ret.index.equals(ew.index),"Funding benchmark mismatch")
            rows.append(dict(training_end=year,model=model,annual_active=252*(ret-ew).mean(),**market.performance(d)))
            if model in ["v3_final",u.final.V1[0]]:
                primary_daily[model]=ret
        difference=primary_daily["v3_final"]-primary_daily[u.final.V1[0]]
        comparisons[str(year)]=u.circular_block_interval(252*difference.to_numpy(),20,2000,20261110)
    pd.DataFrame(rows).to_csv(out/"portfolio_metrics.csv",index=False)
    atomic_json(out/"v3_minus_raw_ridge.json",comparisons)
    journal.event("completed",metrics_sha256=sha(out/"portfolio_metrics.csv"),
                  primary_sha256=sha(out/"v3_minus_raw_ridge.json"))


def execute(path,resume=False):
    with guard.attempt(ROOT,"portfolio",path,resume) as journal:
        portfolio_work(path,journal)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--audited-manifest",type=Path)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.audited_manifest: execute(args.audited_manifest,args.resume)
