"""One prospective prediction evaluation from a reviewed vintage and dated inputs."""
import argparse
import json
from pathlib import Path
import unseen_guard_v3 as guard
if __name__=="__main__":
    guard.seal()
import joblib
import numpy as np
import pandas as pd

import final_v3 as final
from equity_research.models.common import load_lightgbm,circular_block_interval
from equity_research.models.preprocessing import cross_sectional_rank,transform_features,require
from equity_research.models.context_selection import contexts
from equity_research.models.robustness_ledger import sha,atomic_json
from equity_research.data.date_gate import signal_schedule
from unseen_inputs_v3 import Sources,build_features,panel

ROOT=final.ROOT
KEY=final.b.KEY
START,END=map(pd.Timestamp,["2026-01-01","2026-12-31"])
V2="v2_enhanced_ridge_alpha_1000"


def verified(root,record,key="path",digest="sha256"):
    path=root/record[key]
    require(sha(path)==record[digest],"Model or input seal mismatch")
    return path


def score(f,year):
    require(not any(c.startswith(("label_","y_")) for c in f),"Future label in features")
    f=contexts(f.sort_values(KEY).reset_index(drop=True))
    predictions=f[KEY+["signal_eligible"]].copy()
    folder=ROOT/f"outputs/v3_final_{year}"
    official=final.b.base.read_json(folder/"verification.json")
    require(sha(folder/"fit_ledger.json")==official["ledger_sha256"],"Fit ledger changed")
    ledger=final.b.base.read_json(folder/"fit_ledger.json")
    for n,entry in ledger["entries"].items():
        for rel,digest in entry["artifacts"].items():
            require(sha(folder/rel)==digest,"Final estimator changed")
        p=folder/n
        family=final.b.base.read_json(p/"summary.json")["family"]
        predictions[n]=final.apply(f,joblib.load(p/"model.joblib"),final.b.base.read_json(p/"state.json"),family)
    momentum=official["momentum_missing"]
    if year==2021:
        b1=final.b.base.read_json(ROOT/"outputs/final_models_v1/model_bundle.json")
        b2=final.b.base.read_json(ROOT/"outputs/final_models_v2/model_bundle.json")
        orig=b1["features"]
        for entry in b1["models"]:
            name=entry["id"]
            if name not in final.V1: continue
            state=final.b.base.read_json(verified(ROOT,entry,"preprocessing","preprocessing_sha256"))
            path=verified(ROOT,entry)
            m=joblib.load(path) if name==final.V1[0] else load_lightgbm(path)
            x=f[orig] if name==final.V1[0] else cross_sectional_rank(f,orig)
            predictions[name]=m.predict(transform_features(x,state))
        entry=next(r for r in b2["models"] if r["id"]=="ridge_enhanced")
        raw,ctx=[final.b.base.read_json(verified(ROOT,b2["preprocessing"][k])) for k in ["raw","context"]]
        x=pd.concat([transform_features(f[s["features"]],s) for s in [raw,ctx]],axis=1)
        predictions[V2]=joblib.load(verified(ROOT,entry)).predict(x)
        momentum=final.b.base.read_json(ROOT/b1["baseline_state"])["momentum_missing"]
    predictions["momentum20"]=f.momentum_20.replace([np.inf,-np.inf],np.nan).fillna(momentum)
    return predictions


def evaluate_scores(f,forecasts,labels,schedule):
    dates=schedule.signal_date.tolist()
    audit=f[KEY].merge(labels,on=KEY,how="outer",indicator=True,validate="one_to_one")
    require(audit._merge.eq("both").all(),"Missing prospective target status or extra keys")
    expected=labels.signal_date.map(schedule.set_index("signal_date").end_date)
    require(labels.label_end_date.eq(expected).all(),"Five-day label endpoint mismatch")
    summary={}
    for year,p in forecasts.items():
        g=p.merge(labels,on=KEY,validate="one_to_one")
        mask=g.signal_eligible & g.label_status.isin(["ok","ok_bridged"]) & np.isfinite(g.y_excess_5d)
        names=["v3_final",*final.V1,V2,"momentum20"]
        ic=pd.DataFrame({n:final.b.v2.daily_ic(g,g[n],mask) for n in names})
        require(ic.index.tolist()==dates,"Missing prospective IC dates")
        best=ic[names[1:]].mean().idxmax()
        primary=final.V1[0]
        summary[str(year)]=dict(mean_ic=ic.mean().to_dict(),primary_control=primary,
            v3_minus_raw_ridge=circular_block_interval((ic.v3_final-ic[primary]).to_numpy(),10,2000,20261110),
            best_fixed=best,
            interval=circular_block_interval((ic.v3_final-ic[best]).to_numpy(),10,2000,20261110),dates=len(ic))
    return summary


def approved_source(file):
    from check_v3_vintage import overlap_digest,projected
    records=json.loads(file.read_text(encoding="utf8"))
    overlap=ROOT/"outputs/v3_unseen_2026_vintage/overlap.json"
    require(sha(overlap)==records["overlap_sha256"] and records["vintage_decision"]=="old_through_2025_new_from_2026"
            and bool(records["decision_reason"].strip()),"2025 overlap decision required")
    require(records["data_through"]=="2026-12-31" and records["cohort"]=="CRSP_US_equities_2026","Cohort mismatch")
    parts=[verified(file.parent,r) for r in records["parts"]]
    require(len(parts)>0 and len(set(p.resolve() for p in parts))==len(parts),"Invalid part list")
    checked=final.b.base.read_json(overlap)
    require(checked["new_overlap_sha256"]==overlap_digest(pd.concat([projected(p) for p in parts])),
            "Input parts differ from reviewed 2025 overlap")
    calendar=verified(file.parent,records["new_calendar"])
    return Sources(ROOT,START,END,(parts,calendar))


def prediction_work(journal,out,source,cfg):
    saved=journal.record.get("saved_files")
    if saved is None:
        require(not journal.record.get("labels_exposed"),"Predictions must survive label access")
        if journal.record["attempt"]==1: out.mkdir()
        else: out.mkdir(exist_ok=True)
        picks,f=build_features(source,cfg)
        picks.to_parquet(out/"monthly_universe_local.parquet",index=False)
        f.to_parquet(out/"features_local.parquet",index=False)
        pd.DataFrame({"date":source.market.index}).to_parquet(out/"calendar.parquet",index=False)
        names=["monthly_universe_local.parquet","features_local.parquet","calendar.parquet"]
        for year in [2025,2021]:
            name=str(year)+"_predictions_local.parquet"
            score(f,year).to_parquet(out/name,index=False)
            names.append(name)
        saved={name:sha(out/name) for name in names}
        journal.event("predictions_saved",saved_files=saved)
    for name,digest in saved.items():
        require(sha(out/name)==digest,"Prediction checkpoint changed: "+name)
    f=pd.read_parquet(out/"features_local.parquet",columns=KEY+["signal_eligible"])
    forecasts={y:pd.read_parquet(out/(str(y)+"_predictions_local.parquet")) for y in [2025,2021]}
    journal.event("labels_opened",labels_exposed=True)
    labels=panel(source,f[KEY],cfg,labels=True)
    summary=evaluate_scores(f,forecasts,labels,source.schedule)
    atomic_json(out/"prediction_summary.json",summary)
    journal.event("completed",summary_sha256=sha(out/"prediction_summary.json"))


def execute(manifest_path,resume=False):
    file=Path(manifest_path)
    with guard.attempt(ROOT,"prediction",file,resume) as journal:
        final.locked()
        source=approved_source(file)
        cfg=final.b.base.read_json(ROOT/"config/universe.json")
        prediction_work(journal,ROOT/"outputs/v3_unseen_2026_prediction",source,cfg)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--audited-manifest",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.audited_manifest: execute(args.audited_manifest,args.resume)
