"""Refit the frozen V3 residual ranker and, for the prospective cohort, fixed baselines."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import stage_e_v3 as e
from equity_research.models import stage_b_v3 as b
from equity_research.models.preprocessing import cross_sectional_rank,transform_features,frame_digest,require
from equity_research.models.context_selection import contexts
from equity_research.models.robustness_ledger import FitLedger,sha,atomic_json
from equity_research.backtest import portfolio_v3 as port,market
from equity_research.evaluation.exposures import sic_group

ROOT=e.ROOT
V1=["raw__ridge_03","cross_sectional_rank__lightgbm_10"]


def locked():
    require(sha(e.FINAL)==e.FINAL.with_suffix(".sha256").read_text().strip(),"Final V3 seal mismatch")
    spec=b.base.read_json(e.FINAL)
    for name,digest in spec["sources"].items():
        require(sha(ROOT/name)==digest,"Frozen source changed: "+name)
    return spec


def risk_history(frame,cutoff,source):
    """Use only members identified by the final available training snapshot."""
    cutoff=pd.Timestamp(cutoff)
    cal=pd.read_parquet(source/"calendar.parquet",columns=["date"],filters=[("date","<=",str(cutoff.date()))])
    dates=pd.DatetimeIndex(pd.to_datetime(cal.date)).sort_values()[-252:]
    require(len(dates)==252 and dates.is_unique,"PCA dates")
    ids=list(frame.loc[frame.signal_date.eq(frame.signal_date.max())&frame.signal_eligible,"PERMNO"])
    samples=[]
    for part in json.loads((source/"ingest.json").read_text())["parts"]:
        g=pd.read_parquet(source/part,columns=["DlyCalDt","PERMNO","DlyRet","DlyDelFlg","DlyRetMissFlg","DlyRetDurFlg"],
            filters=[("DlyCalDt",">=",dates.min()),("DlyCalDt","<=",cutoff),("PERMNO","in",ids)])
        valid=g.DlyDelFlg.eq("N")&g.DlyRetMissFlg.eq("NA")&g.DlyRetDurFlg.isin(market.SINGLE_DAY)&g.DlyRet.ge(-1)&np.isfinite(g.DlyRet)
        samples.append(g.loc[valid,["DlyCalDt","PERMNO","DlyRet"]])
    return port.pc_loadings(pd.concat(samples).pivot(index="DlyCalDt",columns="PERMNO",values="DlyRet").reindex(dates))


def load_frame(cutoff,cfg):
    require(cutoff in ["2021-12-31","2025-12-31"],"Unsupported training cutoff")
    data=ROOT/"data/processed/research_v1"
    source=ROOT/"data/interim/crsp_audit_v1"
    bounded=[("signal_date",">=",pd.Timestamp("2015-01-01")),("signal_date","<=",pd.Timestamp(cutoff))]
    f=pd.read_parquet(data/"features_only.parquet",columns=b.KEY+["signal_eligible"]+cfg["original"],filters=bounded)
    labels=pd.read_parquet(data/"features_labels.parquet",columns=b.KEY+["label_status","label_end_date","y_excess_5d"],
        filters=bounded+[("label_end_date","<=",pd.Timestamp(cutoff)),("label_status","in",["ok","ok_bridged"]),("signal_eligible","=",True)])
    require(labels.label_end_date.gt(labels.signal_date).all() and np.isfinite(labels.y_excess_5d).all(),"Invalid ordinary targets")
    groups=[]
    for part in json.loads((source/"ingest.json").read_text())["parts"]:
        g=pd.read_parquet(source/part,columns=["DlyCalDt","PERMNO","SICCD","SecInfoStartDt","SecInfoEndDt"],
                          filters=[("DlyCalDt","in",list(f.signal_date.unique())),("PERMNO","in",list(f.PERMNO.unique()))])
        groups.append(g)
    g=pd.concat(groups).rename(columns={"DlyCalDt":"signal_date"})
    g["industry"]=[sic_group(c,s,t,d) for c,s,t,d in zip(g.SICCD,g.SecInfoStartDt,g.SecInfoEndDt,g.signal_date)]
    f=f.merge(g[b.KEY+["industry"]],on=b.KEY,how="left",validate="one_to_one")
    f["industry"]=f.industry.fillna("Unknown")
    f=contexts(f.sort_values(b.KEY).reset_index(drop=True))
    f=f.merge(labels,on=b.KEY,how="left",validate="one_to_one")
    require(int(f.y_excess_5d.notna().sum())==len(labels),"Lost qualified label")
    f["role"]="train"
    if cutoff=="2021-12-31":
        # Strong parity check against the actual development snapshot prior to reserve.
        _,bundle,_=b.plan(ROOT)
        ref=pd.read_parquet(ROOT/bundle["plan"]["snapshot"],columns=b.KEY+["industry","signal_eligible"]+cfg["original"]+cfg["context"],filters=bounded)
        pd.testing.assert_frame_equal(f[ref.columns],ref.sort_values(b.KEY).reset_index(drop=True),check_dtype=False,check_exact=True)
    return f


def fit_one(f,name,cfg,cutoff,v1models,spec):
    window=dict(train=["2015-01-01",cutoff])
    if name=="v3_final":
        family=spec["signal"]
        target=spec["incumbents"][family]["target"]
        require(family=="ranker_enhanced" and target=="residual","Unexpected frozen recipe")
        view=b.training_view(f,target)
        model=lgb.LGBMRanker(**cfg["families"][family]["parameters"])
    else:
        view=f
        family=("ridge_enhanced" if name=="v2_enhanced_ridge_alpha_1000" else
                "ridge_original" if name==V1[0] else "lightgbm_original")
        if name in V1:
            params=v1models[name]["parameters"]
            model=Ridge(**params,fit_intercept=True) if name==V1[0] else lgb.LGBMRegressor(**params)
        else:
            model=Ridge(alpha=1000,solver="cholesky",fit_intercept=True)
    train=view.y_excess_5d.notna()
    state=b.base.build_state(view,train,cfg["original"],cfg["context"],family,window,cfg["preprocessing"])
    x=b.v2.check_state(state,view,train,cfg["original"],cfg["context"],family,window)
    b.v2.fit_candidate(model,dict(design=family),x,view,train,cutoff)
    return model,state,family,x


def apply(frame,model,state,family):
    require(not any(c.startswith(("y_","label_")) for c in frame),"Target in scoring frame")
    cols=state["base"]["features"]
    base=frame[cols] if family.startswith("ridge") else cross_sectional_rank(frame,cols)
    x=transform_features(base,state["base"])
    if family.endswith("enhanced"):
        ctx=state["context"]
        x=pd.concat([x,transform_features(frame[ctx["features"]],ctx)],axis=1)
    require(list(x)==state["columns"],"Final scoring feature order")
    return model.predict(x)


def run(cutoff):
    spec=locked()
    cfg,_,_=b.plan(ROOT)
    names=["v3_final"] if cutoff=="2021-12-31" else ["v3_final",*V1,"v2_enhanced_ridge_alpha_1000"]
    out=ROOT/("outputs/v3_final_"+cutoff[:4])
    require(not out.exists(),"Final fit directory exists")
    f=load_frame(cutoff,cfg)
    v1models={m["id"]:m for m in b.base.read_json(ROOT/"config/final_test_plan.json")["models"]}
    schedule=[dict(key=n,kind="fit") for n in names]
    manifest=dict(plan_sha256=sha(e.FINAL),data_digest=frame_digest(f),cutoff=cutoff,code_sha256=sha(Path(__file__)))
    pcs=risk_history(f,cutoff,ROOT/"data/interim/crsp_audit_v1")
    with FitLedger(out,manifest,schedule,len(schedule)) as ledger:
        pcs.to_parquet(out/"risk_loadings_local.parquet")
        for task in schedule:
            def fit(t):
                n=t["key"]
                model,state,family,x=fit_one(f,n,cfg,cutoff,v1models,spec)
                folder=out/n;folder.mkdir()
                joblib.dump(model,folder/"model.joblib")
                atomic_json(folder/"state.json",state)
                tail=f.loc[f.signal_date.eq(f.signal_date.max())]
                cols=b.KEY+cfg["original"]+cfg["context"]+["signal_eligible"]
                restored=apply(tail[cols],joblib.load(folder/"model.joblib"),state,family)
                np.testing.assert_allclose(restored,model.predict(x.loc[tail.index]),rtol=1e-12,atol=1e-12)
                atomic_json(folder/"summary.json",dict(family=family,fit_rows=state["base"]["fitted_rows"],
                                                     max_label_end=state["max_label_end"]))
                return [p.relative_to(out).as_posix() for p in folder.iterdir()]
            ledger.perform(task,fit)
        ledger.verify()
    atomic_json(out/"verification.json",dict(training_end=cutoff,fits=len(schedule),ledger_sha256=sha(out/"fit_ledger.json"),
        pca_sha256=sha(out/"risk_loadings_local.parquet"),momentum_missing=float(f.loc[f.y_excess_5d.notna(),"momentum_20"].median())))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fit-through",choices=["2021","2025"])
    args=parser.parse_args()
    if args.fit_through: run(args.fit_through+"-12-31")
