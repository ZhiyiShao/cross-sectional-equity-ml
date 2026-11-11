"""Frozen stage C ensemble and neural baseline; default only reads its plan."""
import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from equity_research.models import stage_b_v3 as b
from equity_research.models.ensemble_v3 import fit_component,rank_mean
from equity_research.models.robustness_ledger import FitLedger,atomic_json,sha

ROOT=b.ROOT
PLAN="config/stage_c_v3.json"
RUN="outputs/v3_stage_c"


def incumbent(task,window,root=ROOT):
    family,target=task["family"],task["target"]
    if target=="ordinary":
        return root/b.density.CONTROL/(window+"_"+family+"_"+task["id"])
    return root/b.RUN/(window+"_"+family+"_"+target)


def read_scores(path):
    return pd.read_parquet(path/"predictions.parquet").sort_values(b.KEY).reset_index(drop=True)


def variants(scores,cfg):
    for name,recipe in cfg["variants"].items():
        scores[name]=rank_mean(scores,recipe["members"])
    return scores


def freeze():
    """Resolve the deterministic succession rule using completed stage B only."""
    plan_path=ROOT/PLAN
    pin=plan_path.with_suffix(".sha256")
    if plan_path.exists() or pin.exists():
        raise FileExistsError(plan_path)
    protocol,_,_=b.plan(ROOT)
    summary=b.base.read_json(ROOT/b.RUN/"comparison.json")
    sources={}; incumbents={}; means={}
    for family,spec in protocol["families"].items():
        choice=summary["choices"][family]
        score=summary["metrics"][choice["winner"]]
        target=choice["winner"].removeprefix(family+"_") if choice["adopt"] else "ordinary"
        if family=="ranker_enhanced" and target=="rank":
            target="ordinary"
        inc=dict(family=family,target=target,id=spec["candidate_id"])
        incumbents[family]=inc
        means[family]=score["mean_ic" if choice["adopt"] else "control_ic"]
        for wid in protocol["windows"]:
            p=incumbent(inc,wid)/"predictions.parquet"
            rel=p.relative_to(ROOT).as_posix()
            folder=p.parent.parent
            source=b.base.read_json(folder/"fit_ledger.json")["entries"][p.parent.name]
            assert source["status"]=="completed" and source["artifacts"][p.relative_to(folder).as_posix()]==sha(p)
            sources[rel]=sha(p)
    members={"mlp_seed11":dict(kind="mlp",family="ridge_enhanced",hidden=[32,16],epochs=6,
                                alpha=.001,learning_rate=.001,seed=11)}
    variants={"mlp":dict(members=["mlp_seed11"],baseline="ridge_enhanced")}
    for prefix,family in [("reg","lightgbm_enhanced"),("rank","ranker_enhanced")]:
        names=[]
        for seed in [11,29]:
            name=prefix+"_seed"+str(seed)
            members[name]=dict(kind="tree",family=family,seed=seed,feature_fraction=.8)
            names.append(name)
        variants[prefix+"_mean"]=dict(members=names,baseline=family)
    variants["consensus"]=dict(members=["mlp","reg_mean","rank_mean"],baseline=max(means,key=means.get))
    for rel in [PLAN.replace("stage_c","stage_b"),b.RUN+"/comparison.json",b.RUN+"/fit_ledger.json",
                "scripts/stage_c_v3.py","src/equity_research/models/ensemble_v3.py"]:
        sources[rel]=sha(ROOT/rel)
    cfg=dict(windows=protocol["windows"],members=members,variants=variants,budget=20,
             incumbents=incumbents,incumbent_means=means,sources=sources,
             selection="Among incumbents and candidates beating their named control in mean and >=3 years, take highest mean. Exact ties use incumbent insertion order, then variant insertion order. D cannot change this signal.")
    atomic_json(plan_path,cfg)
    pin.write_text(sha(plan_path)+"\n",encoding="ascii")


def execute(cfg,resume=False):
    protocol,bundle,_=b.plan(ROOT)
    manifest={PLAN:sha(ROOT/PLAN), "scripts/stage_c_v3.py":sha(Path(__file__)),
              "ensemble_module":sha(ROOT/"src/equity_research/models/ensemble_v3.py"),
              "B_ledger":sha(ROOT/b.RUN/"fit_ledger.json")}
    for rel,digest in cfg["sources"].items():
        b.base.require(sha(ROOT/rel)==digest,"C source changed: "+rel)
    # Recheck original feature/label inputs and stored comparator sources.
    _,_,all_tasks=b.plan(ROOT)
    manifest["inputs"]=b.density.inputs(ROOT,bundle,all_tasks)
    label_manifest=b.base.read_json(ROOT/b.targets_v3.DIRECTORY/"manifest.json")
    b.base.require(label_manifest["label_sha256"]==sha(ROOT/b.targets_v3.DIRECTORY/"labels.parquet"),"B labels changed")
    tasks=[dict(key=w+"_"+name,kind="fit",window=w,member=name) for w in cfg["windows"] for name in cfg["members"]]
    out=ROOT/RUN
    rows=[]; schedule=[]
    with FitLedger(out,manifest,tasks,cfg["budget"],resume=resume) as ledger:
        for wid,w in cfg["windows"].items():
            labels=b.density.targets(ROOT/b.density.LABELS,w,"evaluate",five_only=True)
            frame=b.prepare_window(ROOT,protocol,bundle,w,labels)
            ec=frame.role.eq("evaluate")
            scores=frame.loc[ec,b.KEY+["signal_eligible"]].reset_index(drop=True)
            for family,inc in cfg["incumbents"].items():
                folder=incumbent(inc,wid)
                p=read_scores(folder)
                pd.testing.assert_frame_equal(scores[b.KEY+["signal_eligible"]],p[b.KEY+["signal_eligible"]])
                scores[family]=p.prediction
            for t in [t for t in tasks if t["window"]==wid]:
                def operation(t):
                    name=t["member"];r=cfg["members"][name];family=r["family"]
                    target=cfg["incumbents"][family]["target"]
                    view=frame if target=="ordinary" else b.training_view(frame,target)
                    model,state,matrix=fit_component(view,r,family,bundle,protocol,w)
                    pred=model.predict(matrix.loc[ec].to_numpy() if r["kind"]=="mlp" else matrix.loc[ec])
                    folder=out/t["key"];folder.mkdir()
                    joblib.dump(model,folder/"model.joblib")
                    atomic_json(folder/"preprocessing.json",state)
                    scores[b.KEY+["signal_eligible"]].assign(prediction=pred).to_parquet(folder/"predictions.parquet",index=False)
                    return [p.relative_to(out).as_posix() for p in folder.iterdir()]
                ledger.perform(t,operation)
                scores[t["member"]]=read_scores(out/t["key"]).prediction
            scores=variants(scores,cfg)
            q=scores.merge(labels,on=b.KEY,validate="one_to_one")
            def daily(column):
                return b.v2.daily_ic(q,q[column],pd.Series(True,index=q.index))
            for name,variant in cfg["variants"].items():
                comp=variant["baseline"];ic=daily(name);control=daily(comp)
                folder=out/(wid+"_"+name);folder.mkdir(exist_ok=resume)
                scores[b.KEY+["signal_eligible"]].assign(prediction=scores[name]).to_parquet(folder/"predictions.parquet",index=False)
                pd.DataFrame(dict(candidate_ic=ic,control_ic=control,delta=ic-control)).to_csv(folder/"paired_ic.csv")
                rows.append(dict(window=wid,model=name,baseline=comp,ic=ic.mean(),control_ic=control.mean(),delta=(ic-control).mean()))
                schedule.append(dict(window=wid,treatment=name,key=folder.name))
        ledger.verify()
        b.base.require(ledger.used==len(tasks)==cfg["budget"],"C budget mismatch")
    table=pd.DataFrame(rows)
    intervals=b.stage_a.paired_intervals(out,dict(protocol,pairs=cfg["variants"]),schedule)
    summary=[]
    for name,g in table.groupby("model",sort=False):
        summary.append(dict(model=name,ic=g.ic.mean(),control_ic=g.control_ic.mean(),delta=g.delta.mean(),
                            wins=int(g.delta.gt(0).sum()),ci95=intervals[name],adopt=bool(g.delta.mean()>0 and sum(g.delta>0)>=3)))
    accepted=dict(cfg["incumbent_means"])
    accepted.update({s["model"]:s["ic"] for s in summary if s["adopt"]})
    public=ROOT/"results/v3_stage_c";public.mkdir()
    table.to_csv(public/"windows.csv",index=False)
    atomic_json(public/"comparison.json",dict(new_fits=len(tasks),summary=summary,accepted=accepted,
                                             selected=max(accepted,key=accepted.get),plan_sha256=sha(ROOT/PLAN)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group()
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.freeze:
        freeze()
    cfg=b.base.read_json(ROOT/PLAN)
    if sha(ROOT/PLAN)!=(ROOT/"config/stage_c_v3.sha256").read_text().strip():
        raise ValueError("Frozen C plan changed")
    if args.execute:
        execute(cfg,args.resume)
    else:
        print(cfg["budget"],"planned fits; none executed")
