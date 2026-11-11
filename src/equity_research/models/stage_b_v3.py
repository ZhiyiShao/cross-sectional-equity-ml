"""Stage B: frozen training-target contrasts, scored only on the original five-day label."""
from pathlib import Path
import numpy as np
import pandas as pd

from equity_research.data import targets_v3
from equity_research.models import stage_a_v3 as stage_a
from equity_research.models import baseline_v3 as base
from equity_research.models import density_v3 as density
from equity_research.models import parameter_robustness as v2
from equity_research.models.robustness_ledger import FitLedger, atomic_json, sha

ROOT = base.ROOT
CONFIG = "config/stage_b_v3.json"
PIN = "83710de96c2ad3b85a15714c441b4f8f8a6d826c5e97f4ecea29e345973269cb"
RUN = "outputs/v3_stage_b_targets"
KEY = density.KEY


def plan(root):
    base.require(sha(root/CONFIG)==PIN,"Stage B protocol changed")
    cfg = base.read_json(root/CONFIG)
    for name,digest in cfg["source_sha256"].items():
        base.require(sha(root/name)==digest, "Source changed: "+name)
    bundle = base.load_plan(root)
    base.require(cfg["windows"]==bundle["windows"] and cfg["preprocessing"]==bundle["plan"]["preprocessing"],
                 "Control recipe drift")
    original,context = base.feature_lists(root)
    base.require(cfg["original"]==original and cfg["context"]==context and len(set(original+context))==33,
                 "Unexpected features")
    schedule = []
    for control in base.tasks(bundle):
        family = control["design"]
        if family not in cfg["families"]:
            continue
        pair = cfg["families"][family]
        base.require(control["candidate"]["id"]==pair["candidate_id"]
                     and v2.parameters(bundle["v2"],control)==pair["parameters"],"Control parameter drift")
        for target in cfg["target_order"]:
            treatment = family+"_"+target
            reuse = family=="ranker_enhanced" and target=="rank"
            schedule.append(dict(control, source=control["key"], key=control["window"]+"_"+treatment,
                                 treatment=treatment,target=target,kind="reuse" if reuse else "fit"))
    base.require(len(schedule)==36 and len({t["key"] for t in schedule})==36 and
                 sum(t["kind"]=="fit" for t in schedule)==cfg["budget"]["new_fits"]==32,"Schedule mismatch")
    return cfg,bundle,schedule


def multi_labels(path, window, horizons):
    filters = density.date_filters(window,"train")
    for h in horizons:
        filters += [(f"end_{h}","<=",base.period(window,"train")[1]),(f"status_{h}","in",["ok","ok_bridged"])]
    columns = KEY+[c for h in horizons for c in [f"end_{h}",f"excess_{h}"]]
    panel = pd.read_parquet(path,columns=columns,filters=filters)
    parts = []
    previous = panel.signal_date
    for h in horizons:
        end = panel[f"end_{h}"]
        base.require(end.gt(previous).all(),"Invalid horizon endpoint order")
        previous = end
        parts.append(panel[f"excess_{h}"]*(5/h))
    panel["multi"] = np.mean(np.stack(parts),axis=0)
    return panel.loc[np.isfinite(panel.multi),KEY+["multi","end_20"]]


def prepare_window(root,cfg,bundle,w,eval_labels):
    source = root/bundle["plan"]["snapshot"]
    features = KEY+["signal_eligible","industry"]+cfg["original"]+cfg["context"]
    pieces = []
    for role in ["train","evaluate"]:
        bounds = density.date_filters(w,role)
        x = pd.read_parquet(source,columns=features,filters=bounds)
        labels = density.targets(source,w,role) if role=="train" else eval_labels
        if role=="train":
            reference = density.targets(root/density.LABELS,w,role,five_only=True)
            pd.testing.assert_frame_equal(labels,reference,check_exact=True)
            multi = multi_labels(root/targets_v3.DIRECTORY/"labels.parquet",w,cfg["horizons"])
            x = x.merge(multi,on=KEY,how="left",validate="one_to_one")
        part = x.merge(labels,on=KEY,how="outer",indicator=True,validate="one_to_one")
        base.require(not part._merge.eq("right_only").any() and part.loc[part._merge.eq("both"),"signal_eligible"].all(),
                     "Label population mismatch")
        pieces.append(part.drop(columns="_merge").assign(role=role))
    return pd.concat(pieces,ignore_index=True).sort_values(KEY).reset_index(drop=True)


def training_view(frame,target):
    """Adapt the train rows for the existing fit adapter. Keep evaluation labels untouched."""
    f = frame.copy()
    train = f.role.eq("train") & f.y_excess_5d.notna()
    y = f.loc[train,"y_excess_5d"]
    if target=="multi":
        new = f.loc[train,"multi"]
        f.loc[train,"label_end_date"] = f.loc[train,"end_20"]
    elif target=="rank":
        g = y.groupby(f.loc[train,"signal_date"])
        n = g.transform("size")
        new = ((g.rank(method="average")-.5)/n-.5).where(n>=2)
    elif target=="residual":
        new = pd.Series(np.nan,index=y.index)
        for _,g in f.loc[train].groupby("signal_date"):
            cols = ["log_market_cap","market_beta_60"]
            g = g.loc[np.isfinite(g[cols]).all(axis=1)]
            if len(g)<2:
                continue
            X = np.column_stack([pd.get_dummies(g.industry.fillna("Unknown"),dtype=float),g[cols]])
            beta,_,rank,_ = np.linalg.lstsq(X,g.y_excess_5d,rcond=None)
            if len(g)-rank>=2:
                new.loc[g.index] = g.y_excess_5d-X@beta
    else:
        raise ValueError("Unknown target")
    f.loc[train,"y_excess_5d"] = new
    return f


def rank_equivalent(frame,view):
    raw = frame.loc[frame.role.eq("train") & frame.y_excess_5d.notna()]
    ranks = view.loc[view.role.eq("train") & view.y_excess_5d.notna()]
    pd.testing.assert_frame_equal(raw[KEY],ranks[KEY])
    raw_levels,raw_groups = v2.ranking_target(raw)
    levels,groups = v2.ranking_target(ranks)
    np.testing.assert_array_equal(raw_groups,groups)
    np.testing.assert_array_equal(raw_levels,levels)


def summarize(cfg, schedule, out):
    rows = [base.read_json(out/t["key"]/"summary.json") for t in schedule]
    # Same dates and bootstrap draws for all nine family/target comparisons.
    names = list(dict.fromkeys(t["treatment"] for t in schedule))
    intervals = stage_a.paired_intervals(out,dict(cfg,pairs=dict.fromkeys(names)),schedule)
    table = pd.DataFrame(rows)
    metrics,choices = {},{}
    for family in cfg["families"]:
        for target in cfg["target_order"]:
            key = family+"_"+target
            g = table.loc[table.treatment.eq(key)]
            metrics[key] = dict(mean_ic=float(g.rank_ic.mean()),control_ic=float(g.control_ic.mean()),
                                delta=float(g.delta.mean()),wins=int(g.delta.gt(0).sum()),ci95=intervals[key])
        best = max((family+"_"+t for t in cfg["target_order"]),key=lambda k:metrics[k]["mean_ic"])
        m = metrics[best]
        choices[family] = dict(winner=best,adopt=m["delta"]>0 and m["wins"]>=3)
    atomic_json(out/"comparison.json",dict(new_fits=32,reused=4,windows=rows,metrics=metrics,choices=choices))


def run(root,cfg,bundle,schedule,manifest,checked,resume=False):
    expected = {(w,d,t) for w in cfg["windows"] for d in cfg["families"] for t in cfg["target_order"]}
    base.require(len(schedule)==36 and {(t["window"],t["design"],t["target"]) for t in schedule}==expected
                 and all(t["kind"]==("reuse" if t["design"]=="ranker_enhanced" and t["target"]=="rank"
                                     else "fit") for t in schedule)
                 and sum(t["kind"]=="fit" for t in schedule)==32,"Unexpected budget")
    out = root/RUN
    with FitLedger(out,manifest,schedule,32,resume=resume) as ledger:
        for wid,w in cfg["windows"].items():
            pending = [t for t in schedule if t["window"]==wid and not ledger.completed(t["key"])]
            if not pending:
                continue
            labels,controls = checked[wid]
            frame = prepare_window(root,cfg,bundle,w,labels)
            for target in cfg["target_order"]:
                view = training_view(frame,target)
                for task in [t for t in pending if t["target"]==target]:
                    def operation(t):
                        control = controls[t["key"]]
                        if t["kind"]=="fit":
                            return stage_a.fit_one(out,t,view,dict(cfg,directional=[]),bundle,control)
                        rank_equivalent(frame,view)
                        folder = out/t["key"]
                        folder.mkdir()
                        pd.DataFrame(dict(candidate_ic=control,control_ic=control,delta=0.)).to_csv(folder/"paired_ic.csv")
                        atomic_json(folder/"summary.json",dict(window=wid,treatment=t["treatment"],control=t["source"],
                            parameters=v2.parameters(bundle["v2"],t),rank_ic=float(control.mean()),
                            control_ic=float(control.mean()),delta=0.,reuse_reason="Identical LambdaRank grades and queries"))
                        return [p.relative_to(out).as_posix() for p in folder.iterdir()]
                    ledger.perform(task,operation)
        ledger.verify()
        base.require(ledger.used==32 and len(ledger.state["entries"])==36,"Incomplete Stage B")
        summarize(cfg,schedule,out)


def main(*,execute=False,resume=False,check_inputs=False):
    base.require(not resume or execute,"Resume requires execution")
    cfg,bundle,schedule = plan(ROOT)
    if not (check_inputs or execute):
        return dict(tasks=36,new_fits_planned=32,reuse_planned=4,formal_fits_started=0)
    manifest = density.inputs(ROOT,bundle,schedule)
    folder = ROOT/targets_v3.DIRECTORY
    labels = base.read_json(folder/"manifest.json")
    base.require(labels["plan_sha256"]==PIN and labels["parent_sha256"]==sha(ROOT/density.LABELS)
                 and labels["builder_sha256"]==sha(Path(targets_v3.__file__))
                 and labels["label_sha256"]==sha(folder/"labels.parquet"),"Label provenance mismatch")
    manifest.update(stage_b_plan=PIN,labels=sha(folder/"manifest.json"),runner=sha(Path(__file__)))
    checked = {}
    for wid,w in cfg["windows"].items():
        tasks = [t for t in schedule if t["window"]==wid]
        # The three targets share the same five-day control within a family.
        unique = {t["design"]:t for t in tasks}
        labels,ics = density.evaluation(ROOT,bundle,w,list(unique.values()))
        checked[wid] = labels,{t["key"]:ics[unique[t["design"]]["key"]] for t in tasks}
        frame = prepare_window(ROOT,cfg,bundle,w,checked[wid][0])
        for target in cfg["target_order"]:
            view = training_view(frame,target)
            eligible = view.loc[view.role.eq("train") & view.y_excess_5d.notna()]
            base.require(len(eligible)>0 and eligible.label_end_date.le(pd.Timestamp(w["train"][1])).all(),
                         "Training label outside frozen window")
            v2.ranking_target(eligible)
            if target=="rank":
                rank_equivalent(frame,view)
    if execute:
        run(ROOT,cfg,bundle,schedule,manifest,checked,resume)
        return dict(status="completed",new_fits=32,reused=4)
    return dict(status="inputs_verified",formal_fits_started=0)
