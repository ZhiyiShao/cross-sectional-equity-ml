"""Stage 0b: daily training with the fixed five-day 0a evaluation population."""
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from equity_research.models import baseline_v3 as base
from equity_research.models import parameter_robustness as v2
from equity_research.models.robustness_ledger import FitLedger, atomic_json, sha

ROOT = base.ROOT
KEY = ["signal_date", "PERMNO"]
CONFIG = "config/density_0b.json"
CONFIG_SHA = "f492de7a08c0961bbff61808af5123a745183c8f621bc5a2aa6a10eaf73bcf22"
PANEL = "data/processed/research_v3/context_0b"
LABELS = "data/processed/research_v3/features_labels.parquet"
CONTROL = "outputs/v3_baseline"
RUN = "outputs/v3_density_0b"
BUDGET = 8


def plan(root):
    base.require(sha(root/CONFIG) == CONFIG_SHA, "Frozen density rules changed")
    bundle = base.load_plan(root)
    controls = base.read_json(root/CONFIG)["paired_controls"]
    schedule = []
    for task in base.tasks(bundle):
        if task["design"] in controls:
            base.require(task["candidate"]["id"] == controls[task["design"]], "Control parameter mismatch")
            schedule.append(dict(task, kind="fit", source=task["key"]))
    base.require(len(schedule) == BUDGET and len({t["key"] for t in schedule}) == BUDGET, "Invalid 0b budget")
    return bundle, schedule


def date_filters(window, role):
    start, end = base.period(window, role)
    ts, te = base.period(window, "train")
    es, ee = base.period(window, "evaluate")
    base.require(pd.Timestamp("2015-01-01") <= ts <= te < es <= ee <= pd.Timestamp("2021-12-31"),
                 "Window outside development interval or overlapping")
    return [("signal_date", ">=", start), ("signal_date", "<=", end)]


def targets(path, window, role, *, five_only=False):
    filters = date_filters(window, role)
    if five_only:
        filters.append(("is_v1_signal", "=", True))
    filters += [("label_end_date", "<=", base.period(window, role)[1]), ("signal_eligible", "=", True),
                ("label_status", "in", ["ok", "ok_bridged"])]
    columns = KEY+["label_end_date","label_status","y_excess_5d"]
    f = pd.read_parquet(path, columns=columns, filters=filters)
    f = f.loc[f.label_end_date.gt(f.signal_date) & np.isfinite(f.y_excess_5d)]
    base.require(len(f) > 0 and not f.duplicated(KEY).any(), "Empty or duplicate target keys")
    return f.sort_values(KEY).reset_index(drop=True)


def evaluation(root, bundle, window, tasks):
    """Validate candidates, score eligibility and paired dates before reserving any fit."""
    labels = targets(root/LABELS, window, "evaluate", five_only=True)
    reference = targets(root/bundle["plan"]["snapshot"], window, "evaluate")
    pd.testing.assert_frame_equal(labels, reference, check_exact=True)
    filters = date_filters(window, "evaluate") + [("is_v1_signal","=",True)]
    keys = pd.read_parquet(root/PANEL/"daily_features.parquet", columns=KEY+["signal_eligible"], filters=filters)
    keys = keys.sort_values(KEY).reset_index(drop=True)
    base.require(not keys.duplicated(KEY).any(), "Duplicate evaluation candidate")
    control_ics = {}
    for task in tasks:
        folder = root/CONTROL/task["source"]
        pred = pd.read_parquet(folder/"predictions.parquet").sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(keys, pred[keys.columns], check_exact=True)
        scored = labels.merge(pred, on=KEY, how="left", validate="one_to_one")
        base.require(scored.signal_eligible.eq(True).all() and np.isfinite(scored.prediction).all(),
                     "Missing qualified control prediction")
        ic = v2.daily_ic(scored, scored.prediction, pd.Series(True,index=scored.index))
        saved = pd.read_csv(folder/"daily_ic.csv",parse_dates=["signal_date"]).set_index("signal_date").rank_ic
        pd.testing.assert_index_equal(ic.index,saved.index)
        np.testing.assert_allclose(ic,saved,atol=1e-12,rtol=1e-12)
        control_ics[task["key"]] = ic
    return labels, control_ics


def training_frame(root, window, columns, eval_labels):
    pieces = []
    for role in ["train","evaluate"]:
        filters = date_filters(window, role)
        if role == "evaluate":
            filters.append(("is_v1_signal","=",True))
        x = pd.read_parquet(root/PANEL/"daily_features.parquet",columns=KEY+["signal_eligible"]+columns,filters=filters)
        y = targets(root/LABELS, window, role) if role == "train" else eval_labels
        part = x.merge(y,on=KEY,how="outer",indicator=True,validate="one_to_one")
        base.require(not part._merge.eq("right_only").any(), "Unexpected label key")
        base.require(part.loc[part._merge.eq("both"),"signal_eligible"].all(), "Eligibility mismatch")
        pieces.append(part.drop(columns="_merge").assign(role=role))
    frame = pd.concat(pieces,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    return frame


def run_windows(root, out, bundle, schedule, manifest, checked, original, added, *, resume=False):
    """One durable reservation per call to a fixed estimator; single window in memory."""
    expected = {(wid, design) for wid in bundle["windows"] for design in ["ridge_enhanced","lightgbm_enhanced"]}
    base.require(len(schedule) == BUDGET and {(t["window"], t["design"]) for t in schedule} == expected
                 and all(t["kind"] == "fit" for t in schedule), "Incomplete or extra 0b tasks")
    with FitLedger(out, manifest, schedule, BUDGET, resume=resume) as ledger:
        for wid, window in bundle["windows"].items():
            pending = [t for t in schedule if t["window"] == wid and not ledger.completed(t["key"])]
            if not pending:
                continue
            labels, controls = checked[wid]
            frame = training_frame(root, window, original+added, labels)
            train = frame.role.eq("train") & frame.y_excess_5d.notna()
            candidates = frame.role.eq("evaluate")
            evaluate = candidates & frame.y_excess_5d.notna()

            def fit(task):
                state = base.build_state(frame, train, original, added, task["design"], window,
                                         bundle["plan"]["preprocessing"])
                matrix = v2.check_state(state, frame, train, original, added, task["design"], window)
                model = v2.new_estimator(bundle["v2"], task)
                v2.fit_candidate(model, task, matrix, frame, train, window["train"][1])
                pred = np.full(len(frame), np.nan)
                pred[candidates] = model.predict(matrix.loc[candidates])
                base.require(np.isfinite(pred[candidates]).all(), "Nonfinite prediction")
                ic = v2.daily_ic(frame, pred, evaluate)
                paired = controls[task["key"]]
                pd.testing.assert_index_equal(ic.index, paired.index)
                folder = out/task["key"]
                folder.mkdir()
                if task["grid"] == "ridge":
                    joblib.dump(model, folder/"model.joblib")
                else:
                    (folder/"model.txt").write_text(model.booster_.model_to_string(),encoding="utf8")
                atomic_json(folder/"preprocessing.json",state)
                frame.loc[candidates,KEY+["signal_eligible"]].assign(prediction=pred[candidates]).to_parquet(
                    folder/"predictions.parquet",index=False)
                pd.DataFrame(dict(daily_train_ic=ic,control_ic=paired,delta=ic-paired)).to_csv(folder/"paired_ic.csv")
                atomic_json(folder/"summary.json",dict(window=wid, design=task["design"],
                    candidate=task["candidate"]["id"], parameters=v2.parameters(bundle["v2"],task),
                    train_interval=window["train"], eval_interval=window["evaluate"],train_rows=int(train.sum()),
                    max_train_label_end=str(frame.loc[train,"label_end_date"].max()), eval_rows=int(evaluate.sum()),
                    max_eval_label_end=str(labels.label_end_date.max()), dates=len(ic), rank_ic=float(ic.mean()),
                    control_ic=float(paired.mean()), delta=float((ic-paired).mean())))
                return [p.relative_to(out).as_posix() for p in folder.iterdir()]

            for task in pending:
                ledger.perform(task, fit)
        ledger.verify()
        base.require(ledger.used == BUDGET and len(ledger.state["entries"]) == BUDGET,"Incomplete fit budget")
        rows = [base.read_json(out/t["key"]/"summary.json") for t in schedule]
        means = pd.DataFrame(rows).groupby("design")[["rank_ic","control_ic","delta"]].mean()
        atomic_json(out/"comparison.json", dict(new_fits=BUDGET,windows=rows,equal_window=means.to_dict("index")))


def inputs(root, bundle, schedule):
    parent = root/PANEL/"manifest.json"
    base.require(sha(parent) == "3007bf20f849930ca7cbb94953a62751c1a1e7b611234655ea131b66345d21ca",
                 "0b preparation manifest changed")
    hashes = base.read_json(parent)["files"].copy()
    ledger_path = root/CONTROL/"fit_ledger.json"
    base.require(sha(ledger_path) == "09b39f8362e595e87760b894aa141eb78c54880ffe48987c6560d9302663a72c",
                 "0a ledger changed")
    ledger = base.read_json(ledger_path)
    versions = {n:version(n) for n in ledger["manifest"]["versions"]}
    base.require(versions == ledger["manifest"]["versions"], "Environment differs from 0a")
    snap = bundle["plan"]["snapshot"]
    hashes[snap] = ledger["manifest"]["inputs_and_code"][snap]
    for name in base.CODE:
        hashes[name] = ledger["manifest"]["inputs_and_code"][name]
    for task in schedule:
        entry = ledger["entries"][task["source"]]
        base.require(entry["status"] == "completed","Incomplete 0a control")
        for name in ["predictions.parquet","daily_ic.csv"]:
            rel = task["source"]+"/"+name
            hashes[CONTROL+"/"+rel] = entry["artifacts"][rel]
    for name, digest in hashes.items():
        base.require(sha(root/name) == digest, "0b input changed: "+name)
    for name in ["src/equity_research/models/density_v3.py","scripts/density_v3.py"]:
        hashes[name] = sha(root/name)
    return dict(files=hashes, versions=versions, density_rules_sha256=CONFIG_SHA,baseline_plan_sha256=base.PLAN_SHA)


def main(*, execute=False, check_inputs=False, resume=False):
    base.require(not resume or execute,"Resume requires explicit execution")
    bundle, schedule = plan(ROOT)
    result = dict(tasks=len(schedule),fit_budget=BUDGET,formal_fits_started=0)
    if not (execute or check_inputs):
        return result
    manifest = inputs(ROOT,bundle,schedule)
    checked = {wid:evaluation(ROOT,bundle,w,[t for t in schedule if t["window"]==wid])
               for wid,w in bundle["windows"].items()}
    result["qualified_five_day_rows"] = {wid:len(labels) for wid,(labels,_) in checked.items()}
    if execute:
        run_windows(ROOT,ROOT/RUN,bundle,schedule,manifest,checked,*base.feature_lists(ROOT),resume=resume)
        return dict(status="completed",budget_used=BUDGET)
    return result
