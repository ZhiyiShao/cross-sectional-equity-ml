"""Frozen five-day Stage A experiment: three paired comparisons, four windows."""
import joblib
import numpy as np
import pandas as pd

from equity_research.models import baseline_v3 as base
from equity_research.models import density_v3 as density
from equity_research.models import parameter_robustness as v2
from equity_research.models.robustness_ledger import FitLedger, atomic_json, sha

ROOT = base.ROOT
CONFIG = "config/stage_a_v3.json"
CONFIG_SHA = "2c68fe465d5c59194a23378e58b0a53c6f3cacfc565b5eecaf71576474f4e633"
RUN = "outputs/v3_stage_a_directional"
KEY = density.KEY


def frozen_plan(root):
    base.require(sha(root / CONFIG) == CONFIG_SHA, "Stage A configuration changed")
    cfg = base.read_json(root / CONFIG)
    for path, expected in cfg["sources"].items():
        base.require(sha(root / path) == expected, f"Stage A source changed: {path}")
    bundle = base.load_plan(root)
    screen = base.read_json(root / "results/stage_a_screen_v3/screen.json")
    retained = [row["feature"] for row in screen["decisions"] if row["decision"] == "retain"]
    base.require(retained == cfg["directional"] and len(retained) == 5, "Screen mismatch")
    original, context = base.feature_lists(root)
    base.require((original, context) == (cfg["original"], cfg["context"])
                 and len(set(original + context + retained)) == 38, "Feature block mismatch")
    base.require(cfg["windows"] == bundle["windows"] and cfg["preprocessing"] == bundle["plan"]["preprocessing"],
                 "Baseline window or preprocessing drift")
    schedule = []
    for task in base.tasks(bundle):
        for candidate, pair in cfg["pairs"].items():
            if task["design"] != pair["control"]:
                continue
            base.require(task["candidate"]["id"] == pair["candidate_id"]
                         and v2.parameters(bundle["v2"], task) == pair["parameters"], "Estimator mismatch")
            schedule.append(dict(task, kind="fit", source=task["key"], key=f"{task['window']}_{candidate}",
                                 treatment=candidate))
    base.require(len({t["key"] for t in schedule}) == len(schedule) == cfg["budget"]["new_fits"] == 12,
                 "Stage A budget mismatch")
    return cfg, bundle, schedule


def window_frame(root, cfg, bundle, window, evaluation_labels):
    """Use the original 0a feature rows; attach the five new columns on exactly those keys."""
    source = root / bundle["plan"]["snapshot"]
    old_columns = KEY + ["signal_eligible"] + cfg["original"] + cfg["context"]
    pieces = []
    for role in ("train", "evaluate"):
        bounds = density.date_filters(window, role)
        original = pd.read_parquet(source, columns=old_columns, filters=bounds).sort_values(KEY).reset_index(drop=True)
        extra = pd.read_parquet(root / density.PANEL / "daily_features.parquet",
                                columns=KEY + ["signal_eligible"] + cfg["directional"],
                                filters=bounds + [("is_v1_signal", "=", True)]).sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(original[KEY + ["signal_eligible"]], extra[KEY + ["signal_eligible"]],
                                      check_exact=True)
        original[cfg["directional"]] = extra[cfg["directional"]]
        if role == "train":
            labels = density.targets(root / density.LABELS, window, role, five_only=True)
            pd.testing.assert_frame_equal(labels, density.targets(source, window, role), check_exact=True)
        else:
            labels = evaluation_labels
        frame = original.merge(labels, on=KEY, how="outer", validate="one_to_one", indicator=True)
        base.require(not frame._merge.eq("right_only").any() and
                     frame.loc[frame._merge.eq("both"), "signal_eligible"].all(), "Label keys or eligibility differ")
        pieces.append(frame.drop(columns="_merge").assign(role=role))
    return pd.concat(pieces, ignore_index=True).sort_values(KEY).reset_index(drop=True)


def fit_one(out, task, frame, cfg, bundle, control):
    window = cfg["windows"][task["window"]]
    train = frame.role.eq("train") & frame.y_excess_5d.notna()
    candidates = frame.role.eq("evaluate")
    evaluate = candidates & frame.y_excess_5d.notna()
    added = cfg["context"] + cfg["directional"]
    state = base.build_state(frame, train, cfg["original"], added, task["design"], window, cfg["preprocessing"])
    matrix = v2.check_state(state, frame, train, cfg["original"], added, task["design"], window)
    model = v2.new_estimator(bundle["v2"], task)
    v2.fit_candidate(model, task, matrix, frame, train, window["train"][1])
    pred = np.full(len(frame), np.nan)
    pred[candidates] = model.predict(matrix.loc[candidates])
    base.require(np.isfinite(pred[candidates]).all(), "Nonfinite prediction")
    ic = v2.daily_ic(frame, pred, evaluate)
    pd.testing.assert_index_equal(ic.index, control.index)
    folder = out / task["key"]
    folder.mkdir()
    if task["grid"] == "ridge":
        joblib.dump(model, folder / "model.joblib")
    else:
        (folder / "model.txt").write_text(model.booster_.model_to_string(), encoding="utf8")
    atomic_json(folder / "preprocessing.json", state)
    frame.loc[candidates, KEY + ["signal_eligible"]].assign(prediction=pred[candidates]).to_parquet(
        folder / "predictions.parquet", index=False)
    delta = ic - control
    pd.DataFrame(dict(candidate_ic=ic, control_ic=control, delta=delta)).to_csv(folder / "paired_ic.csv")
    atomic_json(folder / "summary.json", dict(window=task["window"], treatment=task["treatment"],
                control=task["source"], parameters=v2.parameters(bundle["v2"], task),
                train_rows=int(train.sum()), eval_rows=int(evaluate.sum()), dates=len(ic),
                max_train_label_end=str(frame.loc[train,"label_end_date"].max()),
                max_eval_label_end=str(frame.loc[evaluate,"label_end_date"].max()),
                rank_ic=float(ic.mean()), control_ic=float(control.mean()), delta=float(delta.mean())))
    return [p.relative_to(out).as_posix() for p in folder.iterdir()]


def paired_intervals(out, cfg, schedule):
    """Share circular block draws across families within each evaluation year."""
    spec = cfg["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    draws = []
    names = list(cfg["pairs"])
    for wid in cfg["windows"]:
        columns = {}
        for task in (t for t in schedule if t["window"] == wid):
            series = pd.read_csv(out / task["key"] / "paired_ic.csv", index_col="signal_date").delta
            base.require(series.index.is_unique and series.index.is_monotonic_increasing, "Invalid IC dates")
            columns[task["treatment"]] = series
        values = pd.DataFrame(columns)[names].to_numpy()
        base.require(np.isfinite(values).all(), "Incomplete paired dates")
        n = len(values)
        block = spec["block_signal_dates"]
        starts = rng.integers(n, size=(spec["replications"], (n + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(spec["replications"], -1)[:, :n]
        draws.append(values[indices].mean(axis=1))
    bounds = np.quantile(np.mean(draws, axis=0), [.025, .975], axis=0)
    return dict(zip(names, bounds.T.tolist()))


def run(root, cfg, bundle, schedule, manifest, checked, *, resume=False):
    out = root / RUN
    budget = cfg["budget"]["new_fits"]
    expected = {(w,c) for w in cfg["windows"] for c in cfg["pairs"]}
    base.require(len(schedule) == budget == 12 and {(t["window"],t["treatment"]) for t in schedule} == expected
                 and all(t["kind"] == "fit" for t in schedule), "Incomplete or extra Stage A tasks")
    with FitLedger(out, manifest, schedule, budget, resume=resume) as ledger:
        for wid, window in cfg["windows"].items():
            pending = [t for t in schedule if t["window"] == wid and not ledger.completed(t["key"])]
            if pending:
                labels, controls = checked[wid]
                frame = window_frame(root, cfg, bundle, window, labels)
                for task in pending:
                    ledger.perform(task, lambda t: fit_one(out, t, frame, cfg, bundle, controls[t["key"]]))
        ledger.verify()
        base.require(ledger.used == budget, "Incomplete Stage A budget")
        rows = [base.read_json(out / t["key"] / "summary.json") for t in schedule]
        comparisons = {}
        intervals = paired_intervals(out, cfg, schedule)
        for candidate, group in pd.DataFrame(rows).groupby("treatment"):
            wins = int(group.delta.gt(0).sum())
            mean = float(group.delta.mean())
            comparisons[candidate] = dict(mean_ic=float(group.rank_ic.mean()), mean_control_ic=float(group.control_ic.mean()),
                                          delta=mean, improved_windows=wins, delta_ci95=intervals[candidate],
                                          adopt=mean > 0 and wins >= 3)
        atomic_json(out / "comparison.json", dict(new_fits=budget, windows=rows, equal_window=comparisons))


def main(*, execute=False, resume=False, check_inputs=False):
    base.require(not resume or execute, "Resume requires explicit execution")
    cfg, bundle, schedule = frozen_plan(ROOT)
    report = dict(status="configuration_verified", tasks=len(schedule), features=38, formal_fits_started=0)
    if not (execute or check_inputs):
        return report
    manifest = density.inputs(ROOT, bundle, schedule)
    manifest["stage_a_plan_sha256"] = CONFIG_SHA
    for path in ["scripts/stage_a_v3.py","src/equity_research/models/stage_a_v3.py"]:
        manifest["files"][path] = sha(ROOT / path)
    checked = {}
    for wid, window in cfg["windows"].items():
        checked[wid] = density.evaluation(ROOT, bundle, window, [t for t in schedule if t["window"] == wid])
        # Validate all training keys and the 38-column projection before opening the ledger.
        window_frame(ROOT, cfg, bundle, window, checked[wid][0])
    if execute:
        run(ROOT, cfg, bundle, schedule, manifest, checked, resume=resume)
        return dict(status="completed", budget_used=12)
    return dict(report, status="inputs_verified")
