"""Apply the W1 label-blind screen for the predeclared directional block."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_context_v3 import OUT, ROOT, config, sha, verify_files

RULE_HASH = "f909dc21753d9b7770a75a083016f17a05ba44b75ead0c88b90fedf69213b11d"


def paired_rho(frame, name, other):
    values = frame[[name, other]].replace([np.inf, -np.inf], np.nan).dropna()
    dates = frame.loc[values.index, "signal_date"]
    ranks = values.groupby(dates).rank(method="average")
    centers = ranks - ranks.groupby(dates).transform("mean")
    x, y = centers[name], centers[other]
    n = x.groupby(dates).size()
    denom = np.sqrt((x*x).groupby(dates).sum() * (y*y).groupby(dates).sum())
    daily = ((x*y).groupby(dates).sum()/denom.where(denom.gt(0))).where(n.ge(100)).dropna()
    if len(daily) < 0.9 * frame.signal_date.nunique():
        raise ValueError(f"Insufficient paired dates for {name} and {other}: {len(daily)}")
    return float(daily.mean()), len(daily)


def adopted(delta):
    values = np.asarray(delta, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Four finite paired window differences required")
    return bool(values.mean() > 0 and (values > 0).sum() >= 3)


def screen():
    rules = ROOT / "config/features_v3.json"
    if sha(rules) != RULE_HASH:
        raise ValueError("Stage A screen definition changed")
    verify_files()
    plan = config("features_v3.json")
    original = config("features_labels.json")["features"]
    context = list(config("context_features_v2.json")["context_features"])
    candidates = [f"{f['name']}_{w}" for f in plan["families"] for w in plan["windows"]]
    kept = [c for c in original + context if c not in plan["screen"]["skip_existing"]]
    source = OUT / "daily_features.parquet"
    frame = pd.read_parquet(source, columns=["signal_date"]+kept+candidates,
                            filters=[("signal_date",">=",pd.Timestamp("2015-01-01")),
                                     ("signal_date","<=",pd.Timestamp("2017-12-31")),
                                     ("is_v1_signal","=",True),("signal_eligible","=",True)])
    decisions = []
    for name in candidates:
        missing = float((~np.isfinite(frame[name])).mean())
        decision = {"feature":name, "missing_fraction":missing, "comparisons":{}}
        reason = "missing" if missing > 0.2 else "retain"
        if reason == "retain":
            for other in kept:
                rho, days = paired_rho(frame, name, other)
                decision["comparisons"][other] = {"mean_daily_rho":rho, "valid_days":days}
                if abs(rho) >= 0.95:
                    reason = "redundant with "+other
                    break
        if reason == "retain":
            kept.append(name)
        decisions.append(dict(decision, decision=reason))
    dest = ROOT / "results/stage_a_screen_v3"
    dest.mkdir(exist_ok=False)
    (dest/"screen.json").write_text(json.dumps(dict(rule_sha256=RULE_HASH, input_sha256=sha(source),
        code_sha256=sha(Path(__file__)),rows=len(frame),dates=frame.signal_date.nunique(),
        decisions=decisions),indent=2),encoding="utf8")
    print([(d["feature"],d["decision"]) for d in decisions])


if __name__ == "__main__":
    screen()
