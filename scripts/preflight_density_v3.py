"""Read-only 0b input gate. Validate frozen files and split counts without fitting."""
import json
import pandas as pd
import numpy as np
from prepare_context_v3 import BASE, OUT, ROOT, sha, verify_files


def preflight():
    public = json.loads((ROOT / "results/context_preparation_v3/verification.json").read_text())
    if sha(OUT / "manifest.json") != public["manifest_sha256"]:
        raise ValueError("0b input manifest differs from the reviewed preparation")
    verify_files()
    windows = json.loads((ROOT / "config/research_v3.json").read_text())["development_windows"]
    rows = []
    for window in windows:
        for role in ["train","evaluate"]:
            start, end = map(pd.Timestamp,window[role])
            filters = [("signal_date",">=",start),("signal_date","<=",end),("label_end_date","<=",end),
                       ("signal_eligible","=",True),("label_status","in",["ok","ok_bridged"])]
            if role == "evaluate":
                filters.append(("is_v1_signal", "=", True))
            f = pd.read_parquet(BASE,columns=["signal_date","label_end_date","is_v1_signal","y_excess_5d"],filters=filters)
            f = f.loc[f.label_end_date.gt(f.signal_date) & np.isfinite(f.y_excess_5d)]
            if role=="evaluate":
                control = pd.read_csv(ROOT/"results/v3_baseline/window_metrics.csv")
                matches = control.loc[control.window.eq(window["id"]), "rows"].unique()
                if matches.tolist() != [int(f.is_v1_signal.sum())]:
                    raise ValueError("0b evaluation mask differs from 0a")
            rows.append(dict(window=window["id"],role=role,eligible_daily=len(f),
                             eligible_five_day=int(f.is_v1_signal.sum())))
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    preflight()
