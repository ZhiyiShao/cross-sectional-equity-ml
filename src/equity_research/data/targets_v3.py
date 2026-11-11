"""Strict multi-horizon training labels on the existing five-day development keys."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from equity_research.data.features import label_one, return_segments
from equity_research.data.universe import regular_trade, filter_codes
from equity_research.models.robustness_ledger import atomic_json, sha

DIRECTORY = "data/processed/research_v3/stage_b_labels"
KEY = ["signal_date", "PERMNO"]


def security_labels(history, market, dates, universe, horizons):
    x = history.set_index("DlyCalDt").reindex(market.index)
    trade = regular_trade(x, universe).to_numpy()
    entry = trade & filter_codes(x, universe["security_filters"]).to_numpy()
    segments = return_segments(x)
    positions = market.index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Signal outside trading calendar")
    rows = []
    for t in positions:
        row = {"signal_date":market.index[t], "PERMNO":int(history.PERMNO.iloc[0])}
        for h in horizons:
            result = label_one(t, x, market, entry, trade, segments, lag=1, horizon=h)
            row[f"end_{h}"] = result["label_end_date"]
            row[f"status_{h}"] = result["label_status"]
            row[f"excess_{h}"] = result["y_excess_5d"]
        rows.append(row)
    return pd.DataFrame(rows)


def build(root, cfg, columns):
    end = pd.Timestamp("2021-12-31")
    base = root / "data/processed/research_v3/features_labels.parquet"
    out = root / DIRECTORY
    if out.exists():
        raise FileExistsError(out)
    source = root / "data/interim/crsp_audit_v1"
    universe = json.loads((root / "config/universe.json").read_text(encoding="utf8"))
    keys = pd.read_parquet(base, columns=KEY, filters=[("is_v1_signal","=",True),("signal_date","<=",end)])
    bounded = pd.read_parquet(base, columns=KEY+["label_end_date","label_status","y_excess_5d"],
        filters=[("is_v1_signal","=",True),("signal_date","<=",end),("label_end_date","<=",end)])
    groups = dict(tuple(keys.groupby("PERMNO")))
    cal = pd.read_parquet(source/"calendar.parquet", columns=["date","vwretd"], filters=[("date","<=","2021-12-31")])
    market = cal.set_index(pd.to_datetime(cal.date)).vwretd
    if not market.index.is_monotonic_increasing or not market.index.is_unique:
        raise ValueError("Unordered or duplicate market sessions")
    panels = []
    digests = {}
    for part in json.loads((source/"ingest.json").read_text(encoding="utf8"))["parts"]:
        raw = pd.read_parquet(source/part, columns=columns,
                              filters=[("DlyCalDt","<=",end),("PERMNO","in",list(groups))])
        digests[part] = hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).to_numpy().tobytes()).hexdigest()
        for permno, history in raw.groupby("PERMNO",sort=False):
            panels.append(security_labels(history, market, groups[permno].signal_date, universe, cfg["horizons"]))
    panel = pd.concat(panels,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(panel[KEY],keys.sort_values(KEY).reset_index(drop=True),check_exact=True)
    rebuilt = panel.loc[panel.end_5.notna(), KEY+["end_5","status_5","excess_5"]].copy()
    rebuilt.columns = bounded.columns
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True),bounded.sort_values(KEY).reset_index(drop=True),check_exact=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        ready = Path(tmp)/"ready"
        ready.mkdir()
        file = ready/"labels.parquet"
        panel.to_parquet(file,index=False,compression="zstd")
        atomic_json(ready/"manifest.json",dict(rows=len(panel), label_sha256=sha(file), cutoff=str(end.date()),
            calendar_projection_sha256=hashlib.sha256(pd.util.hash_pandas_object(cal,index=False).to_numpy().tobytes()).hexdigest(),
            projected_part_hashes=digests, parent_sha256=sha(base), builder_sha256=sha(Path(__file__)),
            plan_sha256=sha(root/"config/stage_b_v3.json")))
        ready.rename(out)
    return {"rows":len(panel),"status":"labels_verified"}
