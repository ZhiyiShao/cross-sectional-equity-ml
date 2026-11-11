"""Prepare 0b and the stage A feature block. Only development features are read."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from equity_research.data.directional_v3 import directional_features, ordinary_returns
from equity_research.evaluation.exposures import sic_group
from equity_research.models.context_selection import contexts

END = pd.Timestamp("2021-12-31")
KEY = ["signal_date", "PERMNO"]
BASE = ROOT / "data/processed/research_v3/features_labels.parquet"
OUT = BASE.parent / "context_0b"
SOURCE = ROOT / "data/interim/crsp_audit_v1"
V2 = ROOT / "outputs/context_selection_v2/development_features_local.parquet"
RULES = ["config/features_v3.json", "config/context_features_v2.json", "config/features_labels.json",
         "config/baseline_v3.json", "config/density_0b.json", "scripts/prepare_context_v3.py",
         "scripts/build_daily_v3.py", "src/equity_research/data/features.py",
         "src/equity_research/data/universe.py", "src/equity_research/evaluation/exposures.py",
         "src/equity_research/models/context_selection.py", "src/equity_research/data/directional_v3.py"]
FIELDS = ["PERMNO", "DlyCalDt", "SICCD", "SecInfoStartDt", "SecInfoEndDt", "DlyRet", "DlyDelFlg",
          "DlyRetDurFlg", "DlyRetMissFlg"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def config(name):
    return json.loads((ROOT / "config" / name).read_text(encoding="utf8"))


def verify_files():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Manifest mismatch: {rel}")
    print("Verified local feature panel, code, configuration and parent panel hashes.")


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    original = config("features_labels.json")["features"]
    context = config("context_features_v2.json")
    added = list(context["context_features"])
    windows = config("features_v3.json")["windows"]
    base = pd.read_parquet(BASE, columns=KEY + ["signal_eligible", "is_v1_signal"] + original,
                           filters=[("signal_date", "<=", END)]).sort_values(KEY).reset_index(drop=True)
    groups = {p: g for p, g in base.groupby("PERMNO")}
    cal = pd.read_parquet(SOURCE / "calendar.parquet", columns=["date"], filters=[("date", "<=", str(END.date()))])
    calendar = pd.DatetimeIndex(pd.to_datetime(cal.date))
    parts = json.loads((SOURCE / "ingest.json").read_text())["parts"]
    blocks, sources = [], {}
    for name in parts:
        raw = pd.read_parquet(SOURCE / name, columns=FIELDS,
                              filters=[("DlyCalDt", "<=", END), ("PERMNO", "in", list(groups))])
        sources[name] = {"rows": len(raw), "projected_rows_digest": rows_digest(raw)}
        for permno, history in raw.groupby("PERMNO", sort=False):
            x = history.set_index("DlyCalDt").reindex(calendar)
            g = groups[permno]
            dates = pd.DatetimeIndex(g.signal_date)
            r = ordinary_returns(x, calendar)
            np.testing.assert_array_equal(r.loc[dates].to_numpy(), -g.reversal_1.to_numpy())
            d = directional_features(r, windows).loc[dates].reset_index()
            s = x.loc[dates]
            d["industry"] = [sic_group(c, start, end, t) for c, start, end, t in
                             zip(s.SICCD, s.SecInfoStartDt, s.SecInfoEndDt, dates)]
            d["PERMNO"] = permno
            blocks.append(d)
    extra = pd.concat(blocks, ignore_index=True)
    daily = base.merge(extra, on=KEY, how="left", validate="one_to_one")
    if daily.industry.isna().any() or len(extra) != len(base):
        raise ValueError("Missing or unexpected daily keys")
    daily = contexts(daily, context["industry_min_count"])
    reference = pd.read_parquet(V2, columns=KEY + ["industry"] + added, filters=[("signal_date", "<=", END)])
    five = daily.loc[daily.is_v1_signal].set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(five[["industry"] + added], reference.set_index(KEY).sort_index(), check_exact=True)
    risk_keys = pd.read_parquet(ROOT / "data/processed/research_v1/features_only.parquet", columns=KEY,
                                filters=[("signal_date", "<=", END)])
    joined = base[KEY].merge(risk_keys, on=KEY, validate="one_to_one").set_index(KEY).sort_index()
    pd.testing.assert_index_equal(joined.index, five.index)
    days = daily.groupby("signal_date").size()
    if len(days) != 1763 or not days.eq(500).all() or len(five) != 176500:
        raise ValueError("Daily candidate coverage changed")
    coverage = dict(rows=len(daily), days=len(days), five_day_rows=len(five), non_five_day_rows=len(daily)-len(five),
                    unknown_industry=int(daily.industry.eq("Unknown").sum()), legacy_risk_join_rows=len(joined),
                    legacy_risk_join_lost=len(daily)-len(joined), finite={c:int(np.isfinite(daily[c]).sum()) for c in added})
    with TemporaryDirectory(dir=BASE.parent) as tmp:
        ready = Path(tmp) / OUT.name
        ready.mkdir()
        file = "daily_features.parquet"
        daily.to_parquet(ready / file, index=False, compression="zstd")
        files = {p: sha(ROOT / p) for p in RULES}
        files[BASE.relative_to(ROOT).as_posix()] = sha(BASE)
        files[(OUT / file).relative_to(ROOT).as_posix()] = sha(ready / file)
        manifest = dict(files=files, versions={n: version(n) for n in ["numpy","pandas","pyarrow"]},
                        inputs={"parent": str(BASE.relative_to(ROOT)), "partition_directory":str(SOURCE.relative_to(ROOT)),
                                "calendar_digest":rows_digest(cal), "columns":FIELDS, "cutoff":str(END.date()),
                                "security_filter":"PERMNO in development panel", "partitions":sources,
                                "v2_parity_reference":str(V2.relative_to(ROOT)), "v2_projection_digest":rows_digest(reference),
                                "legacy_risk_key_digest":rows_digest(risk_keys)},
                        context_definitions=context["context_features"], coverage=coverage)
        (ready / "manifest.json").write_text(json.dumps(manifest,indent=2), encoding="utf8")
        ready.rename(OUT)
    print(json.dumps(coverage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    if parser.parse_args().build:
        build()
    else:
        verify_files()
