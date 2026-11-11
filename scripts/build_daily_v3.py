"""Rebuild the development panel from audited CRSP partitions and check V1 parity."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from equity_research.data.features import label_one, make_features, return_segments, split_masks
from equity_research.data.universe import filter_codes, regular_trade

END = pd.Timestamp("2021-12-31")
KEY = ["signal_date", "PERMNO"]
FIELDS = """PERMNO DlyCalDt DlyDelFlg DlyPrc DlyPrcFlg DlyClose DlyHigh DlyLow DlyVol
ShrOut DlyCap DlyCumFacShr DlyRet DlyRetMissFlg DlyRetDurFlg PrimaryExch SecurityType
SecuritySubType ShareType USIncFlg IssuerType ConditionalType TradingStatusFlg""".split()


def security_panel(history, membership, market, config, universe):
    calendar = market.index
    x = history.set_index("DlyCalDt").reindex(calendar)
    membership = membership.set_index("effective_month")
    months = calendar.strftime("%Y-%m")
    start = calendar.get_loc(pd.Timestamp(config["signal_start"]))
    positions = np.flatnonzero((np.arange(len(calendar)) >= start) & months.isin(membership.index))
    trade = regular_trade(x, universe).to_numpy()
    entry = trade & filter_codes(x, universe["security_filters"]).to_numpy()
    segments = return_segments(x)
    features = make_features(x, market, universe).iloc[positions].reset_index(drop=True)
    dates = calendar[positions]
    features["signal_date"] = dates
    features["PERMNO"] = int(history.PERMNO.iloc[0])
    features["formation_date"] = membership.loc[months[positions], "formation_date"].to_numpy()
    features["signal_eligible"] = entry[positions]
    features["is_v1_signal"] = (positions - start) % config["signal_step_trading_days"] == 0
    labels = [label_one(t, x, market, entry, trade, segments, config["entry_lag_trading_days"],
                        config["holding_period_trading_days"]) for t in positions]
    return pd.concat([features, pd.DataFrame(labels)], axis=1)


def verify(panel, reference, config, market):
    panel = panel.sort_values(KEY).reset_index(drop=True)
    counts = panel.groupby("signal_date").size()
    dates = market.loc[config["signal_start"]:].index
    if (panel.duplicated(KEY).any() or not counts.index.equals(dates) or not counts.eq(500).all()
            or not panel.formation_date.lt(panel.signal_date).all()):
        raise ValueError("Daily universe, formation dates or coverage mismatch")
    five = panel.loc[panel.is_v1_signal].set_index(KEY)
    common = ["signal_eligible", *config["features"]]
    filters = [("signal_date", "<=", END)]
    v1 = pd.read_parquet(reference, columns=KEY + common, filters=filters).set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(five[common], v1, check_exact=True)
    labels = ["entry_date", "label_end_date", "label_status", "y_excess_5d", "y_excess_5d_event_proxy"]
    valid = five.label_end_date.notna()
    old = pd.read_parquet(reference, columns=KEY + labels,
                          filters=filters + [("label_end_date", "<=", END)]).set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(five.loc[valid, labels], old, check_exact=True)
    if not five.loc[~valid, "y_excess_5d"].isna().all():
        raise ValueError("Label extends past development end")
    return dict(rows=len(panel), dates=len(dates), five_day_feature_rows=len(five),
                in_bounds_label_rows=len(old), right_censored_five_day_rows=int((~valid).sum()),
                cutoff=str(END.date()), feature_columns=len(config["features"]))


def build():
    processed = ROOT / "data/processed"
    out = processed / "research_v3"
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / "data/interim/crsp_audit_v1"
    cfg = json.loads((ROOT / "config/features_labels.json").read_text())
    uc = json.loads((ROOT / "config/universe.json").read_text())
    cal = pd.read_parquet(source / "calendar.parquet", columns=["date", "vwretd"],
                          filters=[("date", "<=", str(END.date()))])
    market = cal.set_index(pd.to_datetime(cal.date)).vwretd
    members = pd.read_parquet(processed / "crsp_v1/monthly_universe.parquet",
                              columns=["effective_month", "formation_date", "PERMNO"],
                              filters=[("effective_month", "<=", "2021-12")])
    by_id = dict(tuple(members.groupby("PERMNO")))
    parts = json.loads((source / "ingest.json").read_text())["parts"]
    panels = []
    for part in parts:
        rows = pd.read_parquet(source / part, columns=FIELDS,
                               filters=[("DlyCalDt", "<=", END), ("PERMNO", "in", list(by_id))])
        for permno, history in rows.groupby("PERMNO", sort=False):
            panels.append(security_panel(history, by_id[permno], market, cfg, uc))
    panel = split_masks(pd.concat(panels, ignore_index=True), cfg).sort_values(KEY).reset_index(drop=True)
    summary = verify(panel, processed / "research_v1/features_labels.parquet", cfg, market)
    with TemporaryDirectory(dir=processed) as temp:
        ready = Path(temp) / "verified"
        ready.mkdir()
        panel.to_parquet(ready / "features_labels.parquet", index=False, compression="zstd")
        (ready / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        ready.rename(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if parser.parse_args().execute:
        build()
    else:
        print("Development panel, cutoff 2021-12-31. Pass --execute to build and verify.")
