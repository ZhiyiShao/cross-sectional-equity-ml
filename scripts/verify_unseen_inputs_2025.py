"""Rowwise historical parity; no strategy simulation, estimator fitting or performance metrics."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from equity_research.models.context_selection import contexts
from unseen_inputs_v3 import Sources,build_features,panel,KEY


def verify():
    old_source=ROOT/"data/interim/crsp_audit_v1"
    parts=json.loads((old_source/"ingest.json").read_text())["parts"]
    # Use the real two-vintage read path with disjoint predicates on a known
    # archive. This checks joins across New Year without assuming two releases agree.
    extension=([old_source/p for p in parts],old_source/"calendar.parquet")
    source=Sources(ROOT,"2025-01-01","2025-12-31",extension,history_end="2024-12-31")
    cfg=json.loads((ROOT/"config/universe.json").read_text())
    picks,rebuilt=build_features(source,cfg)
    old=ROOT/"data/processed"
    cols=["effective_month","formation_date","PERMNO","PERMCO","liquidity_rank"]
    ref=pd.read_parquet(old/"crsp_v1/monthly_universe.parquet",columns=cols,
                        filters=[("effective_month",">=","2025-01"),("effective_month","<=","2025-12")])
    def sorted_members(f):
        return f[cols].sort_values(["effective_month","liquidity_rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_members(picks),sorted_members(ref),check_dtype=False)
    dates=source.schedule.signal_date.tolist()
    filters=[("signal_date","in",dates)]
    orig=json.loads((ROOT/"config/features_labels.json").read_text())["features"]
    f=pd.read_parquet(old/"research_v1/features_only.parquet",columns=KEY+["signal_eligible"]+orig,filters=filters)
    ind=pd.read_parquet(ROOT/"outputs/risk_comparison_v2/seen_test_risk_inputs_local.parquet",
                        columns=KEY+["industry"],filters=filters)
    f=f.merge(ind,on=KEY,validate="one_to_one").sort_values(KEY).reset_index(drop=True)
    ctx=list(json.loads((ROOT/"config/context_features_v2.json").read_text())["context_features"])
    a,b=contexts(rebuilt),contexts(f)
    pd.testing.assert_frame_equal(a[KEY+["signal_eligible","industry"]],b[KEY+["signal_eligible","industry"]],check_dtype=False)
    np.testing.assert_allclose(a[orig+ctx],b[orig+ctx],atol=1e-12,rtol=1e-12,equal_nan=True)
    y=panel(source,rebuilt[KEY],cfg,labels=True)
    reference=pd.read_parquet(old/"research_v1/features_labels.parquet",columns=list(y),filters=filters).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(y,reference,check_dtype=False,check_exact=True)
    return dict(history_cutoff="2024-12-31",months=12,membership_rows=len(ref),signals=len(dates),candidate_rows=len(y),features=len(orig+ctx),
                max_feature_abs_diff=float(np.nanmax(abs(a[orig+ctx]-b[orig+ctx]))),exact_labels=len(y),
                max_label_end=str(y.label_end_date.max().date()),source_end=str(source.market.index.max().date()))


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
