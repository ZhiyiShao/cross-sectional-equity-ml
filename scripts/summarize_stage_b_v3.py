"""Verify stored B predictions against their original five-day evaluation labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from equity_research.models import stage_b_v3 as b
from equity_research.models.robustness_ledger import atomic_json,sha


def main():
    cfg,_,tasks=b.plan(b.ROOT)
    out=b.ROOT/b.RUN
    ledger=b.base.read_json(out/"fit_ledger.json")
    summary=b.base.read_json(out/"comparison.json")
    checked=0
    for wid,w in cfg["windows"].items():
        labels=b.density.targets(b.ROOT/b.density.LABELS,w,"evaluate",five_only=True)
        for t in [t for t in tasks if t["window"]==wid]:
            entry=ledger["entries"][t["key"]]
            b.base.require(entry["status"]=="completed","Incomplete fit")
            for rel,digest in entry["artifacts"].items():
                b.base.require(sha(out/rel)==digest,"Changed artifact "+rel)
            control=b.ROOT/b.density.CONTROL/t["source"]
            treatment=out/t["key"] if t["kind"]=="fit" else control
            pair=pd.read_csv(out/t["key"]/"paired_ic.csv",index_col="signal_date",parse_dates=True)
            for folder,col in [(control,"control_ic"),(treatment,"candidate_ic")]:
                pred=pd.read_parquet(folder/"predictions.parquet")
                f=labels.merge(pred,on=b.KEY,how="left",validate="one_to_one")
                assert f.signal_eligible.all() and np.isfinite(f.prediction).all()
                ranked=f.groupby("signal_date")[["prediction","y_excess_5d"]].rank()
                ic=ranked.groupby(f.signal_date).apply(lambda g:g.prediction.corr(g.y_excess_5d))
                pd.testing.assert_index_equal(ic.index,pair.index)
                np.testing.assert_allclose(ic,pair[col],atol=1e-13)
            checked+=len(pair)
    public=b.ROOT/"results/v3_stage_b_targets"
    public.mkdir()
    pd.DataFrame(summary["windows"]).drop(columns=["parameters"]).to_csv(public/"windows.csv",index=False)
    pd.DataFrame(summary["metrics"]).T.to_csv(public/"comparison.csv")
    atomic_json(public/"verification.json",dict(plan_sha256=b.PIN,ledger_sha256=sha(out/"fit_ledger.json"),
        paired_signal_dates=checked,new_fits=summary["new_fits"],reused=summary["reused"],choices=summary["choices"]))


if __name__=="__main__":
    main()
