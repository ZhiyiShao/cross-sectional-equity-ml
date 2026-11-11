"""Read-only verification of the finished replay and prospective model files."""
import argparse
import json
import pandas as pd
import numpy as np

import final_v3 as final
import replay_v3 as replay
from equity_research.models.robustness_ledger import sha
from equity_research.models.preprocessing import require

ROOT=final.ROOT


def audit():
    final.locked()
    counts={}
    for year in [2021,2025]:
        path=ROOT/f"outputs/v3_final_{year}"
        v=final.b.base.read_json(path/"verification.json")
        l=final.b.base.read_json(path/"fit_ledger.json")
        require(sha(path/"fit_ledger.json")==v["ledger_sha256"],"Final ledger changed")
        require(sha(path/"risk_loadings_local.parquet")==v["pca_sha256"],"Changed risk loadings")
        for key,entry in l["entries"].items():
            require(entry["status"]=="completed" and entry["kind"]=="fit","Incomplete final estimator")
            for name,digest in entry["artifacts"].items(): require(sha(path/name)==digest,"Final artifact changed")
            z=final.b.base.read_json(path/key/"summary.json")
            require(pd.Timestamp(z["max_label_end"])<=pd.Timestamp(year,12,31),"Final cutoff breach")
        counts[year]=len(l["entries"])
    prior=pd.read_csv(ROOT/"outputs/frozen_test_v1/daily_portfolios.csv",parse_dates=["date"])
    prior=prior.loc[prior.cost_bps.eq(10)&prior.scenario.eq("lag_2")]
    for name in [*final.V1,"momentum20","equal_weight"]:
        daily=pd.read_parquet(replay.OUT/(name+"_daily.parquet"))
        ref=prior.loc[prior.model.eq(name)].sort_values("date")
        np.testing.assert_array_equal(daily.date,ref.date)
        np.testing.assert_allclose(daily.nav,ref.nav,rtol=1e-10,atol=1e-10)
    old=pd.read_parquet(ROOT/"outputs/historical_replay_v2/daily_portfolios.parquet",
        columns=["date","model","variant","cost_bps","nav"],filters=[("model","=","ridge_enhanced"),("variant","=","constrained"),("cost_bps","=",10)])
    daily=pd.read_parquet(replay.OUT/(replay.V2+"_daily.parquet"))
    old=old.sort_values("date")
    np.testing.assert_array_equal(daily.date,old.date)
    np.testing.assert_allclose(daily.nav,old.nav,rtol=1e-10,atol=1e-10)
    public=pd.read_csv(replay.PUBLIC/"portfolios.csv").set_index("strategy")
    ew=pd.read_parquet(replay.OUT/"equal_weight_daily.parquet").set_index("date").return_net.iloc[1:]
    for name,metrics in public.iterrows():
        ret=pd.read_parquet(replay.OUT/(name+"_daily.parquet")).set_index("date").return_net.iloc[1:]
        pd.testing.assert_index_equal(ew.index,ret.index)
        np.testing.assert_allclose([252*(ret-ew).mean(),(1+ret).prod()**(252/len(ret))-1],
                                   metrics[["annual_active","cagr"]],rtol=1e-12,atol=1e-12)
    return dict(verified_final_fit_counts=counts,previous_NAV_series_reproduced=5,portfolio_statistics_recomputed=len(public))


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
