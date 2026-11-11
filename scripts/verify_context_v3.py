"""Independent arithmetic checks on every non-five-day cross section (no labels)."""
import json
import numpy as np
import pandas as pd
from prepare_context_v3 import OUT, config, sha, verify_files


def verify():
    verify_files()
    features = list(config("context_features_v2.json")["context_features"])
    frame = pd.read_parquet(OUT/"daily_features.parquet", columns=["signal_date","industry","signal_eligible",
        "momentum_20","momentum_60","volatility_20","reversal_1"]+features, filters=[("is_v1_signal","=",False)])
    for date, g in frame.groupby("signal_date"):
        m = g.momentum_20.where(g.signal_eligible)
        vol = g.volatility_20.where(g.signal_eligible).median()
        breadth = (m.dropna()>0).mean()
        expected = dict(universe_mean_mom20=m.mean(), universe_breadth20=breadth, universe_dispersion20=m.std(),
                        universe_median_vol20=vol, momentum_vol_interaction=g.momentum_20*vol,
                        reversal_breadth_interaction=g.reversal_1*(breadth-.5),
                        momentum_dispersion_interaction=g.momentum_60*m.std())
        for w in [20,60]:
            s = g[f"momentum_{w}"].where(g.signal_eligible)
            stats = s.groupby(g.industry).agg(["sum","count"])
            count = g.industry.map(stats["count"])
            local = (count>=5) & g.industry.ne("Unknown")
            mean = (g.industry.map(stats["sum"])/count.where(count>0)).where(local,s.mean())
            expected[f"industry_mom{w}"] = mean
            expected[f"relative_mom{w}"] = s-mean
            if w==20:
                rank = s.groupby(g.industry).rank(method="average")
                expected["industry_rank20"] = ((rank-.5)/count-.5).where(local,(s.rank()-.5)/s.count()-.5)
        for name in features:
            np.testing.assert_allclose(g[name],expected[name],equal_nan=True,rtol=1e-11,atol=1e-12,
                                       err_msg=f"{date} {name}")
    return {"rows":len(frame),"dates":frame.signal_date.nunique(),"context_columns":len(features),
            "manifest_sha256":sha(OUT/"manifest.json")}


if __name__ == "__main__":
    print(json.dumps(verify()))
