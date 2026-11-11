"""Frozen turnover and statistical exposure constraints for development and replay."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye, hstack
from threadpoolctl import threadpool_limits

from equity_research.backtest import market, risk_comparison
from equity_research.models.preprocessing import require


def pc_loadings(returns, minimum=60):
    """Annual volatility-unit loadings on the leading three return PCs."""
    keep = returns.count().ge(minimum)
    history = returns.loc[:, keep]
    require(history.shape[1] >= 3 and len(history) > 3, "Insufficient PCA history")
    x = history.sub(history.mean()).fillna(0).to_numpy()
    with threadpool_limits(limits=3):
        eigen, directions = np.linalg.eigh(x @ x.T)
        require(eigen[-3] > 1e-12, "Insufficient statistical risk rank")
        loadings = x.T @ directions[:, -3:] * np.sqrt(252 / (len(x) - 1))
    return pd.DataFrame(loadings, index=history.columns, columns=["pc1","pc2","pc3"])


def training_pcs(root, snapshot, cutoff):
    cutoff = pd.Timestamp(cutoff)
    require(cutoff <= pd.Timestamp("2021-12-31"), "PCA training cutoff")
    source = root / "data/interim/crsp_audit_v1"
    cal = pd.read_parquet(source/"calendar.parquet", columns=["date"],
                          filters=[("date","<=",str(cutoff.date()))])
    calendar = pd.DatetimeIndex(pd.to_datetime(cal.date)).sort_values()[-252:]
    require(len(calendar)==252 and calendar.is_unique, "PCA session count")
    population = pd.read_parquet(root/snapshot, columns=["signal_date","PERMNO","signal_eligible"],
        filters=[("signal_date",">=",calendar[-12]),("signal_date","<=",cutoff)])
    ids = population.loc[population.signal_date.eq(population.signal_date.max()) & population.signal_eligible,"PERMNO"]
    chunks = []
    for part in json.loads((source/"ingest.json").read_text())["parts"]:
        raw = pd.read_parquet(source/part,columns=["DlyCalDt","PERMNO","DlyRet","DlyDelFlg","DlyRetMissFlg","DlyRetDurFlg"],
            filters=[("DlyCalDt",">=",calendar.min()),("DlyCalDt","<=",calendar.max()),("PERMNO","in",ids.tolist())])
        mask = (raw.DlyDelFlg.eq("N") & raw.DlyRetMissFlg.eq("NA") & raw.DlyRetDurFlg.isin(market.SINGLE_DAY)
                & np.isfinite(raw.DlyRet) & raw.DlyRet.ge(-1))
        chunks.append(raw.loc[mask,["DlyCalDt","PERMNO","DlyRet"]])
    history = pd.concat(chunks).pivot(index="DlyCalDt",columns="PERMNO",values="DlyRet").reindex(calendar)
    return pc_loadings(history)


def weights(frame, name, limits, recipe, previous, loadings):
    if recipe["risk"]=="vol" and recipe["penalty"]==0:
        return risk_comparison.optimize(frame,name,limits)[0]
    g = frame.sort_values("PERMNO").set_index("PERMNO")
    n = len(g)
    sectors = pd.get_dummies(g.industry.fillna("Unknown"),dtype=float).to_numpy().T
    beta = g.market_beta_60.replace([np.inf,-np.inf],np.nan)
    exp = np.vstack([sectors,beta.fillna(beta.median())])
    bands = np.r_[np.full(len(sectors),limits["industry_active"]),limits["beta_active"]]
    if recipe["risk"]=="pca":
        # New entrants use the training-universe median loading for each component.
        factors = loadings.reindex(g.index).fillna(loadings.median()).to_numpy().T
        exp = np.vstack([exp,factors])
        bands = np.r_[bands,np.full(3,recipe["pc_band"])]
    A = np.vstack([exp,-exp])
    rhs = np.r_[exp.mean(axis=1)+bands,-exp.mean(axis=1)+bands]
    if recipe["risk"]=="vol":
        vol = g.annual_vol.replace([np.inf,-np.inf],np.nan)
        vol = vol.fillna(vol.median()).to_numpy()
        A = np.vstack([A,vol])
        rhs = np.r_[rhs,vol.mean()+limits["annual_stock_vol_active_upper"]]
    rank = g[name].rank(method="average",pct=True).to_numpy()
    prior = np.array([previous.get(i,0.) for i in g.index])
    # Each auxiliary variable bounds |new target - previous target|.
    I = eye(n,format="csr")
    constraint = bmat([[csr_matrix(A),csr_matrix(A.shape)],[I,-I],[-I,-I]],format="csr")
    c = np.r_[-rank,np.full(n,recipe["penalty"])]
    result = linprog(c,A_ub=constraint,b_ub=np.r_[rhs,prior,-prior],
        A_eq=hstack([np.ones((1,n)),csr_matrix((1,n))]),b_eq=[1.],
        bounds=[(0,limits["max_weight"])]*n+[(0,None)]*n,method="highs")
    require(result.success,"Portfolio solver: "+result.message)
    w = result.x[:n]
    require(abs(w.sum()-1)<1e-8 and (A@w-rhs).max()<1e-8,"Portfolio constraint violation")
    return dict(zip(g.index,w))


def targets(predictions,schedule,name,limits,recipe,pcs):
    by_date = dict(tuple(predictions.loc[predictions.signal_eligible].groupby("signal_date")))
    result,previous = {},{}
    for row in schedule.itertuples():
        g = by_date[row.signal_date]
        previous = weights(g,name,limits,recipe,previous,pcs.get(row.signal_date.year))
        result[row.entry_date] = previous
    return result
