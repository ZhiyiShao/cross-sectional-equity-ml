"""Develop portfolio rules on the two years with audited funding and corporate actions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import stage_c_v3 as c
from equity_research.backtest import market, portfolio_v3, validation
from equity_research.models.common import circular_block_interval
from equity_research.models.robustness_ledger import atomic_json,sha

ROOT=c.ROOT
PLAN=ROOT/"config/stage_d_v3.json"
PUBLIC=ROOT/"results/v3_stage_d"
OUT=ROOT/"outputs/v3_stage_d"


def score_folder(name,w,cfg):
    if name in cfg["incumbents"]:
        return c.incumbent(cfg["incumbents"][name],w)
    return ROOT/c.RUN/(w+"_"+name)


def freeze():
    require=market.require
    require(not PLAN.exists() and not PLAN.with_suffix(".sha256").exists(),"D plan exists")
    cfg=c.b.base.read_json(ROOT/c.PLAN)
    name=c.b.base.read_json(ROOT/"results/v3_stage_c/comparison.json")["selected"]
    recipes={risk+"_"+str(p):dict(risk=risk,penalty=p,pc_band=.02) for risk in ["vol","pca"] for p in [0.,.05,.15]}
    files=[ROOT/c.PLAN,ROOT/"results/v3_stage_c/comparison.json",ROOT/"config/research_v2.json",
           ROOT/"src/equity_research/backtest/portfolio_v3.py",Path(__file__)]
    files+=[score_folder(name,w,cfg)/"predictions.parquet" for w in ["W3","W4"]]
    spec=dict(signal=name,recipes=recipes,baseline="vol_0.0",cost_bps=10,action_lag=2,windows=["W3","W4"],
              rules="Maximize 2020-21 arithmetic annual net active return, then require both annual returns higher than control. Otherwise retain control. Exact ties follow recipe order. EW is a funding-path reference.",
              turnover="L1 distance from previous target, zero at inception. Exit of names outside current eligible pool is a constant in the objective.",
              pca="Last 252 training sessions, eligible pool at final training snapshot, minimum 60 ordinary observations. Center individual returns, missing centered observations zero. First three covariance PCs; annualized loading bands +/-0.02. New entrants receive training median loadings.",
              sources={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    atomic_json(PLAN,spec)
    PLAN.with_suffix(".sha256").write_text(sha(PLAN)+"\n",encoding="ascii")


def execute(spec):
    for rel,digest in spec["sources"].items():
        market.require(sha(ROOT/rel)==digest,"D input changed: "+rel)
    market.require(not OUT.exists() and not PUBLIC.exists(),"Existing stage D run")
    cfg=c.b.base.read_json(ROOT/c.PLAN)
    _,bundle,_=c.b.plan(ROOT)
    snapshot=bundle["plan"]["snapshot"]
    _,terms,_,targets,dates,m,_=validation.prepare_inputs()
    schedule=pd.DataFrame({"entry_date":sorted(targets["equal_weight"])})
    schedule["signal_date"]=[dates[dates.get_loc(day)-1] for day in schedule.entry_date]
    scores=pd.concat([c.read_scores(score_folder(spec["signal"],w,cfg)) for w in spec["windows"]])
    keys=c.b.KEY
    f=pd.read_parquet(ROOT/snapshot,columns=keys+["signal_eligible","industry","market_beta_60","volatility_20"],
                      filters=[("signal_date","in",list(schedule.signal_date))]).sort_values(keys).reset_index(drop=True)
    scores=scores.loc[scores.signal_date.isin(schedule.signal_date)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[keys+["signal_eligible"]],scores[keys+["signal_eligible"]])
    f["signal"]=scores.prediction
    f["annual_vol"]=f.volatility_20
    pcs={int(cfg["windows"][w]["evaluate"][0][:4]):portfolio_v3.training_pcs(ROOT,snapshot,cfg["windows"][w]["train"][1])
         for w in spec["windows"]}
    limits=c.b.base.read_json(ROOT/"config/research_v2.json")["constraints"]
    ew=market.make_targets(f,schedule,models=["equal_weight"])["equal_weight"]
    market.require(ew==targets["equal_weight"],"Equal weight population changed")
    OUT.mkdir()
    paths={}; rows=[]; checks=0
    for name in ["equal_weight",*spec["recipes"]]:
        weights=ew if name=="equal_weight" else portfolio_v3.targets(f,schedule,"signal",limits,spec["recipes"][name],pcs)
        d,o,h,_=validation.simulate(m,weights,terms,spec["cost_bps"],spec["action_lag"],False,True)
        checks+=validation.verify(d,o,h)
        if name=="equal_weight":
            old=pd.read_csv(ROOT/"outputs/final_funding_backtest_v1/daily_portfolios.csv",parse_dates=["date"])
            ref=old.loc[old.model.eq(name)&old.cost_bps.eq(10)&old.scenario.eq("lag_2")].sort_values("date")
            np.testing.assert_array_equal(ref.date,d.date)
            np.testing.assert_allclose(ref.nav,d.nav,atol=1e-10,rtol=1e-10)
        d.to_parquet(OUT/(name+".parquet"),index=False)
        paths[name]=d.set_index("date").return_net.iloc[1:]
        active=paths[name]-paths["equal_weight"]
        rows.append(dict(path=name,annual_active=active.mean()*252,
                         **{str(y):v.mean()*252 for y,v in active.groupby(active.index.year)},**market.performance(d)))
        print("Verified",name,flush=True)
    table=pd.DataFrame(rows).set_index("path")
    best=table.loc[list(spec["recipes"]),"annual_active"].idxmax()
    base=spec["baseline"]
    adopt=bool(best!=base and (table.loc[best,["2020","2021"]]>table.loc[base,["2020","2021"]]).all())
    PUBLIC.mkdir()
    table.to_csv(PUBLIC/"portfolios.csv")
    intervals={n:circular_block_interval((paths[n]-paths[base]).to_numpy()*252,20,2000,20261110)
               for n in spec["recipes"]}
    atomic_json(PUBLIC/"comparison.json",dict(selected=best if adopt else base,highest_mean=best,adopt=adopt,
        signal=spec["signal"],checks=int(checks),paths=len(rows),paired_intervals=intervals,plan_sha256=sha(PLAN)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    grp=parser.add_mutually_exclusive_group()
    grp.add_argument("--freeze",action="store_true")
    grp.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.freeze: freeze()
    market.require(sha(PLAN)==PLAN.with_suffix(".sha256").read_text().strip(),"D pin mismatch")
    if args.execute: execute(c.b.base.read_json(PLAN))
