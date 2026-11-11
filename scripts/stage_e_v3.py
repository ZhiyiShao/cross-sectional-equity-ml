"""Account for research trials, quantify multiplicity and lock final evaluation."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm,skew,kurtosis

import stage_d_v3 as d
from equity_research.models.robustness_ledger import atomic_json,sha

ROOT=d.ROOT
PUBLIC=ROOT/"results/v3_synthesis"
FINAL=ROOT/"config/final_v3.json"


def inventory():
    rows=[]
    for name,file in [("baselines_v1","trials.csv"),("temporal_cv_v1","fold_trials.csv"),("feature_subset_cv_v1","fold_trials.csv")]:
        path=ROOT/"outputs"/name/file
        frame=pd.read_csv(path)
        configs=len(frame) if name=="baselines_v1" else frame.configuration.nunique()
        reused=0
        if name=="feature_subset_cv_v1":
            reused=json.loads((path.parent/"comparison_verification.json").read_text())["reused_reference_fits"]
        rows.append(dict(run=name,role="selection",new_fits=len(frame),reused=reused,designs=configs,source=path))
    for name in ["cv_selected_validation_v1","full_train_selection_v1","final_models_v1","context_selection_v2",
                 "ranking_selection_v2","final_models_v2","parameter_robustness_v2","v3_baseline","v3_density_0b",
                 "v3_stage_a_directional","v3_stage_b_targets","v3_stage_c"]:
        path=ROOT/"outputs"/name/"fit_ledger.json"
        data=json.loads(path.read_text())
        if isinstance(data,list):
            fits=len(data); reused=0
            configs=len(set((e["design"],e["candidate"]) for e in data)) if "window" in data[0] else fits
        else:
            assert set(data["entries"])==set(data["tasks"])
            assert all(e["status"]=="completed" for e in data["entries"].values())
            fits=sum(t["kind"]=="fit" for t in data["entries"].values())
            reused=len(data["entries"])-fits
            configs=len(set(k.split("_",1)[1] for k in data["entries"]))
        role="refit" if name.startswith(("cv_selected","final_models")) else "selection"
        rows.append(dict(run=name,role=role,new_fits=fits,reused=reused,designs=configs,source=path))
    for r in rows:
        r["file_sha256"]=sha(r["source"])
        r["source"]=r["source"].relative_to(ROOT).as_posix()
    return pd.DataFrame(rows)


def adjusted_ic():
    """Center each year's paired series, circular-block sample each year, then equal-weight years."""
    outputs={}
    for folder in ["v3_density_0b","v3_stage_a_directional","v3_stage_b_targets","v3_stage_c"]:
        out=ROOT/"outputs"/folder
        for path in sorted(out.glob("W*_*/paired_ic.csv")):
            w,name=path.parent.name.split("_",1)
            g=pd.read_csv(path,parse_dates=["signal_date"]).set_index("signal_date").delta
            assert g.index.year.unique().tolist()==[2017+int(w[1:])]
            outputs.setdefault((folder,name),{})[w]=g.to_numpy()
    records=[]
    for (stage,name),parts in outputs.items():
        assert set(parts)=={"W1","W2","W3","W4"}
        rng=np.random.default_rng(20261110)
        draws=[]
        for w in sorted(parts):
            x=parts[w]; n=len(x); starts=rng.integers(n,size=(10000,(n+9)//10))
            ix=((starts[:,:,None]+np.arange(10))%n).reshape(len(starts),-1)[:,:n]
            draws.append((x-x.mean())[ix].mean(axis=1))
        effect=np.mean([x.mean() for x in parts.values()])
        p=(1+sum(np.mean(draws,axis=0)>=effect))/(1+len(draws[0]))
        records.append(dict(stage=stage,candidate=name,mean_delta=effect,p_raw=p))
    table=pd.DataFrame(records).sort_values("p_raw").reset_index(drop=True)
    assert len(table)==18
    table["p_holm"]=np.minimum(1,np.maximum.accumulate(table.p_raw*(len(table)-np.arange(len(table)))))
    return table


def deflated(returns,chosen):
    """Bailey–López de Prado DSR, nominal independent path count, observed daily SR dispersion."""
    from math import sqrt
    sr=returns.mean()/returns.std()
    n=len(sr)
    gamma=.5772156649015329
    threshold=sr.std()*((1-gamma)*norm.ppf(1-1/n)+gamma*norm.ppf(1-1/(n*np.e)))
    x=returns[chosen]
    s=sr[chosen]
    se=sqrt((1-skew(x,bias=False)*s+(kurtosis(x,fisher=False,bias=False)-1)*s*s/4)/(len(x)-1))
    return dict(paths=n,days=len(x),selected=chosen,annualized_sharpe=s*np.sqrt(252),
                annualized_hurdle=threshold*np.sqrt(252),probability=norm.cdf((s-threshold)/se),
                interpretation="Six correlated strategy paths, fixed zero risk-free rate, daily SR dispersion. DSR assumes independent daily observations. Conditional on stage D; prior prediction searches are not independent portfolio paths.")


def execute():
    assert not PUBLIC.exists() and not FINAL.exists()
    cplan=d.c.b.base.read_json(ROOT/d.c.PLAN)
    protocol=d.c.b.base.read_json(ROOT/"config/research_v3.json")
    ds=d.c.b.base.read_json(d.PUBLIC/"comparison.json")
    dc=d.c.b.base.read_json(d.PLAN)
    inv=inventory()
    tests=adjusted_ic()
    daily=pd.DataFrame({n:pd.read_parquet(d.OUT/(n+".parquet"),columns=["date","return_net"]).set_index("date").return_net
                        for n in dc["recipes"]}).iloc[1:]
    assert daily.notna().all().all() and set(daily.index.year)=={2020,2021}
    dsr=deflated(daily,ds["selected"])
    refs=[ROOT/d.c.PLAN,d.PLAN,ROOT/"results/v3_stage_c/comparison.json",d.PUBLIC/"comparison.json",
          ROOT/"config/research_v3.json",ROOT/"config/research_v2.json",ROOT/"config/stage_b_v3.json",
          ROOT/"src/equity_research/backtest/portfolio_v3.py",ROOT/"src/equity_research/models/ensemble_v3.py",
          ROOT/"src/equity_research/models/stage_b_v3.py",ROOT/"scripts/stage_c_v3.py",ROOT/"scripts/stage_e_v3.py",
          ROOT/"scripts/final_v3.py",ROOT/"scripts/replay_v3.py",ROOT/"config/final_test_plan.json",ROOT/"config/final_evaluation_v2.json",
          ROOT/"src/equity_research/backtest/engine.py",ROOT/"src/equity_research/backtest/replay_v2.py"]
    selection=dict(signal=dc["signal"],portfolio=dc["recipes"][ds["selected"]],portfolio_id=ds["selected"],
                   incumbents=cplan["incumbents"],members=cplan["members"],variants=cplan["variants"],
                   seen=dict(period=["2022-01-01","2025-12-31"],training=["2015-01-01","2021-12-31"],
                             runs=1,status="previously_seen",baseline_portfolios="Original V1 top decile, V2 enhanced ridge original V2 risk LP, momentum top decile and universe EW. Extra fixed same-construction signal comparison evaluated under selected V3 rule."),
                   unseen=protocol["unseen_period"],funding=d.c.b.base.read_json(ROOT/"config/research_v2.json")["execution"],
                   sources={p.relative_to(ROOT).as_posix():sha(p) for p in refs})
    PUBLIC.mkdir()
    inv.to_csv(PUBLIC/"trial_inventory.csv",index=False)
    tests.to_csv(PUBLIC/"paired_ic_adjustment.csv",index=False)
    atomic_json(PUBLIC/"deflated_sharpe.json",dsr)
    atomic_json(FINAL,selection)
    FINAL.with_suffix(".sha256").write_text(sha(FINAL)+"\n",encoding="ascii")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--freeze",action="store_true")
    if p.parse_args().freeze:
        execute()
