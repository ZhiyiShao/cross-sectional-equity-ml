"""Exercise the frozen Stage A pipeline only on invented stocks and dates."""
import json

import numpy as np
import pandas as pd
import pytest

from equity_research.models import stage_a_v3 as a


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def experiment(tmp_path):
    cfg, bundle, tasks = a.frozen_plan(a.ROOT)
    rng = np.random.default_rng(914)
    calendar = pd.bdate_range("2015-01-01","2022-01-14")
    dates = calendar[calendar.month.isin([1,12])]
    pos = calendar.get_indexer(dates)
    n = 72
    f = pd.DataFrame(dict(signal_date=np.repeat(dates,n),PERMNO=np.tile(np.arange(n),len(dates)),
                          signal_eligible=True, is_v1_signal=np.repeat(pos % 5 == 0,n)))
    columns = cfg["original"] + cfg["context"] + cfg["directional"]
    f[columns] = rng.normal(size=(len(f),len(columns)))
    f["label_end_date"] = np.repeat(calendar[np.minimum(pos+6,len(calendar)-1)],n)
    f["label_status"] = "ok"
    f["y_excess_5d"] = f[columns[0]] + f[columns[-1]] * .5
    f.loc[f.PERMNO==0,"signal_eligible"] = False
    f.loc[f.PERMNO==1,"label_status"] = "delisted"
    f.loc[f.PERMNO==2,"label_end_date"] = f.signal_date
    f.loc[f.PERMNO==3,"y_excess_5d"] = np.nan
    f.loc[f.PERMNO==4,"label_status"] = "ok_bridged"
    meta = a.KEY+["signal_eligible","is_v1_signal"]
    write(f[meta+["label_status","label_end_date","y_excess_5d"]], tmp_path/a.density.LABELS)
    write(f[meta+columns].assign(gain_share_60=123), tmp_path/a.density.PANEL/"daily_features.parquet")
    ref = f.loc[f.is_v1_signal]
    write(ref[a.KEY+["signal_eligible","label_status","label_end_date","y_excess_5d"]+columns[:33]],
          tmp_path/bundle["plan"]["snapshot"])
    for t in tasks:
        lo,hi=cfg["windows"][t["window"]]["evaluate"]
        ec=ref.loc[ref.signal_date.between(lo,hi)].copy()
        # Different controls catch swapped families.
        ec["prediction"]=ec[columns[list(cfg["pairs"]).index(t["treatment"])]]
        folder=tmp_path/a.density.CONTROL/t["source"]
        write(ec[a.KEY+["signal_eligible","prediction"]],folder/"predictions.parquet")
        ok=ec.loc[ec.PERMNO.ge(4)&ec.label_end_date.le(pd.Timestamp(hi))]
        ic=ok.groupby("signal_date")[["prediction","y_excess_5d"]].corr(method="spearman").xs("prediction",level=1).y_excess_5d
        ic.rename("rank_ic").to_csv(folder/"daily_ic.csv")
    checked={wid:a.density.evaluation(tmp_path,bundle,w,[t for t in tasks if t["window"]==wid])
             for wid,w in cfg["windows"].items()}
    return cfg,bundle,tasks,checked,f


def test_preflight_and_hash_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(a.density,"inputs",lambda *args:pytest.fail("Unexpected private read"))
    monkeypatch.setattr(pd,"read_parquet",lambda *args,**kwargs:pytest.fail("Data read"))
    assert a.main()==dict(status="configuration_verified",tasks=12,features=38,formal_fits_started=0)
    with pytest.raises(ValueError,match="Resume"):
        a.main(resume=True)
    p=tmp_path/a.CONFIG
    p.parent.mkdir()
    p.write_text("{}")
    with pytest.raises(ValueError,match="configuration changed"):
        a.frozen_plan(tmp_path)


def test_complete_synthetic_run(tmp_path,experiment,monkeypatch):
    cfg,bundle,tasks,checked,f=experiment
    calls=[]
    fit=a.v2.fit_candidate
    def checked_fit(model,t,matrix,frame,mask,cutoff):
        calls.append(t["key"])
        assert matrix.columns.tolist()==cfg["original"]+cfg["context"]+cfg["directional"]
        w=cfg["windows"][t["window"]]
        for role in ["train","evaluate"]:
            lo,hi=map(pd.Timestamp,w[role])
            eligible=f.is_v1_signal & f.signal_date.between(lo,hi) & f.label_end_date.le(hi) & f.PERMNO.ge(4)
            actual=mask if role=="train" else frame.role.eq("evaluate")&frame.y_excess_5d.notna()
            pd.testing.assert_frame_equal(frame.loc[actual,a.KEY].reset_index(drop=True),
                                          f.loc[eligible,a.KEY].reset_index(drop=True))
        assert a.v2.parameters(bundle["v2"],t)==cfg["pairs"][t["treatment"]]["parameters"]
        return fit(model,t,matrix,frame,mask,cutoff)
    monkeypatch.setattr(a.v2,"fit_candidate",checked_fit)
    a.run(tmp_path,cfg,bundle,tasks,{},checked)
    out=tmp_path/a.RUN
    ledger=json.loads((out/"fit_ledger.json").read_text())
    assert len(set(calls))==len(calls)==len(ledger["entries"])==ledger["budget"]==12
    assert all(e["status"]=="completed" for e in ledger["entries"].values())
    for t in tasks:
        folder=out/t["key"]
        state=json.loads((folder/"preprocessing.json").read_text())
        assert state["context"]["features"]==cfg["context"]+cfg["directional"]
        score=pd.read_csv(folder/"paired_ic.csv")
        np.testing.assert_allclose(score.delta,score.candidate_ic-score.control_ic,atol=1e-14)
        assert (folder/("model.joblib" if t["grid"]=="ridge" else "model.txt")).is_file()
        # Replay the stored estimator using only the evaluation cross sections.
        w=cfg["windows"][t["window"]]
        ec=f.loc[f.is_v1_signal & f.signal_date.between(*w["evaluate"])].reset_index(drop=True)
        from equity_research.models.preprocessing import cross_sectional_rank, transform_features
        original=ec[cfg["original"]] if t["grid"]=="ridge" else cross_sectional_rank(ec,cfg["original"])
        matrix=pd.concat([transform_features(original,state["base"]),
                          transform_features(ec[cfg["context"]+cfg["directional"]],state["context"])],axis=1)
        if t["grid"]=="ridge":
            import joblib
            model=joblib.load(folder/"model.joblib")
        else:
            import lightgbm as lgb
            model=lgb.Booster(model_str=(folder/"model.txt").read_text(encoding="utf8"))
        saved=pd.read_parquet(folder/"predictions.parquet")
        pd.testing.assert_frame_equal(ec[a.KEY],saved[a.KEY].reset_index(drop=True))
        np.testing.assert_allclose(model.predict(matrix),saved.prediction,rtol=1e-12,atol=1e-12)
    result=json.loads((out/"comparison.json").read_text())
    for name,g in pd.DataFrame(result["windows"]).groupby("treatment"):
        assert result["equal_window"][name]["adopt"]==bool(g.delta.mean()>0 and sum(g.delta>0)>=3)
        lo,hi=result["equal_window"][name]["delta_ci95"]
        assert np.isfinite([lo,hi]).all() and lo<=hi
    a.run(tmp_path,cfg,bundle,tasks,{},checked,resume=True)
    assert len(calls)==12


def test_failure_counts_and_blocks_retry(tmp_path,experiment,monkeypatch):
    cfg,bundle,tasks,checked,_=experiment
    out=tmp_path/a.RUN
    with pytest.raises(ValueError,match="extra"):
        a.run(tmp_path,cfg,bundle,tasks+[tasks[-1]],{},checked)
    assert not out.exists()
    def interrupt(*args):
        entry,=a.base.read_json(out/"fit_ledger.json")["entries"].values()
        assert entry["status"]=="running"
        raise KeyboardInterrupt()
    monkeypatch.setattr(a.v2,"fit_candidate",interrupt)
    with pytest.raises(KeyboardInterrupt):
        a.run(tmp_path,cfg,bundle,tasks,{},checked)
    entry,=a.base.read_json(out/"fit_ledger.json")["entries"].values()
    assert entry==dict(kind="fit",status="failed",error_type="KeyboardInterrupt")
    with pytest.raises(ValueError,match="Interrupted or failed"):
        a.run(tmp_path,cfg,bundle,tasks,{},checked,resume=True)


@pytest.mark.parametrize("fault",["feature_key","training_label","window"])
def test_invalid_training_inputs(tmp_path,experiment,fault):
    cfg,bundle,tasks,checked,_=experiment
    w=dict(cfg["windows"]["W1"])
    if fault=="window":
        w["evaluate"]=["2018-01-01","2022-01-01"]
        error=ValueError
    else:
        path=tmp_path/(a.density.LABELS if fault=="training_label" else a.density.PANEL+"/daily_features.parquet")
        f=pd.read_parquet(path)
        rows=f.is_v1_signal & f.signal_date.dt.year.eq(2016) & f.PERMNO.eq(5)
        if fault=="feature_key":
            f.loc[rows,"PERMNO"]=10000
        else:
            f.loc[rows,"y_excess_5d"]+=1
        write(f,path)
        error=AssertionError
    with pytest.raises(error):
        a.window_frame(tmp_path,cfg,bundle,w,checked["W1"][0])
    assert not (tmp_path/a.RUN).exists()
