"""No production observations or models: check the density runner on invented panels."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from equity_research.models import density_v3 as d


def write(frame, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(path,index=False)


@pytest.fixture
def experiment(tmp_path):
    bundle, schedule = d.plan(d.ROOT)  # frozen public files only
    calendar = pd.bdate_range("2015-01-01","2022-01-14")
    # Invent small cross sections; years, five-session spacing and t+6 are independent oracles.
    dates = calendar[calendar.month.isin([1,12])]
    pos = calendar.get_indexer(dates)
    n = 24
    rng = np.random.default_rng(184)
    f = pd.DataFrame(dict(signal_date=np.repeat(dates,n),PERMNO=np.tile(np.arange(n),len(dates)),
                          is_v1_signal=np.repeat(pos % 5 == 0,n),signal_eligible=True,
                          a=rng.normal(size=len(dates)*n),b=rng.normal(size=len(dates)*n)))
    f["label_status"] = "ok"
    f["label_end_date"] = np.repeat(calendar[np.minimum(pos+6,len(calendar)-1)],n)
    f["y_excess_5d"] = f.a + .1*f.b
    f.loc[f.PERMNO==0,"signal_eligible"] = False
    f.loc[f.PERMNO==1,"label_status"] = "delisted"
    f.loc[f.PERMNO==2,"label_end_date"] = f.signal_date
    f.loc[f.PERMNO==3,"y_excess_5d"] = np.nan
    f.loc[f.PERMNO==4,"label_status"] = "ok_bridged"
    meta = d.KEY+["signal_eligible","is_v1_signal"]
    write(f[meta+["a","b"]], tmp_path/d.PANEL/"daily_features.parquet")
    write(f.drop(columns=["a","b"]),tmp_path/d.LABELS)
    ref = f.loc[f.is_v1_signal]
    write(ref,tmp_path/bundle["plan"]["snapshot"])
    for task in schedule:
        start,end = bundle["windows"][task["window"]]["evaluate"]
        g = ref.loc[ref.signal_date.between(start,end)]
        path = tmp_path/d.CONTROL/task["source"]
        write(g[d.KEY+["signal_eligible"]].assign(prediction=g.a),path/"predictions.parquet")
        valid = g.loc[g.PERMNO.ge(4) & g.label_end_date.le(pd.Timestamp(end))]
        ic = valid.groupby("signal_date")[["a","y_excess_5d"]].corr(method="spearman").xs("a",level=1).y_excess_5d
        ic.rename("rank_ic").to_csv(path/"daily_ic.csv")
    return bundle,schedule,f


def check_all(root,bundle,schedule):
    return {wid:d.evaluation(root,bundle,w,[t for t in schedule if t["window"]==wid])
            for wid,w in bundle["windows"].items()}


def test_default_is_configuration_only(monkeypatch):
    monkeypatch.setattr(d,"inputs",lambda *args:pytest.fail("private input read"))
    assert d.main()==dict(tasks=8,fit_budget=8,formal_fits_started=0)
    with pytest.raises(ValueError,match="Resume"):
        d.main(resume=True)


def test_full_synthetic_fit_and_boundary_oracle(tmp_path,experiment,monkeypatch):
    bundle,schedule,f = experiment
    checked = check_all(tmp_path,bundle,schedule)
    for wid,w in bundle["windows"].items():
        for role in ["train","evaluate"]:
            lo,hi = map(pd.Timestamp,w[role])
            good = f.signal_date.between(lo,hi) & f.label_end_date.le(hi) & f.PERMNO.ge(4)
            if role=="evaluate":
                good &= f.is_v1_signal
            got = d.targets(tmp_path/d.LABELS,w,role,five_only=role=="evaluate")
            pd.testing.assert_frame_equal(got[d.KEY],f.loc[good,d.KEY].reset_index(drop=True))
            assert f.loc[f.signal_date.between(lo,hi),"label_end_date"].gt(hi).any()
    out = tmp_path/"run"
    d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"])
    ledger = d.base.read_json(out/"fit_ledger.json")
    assert len(ledger["entries"]) == ledger["budget"] == 8
    assert all(e["status"]=="completed" for e in ledger["entries"].values())
    for t in schedule:
        s = d.base.read_json(out/t["key"]/"summary.json")
        w = bundle["windows"][t["window"]]
        assert pd.Timestamp(s["max_train_label_end"]) <= pd.Timestamp(w["train"][1])
        assert pd.Timestamp(s["max_eval_label_end"]) <= pd.Timestamp(w["evaluate"][1])
        assert s["eval_rows"] == len(checked[t["window"]][0])
        train = f.signal_date.between(*w["train"]) & f.label_end_date.le(pd.Timestamp(w["train"][1])) & f.PERMNO.ge(4)
        assert s["train_rows"] == train.sum() and (train & ~f.is_v1_signal).any()
    monkeypatch.setattr(d.v2,"fit_candidate",lambda *args:pytest.fail("duplicated fit"))
    d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"],resume=True)
    with pytest.raises(ValueError,match="manifest"):
        d.run_windows(tmp_path,out,bundle,schedule,{"changed":True},checked,["a"],["b"],resume=True)


@pytest.mark.parametrize("mismatch",["candidate","qualified"])
def test_equal_count_wrong_stock_rejected(tmp_path,experiment,mismatch):
    bundle,schedule,_ = experiment
    w = bundle["windows"]["W1"]
    task = schedule[0]
    if mismatch=="candidate":
        path = tmp_path/d.CONTROL/task["source"]/"predictions.parquet"
        f = pd.read_parquet(path)
        f.loc[0,"PERMNO"] = 99
    else:
        path = tmp_path/bundle["plan"]["snapshot"]
        f = pd.read_parquet(path)
        f["PERMNO"] = f.PERMNO.replace({5:99})
    write(f,path)
    with pytest.raises(AssertionError):
        d.evaluation(tmp_path,bundle,w,[task])
    assert not (tmp_path/d.RUN).exists()


@pytest.mark.parametrize("failure",[RuntimeError,KeyboardInterrupt])
def test_failed_attempt_is_durable(tmp_path,experiment,monkeypatch,failure):
    bundle,schedule,_ = experiment
    checked = check_all(tmp_path,bundle,schedule)
    out = tmp_path/"failed"
    def abort(*args):
        ledger = d.base.read_json(out/"fit_ledger.json")
        assert [e["status"] for e in ledger["entries"].values()]==["running"]
        raise failure()
    monkeypatch.setattr(d.v2,"fit_candidate",abort)
    with pytest.raises(failure):
        d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"])
    entry, = d.base.read_json(out/"fit_ledger.json")["entries"].values()
    assert entry == dict(kind="fit",status="failed",error_type=failure.__name__)
    with pytest.raises(ValueError,match="Interrupted or failed"):
        d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"],resume=True)


def test_process_kill_preserves_running_reservation(tmp_path):
    _,tasks = d.plan(d.ROOT)
    out = tmp_path/"hard_stop"
    code = """import json,os,sys
from equity_research.models.robustness_ledger import FitLedger
tasks=json.loads(sys.argv[2])
with FitLedger(sys.argv[1],{},tasks,8) as ledger:
    ledger.perform(tasks[0],lambda _:os._exit(17))
"""
    env = dict(os.environ,PYTHONPATH=str(d.ROOT/"src"))
    result = subprocess.run([sys.executable,"-c",code,str(out),json.dumps(tasks)],env=env)
    assert result.returncode==17
    ledger = d.base.read_json(out/"fit_ledger.json")
    assert list(ledger["entries"].values())==[dict(kind="fit",status="running")]
    with pytest.raises(FileExistsError):
        d.FitLedger(out,{},tasks,8,resume=True)
    # Only remove the dead subprocess lock in this disposable synthetic case.
    (out/".writer.lock").unlink()
    with pytest.raises(ValueError,match="Interrupted or failed"):
        d.FitLedger(out,{},tasks,8,resume=True)


@pytest.mark.parametrize("train_end,eval_end",[("2018-01-01","2018-12-31"),("2017-12-31","2022-01-01")])
def test_invalid_window_stops_before_io(tmp_path,train_end,eval_end):
    w = dict(train=["2015-01-01",train_end],evaluate=["2018-01-01",eval_end])
    with pytest.raises(ValueError,match="Window"):
        d.targets(tmp_path/"must_not_read",w,"train")


def test_clean_pause_resumes_only_remaining_tasks(tmp_path,experiment,monkeypatch):
    bundle,schedule,_ = experiment
    checked = check_all(tmp_path,bundle,schedule)
    out = tmp_path/"paused"
    perform = d.FitLedger.perform
    def pause_before_second_reservation(ledger,task,operation):
        if ledger.used == 1:
            raise KeyboardInterrupt
        return perform(ledger,task,operation)
    monkeypatch.setattr(d.FitLedger,"perform",pause_before_second_reservation)
    with pytest.raises(KeyboardInterrupt):
        d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"])
    original = d.base.read_json(out/"fit_ledger.json")["entries"]
    assert len(original)==1 and next(iter(original.values()))["status"]=="completed"
    monkeypatch.setattr(d.FitLedger,"perform",perform)
    calls = []
    fit = d.v2.fit_candidate
    def count_fit(model,task,*args):
        calls.append(task["key"])
        return fit(model,task,*args)
    monkeypatch.setattr(d.v2,"fit_candidate",count_fit)
    d.run_windows(tmp_path,out,bundle,schedule,{},checked,["a"],["b"],resume=True)
    assert len(calls)==7 and set(original).isdisjoint(calls)
    assert len(d.base.read_json(out/"fit_ledger.json")["entries"])==8


def test_no_ninth_reservation(tmp_path):
    bundle,tasks = d.plan(d.ROOT)
    extra = dict(tasks[0], key="extra")
    with pytest.raises(ValueError,match="extra"):
        d.run_windows(tmp_path,tmp_path/"run",bundle,tasks+[extra],{}, {},[],[])
    assert not (tmp_path/"run").exists()
    with d.FitLedger(tmp_path/"ledger",{},tasks+[extra],8) as ledger:
        def operation(task):
            path = ledger.directory/task["key"]
            path.write_bytes(b"test")
            return [path.name]
        for task in tasks:
            ledger.perform(task,operation)
        with pytest.raises(ValueError,match="exhausted"):
            ledger.perform(extra,operation)
        assert ledger.used==8 and extra["key"] not in ledger.state["entries"]
