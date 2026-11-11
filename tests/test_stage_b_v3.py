"""Stage B contracts: all observations, labels, predictions and weights are synthetic."""
import json
import numpy as np
import pandas as pd
import pytest

from equity_research.data.targets_v3 import security_labels
from equity_research.models import stage_b_v3 as b
from test_stage_a_v3 import experiment as stage_a_fixture, write


def test_default_public_only_and_exact_schedule(monkeypatch):
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Unexpected observations"))
    assert b.main()==dict(tasks=36,new_fits_planned=32,reuse_planned=4,formal_fits_started=0)
    cfg,_,tasks=b.plan(b.ROOT)
    assert len(cfg["original"]+cfg["context"])==33
    assert all((t["kind"]=="reuse")== (t["design"]=="ranker_enhanced" and t["target"]=="rank") for t in tasks)
    with pytest.raises(ValueError):
        b.main(resume=True)


def test_horizon_entry_exit_compounding_and_truncation():
    dates=pd.bdate_range("2017-11-01",periods=60)
    market=pd.Series(np.arange(60)*0.0001,index=dates)
    stock=np.linspace(.0001,.012,60)
    cfg=b.base.read_json(b.ROOT/"config/universe.json")
    hist=pd.DataFrame(dict(DlyCalDt=dates,PERMNO=1,DlyRet=stock,DlyRetDurFlg="D1",DlyRetMissFlg="NA",
                           DlyPrc=10.,DlyClose=10.,DlyVol=1000.))
    for key,values in {**cfg["regular_trading_filters"],**cfg["security_filters"]}.items():
        hist[key]=values[0]
    hs=[1,5,10,20]
    signals=dates[[0,39,40,57]]
    got=security_labels(hist,market,signals,cfg,hs)
    for j,t in enumerate([0,39,40,57]):
        for h in hs:
            exit_index=t+1+h
            row=got.iloc[j]
            if exit_index>=len(dates):
                assert pd.isna(row[f"end_{h}"]) and pd.isna(row[f"excess_{h}"])
                assert row[f"status_{h}"]=="right_censored"
            else:
                assert row[f"end_{h}"]==dates[exit_index]
                expected=np.prod(1+stock[t+2:exit_index+1])-np.prod(1+market.iloc[t+2:exit_index+1])
                np.testing.assert_allclose(row[f"excess_{h}"],expected,atol=1e-14)
    # Dropping future sessions cannot change labels that were already resolved.
    short=security_labels(hist.iloc[:20],market.iloc[:20],dates[:1],cfg,hs)
    for h in [1,5,10]:
        assert short.at[0,f"excess_{h}"]==got.at[0,f"excess_{h}"]
    assert short.at[0,"status_20"]=="right_censored"


@pytest.fixture
def synthetic(tmp_path,stage_a_fixture):
    _,bundle,_,_,f=stage_a_fixture
    cfg,_,tasks=b.plan(b.ROOT)
    snapshot=tmp_path/bundle["plan"]["snapshot"]
    ref=pd.read_parquet(snapshot)
    ref["industry"]=np.where(ref.PERMNO%2,"01","02")
    write(ref,snapshot)
    raw=f.loc[f.is_v1_signal,b.KEY].copy()
    cal=pd.bdate_range("2015-01-01","2022-01-14")
    pos=cal.get_indexer(raw.signal_date)
    for h in cfg["horizons"]:
        ok=pos+1+h<len(cal)
        raw[f"end_{h}"]=pd.to_datetime(np.where(ok,cal[np.minimum(pos+1+h,len(cal)-1)].to_numpy(),np.datetime64("NaT")))
        raw[f"status_{h}"]=np.where(ok,"ok","right_censored")
        raw[f"excess_{h}"]=np.where(ok,f.loc[f.is_v1_signal,"y_excess_5d"]*h/5,np.nan)
    write(raw,tmp_path/b.targets_v3.DIRECTORY/"labels.parquet")
    checked={wid:b.density.evaluation(tmp_path,bundle,w,[t for t in tasks if t["window"]==wid])
             for wid,w in cfg["windows"].items()}
    return cfg,bundle,tasks,checked,f


def test_residual_rank_and_evaluation_isolation():
    rng=np.random.default_rng(7)
    n=55
    frame=pd.DataFrame(dict(signal_date=pd.Timestamp("2018-01-02"),industry=np.where(np.arange(n)%2,"02","Unknown"),
        log_market_cap=rng.normal(size=n),market_beta_60=rng.normal(size=n),y_excess_5d=rng.integers(0,6,n).astype(float),
        role=["train"]*50+["evaluate"]*5))
    frame.loc[0,"market_beta_60"]=np.nan
    for target in ["rank","residual"]:
        v=b.training_view(frame,target)
        pd.testing.assert_frame_equal(frame.iloc[50:],v.iloc[50:])
        mask=frame.role.eq("train")
        if target=="rank":
            y=frame.loc[mask,"y_excess_5d"].to_numpy()
            oracle=[((y<q).sum()+.5*(y==q).sum())/len(y)-.5 for q in y]
            np.testing.assert_allclose(v.loc[mask,"y_excess_5d"],oracle)
        else:
            assert pd.isna(v.at[0,"y_excess_5d"])
            g=v.loc[mask].dropna(subset=["y_excess_5d"])
            X=np.column_stack([pd.get_dummies(g.industry,dtype=float),g[["log_market_cap","market_beta_60"]]])
            np.testing.assert_allclose(X.T@g.y_excess_5d,0,atol=1e-10)


def test_complete_run_only_bounded_training_and_original_five_day_ic(tmp_path,synthetic,monkeypatch):
    cfg,bundle,tasks,checked,f=synthetic
    original_fit=b.v2.fit_candidate
    calls=[]
    cal=pd.bdate_range("2015-01-01","2022-01-14")
    def inspect_fit(model,t,matrix,frame,mask,cutoff):
        calls.append(t["key"])
        assert matrix.shape[1]==33
        w=cfg["windows"][t["window"]]
        lo,hi=map(pd.Timestamp,w["train"])
        expected=f.loc[f.is_v1_signal & f.signal_date.between(lo,hi) & f.PERMNO.ge(4) & f.label_end_date.le(hi)].copy()
        if t["target"]=="multi":
            limit=cal.get_loc(hi) if hi in cal else cal.searchsorted(hi,side="right")-1
            expected=expected.loc[cal.get_indexer(expected.signal_date)+21<=limit]
            np.testing.assert_allclose(frame.loc[mask,"y_excess_5d"],expected.y_excess_5d)
        pd.testing.assert_frame_equal(frame.loc[mask,b.KEY].reset_index(drop=True),expected[b.KEY].reset_index(drop=True))
        oracle=checked[t["window"]][0]
        ev=frame.loc[frame.role.eq("evaluate") & frame.y_excess_5d.notna(),b.KEY+["y_excess_5d"]]
        pd.testing.assert_frame_equal(ev.reset_index(drop=True),oracle[ev.columns])
        return original_fit(model,t,matrix,frame,mask,cutoff)
    monkeypatch.setattr(b.v2,"fit_candidate",inspect_fit)
    b.run(tmp_path,cfg,bundle,tasks,{},checked)
    out=tmp_path/b.RUN
    ledger=b.base.read_json(out/"fit_ledger.json")
    assert len(set(calls))==len(calls)==ledger["budget"]==32
    assert len(ledger["entries"])==36 and all(v["status"]=="completed" for v in ledger["entries"].values())
    result=b.base.read_json(out/"comparison.json")
    assert result["metrics"]["ranker_enhanced_rank"]["ci95"]==[0.,0.]
    assert len(result["metrics"])==9 and len(result["choices"])==3
    b.run(tmp_path,cfg,bundle,tasks,{},checked,resume=True)
    assert len(calls)==32


def test_invalid_target_or_failed_fit_no_retry(tmp_path,synthetic,monkeypatch):
    cfg,bundle,tasks,checked,_=synthetic
    with pytest.raises(ValueError,match="budget"):
        b.run(tmp_path,cfg,bundle,tasks+[tasks[0]],{},checked)
    def fail(*args):
        ledger=b.base.read_json(tmp_path/b.RUN/"fit_ledger.json")
        assert [v["status"] for v in ledger["entries"].values()]==["running"]
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(b.v2,"fit_candidate",fail)
    with pytest.raises(RuntimeError,match="synthetic"):
        b.run(tmp_path,cfg,bundle,tasks,{},checked)
    with pytest.raises(ValueError,match="Interrupted or failed"):
        b.run(tmp_path,cfg,bundle,tasks,{},checked,resume=True)


def test_invalid_horizon_data_and_window(tmp_path):
    w=dict(train=["2015-01-01","2017-12-31"],evaluate=["2018-01-01","2018-12-31"])
    rows=[]
    for i in range(5):
        row=dict(signal_date=pd.Timestamp("2017-11-01"),PERMNO=i)
        for h in [1,5,10,20]:
            row.update({f"end_{h}":row["signal_date"]+pd.offsets.BDay(1+h), f"excess_{h}":h*.01,
                        f"status_{h}":"ok"})
        rows.append(row)
    rows[1]["end_20"]=pd.Timestamp("2018-01-02")
    rows[2]["status_10"]="span_crosses_exit"
    rows[3]["excess_1"]=np.nan
    rows[4]["excess_5"]=np.inf
    p=tmp_path/"labels.parquet"
    write(pd.DataFrame(rows),p)
    out=b.multi_labels(p,w,[1,5,10,20])
    assert out.PERMNO.tolist()==[0]
    np.testing.assert_allclose(out.multi,[.05])
    rows[0]["end_10"]=rows[0]["end_1"]
    write(pd.DataFrame(rows),p)
    with pytest.raises(ValueError,match="endpoint"):
        b.multi_labels(p,w,[1,5,10,20])
    with pytest.raises(ValueError,match="Window"):
        b.multi_labels(tmp_path/"nonexistent",dict(w,evaluate=["2018-01-01","2022-01-01"]),[1,5,10,20])


def test_label_builder_publishes_only_after_five_day_parity(tmp_path):
    from equity_research.data.targets_v3 import build, DIRECTORY
    dates=pd.bdate_range("2021-11-01","2022-01-14")
    cutoff=pd.Timestamp("2021-12-31")
    raw_source=tmp_path/"data/interim/crsp_audit_v1"
    market=np.linspace(.0001,.004,len(dates))
    returns=np.linspace(-.002,.01,len(dates))
    write(pd.DataFrame(dict(date=dates.strftime("%Y-%m-%d"),vwretd=market)),raw_source/"calendar.parquet")
    hist=pd.DataFrame(dict(DlyCalDt=dates,PERMNO=11,DlyRet=returns,DlyRetDurFlg="D1",DlyRetMissFlg="NA",
                           DlyPrc=10.,DlyClose=10.,DlyVol=1000.))
    uc=b.base.read_json(b.ROOT/"config/universe.json")
    for key,values in {**uc["regular_trading_filters"],**uc["security_filters"]}.items():
        hist[key]=values[0]
    write(hist,raw_source/"part.parquet")
    (raw_source/"ingest.json").write_text(json.dumps(dict(parts=["part.parquet"])))
    (tmp_path/"config").mkdir()
    (tmp_path/"config/universe.json").write_text(json.dumps(uc))
    (tmp_path/b.CONFIG).write_bytes((b.ROOT/b.CONFIG).read_bytes())
    cfg,_,_=b.plan(b.ROOT)
    expected=[]
    for t in range(0,len(dates),5):
        valid=t+6<len(dates) and dates[t+6]<=cutoff
        expected.append(dict(signal_date=dates[t],PERMNO=11,is_v1_signal=True,
                             label_end_date=dates[t+6] if valid else pd.NaT,label_status="ok" if valid else "right_censored",
                             y_excess_5d=np.prod(1+returns[t+2:t+7])-np.prod(1+market[t+2:t+7]) if valid else np.nan))
    ref=pd.DataFrame(expected)
    # Match the documented stock-minus-market arithmetic, including float operation order.
    for t in range(0,len(dates),5):
        if pd.notna(ref.loc[t//5,"label_end_date"]):
            ref.loc[t//5,"y_excess_5d"]=(np.prod(1+returns[t+2:t+7])-1)-(np.prod(1+market[t+2:t+7])-1)
    write(ref.assign(y_excess_5d=1.),tmp_path/b.density.LABELS)
    with pytest.raises(AssertionError):
        build(tmp_path,cfg,hist.columns.tolist())
    assert not (tmp_path/DIRECTORY).exists()
    write(ref,tmp_path/b.density.LABELS)
    report=build(tmp_path,cfg,hist.columns.tolist())
    got=pd.read_parquet(tmp_path/DIRECTORY/"labels.parquet")
    assert report["rows"]==len(ref.loc[ref.signal_date<=cutoff])
    assert got.signal_date.max()<=cutoff
    for h in cfg["horizons"]:
        assert got[f"end_{h}"].dropna().max()<=cutoff
        assert got.loc[got[f"end_{h}"].isna(),f"excess_{h}"].isna().all()
    with pytest.raises(FileExistsError):
        build(tmp_path,cfg,hist.columns.tolist())
