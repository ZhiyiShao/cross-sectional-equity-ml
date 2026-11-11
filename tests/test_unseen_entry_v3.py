"""Exercise the public entry points on invented dates/prices and isolated Git refs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import unseen_guard_v3 as guard
import unseen_v3 as u
import unseen_portfolios_v3 as portfolios
from unseen_inputs_v3 import Sources


def save(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_bytes((json.dumps(value,indent=2)+"\n").encode())
    return file


@pytest.fixture
def repo(tmp_path):
    root=tmp_path
    (root/"outputs").mkdir()
    source=save(root/"source.json",{"test":1})
    seal=save(root/guard.SEAL,{"sources":{source.name:guard.digest(source)}})
    guard.git(root,"init","-q")
    guard.git(root,"config","core.autocrlf","false")
    guard.git(root,"add","--",source.name,guard.SEAL)
    guard.git(root,"commit","-qm","Synthetic release")
    guard.git(root,"tag",guard.TAG.removeprefix("refs/tags/"))
    for s in guard.STAGES: guard.git(root,"update-ref",guard.PREFIX+s,"HEAD","0"*40)
    return root


def test_release_tamper_and_missing_ref(repo):
    m=save(repo/"request.json",{})
    (repo/"source.json").write_bytes(b"edited")
    with pytest.raises(ValueError,match="Unsealed input"):
        with guard.attempt(repo,"prediction",m): pytest.fail("Reserved after tampering")
    save(repo/"source.json",{"test":1})
    guard.git(repo,"update-ref","-d",guard.PREFIX+"prediction")
    with pytest.raises(ValueError,match="Git audit"):
        with guard.attempt(repo,"prediction",m): pytest.fail("Recreated removed journal")


def test_crash_and_lock(repo):
    m=save(repo/"request.json",{})
    script_dir=Path(guard.__file__).parent
    code=("import os,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
          "import unseen_guard_v3 as g;r=Path(sys.argv[2]);"
          "ctx=g.attempt(r,'prediction',r/'request.json');j=ctx.__enter__();"
          "j.event('labels_opened',labels_exposed=True);os._exit(19)")
    assert subprocess.run([sys.executable,"-c",code,str(script_dir),str(repo)]).returncode==19
    with pytest.raises(ValueError,match="already reserved"):
        with guard.attempt(repo,"prediction",m): pass
    with guard.attempt(repo,"prediction",m,resume=True) as j:
        assert j.record["labels_exposed"] and j.record["attempt"]==2
        with pytest.raises((OSError,BlockingIOError)):
            with guard.exclusive(repo,"prediction"): pass
        j.event("completed")


def test_manifest_requires_reviewed_overlap(repo,monkeypatch):
    from check_v3_vintage import overlap_digest
    monkeypatch.setattr(u,"ROOT",repo)
    g=pd.DataFrame(dict(PERMNO=[8],DlyCalDt=[pd.Timestamp("2025-01-03")],DlyRet=[.01],
                        DlyRetx=[.01],vwretd=[.001],DlyRetMissFlg=["NA"],DlyRetDurFlg=["D1"],DlyDelFlg=["N"]))
    part=repo/"part.parquet"
    g.to_parquet(part,index=False)
    cal=repo/"calendar.parquet"
    pd.DataFrame(dict(date=["2025-01-03"],vwretd=[.001])).to_parquet(cal,index=False)
    receipt=save(repo/"outputs/v3_unseen_2026_vintage/overlap.json",dict(new_overlap_sha256=overlap_digest(g)))
    fields=dict(parts=[dict(path=part.name,sha256=guard.digest(part))],overlap_sha256=guard.digest(receipt),
                new_calendar=dict(path=cal.name,sha256=guard.digest(cal)),data_through="2026-12-31",
                cohort="CRSP_US_equities_2026",vintage_decision="old_through_2025_new_from_2026",decision_reason="synthetic")
    file=save(repo/"inputs.json",fields)
    # Only the 2025 projection is materialized in this test.
    monkeypatch.setattr(u,"Sources",lambda *args:args[-1])
    assert u.approved_source(file)==([part],cal)
    g.assign(DlyRet=.02).to_parquet(part,index=False)
    fields["parts"][0]["sha256"]=guard.digest(part)
    save(file,fields)
    with pytest.raises(ValueError,match="reviewed 2025 overlap"): u.approved_source(file)


def invented_inputs(repo):
    days=pd.bdate_range("2014-01-02","2015-04-30")
    start=pd.Timestamp("2015-01-01")
    n=60
    data=pd.DataFrame({"PERMNO":np.repeat(np.arange(n),len(days)),"DlyCalDt":np.tile(days,n)})
    i=data.PERMNO
    t=np.tile(np.arange(len(days)),n)
    r=.001*np.sin(.03*t+.1*i)+.0001*i/n
    price=30*(1+r).groupby(i).cumprod()
    cfg=u.final.b.base.read_json(u.final.ROOT/"config/universe.json")
    cfg=dict(cfg,max_securities=55)
    for k,v in {**cfg["regular_trading_filters"],**cfg["security_filters"]}.items(): data[k]=v[0]
    for k,v in dict(PERMCO=i,DlyPrc=price,DlyClose=price,DlyHigh=price*1.01,DlyLow=price*.99,
                    DlyVol=10000+i*100,ShrOut=1e5,DlyCap=1e7,DlyCumFacShr=1.,DlyRet=r,DlyRetx=r,
                    DlyCumFacPr=1.,DlyRetMissFlg="NA",DlyRetDurFlg="D1",SICCD=3500,
                    SecInfoStartDt=days[0],SecInfoEndDt=days[-1]).items(): data[k]=v
    p=repo/"data/interim/crsp_audit_v1";p.mkdir(parents=True)
    save(p/"ingest.json",dict(parts=["part.parquet"],first_date="2014-01-02",last_date="2015-04-30"))
    data.to_parquet(p/"part.parquet",index=False)
    pd.DataFrame(dict(date=days.strftime("%Y-%m-%d"),vwretd=.0005*np.cos(np.arange(len(days))*.1))).to_parquet(p/"calendar.parquet")
    save(repo/"config/universe.json",cfg)
    return Sources(repo,start,days[-1]),data


def test_public_entries_resume_sealed_predictions_and_paths(repo,monkeypatch):
    real_root=u.ROOT
    source,raw=invented_inputs(repo)
    request=save(repo/"prediction.json",{"synthetic":True})
    monkeypatch.setattr(u,"ROOT",repo)
    monkeypatch.setattr(portfolios,"ROOT",repo)
    monkeypatch.setattr(u,"START",source.start)
    monkeypatch.setattr(u,"END",source.end)
    monkeypatch.setattr(u,"approved_source",lambda f:source)
    monkeypatch.setattr(u.final,"locked",lambda:{"portfolio":dict(risk="pca",pc_band=.02,penalty=.15)})
    def fake_score(f,year):
        p=f[u.KEY+["signal_eligible"]].copy()
        for n in ["v3_final",*u.final.V1,u.V2,"momentum20"]: p[n]=p.PERMNO*.1
        return p
    monkeypatch.setattr(u,"score",fake_score)
    original=u.panel
    def failure(*a,**kw):
        j=guard.Journal(repo,"prediction")
        assert j.record["status"]=="labels_opened" and len(j.record["saved_files"])==5
        raise RuntimeError("simulated post-release fault")
    monkeypatch.setattr(u,"panel",failure)
    with pytest.raises(RuntimeError,match="post-release"): u.execute(request)
    pred=repo/"outputs/v3_unseen_2026_prediction"
    checkpoint=pred/"2025_predictions_local.parquet"
    backup=checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(u,"score",lambda *a:pytest.fail("Rescoring after label access"))
    monkeypatch.setattr(u,"panel",original)
    with pytest.raises(ValueError,match="checkpoint changed"): u.execute(request,resume=True)
    checkpoint.write_bytes(backup)
    other=save(repo/"changed.json",{"different":True})
    with pytest.raises(ValueError,match="identity mismatch"): u.execute(other,resume=True)
    u.execute(request,resume=True)
    result=json.loads((pred/"prediction_summary.json").read_bytes())
    assert result["2025"]["primary_control"]=="raw__ridge_03"

    def record(p): return dict(path=p.name,sha256=guard.digest(p))
    market=repo/"market.parquet"
    raw.assign(sellable=True,buyable=True,ordinary_return=True,known_terminal=False).to_parquet(market,index=False)
    cal=save(repo/"settlement.json",dict(non_settlement_weekdays=[],supported_windows=[["2014-01-01","2015-05-31"]],
             exchange_supported_window=["2014-01-01","2015-05-31"],exchange_closed_weekdays=[]))
    registry=save(repo/"events.json",{"events":[]})
    audit=save(repo/"audit.json",dict(unresolved_events=0,settlement_verified=True,market_verified=True,
               execution_market=record(market),settlement_calendar=record(cal),event_registry=record(registry)))
    command=save(repo/"portfolio.json",dict(prediction_manifest=record(request),execution_audit=record(audit)))
    shutil.copyfile(real_root/"config/research_v2.json",repo/"config/research_v2.json")
    for y in [2025,2021]:
        folder=repo/f"outputs/v3_final_{y}";folder.mkdir()
        pc=folder/"risk_loadings_local.parquet"
        pd.DataFrame(np.random.default_rng(4).normal(0,.05,(60,3))).to_parquet(pc)
        save(folder/"verification.json",dict(pca_sha256=guard.digest(pc)))
    simulate=portfolios.engine.simulate
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        if len(calls)==2: raise RuntimeError("path fault")
        return simulate(*args,**kwargs)
    monkeypatch.setattr(portfolios.engine,"simulate",interrupted)
    with pytest.raises(RuntimeError,match="path fault"): portfolios.execute(command)
    assert len(guard.Journal(repo,"portfolio").record["paths"])==1
    portfolios.execute(command,resume=True)
    assert len(calls)==13  # 12 completed paths plus one failed, first path reused
    assert len(guard.Journal(repo,"portfolio").record["paths"])==12
    shutil.rmtree(pred)
    for resume in [False,True]:
        with pytest.raises(ValueError,match="already reserved or completed"): u.execute(request,resume)
    # Simulate deletion/reinitialization of the audit branch too: the Git
    # reservation tag must still prevent a second first-use transaction.
    guard.git(repo,"update-ref",guard.PREFIX+"prediction","HEAD")
    with pytest.raises(ValueError,match="Git audit"): u.execute(request)
