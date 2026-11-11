from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import final_v3
import replay_v3
import unseen_v3
import unseen_portfolios_v3 as future_ports
import check_v3_vintage as vintage
from stage_e_v3 import deflated
from test_stage_b_v3 import synthetic,stage_a_fixture


def test_final_ranker_roundtrip_and_horizon(tmp_path,synthetic):
    b=final_v3.b
    cfg,bundle,_,checked,_=synthetic
    w=cfg["windows"]["W1"]
    f=b.prepare_window(tmp_path,cfg,bundle,w,checked["W1"][0])
    f=f.loc[f.role.eq("train")].reset_index(drop=True)
    family="ranker_enhanced"
    spec=dict(signal=family,incumbents={family:dict(target="residual")})
    cfg["families"][family]["parameters"].update(min_child_samples=2,n_estimators=3)
    model,state,_,matrix=final_v3.fit_one(f,"v3_final",cfg,w["train"][1],{},spec)
    cols=b.KEY+cfg["original"]+cfg["context"]+["signal_eligible"]
    predicted=final_v3.apply(f[cols],model,state,family)
    np.testing.assert_allclose(predicted,model.predict(matrix))
    with pytest.raises(ValueError,match="Target"):
        final_v3.apply(f,model,state,family)
    f.loc[f.y_excess_5d.notna(),"label_end_date"]=pd.Timestamp("2018-01-10")
    with pytest.raises(ValueError,match="boundary"):
        final_v3.fit_one(f,"v3_final",cfg,w["train"][1],{},spec)


def test_pair_keys_and_deflation():
    rng=np.random.default_rng(12)
    x=pd.DataFrame(rng.normal(.0004,.01,(500,6)))
    result=deflated(x,0)
    assert 0<=result["probability"]<=1 and result["annualized_hurdle"]>0
    dates=pd.DataFrame(dict(signal_date=pd.Timestamp("2019-01-02"),PERMNO=[1,2],signal_eligible=True))
    cached=dates.assign(score=[.1,.4])
    assert replay_v3.align(dates,cached.iloc[::-1],["score"]).score.tolist()==[.1,.4]
    with pytest.raises(AssertionError):
        replay_v3.align(dates,cached.assign(PERMNO=[2,3]),["score"])


def test_vintage_counts_and_storage_predicate(tmp_path):
    frame=pd.DataFrame(dict(DlyCalDt=pd.to_datetime(["2025-12-30","2025-12-30","2026-01-02"]),
                            PERMNO=[1,2,3],DlyRet=[.01,.02,.03],DlyRetx=[.01,.02,.03],vwretd=.001,
                            DlyDelFlg="N",DlyRetMissFlg="NA",DlyRetDurFlg="D1"))
    path=tmp_path/"projected.parquet"
    frame.to_parquet(path)
    old=vintage.projected(path)
    new=old.copy()
    new.loc[0,"DlyRet"]=.013
    new.loc[1,"PERMNO"]=4
    result=vintage.compare(old,new)
    assert result["old_only"]==result["new_only"]==1 and result["old_rows"]==2
    assert result["columns"]["DlyRet"]["different_nonmissing"]==1
    with pytest.raises(ValueError,match="date breach"):
        vintage.compare(old,frame)


def test_prospective_scoring_uses_saved_cohorts_only():
    # Two simulated future cross-sections exercise the actual serialized model adapters.
    rng=np.random.default_rng(13)
    cfg,_,_=final_v3.b.plan(final_v3.ROOT)
    f=pd.DataFrame(rng.normal(size=(100,21)),columns=cfg["original"])
    f["signal_date"]=np.repeat(pd.to_datetime(["2016-01-04","2016-01-11"]),50)
    f["PERMNO"]=np.tile(np.arange(50),2)
    f["signal_eligible"]=True
    f["industry"]="01"
    for year in [2021,2025]:
        if not (final_v3.ROOT/f"outputs/v3_final_{year}/verification.json").exists():
            pytest.skip("Local frozen cohort required")
        s=unseen_v3.score(f,year)
        names=["v3_final",*final_v3.V1,unseen_v3.V2,"momentum20"]
        assert np.isfinite(s[names]).all().all() and len(s)==100


def test_prospective_portfolio_assignment():
    rng=np.random.default_rng(9)
    days=pd.bdate_range("2026-01-05",periods=7)
    sched=pd.DataFrame(dict(signal_date=days[[0,5]],entry_date=days[[1,6]]))
    frame=pd.DataFrame(dict(signal_date=np.repeat(sched.signal_date,100),PERMNO=np.tile(np.arange(100),2),
                            industry=np.tile(["01","03"],100),market_beta_60=rng.normal(1,.1,200),
                            annual_vol=rng.uniform(.1,.5,200),signal_eligible=True))
    for n in ["v3_final",*final_v3.V1,unseen_v3.V2,"momentum20"]:
        frame[n]=rng.normal(size=len(frame))
    pc=pd.DataFrame(rng.normal(0,.15,(100,3)))
    p=future_ports.portfolio_set(frame,sched,dict(portfolio=dict(risk="pca",penalty=.15,pc_band=.02)),pc)
    assert set(p)=={"v3_final",*final_v3.V1,unseen_v3.V2,"momentum20","equal_weight"}
    for day in sched.entry_date:
        assert len(p["equal_weight"][day])==100 and len(p[final_v3.V1[0]][day])==10
        assert np.isclose(sum(p["v3_final"][day].values()),1)
        assert max(p["v3_final"][day].values())<=.020000001
