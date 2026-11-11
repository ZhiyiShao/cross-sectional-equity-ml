"""Synthetic tests for the stage C model recipes and stage D risk budgets."""
import copy

import numpy as np
import pandas as pd

from equity_research.backtest import portfolio_v3 as portfolio
from equity_research.models import stage_b_v3 as b
from equity_research.models.ensemble_v3 import fit_component, rank_mean
from test_stage_b_v3 import synthetic, stage_a_fixture


def test_components_use_training_only(tmp_path, synthetic):
    cfg,bundle,tasks,checked,_=synthetic
    w=cfg["windows"]["W1"]
    frame=b.prepare_window(tmp_path,cfg,bundle,w,checked["W1"][0])
    recipes=[("ridge_enhanced",dict(kind="mlp",seed=11,hidden=[4],alpha=.001,epochs=2,learning_rate=.001)),
             ("lightgbm_enhanced",dict(kind="tree",seed=11,feature_fraction=.8)),
             ("ranker_enhanced",dict(kind="tree",seed=29,feature_fraction=.8))]
    for family,recipe in recipes:
        small=copy.deepcopy(cfg)
        small["families"][family]["parameters"].update(n_estimators=4,min_child_samples=2)
        model,state,matrix=fit_component(frame,recipe,family,bundle,small,w)
        evaluation=frame.role.eq("evaluate")
        def predict(m,x):
            return m.predict(x.to_numpy() if recipe["kind"]=="mlp" else x)
        scores=predict(model,matrix.loc[evaluation])
        other=frame.copy()
        other.loc[evaluation,"y_excess_5d"]=1000
        model2,state2,x=fit_component(other,recipe,family,bundle,small,w)
        np.testing.assert_allclose(scores,predict(model2,x.loc[evaluation]),rtol=0,atol=0)
        assert state==state2 and np.isfinite(scores).all()


def test_equal_rank_uses_eligible_same_date():
    f=pd.DataFrame(dict(signal_date=[1,1,1,2,2],PERMNO=[1,2,3,1,2],
                        signal_eligible=[True,True,False,True,True],a=[1,2,8,4,4],b=[1,1,3,1,0]))
    np.testing.assert_allclose(rank_mean(f,["a","b"]),[.625,.875,0,.875,.625])


def test_loadings_and_optimizer():
    rng=np.random.default_rng(17)
    history=pd.DataFrame(rng.normal(0,.015,(100,80)))
    history.iloc[:80,0]=np.nan
    pcs=portfolio.pc_loadings(history)
    assert pcs.shape==(79,3)
    g=pd.DataFrame(dict(PERMNO=np.arange(80),industry=np.tile(["a","b"],40),market_beta_60=rng.normal(1,.2,80),
        annual_vol=rng.uniform(.1,.6,80),score=rng.normal(size=80)))
    limits=dict(industry_active=.02,beta_active=.1,max_weight=.02,annual_stock_vol_active_upper=.1)
    prior=dict.fromkeys(g.PERMNO,1/80)
    for risk in ["pca","vol"]:
        def solve(penalty):
            return pd.Series(portfolio.weights(g,"score",limits,dict(risk=risk,pc_band=.02,penalty=penalty),prior,pcs))
        w=solve(0)
        assert abs(w.sum()-1)<1e-9 and w.max()<=.02000001
        assert (solve(.15)-1/80).abs().sum()<=(w-1/80).abs().sum()+1e-9
        if risk=="pca":
            assert np.max(np.abs((w-1/80)@pcs.reindex(w.index).fillna(pcs.median())))<=.02000001
