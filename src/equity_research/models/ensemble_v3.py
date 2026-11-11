"""Model recipes used by stage C and its later frozen final fit."""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from equity_research.models import baseline_v3 as base
from equity_research.models import parameter_robustness as v2


def fit_component(frame, recipe, family, bundle, config, window):
    mask=frame.role.eq("train") & frame.y_excess_5d.notna()
    state=base.build_state(frame,mask,config["original"],config["context"],family,window,config["preprocessing"])
    matrix=v2.check_state(state,frame,mask,config["original"],config["context"],family,window)
    with threadpool_limits(limits=4):
        if recipe["kind"]=="mlp":
            model=MLPRegressor(hidden_layer_sizes=tuple(recipe["hidden"]),activation="relu",solver="adam",
                alpha=recipe["alpha"],learning_rate_init=recipe["learning_rate"],shuffle=False,
                batch_size="auto",early_stopping=False,random_state=recipe["seed"])
            train=frame.loc[mask]
            base.require(train.label_end_date.le(pd.Timestamp(window["train"][1])).all(),"MLP cutoff")
            batches=[(matrix.loc[g.index].to_numpy(),g.y_excess_5d.to_numpy()) for _,g in train.groupby("signal_date")]
            rng=np.random.default_rng(recipe["seed"])
            for _ in range(recipe["epochs"]):
                for i in rng.permutation(len(batches)):
                    x,y=batches[i]
                    model.batch_size=len(y)
                    model.partial_fit(x,y)
        else:
            import lightgbm as lgb
            params=dict(config["families"][family]["parameters"],random_state=recipe["seed"],
                        colsample_bytree=recipe["feature_fraction"])
            cls=lgb.LGBMRanker if family.startswith("ranker") else lgb.LGBMRegressor
            model=cls(**params)
            v2.fit_candidate(model,dict(design=family),matrix,frame,mask,window["train"][1])
    return model,state,matrix


def rank_mean(frame,columns):
    base.require(not frame.duplicated(["signal_date","PERMNO"]).any(),"Duplicate score keys")
    eligible=frame.signal_eligible
    base.require(np.isfinite(frame.loc[eligible,columns]).all().all(),"Missing member score")
    scores=frame[columns].where(eligible,axis=0)
    ranks=scores.groupby(frame.signal_date).rank(method="average",pct=True)
    # Ineligible predictions never enter IC, rank population or portfolios.
    return ranks.mean(axis=1).fillna(0.)
