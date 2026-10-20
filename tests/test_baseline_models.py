"""Synthetic checks for development isolation and ranking evaluation."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from equity_research.data.ingest import ROOT
from equity_research.models.common import (load_development_data,sample_masks,fit_preprocessing,
    transform_features,daily_rank_ic,circular_block_interval,model_grid,new_model,save_lightgbm,load_lightgbm)


def config():
    return json.loads((ROOT/'config/baseline_models.json').read_text(encoding='utf-8'))


def test_parquet_filter_excludes_test_rows_and_preserves_unlabeled_candidates(tmp_path):
    frame=pd.DataFrame({'PERMNO':[1,2,1,1],
        'signal_date':pd.to_datetime(['2019-01-02','2019-01-02','2020-01-02','2022-01-03']),
        'split':['train','train','validation','test'],
        'label_end_date':pd.to_datetime(['2019-01-10','2019-01-10','2020-01-10','2022-01-11']),
        'signal_eligible':[True]*4,'eligible_main_sample':[True,False,True,True],
        'purged_boundary':[False]*4,'label_status':['ok','unresolved_delist_MV','ok','ok'],
        'y_excess_5d':[.01,np.nan,.02,1e99],'y_excess_5d_event_proxy':[np.nan]*4,
        'label_proxy_assumption':[False]*4,'feature':[1.,2.,3.,1e99]})
    path=tmp_path/'data.parquet'; frame.to_parquet(path,index=False)
    loaded=load_development_data(path,['feature'],config())
    assert len(loaded)==3 and loaded.feature.max()==3
    assert loaded.y_excess_5d.isna().sum()==1
    masks=sample_masks(loaded,config())
    assert masks['train'].sum()==1 and masks['validation'].sum()==1


def test_preprocessing_fits_training_only_and_transform_does_not_mutate_state():
    train=pd.DataFrame({'a':[0.,1.,2.,np.nan,100.],'b':[3.]*5})
    state=fit_preprocessing(train,[.005,.995]); frozen=json.dumps(state)
    extreme=pd.DataFrame({'a':[1e50,np.nan],'b':[-1e50,np.nan]})
    transformed=transform_features(extreme,state)
    assert np.isfinite(transformed).all().all() and transformed.b.eq(0).all()
    assert json.dumps(state)==frozen
    assert state['median'][0]==1.5 and state['fitted_rows']==5
    assert state!=fit_preprocessing(pd.concat([train,extreme]),[.005,.995])


def test_all_missing_training_feature_fails_and_feature_order_is_checked():
    with pytest.raises(ValueError):
        fit_preprocessing(pd.DataFrame({'a':[np.nan]}),[.005,.995])
    x=pd.DataFrame({'a':[1.,2.],'b':[2.,4.]})
    with pytest.raises(AssertionError):
        transform_features(x[['b','a']],fit_preprocessing(x,[.005,.995]))


def test_rank_ic_is_cross_sectional_with_average_ties():
    dates=pd.to_datetime(['2020-01-02']*4+['2020-01-09']*3)
    pred=[1,1,2,3,3,2,1]; target=[1,3,2,4,1,2,3]
    got=daily_rank_ic(dates,pred,target)
    assert np.isclose(got.iloc[0],spearmanr(pred[:4],target[:4]).statistic)
    assert np.isclose(got.iloc[1],-1.)
    assert not np.isclose(got.mean(),spearmanr(pred,target).statistic)


def test_constant_score_has_undefined_ic_not_zero():
    ic=daily_rank_ic(['2020-01-02']*3,[1,1,1],[1,2,3])
    assert np.isnan(ic.iloc[0])


def test_paired_block_bootstrap_preserves_constant_difference_and_seed():
    result=circular_block_interval(np.full(30,.02),10,100,42)
    assert np.isclose(result['ci_low'],.02) and np.isclose(result['ci_high'],.02)
    a=np.sin(np.arange(30))
    assert circular_block_interval(a,5,100,42)==circular_block_interval(a,5,100,42)
    with pytest.raises(ValueError):
        circular_block_interval([np.nan]*30,5,100,42)


def test_fixed_equal_grid_budget_and_no_test_authorization():
    cfg=config()
    assert len(model_grid(cfg,'ridge'))==len(model_grid(cfg,'lightgbm'))==20
    assert not cfg['test_performance_authorized'] and not cfg['lightgbm']['early_stopping']


@pytest.mark.parametrize('name',['ridge','lightgbm'])
def test_models_fit_same_synthetic_features_with_finite_predictions(name):
    rng=np.random.default_rng(7)
    X=pd.DataFrame(rng.normal(size=(800,3)),columns=['a','b','c'])
    y=X.a*.01+X.b**2*.02
    cfg=config()
    model=new_model(cfg,name,model_grid(cfg,name)[0]).fit(X,y)
    p=model.predict(X.iloc[:20])
    assert len(p)==20 and np.isfinite(p).all()


def test_lightgbm_roundtrip_in_unicode_directory(tmp_path):
    X=pd.DataFrame({'feature':np.arange(800,dtype=float)})
    model=new_model(config(),'lightgbm',model_grid(config(),'lightgbm')[0]).fit(X,np.sin(X.feature))
    path=tmp_path/'量化模型'; path.mkdir()
    save_lightgbm(model,path/'模型.txt')
    loaded=load_lightgbm(path/'模型.txt')
    np.testing.assert_allclose(model.predict(X),loaded.predict(X,num_threads=2),atol=1e-12)
