"""Independent checks of local ranker artifacts; never fit or inspect test data."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata,spearmanr
from equity_research.models.ranking_selection import OUT,SOURCE,PUBLIC
from equity_research.models.context_selection import period_mask
from equity_research.models.preprocessing import cross_sectional_rank,transform_features
from equity_research.models.common import load_lightgbm,sha
from equity_research.backtest.market import ROOT

pytestmark=pytest.mark.skipif(not (OUT/'verification.json').exists(),reason='Local ranking experiment absent')


def test_ranker_budget_group_labels_and_boundaries():
    ledger=json.loads((OUT/'fit_ledger.json').read_text())
    assert len(ledger)==16 and len({(r['window'],r['design'],r['candidate']) for r in ledger})==16
    for wi in [1,2]:
        labels=pd.read_parquet(OUT/f'w{wi}_training_labels_local.parquet')
        for date,g in labels.groupby('signal_date'):
            expected=np.minimum(9,np.floor(10*(rankdata(g.y_excess_5d,method='average')-.5)/len(g))).astype(int)
            np.testing.assert_array_equal(g.relevance,expected)
        for r in [r for r in ledger if r['window']==wi]:
            assert r['train_rows']==len(labels) and r['queries']==labels.signal_date.nunique()
            assert labels.label_end_date.max()<pd.Timestamp(r['eval_start'])
    old=json.loads((SOURCE/'fit_ledger.json').read_text())
    assert len(old)+len(ledger)==48


def test_ranker_selected_predictions_ic_and_preprocessing():
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    f=pd.read_parquet(SOURCE/'development_features_local.parquet')
    assert f.signal_date.max()<pd.Timestamp('2022-01-01')
    features=json.loads((ROOT/'config/features_labels.json').read_text())['features']
    ranked=cross_sectional_rank(f,features)
    selected=json.loads((OUT/'selected.json').read_text())
    metrics=pd.read_csv(PUBLIC/'candidate_metrics.csv').groupby(['design','candidate']).rank_ic.mean()
    daily=pd.read_csv(OUT/'candidate_daily_ic.csv',parse_dates=['signal_date'])
    for row in selected:assert row['candidate']==metrics.loc[row['design']].idxmax()
    for wi,window in enumerate(plan['parent']['development_windows'],1):
        pred=pd.read_parquet(OUT/f'w{wi}_predictions_local.parquet')
        ec=f.signal_date.between(*window['evaluate']);mask=period_mask(f,window['evaluate'])
        pd.testing.assert_frame_equal(pred[['PERMNO','signal_date']],f.loc[ec,['PERMNO','signal_date']].reset_index(drop=True))
        pred.index=f.index[ec]
        for row in selected:
            design=row['design'];name=f"{design}_{row['candidate']}"
            statepath=OUT/'preprocessing'/f'w{wi}_{design}.json'
            assert sha(statepath)==sha(SOURCE/'preprocessing'/f'w{wi}_{design.replace("ranker_","lightgbm_")}.json')
            state=json.loads(statepath.read_text());X=transform_features(ranked,state['base'])
            if 'context' in state:X=pd.concat([X,transform_features(f[state['context']['features']],state['context'])],axis=1)
            model=load_lightgbm(OUT/'models'/f'w{wi}_{name}.txt')
            assert 'lambdarank' in model.dump_model()['objective']
            np.testing.assert_allclose(model.predict(X.loc[ec],num_threads=4),pred[name],atol=1e-12)
            actual={d:spearmanr(pred.loc[g.index,name],g.y_excess_5d).statistic for d,g in f[mask].groupby('signal_date')}
            expected=daily[daily.window.eq(wi)&daily.design.eq(design)&daily.candidate.eq(row['candidate'])].set_index('signal_date').rank_ic
            np.testing.assert_allclose(pd.Series(actual).reindex(expected.index),expected,atol=1e-12)


def test_ranker_portfolio_constraints():
    x=pd.read_csv(PUBLIC/'exposures.csv');x=x[x.stage.eq('target')]
    assert len(x)==200
    assert x.max_weight.le(.02+1e-8).all() and x.max_industry_active.le(.02+1e-8).all()
    assert (x.beta-x.benchmark_beta).abs().le(.1+1e-8).all()
    assert (x.annual_stock_vol-x.benchmark_vol).le(.1+1e-8).all()
