"""Optional independent checks of the saved local V2 experiment; no fitting."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from equity_research.backtest.market import ROOT
from equity_research.models.common import load_lightgbm
from equity_research.models.preprocessing import cross_sectional_rank,transform_features
from equity_research.models.context_selection import period_mask

OUT=ROOT/'outputs/context_selection_v2'
pytestmark=pytest.mark.skipif(not (OUT/'verification.json').exists(),reason='Licensed local experiment absent')


def test_saved_selection_predictions_and_independent_ic():
    f=pd.read_parquet(OUT/'development_features_local.parquet')
    assert f.signal_date.max()<pd.Timestamp('2022-01-01')
    cfg=json.loads((OUT/'frozen_plan.json').read_text())
    features=json.loads((ROOT/'config/features_labels.json').read_text())['features']
    ranked=cross_sectional_rank(f,features)
    selected=json.loads((OUT/'selected.json').read_text())
    all_scores=pd.read_csv(ROOT/'results/context_selection_v2/candidate_metrics.csv')
    expected=all_scores.groupby(['design','candidate']).rank_ic.mean()
    for row in selected:
        candidates=expected.loc[row['design']]
        assert row['candidate']==candidates.idxmax()
    daily=pd.read_csv(OUT/'candidate_daily_ic.csv',parse_dates=['signal_date'])
    for wi,window in enumerate(cfg['parent']['development_windows'],1):
        pred=pd.read_parquet(OUT/f'w{wi}_predictions_local.parquet')
        ec=f.signal_date.between(*window['evaluate']);mask=period_mask(f,window['evaluate'])
        pd.testing.assert_frame_equal(pred[['PERMNO','signal_date']].reset_index(drop=True),
            f.loc[ec,['PERMNO','signal_date']].reset_index(drop=True))
        pred.index=f.index[ec]
        for row in selected:
            design=row['design'];name=design+'_'+str(row['candidate'])
            state=json.loads((OUT/'preprocessing'/f'w{wi}_{design}.json').read_text())
            assert pd.Timestamp(state['max_label_end'])<=pd.Timestamp(window['train'][1])
            train=period_mask(f,window['train'])
            assert state['base']['fitted_rows']==int(train.sum())
            if 'context' in state:
                cols=state['context']['features'];a=f.loc[train,cols].to_numpy(float)
                a[~np.isfinite(a)]=np.nan
                low,high=np.nanquantile(a,[.005,.995],axis=0)
                median=np.nanmedian(a,axis=0)
                np.testing.assert_allclose(low,state['context']['lower'],atol=1e-12)
                np.testing.assert_allclose(high,state['context']['upper'],atol=1e-12)
                np.testing.assert_allclose(median,state['context']['median'],atol=1e-12)
                clean=np.where(np.isnan(a),median,np.clip(a,low,high))
                np.testing.assert_allclose(clean.mean(axis=0),state['context']['mean'],atol=1e-12)
            x=f[features] if design.startswith('ridge') else ranked
            X=transform_features(x,state['base'])
            if 'context' in state:X=pd.concat([X,transform_features(f[state['context']['features']],state['context'])],axis=1)
            path=OUT/'models'/f'w{wi}_{name}'
            model=joblib.load(path.with_suffix('.joblib')) if design.startswith('ridge') else load_lightgbm(path.with_suffix('.txt'))
            np.testing.assert_allclose(model.predict(X.loc[ec]),pred[name],atol=1e-12)
            observed={date:spearmanr(pred.loc[g.index,name],g.y_excess_5d).statistic for date,g in f.loc[mask].groupby('signal_date')}
            stored=daily[daily.window.eq(wi)&daily.design.eq(design)&daily.candidate.eq(row['candidate'])].set_index('signal_date').rank_ic
            np.testing.assert_allclose(pd.Series(observed).reindex(stored.index),stored,atol=1e-12)


def test_fit_budget_and_target_constraints():
    ledger=json.loads((OUT/'fit_ledger.json').read_text())
    assert len(ledger)==32 and len({(x['window'],x['design'],x['candidate']) for x in ledger})==32
    for row in ledger:
        assert pd.Timestamp(row['max_train_label'])<pd.Timestamp(row['eval_start'])
    x=pd.read_csv(ROOT/'results/context_selection_v2/exposures.csv')
    x=x[x.stage.eq('target')]
    assert len(x)==400
    assert x.max_weight.le(.02+1e-8).all()
    assert x.max_industry_active.le(.02+1e-8).all()
    assert (x.beta-x.benchmark_beta).abs().le(.1+1e-8).all()
    assert (x.annual_stock_vol-x.benchmark_vol).le(.1+1e-8).all()
