"""Read only completed, local 2022–2025 replay artifacts; no fits or raw-data IO."""
import numpy as np
import pandas as pd
import pytest
from equity_research.backtest.replay_v2 import ROOT,OUT,PUBLIC,START,END,check_targets
from equity_research.models.final_fit_v2 import read_json,check_metadata,IDS
from equity_research.models.common import sha
from equity_research.evaluation.v2 import locked_bundle

pytestmark=pytest.mark.skipif(not (OUT/'replay_audit.json').exists(),reason='Local historical V2 replay artifacts absent')


def test_replay_budget_models_sources_and_preserved_history():
    audit=read_json(PUBLIC/'verification.json')
    assert audit['paths']==56 and audit['model_fits']==0 and not audit['unseen_data_read']
    assert audit['ledger_days']==audit['market_dates']*14
    _,_,manifest=locked_bundle()
    assert manifest==read_json(OUT/'model_manifest.json')
    for path,digest in read_json(OUT/'source_hashes.json').items():assert sha(ROOT/path)==digest
    check_metadata(read_json(OUT/'protected_metadata.json'))
    log=read_json(OUT/'observation_access_log.json')
    assert all(pd.Timestamp(r['maximum'])<=END for r in log)
    for r in log:
        if r['minimum']<'2022':assert r['columns']==['date']


def test_all_paths_common_grid_and_benchmarks_reproduce_v1():
    daily=pd.read_parquet(OUT/'daily_portfolios.parquet',filters=[('date','>=',START),('date','<=',END)])
    assert daily.groupby(['model','variant','cost_bps']).ngroups==56
    reference=pd.read_csv(ROOT/'outputs/frozen_test_v1/daily_portfolios.csv',parse_dates=['date'])
    expected_dates=None
    for (model,variant,cost),g in daily.groupby(['model','variant','cost_bps']):
        g=g.sort_values('date')
        if expected_dates is None:expected_dates=g.date.to_numpy()
        np.testing.assert_array_equal(g.date,expected_dates)
        assert g.nav.iloc[0]==1 and g.equity.iloc[0]==0
        np.testing.assert_allclose(g.nav,g.cash+g.equity+g.receivable-g.payable,atol=1e-10)
        if model in ['equal_weight','momentum20']:
            old=reference[reference.model.eq(model)&reference.scenario.eq('lag_2')&reference.cost_bps.eq(cost)].sort_values('date')
            np.testing.assert_allclose(g.nav,old.nav,atol=1e-10,rtol=1e-10)


def test_primary_interval_independent_bootstrap():
    primary=read_json(PUBLIC/'primary_comparison.json')
    d=pd.read_parquet(OUT/'primary_daily_local.parquet')
    wide=d.pivot(index='date',columns='model',values='return_net').sort_index()
    delta=(wide.ridge_enhanced-wide.ridge_original).to_numpy()*252
    rng=np.random.default_rng(20261107);draws=[];n=len(delta)
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/20)))
        ix=np.concatenate([(start+np.arange(20))%n for start in starts])[:n]
        draws.append(delta[ix].mean())
    np.testing.assert_allclose(primary['annual_mean_net_difference'],delta.mean(),atol=1e-12)
    np.testing.assert_allclose([primary['ci_low'],primary['ci_high']],np.quantile(draws,[.025,.975]),atol=1e-12)


def test_targets_label_coverage_and_top_deciles():
    plan,_,_=locked_bundle()
    check_targets(pd.read_csv(OUT/'target_exposures.csv'),plan['portfolio']['constraints'])
    coverage=pd.read_csv(PUBLIC/'label_coverage.csv')
    assert len(coverage)==199 and coverage.candidates.eq(500).all()
    assert coverage.valid_labels.le(coverage.eligible).all()
    top=pd.read_csv(PUBLIC/'top_decile_labels.csv')
    assert set(top.model)==set(IDS+['momentum20'])
    assert top.valid_labels.le(top.selected).all()
    assert top.groupby('model').size().eq(199).all()
