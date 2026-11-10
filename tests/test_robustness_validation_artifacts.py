"""Read-only checks of saved 2020-2021 portfolio results; no models are loaded."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.models.robustness_ledger import sha

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'results/robustness_validation_backtest_v2'
LOCAL=ROOT/'outputs/robustness_validation_backtest_v2'
pytestmark=pytest.mark.skipif(not (PUBLIC/'verification.json').exists(),reason='Local validation run not available')


def test_validation_results_and_portfolio_statistics():
    d=pd.read_csv(PUBLIC/'daily_portfolios.csv',parse_dates=['date'])
    m=pd.read_csv(PUBLIC/'portfolio_metrics.csv').set_index(['design','variant'])
    assert d.date.between('2020-01-01','2021-12-31').all()
    assert len(m)==14 and len(d)==14*502
    for key,g in d.groupby(['design','variant']):
        g=g.sort_values('date');r=g.return_net.iloc[1:].to_numpy()
        np.testing.assert_allclose(g.nav.iloc[1:].to_numpy()/g.nav.iloc[:-1].to_numpy()-1,r,atol=1e-12)
        np.testing.assert_allclose(m.loc[key,'cagr'],np.prod(1+r)**(252/len(r))-1)
        np.testing.assert_allclose(m.loc[key,'annual_fee_drag'],(g.fees.iloc[1:]/g.pretrade_nav.iloc[1:]).mean()*252)


def test_only_one_design_changes_and_old_paths_reproduce():
    c=pd.read_csv(PUBLIC/'comparison.csv')
    unchanged=c[c.design.ne('lightgbm_original')]
    assert len(unchanged)==10
    np.testing.assert_allclose(unchanged.delta_cagr,0,atol=1e-12)
    reproduction=pd.read_csv(PUBLIC/'reproduced_paths.csv')
    assert len(reproduction)==12 and reproduction.max_nav_error.max()<1e-10
    a=json.loads((PUBLIC/'verification.json').read_text())
    assert a['model_fits']==a['final_refits']==a['new_prediction_calls']==a['holdout_performance_reads']==0
    assert a['target_checks']==700 and a['independently_verified_ledger_days']==7028


def test_saved_sources_ledger_and_predictions_unchanged():
    for name,digest in json.loads((LOCAL/'source_hashes.json').read_text()).items():
        assert sha(ROOT/name)==digest,name
    for name,digest in json.loads((PUBLIC/'manifest.json').read_text()).items():
        assert sha(PUBLIC/name)==digest,name
