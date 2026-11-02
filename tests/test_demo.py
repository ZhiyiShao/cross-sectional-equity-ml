"""End-to-end public demo: cross-stage contracts, not synthetic profitability."""
import json

import numpy as np
import pandas as pd
import pytest

from equity_research.demo import run_demo, fit_models, MODEL_NAMES
from equity_research.models.preprocessing import fit_state, cross_sectional_rank


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    out = run_demo(tmp_path_factory.mktemp('synthetic_demo'))
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    panel = pd.read_parquet(out / 'features_labels.parquet')
    return out, manifest, panel


def test_demo_runs_all_stages_and_reconciles_cash_event(demo):
    out, manifest, panel = demo
    assert manifest['status'] == 'completed' and manifest['synthetic_data_only']
    assert manifest['selection_fits'] == 8 and manifest['final_fits'] == 4
    assert manifest['accounting_verified'] and len(manifest['features']) == 21
    summary = pd.read_csv(out / 'summary.csv')
    assert set(summary.model) == set(MODEL_NAMES)
    assert np.isfinite(summary[['cagr', 'sharpe', 'max_drawdown']]).all().all()
    assert summary.total_fees.gt(0).all()  # Fees must actually be charged.
    assert (out / 'demo_overview.png').stat().st_size > 1000
    daily = pd.read_csv(out / 'daily.csv')
    np.testing.assert_allclose(daily.nav, daily.cash + daily.equity + daily.receivable - daily.payable)
    assert daily.sale_receivable.gt(0).any() and daily.payable.gt(0).any()
    events = pd.read_csv(out / 'events.csv')
    assert ((events.model == 'equal_weight') & (events.kind == 'conversion')).sum() == 1
    assert daily.loc[daily.model.eq('equal_weight'), 'released_cash'].sum() > 0
    assert panel.label_status.eq('delist_settlement_unverified').any()


def test_demo_boundaries_and_saved_preprocessing(demo):
    out, manifest, panel = demo
    cfg, features = manifest['configuration'], manifest['features']
    ledger = pd.DataFrame(json.loads((out / 'fit_ledger.json').read_text(encoding='utf-8')))
    assert ledger.loc[ledger.phase.eq('selection'), 'max_label_end'].le('2020-12-31').all()
    assert ledger.loc[ledger.phase.eq('final'), 'max_label_end'].le('2021-12-31').all()
    assert ledger.max_signal.lt('2022-01-01').all()
    universe = pd.read_csv(out / 'universe.csv')
    assert not universe.duplicated(['effective_month', 'PERMCO']).any()
    assert not universe.PERMNO.isin([900000, 900001, 900002, 900003]).any()
    assert universe.loc[universe.PERMNO.eq(900006), 'history_days'].ge(252).all()
    assert panel.formation_date.lt(panel.signal_date).all()
    assert panel.loc[panel.purged_boundary, 'eligible_main_sample'].eq(False).all()
    development = panel.loc[panel.split.isin(['train', 'validation'])]
    train = development.split.eq('train') & development.eligible_main_sample
    final = (development.signal_eligible & development.label_status.isin(['ok', 'ok_bridged'])
             & development.label_end_date.le('2021-12-31'))
    assert (final & development.purged_boundary).any()  # Internal boundary restored on refit.
    for rep in cfg['representations']:
        x = development[features] if rep == 'raw' else cross_sectional_rank(development, features)
        for phase, mask in [('selection', train), ('final', final)]:
            saved = json.loads((out / 'preprocessing' / f'{rep}_{phase}.json').read_text(encoding='utf-8'))
            assert saved == fit_state(x.loc[mask], rep, cfg)
    selected = json.loads((out / 'selected.json').read_text(encoding='utf-8'))
    scores = pd.read_csv(out / 'validation_scores.csv')
    for candidate in selected:
        group = scores.loc[scores.model.eq(candidate['model'])]
        assert candidate['trial'] == group.loc[group.mean_rank_ic.idxmax(), 'trial']


def test_demo_rejects_holdout_at_fit_boundary(demo, tmp_path):
    _, manifest, panel = demo
    with pytest.raises(ValueError, match='Holdout passed to fit'):
        fit_models(panel, manifest['features'], manifest['configuration'], tmp_path)
    assert not list(tmp_path.iterdir())


def test_demo_refuses_to_overwrite_an_existing_run(demo):
    out, _, _ = demo
    original = (out / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='empty directory'):
        run_demo(out)
    assert (out / 'manifest.json').read_bytes() == original
