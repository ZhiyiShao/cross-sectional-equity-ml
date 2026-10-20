"""Synthetic temporal CV isolation and representation checks; no real model fits."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from reference.cv import CONFIG, load_training_only, fold_masks, configurations, new_model
from equity_research.models.preprocessing import cross_sectional_rank, fit_state, checked_ic
from equity_research.models.common import transform_features


def config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def test_training_loader_excludes_both_holdouts(tmp_path):
    a = pd.DataFrame({'PERMNO': [1, 2, 1, 1],
        'signal_date': pd.to_datetime(['2016-01-01', '2016-01-01', '2020-01-01', '2022-01-01']),
        'split': ['train', 'train', 'validation', 'test'],
        'label_end_date': pd.to_datetime(['2016-01-10']*2+['2020-01-10', '2022-01-10']),
        'signal_eligible': [True]*4, 'label_status': ['ok', 'missing', 'ok', 'ok'],
        'y_excess_5d': [.1, np.nan, 1e90, 1e99], 'f': [1., 2., 1e90, 1e99]})
    path = tmp_path/'data.parquet'; a.to_parquet(path, index=False)
    loaded = load_training_only(path, ['f'])
    assert len(loaded) == 2 and loaded.f.max() == 2
    assert loaded.y_excess_5d.isna().sum() == 1


def test_fold_repurges_internal_boundaries_and_keeps_unlabeled_candidates():
    a = pd.DataFrame({'signal_date': pd.to_datetime(['2016-12-20', '2016-12-29', '2017-01-04', '2017-12-29', '2017-02-01']),
        'label_end_date': pd.to_datetime(['2016-12-28', '2017-01-06', '2017-01-12', '2018-01-08', None]),
        'signal_eligible': [True]*5, 'label_status': ['ok']*4+['missing'],
        'y_excess_5d': [.01]*4+[np.nan]})
    masks = fold_masks(a, config()['folds'][0])
    assert masks['train'].tolist() == [True, False, False, False, False]
    assert masks['validation'].tolist() == [False, False, True, False, False]
    assert masks['validation_candidates'].sum() == 3


def test_ranking_uses_eligible_unlabeled_rows_ties_and_no_future_dates():
    a = pd.DataFrame({'signal_date': pd.to_datetime(['2016-01-01']*5+['2017-01-01']),
        'signal_eligible': [True, True, True, False, True, True],
        'f': [10., 10., 30., -100., np.nan, 1e99],
        'y_excess_5d': [0., np.nan, 0., 0., 0., 0.]})
    r = cross_sectional_rank(a, ['f']).f
    np.testing.assert_allclose(r.iloc[:3], [-1/6, -1/6, 1/3])
    assert r.iloc[3:5].isna().all() and r.iloc[5] == 0
    changed = a.copy(); changed.y_excess_5d = 100.; changed.loc[5, 'f'] = -1e99
    pd.testing.assert_series_equal(r, cross_sectional_rank(changed, ['f']).f)


def test_rank_all_missing_stays_missing_and_does_not_leak_into_fit_state():
    cfg = config()
    training = pd.DataFrame({'f': [-.4, .1, np.nan]})
    state = fit_state(training, 'cross_sectional_rank', cfg)
    frozen = json.dumps(state)
    transformed = transform_features(pd.DataFrame({'f': [.49, np.nan]}), state)
    assert np.isfinite(transformed).all().all() and json.dumps(state) == frozen
    assert state['median'][0] == pytest.approx(-.15)
    with pytest.raises(ValueError, match='All-missing'):
        fit_state(pd.DataFrame({'f': [np.nan]}), 'cross_sectional_rank', cfg)


def test_fixed_budget_and_fold_normalized_ridge_penalty():
    cfg = config(); grid = configurations(cfg)
    assert len(grid) == 22 and len(grid)*len(cfg['folds']) == 66
    ridge = grid[0]
    assert new_model(cfg, ridge, 100).alpha == pytest.approx(100*ridge['parameters']['lambda'])
    assert new_model(cfg, ridge, 200).alpha == pytest.approx(2*new_model(cfg, ridge, 100).alpha)
    raw = [g['parameters'] for g in grid if g['representation'] == 'raw']
    ranked = [g['parameters'] for g in grid if g['representation'] == 'cross_sectional_rank']
    assert raw == ranked


def test_undefined_daily_ic_aborts_instead_of_dropping_a_date():
    dates = pd.Series(pd.to_datetime(['2017-01-01']*3))
    with pytest.raises(ValueError, match='Undefined'):
        checked_ic(dates, [1., 1., 1.], [.1, .2, .3])
