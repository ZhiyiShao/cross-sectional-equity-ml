"""Search diagnostics use complete paired windows and the frozen weighting."""
import pandas as pd
import pytest

from equity_research.evaluation.search_audit_v2 import DESIGNS, summarize


def candidates():
    records = []
    for design in DESIGNS:
        grid = [0.1, 10., 1000., 100000.] if design.startswith('ridge') else [7., 15., 31., 63.]
        for window, values in [(1, [.04, .02, .01, 0]), (2, [0, .01, .01, .03])]:
            for candidate, (parameter, ic) in enumerate(zip(grid, values)):
                records.append(dict(design=design, candidate=candidate, parameter=parameter,
                                    window=window, rank_ic=ic + (.005 if design.endswith('enhanced') else 0),
                                    dates={1: 49, 2: 100}[window], rows={1: 24427, 2: 49860}[window]))
    return pd.DataFrame(records)


def test_equal_windows_not_date_weighted_and_matched_contrasts():
    curve, summary, contrasts = summarize(candidates())
    # Date weighting would prefer candidate 3; frozen equal weights prefer 0.
    assert summary.boundary.eq('lower').all()
    assert curve.loc[curve.candidate.eq(0) & curve.design.eq('ridge_original'), 'equal_window_ic'].item() == .02
    assert summary.best_2019.eq(summary.selected_parameter).all()
    assert (summary.best_2020_2021 > summary.selected_parameter).all()
    assert contrasts.loc[contrasts.contrast.str.endswith('features'), 'equal_window_delta'].to_list() == pytest.approx([.005]*12)
    assert contrasts.loc[contrasts.contrast.str.endswith('objective'), 'equal_window_delta'].to_list() == pytest.approx([0.]*8)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'grid', 'coverage', 'nan'])
def test_rejects_incomplete_or_changed_comparison(fault):
    frame = candidates()
    if fault == 'missing':
        frame = frame.iloc[:-1]
    elif fault == 'duplicate':
        frame.iloc[1] = frame.iloc[0]
    elif fault == 'grid':
        frame.loc[0, 'parameter'] = 2
    elif fault == 'coverage':
        frame.loc[0, 'dates'] = 48
    else:
        frame.loc[0, 'rank_ic'] = float('nan')
    with pytest.raises(ValueError):
        summarize(frame)


def test_exact_tie_uses_original_candidate_order():
    frame = candidates()
    frame['rank_ic'] = .01
    _, summary, _ = summarize(frame)
    assert summary.boundary.eq('lower').all()
    assert summary.winner_margin.eq(0).all()
