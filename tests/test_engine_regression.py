"""Numerical comparison with the pre-refactor engine, including event accounting."""
import pandas as pd
import pytest
from test_action_portfolios import market
from reference.preflight import simulate as original
from equity_research.backtest.validation import simulate, verify


@pytest.mark.parametrize("cost", [0, 10, 20])
@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("settled_only", [False, True])
@pytest.mark.parametrize("event", ["ordinary", "cash", "stock"])
def test_refactor_preserves_old_cash_positions_and_trades(cost, lag, settled_only, event):
    m = market()
    terms = {}
    if event != "ordinary":
        m["ordinary"][2, 0] = False
        m["terminal"][2, 0] = True
        terms = {1: dict(event_date=str(m["dates"][2].date()),
                        cash_per_share=10 if event == "cash" else 2,
                        special_per_share=0, special_expected_date=None,
                        ratio=0 if event == "cash" else .5,
                        successor_permno=0 if event == "cash" else 2)}
    targets = {m["dates"][1]: (1,), m["dates"][3]: (2,), m["dates"][6]: (2,)}
    expected = original(m, targets, terms, cost, lag, settled_only, True)
    actual = simulate(m, targets, terms, cost, lag, settled_only, True)
    for left, right in zip(actual, expected):
        pd.testing.assert_frame_equal(left[right.columns], right)
    verify(*actual[:3])
