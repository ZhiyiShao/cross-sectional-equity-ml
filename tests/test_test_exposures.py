"""Independent edge cases for post-test attribution, without historical data."""
import numpy as np
import pandas as pd
import pytest

from test_final_test_engine import event_market
from equity_research.backtest.engine import simulate
from equity_research.evaluation.exposures import security_ledger
from equity_research.evaluation.test_exposures import period_contributions


@pytest.mark.parametrize('event', ['otc', 'cvr'])
def test_delisting_attribution_has_no_phantom_transfer_profit(event):
    m = event_market()
    term = dict(event_date=str(m['dates'][2].date()), previous_price=10.,
                cash_per_share=8., special_per_share=0., special_expected_date=None,
                ratio=0., successor_permno=0, cvr_max_per_share=3.)
    if event == 'otc':
        term.update(election_policy='first_post_delisting_quote_forced_sale_proxy',
                    otc_first_observation_date=str(m['dates'][5].date()), otc_first_observation_price=.4)
    d, t, h, e = simulate(m, {m['dates'][1]: (1,), m['dates'][5]: (2,)}, {1: term}, 10, 2, False, True)
    for frame in [d, t, h, e]:
        frame['model'] = 'equal_weight'
    d['prev_nav'] = d.nav.shift().fillna(1.)
    p = security_ledger(d, h, t, m['dates'], e, pd.DataFrame.from_dict({1: term}, orient='index'))
    qty = (1 / 1.001) / 10
    expected = qty * (.4 - 10) if event == 'otc' else qty * (8 - 10)
    date = m['dates'][5 if event == 'otc' else 2]
    np.testing.assert_allclose(p.loc[p.date.eq(date) & p.PERMNO.eq(1), 'pnl'].sum(), expected)
    if event == 'otc':
        np.testing.assert_allclose(p.loc[p.date.between(m['dates'][2], m['dates'][4]), 'pnl'].sum(), 0.)
    np.testing.assert_allclose(p.net_pnl.sum(), d.nav.iloc[-1] - 1)


def test_monthly_contributions_use_distinct_annual_and_monthly_bases():
    dates = pd.to_datetime(['2022-01-28', '2022-01-31', '2022-02-01'])
    d = pd.DataFrame(dict(model='m', date=dates, prev_nav=[1., 1., .8], nav=[1., .8, 1.], pnl=[0., -.2, .2], fees=0.))
    p = pd.DataFrame(dict(model='m', date=dates[1:], PERMNO=1, industry='36',
                          net_pnl=[-.2, .2], cost=0., equity_weight=1.))
    monthly, industry, stock = period_contributions(p, d)
    np.testing.assert_allclose(monthly.monthly_return, [-.2, .25])
    np.testing.assert_allclose(monthly.annual_contribution, [-.2, .2])
    for table in [industry, stock]:
        np.testing.assert_allclose(table.annual_contribution.sum(), 0., atol=1e-12)
