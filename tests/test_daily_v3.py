import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_daily", ROOT / "scripts/build_daily_v3.py")
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)
from equity_research.data.directional_v3 import directional_features


def test_daily_features_calendar_and_label_boundary():
    cfg = json.loads((ROOT / "config/features_labels.json").read_text())
    uc = json.loads((ROOT / "config/universe.json").read_text())
    calendar = pd.bdate_range(end="2021-12-31", periods=160)
    cfg["signal_start"] = str(calendar[120].date())
    market = pd.Series(np.linspace(-0.001, 0.002, len(calendar)), index=calendar)
    history = pd.DataFrame(dict(DlyCalDt=calendar, PERMNO=1, DlyRet=0.01, DlyRetDurFlg="D1",
                                DlyRetMissFlg="NA", DlyPrc=10., DlyClose=10., DlyHigh=11., DlyLow=9.,
                                DlyVol=1000., ShrOut=1000., DlyCap=10000., DlyCumFacShr=1.))
    for key, values in {**uc["regular_trading_filters"], **uc["security_filters"]}.items():
        history[key] = values[0]
    membership = pd.DataFrame(dict(effective_month=calendar.strftime("%Y-%m").unique(),
                                   formation_date=calendar[0]-pd.Timedelta(days=1)))
    base = daily.security_panel(history, membership, market, cfg, uc)
    assert base.signal_date.tolist() == list(calendar[120:])
    assert base.loc[base.is_v1_signal, "signal_date"].tolist() == list(calendar[120::5])
    assert base.iloc[-6:].label_end_date.isna().all()
    expected = 1.01**5 - np.prod(1 + market.iloc[122:127])
    np.testing.assert_allclose(base.iloc[0].y_excess_5d, expected)
    history.loc[130:, "DlyRet"] = -0.1
    changed = daily.security_panel(history, membership, market, cfg, uc)
    pd.testing.assert_frame_equal(base.iloc[:10][cfg["features"]], changed.iloc[:10][cfg["features"]])


def test_each_draft_feature_against_explicit_windows_and_future_perturbation():
    plan = json.loads((ROOT / "config/features_v3.json").read_text())
    r = pd.Series(np.tile([0.01, -0.03, 0., 0.02, 0.04], 30))
    r.iloc[4] = np.nan
    observed = directional_features(r, plan["windows"])
    expected = set()
    for w in plan["windows"]:
        sample = r.iloc[81-w:81]
        values = [(sample > 0).mean(), sample.clip(lower=0).sum()/sample.abs().sum()]
        for family, value in zip(plan["families"], values, strict=True):
            column = f"{family['name']}_{w}"
            expected.add(column)
            np.testing.assert_allclose(observed.at[80, column], value)
            assert pd.isna(observed.at[w-1, column])
    assert set(observed.columns) == expected and len(expected) == plan["feature_count"]
    r.iloc[81:] = -1.
    pd.testing.assert_frame_equal(observed.iloc[:81], directional_features(r, plan["windows"]).iloc[:81])
    flat = directional_features(pd.Series(0., index=r.index), plan["windows"])
    assert flat.filter(like="gain_share_").isna().all().all()

def test_gain_share_endpoints_after_long_history():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        history = rng.normal(size=1500)
        for tail, expected in [(np.ones(60), 1.), (-np.ones(60), 0.), (np.zeros(60), np.nan)]:
            f = directional_features(pd.Series(np.r_[history,tail]), [5,20,60])
            for w in [5,20,60]:
                gain = f[f"gain_share_{w}"]
                assert gain.dropna().between(0,1).all()
                assert gain.first_valid_index() == w-1
                np.testing.assert_equal(gain.iloc[-1],expected)


def test_raw_mask_and_market_calendar():
    import pytest
    from equity_research.data.directional_v3 import ordinary_returns
    dates = pd.bdate_range("2020-01-02",periods=7)
    h = pd.DataFrame(dict(DlyRet=[0.,-1.,.1,.2,.3,-1.1,np.inf],DlyRetDurFlg=["D1","D2","P2","D1","D1","D1","D1"],
                          DlyDelFlg=["N","N","N","Y","N","N","N"],DlyRetMissFlg=["NA"]*4+["MP","NA","NA"]),index=dates)
    np.testing.assert_array_equal(ordinary_returns(h,dates),[0.,-1.,np.nan,np.nan,np.nan,np.nan,np.nan])
    with pytest.raises(ValueError):
        ordinary_returns(h.iloc[1:],dates)


def test_signed_screen_and_paired_adoption():
    import sys
    import pytest
    sys.path.insert(0,str(ROOT/"scripts"))
    from screen_directional_v3 import adopted, paired_rho
    a = np.arange(100)
    f = pd.DataFrame(dict(signal_date=np.repeat([1,2],100),a=np.tile(a,2),b=np.r_[a,-a]))
    rho,days = paired_rho(f,"a","b")
    assert rho == 0 and days == 2
    for bad in [f.assign(b=1), f.iloc[1:]]:
        with pytest.raises(ValueError):
            paired_rho(bad,"a","b")
    for diff,expected in [([1]*4,True),([1,1,1,0],True),([2,2,0,0],False),([1,1,1,-3],False)]:
        assert adopted(diff) == expected
