"""Directional features for one security aligned to the complete market calendar."""
import numpy as np
import pandas as pd

from equity_research.data.universe import SINGLE_DAY


def ordinary_returns(history, calendar):
    if (not calendar.is_unique or not calendar.is_monotonic_increasing
            or not history.index.equals(calendar)):
        raise ValueError("Align the security to the complete ordered market calendar first")
    r = history.DlyRet
    valid = (history.DlyDelFlg.eq("N") & history.DlyRetMissFlg.eq("NA")
             & history.DlyRetDurFlg.isin(SINGLE_DAY) & r.ge(-1) & np.isfinite(r))
    return r.where(valid)


def directional_features(ordinary_returns, windows):
    r = ordinary_returns
    up = r.gt(0).astype(float).where(r.notna())
    columns = {}
    for w in windows:
        windows_view = np.lib.stride_tricks.sliding_window_view(r.to_numpy(dtype=float), w) if len(r) >= w else []
        share = np.full(len(r), np.nan)
        if len(windows_view):
            gains = windows_view.clip(min=0).sum(axis=1)
            losses = (-windows_view).clip(min=0).sum(axis=1)
            total = gains + losses
            np.divide(gains, total, out=share[w-1:], where=total > 0)
        columns[f"positive_fraction_{w}"] = up.rolling(w).mean()
        columns[f"gain_share_{w}"] = share
    return pd.DataFrame(columns, index=r.index)
