"""Causal cross-sectional ranks, training-fitted transforms and ranking metrics."""
import numpy as np
import pandas as pd
from equity_research.models.common import fit_preprocessing, daily_rank_ic


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame_digest(frame):
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def cross_sectional_rank(frame, features):
    # Only current-date eligibility and finite feature values enter the universe.
    values = frame[features].replace([np.inf, -np.inf], np.nan).where(frame.signal_eligible, axis=0)
    groups = values.groupby(frame.signal_date)
    return (groups.rank(method='average')-.5)/groups.transform('count')-.5


def fit_state(x, representation, cfg):
    if representation == 'raw':
        return fit_preprocessing(x, cfg['raw_clip_quantiles'])
    clean = x.replace([np.inf, -np.inf], np.nan)
    require(not clean.isna().all().any(), 'All-missing ranked training column')
    median = clean.median()
    clean = clean.fillna(median)
    return dict(features=x.columns.tolist(), fitted_rows=len(x),
                lower=[cfg['rank_clip_bounds'][0]]*len(x.columns),
                upper=[cfg['rank_clip_bounds'][1]]*len(x.columns), median=median.tolist(),
                mean=clean.mean().tolist(), scale=clean.std(ddof=0).replace(0., 1.).tolist())


def checked_ic(dates, predictions, targets):
    ic = daily_rank_ic(dates, predictions, targets)
    require(len(ic) == dates.nunique() and np.isfinite(ic).all(), 'Undefined or dropped date IC')
    return ic


def metric(dates, predictions, targets):
    ic = checked_ic(dates, predictions, targets)
    return dict(rows=len(targets), dates=len(ic), mean_rank_ic=float(ic.mean()),
                positive_fraction=float(ic.gt(0).mean()), ic_std=float(ic.std()),
                rmse=float(np.sqrt(np.mean(np.square(np.asarray(predictions)-np.asarray(targets))))))
