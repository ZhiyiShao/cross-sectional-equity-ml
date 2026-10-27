"""Causal cross-sectional ranks, training-fitted transforms and ranking metrics."""
import numpy as np
import pandas as pd

def fit_preprocessing(x, quantiles):
    """Fit raw-feature clipping, imputation and scaling on training rows only."""
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.isna().all().any():
        raise ValueError('Training contains an entirely missing feature')
    lower = x.quantile(quantiles[0])
    upper = x.quantile(quantiles[1])
    median = x.median()
    clean = x.clip(lower=lower, upper=upper, axis=1).fillna(median)
    std = clean.std(ddof=0).replace(0.0, 1.0)
    return {
        'features': x.columns.tolist(), 'fitted_rows': len(x),
        'lower': lower.tolist(), 'upper': upper.tolist(),
        'median': median.tolist(), 'mean': clean.mean().tolist(), 'scale': std.tolist(),
    }

def transform_features(x, state):
    """Apply a saved state without learning from validation or test rows."""
    assert x.columns.tolist() == state['features']
    a = x.to_numpy(dtype=float, copy=True)
    a[~np.isfinite(a)] = np.nan
    a = np.clip(a, state['lower'], state['upper'])
    a = np.where(np.isnan(a), np.asarray(state['median']), a)
    a = (a - state['mean']) / state['scale']
    assert np.isfinite(a).all()
    return pd.DataFrame(a, index=x.index, columns=x.columns)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def frame_digest(frame):
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()

def cross_sectional_rank(frame, features):
    """Rank eligible stocks within each signal date, centered around zero."""
    # Eligibility and finite values are evaluated at the current signal date.
    values = (
        frame[features].replace([np.inf, -np.inf], np.nan)
        .where(frame.signal_eligible, axis=0)
    )
    groups = values.groupby(frame.signal_date)
    return (groups.rank(method='average') - 0.5) / groups.transform('count') - 0.5

def fit_state(x, representation, cfg):
    """Select the raw or ranked preprocessing recipe; x contains training rows."""
    if representation == 'raw':
        return fit_preprocessing(x, cfg['raw_clip_quantiles'])
    clean = x.replace([np.inf, -np.inf], np.nan)
    require(not clean.isna().all().any(), 'All-missing ranked training column')
    median = clean.median()
    clean = clean.fillna(median)
    return dict(
        features=x.columns.tolist(), fitted_rows=len(x),
        lower=[cfg['rank_clip_bounds'][0]] * len(x.columns),
        upper=[cfg['rank_clip_bounds'][1]] * len(x.columns),
        median=median.tolist(), mean=clean.mean().tolist(),
        scale=clean.std(ddof=0).replace(0.0, 1.0).tolist(),
    )

def checked_ic(dates, predictions, targets):
    # Lazy import keeps preprocessing usable while common re-exports old names.
    from equity_research.models.common import daily_rank_ic
    ic = daily_rank_ic(dates, predictions, targets)
    require(len(ic) == dates.nunique() and np.isfinite(ic).all(), 'Undefined or dropped date IC')
    return ic

def metric(dates, predictions, targets):
    ic = checked_ic(dates, predictions, targets)
    return dict(
        rows=len(targets), dates=len(ic), mean_rank_ic=float(ic.mean()),
        positive_fraction=float(ic.gt(0).mean()), ic_std=float(ic.std()),
        rmse=float(np.sqrt(np.mean(np.square(np.asarray(predictions) - np.asarray(targets))))),
    )
