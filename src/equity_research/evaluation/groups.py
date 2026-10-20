"""Read-only group assignment, coverage and paired uncertainty diagnostics."""
import numpy as np
import pandas as pd
from equity_research.models.preprocessing import require


def load_validation(path, columns):
    frame = pd.read_parquet(path, columns=columns, engine='pyarrow', filters=[
        ('signal_date', '>=', pd.Timestamp('2020-01-01')),
        ('signal_date', '<', pd.Timestamp('2022-01-01')), ('split', '==', 'validation')])
    require(len(frame) > 0 and frame.signal_date.between('2020-01-01', '2021-12-31').all(), 'Validation scope breach')
    require(not frame.duplicated(['signal_date', 'PERMNO']).any(), 'Duplicate predictions')
    return frame.sort_values(['signal_date', 'PERMNO']).reset_index(drop=True)


def assign_groups(frame, score, groups=10):
    """Never use a label, label status or future tradability to assign a group."""
    chosen = frame.loc[frame.signal_eligible].copy()
    require(np.isfinite(chosen[score]).all(), 'Nonfinite eligible prediction')
    chosen = chosen.sort_values(['signal_date', score, 'PERMNO'], ascending=[True, False, True])
    position = chosen.groupby('signal_date').cumcount()
    size = chosen.groupby('signal_date')[score].transform('size')
    chosen['group'] = groups-(groups*position//size)
    return chosen


def boundary_dates(frame):
    """Date-wide administrative horizon boundary, never inferred from label absence."""
    flags = frame.label_end_date.gt('2021-12-31')
    dates = pd.DataFrame({'date': frame.signal_date, 'boundary': flags}).groupby('date').boundary
    require((dates.nunique() == 1).all(), 'Mixed horizon boundaries within date')
    require(flags.equals(frame.purged_boundary.astype(bool)), 'Stored boundary differs from horizon')
    return dates.first()


def circular_interval(values, block, repetitions, seed):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all() and len(values) >= block, 'Invalid bootstrap input')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(repetitions, int(np.ceil(len(values)/block))))
    indices = (starts[:, :, None]+np.arange(block)) % len(values)
    draws = values[indices.reshape(repetitions, -1)[:, :len(values)]].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return dict(dates=len(values), mean_difference=float(values.mean()), ci_low=float(low), ci_high=float(high))


def group_statistics(assigned, boundary):
    rows, reasons = [], []
    for date, day in assigned.groupby('signal_date'):
        is_boundary = bool(boundary.loc[date])
        market_pool_mean = day.y_excess_5d.mean()
        for group, part in day.groupby('group'):
            primary = part.y_excess_5d
            augmented = primary.fillna(part.y_excess_5d_event_proxy) if not is_boundary else primary.copy()
            n, observed, proxy_observed = len(part), int(primary.notna().sum()), int(augmented.notna().sum())
            rows.append(dict(signal_date=date, year=date.year, group=int(group), boundary=is_boundary,
                selected=n, observed=observed, missing=n-observed, coverage=observed/n,
                proxy_observed=proxy_observed, proxy_added=proxy_observed-observed,
                proxy_remaining_missing=n-proxy_observed, proxy_coverage=proxy_observed/n,
                mean_excess=primary.mean(), median_stock_excess=primary.median() if observed else np.nan,
                augmented_mean_excess=augmented.mean(), pool_mean_excess=market_pool_mean,
                mean_minus_pool=primary.mean()-market_pool_mean))
            missing = part.loc[primary.isna()]
            for status, count in missing.label_status.value_counts(dropna=False).items():
                reasons.append(dict(signal_date=date, year=date.year, group=int(group), boundary=is_boundary,
                                    reason='administrative_boundary' if is_boundary else str(status), count=int(count)))
    return pd.DataFrame(rows), pd.DataFrame(reasons)


def summarize_groups(daily):
    usable = daily.loc[~daily.boundary]
    rows = []
    periods = [('all', usable)]+[(str(year), part) for year, part in usable.groupby('year')]
    for period, part in periods:
        for (model, group), g in part.groupby(['model', 'group'], sort=False):
            returns = g.mean_excess.dropna()
            rows.append(dict(period=period, model=model, group=int(group), dates=len(g),
                observed_return_dates=len(returns), no_observed_return_dates=int(g.mean_excess.isna().sum()),
                selected=int(g.selected.sum()), observed=int(g.observed.sum()), missing=int(g.missing.sum()),
                coverage=float(g.observed.sum()/g.selected.sum()), min_date_coverage=float(g.coverage.min()),
                dates_with_missing=int(g.missing.gt(0).sum()), mean_excess=float(returns.mean()),
                median_date_excess=float(returns.median()), positive_date_fraction=float(returns.gt(0).mean()),
                mean_minus_pool=float(g.mean_minus_pool.mean()),
                mean_without_best5_dates=float(returns.sort_values().iloc[:-5].mean()) if len(returns)>5 else np.nan,
                augmented_mean_excess=float(g.augmented_mean_excess.mean()), proxy_added=int(g.proxy_added.sum()),
                proxy_remaining_missing=int(g.proxy_remaining_missing.sum()),
                proxy_coverage=float(g.proxy_observed.sum()/g.selected.sum())))
    return pd.DataFrame(rows)


def verify_group_assignment(frame, score, assigned, daily, boundary):
    """Independent top selection using numpy lexsort, plus partition conservation."""
    for date, day in frame.loc[frame.signal_eligible].groupby('signal_date'):
        actual = assigned.loc[assigned.signal_date.eq(date)]
        require(set(actual.PERMNO) == set(day.PERMNO), 'Group partition changed eligible universe')
        order = np.lexsort((day.PERMNO.to_numpy(), -day[score].to_numpy()))
        top = day.iloc[order[:int(np.ceil(len(day)/10))]]
        chosen = actual.loc[actual.group.eq(10)]
        require(set(top.PERMNO) == set(chosen.PERMNO), 'Top-decile membership mismatch')
        r = daily.loc[daily.signal_date.eq(date)]
        require(r.selected.sum() == len(day) and r.observed.sum() == day.y_excess_5d.notna().sum(), 'Group totals mismatch')
        q10 = r.loc[r.group.eq(10)].iloc[0]
        np.testing.assert_allclose(q10.mean_excess, top.y_excess_5d.mean(), equal_nan=True)
        require(q10.missing == top.y_excess_5d.isna().sum(), 'Top coverage mismatch')
        if not boundary.loc[date]:
            np.testing.assert_allclose(q10.augmented_mean_excess,
                top.y_excess_5d.fillna(top.y_excess_5d_event_proxy).mean(), equal_nan=True)
