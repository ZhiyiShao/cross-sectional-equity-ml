"""Observation access for the pre-test engineering release. No test unlock switch."""
from pathlib import Path
import pandas as pd

from equity_research.paths import ROOT
START, END = pd.Timestamp('2020-01-01'), pd.Timestamp('2021-12-31')

class DateGate:
    def __init__(self):
        self.audit = []

    def read_parquet(self, path, *, columns, filters):
        # Validate BEFORE opening any dataset; nested OR filters are forbidden.
        path = Path(path).resolve()
        allowed = [ROOT/'data/interim/crsp_audit_v1',
                   ROOT/'outputs/full_train_selection_v1',
                   ROOT/'outputs/final_funding_backtest_v1']
        if not any(path.is_relative_to(p) for p in allowed):
            raise PermissionError('Unregistered observation source')
        dates = [c for c in ['DlyCalDt', 'signal_date', 'date'] if c in columns]
        if len(dates) != 1 or not filters or any(not isinstance(f, tuple) or len(f)!=3 for f in filters):
            raise PermissionError('One date column and flat AND date filters required')
        col = dates[0]
        lower, upper = [], []
        for key, op, value in filters:
            if key == col:
                if op not in ('>=','>','<=','<'):
                    raise PermissionError('Unsupported date predicate')
                stamp = pd.Timestamp(value)
                if pd.isna(stamp) or stamp.tz is not None:
                    raise PermissionError('Invalid observation boundary')
                if op in ('>=','>'): lower.append(stamp)
                else: upper.append(stamp - pd.Timedelta(nanoseconds=1) if op=='<' else stamp)
        if not lower or not upper or min(lower)<START or max(upper)>END+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1):
            raise PermissionError('Observation request outside validation; test remains sealed')
        if max(lower)>min(upper):
            raise PermissionError('Inverted date request')
        enforced = list(filters)+[(col,'>=',START),(col,'<=',END)]
        # The audited CRSP calendar stores ISO dates as strings; price/prediction
        # datasets store timestamps. Preserve native types for Arrow pushdown.
        if path == ROOT/'data/interim/crsp_audit_v1/calendar.parquet':
            enforced=[(k,op,str(pd.Timestamp(v).date()) if k==col else v) for k,op,v in enforced]
        frame = pd.read_parquet(path, columns=columns, filters=enforced)
        self.check_frame(frame, col)
        self.audit.append(dict(path=str(path.relative_to(ROOT)),date_column=col,
            filters=repr(enforced),rows=len(frame),
            minimum=None if frame.empty else str(pd.to_datetime(frame[col]).min()),
            maximum=None if frame.empty else str(pd.to_datetime(frame[col]).max())))
        return frame

    @staticmethod
    def check_frame(frame, col):
        dates = pd.to_datetime(frame[col])
        if dates.isna().any() or not dates.between(START, END).all():
            raise PermissionError('Returned observation date breach')
        if 'split' in frame and not frame['split'].eq('validation').all():
            raise PermissionError('Split/date disagreement')

def deny_test_execution():
    # This release cannot fit, predict, audit test events or publish test results.
    raise PermissionError('Test execution sealed: final fits and separately authorized test event audit are pending')

def signal_schedule(calendar, start, end, anchor='2015-01-02'):
    """Keep the global five-session phase; input must include the anchor."""
    cal = pd.DatetimeIndex(calendar)
    if cal.has_duplicates or not cal.is_monotonic_increasing or pd.Timestamp(anchor) not in cal:
        raise ValueError('A unique ordered calendar containing the original anchor is required')
    first = cal.get_loc(pd.Timestamp(anchor))
    rows=[]
    for i in range(first, len(cal), 5):
        if pd.Timestamp(start)<=cal[i]<=pd.Timestamp(end) and i+6<len(cal) and cal[i+6]<=pd.Timestamp(end):
            rows.append(dict(signal_date=cal[i],entry_date=cal[i+1],end_date=cal[i+6]))
    return pd.DataFrame(rows, columns=['signal_date','entry_date','end_date'])
