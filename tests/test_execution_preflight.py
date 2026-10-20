import ast
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from equity_research.data.date_gate import DateGate, ROOT, deny_test_execution, signal_schedule
from equity_research.backtest.calendar import settlement_date, is_settlement_day, is_exchange_day, Ledger
from reference.preflight import simulate, verify
from equity_research.data.features import make_features, split_masks
from test_action_portfolios import market

@pytest.mark.parametrize('trade,due',[
 ('2022-01-03','2022-01-05'),('2022-04-13','2022-04-18'),
 ('2022-06-17','2022-06-22'),('2022-10-07','2022-10-12'),
 ('2022-12-30','2023-01-04'),('2023-04-05','2023-04-10'),
 ('2023-11-08','2023-11-10'),('2023-11-09','2023-11-13'),
 ('2023-12-29','2024-01-03'),('2024-03-27','2024-04-01'),
 ('2024-05-23','2024-05-28'),('2024-05-24','2024-05-29'),
 ('2024-05-28','2024-05-29'),('2024-06-18','2024-06-20'),
 ('2024-10-11','2024-10-15'),('2024-12-31','2025-01-02'),
 ('2025-01-08','2025-01-09'),('2025-04-17','2025-04-21'),
 ('2025-11-10','2025-11-12'),('2025-12-31','2026-01-02')])
def test_public_settlement_fixtures(trade,due):
    assert settlement_date(trade)==pd.Timestamp(due)

def test_calendar_fail_closed_and_special_open():
    with pytest.raises(ValueError):settlement_date('2026-01-09')
    with pytest.raises(ValueError):settlement_date('2019-12-30')
    assert is_settlement_day('2023-11-10') and is_settlement_day('2025-01-09')
    assert not is_exchange_day('2025-01-09') and is_exchange_day('2025-11-11')
    assert not is_settlement_day('2025-11-11')

@pytest.mark.parametrize('filters',[
 [],[('date','>=','2022-01-01'),('date','<=','2025-12-31')],
 [('date','>=','2021-01-01'),('date','<=','2022-01-03')],
 [[('date','<=','2021-12-31')],[('date','>=','2022-01-01')]],
 [('date','>=','2019-12-31'),('date','<=','2021-12-31')]])
def test_denied_before_io(monkeypatch,filters):
    def forbidden(*a,**k):pytest.fail('I/O occurred before rejection')
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    with pytest.raises(PermissionError):DateGate().read_parquet(ROOT/'data/interim/crsp_audit_v1/fake.parquet',columns=['date'],filters=filters)

def test_test_mode_always_denied_before_io(monkeypatch):
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('Test data opened'))
    with pytest.raises(PermissionError):deny_test_execution()

def test_pushdown_and_returned_dates(monkeypatch):
    calls=[]
    def fake(path,**kw):
        calls.append(kw);return pd.DataFrame({'date':pd.to_datetime(['2020-01-02'])})
    monkeypatch.setattr(pd,'read_parquet',fake)
    gate=DateGate();path=ROOT/'data/interim/crsp_audit_v1/fake.parquet'
    gate.read_parquet(path,columns=['date'],filters=[('date','>=','2020-01-01'),('date','<','2022-01-01')])
    assert ('date','<=',pd.Timestamp('2021-12-31')) in calls[0]['filters'] and len(gate.audit)==1
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pd.DataFrame({'date':pd.to_datetime(['2022-01-03'])}))
    with pytest.raises(PermissionError):gate.read_parquet(path,columns=['date'],filters=[('date','>=','2020-01-01'),('date','<=','2021-12-31')])

def test_split_mislabel_and_unknown_path():
    with pytest.raises(PermissionError):DateGate.check_frame(pd.DataFrame({'date':pd.to_datetime(['2021-01-04']),'split':['test']}),'date')
    with pytest.raises(PermissionError):DateGate().read_parquet(ROOT/'other.parquet',columns=['date'],filters=[('date','>=','2020-01-01'),('date','<=','2021-12-31')])

def test_global_schedule_no_year_reset_and_end_purge():
    cal=pd.bdate_range('2015-01-02','2026-01-09') # Deliberately synthetic, not market calendar.
    s=signal_schedule(cal,'2022-01-01','2025-12-31')
    assert all(cal.get_loc(d)%5==0 for d in s.signal_date)
    assert all(cal.get_loc(b)-cal.get_loc(a)==6 for a,b in zip(s.signal_date,s.end_date))
    assert s.end_date.max()<=pd.Timestamp('2025-12-31')
    with pytest.raises(ValueError):signal_schedule(cal[20:],'2022-01-01','2025-12-31')

def test_final_fit_boundary_without_fitting():
    p=pd.DataFrame({'signal_date':pd.to_datetime(['2019-12-30','2021-12-30','2022-01-03']),
       'label_end_date':pd.to_datetime(['2020-01-08','2022-01-10','2022-01-11']),
       'signal_eligible':[True]*3,'label_status':['ok']*3})
    cfg=json.loads((ROOT/'config/features_labels.json').read_text(encoding='utf-8'))
    assert split_masks(p,cfg).eligible_refit_through_2021.tolist()==[True,False,False]

def test_features_future_perturbation_and_prefix():
    dates=pd.bdate_range('2024-01-01',periods=210);rng=np.random.default_rng(42)
    x=pd.DataFrame(index=dates)
    for col,val in {'DlyDelFlg':'N','DlyRetMissFlg':'NA','DlyRetDurFlg':'D1','DlyPrcFlg':'TR','ConditionalType':'RW','TradingStatusFlg':'A',
                    'DlyPrc':20.,'DlyClose':20.,'DlyHigh':21.,'DlyLow':19.,'DlyVol':1000.,'DlyCumFacShr':1.,'ShrOut':10000.,'DlyCap':200000.}.items():x[col]=val
    x['DlyRet']=rng.normal(0,.01,len(x));m=pd.Series(rng.normal(0,.01,len(x)),index=dates)
    cfg=json.loads((ROOT/'config/universe.json').read_text())
    original=make_features(x,m,cfg)
    assert original.iloc[150].notna().all(), 'Fixture must exercise all 21 features'
    changed=x.copy();changed.loc[dates[151]:,'DlyRet']=.9;changed.loc[dates[151]:,'DlyVol']=1e9
    changed_m=m.copy();changed_m.iloc[151:]=-.5
    pd.testing.assert_frame_equal(original.iloc[:151],make_features(changed,changed_m,cfg).iloc[:151])
    pd.testing.assert_frame_equal(original.iloc[:151],make_features(x.iloc[:151],m.iloc[:151],cfg))

def test_portfolio_future_perturbation_and_prefix():
    m=market();t={m['dates'][1]:(1,),m['dates'][3]:(2,)}
    before=simulate(m,t,{},10,2,False,True)[0]
    changed={k:v.copy() for k,v in m.items()};changed['ret'][6:]=.5
    after=simulate(changed,t,{},10,2,False,True)[0]
    pd.testing.assert_frame_equal(before.iloc[:6],after.iloc[:6])
    cut={k:(v if k=='ids' else v[:6]) for k,v in m.items()}
    short=simulate(cut,t,{},10,2,False,True)[0]
    # Prefix final liquidation is an intentional terminal-policy difference.
    pd.testing.assert_frame_equal(before.iloc[:5],short.iloc[:5])

def test_carter_settles_during_exchange_closure():
    m=market();m['dates']=pd.DatetimeIndex(['2025-01-06','2025-01-07','2025-01-08','2025-01-10','2025-01-13','2025-01-14','2025-01-15','2025-01-16','2025-01-17'])
    d,o,h,e=simulate(m,{m['dates'][2]:(1,)},{},10,2,False,True)
    assert o.iloc[0].due_date==pd.Timestamp('2025-01-09')
    assert d.iloc[2].payable>0 and d.iloc[3].payable==0
    verify(d,o,h)

def test_unavailable_receivable_and_double_settlement():
    l=Ledger(0);a=pd.Timestamp('2024-05-24');b=pd.Timestamp('2024-05-28');due=settlement_date(a)
    assert due==settlement_date(b)
    l.post(a,due,100,1);l.post(b,due,-100,2);l.check()
    assert l.budget(due)==0 and l.settle(due)==0

def test_validation_and_test_share_accounting_implementation():
    from equity_research.backtest.ledger import Ledger as SharedLedger
    from equity_research.backtest.validation_calendar import Ledger as ValidationLedger
    assert issubclass(Ledger, SharedLedger)
    assert issubclass(ValidationLedger, SharedLedger)

def test_old_calendar_validation_dates_identical():
    from equity_research.backtest.validation_calendar import settlement_date as old_due
    for d in pd.bdate_range('2020-01-01','2021-12-31'):
        assert settlement_date(d)==old_due(d)

def test_string_calendar_pushdown(monkeypatch):
    def fake(path,**kw):
        assert all(isinstance(v,str) for k,op,v in kw['filters'] if k=='date')
        return pd.DataFrame({'date':['2020-01-02'],'vwretd':[0.]})
    monkeypatch.setattr(pd,'read_parquet',fake)
    DateGate().read_parquet(ROOT/'data/interim/crsp_audit_v1/calendar.parquet',columns=['date','vwretd'],
        filters=[('date','>=','2020-01-01'),('date','<=','2021-12-31')])
