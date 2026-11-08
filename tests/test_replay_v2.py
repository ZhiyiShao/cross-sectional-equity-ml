import ast
import numpy as np
import pandas as pd
import pytest
from equity_research.backtest import replay_v2 as replay
from equity_research.models.final_fit_v2 import IDS, load_plan
from equity_research.evaluation.replay_v2 import prediction_diagnostics


@pytest.mark.parametrize('source,start,end',[
    ('unknown.parquet','2022-01-01','2025-12-31'),
    ('data/processed/research_v1/features_only.parquet','2022-01-01','2026-01-01'),
    ('data/processed/research_v1/features_only.parquet','2021-01-01','2025-12-31'),
    ('data/processed/research_v1/features_only.parquet','2025-01-01','2024-01-01')])
def test_scope_rejected_before_io(monkeypatch,source,start,end):
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('Forbidden read'))
    with pytest.raises(ValueError):replay.ReplayReader().read(source,['signal_date'],start,end)


def test_calendar_warmup_cannot_read_market_returns(monkeypatch):
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('Forbidden read'))
    with pytest.raises(ValueError,match='may not read returns'):
        replay.ReplayReader().read('data/interim/crsp_audit_v1/calendar.parquet',['date','vwretd'],'2015-01-02')


def test_label_horizon_pushdown_and_returned_date_check(monkeypatch):
    def read(path,columns,filters):
        assert ('label_end_date','<=',replay.END) in filters
        assert ('signal_date','<=',replay.END) in filters
        return pd.DataFrame(dict(signal_date=[pd.Timestamp('2025-12-16')],label_end_date=[pd.Timestamp('2026-01-02')],y_excess_5d=[.1]))
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='target horizon breach'):
        replay.ReplayReader().read('data/processed/research_v1/features_labels.parquet',['signal_date','label_end_date','y_excess_5d'])


def fixture():
    n=100;dates=pd.to_datetime(['2022-01-05','2022-01-12'])
    f=pd.DataFrame(dict(PERMNO=np.tile(np.arange(n),2),signal_date=np.repeat(dates,n),signal_eligible=True,
        industry=np.tile(['A']*50+['B']*50,2),market_beta_60=1.,volatility_20=.3,log_market_cap=20.))
    pred=f[['PERMNO','signal_date','signal_eligible']].copy()
    for i,model in enumerate(IDS+['momentum20']):pred[model]=np.sin(np.arange(n*2)+i)
    s=pd.DataFrame(dict(signal_date=dates,entry_date=dates+pd.Timedelta(days=1)))
    return f,pred,s


def test_six_by_two_plus_benchmarks_on_same_schedule():
    plan,_,_=load_plan();f,p,s=fixture()
    targets,snapshots,ex=replay.target_sets(p,f,s,plan['portfolio']['constraints'])
    assert len(targets)==14 and len(ex)==28 and len(snapshots)==2
    for dates in targets.values():assert set(dates)==set(s.entry_date)
    replay.check_targets(ex,plan['portfolio']['constraints'])
    for m in IDS:assert len(targets[(m,'original')][s.entry_date.iloc[0]])==10


def test_exposure_does_not_hide_inherited_outside_universe():
    f,_,_=fixture();g=f[f.signal_date.eq(f.signal_date.iloc[0])].copy();g['annual_vol']=g.volatility_20
    r=replay.exposure(g,{0:.5,1000:.5})
    assert r['coverage']==.5 and r['outside_universe_weight']==.5 and r['total_equity_weight']==1


def test_replay_sources_have_no_fit_calls():
    for path in [replay.ROOT/'src/equity_research/backtest/replay_v2.py',replay.ROOT/'src/equity_research/evaluation/replay_v2.py']:
        tree=ast.parse(path.read_text(encoding='utf-8'))
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['fit','fit_transform'] for n in ast.walk(tree))


def test_top_membership_precedes_label_availability_filter():
    plan,_,_=load_plan()
    dates=pd.date_range('2022-01-05',periods=10,freq='7D')
    f=pd.DataFrame([dict(PERMNO=i,signal_date=d,label_end_date=d+pd.Timedelta(days=8),
        signal_eligible=True,label_status='ok' if i<19 else 'missing',
        y_excess_5d=float(i) if i<19 else np.nan,
        **{model:float(i) for model in IDS+['momentum20']}) for d in dates for i in range(20)])
    coverage,ic,top,summary,pairs=prediction_diagnostics(f,plan)
    assert coverage.eligible.eq(20).all() and coverage.valid_labels.eq(19).all()
    # Top two are IDs 19 and 18. The missing label must not promote ID 17.
    assert top.selected.eq(2).all() and top.valid_labels.eq(1).all()
    assert top.top_label_mean.eq(18).all() and top.coverage.eq(.5).all()
    np.testing.assert_allclose(ic.rank_ic,1.)
