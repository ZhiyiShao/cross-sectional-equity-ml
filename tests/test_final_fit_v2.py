"""No estimator fits and no historical/unseen data reads in these tests."""
import numpy as np
import pandas as pd
import pytest

from equity_research.models import final_fit_v2 as final
from equity_research.models.preprocessing import cross_sectional_rank
from equity_research.models.verification_v2 import independent_components, verify_contexts
from equity_research.models.context_selection import contexts
from equity_research.evaluation import v2


def synthetic_frame():
    rng = np.random.default_rng(42)
    _, original, added = final.load_plan()
    f = pd.DataFrame(rng.normal(size=(80, len(original))), columns=original)
    f['PERMNO'] = np.tile(np.arange(20), 4)
    f['signal_date'] = np.repeat(pd.to_datetime(['2019-12-30', '2020-01-06', '2021-06-01', '2021-12-30']), 20)
    f['signal_eligible'] = True
    f.loc[0, 'signal_eligible'] = False
    f['industry'] = np.tile(['A']*8+['B']*8+['Unknown']*4, 4)
    f.loc[1, original[0]] = np.nan
    return contexts(f), original, added


def test_components_only_use_qualified_training_rows_and_match_independent_math():
    f, original, added = synthetic_frame()
    mask = f.signal_date.lt('2021-12-01') & f.signal_eligible
    states = final.fit_components(f, mask, original, added)
    independent_components(f, mask, original, added, states)
    # Mutating excluded future dates cannot affect earlier-date ranks or states.
    altered = f.copy()
    altered.loc[f.signal_date.ge('2021-12-01'), original+added] = 1e10
    assert states == final.fit_components(altered, mask, original, added)
    verify_contexts(f)


def test_four_input_bundles_are_shared_by_six_designs():
    f, original, added = synthetic_frame()
    states = final.fit_components(f, f.signal_eligible, original, added)
    ranked = cross_sectional_rank(f, original)
    plan, _, _ = final.load_plan()
    matrices = {c['id']: final.model_matrix(f, ranked, states, c) for c in plan['models']}
    assert [matrices[c['id']].shape[1] for c in plan['models']] == [21, 33, 21, 33, 21, 33]
    for suffix in ['original', 'enhanced']:
        pd.testing.assert_frame_equal(matrices['lightgbm_'+suffix], matrices['ranker_'+suffix])


def test_contexts_ignore_label_status_and_future_dates():
    f, original, _ = synthetic_frame()
    changed = f.copy()
    changed['label_status'] = 'missing'
    changed['y_excess_5d'] = 1e9
    changed.loc[changed.signal_date.eq(changed.signal_date.max()), original] *= 100
    expected = contexts(f)
    actual = contexts(changed)
    pd.testing.assert_frame_equal(expected.iloc[:60], actual[expected.columns].iloc[:60])


def test_industry_reader_projects_no_labels_and_rejects_out_of_range(monkeypatch):
    f, original, _ = synthetic_frame()
    base = f.drop(columns='industry')
    monkeypatch.setattr(final, 'read_final_inputs', lambda _: (base, base.signal_eligible, []))
    def read(path, columns, filters):
        assert columns == ['PERMNO', 'signal_date', 'industry']
        assert ('signal_date', '<=', final.END) in filters
        return f[columns].assign(signal_date=pd.Timestamp('2022-01-03'))
    monkeypatch.setattr(pd, 'read_parquet', read)
    with pytest.raises(ValueError, match='Industry date breach'):
        final.load_training(original)


def test_existing_output_blocks_fit_before_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'OUT', tmp_path)
    monkeypatch.setattr(final, 'load_plan', lambda: pytest.fail('Must reject before input access'))
    with pytest.raises(ValueError, match='no implicit refit'):
        final.main()


def test_prediction_dates_and_targets_rejected_before_model_access(monkeypatch):
    f, _, _ = synthetic_frame()
    monkeypatch.setattr(v2, 'read_json', lambda _: pytest.fail('No artifact access before boundary checks'))
    with pytest.raises(ValueError, match='date breach'):
        v2.predict_frame(f, {}, (pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31')))
    with pytest.raises(ValueError, match='feature-only'):
        v2.predict_frame(f.assign(y_excess_5d=0), {}, (final.START, final.END))


def release_fixture():
    plan, _, _ = final.load_plan()
    manifest = {'model': 'hash'}
    release = dict(authorized=True, authorization_record='explicit user authorization',
        plan_sha256=final.sha(final.PLAN), model_manifest=manifest, mode='historical',
        period=['2022-01-01', '2025-12-31'], full_period_coverage=True,
        calendar_verified=True, events_resolved=True, ledger_verified=True,
        same_frozen_models=True, audit_record={'path': 'unused', 'sha256': 'unused'})
    return plan, manifest, release


@pytest.mark.parametrize('key', ['authorized', 'full_period_coverage', 'calendar_verified', 'events_resolved', 'ledger_verified', 'same_frozen_models'])
def test_release_fails_closed(key):
    plan, manifest, release = release_fixture()
    release[key] = False
    with pytest.raises(ValueError):
        v2.validate_release(release, plan, manifest)


def test_unseen_requires_eligibility_and_complete_year_before_opening():
    plan, manifest, release = release_fixture()
    release.update(mode='unseen', period=['2026-01-01', '2026-12-31'], as_of='2026-11-07')
    with pytest.raises(ValueError, match='eligibility'):
        v2.validate_release(release, plan, manifest)
    release.update(access_audit_status='eligible', access_audit_record='documented audit')
    with pytest.raises(ValueError, match='Incomplete calendar year'):
        v2.validate_release(release, plan, manifest)
    release['as_of'] = '2027-02-01'
    assert v2.validate_release(release, plan, manifest)[0] == pd.Timestamp('2026-01-01')
    release['fallback_trigger'] = 'poor_returns'
    with pytest.raises(ValueError, match='fallback trigger'):
        v2.validate_release(release, plan, manifest)


def synthetic_daily(n=252):
    return pd.DataFrame([dict(date=d, model=m, return_net=.001 if m=='ridge_enhanced' else 0.)
                         for d in pd.bdate_range('2020-01-01', periods=n) for m in final.IDS])


def test_primary_is_arithmetic_paired_difference_not_cagr():
    plan, _, _ = final.load_plan()
    result = v2.primary_comparison(synthetic_daily(), plan)
    assert result['annual_mean_net_difference'] == pytest.approx(.252)
    assert result['ci_low'] == pytest.approx(.252)
    assert result['conclusion'] == 'supports_predeclared_improvement'
    assert v2.primary_comparison(synthetic_daily(19), plan)['conclusion'] == 'descriptive_only'


def test_missing_model_date_cannot_be_silently_inner_joined():
    plan, _, _ = final.load_plan()
    with pytest.raises(ValueError, match='identical finite dates'):
        v2.primary_comparison(synthetic_daily().iloc[1:], plan)


def test_prediction_metrics_continuous_ic_frozen_ndcg_and_boundary():
    plan, _, _ = final.load_plan()
    f = pd.DataFrame({'PERMNO': range(20), 'signal_date': pd.Timestamp('2020-01-02'),
                      'label_end_date': pd.Timestamp('2020-01-10'), 'label_status': 'ok',
                      'signal_eligible': True, 'y_excess_5d': np.arange(20.)})
    for name in final.IDS:
        f[name] = np.arange(20.)
    result = v2.prediction_metrics(f, plan, (pd.Timestamp('2020-01-01'), final.END))
    np.testing.assert_allclose(result[['rank_ic', 'ndcg_top10']], 1.)
    f['label_end_date'] = pd.Timestamp('2022-01-03')
    with pytest.raises(ValueError, match='horizon breach'):
        v2.prediction_metrics(f, plan, (pd.Timestamp('2020-01-01'), final.END))


def test_no_release_stops_before_observation_read(monkeypatch, tmp_path):
    plan, manifest, _ = release_fixture()
    monkeypatch.setattr(v2, 'locked_bundle', lambda: (plan, {}, manifest))
    path = tmp_path/'release.json'
    path.write_text('{}')
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('Must not read observations'))
    with pytest.raises(ValueError, match='authorization'):
        v2.run_evaluation(path)


def test_evaluation_entry_with_synthetic_audited_bundle(tmp_path, monkeypatch):
    """Exercise the actual IO/summary entry, but only with manufactured observations."""
    plan, manifest, release = release_fixture()
    plan['historical_evaluation']['dates'] = ['2020-01-01', '2021-12-31']
    release['period'] = plan['historical_evaluation']['dates']
    folder = tmp_path/'outputs/synthetic'
    folder.mkdir(parents=True)
    daily = synthetic_daily()
    scores = pd.DataFrame([dict(PERMNO=i, signal_date=date,
        label_end_date=date+pd.Timedelta(days=8), signal_eligible=True, label_status='ok',
        y_excess_5d=float(i), **{m: float(i) for m in final.IDS})
        for date in pd.date_range('2020-01-02', periods=50, freq='7D') for i in range(20)])
    release['sources'] = {}
    for name, table in [('daily', daily), ('predictions', scores)]:
        path = folder/(name+'.parquet')
        table.to_parquet(path, index=False)
        release['sources'][name] = dict(path=path.relative_to(tmp_path).as_posix(), sha256=final.sha(path), rows=len(table))
    audit = dict(status='passed', model_manifest=manifest, period=release['period'],
        sources=release['sources'], cost_bps_per_side=10, variant='constrained', nav_anchor_excluded=True)
    path = folder/'audit.json'
    final.write_json(path, audit)
    release['audit_record'] = dict(path=path.relative_to(tmp_path).as_posix(), sha256=final.sha(path))
    release_path = folder/'release.json'
    final.write_json(release_path, release)
    monkeypatch.setattr(v2, 'ROOT', tmp_path)
    monkeypatch.setattr(v2, 'locked_bundle', lambda: (plan, {}, manifest))
    result = v2.run_evaluation(release_path)
    assert result['annual_mean_net_difference'] == pytest.approx(.252)
    assert result['valid_signal_dates'] == 50
    assert result['conclusion'] == 'supports_predeclared_improvement'
    with pytest.raises(ValueError, match='Preserve existing evaluation'):
        v2.run_evaluation(release_path)
