"""Offline synthetic walkthrough using the research pipeline's core functions.

Only this module generates fixtures and orchestrates a small experiment. Universe
rules, features, labels, preprocessing, estimators, targets and accounting come
from the production modules. No historical research entry point is invoked.
"""
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from equity_research.paths import ROOT
from equity_research.data.ingest import write_json
from equity_research.data.universe import (
    SINGLE_DAY, filter_codes, monthly_snapshots, regular_trade, select_month,
)
from equity_research.data.features import make_features, return_segments, label_one, split_masks
from equity_research.models.preprocessing import (
    cross_sectional_rank, fit_state, transform_features, require,
)
from equity_research.models.common import (
    daily_rank_ic, model_grid, new_model, sample_masks, save_lightgbm, load_lightgbm, sha,
)
from equity_research.backtest.market import build_market, make_targets, performance
from equity_research.backtest.engine import simulate, verify
from equity_research.backtest.ledger import Ledger
from equity_research.backtest.settlement import SettlementCalendar

MODEL_NAMES = {
    'raw__ridge': 'Raw Ridge', 'cross_sectional_rank__ridge': 'Rank Ridge',
    'raw__lightgbm': 'Raw LightGBM', 'cross_sectional_rank__lightgbm': 'Rank LightGBM',
    'momentum20': 'Momentum 20', 'equal_weight': 'Universe EW',
}


def synthetic_market(cfg):
    """Fictional CIZ-shaped rows, including filters, a late listing and cash exit."""
    rng = np.random.default_rng(cfg['seed'])
    dates = pd.bdate_range(*cfg['data_dates'])  # Deliberately fictional weekday calendar.
    market = pd.Series(rng.normal(.0001, .008, len(dates)), index=dates, name='vwretd')
    frames, terms = [], {}
    for i in range(cfg['securities']):
        permno = 900000 + i
        noise = rng.normal(0, .012, len(dates))
        returns = (.7 + .6 * rng.random()) * market.to_numpy() + noise
        returns[1:] += .08 * noise[:-1]
        price = (1. if i == 0 else rng.uniform(25, 70)) * np.cumprod(1 + returns)
        volume = rng.lognormal(12.5, .3, len(dates))
        volume *= np.exp(.7 * np.sin(np.arange(len(dates)) / 80 + i))
        if i == 7:
            volume *= 100  # Keep the synthetic cash event in the EW portfolio.
        shares_thousands = rng.uniform(5000, 50000)
        x = pd.DataFrame(dict(
            PERMNO=permno, PERMCO=800000 + (4 if i == 5 else i), Ticker=f'SYN{i:03d}',
            DlyCalDt=dates, DlyClose=price, DlyPrc=price, DlyRet=returns, DlyRetx=returns,
            DlyHigh=price * (1 + rng.uniform(.001, .025, len(dates))),
            DlyLow=price * (1 - rng.uniform(.001, .025, len(dates))),
            DlyVol=np.round(volume), ShrOut=shares_thousands, DlyCap=price * shares_thousands,
            DlyCumFacPr=1., DlyCumFacShr=1., DlyDelFlg='N', DlyPrcFlg='TR',
            DlyRetMissFlg='NA', DlyRetDurFlg='D1', ConditionalType='RW',
            TradingStatusFlg='A', PrimaryExch='N', SecurityType='EQTY',
            SecuritySubType='ETF' if i == 2 else 'COM', ShareType='NS',
            USIncFlg='N' if i == 1 else 'Y', IssuerType='CORP',
        ))
        if i == 3:
            x['ConditionalType'] = 'NW'
        if i == 6:
            x = x.loc[x.DlyCalDt.ge('2020-07-01')].copy()
        if i == 7:
            event = pd.Timestamp(cfg['cash_event_date'])
            terminal_price = float(x.loc[x.DlyCalDt.lt(event), 'DlyClose'].iloc[-1] * 1.05)
            x = x.loc[x.DlyCalDt.le(event)].copy()
            terminal = x.DlyCalDt.eq(event)
            x.loc[terminal, ['DlyDelFlg', 'DlyRetDurFlg']] = ['Y', 'DD']
            x.loc[terminal, ['DlyRet', 'DlyRetx']] = .05
            x.loc[terminal, ['DlyPrc', 'DlyClose', 'DlyHigh', 'DlyLow']] = terminal_price
            x.loc[terminal, 'DlyCap'] = terminal_price * shares_thousands
            terms[permno] = dict(event_date=str(event.date()), cash_per_share=terminal_price,
                                special_per_share=0., special_expected_date=None,
                                ratio=0., successor_permno=0)
        frames.append(x)
    return pd.concat(frames, ignore_index=True), market, terms


def build_dataset(raw, market, cfg, universe_cfg, feature_cfg):
    """Previous-month snapshots -> historical members -> causal features/labels."""
    dates = market.index
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period('M')).max()
    first = pd.Period(cfg['train_dates'][0], 'M')
    last = pd.Period(cfg['holdout_dates'][1], 'M')
    formations = month_ends.loc[(month_ends.index >= first - 1) & (month_ends.index < last)]
    snapshots = pd.concat([
        monthly_snapshots(g, dates, formations.to_numpy(), universe_cfg)
        for _, g in raw.groupby('PERMNO', sort=True)
    ], ignore_index=True)
    members, funnels = [], []
    for date, group in snapshots.groupby('formation_date', sort=True):
        selected, funnel = select_month(group, universe_cfg)
        selected['effective_month'] = str(date.to_period('M') + 1)
        members.append(selected)
        funnels.append(dict(formation_date=date, **funnel))
    universe = pd.concat(members, ignore_index=True)
    require(not universe.duplicated(['effective_month', 'PERMCO']).any(), 'Duplicate company')
    require(universe.effective_month.nunique() == len(pd.period_range(first, last, freq='M')),
            'Incomplete demo universe')

    start = dates.searchsorted(pd.Timestamp(cfg['train_dates'][0]))
    positions = np.arange(start, len(dates), feature_cfg['signal_step_trading_days'])
    signals = dates[positions]
    panels = []
    by_security = {int(p): g for p, g in raw.groupby('PERMNO')}
    for permno, membership in universe.groupby('PERMNO'):
        x = by_security[int(permno)].set_index('DlyCalDt').reindex(dates)
        features = make_features(x, market, universe_cfg)
        trade = regular_trade(x, universe_cfg).to_numpy()
        entry_trade = trade & filter_codes(x, universe_cfg['security_filters']).to_numpy()
        segments = return_segments(x)
        membership = membership.set_index('effective_month')
        keep = signals.to_period('M').astype(str).isin(membership.index)
        chosen = signals[keep]
        part = features.loc[chosen].reset_index(drop=True)
        part.insert(0, 'signal_date', chosen)
        part.insert(0, 'PERMNO', int(permno))
        part['formation_date'] = [membership.loc[str(d.to_period('M')), 'formation_date'] for d in chosen]
        part['signal_eligible'] = entry_trade[positions[keep]]
        labels = [label_one(int(t), x, market, entry_trade, trade, segments,
                            feature_cfg['entry_lag_trading_days'], feature_cfg['holding_period_trading_days'])
                  for t in positions[keep]]
        panels.append(pd.concat([part, pd.DataFrame(labels)], axis=1))
    panel = pd.concat(panels, ignore_index=True).sort_values(['signal_date', 'PERMNO']).reset_index(drop=True)
    panel = split_masks(panel, dict(splits={
        'train': cfg['train_dates'], 'validation': cfg['validation_dates'], 'test': cfg['holdout_dates'],
    }))
    require(panel.formation_date.lt(panel.signal_date).all(), 'Universe look-ahead')
    require(not panel.duplicated(['signal_date', 'PERMNO']).any(), 'Duplicate sample')
    return universe, pd.DataFrame(funnels), panel


def fit_models(development, features, cfg, out):
    """Eight candidate fits on train; choose on validation; four final refits.

    Holdout rows are never accepted here. Refit may restore labels crossing the
    internal train/validation boundary, but never the final development boundary.
    """
    require(development.signal_date.lt(cfg['test_start_exclusive']).all(), 'Holdout passed to fit')
    masks = sample_masks(development, cfg)
    ranked = cross_sectional_rank(development, features)
    final_end = pd.Timestamp(cfg['validation_dates'][1])
    refit = (development.signal_eligible & development.label_status.isin(['ok', 'ok_bridged'])
             & development.label_end_date.le(final_end)
             & development.label_end_date.gt(development.signal_date)
             & development.y_excess_5d.notna())
    scores, selected, ledger, states, models = [], [], [], {}, {}
    (out / 'models').mkdir()
    (out / 'preprocessing').mkdir()

    def record(name, phase, mask, params):
        rows = development.loc[mask]
        ledger.append(dict(model=name, phase=phase, parameters=params, rows=len(rows),
                           min_signal=str(rows.signal_date.min().date()),
                           max_signal=str(rows.signal_date.max().date()),
                           max_label_end=str(rows.label_end_date.max().date())))

    for representation in cfg['representations']:
        x = development[features] if representation == 'raw' else ranked
        state = fit_state(x.loc[masks['train']], representation, cfg)
        write_json(out / 'preprocessing' / f'{representation}_selection.json', state)
        transformed = transform_features(x, state)
        for family in ['ridge', 'lightgbm']:
            name = f'{representation}__{family}'
            candidates = []
            for trial, params in enumerate(model_grid(cfg, family), 1):
                model = new_model(cfg, family, params)
                model.fit(transformed.loc[masks['train']], development.loc[masks['train'], 'y_excess_5d'])
                validation = development.loc[masks['validation']]
                prediction = model.predict(transformed.loc[masks['validation']])
                ic = daily_rank_ic(validation.signal_date, prediction, validation.y_excess_5d)
                require(np.isfinite(ic).all(), 'Undefined validation IC')
                candidate = dict(model=name, trial=trial, parameters=params, mean_rank_ic=float(ic.mean()))
                candidates.append(candidate)
                scores.append(candidate)
                record(name, 'selection', masks['train'], params)
            selected.append(max(candidates, key=lambda item: item['mean_rank_ic']))
    require(len(ledger) == cfg['selection_fit_budget'], 'Selection budget mismatch')
    # Persist all selections before any final fit or holdout prediction.
    write_json(out / 'selected.json', selected)
    pd.DataFrame(scores).to_csv(out / 'validation_scores.csv', index=False)
    for representation in cfg['representations']:
        x = development[features] if representation == 'raw' else ranked
        state = fit_state(x.loc[refit], representation, cfg)
        states[representation] = state
        write_json(out / 'preprocessing' / f'{representation}_final.json', state)
        transformed = transform_features(x.loc[refit], state)
        for family in ['ridge', 'lightgbm']:
            name = f'{representation}__{family}'
            params = next(item['parameters'] for item in selected if item['model'] == name)
            model = new_model(cfg, family, params)
            model.fit(transformed, development.loc[refit, 'y_excess_5d'])
            path = out / 'models' / (name + ('.joblib' if family == 'ridge' else '.txt'))
            if family == 'ridge':
                joblib.dump(model, path)
                loaded = joblib.load(path)
            else:
                save_lightgbm(model, path)
                loaded = load_lightgbm(path)
            np.testing.assert_allclose(model.predict(transformed), loaded.predict(transformed), atol=1e-12, rtol=1e-12)
            models[name] = loaded
            record(name, 'final', refit, params)
    require(len(ledger) == cfg['selection_fit_budget'] + cfg['final_fit_budget'], 'Final budget mismatch')
    write_json(out / 'fit_ledger.json', ledger)
    momentum_fill = float(development.loc[refit, 'momentum_20'].median())
    write_json(out / 'baseline_state.json', dict(momentum_missing=momentum_fill))
    return states, models, momentum_fill


def predict_holdout(frame, features, states, models, momentum_fill):
    """Input contains features and eligibility only; saved models produce scores."""
    pred = frame[['PERMNO', 'signal_date', 'signal_eligible']].copy()
    for representation, state in states.items():
        x = frame[features] if representation == 'raw' else cross_sectional_rank(frame, features)
        transformed = transform_features(x, state)
        for name, model in models.items():
            if name.startswith(representation + '__'):
                pred[name] = model.predict(transformed)
    pred['momentum20'] = frame.momentum_20.replace([np.inf, -np.inf], np.nan).fillna(momentum_fill)
    require(np.isfinite(pred.loc[pred.signal_eligible, list(MODEL_NAMES)[:-1]]).all().all(),
            'Nonfinite holdout prediction')
    return pred


def evaluate_portfolios(raw, market, pred, labels, terms, cfg, feature_cfg, out):
    """Score-only selection, shared execution/ledger, independent reconciliation."""
    dates = market.index[market.index.to_series().between(*cfg['holdout_dates']).to_numpy()]
    lag, horizon = feature_cfg['entry_lag_trading_days'], feature_cfg['holding_period_trading_days']
    schedule = []
    for signal in sorted(pred.signal_date.unique()):
        pos = dates.get_loc(signal)
        if pos + lag + horizon < len(dates):
            schedule.append(dict(signal_date=signal, entry_date=dates[pos + lag], end_date=dates[pos + lag + horizon]))
    schedule = pd.DataFrame(schedule)
    dates = dates[(dates >= schedule.signal_date.min()) & (dates <= schedule.end_date.max())]
    pred = pred.loc[pred.signal_date.isin(schedule.signal_date)]
    targets = make_targets(pred, schedule, models=list(MODEL_NAMES))
    raw = raw.loc[raw.DlyCalDt.isin(dates) & raw.PERMNO.isin(pred.PERMNO)].copy()
    universe_cfg = json.loads((ROOT / 'config/universe.json').read_text(encoding='utf-8'))
    raw['sellable'] = regular_trade(raw, universe_cfg)
    raw['buyable'] = raw.sellable & filter_codes(raw, universe_cfg['security_filters'])
    valid = raw.DlyRetMissFlg.eq('NA') & np.isfinite(raw.DlyRet) & raw.DlyRet.ge(-1)
    raw['ordinary_return'] = valid & raw.DlyDelFlg.eq('N') & raw.DlyRetDurFlg.isin(SINGLE_DAY)
    raw['known_terminal'] = valid & raw.DlyDelFlg.eq('Y') & raw.DlyRetDurFlg.isin(SINGLE_DAY + ['DD'])
    arrays = build_market(raw, dates, set(pred.PERMNO))
    arrays['price'] = raw.pivot(index='DlyCalDt', columns='PERMNO', values='DlyPrc').reindex(index=dates, columns=arrays['ids']).to_numpy()
    calendar_path = out / 'synthetic_settlement_calendar.json'
    write_json(calendar_path, dict(synthetic_data_only=True, supported_windows=[cfg['data_dates']],
                                  non_settlement_weekdays=[], note='Fictional weekdays; no US holiday modelling.'))
    calendar = SettlementCalendar(calendar_path)
    ledger_factory = partial(Ledger, is_settlement_day=calendar.is_settlement_day)
    joined = pred.merge(labels[['PERMNO', 'signal_date', 'y_excess_5d', 'eligible_main_sample']],
                        on=['PERMNO', 'signal_date'], validate='one_to_one')
    valid_labels = joined.loc[joined.eligible_main_sample]
    metrics, daily_rows, orders, holdings, events, ic_rows = [], [], [], [], [], []
    for name in MODEL_NAMES:
        daily, trades, positions, actions = simulate(
            arrays, targets[name], terms, cfg['cost_bps'], cfg['action_availability_trading_days'],
            record=True, ledger_factory=ledger_factory, settlement_function=calendar.settlement_date,
        )
        verify(daily, trades, positions, settlement_function=calendar.settlement_date)
        ic = (daily_rank_ic(valid_labels.signal_date, valid_labels[name], valid_labels.y_excess_5d)
              if name != 'equal_weight' else pd.Series(dtype=float, name='rank_ic'))
        metrics.append(dict(model=name, synthetic_data_only=True, **performance(daily),
                            mean_rank_ic=float(ic.mean()), ic_dates=int(ic.notna().sum()),
                            label_coverage=len(valid_labels) / int(pred.signal_eligible.sum())))
        for target, frame in [(daily_rows, daily), (orders, trades), (holdings, positions), (events, actions)]:
            if not frame.empty:
                target.append(frame.assign(model=name))
        if not ic.empty:
            ic_rows.append(ic.rename_axis('signal_date').reset_index().assign(model=name))
    daily = pd.concat(daily_rows, ignore_index=True)
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / 'summary.csv', index=False)
    daily.to_csv(out / 'daily.csv', index=False)
    pd.concat(orders, ignore_index=True).to_csv(out / 'orders.csv', index=False)
    pd.concat(holdings, ignore_index=True).to_parquet(out / 'holdings.parquet', index=False)
    pd.concat(events, ignore_index=True).to_csv(out / 'events.csv', index=False)
    pd.concat(ic_rows, ignore_index=True).to_csv(out / 'daily_rank_ic.csv', index=False)
    schedule.to_csv(out / 'schedule.csv', index=False)
    return summary, daily


def plot_results(summary, daily, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for name, label in MODEL_NAMES.items():
        rows = daily.loc[daily.model.eq(name)]
        axes[0].plot(rows.date, rows.nav, label=label, linewidth=1)
    axes[0].set(title='Synthetic holdout NAV | 10 bps per side', ylabel='NAV (initial = 1)')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x', labelrotation=30)
    scores = summary.loc[summary.model.ne('equal_weight')]
    axes[1].bar([MODEL_NAMES[n] for n in scores.model], scores.mean_rank_ic)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(title='Synthetic 5-day Rank IC')
    axes[1].tick_params(axis='x', labelrotation=30)
    fig.suptitle('SYNTHETIC DEMO ONLY — pipeline illustration, not investment evidence')
    fig.savefig(out / 'demo_overview.png', dpi=140)
    plt.close(fig)


def run_demo(output=None):
    """Write one self-contained run into an empty directory; never overwrite."""
    out = Path(output) if output is not None else ROOT / 'outputs/demo'
    out = out.resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            f'Demo output must be an empty directory: {out}')
    cfg = json.loads((ROOT / 'config/demo.json').read_text(encoding='utf-8'))
    require(cfg['synthetic_data_only'], 'Demo must use synthetic data')
    universe_cfg = json.loads((ROOT / 'config/universe.json').read_text(encoding='utf-8'))
    universe_cfg['max_securities'] = cfg['max_securities']
    feature_cfg = json.loads((ROOT / 'config/features_labels.json').read_text(encoding='utf-8'))
    features = feature_cfg['features']
    out.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / 'config' / name for name in ['demo.json', 'universe.json', 'features_labels.json']]
    inputs += sorted((ROOT / 'src/equity_research').rglob('*.py'))
    manifest = dict(status='started', synthetic_data_only=True, configuration=cfg, features=features,
                    input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                    versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'pyarrow']})
    write_json(out / 'manifest.json', manifest)
    print('[1/5] Generate fictional daily bars and a cash corporate action', flush=True)
    raw, market, terms = synthetic_market(cfg)
    raw.to_parquet(out / 'synthetic_prices.parquet', index=False)
    market.rename_axis('date').reset_index().to_csv(out / 'synthetic_market.csv', index=False)
    write_json(out / 'synthetic_actions.json', terms)
    print('[2/5] Build monthly universe, 21 features and purged 5-day labels', flush=True)
    universe, funnel, panel = build_dataset(raw, market, cfg, universe_cfg, feature_cfg)
    universe.to_csv(out / 'universe.csv', index=False)
    funnel.to_csv(out / 'universe_funnel.csv', index=False)
    panel.to_parquet(out / 'features_labels.parquet', index=False)
    development = panel.loc[panel.split.isin(['train', 'validation'])].copy()
    holdout = panel.loc[panel.split.eq('test')].copy()
    print('[3/5] Select four models with 8 fits, then refit on development (4 fits)', flush=True)
    with threadpool_limits(limits=2):
        states, models, momentum_fill = fit_models(development, features, cfg, out)
        pred = predict_holdout(holdout[['PERMNO', 'signal_date', 'signal_eligible'] + features],
                               features, states, models, momentum_fill)
    pred.to_parquet(out / 'predictions.parquet', index=False)
    print('[4/5] Run six portfolios through the shared trading and settlement ledger', flush=True)
    summary, daily = evaluate_portfolios(raw, market, pred, holdout, terms, cfg, feature_cfg, out)
    print('[5/5] Reconciliation passed; save metrics, chart and reproducibility manifest', flush=True)
    plot_results(summary, daily, out)
    manifest.update(status='completed', selection_fits=cfg['selection_fit_budget'], final_fits=cfg['final_fit_budget'],
                    candidate_rows=len(panel), portfolios=len(summary), accounting_verified=True,
                    sample_counts=panel.groupby('split').eligible_main_sample.sum().astype(int).to_dict(),
                    output_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
                                   if p.is_file() and p.name != 'manifest.json'})
    write_json(out / 'manifest.json', manifest)
    print('\nSYNTHETIC DEMO ONLY: these numbers do not measure investment performance.')
    print(summary[['model', 'cagr', 'sharpe', 'max_drawdown', 'mean_rank_ic']].to_string(index=False))
    print(f'\nOutputs: {out}')
    return out
