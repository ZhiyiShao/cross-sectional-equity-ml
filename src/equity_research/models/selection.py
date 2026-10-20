"""Fixed-budget selection of four model families on development data only."""
from equity_research.provenance import hash_matches
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from equity_research.data.ingest import ROOT, write_json
from equity_research.models.common import (model_grid, new_model as baseline_model, sha, load_development_data, sample_masks, transform_features,
                             save_lightgbm, circular_block_interval)
from equity_research.models.preprocessing import (require, frame_digest, cross_sectional_rank, fit_state,
                               checked_ic, metric)

OUT = ROOT/'outputs/full_train_selection_v1'
CV = ROOT/'outputs/temporal_cv_v1'
CONFIG = ROOT/'config/full_train_selection.json'
SOURCE = ROOT/'data/processed/research_v1/features_labels.parquet'


def selected_candidates(cfg, unused=None):
    result=[]
    for rep in cfg['representations']:
        for model in ['ridge','lightgbm']:
            for i,params in enumerate(model_grid(cfg,model),1):
                result.append(dict(id=f'{rep}__{model}_{i:02d}',representation=rep,model=model,parameters=params))
    require(len(result)==cfg['model_fit_budget']==80,'Budget mismatch')
    return result


def prior_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for folder in ['baselines_v1', 'temporal_cv_v1', 'cv_selected_validation_v1', 'validation_backtest_v1', 'validation_exposures_v1']
            for p in (ROOT/'outputs'/folder).rglob('*') if p.is_file()}


def pair_results(daily, candidates, cfg):
    table = daily.loc[daily.split.eq('validation')].pivot(index='signal_date', columns='configuration', values='rank_ic')
    output = []
    for model in ['ridge', 'lightgbm']:
        raw = next(c['id'] for c in candidates if c['model'] == model and c['representation'] == 'raw')
        ranked = next(c['id'] for c in candidates if c['model'] == model and c['representation'] == 'cross_sectional_rank')
        difference = table[ranked]-table[raw]
        require(np.isfinite(difference).all(), 'Nonfinite paired IC')
        for block in cfg['bootstrap']['blocks']:
            output.append(dict(model=model, raw=raw, ranked=ranked,
                **circular_block_interval(difference, block, cfg['bootstrap']['replications'], cfg['bootstrap']['seed'])))
    return output


def main():
    require(not OUT.exists() or not any(OUT.iterdir()), 'Nonempty experiment directory; no silent rerun')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    cv_cfg = cfg
    require(not cfg['early_stopping'] and not cfg['test_performance_authorized'] and not cfg['refit_through_2021'], 'Scope mismatch')
    candidates = selected_candidates(cfg, cv_cfg)
    features = json.loads((ROOT/'config/features_labels.json').read_text(encoding='utf-8'))['features']
    frame = load_development_data(SOURCE, features, cfg)
    masks = sample_masks(frame, cfg)
    require(frame.loc[masks['train'], 'label_end_date'].max() < frame.loc[frame.split.eq('validation'), 'signal_date'].min(), 'Train labels cross validation')
    ranked = cross_sectional_rank(frame, features)
    old_hashes = prior_hashes()
    input_paths = [CONFIG, ROOT/'docs/FULL_TRAIN_SELECTION_PROTOCOL.md', ROOT/'config/features_labels.json', ROOT/'src/equity_research/models/selection.py', ROOT/'src/equity_research/models/validation.py', ROOT/'src/equity_research/models/preprocessing.py', ROOT/'src/equity_research/models/common.py']
    input_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths}
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ['models', 'preprocessing']:
        (OUT/folder).mkdir(exist_ok=True)
    write_json(OUT/'run_started.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        configuration=cfg, cv_configuration=cv_cfg, candidates=candidates, features=features,
        input_sha256=input_hashes, original_artifact_sha256=old_hashes,
        filtered_development_sha256=frame_digest(frame), frozen_before_fit=True))
    columns = ['PERMNO', 'signal_date', 'split', 'label_end_date', 'signal_eligible',
               'label_status', 'purged_boundary', 'eligible_main_sample']
    pred = frame[columns].copy()
    pred['y_excess_5d'] = frame.y_excess_5d.where(masks['train'] | masks['validation'])
    pred['momentum20'] = frame.momentum_20.replace([np.inf, -np.inf], np.nan).fillna(frame.loc[masks['train'], 'momentum_20'].median())
    fits, ledger = 0, []
    with threadpool_limits(limits=4):
        for representation in cv_cfg['representations']:
            x = frame[features] if representation == 'raw' else ranked
            state = fit_state(x.loc[masks['train']], representation, cv_cfg)
            state.update(representation=representation, fit_start='2015-01-01', fit_end='2019-12-31',
                         max_fit_label_end=str(frame.loc[masks['train'], 'label_end_date'].max().date()))
            write_json(OUT/'preprocessing'/f'{representation}.json', state)
            X = transform_features(x, state)
            for c in [c for c in candidates if c['representation'] == representation]:
                model = baseline_model(cfg, c['model'], c['parameters'])
                fits += 1
                require(fits <= 80, 'Fit budget exceeded')
                model.fit(X.loc[masks['train']], frame.loc[masks['train'], 'y_excess_5d'])
                pred[c['id']] = model.predict(X)
                if c['model'] == 'ridge':
                    joblib.dump(model, OUT/'models'/f'{c["id"]}.joblib')
                else:
                    save_lightgbm(model, OUT/'models'/f'{c["id"]}.txt')
                ledger.append(dict(configuration=c['id'], parameters=c['parameters'], training_rows=int(masks['train'].sum()),
                                   effective_alpha=float(model.alpha) if c['model'] == 'ridge' else None))
                write_json(OUT/'fit_ledger.json', ledger)
                print(f'Completed fixed fit {fits}/80: {c["id"]}', flush=True)
    require(fits == 80, 'Incomplete experiment')
    pred.to_parquet(OUT/'development_predictions.parquet', index=False)
    metrics, daily_rows, coverage = [], [], []
    for split, mask in masks.items():
        part = frame.loc[mask]
        coverage.append(dict(split=split, candidates=int(frame.split.eq(split).sum()),
            signal_eligible=int((frame.split.eq(split) & frame.signal_eligible).sum()), main_rows=len(part),
            main_dates=part.signal_date.nunique(), boundary_rows=int((frame.split.eq(split) & frame.purged_boundary).sum())))
        for name in ['momentum20']+[c['id'] for c in candidates]:
            values = pred.loc[mask, name]
            ic = checked_ic(part.signal_date, values, part.y_excess_5d)
            daily_rows += [dict(split=split, configuration=name, signal_date=date, rank_ic=float(value)) for date, value in ic.items()]
            groups = [('all', part.index)]+[(str(year), g.index) for year, g in part.groupby(part.signal_date.dt.year)]
            for year, index in groups:
                m = metric(frame.loc[index, 'signal_date'], pred.loc[index, name], frame.loc[index, 'y_excess_5d'])
                if name == 'momentum20':
                    m['rmse'] = np.nan  # Past momentum is a ranking baseline, not a calibrated forward-return estimate.
                metrics.append(dict(split=split, year=year, configuration=name, **m))
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT/'daily_rank_ic.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    scores=pd.DataFrame(metrics).query("split == 'validation' and year == 'all'").set_index('configuration')
    selected=[]
    for rep in cfg['representations']:
        for family in ['ridge','lightgbm']:
            group=[c for c in candidates if c['representation']==rep and c['model']==family]
            selected.append(max(group,key=lambda c: scores.loc[c['id'],'mean_rank_ic']))
    write_json(OUT/'selected.json',selected)
    write_json(OUT/'paired_bootstrap.json', pair_results(daily, selected, cfg))
    pred[[c for c in pred.columns if c not in [x['id'] for x in candidates] or c in [x['id'] for x in selected]]].to_parquet(OUT/'selected_predictions.parquet',index=False)
    require(prior_hashes() == old_hashes, 'Prior experiments changed')
    require(all(hash_matches(ROOT/path, digest) for path, digest in input_hashes.items()), 'Inputs changed during fit')
    outputs = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()}
    write_json(OUT/'run_summary.json', dict(status='completed', model_fits=fits,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), candidate_rows=len(frame),
        train_rows=int(masks['train'].sum()), validation_rows=int(masks['validation'].sum()),
        test_rows_returned=0, test_predictions_generated=0, test_performance_evaluated=False,
        retuning_performed=True, refit_through_2021_performed=False, prior_artifacts_unchanged=True,
        output_sha256=outputs))
    print('All 80 full-period fits complete; independent validation required. Test untouched.', flush=True)
