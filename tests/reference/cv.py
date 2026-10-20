"""Frozen-budget internal temporal CV, reading only original training-period rows."""
import itertools
import json
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from equity_research.data.ingest import ROOT, write_json
from equity_research.models.common import fit_preprocessing, transform_features, daily_rank_ic, save_lightgbm, sha

OUT = ROOT/'outputs/temporal_cv_v1'
CONFIG = ROOT/'config/temporal_cv.json'
SOURCE = ROOT/'data/processed/research_v1/features_labels.parquet'


def load_training_only(path, features):
    columns = ['PERMNO', 'signal_date', 'split', 'label_end_date', 'signal_eligible',
               'label_status', 'y_excess_5d'] + features
    frame = pd.read_parquet(path, columns=columns, engine='pyarrow', filters=[
        ('signal_date', '>=', pd.Timestamp('2015-01-01')),
        ('signal_date', '<', pd.Timestamp('2020-01-01')), ('split', '==', 'train')])
    require(len(frame) > 0 and set(frame.split) == {'train'}, 'Not original training data')
    require(frame.signal_date.between('2015-01-01', '2019-12-31').all(), 'Scope breach')
    require(not frame.duplicated(['signal_date', 'PERMNO']).any(), 'Duplicate candidates')
    return frame.sort_values(['signal_date', 'PERMNO']).reset_index(drop=True)


def fold_masks(frame, fold):
    result = {}
    known = (frame.signal_eligible & frame.label_status.isin(['ok', 'ok_bridged'])
             & np.isfinite(frame.y_excess_5d) & frame.label_end_date.gt(frame.signal_date))
    for side in ['train', 'validation']:
        start, end = pd.Timestamp(fold[side+'_start']), pd.Timestamp(fold[side+'_end'])
        candidates = frame.signal_date.between(start, end)
        result[side+'_candidates'] = candidates
        result[side] = candidates & known & frame.label_end_date.le(end)
        require(result[side].any(), 'Empty fold sample')
    require(not (result['train_candidates'] & result['validation_candidates']).any(), 'Fold overlap')
    require(frame.loc[result['train'], 'label_end_date'].max() <
            frame.loc[result['validation_candidates'], 'signal_date'].min(), 'Training labels overlap validation')
    return result


def configurations(cfg):
    grid = []
    for representation in cfg['representations']:
        for trial, strength in enumerate(cfg['ridge_lambda'], 1):
            grid.append(dict(id=f'{representation}__ridge_{trial:02d}', representation=representation,
                             model='ridge', trial=trial, parameters={'lambda': strength}))
        for trial, (leaves, child) in enumerate(itertools.product(
                cfg['lightgbm']['num_leaves'], cfg['lightgbm']['min_child_samples']), 1):
            grid.append(dict(id=f'{representation}__lightgbm_{trial:02d}', representation=representation,
                             model='lightgbm', trial=trial,
                             parameters={'num_leaves': leaves, 'min_child_samples': child}))
    require(len(grid) == cfg['configuration_budget'], 'Configuration budget mismatch')
    require(len(grid)*len(cfg['folds']) == cfg['model_fit_budget'], 'Fit budget mismatch')
    return grid


def new_model(cfg, candidate, rows):
    if candidate['model'] == 'ridge':
        return Ridge(alpha=rows*candidate['parameters']['lambda'], solver='cholesky', fit_intercept=True)
    return lgb.LGBMRegressor(**cfg['lightgbm']['fixed'], **candidate['parameters'])


from equity_research.models.preprocessing import require
