"""Development data boundaries, reusable preprocessing, estimators and metrics."""
import itertools
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from equity_research.provenance import sha256 as sha
# Compatibility exports: preprocessing calculations live in one module.
from equity_research.models.preprocessing import fit_preprocessing, transform_features


def load_development_data(path, features, cfg):
    columns=['PERMNO','signal_date','split','label_end_date','signal_eligible',
             'eligible_main_sample','purged_boundary','label_status',
             'y_excess_5d','y_excess_5d_event_proxy','label_proxy_assumption']+features
    # Arrow row filtering happens before the returned frame exists. No test
    # prediction or target is passed to the fitting/evaluation functions.
    frame=pd.read_parquet(path,columns=columns,engine='pyarrow',filters=[
        ('signal_date','>=',pd.Timestamp(cfg['train_dates'][0])),
        ('signal_date','<',pd.Timestamp(cfg['test_start_exclusive'])),
        ('split','in',['train','validation'])])
    assert len(frame) and frame.signal_date.max()<pd.Timestamp(cfg['test_start_exclusive'])
    assert set(frame.split)=={'train','validation'}
    assert not frame.duplicated(['PERMNO','signal_date']).any()
    return frame.sort_values(['signal_date','PERMNO']).reset_index(drop=True)


def sample_masks(frame,cfg):
    masks={}
    for split in ['train','validation']:
        start,end=map(pd.Timestamp,cfg[split+'_dates'])
        in_split=frame.split.eq(split)
        assert frame.loc[in_split,'signal_date'].between(start,end).all()
        masks[split]=(in_split & frame.signal_eligible & ~frame.purged_boundary
            & frame.label_end_date.le(end) & frame.label_end_date.gt(frame.signal_date)
            & frame.label_status.isin(['ok','ok_bridged']) & frame.y_excess_5d.notna())
    expected=masks['train']|masks['validation']
    assert expected.equals(frame.eligible_main_sample.astype(bool))
    return masks




def daily_rank_ic(dates,pred,target):
    """Cross-sectional Spearman with average ties; constant ranks stay NaN."""
    a=pd.DataFrame({'date':np.asarray(dates),'p':np.asarray(pred,float),'y':np.asarray(target,float)})
    a=a.loc[np.isfinite(a.p)&np.isfinite(a.y)]
    ranks=a.groupby('date')[['p','y']].rank(method='average')
    centered=ranks-ranks.groupby(a.date).transform('mean')
    cross=(centered.p*centered.y).groupby(a.date).sum()
    denom=np.sqrt(centered.p.pow(2).groupby(a.date).sum()*centered.y.pow(2).groupby(a.date).sum())
    return (cross/denom.where(denom.gt(0))).rename('rank_ic').sort_index()


def circular_block_interval(values,block,replications,seed):
    values=np.asarray(values,float)
    if not np.isfinite(values).all() or len(values)<block:
        raise ValueError('Bootstrap requires finite paired ICs and sufficient dates')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(values),size=(replications,int(np.ceil(len(values)/block))))
    indexes=(starts[:,:,None]+np.arange(block))%len(values)
    draws=values[indexes.reshape(replications,-1)[:,:len(values)]].mean(axis=1)
    low,high=np.quantile(draws,[.025,.975])
    return {'block_signals':block,'replications':replications,'paired_dates':len(values),
            'mean_difference':float(values.mean()),'ci_low':float(low),'ci_high':float(high)}


def model_grid(cfg,name):
    if name=='ridge':
        c=cfg[name]
        return [{'alpha':float(a)} for a in np.logspace(c['alpha_log10_start'],c['alpha_log10_end'],c['n_trials'])]
    c=cfg[name]
    return [dict(zip(['num_leaves','n_estimators','min_child_samples'],values))
            for values in itertools.product(c['num_leaves'],c['n_estimators'],c['min_child_samples'])]


def new_model(cfg,name,params):
    if name=='ridge':
        return Ridge(**params,solver=cfg['ridge']['solver'],fit_intercept=True)
    return lgb.LGBMRegressor(**cfg['lightgbm']['fixed'],**params)


def save_lightgbm(model,path):
    # Native C++ file IO on Windows may reject Unicode workspace paths.
    Path(path).write_text(model.booster_.model_to_string(),encoding='utf-8')


def load_lightgbm(path):
    return lgb.Booster(model_str=Path(path).read_text(encoding='utf-8'))
