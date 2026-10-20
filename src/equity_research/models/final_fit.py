"""Exactly four frozen final fits. No test observations, selection or evaluation."""
from equity_research.provenance import hash_matches
import json
import platform
from datetime import datetime, timezone
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from equity_research.data.ingest import ROOT, write_json
from equity_research.models.common import sha, transform_features, save_lightgbm, load_lightgbm
from equity_research.models.preprocessing import require, frame_digest, cross_sectional_rank, fit_state

OUT=ROOT/'outputs/final_models_v1'
SOURCE=ROOT/'data/processed/research_v1/features_labels.parquet'
PLAN=ROOT/'config/final_test_plan.json'
START=pd.Timestamp('2015-01-01')
END=pd.Timestamp('2021-12-31')

def eligibility(frame):
    return (frame.signal_date.between(START,END)&frame.signal_eligible&
            frame.label_status.isin(['ok','ok_bridged'])&
            frame.label_end_date.le(END)&frame.label_end_date.gt(frame.signal_date))

def read_final_inputs(features):
    # Read feature candidates first, including invalid-label candidates for ranks.
    # Never request values of labels whose horizon enters the sealed period.
    filters=[('signal_date','>=',START),('signal_date','<=',END),('split','in',['train','validation'])]
    cols=['PERMNO','signal_date','split','signal_eligible','label_status','label_end_date',
          'eligible_refit_through_2021','eligible_main_sample','purged_boundary']+features
    frame=pd.read_parquet(SOURCE,columns=cols,filters=filters)
    require(len(frame)>0 and frame.signal_date.between(START,END).all(),'Candidate dates outside development')
    require(frame.loc[frame.split.eq('train'),'signal_date'].le('2019-12-31').all(),'Original training split mismatch')
    require(frame.loc[frame.split.eq('validation'),'signal_date'].ge('2020-01-01').all(),'Original validation split mismatch')
    require(not frame.duplicated(['PERMNO','signal_date']).any(),'Duplicate candidate keys')
    frame=frame.sort_values(['signal_date','PERMNO']).reset_index(drop=True)
    mask=eligibility(frame)
    require(mask.equals(frame.eligible_refit_through_2021.astype(bool)),'Stored final-fit eligibility disagrees')
    target_filters=filters+[('label_end_date','<=',END),('signal_eligible','==',True),('label_status','in',['ok','ok_bridged'])]
    y=pd.read_parquet(SOURCE,columns=['PERMNO','signal_date','label_end_date','y_excess_5d'],filters=target_filters)
    require(y.signal_date.between(START,END).all() and y.label_end_date.le(END).all(),'Target date breach')
    require(y.label_end_date.gt(y.signal_date).all() and np.isfinite(y.y_excess_5d).all(),'Invalid strict label')
    keys=['PERMNO','signal_date','label_end_date']
    target=frame.loc[mask,keys].merge(y,on=keys,how='outer',validate='one_to_one',indicator=True)
    require(target['_merge'].eq('both').all(),'Target and fit row mismatch')
    frame=frame.merge(y,on=keys,how='left',validate='one_to_one',sort=False)
    require(frame.loc[~mask,'y_excess_5d'].isna().all(),'Excluded label loaded')
    access=[dict(source=str(SOURCE.relative_to(ROOT)),columns=cols,filters=repr(filters),rows=len(frame),
                 min_signal=str(frame.signal_date.min()),max_signal=str(frame.signal_date.max()),kind='features_and_boundary_metadata'),
            dict(source=str(SOURCE.relative_to(ROOT)),columns=list(y.columns),filters=repr(target_filters),rows=len(y),
                 min_signal=str(y.signal_date.min()),max_signal=str(y.signal_date.max()),max_label_end=str(y.label_end_date.max()),kind='strict_fit_targets')]
    return frame,mask,access

def protected_hashes():
    result={}
    for folder in ['final_test_plan_v1','test_execution_preflight_v1']:
        result.update(json.loads((ROOT/'outputs'/folder/'manifest.json').read_text(encoding='utf-8')))
    result.update(json.loads((ROOT/'outputs/final_test_plan_v1/readiness.json').read_text(encoding='utf-8'))['protected_hashes'])
    for p,h in result.items():require(hash_matches(ROOT/p, h),'Frozen artifact changed: '+p)
    return result

def main():
    require(not OUT.exists(),'Final fit directory already exists; no implicit refit/resume')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'config/full_train_selection.json').read_text(encoding='utf-8'))
    features=plan['features'];candidates=plan['models']
    require(plan['final_fit_dates']==['2015-01-01','2021-12-31'] and plan['final_fit_budget']==4 and len(candidates)==4,'Plan drift')
    require(features==json.loads((ROOT/'config/features_labels.json').read_text(encoding='utf-8'))['features'],'Feature drift')
    require(not plan['test_data_access_authorized'] and not plan['test_performance_authorized'],'Test must stay sealed')
    require(prior['raw_clip_quantiles']==plan['preprocessing']['clip_quantiles'] and prior['rank_clip_bounds']==[-.5,.5],'Preprocessing drift')
    protected=protected_hashes()
    frame,mask,access=read_final_inputs(features)
    training=frame.loc[mask];require(len(training)>0,'Empty final training')
    rank=cross_sectional_rank(frame,features)
    restored=mask&~frame.eligible_main_sample
    require(frame.loc[restored,'split'].eq('train').all() and frame.loc[restored,'purged_boundary'].all(),'Unexpected restored rows')
    OUT.mkdir();(OUT/'models').mkdir();(OUT/'preprocessing').mkdir()
    inputs=[PLAN,ROOT/'config/full_train_selection.json',ROOT/'config/features_labels.json',
            ROOT/'src/equity_research/models/final_fit.py',ROOT/'src/equity_research/models/verification.py',ROOT/'src/equity_research/models/preprocessing.py',ROOT/'src/equity_research/models/common.py']
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    boundary=dict(candidates=len(frame),fit_rows=int(mask.sum()),signal_start=str(training.signal_date.min().date()),
        signal_end=str(training.signal_date.max().date()),max_label_end=str(training.label_end_date.max().date()),
        restored_internal_boundary_rows=int(restored.sum()),excluded_cross_2022_rows=int(frame.label_end_date.gt(END).sum()),
        original_train_fit_rows=int((mask&frame.split.eq('train')).sum()),original_validation_fit_rows=int((mask&frame.split.eq('validation')).sum()))
    write_json(OUT/'run_started.json',dict(started_at=datetime.now(timezone.utc).isoformat(),authorization='User authorized four frozen final fits only',
        plan=plan,boundary=boundary,filtered_candidate_digest=frame_digest(frame),fit_digest=frame_digest(training),input_hashes=input_hashes,
        protected_hashes=protected,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lgb.__version__)))
    write_json(OUT/'observation_access_log.json',access)
    training[['PERMNO','signal_date','label_end_date','label_status','split']].to_parquet(OUT/'fit_rows_local.parquet',index=False)
    annual=training.groupby(training.signal_date.dt.year).agg(rows=('PERMNO','size'),signal_dates=('signal_date','nunique'),max_label_end=('label_end_date','max'))
    annual.to_csv(OUT/'annual_fit_coverage.csv')
    write_json(OUT/'baseline_state.json',dict(momentum_direction='positive',momentum_missing=float(training.momentum_20.replace([np.inf,-np.inf],np.nan).median()),fit_rows=len(training)))
    ledger=[];checks=[];reload_frames=[]
    print('Final-fit boundary:',boundary,flush=True)
    with threadpool_limits(limits=4):
        for rep in ['raw','cross_sectional_rank']:
            x=(frame[features] if rep=='raw' else rank).loc[mask]
            state=fit_state(x,rep,prior)
            state.update(representation=rep,fit_signal_start=boundary['signal_start'],fit_signal_end=boundary['signal_end'],max_fit_label_end=boundary['max_label_end'])
            write_json(OUT/'preprocessing'/f'{rep}.json',state)
            X=transform_features(x,state)
            for c in [c for c in candidates if c['representation']==rep]:
                require(len(ledger)<4,'Fit budget exhausted')
                model=Ridge(**c['parameters'],fit_intercept=True) if c['model']=='ridge' else lgb.LGBMRegressor(**c['parameters'])
                record=dict(id=c['id'],parameters=c['parameters'],status='fit_started',fit_rows=len(X),representation=rep,
                    signal_start=boundary['signal_start'],signal_end=boundary['signal_end'],max_label_end=boundary['max_label_end'],
                    preprocessing_sha256=sha(OUT/'preprocessing'/f'{rep}.json'),transformed_fit_digest=frame_digest(X))
                ledger.append(record);write_json(OUT/'fit_ledger.json',ledger)
                model.fit(X,training.y_excess_5d)
                path=OUT/'models'/(c['id']+('.joblib' if c['model']=='ridge' else '.txt'))
                if c['model']=='ridge':joblib.dump(model,path);loaded=joblib.load(path)
                else:save_lightgbm(model,path);loaded=load_lightgbm(path)
                p=model.predict(X);q=loaded.predict(X)
                np.testing.assert_allclose(p,q,atol=1e-12,rtol=1e-12)
                require(np.isfinite(q).all(),'Nonfinite round-trip prediction')
                check=training[['PERMNO','signal_date']].copy();check[c['id']]=q
                reload_frames.append(check.set_index(['PERMNO','signal_date']))
                record.update(status='completed',model_sha256=sha(path),model_path=str(path.relative_to(ROOT)),
                    effective_parameters=model.get_params(),trees=None if c['model']=='ridge' else model.booster_.num_trees())
                write_json(OUT/'fit_ledger.json',ledger)
                checks.append(dict(id=c['id'],rows=len(X),max_serialization_prediction_error=float(np.max(np.abs(p-q)))))
                print('Completed final fit',len(ledger),'/4:',c['id'],flush=True)
    pd.concat(reload_frames,axis=1).reset_index().to_parquet(OUT/'reload_reference_local.parquet',index=False)
    for p,h in {**protected,**input_hashes}.items():require(hash_matches(ROOT/p, h),'Protected input changed: '+p)
    write_json(OUT/'fit_summary.json',dict(status='four_fits_completed_pending_independent_verification',fits=len(ledger),boundary=boundary,
        roundtrip_checks=checks,preprocessing_fits=2,test_rows_returned=0,test_predictions_generated=0,performance_evaluated=False,
        protected_hashes_verified=len(protected),completed_at=datetime.now(timezone.utc).isoformat()))
