"""Independent arithmetic and artifact checks; never fits or evaluates performance."""
from equity_research.provenance import hash_matches
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from equity_research.data.ingest import write_json
from equity_research.models.final_fit import ROOT, OUT, PLAN, START, END, read_final_inputs, protected_hashes
from equity_research.models.common import sha, load_lightgbm
from equity_research.models.preprocessing import require, frame_digest

def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8'));features=plan['features']
    started=json.loads((OUT/'run_started.json').read_text(encoding='utf-8'))
    for p,h in {**started['input_hashes'],**started['protected_hashes']}.items():require(hash_matches(ROOT/p, h),'Input hash mismatch: '+p)
    frame,_,_=read_final_inputs(features)
    independent=(frame.signal_date.ge(START)&frame.signal_date.le(END)&frame.signal_eligible.eq(True)&
        frame.label_status.isin(['ok','ok_bridged'])&frame.label_end_date.gt(frame.signal_date)&frame.label_end_date.le(END)&np.isfinite(frame.y_excess_5d))
    fit=frame.loc[independent]
    require(frame_digest(frame)==started['filtered_candidate_digest'] and frame_digest(fit)==started['fit_digest'],'Input digest changed')
    saved=pd.read_parquet(OUT/'fit_rows_local.parquet')
    pd.testing.assert_frame_equal(saved,fit[saved.columns].reset_index(drop=True))
    ledger=json.loads((OUT/'fit_ledger.json').read_text(encoding='utf-8'))
    require(len(ledger)==4 and {c['id'] for c in ledger}=={c['id'] for c in plan['models']},'Four fixed fits required')
    # Recompute all eligible-date ranks with scipy, separately from pandas rank helper.
    ranked=pd.DataFrame(np.nan,index=frame.index,columns=features)
    for _,g in frame.groupby('signal_date'):
        for name in features:
            ix=g.index[g.signal_eligible&np.isfinite(g[name])]
            if len(ix):ranked.loc[ix,name]=(rankdata(frame.loc[ix,name],method='average')-.5)/len(ix)-.5
    reference=pd.read_parquet(OUT/'reload_reference_local.parquet')
    pd.testing.assert_frame_equal(reference[['PERMNO','signal_date']],fit[['PERMNO','signal_date']].reset_index(drop=True))
    checks=[]
    with threadpool_limits(limits=4):
        for rep in ['raw','cross_sectional_rank']:
            state=json.loads((OUT/'preprocessing'/f'{rep}.json').read_text(encoding='utf-8'))
            require(state['features']==features and state['fitted_rows']==len(fit),'Preprocessing schema/rows mismatch')
            a=(frame[features] if rep=='raw' else ranked).loc[independent].to_numpy(float)
            a[~np.isfinite(a)]=np.nan
            median=np.nanmedian(a,axis=0)
            if rep=='raw':lo,hi=np.nanquantile(a,plan['preprocessing']['clip_quantiles'],axis=0)
            else:lo=np.full(21,-.5);hi=np.full(21,.5)
            clean=np.where(np.isnan(a),median,np.clip(a,lo,hi))
            mean=np.mean(clean,axis=0);scale=np.std(clean,axis=0,ddof=0);scale[scale==0]=1
            for key,value in [('lower',lo),('upper',hi),('median',median),('mean',mean),('scale',scale)]:
                np.testing.assert_allclose(state[key],value,atol=1e-12,rtol=1e-12)
            # The saved state is the production transform; independently validate it above.
            X=pd.DataFrame((clean-np.asarray(state['mean']))/state['scale'],columns=features)
            for c in [c for c in plan['models'] if c['representation']==rep]:
                rec=next(v for v in ledger if v['id']==c['id'])
                require(rec['status']=='completed' and rec['parameters']==c['parameters'],'Fit ledger drift')
                require(rec['fit_rows']==len(fit) and rec['max_label_end']<=str(END.date()),'Fit boundary drift')
                require(sha(OUT/'preprocessing'/f'{rep}.json')==rec['preprocessing_sha256'],'Preprocessing hash mismatch')
                require(sha(ROOT/rec['model_path'])==rec['model_sha256'],'Model hash mismatch')
                for k,v in c['parameters'].items():require(rec['effective_parameters'][k]==v,'Effective parameter mismatch: '+k)
                model=joblib.load(ROOT/rec['model_path']) if c['model']=='ridge' else load_lightgbm(ROOT/rec['model_path'])
                if c['model']=='ridge':
                    require(model.alpha==c['parameters']['alpha'] and model.solver=='cholesky' and model.fit_intercept,'Ridge parameter drift')
                    require(model.feature_names_in_.tolist()==features,'Ridge feature order drift')
                else:
                    require(model.feature_name()==features and model.num_trees()==100,'LightGBM feature order/tree count drift')
                    for k,v in c['parameters'].items():
                        if k in model.params:require(model.params[k]==v,'Serialized LightGBM parameter drift: '+k)
                p=model.predict(X);expected=reference[c['id']].to_numpy()
                np.testing.assert_allclose(p,expected,atol=1e-11,rtol=1e-11)
                checks.append(dict(id=c['id'],rows=len(fit),independent_reload_max_error=float(np.max(np.abs(p-expected)))))
    baseline=json.loads((OUT/'baseline_state.json').read_text())
    require(baseline['momentum_missing']==float(fit.momentum_20.median()),'Momentum imputation mismatch')
    protected=protected_hashes()
    result=dict(status='passed',fits=4,additional_verification_fits=0,independent_preprocessing_recomputed=True,
        independent_rank_method='scipy average rank on eligible finite same-date candidates',checks=checks,
        verified_fit_rows=len(fit),max_fit_label_end=str(fit.label_end_date.max().date()),protected_hashes_verified=len(protected),
        test_rows_returned=0,test_predictions_generated=0,performance_evaluated=False)
    write_json(OUT/'verification.json',result)
    print(json.dumps(result,indent=2))
