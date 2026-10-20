"""Review the original 16 exceptions locally; print only quality diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from equity_research.data.ingest import ROOT, write_json


def selected_parts(source, ids, columns=None):
    manifest=json.loads((source/'ingest.json').read_text(encoding='utf-8'))
    for name in manifest['parts']:
        path=source/name
        meta=pq.read_metadata(path)
        # PERMNO is the first column in the verified original schema.
        lo=min(meta.row_group(i).column(0).statistics.min for i in range(meta.num_row_groups))
        hi=max(meta.row_group(i).column(0).statistics.max for i in range(meta.num_row_groups))
        wanted=[int(x) for x in ids if lo<=x<=hi]
        if wanted:
            yield name,pd.read_parquet(path,columns=columns,filters=[('PERMNO','in',wanted)])


def main():
    source=ROOT/'data/interim/crsp_audit_v1'
    out=ROOT/'data/processed/research_v1'
    out.mkdir(parents=True,exist_ok=True)
    q=pd.read_csv(ROOT/'data/processed/crsp_v1/return_review_queue.csv',keep_default_na=False)
    q['DlyCalDt']=pd.to_datetime(q.DlyCalDt)
    q=q.sort_values(['PERMNO','DlyCalDt']).reset_index(drop=True)
    assert len(q)==16
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(source/'calendar.parquet').date))
    d=pd.concat([x for _,x in selected_parts(source,set(q.PERMNO))],ignore_index=True)
    details=[]; evidence=[]
    for i,row in q.iterrows():
        cid=f'R{i+1:02d}'
        g=d.loc[d.PERMNO.eq(row.PERMNO)].set_index('DlyCalDt').reindex(cal)
        j=cal.get_loc(row.DlyCalDt)
        r=g.iloc[j]
        valid_price=g.DlyPrc.gt(0)&g.DlyPrcFlg.isin(['TR','BA'])&g.DlyDelFlg.eq('N')
        prior=np.flatnonzero(valid_price.to_numpy()[:j])
        following=np.flatnonzero(valid_price.to_numpy()[j+1:])+j+1
        previous=int(prior[-1]) if len(prior) else None
        next_good=int(following[0]) if len(following) else None
        duration=str(r.DlyRetDurFlg)
        span=int(duration[1:])+1 if duration.startswith('P') else 1
        bridged=False
        price_return_check=None
        if duration.startswith('P') and previous is not None:
            between=g.iloc[previous+1:j]
            bridged=(j-previous==span and between.DlyRet.isna().all() and between.DlyRetMissFlg.eq('MP').all())
            old=g.iloc[previous]
            if r.DlyCumFacPr>0 and old.DlyCumFacPr>0:
                expected=r.DlyPrc/old.DlyPrc*old.DlyCumFacPr/r.DlyCumFacPr-1
                price_return_check=bool(np.isclose(expected,r.DlyRetx,atol=1e-5,rtol=0))
        rec={'case_id':cid,'PERMNO':int(row.PERMNO),'event_date':str(row.DlyCalDt.date()),
             'delist':r.DlyDelFlg,'missing_flag':r.DlyRetMissFlg,'duration':duration,
             'price_flag':r.DlyPrcFlg,'numeric_return_present':bool(pd.notna(r.DlyRet)),
             'previous_valid_price_distance':None if previous is None else j-previous,
             'next_valid_price_distance':None if next_good is None else next_good-j,
             'next_return_duration':None if next_good is None else g.iloc[next_good].DlyRetDurFlg,
             'multi_period_covers_only_MP':bool(bridged),
             'adjusted_endpoint_price_matches_retx':price_return_check}
        if r.DlyRetMissFlg=='MP':
            rec['decision']='missing_daily_return_keep_NA; use_later_vendor_span_only_if_entire_span_inside_label'
        elif duration.startswith('P') and r.DlyRetMissFlg=='NA':
            rec['decision']='verified_whole_span_return; not_a_daily_volatility_observation' if bridged else 'unverified_span_quarantine'
        elif r.DlyRetMissFlg=='MV':
            rec['decision']='unresolved_corporate_action_value; keep_numeric_source_but_label_NA'
        elif r.DlyRetMissFlg=='DG':
            rec['decision']='unresolved_late_delisting_valuation; label_NA_not_zero'
        else:
            rec['decision']='manual_review_required'
        details.append(rec)
        ev=g.iloc[max(0,j-12):min(len(cal),j+13)].reset_index(names='date')
        ev['case_id']=cid
        evidence.append(ev)
    result=pd.DataFrame(details)
    result.to_csv(out/'return_review_decisions.csv',index=False)
    pd.concat(evidence,ignore_index=True).to_parquet(out/'return_review_context.parquet',index=False)
    write_json(out/'return_review_summary.json',{'reviewed_records':len(q),'securities':int(q.PERMNO.nunique()),
        'decision_counts':result.decision.value_counts().to_dict(),
        'raw_returns_rewritten':0,'all_multi_period_cases_verified':bool(result.loc[result.duration.str.startswith('P'),'multi_period_covers_only_MP'].all())})
    print(result.drop(columns=['PERMNO','event_date']).to_json(orient='records',force_ascii=False,indent=2),flush=True)


def review_additional_events():
    source=ROOT/'data/interim/crsp_audit_v1'
    out=ROOT/'data/processed/research_v1'
    p=pd.read_parquet(out/'features_labels.parquet')
    affected=p.loc[p.label_status.eq('unresolved_delist_NA')]
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(source/'calendar.parquet').date))
    contexts=[]; cases=[]
    for _,d in selected_parts(source,set(affected.PERMNO)):
        for _,row in affected.loc[affected.PERMNO.isin(d.PERMNO)].iterrows():
            x=d.loc[d.PERMNO.eq(row.PERMNO)].set_index('DlyCalDt').reindex(cal)
            window=x.loc[row.entry_date:row.label_end_date]
            for date,event in window.loc[window.DlyDelFlg.eq('Y')].iterrows():
                j=cal.get_loc(date)
                case_id=f'EX{len(cases)+1:02d}'
                context=x.iloc[max(0,j-4):j+2].copy()
                context.insert(0,'observation_date',context.index)
                context.insert(0,'case_id',case_id)
                contexts.append(context.reset_index(drop=True))
                cases.append({'case_id':case_id,'duration':event.DlyRetDurFlg,
                    'missing_flag':event.DlyRetMissFlg,'price_flag':event.DlyPrcFlg,
                    'return_present':bool(pd.notna(event.DlyRet)),
                    'prior_two_returns_normal_daily':bool(x.iloc[j-2:j].DlyRetMissFlg.eq('NA').all()
                        and x.iloc[j-2:j].DlyRetDurFlg.eq('D1').all()
                        and x.iloc[j-2:j].DlyRet.notna().all()),
                    'decision':'Keep main and proxy targets missing; special delist span not verified.'})
    if contexts:
        pd.concat(contexts,ignore_index=True).to_parquet(out/'additional_event_context.parquet',index=False)
    affected.to_parquet(out/'additional_event_label.parquet',index=False)
    result={'affected_label_rows':len(affected),'cases':cases}
    write_json(out/'additional_event_review.json',result)
    return result
