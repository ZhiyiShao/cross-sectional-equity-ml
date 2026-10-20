"""Validation holdings exposure and reconciled PnL diagnostics; zero model fitting."""
from equity_research.provenance import hash_matches
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from equity_research.backtest.market import ROOT, MODELS, NAMES, require, sha

OUT=ROOT/'outputs/final_funding_exposures_v1'
BT=ROOT/'outputs/final_funding_backtest_v1'
FEATURES=['log_market_cap','volatility_20','market_beta_60']
GROUPS={1:'农作物',2:'畜牧',7:'农业服务',8:'林业',9:'渔业',10:'金属矿',12:'煤矿',13:'油气开采',14:'非金属矿',15:'建筑',16:'土木工程',17:'专业承包',20:'食品',21:'烟草',22:'纺织',23:'服装',24:'木材',25:'家具',26:'纸业',27:'印刷出版',28:'化工及相关产品（含医药）',29:'炼油及相关产品',30:'橡塑',31:'皮革',32:'非金属制品',33:'基础金属',34:'金属制品',35:'工业机械及计算机设备',36:'电子电气设备',37:'交通设备',38:'仪器及医疗器械',39:'其他制造',40:'铁路',41:'客运',42:'货运仓储',43:'邮政',44:'水运',45:'航空',46:'管道',47:'运输服务',48:'通信',49:'公用事业',50:'耐用品批发',51:'非耐用品批发',52:'建材零售',53:'综合零售',54:'食品零售',55:'汽车零售',56:'服装零售',57:'家居零售',58:'餐饮',59:'其他零售',60:'存款机构',61:'非存款信贷',62:'证券及商品服务',63:'保险承保',64:'保险代理',65:'房地产',67:'控股及其他投资',70:'酒店',72:'个人服务',73:'商业服务（含软件）',75:'汽车维修服务',76:'其他维修',78:'影视',79:'娱乐',80:'医疗服务',81:'法律',82:'教育',83:'社会服务',84:'博物馆',86:'会员组织',87:'工程会计研究管理服务',88:'私人家庭',89:'其他服务',91:'行政立法',92:'司法',93:'公共财政',94:'人力资源行政',95:'环境行政',96:'经济行政',97:'国家安全',99:'不可分类'}


def sic_group(code,start,end,date):
    if pd.isna(code) or pd.isna(start) or pd.isna(end) or not start<=date<=end:return 'Unknown'
    if not np.isfinite(code) or code!=int(code) or code<100 or code>9999:return 'Unknown'
    group=int(code)//100
    return f'{group:02d}' if group in GROUPS and group!=99 else 'Unknown'


def division(group):
    if group=='Unknown':return group
    g=int(group)
    for lo,hi,name in [(1,9,'A Agriculture'),(10,14,'B Mining'),(15,17,'C Construction'),(20,39,'D Manufacturing'),(40,49,'E Transport/utilities'),(50,51,'F Wholesale'),(52,59,'G Retail'),(60,67,'H Finance/real estate'),(70,89,'I Services'),(91,97,'J Public administration')]:
        if lo<=g<=hi:return name
    return 'Unknown'


def style_snapshots(f):
    f=f.copy()
    for col in FEATURES:
        x=f[col].where(f.signal_eligible & np.isfinite(f[col]))
        ranks=x.groupby(f.signal_date).rank(method='average')
        n=x.groupby(f.signal_date).transform('count')
        q=(ranks-.5)/n
        f[col+'_bucket']=np.select([q.le(.2),q.ge(.8),q.notna()],['Low20','High20','Middle60'],default='Unknown')
    return f


def attach_features(keys,features):
    # asof_date is strictly prior to the attribution date.
    left=keys.sort_values(['asof_date','PERMNO'])
    right=features.rename(columns={'signal_date':'feature_date'}).sort_values(['feature_date','PERMNO'])
    out=pd.merge_asof(left,right,on=None,left_on='asof_date',right_on='feature_date',by='PERMNO',direction='backward',allow_exact_matches=True)
    age=(out.asof_date-out.feature_date).dt.days
    invalid=age.isna()|age.gt(14)
    out.loc[invalid,FEATURES]=np.nan
    for col in FEATURES:out.loc[invalid,col+'_bucket']='Unknown'
    out['feature_age_days']=age
    require((out.feature_date.dropna()<=out.loc[out.feature_date.notna(),'asof_date']).all(),'Future feature match')
    return out


def load_inputs():
    paths=[BT/n for n in ['holdings_local.parquet','trades_local.parquet','daily_portfolios.csv','verification.json']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    manifest=json.loads((BT/'manifest.json').read_text(encoding='utf-8'))
    for p in paths:require(manifest[str(p.relative_to(ROOT))]==sha(p),'Backtest source hash changed')
    d=pd.read_csv(BT/'daily_portfolios.csv',parse_dates=['date'])
    d=d.loc[d.scenario.eq('lag_2')&d.cost_bps.eq(10)].copy().sort_values(['model','date'])
    require(d.date.between('2020-01-01','2021-12-31').all(),'Daily scope breach')
    d['prev_nav']=d.groupby('model').nav.shift().fillna(1.)
    dates=pd.DatetimeIndex(sorted(d.date.unique()))
    h=pd.read_parquet(paths[0],filters=[('date','>=',dates.min()),('date','<=',dates.max())])
    t=pd.read_parquet(paths[1],filters=[('date','>=',dates.min()),('date','<=',dates.max())])
    require(set(h.model)==set(MODELS),'Missing models')
    f=pd.read_parquet(ROOT/'data/processed/research_v1/features_only.parquet',columns=['PERMNO','signal_date','signal_eligible']+FEATURES,
        filters=[('signal_date','>=',dates.min()),('signal_date','<=',dates.max())])
    f=style_snapshots(f)
    ids=set(h.PERMNO)|set(t.PERMNO)
    source=ROOT/'data/interim/crsp_audit_v1';parts=json.loads((source/'ingest.json').read_text(encoding='utf-8'))['parts']
    data=[];partmeta=[]
    for name in parts:
        p=source/name;md=pq.read_metadata(p)
        lo=min(md.row_group(i).column(0).statistics.min for i in range(md.num_row_groups));hi=max(md.row_group(i).column(0).statistics.max for i in range(md.num_row_groups))
        wanted=sorted(i for i in ids if lo<=i<=hi)
        if wanted:
            x=pd.read_parquet(p,columns=['PERMNO','DlyCalDt','SICCD','SecInfoStartDt','SecInfoEndDt'],filters=[('PERMNO','in',wanted),('DlyCalDt','>=',dates.min()),('DlyCalDt','<=',dates.max())])
            data.append(x);partmeta.append(dict(name=name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    raw=pd.concat(data,ignore_index=True)
    require(raw.DlyCalDt.between(dates.min(),dates.max()).all() and f.signal_date.between(dates.min(),dates.max()).all(),'Source scope breach')
    require(not raw.duplicated(['PERMNO','DlyCalDt']).any(),'Duplicate industry source')
    valid=raw.DlyCalDt.ge(raw.SecInfoStartDt)&raw.DlyCalDt.le(raw.SecInfoEndDt)
    raw['industry']=[sic_group(c,s,e,dt) for c,s,e,dt in zip(raw.SICCD,raw.SecInfoStartDt,raw.SecInfoEndDt,raw.DlyCalDt)]
    raw['division']=raw.industry.map(division)
    audit=dict(raw_rows=len(raw),invalid_effective_date_rows=int((~valid).sum()),unknown_industry_rows=int(raw.industry.eq('Unknown').sum()),
        securities_with_sic_changes=int(raw.groupby('PERMNO').SICCD.nunique().gt(1).sum()),
        feature_rows=len(f),external_test_rows=0,new_model_fits=0,new_predictions=0,source_hashes=hashes,market_parts=partmeta)
    for folder in ['final_funding_backtest_v1','full_train_selection_v1','full_selection_exposures_v1']:
        manifest=json.loads((ROOT/'outputs'/folder/'manifest.json').read_text(encoding='utf-8'))
        for path,digest in manifest.items():require(hash_matches(ROOT/path, digest),'Protected artifact changed '+path)
        hashes.update(manifest)
    for name,frame in [('features',f),('industry',raw)]:audit[name+'_filtered_sha256']=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
    return d,h,t,f,raw,dates,audit


def security_ledger(d,h,t,dates,events,registry):
    key=['model','date','PERMNO']
    # Fixed cash claims and trade settlements do not generate investment PnL.
    stock=h.loc[h.kind.isin(['equity','inherited'])].groupby(key,as_index=False).value.sum().rename(columns={'value':'equity'})
    nxt=dict(zip(dates[:-1],dates[1:]))
    prev=stock.copy();prev['date']=prev.date.map(nxt);prev=prev.dropna(subset=['date']).rename(columns={'equity':'prev_equity'})
    p=stock.merge(prev,on=key,how='outer',validate='one_to_one')
    tr=t.pivot_table(index=key,columns='side',values='amount',aggfunc='sum',fill_value=0).reset_index()
    fees=t.groupby(key,as_index=False).cost.sum()
    adjustments=[]
    for e in events.itertuples():
        r=registry.loc[e.PERMNO];qty=e.old_value/r.previous_price
        stock_value=e.value-qty*(r.cash_per_share+r.special_per_share)
        require(stock_value>=-1e-10,'Negative converted stock value')
        adjustments.append(dict(model=e.model,date=e.date,PERMNO=e.PERMNO,conversion_adjustment=e.value))
        if r.ratio:
            adjustments.append(dict(model=e.model,date=e.date,PERMNO=int(r.successor_permno),conversion_adjustment=-stock_value))
    adj=pd.DataFrame(adjustments,columns=key+['conversion_adjustment']).groupby(key,as_index=False).conversion_adjustment.sum()
    p=p.merge(tr,on=key,how='outer',validate='one_to_one').merge(fees,on=key,how='outer',validate='one_to_one').merge(adj,on=key,how='outer',validate='one_to_one')
    for col in ['equity','prev_equity','buys','sells','cost','conversion_adjustment']:
        if col not in p:p[col]=0.
        p[col]=p[col].fillna(0.)
    p['pnl']=p.equity-p.prev_equity-p.buys+p.sells+p.conversion_adjustment
    p['net_pnl']=p.pnl-p.cost
    p=p.merge(d[['model','date','prev_nav']],on=['model','date'],validate='many_to_one')
    before=dict(zip(dates[1:],dates[:-1]));p['asof_date']=p.date.map(before)
    require(p.asof_date.notna().all(),'Unexpected asset or trade on initial anchor')
    p['equity_weight']=p.prev_equity/p.prev_nav
    p['receivable_weight']=0.;p['prev_receivable']=0.
    agg=p.groupby(['model','date'])[['pnl','cost','net_pnl']].sum().reindex(pd.MultiIndex.from_frame(d[['model','date']]),fill_value=0)
    np.testing.assert_allclose(agg.pnl,d.pnl,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(agg.cost,d.fees,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(agg.net_pnl,d.nav-d.prev_nav,atol=1e-10,rtol=1e-10)
    return p


def summarize(p,d):
    keys=['model','date'];rows=[]
    for (model,date),g in p.groupby(keys,sort=False):
        row=dict(model=model,date=date,equity_weight=g.equity_weight.sum(),receivable_weight=g.receivable_weight.sum(),
            industry_unknown_weight=g.loc[g.industry.eq('Unknown'),'equity_weight'].sum())
        for col in FEATURES:
            mask=np.isfinite(g[col]);covered=g.loc[mask,'equity_weight'].sum()
            row[col]=float((g.loc[mask,col]*g.loc[mask,'equity_weight']).sum()/covered) if covered>0 else np.nan
            row[col+'_coverage']=covered/row['equity_weight'] if row['equity_weight']>0 else np.nan
        covered=g.feature_age_days.notna()&g.feature_age_days.le(14)
        row['feature_age_days']=np.average(g.loc[covered,'feature_age_days'],weights=g.loc[covered,'equity_weight']) if g.loc[covered,'equity_weight'].sum()>0 else np.nan
        rows.append(row)
    exposure=pd.DataFrame(rows)
    # Keep first-entry day (all cash) and final sale day; omit only anchor.
    base=d.loc[d.date.gt(d.date.min()),['model','date','prev_nav','return_net','fees']].copy()
    daily=base.merge(exposure,on=keys,how='left',validate='one_to_one')
    for col in ['equity_weight','receivable_weight','industry_unknown_weight']:daily[col]=daily[col].fillna(0.)
    daily['cash_weight']=1-daily.equity_weight
    funding=d[['model','date','prev_nav']].copy()
    for col in ['cash','sale_receivable','payable','action_receivable','net_funding','inherited']:
        initial=1. if col in ['cash','net_funding'] else 0.
        funding[col+'_weight']=d.groupby('model')[col].shift().fillna(initial)/d.prev_nav
    daily=daily.rename(columns={'cash_weight':'non_equity_weight'}).merge(funding.drop(columns='prev_nav'),on=keys,validate='one_to_one')
    require(daily.non_equity_weight.ge(-1e-10).all(),'Invalid exposure balance')
    np.testing.assert_allclose(daily.equity_weight+daily.net_funding_weight+daily.action_receivable_weight,1.,atol=1e-10)
    daily['year']=daily.date.dt.year;daily['month']=daily.date.dt.strftime('%Y-%m')
    cols=FEATURES+[x+'_coverage' for x in FEATURES]+['equity_weight','receivable_weight','cash_weight','non_equity_weight','sale_receivable_weight','payable_weight','action_receivable_weight','net_funding_weight','inherited_weight','industry_unknown_weight','feature_age_days']
    yearly=daily.groupby(['model','year'])[cols].mean().reset_index();monthly=daily.groupby(['model','month'])[cols].mean().reset_index()
    for x in [yearly,monthly]:x['geometric_market_cap_bn']=np.exp(x.log_market_cap)/1e9
    for period,table in [('year',yearly),('month',monthly)]:
        eq=table.loc[table.model.eq('equal_weight'),[period]+FEATURES].rename(columns={x:x+'_pool' for x in FEATURES})
        merged=table.merge(eq,on=period,validate='many_to_one')
        for col in FEATURES:table[col+'_active']=merged[col]-merged[col+'_pool']
    return daily,yearly,monthly


def contribution_tables(p,d):
    p=p.copy();p['year']=p.date.dt.year
    annual=[]
    for (model,year),g in d.loc[d.date.gt(d.date.min())].groupby(['model',d.date.dt.year]):
        start=float(g.prev_nav.iloc[0]);end=float(g.nav.iloc[-1]);annual.append(dict(model=model,year=int(year),start_nav=start,net_return=end/start-1,fees_return=g.fees.sum()/start))
    annual=pd.DataFrame(annual)
    p=p.merge(annual[['model','year','start_nav']],on=['model','year'],validate='many_to_one');p['contribution']=p.net_pnl/p.start_nav;p['fee_contribution']=-p.cost/p.start_nav
    datecount=d.loc[d.date.gt(d.date.min())].assign(year=lambda x:x.date.dt.year).groupby(['model','year']).size().rename('days')
    result=[]
    for dim in ['industry','division']+[c+'_bucket' for c in FEATURES]:
        g=p.copy();g['category']=g[dim].fillna('Unknown')
        claim=g.prev_receivable.gt(0)&g.prev_equity.le(1e-12)&g.buys.le(1e-12)
        g.loc[claim,'category']='Receivable'
        g['weight']=g.equity_weight+g.receivable_weight
        group=g.groupby(['model','year','category']).agg(contribution=('contribution','sum'),fee_contribution=('fee_contribution','sum'),weight_sum=('weight','sum')).reset_index()
        group=group.merge(datecount.reset_index(),on=['model','year'],validate='many_to_one');group['average_weight']=group.weight_sum/group.days
        for (model,year),z in group.groupby(['model','year']):
            np.testing.assert_allclose(z.contribution.sum(),annual.loc[annual.model.eq(model)&annual.year.eq(year),'net_return'].iloc[0],atol=1e-10)
        categories=sorted(group.category.unique());full=pd.MultiIndex.from_product([MODELS,[2020,2021],categories],names=['model','year','category'])
        group=group.set_index(['model','year','category']).reindex(full).reset_index()
        for col in ['contribution','fee_contribution','average_weight']:group[col]=group[col].fillna(0.)
        eq=group.loc[group.model.eq('equal_weight'),['year','category','contribution','average_weight']].rename(columns={'contribution':'pool_contribution','average_weight':'pool_weight'})
        group=group.merge(eq,on=['year','category'],validate='many_to_one')
        group['active_contribution']=group.contribution-group.pool_contribution;group['active_weight']=group.average_weight-group.pool_weight;group['dimension']=dim
        result.append(group)
    return pd.concat(result,ignore_index=True),annual


def main():
    require(not OUT.exists() or not any(OUT.iterdir()),'Nonempty output; preserve existing analysis')
    d,h,t,f,raw,dates,audit=load_inputs()
    events=pd.read_parquet(BT/'events_local.parquet')
    events=events.loc[events.scenario.eq('lag_2')&events.cost_bps.eq(10)]
    registry=pd.read_csv(BT/'event_registry_local.csv').set_index('permno')
    p=security_ledger(d,h,t,dates,events,registry)
    keys=p[['asof_date','PERMNO']].drop_duplicates()
    joined=attach_features(keys,f)
    ind=raw.rename(columns={'DlyCalDt':'asof_date'})[['asof_date','PERMNO','industry','division']]
    joined=joined.merge(ind,on=['asof_date','PERMNO'],how='left',validate='one_to_one')
    p=p.merge(joined,on=['asof_date','PERMNO'],how='left',validate='many_to_one')
    p['industry']=p.industry.fillna('Unknown');p['division']=p.division.fillna('Unknown')
    daily,yearly,monthly=summarize(p,d);contributions,annual=contribution_tables(p,d)
    # Full month x industry contribution, both annual-start and monthly-start denominators.
    q=p.assign(year=p.date.dt.year,month=p.date.dt.strftime('%Y-%m'))
    q=q.merge(annual[['model','year','start_nav']],on=['model','year'],validate='many_to_one')
    q['annual_contribution']=q.net_pnl/q.start_nav
    q['annual_fee_contribution']=-q.cost/q.start_nav
    mon=q.groupby(['model','year','month','industry']).agg(annual_contribution=('annual_contribution','sum'),annual_fee_contribution=('annual_fee_contribution','sum')).reset_index()
    starts=d[d.date.gt(d.date.min())].assign(month=lambda x:x.date.dt.strftime('%Y-%m')).groupby(['model','month']).prev_nav.first().rename('month_start').reset_index()
    mon=mon.merge(starts,on=['model','month']).merge(annual[['model','year','start_nav']],on=['model','year'])
    mon['monthly_contribution']=mon.annual_contribution*mon.start_nav/mon.month_start
    for (model,year),g in mon.groupby(['model','year']):
        np.testing.assert_allclose(g.annual_contribution.sum(),annual.loc[annual.model.eq(model)&annual.year.eq(year),'net_return'].iloc[0],atol=1e-10)
    audit.update(security_ledger_rows=len(p),daily_reconciliations=len(d),annual_dimension_reconciliations=6*2*5,
        maximum_feature_age_days=float(p.loc[p.equity_weight.gt(0)&p.feature_age_days.le(14),'feature_age_days'].max()),
        minimum_feature_coverage=float(yearly[[x+'_coverage' for x in FEATURES]].min().min()),
        library_classification_note='SIC history effective-date alignment; annual vintage not unrevised historical publication')
    for path,digest in audit['source_hashes'].items():require(hash_matches(ROOT/path, digest),'Original backtest changed')
    OUT.mkdir(parents=True)
    for name,frame in [('daily_exposures',daily),('yearly_exposures',yearly),('monthly_exposures',monthly),('contributions',contributions),('annual_reconciliation',annual),('monthly_industry_contributions',mon)]:frame.to_csv(OUT/(name+'.csv'),index=False)
    p.to_parquet(OUT/'exposure_ledger_local.parquet',index=False)
    audit['code_sha256']=sha(Path(__file__));audit['protocol_sha256']=sha(ROOT/'docs/FINAL_FUNDING_EXPOSURE_PROTOCOL.md')
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(yearly[['model','year','geometric_market_cap_bn','volatility_20','market_beta_60','industry_unknown_weight']].to_string(index=False),flush=True)
    print('All daily PnL and annual contributions reconcile; zero fits and no test observations.',flush=True)
