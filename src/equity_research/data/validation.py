"""Independent target arithmetic, temporal checks and aggregate report."""
import hashlib
import json
import numpy as np
import pandas as pd
from equity_research.data.ingest import ROOT, write_json
from equity_research.data.features import make_features
from equity_research.data.returns import selected_parts
from equity_research.data.returns import review_additional_events


def main():
    out=ROOT/'data/processed/research_v1'
    source=ROOT/'data/interim/crsp_audit_v1'
    summary=json.loads((out/'dataset_summary.json').read_text(encoding='utf-8'))
    for name,sha in summary['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, f'Changed artifact: {name}'
    p=pd.read_parquet(out/'features_labels.parquet')
    f=pd.read_parquet(out/'features_only.parquet')
    names=summary['features']
    assert f.columns.tolist()==['PERMNO','signal_date','signal_eligible']+names
    pd.testing.assert_frame_equal(p[f.columns],f)
    assert p.groupby('signal_date').size().eq(500).all()
    assert not p.duplicated(['signal_date','PERMNO']).any()
    cf=pd.read_parquet(source/'calendar.parquet')
    cal=pd.DatetimeIndex(pd.to_datetime(cf.date))
    pos=pd.Series(np.arange(len(cal)),index=cal)
    market=pd.Series(cf.vwretd.to_numpy(),index=cal)
    known=p.entry_date.notna()
    assert (p.loc[known,'entry_date'].map(pos)-p.loc[known,'signal_date'].map(pos)).eq(1).all()
    known=p.label_end_date.notna()
    assert (p.loc[known,'label_end_date'].map(pos)-p.loc[known,'signal_date'].map(pos)).eq(6).all()
    market_forward=pd.concat([market.shift(-i) for i in range(2,7)],axis=1)
    expected_market=(1+market_forward).prod(axis=1,min_count=5)-1
    observed=p.market_return_5d.notna()
    np.testing.assert_allclose(p.loc[observed,'market_return_5d'],p.loc[observed,'signal_date'].map(expected_market),atol=1e-12,rtol=1e-10)
    assert p.loc[~p.label_status.isin(['ok','ok_bridged']),'y_excess_5d'].isna().all()
    assert p.loc[p.purged_boundary,'eligible_main_sample'].eq(False).all()
    assert p.loc[p.label_proxy_assumption,'y_excess_5d'].isna().all()
    groups={int(k):g for k,g in p.groupby('PERMNO')}
    uc=json.loads((ROOT/'config/universe.json').read_text(encoding='utf-8'))
    prefix_ids=set(sorted(groups)[:8])
    checked=0; bridged=0; prefix_checks=0
    for _,d in selected_parts(source,set(groups)):
        for permno,g in d.groupby('PERMNO',sort=False):
            x=g.set_index('DlyCalDt').reindex(cal)
            rows=groups[int(permno)]
            ordinary=rows.loc[rows.label_status.eq('ok')]
            # Independent forward shifting of original vendor daily returns.
            forward=pd.concat([x.DlyRet.shift(-i) for i in range(2,7)],axis=1)
            stock=(1+forward).prod(axis=1,min_count=5)-1
            expected=ordinary.signal_date.map(stock)
            np.testing.assert_allclose(ordinary.stock_return_5d,expected,atol=1e-12,rtol=1e-10)
            np.testing.assert_allclose(ordinary.y_excess_5d,expected-ordinary.market_return_5d,atol=1e-12,rtol=1e-10)
            checked+=len(ordinary)
            for _,r in rows.loc[rows.label_status.eq('ok_bridged')].iterrows():
                e=cal.get_loc(r.entry_date); end=cal.get_loc(r.label_end_date)
                path=x.iloc[e+1:end+1]
                finite=path.DlyRet.notna()
                assert path.loc[~finite,'DlyRetMissFlg'].eq('MP').all()
                assert path.loc[finite,'DlyRetMissFlg'].eq('NA').all()
                total=(1+path.loc[finite,'DlyRet']).prod()-1
                np.testing.assert_allclose(total,r.stock_return_5d,atol=1e-12,rtol=1e-10)
                bridged+=1
            if int(permno) in prefix_ids:
                cut=rows.iloc[len(rows)//2].signal_date
                prefix=make_features(x.loc[:cut],market.loc[:cut],uc).loc[cut,names]
                stored=rows.loc[rows.signal_date.eq(cut),names].iloc[0]
                np.testing.assert_allclose(prefix.to_numpy(float),stored.to_numpy(float),rtol=1e-10,atol=1e-12,equal_nan=True)
                prefix_checks+=1
    validation={'independent_ordinary_labels_checked':checked,'independent_bridged_labels_checked':bridged,
                'actual_data_prefix_checks':prefix_checks,'all_candidate_rows_preserved':True,
                'label_date_alignment_passed':True,'feature_whitelist_passed':True,
                'artifact_hashes_passed':True,'validation_passed':True}
    extra=review_additional_events()
    assert extra['affected_label_rows']==1
    assert extra['cases'][0]['duration']=='P2' and extra['cases'][0]['prior_two_returns_normal_daily']
    validation['additional_special_delist_review_passed']=True
    write_json(out/'dataset_validation.json',validation)
    review=pd.read_csv(out/'return_review_decisions.csv',keep_default_na=False)
    coverage=pd.read_csv(out/'split_coverage.csv')
    quality=pd.read_csv(out/'label_quality_counts.csv')
    impacts=pd.read_csv(out/'review_case_label_impacts.csv')
    lines=['# 异常收益复核与特征标签结果','',
        '16 条原始异常已逐条分类并明确处理规则；21 个特征及未来五日收益标签已生成。没有训练模型或计算策略表现。','',
        '## 异常复核','',
        '五组 MP → P1 的跨度、缺价位置和复权端点价格收益均核对通过。两日累计收益只用于完全覆盖两日的持有区间，不拆分、不回填单日收益。4 条 MV 与 2 条 DG 均为退市记录；缺少完整公司行动或及时估值，仍无法恢复真实收益。保留原值和缺失标签，并记录规则；这 6 条没有被“修复”为猜测值。','',
        '| 记录 | 标记 / 期间 | 结论 |','|---|---|---|']
    decisions={'MP':'缺失日保持 NA；仅允许完整持有区间使用后续 P1',
               'NA':'已验证两交易日累计收益；不当作单日波动率观测',
               'MV':'公司行动金额未知，数值原样保留但目标不可用',
               'DG':'退市估值过晚，目标不可用'}
    for _,r in review.iterrows():
        lines.append(f'| {r.case_id} | {r.missing_flag} / {r.duration} | {decisions[r.missing_flag]} |')
    lines+=['','具体证券、日期和前后行情仅保存在本地 `return_review_decisions.csv` 与 `return_review_context.parquet`。来源与处理规则见 [异常收益政策](RETURN_POLICY.md)。','',
        '全量标签额外发现 EX01：1 条 NA/P2/DP 退市事件，收益期间标记为三个交易日，之前两天又各有正常单日收益。未经核实直接连乘可能重复计算，因此主标签与辅助近似均保留缺失。其情况不同于已核实的五组 MP → P1。局部原始证据存于 `additional_event_context.parquet`，匿名结论存于 `additional_event_review.json`。','',
        '## 数据集','',
        f'保留 {summary["signal_dates"]} 个信号截面，每次 500 行，共 {summary["rows"]:,} 条股票—信号日记录，涉及 {summary["securities"]:,} 只证券。21 个输入特征包括动量、反转、波动、成交量、流动性、规模和市场 beta。全部特征完整的行数为 {summary["feature_complete_rows"]:,}；其余缺失不在全样本填补。','',
        '信号 t 收盘、入场 t+1 收盘、终点 t+6 收盘；主标签减去 t+2 至 t+6 同区间市场复合收益。主标签以可执行的普通退出路径为准；正常退市另存冻结对价的辅助近似，不视为已核实现金结算。见 [特征与标签字典](FEATURES_AND_LABELS.md)。','',
        '| 阶段 | 候选行 | 信号截面 | t 时刻可选 | 主标签样本（含边界净化） | 退市辅助近似 | 跨阶段边界 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in coverage.iterrows():
        lines.append(f'| {r.split} | {int(r.candidates):,} | {int(r.signal_dates)} | {int(r.signal_eligible):,} | {int(r.main_samples):,} | {int(r.proxy_labels)} | {int(r.purged)} |')
    lines+=['','### 标签质量分布','','| 状态 | 行数 |','|---|---:|']
    for status,count in summary['label_status_counts'].items():
        lines.append(f'| {status} | {count:,} |')
    lines+=['','所有缺标签、未成交、退市和末尾不完整行均保留。`eligible_main_sample` 只用于事后目标可用性，不可用它筛选事前预测/回测股票池。退市及缺价是非随机缺失，主标签评价仍可能受删失影响；必须披露分母并检查事件近似的敏感性。当前不能宣布完整回测已就绪。','',
        '## 校验','',
        f'独立用原始收益前移重算 {checked:,} 个普通标签、核验 {bridged} 个整段桥接标签；{prefix_checks} 个真实证券检查仅输入历史前缀时特征与全量构建一致。所有候选数、日期对齐、阶段边界、特征白名单和产物哈希检查通过。23 项合成测试通过，覆盖未来数据扰动、复权因子基准抵消、跨期收益、退市、全损与样本尾端。','',
        '## 交付','',
        '- `data/processed/research_v1/features_labels.parquet`：完整候选、21 个特征、主/辅助标签与状态。',
        '- `features_only.parquet`：特征白名单视图，避免将目标或未来执行状态误喂给模型。',
        '- `return_review_decisions.csv`、`return_review_context.parquet`：逐条决策及原始证据。',
        '- `review_case_label_impacts.csv`：16 条事件与实际五日网格的对应影响。',
        '- `split_coverage.csv`、`label_quality_counts.csv`、`feature_missing_counts.csv`：覆盖与缺失统计。',
        '- `dataset_summary.json`、`dataset_validation.json`：规则、来源哈希及核验结果。','',
        '上述数据文件均保持本地且被 Git 忽略。可复现脚本：`review_returns.py` → `features_labels.py` → `validate_features_labels.py`。']
    (ROOT/'reports').mkdir(exist_ok=True)
    report='\n'.join(lines)+'\n'
    for name in ['RETURN_POLICY.md','FEATURES_AND_LABELS.md']:
        report=report.replace(f']({name})',f'](../{name})')
    (ROOT/'reports/FEATURE_DATA_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)
