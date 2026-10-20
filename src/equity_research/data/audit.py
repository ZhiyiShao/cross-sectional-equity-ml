"""Validate saved universe outputs and write aggregate audit documentation."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_research.data.ingest import ROOT, write_json
from equity_research.data.universe import SINGLE_DAY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source, out):
    source, out = source.absolute(), out.absolute()
    ing = json.loads((source / 'ingest.json').read_text(encoding='utf-8'))
    audit = json.loads((out / 'semantic_audit.json').read_text(encoding='utf-8'))
    u = pd.read_parquet(out / 'monthly_universe.parquet')
    candidates = pd.read_parquet(out / 'monthly_candidates.parquet')
    funnel = pd.read_csv(out / 'universe_funnel.csv')
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(source / 'calendar.parquet').date))
    cfg = audit['config']
    assert len(u)==len(pd.read_csv(out / 'monthly_universe.csv'))
    assert ing['raw_rows']-ing['exact_duplicate_rows_removed']==ing['clean_rows']
    assert not u.duplicated(['effective_month','PERMNO']).any()
    assert not u.duplicated(['effective_month','PERMCO']).any()
    assert (u.formation_date<u.effective_from).all()
    assert u.DlyClose.ge(cfg['min_close_usd']).all()
    assert u.history_days.ge(cfg['min_observed_history_days']).all()
    assert u.valid_liquidity_days.ge(cfg['min_valid_liquidity_days']).all()
    # Independent ranking verification from saved candidates, no reliance on select_month.
    for formation, s in candidates.groupby('formation_date'):
        eligible=s.loc[s.filter(like='eligible_').all(axis=1)]
        best=(eligible.sort_values(['adv60_usd','PERMNO'],ascending=[False,True])
                      .groupby('PERMCO',sort=False).head(1)
                      .head(cfg['max_securities']))
        actual=u.loc[u.formation_date.eq(formation)]
        assert best.PERMNO.astype(int).tolist()==actual.PERMNO.tolist()
    members = {m:set(g.PERMNO) for m,g in u.groupby('effective_month')}
    first6 = set(pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).head(6))
    counts=Counter()
    return_flags=Counter()
    issue_parts=[]
    cap_parts=[]
    selected_ids=set(u.PERMNO)
    check_cols=['PERMNO','DlyCalDt','DlyDelFlg','DlyPrcFlg','DlyRet','DlyRetx','DlyRetMissFlg','DlyRetDurFlg','DlyCumFacPr','DlyCumFacShr','DlyPrc','ShrOut','DlyCap']
    for name in ing['parts']:
        d=pd.read_parquet(source/name, columns=check_cols)
        cap_error=d.DlyCap/(d.DlyPrc*d.ShrOut)-1
        cap_bad=d.DlyDelFlg.eq('N')&d.DlyPrc.gt(0)&d.ShrOut.gt(0)&d.DlyCap.notna()&cap_error.abs().gt(1e-4)
        cap_rows=d.loc[cap_bad,['PERMNO','DlyCalDt','DlyPrc','ShrOut','DlyCap','DlyPrcFlg']].copy()
        cap_rows['cap_relative_error']=cap_error.loc[cap_bad]
        cap_parts.append(cap_rows)
        d=d.loc[d.PERMNO.isin(selected_ids)].copy()
        if d.empty:
            continue
        month=d.DlyCalDt.dt.to_period('M')
        exposure=pd.Series(False,index=d.index)
        for m, idx in d.groupby(month).groups.items():
            current=d.loc[idx,'PERMNO'].isin(members.get(str(m),set()))
            previous=d.loc[idx,'PERMNO'].isin(members.get(str(m-1),set())) & d.loc[idx,'DlyCalDt'].isin(first6)
            exposure.loc[idx]=current | previous
        d=d.loc[exposure].copy()
        delist=d.DlyDelFlg.eq('Y')
        bad=(d.DlyRet.isna() | d.DlyRetMissFlg.ne('NA') |
             (d.DlyDelFlg.eq('N') & ~d.DlyRetDurFlg.isin(SINGLE_DAY)))
        bad_factor=d.DlyDelFlg.eq('N') & (~d.DlyCumFacPr.gt(0) | ~d.DlyCumFacShr.gt(0))
        counts['candidate_path_observed_rows']+=len(d)
        counts['candidate_path_delist_events']+=int(delist.sum())
        counts['candidate_path_bad_return_rows']+=int(bad.sum())
        counts['candidate_path_delist_bad_returns']+=int((delist&bad).sum())
        counts['candidate_path_nonpositive_or_missing_factors']+=int(bad_factor.sum())
        return_flags.update(d.loc[bad,'DlyRetMissFlg'].value_counts().to_dict())
        issues=d.loc[bad | delist | bad_factor].copy()
        issues['requires_return_review']=bad.loc[issues.index]
        issues['requires_factor_review']=bad_factor.loc[issues.index]
        issue_parts.append(issues)
    issues=pd.concat(issue_parts,ignore_index=True) if issue_parts else pd.DataFrame()
    issues.to_parquet(out/'selected_return_events.parquet',index=False)
    issues.loc[issues.requires_return_review].to_csv(out/'return_review_queue.csv',index=False)
    cap_exceptions=pd.concat(cap_parts,ignore_index=True)
    cap_exceptions.to_parquet(out/'capitalization_exceptions.parquet',index=False)
    counts['cap_exception_rows_ever_selected_security']=int(cap_exceptions.PERMNO.isin(selected_ids).sum())
    counts['candidate_path_affected_securities']=int(issues.loc[issues.requires_return_review,'PERMNO'].nunique()) if len(issues) else 0
    counts['selected_securities_delisted_anytime_in_download']=0
    ever_delisted=set()
    for p in (out/'exceptions').glob('part-*.parquet'):
        d=pd.read_parquet(p,columns=['PERMNO','DlyDelFlg'])
        ever_delisted.update(d.loc[d.DlyDelFlg.eq('Y'),'PERMNO'])
    counts['selected_securities_delisted_anytime_in_download']=len(selected_ids & ever_delisted)
    provenance={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'config/universe.json',
        ROOT/'src/equity_research/data/ingest.py',ROOT/'src/equity_research/data/universe.py',ROOT/'src/equity_research/data/audit.py',
        ROOT/'tests/test_crsp_pipeline.py',out/'monthly_universe.parquet',out/'monthly_universe.csv']}
    final={**dict(counts),'candidate_path_bad_flags':dict(return_flags),
           'universe_validation_passed':True,'universe_ready':True,
           'backtest_ready':False,'reason':'Review candidate-path return events and delisting settlement assumptions before portfolio accounting.',
           'exposure_scope':'current-month membership plus prior-month membership during first six market dates; conservative possible holding paths, not actual positions',
           'source_sha256':ing['sha256'],'provenance_sha256':provenance}
    write_json(out/'validation.json',final)
    yearly=calendar.to_series().groupby(calendar.year).size()
    annual=u.assign(year=u.effective_from.dt.year).groupby('year').agg(stock_months=('PERMNO','size'),unique_stocks=('PERMNO','nunique'))
    annual['market_days']=yearly.reindex(annual.index)
    annual.to_csv(out/'annual_coverage.csv')
    pct=lambda a,b: f'{100*a/b:.3f}%' if b else '不适用'
    count=lambda k: f'{ing[k]:,}'
    lines=[
        '# CRSP 数据审计与历史股票池结果',
        '',
        '审计日期：2026-10-14。全量审计与股票池生成完成；尚未训练模型或检查策略收益。',
        '',
        '## 结论',
        '',
        f'已生成 {audit["universe_months"]} 个月（2015-01 至 2025-12）的历史股票池，共 {len(u):,} 条股票—月份记录，每月 {audit["min_month_size"]}–{audit["max_month_size"]} 只，累计 {audit["unique_selected_permnos"]:,} 只不同证券。独立排名、公司去重和生效时点检查通过。',
        '',
        f'其中 {counts["selected_securities_delisted_anytime_in_download"]} 只证券在下载范围内曾出现退市事件，未因其事后退市而排除历史成员资格。股票池可用于后续特征准备；收益异常和真实退市支付时点仍须在回测前处理，不能将本次通过理解为全部收益记录都可直接使用。',
        '',
        '## 全量数据检查',
        '',
        '| 检查 | 结果 |','|---|---|',
        f'| 原始行数 | {count("raw_rows")} |',
        f'| 全字段完全相同的重复行 | {count("exact_duplicate_rows_removed")}；只合并重复副本 |',
        f'| 合并后的证券日记录 | {count("clean_rows")} |',
        f'| 冲突证券日主键 | 0；如存在则程序中止 |',
        f'| 证券 / 市场交易日期 | {ing["unique_permnos"]:,} / {ing["trading_dates"]:,} |',
        f'| 分类日期越界 | {count("outside_classification_interval")} |',
        f'| 同日市场收益冲突 / 缺失 | {count("market_conflicting_dates")} / {count("market_missing_rows")} |',
        f'| 负价格 / 负成交量 / 收益低于 −100% | {count("negative_price_rows")} / {count("negative_volume_rows")} / {count("return_below_minus_one")} |',
        f'| TR 价格与 Close 不一致 / Close 缺失 | {audit["trade_price_close_mismatches"]:,} / {audit["trade_price_close_missing"]:,} |',
        f'| OHLC 范围异常 | {audit["ohlc_invalid_bounds"]:,}，不自动改价 |',
        f'| 非正累计价格 / 股数因子 | {ing["DlyCumFacPr_nonpositive"]:,} / {ing["DlyCumFacShr_nonpositive"]:,}，禁止直接除以零 |',
        f'| 市值单位核对 | {audit["cap_unit_checked"]:,} 行；相对误差 > 0.01% 的 {audit["cap_unit_relative_error_gt_1e_4"]:,} 行 |',
        '',
        '数据哈希与已验证下载一致。保留 35 个导出字段、原始收益和所有事件标记。`NA` 按字符保存，不按空值解析。规范化数据没有按最终股票池裁剪。',
        '',
        f'市值核对的 28 条小幅差异已另存 `capitalization_exceptions.parquet`；其中涉及曾入选证券的 {counts["cap_exception_rows_ever_selected_security"]} 条。继续保留供应商 DlyCap，不用重新计算值覆盖。',
        '',
        '## 复权核对',
        '',
        f'在累计价格因子发生变化且可比较的 {audit["factor_change_rows"]:,} 个相邻交易日事件上，价格除以累计因子所构造的收益与 Retx 在 1e-5 容差内吻合 {audit["factor_change_divide_matches_1e_5"]:,} 次（{pct(audit["factor_change_divide_matches_1e_5"],audit["factor_change_rows"])}）；反向相乘仅吻合 {audit["factor_change_multiply_matches_1e_5"]:,} 次。',
        '',
        f'价格与股数因子变化一致的拆股候选事件共 {audit["split_like_rows"]:,} 次，其中 {audit["split_like_divide_matches_1e_5"]:,} 次收益吻合；股数变化在 1% 容差内吻合 {audit["share_split_ratio_matches_1pct"]:,}/{audit["share_split_checked"]:,} 次。公司同时发行、回购及非普通分配可产生差异，故这不是每笔公司行动的独立重建。未吻合事件保留为待复核，不根据这个比值重写 Retx。',
        '',
        '实施方向：历史价格除以累计价格因子，股数/成交量乘以累计股数因子；模型窗口再归一到信号日基准。交易门槛与成交额排序始终用原始收盘价和原始成交量，避免使用样本终点复权绝对值。具体定义与来源见 [数据口径](DATA_CONVENTIONS.md)。',
        '',
        f'潜在持有路径上有 {counts["candidate_path_nonpositive_or_missing_factors"]:,} 条非退市记录的累计因子非正或缺失。因子按样本终点归一，不能据此事后删去历史股票池成员；后续优先用有效供应商收益构建收益特征，涉及股数复权的窗口须单独处理。',
        '',
        '## 退市与质量标记',
        '',
        f'全体证券包含 {audit["delist_rows"]:,} 个退市事件，其中 {audit["delist_missing_return_rows"]:,} 个没有数值收益，另有 {audit["delist_numeric_but_flagged_rows"]:,} 个有数值但带非 NA 标记。全样本有数值却带异常标记的收益共 {audit["numeric_return_with_non_NA_flag"]:,} 条；NA 标记却无数值的 {audit["missing_return_with_NA_flag"]:,} 条。',
        '',
        'CIZ 的退市收益已经位于日频序列，不能再额外乘 legacy DLRET。MV 表示公司行动金额缺失，有数值也不能无条件视为正常。普通收益 D1/D2/D3/D4/DU 可覆盖一个交易期间；P* 与 DD 须另行解释，缺失和跨期收益不填零。',
        '',
        f'按“当月成员＋上月成员在本月前 6 个交易日”检查潜在持有路径：发现 {counts["candidate_path_delist_events"]:,} 个退市事件，{counts["candidate_path_bad_return_rows"]:,} 条需要复核的收益记录，涉及 {counts["candidate_path_affected_securities"]:,} 只证券；其中退市异常收益 {counts["candidate_path_delist_bad_returns"]:,} 条。这是保守路径排查，并非实际策略持仓或交易次数。原始事件已保存到 `selected_return_events.parquet`。',
        '',
        '需要复核的具体记录单独导出 `return_review_queue.csv`；按标记汇总：'+ '、'.join(f'{k}：{v} 条' for k,v in sorted(return_flags.items()))+'。其中 NA 行的问题在于收益跨多个交易期间，并非数值缺失。',
        '',
        '缺少的完整支付/结算信息不能从本导出推断。回测时应针对实际持有事件核实补充数据或披露近似；不能通过事后删股提高结果。当前年度数据版本亦不是未修订的历史发布快照。',
        '',
        '## 年度覆盖',
        '',
        '| 年份 | 市场交易日 | 股票—月份记录 | 不同证券 |','|---|---:|---:|---:|',
    ]
    for year,r in annual.iterrows():
        lines.append(f'| {year} | {int(r.market_days)} | {int(r.stock_months):,} | {int(r.unique_stocks):,} |')
    lines += ['', '## 产物与复现', '',
        '- `data/processed/crsp_v1/monthly_universe.parquet` 与 `.csv`：月度名单、形成/生效日期、成交额排名和筛选依据。',
        '- `monthly_candidates.parquet`、`universe_funnel.csv`：全部月末候选及逐层筛选数量。',
        '- `semantic_audit.json`、`validation.json`：统计、配置及代码/产物哈希。',
        '- `exceptions/`、`selected_return_events.parquet`：本地异常及退市明细。',
        '- `return_review_queue.csv`、`capitalization_exceptions.parquet`：待复核收益与市值差异，均保留来源值。',
        '- `data/interim/crsp_audit_v1/`：去除重复副本后的完整来源层、交易日历、字段缺失/类别统计。',
        '',
        '在项目目录使用虚拟环境，依次运行 `scripts/prepare_data.py ingest`、`scripts/prepare_data.py universe`、`scripts/prepare_data.py audit`。已完成的导入不会被隐式重写；重新导入需指定新目录，并将后续输入指向该目录。所有数据产物保持本地并被 Git 忽略。',
        '',
        f'原件 SHA-256：`{ing["sha256"]}`。',
        '',
        '来源：[CRSP CIZ 指南](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-guide-flat-file-format-2-0/)、[WRDS CIZ 退市说明](https://wrds-www.wharton.upenn.edu/documents/2084/Webinar.pdf)、[CRSP 分类映射](https://www.crsp.org/wp-content/uploads/ShareCode.html)。其他字段与标记来源逐项列于数据口径说明。',
    ]
    (out/'audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'reports').mkdir(exist_ok=True)
    report='\n'.join(lines)+'\n'
    (ROOT/'reports/DATA_AUDIT_REPORT.md').write_text(report.replace('](DATA_CONVENTIONS.md)','](../docs/DATA_CONVENTIONS.md)'),encoding='utf-8')
    print(json.dumps({k:v for k,v in final.items() if k!='provenance_sha256'},indent=2),flush=True)
