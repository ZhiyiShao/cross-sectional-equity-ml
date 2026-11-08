"""Aggregate V2 historical replay diagnostics; no training or unseen observation IO."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from equity_research.models.common import circular_block_interval, daily_rank_ic, sha
from equity_research.models.preprocessing import require
from equity_research.models.ranking_selection import ranking_labels, ndcg_top
from equity_research.models.final_fit_v2 import read_json, write_json, check_metadata, IDS
from equity_research.evaluation.v2 import locked_bundle
from equity_research.backtest.replay_v2 import ROOT, OUT, PUBLIC, START, END, check_targets

NAMES = dict(zip(IDS, ['Ridge 21','Ridge 33','LGBM regression 21','LGBM regression 33','LGBM ranking 21','LGBM ranking 33']))
NAMES.update(momentum20='Momentum20', equal_weight='Universe EW')


def interval(values, block, scale=1):
    r = circular_block_interval(np.asarray(values)*scale, block, 2000, 20261107)
    return dict(mean=r['mean_difference'], ci_low=r['ci_low'], ci_high=r['ci_high'], observations=r['paired_dates'])


def prediction_diagnostics(full, plan):
    valid = (full.signal_eligible & full.label_status.isin(['ok','ok_bridged']) &
             np.isfinite(full.y_excess_5d) & full.label_end_date.gt(full.signal_date) & full.label_end_date.le(END))
    strict = full.loc[valid]
    coverage = full.assign(valid_label=valid).groupby('signal_date').agg(
        candidates=('PERMNO','size'),eligible=('signal_eligible','sum'),valid_labels=('valid_label','sum'))
    coverage['label_coverage'] = coverage.valid_labels/coverage.eligible
    daily, top, summaries = [], [], []
    for model in IDS+['momentum20']:
        ic = daily_rank_ic(strict.signal_date, strict[model], strict.y_excess_5d)
        require(np.isfinite(ic).all() and len(ic)==len(coverage), 'Undefined/missing signal IC')
        for date,g in strict.groupby('signal_date'):
            independent = spearmanr(g[model],g.y_excess_5d).statistic
            np.testing.assert_allclose(ic.loc[date],independent,atol=1e-12)
            relevance,_ = ranking_labels(g)
            daily.append(dict(model=model,signal_date=date,rank_ic=ic.loc[date],ndcg_top10=ndcg_top(relevance,g[model],g.PERMNO)))
        for date,g in full[full.signal_eligible].groupby('signal_date'):
            selected = g.sort_values([model,'PERMNO'],ascending=[False,True]).head(int(np.ceil(len(g)/10)))
            qualified = valid.loc[selected.index]
            top.append(dict(model=model,signal_date=date,selected=len(selected),valid_labels=int(qualified.sum()),
                top_label_mean=float(selected.loc[qualified,'y_excess_5d'].mean()),coverage=float(qualified.mean())))
        summaries.append(dict(model=model,rank_ic=ic.mean(),positive_ic_fraction=ic.gt(0).mean(),
            ic_ci_low=interval(ic,10)['ci_low'],ic_ci_high=interval(ic,10)['ci_high'],
            label_coverage=len(strict)/full.signal_eligible.sum()))
    daily, top = pd.DataFrame(daily), pd.DataFrame(top)
    summaries = pd.DataFrame(summaries).merge(daily.groupby('model',as_index=False).ndcg_top10.mean(),on='model')
    summaries = summaries.merge(top.groupby('model',as_index=False).agg(top_label_mean=('top_label_mean','mean'),top_label_coverage=('coverage','mean')),on='model')
    wide = daily.pivot(index='signal_date',columns='model',values='rank_ic')
    pairs = [plan['evaluation']['primary_contrast']]+plan['evaluation']['predeclared_secondary_contrasts']
    comparisons = pd.DataFrame([dict(model=a,comparator=b,**interval(wide[a]-wide[b],10)) for a,b in pairs])
    return coverage,daily,top,summaries,comparisons


def portfolio_diagnostics(daily, metrics, plan):
    rows = []
    for r in metrics.to_dict('records'):
        g = daily[daily.model.eq(r['model'])&daily.variant.eq(r['variant'])&daily.cost_bps.eq(r['cost_bps'])].sort_values('date').iloc[1:]
        ew = daily[daily.model.eq('equal_weight')&daily.cost_bps.eq(r['cost_bps'])].sort_values('date').iloc[1:]
        require(g.date.equals(ew.date) or np.array_equal(g.date.to_numpy(),ew.date.to_numpy()),'Paired portfolio dates')
        active = g.return_net.to_numpy()-ew.return_net.to_numpy()
        te = active.std(ddof=1)*np.sqrt(252)
        rows.append(dict(**r,annual_mean_active=active.mean()*252,tracking_error=te,
            information_ratio=active.mean()*252/te if te>0 else np.nan))
    metrics = pd.DataFrame(rows)
    main = daily[daily.cost_bps.eq(10)]
    comparisons = []
    pairs = [plan['evaluation']['primary_contrast']]+plan['evaluation']['predeclared_secondary_contrasts']
    for variant in ['original','constrained']:
        wide = main[main.variant.eq(variant)].pivot(index='date',columns='model',values='return_net').sort_index().iloc[1:]
        for a,b in pairs:
            comparisons.append(dict(model=a,comparator=b,variant=variant,
                role='primary' if variant=='constrained' and [a,b]==pairs[0] else 'exploratory',
                **interval(wide[a]-wide[b],20,252)))
    for model in IDS:
        wide = main[main.model.eq(model)].pivot(index='date',columns='variant',values='return_net').sort_index().iloc[1:]
        comparisons.append(dict(model=model,comparator='same model original',variant='constrained-minus-original',
            role='exploratory',**interval(wide.constrained-wide.original,20,252)))
    quarterly = []
    for (model,variant),g in main.groupby(['model','variant']):
        g=g.sort_values('date').iloc[1:]
        for quarter,q in g.groupby(g.date.dt.to_period('Q')):
            quarterly.append(dict(model=model,variant=variant,quarter=str(quarter),net_return=(1+q.return_net).prod()-1))
    return metrics,pd.DataFrame(comparisons),pd.DataFrame(quarterly)


def figure(daily,metrics,predictive,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    colors=dict(zip(IDS+['momentum20','equal_weight'],plt.get_cmap('tab10').colors[:8]))
    for ax,variant in zip(axes[0],['original','constrained']):
        for model in IDS+['momentum20','equal_weight']:
            g=daily[daily.cost_bps.eq(10)&daily.model.eq(model)&daily.variant.eq(variant if model in IDS else 'benchmark')]
            ax.plot(g.date,g.nav,label=NAMES[model],color=colors[model],linewidth=1.3)
        ax.set_title(f'{variant.title()} portfolios | 10 bps per side')
        ax.set_ylabel('Net NAV');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    x=np.arange(len(predictive));axes[1,0].bar(x,predictive.rank_ic,color=[colors[m] for m in predictive.model])
    axes[1,0].errorbar(x,predictive.rank_ic,yerr=np.vstack([predictive.rank_ic-predictive.ic_ci_low,predictive.ic_ci_high-predictive.rank_ic]),fmt='none',ecolor='black',capsize=3)
    axes[1,0].set_xticks(x,[NAMES[m] for m in predictive.model],rotation=25,ha='right',fontsize=8)
    axes[1,0].axhline(0,color='black',lw=.6);axes[1,0].set_title('Mean Rank IC | 95% block intervals')
    for model in IDS+['momentum20','equal_weight']:
        g=metrics[metrics.model.eq(model)&metrics.variant.eq('constrained' if model in IDS else 'benchmark')].sort_values('cost_bps')
        axes[1,1].plot(g.cost_bps,g.cagr*100,marker='o',color=colors[model],label=NAMES[model])
    axes[1,1].set_title('Constrained CAGR vs one-way costs');axes[1,1].set_xlabel('bps per side');axes[1,1].set_ylabel('CAGR (%)');axes[1,1].grid(alpha=.2)
    fig.suptitle('V2 2022–2025 POST-TEST HISTORICAL REPLAY | fixed 2015–2021 models\nOTC quote proxies; CVR rights excluded from investable NAV',fontsize=11)
    fig.tight_layout();fig.savefig(PUBLIC/'overview.png',dpi=170);plt.close(fig)


def report(metrics,predictive,comparisons,audit,primary):
    main=metrics[metrics.cost_bps.eq(10)]
    text=['# V2：2022–2025 事后历史稳健性评价','',
        '六组模型使用冻结参数及 2015–2021 最终拟合权重。本次新增拟合和调参均为 0；原结果保留，未读取新的未见数据。2022–2025 已在此前研究中看过，本轮结果不属于独立未见测试。','',
        '主模型保持增强 Ridge，主要对照保持原有 Ridge；增强排序 LightGBM 仍为挑战模型。', '',
        '## 同条件组合结果','',
        '单边 10 bps、公司行动 lag 2、普通卖款可再投资、历史应收应付及 T+2/T+1 日历。原始组合为前 10% 等权；风险约束沿用单股 2%、行业主动权重 ±2%、beta 主动暴露 ±0.1、个股年化波动率加权均值不超过等权基准 +0.1。最后一项不是组合波动率目标。','',
        '| 模型 | 原始 CAGR | 约束 CAGR | 约束 Sharpe | 约束最大回撤 | 平均 Rank IC |','|---|---:|---:|---:|---:|---:|']
    for model in IDS:
        a=main[main.model.eq(model)&main.variant.eq('original')].iloc[0]
        b=main[main.model.eq(model)&main.variant.eq('constrained')].iloc[0]
        p=predictive[predictive.model.eq(model)].iloc[0]
        text.append(f'| {NAMES[model]} | {a.cagr:.2%} | {b.cagr:.2%} | {b.sharpe:.3f} | {b.max_drawdown:.2%} | {p.rank_ic:.4f} |')
    text+=['','| 基准 | CAGR | Sharpe | 最大回撤 |','|---|---:|---:|---:|']
    for model in ['momentum20','equal_weight']:
        r=main[main.model.eq(model)].iloc[0]
        text.append(f'| {NAMES[model]} | {r.cagr:.2%} | {r.sharpe:.3f} | {r.max_drawdown:.2%} |')
    text+=['','## 预先指定的主要比较','',
        f"增强 Ridge 减原有 Ridge，风险约束组合的年化平均日净收益差为 **{primary['annual_mean_net_difference']:.2%}**，20 日配对循环区块 Bootstrap、2,000 次重采样的 95% 区间为 **[{primary['ci_low']:.2%}, {primary['ci_high']:.2%}]**。这是日收益差均值乘 252，不是 CAGR 差。",'',
        '区间判读：'+('该历史期间的区间下界大于零，支持这个预先指定对照的净收益改善；由于期间已被看过，不能作为独立未见期的确认。' if primary['ci_low']>0 else '区间未排除零或指向负向，不能据此宣布增强信息带来稳定净收益改善；主模型身份不变。'),'',
        '其他参数、模型和组合的差异均作辅助解释。逐日 IC 使用连续严格标签，NDCG 使用冻结的 0–9 级与线性 gain。前 10% 选股先在全部可选股票中确定，再统计有效标签覆盖；顶部标签收益与实际净值收益分开报告。','',
        '## 覆盖、风险与执行核验','',
        f"- {audit['signals']} 个完整信号，{audit['market_dates']} 个净值日期；{audit['candidate_rows']:,} 条候选、{audit['strict_label_rows']:,} 条合格严格标签。",
        f"- 14 条主组合与 42 条成本敏感性路径，共 {audit['paths']} 条；{audit['ledger_days']:,} 个主组合日期完成独立资金/净值核对。",
        '- 原 2015-01-02 锚定的五交易日信号网格保持不变；第一信号为 2022-01-05、首次成交为 2022-01-06、最后清算为 2025-12-24。年度统计因此首尾为实际完整策略窗口，不包括被边界规则剔除的零碎持有期。',
        '- 等权基准复现原 V1 净值。每条路径独立由现金 NAV=1 开始；无模型滚动更新。',
        '- 目标约束单独核验；成交后的暴露、覆盖不足、调仓间漂移和事件继承不会被当成目标达标。公开暴露为调仓成交时点，用当期信号暴露衡量；不等于全日频因子归因。',
        '- 银行退市仍使用已批准的首个退市后报价强制处置代理；CVR 登记但不进入可投资净值。报价不是实际成交验证，CVR 最大金额不是公允价值。',
        '- 原有模型、结果及冻结参数源保持不变。新的未见期资格仍待审计。','',
        '## 文件','',
        '- [组合指标与成本敏感性](portfolio_metrics.csv)、[年度净收益](annual_returns.csv)、[季度净收益](quarterly_returns.csv)。',
        '- [预测指标](predictive_metrics.csv)、[年度 IC](annual_ic.csv)、[配对 IC 区间](paired_ic.csv)、[顶部标签及覆盖](top_decile_labels.csv)。',
        '- [收益差配对区间](paired_return_intervals.csv)、[目标与成交暴露](exposure_summary.csv)、[逐日标签覆盖](label_coverage.csv)。',
        '- [主要比较](primary_comparison.json)、[执行核验](verification.json)。个股预测、交易、持仓与事件明细保存在本地 `outputs/historical_replay_v2/`。','',
        '![统一图表](overview.png)','']
    (PUBLIC/'README.md').write_text('\n'.join(text),encoding='utf-8')


def main():
    plan,_,_=locked_bundle()
    audit=read_json(OUT/'replay_audit.json')
    require(audit['status']=='passed','Replay audit pending')
    PUBLIC.mkdir(exist_ok=False)
    daily=pd.read_parquet(OUT/'daily_portfolios.parquet',filters=[('date','>=',START),('date','<=',END)])
    full=pd.read_parquet(OUT/'scored_population_local.parquet',filters=[('signal_date','>=',START),('signal_date','<=',END)])
    require(daily.date.between(START,END).all() and full.signal_date.between(START,END).all(),'Historical scope')
    coverage,ic,top,predictive,paired_ic=prediction_diagnostics(full,plan)
    metrics,paired,quarterly=portfolio_diagnostics(daily,pd.read_csv(OUT/'portfolio_metrics.csv'),plan)
    ex=pd.read_csv(OUT/'exposures.csv',parse_dates=['date'])
    check_targets(ex,plan['portfolio']['constraints'])
    ex['year']=ex.date.dt.year
    ex['beta_active']=ex.beta-ex.benchmark_beta
    ex['vol_active']=ex.annual_stock_vol-ex.benchmark_vol
    for key,flag in [('beta_breach',ex.beta_active.abs()>.10000001),('industry_breach',ex.max_industry_active>.02000001),
                     ('weight_breach',ex.max_weight>.02000001),('vol_breach',ex.vol_active>.10000001)]:
        ex[key]=flag
    exposure=ex.groupby(['model','variant','stage','year'],as_index=False).agg(
        dates=('date','size'),coverage=('coverage','mean'),outside_universe_weight=('outside_universe_weight','mean'),
        beta=('beta','mean'),beta_active=('beta_active','mean'),annual_stock_vol=('annual_stock_vol','mean'),
        vol_active=('vol_active','mean'),log_market_cap=('log_market_cap','mean'),
        max_industry_active=('max_industry_active','mean'),max_weight=('max_weight','max'),
        beta_breach_fraction=('beta_breach','mean'),industry_breach_fraction=('industry_breach','mean'),
        weight_breach_fraction=('weight_breach','mean'),vol_breach_fraction=('vol_breach','mean'))
    annual_ic=ic.assign(year=ic.signal_date.dt.year).groupby(['model','year'],as_index=False).agg(rank_ic=('rank_ic','mean'),ndcg_top10=('ndcg_top10','mean'),signal_dates=('signal_date','size'))
    for name,table in [('portfolio_metrics',metrics),('predictive_metrics',predictive),('paired_return_intervals',paired),
                       ('quarterly_returns',quarterly),('daily_prediction_metrics',ic),('annual_ic',annual_ic),
                       ('top_decile_labels',top),('paired_ic',paired_ic),('exposure_summary',exposure)]:
        table.to_csv(PUBLIC/(name+'.csv'),index=False)
    coverage.to_csv(PUBLIC/'label_coverage.csv')
    (PUBLIC/'annual_returns.csv').write_bytes((OUT/'annual_returns.csv').read_bytes())
    evaluation=ROOT/'outputs/evaluation_v2_historical_2022'
    primary=read_json(evaluation/'primary_comparison.json')
    primary.update(interpretation_scope='post-test historical replay, not independent confirmation')
    write_json(PUBLIC/'primary_comparison.json',primary)
    write_json(PUBLIC/'verification.json',audit)
    figure(daily,metrics,predictive,paired)
    report(metrics,predictive,paired,audit,primary)
    check_metadata(read_json(OUT/'protected_metadata.json'))
    locked_bundle()
    write_json(PUBLIC/'manifest.json',{p.relative_to(ROOT).as_posix():sha(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(metrics[metrics.cost_bps.eq(10)][['model','variant','cagr','sharpe','max_drawdown']].to_string(index=False),flush=True)
    print('Primary historical comparison:',primary,flush=True)
