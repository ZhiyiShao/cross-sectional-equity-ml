"""Independent frozen prediction/metric checks and report for four CV-selected fits."""
from equity_research.provenance import hash_matches
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

from equity_research.data.ingest import ROOT, write_json
from equity_research.models.common import sha, load_development_data, sample_masks, transform_features, load_lightgbm
from equity_research.models.preprocessing import require, frame_digest, cross_sectional_rank
from equity_research.models.selection import OUT, SOURCE, CONFIG, selected_candidates


def report(started, validation):
    selected=json.loads((OUT/'selected.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'metrics.csv',dtype={'year':str})
    aliases={'raw':'原始','cross_sectional_rank':'排名'}
    lines=['# 完整 2015–2019 训练、2020–2021 选参：四组方案','',
        '本轮恢复开发验证期选参，固定完成 80 次拟合（四组各20配置）。每个模型均使用完整2015–2019训练；未合并验证期重训，测试期保持封存。原21特征、标签及样本口径不变。','',
        '协议：[固定实验协议](../docs/FULL_TRAIN_SELECTION_PROTOCOL.md)；配置：[参数网格](../config/full_train_selection.json)。','',
        '## 四组入选结果','',
        '| 方案 | 参数 | 训练 IC | 验证 IC | 2020 IC | 2021 IC | 正 IC 日期比例 | 验证 RMSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    plot={}
    for c in selected+[dict(id='momentum20',model='动量',representation='raw',parameters={})]:
        g=metrics.loc[metrics.configuration.eq(c['id'])]
        tr=g.loc[g.split.eq('train')&g.year.eq('all')].iloc[0]
        va=g.loc[g.split.eq('validation')].set_index('year');v=va.loc['all']
        name=aliases[c['representation']]+' '+c['model']
        lines.append(f'| {name} | `{json.dumps(c["parameters"])}` | {tr.mean_rank_ic:.6f} | {v.mean_rank_ic:.6f} | {va.loc["2020","mean_rank_ic"]:.6f} | {va.loc["2021","mean_rank_ic"]:.6f} | {v.positive_fraction:.1%} | '+(f'{v.rmse:.6f}' if c['id']!='momentum20' else '—')+' |')
        plot[c['id']]=[v.mean_rank_ic,va.loc['2020','mean_rank_ic'],va.loc['2021','mean_rank_ic']]
    lines+=['','按验证平均 IC 选择；精确同分按网格顺序。四组分别入选，没有额外根据分年表现或回测收益筛选。原始 Ridge/LightGBM 恢复第一版网格，排名输入使用同一网格。Ridge 此次使用直接 alpha，而非旧 CV 的 lambda 参数化。','',
        '## 成对比较','', '| 模型 | 区块长度 | 排名−原始 IC | 95% 区间 |','|---|---:|---:|---|']
    pairs=json.loads((OUT/'paired_bootstrap.json').read_text(encoding='utf-8'))
    for p in pairs:lines.append(f'| {p["model"]} | {p["block_signals"]} | {p["mean_difference"]:+.6f} | [{p["ci_low"]:+.6f}, {p["ci_high"]:+.6f}] |')
    lines+=['','这是分别选参后的整套流程差异，超参数可以不同。区间条件于入选预测，没有校正选参、多次研究选择或多重检验；2020–2021 是开发选参集，不是最终独立测试。','',
        '## 全部配置与复现','',
        f'全部80模型已独立重载；SciPy 重算 {validation["independent_daily_ic_checks"]:,} 个逐日 IC。训练/验证候选、严格标签及动量与原基线一致，预处理和排名均核验。旧实验保持不变。','',
        '原始表示缩尾0.5%/99.5%、中位数填补、标准化只在训练主样本拟合；排名总体使用当天事前合格股票，不使用未来标签可用性。训练和验证边界按标签终点净化。','',
        '![验证结果](../outputs/full_train_selection_v1/selected_ic.png)','',
        '本轮不包含新入选组合的回测、收益或暴露归因。此前20.03%等组合收益属于旧CV入选模型，不能转用于本轮模型；后续需明确使用本轮 selected_predictions.parquet 重新计算。','',
        '训练与验证主样本覆盖见 coverage.csv。全部配置逐年指标在 metrics.csv，逐日IC在 daily_rank_ic.csv，80次拟合记录在 fit_ledger.json，四个入选配置在 selected.json，全部候选预测在 development_predictions.parquet，入选四组预测在 selected_predictions.parquet。模型与证券级数据只存本地忽略目录。','',
        '```powershell','.\\.venv\\Scripts\\python.exe scripts/train_models.py select','.\\.venv\\Scripts\\python.exe scripts/verify.py selection','```','',
        '训练拒绝非空目录，已有实验只运行核验脚本，不重复拟合。']
    pd.DataFrame(plot,index=['all','2020','2021']).T.plot.bar(figsize=(12,5),rot=15)
    plt.axhline(0,color='gray',lw=.7);plt.ylabel('Mean Rank IC');plt.title('Validation-selected configurations (selection-biased)');plt.tight_layout();plt.savefig(OUT/'selected_ic.png',dpi=160);plt.close()
    (ROOT/'reports/FULL_TRAIN_SELECTION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    started = json.loads((OUT/'run_started.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT/'run_summary.json').read_text(encoding='utf-8'))
    require(summary['status'] == 'completed' and summary['model_fits'] == 80, 'Incomplete fixed fit run')
    for key in ['input_sha256', 'original_artifact_sha256']:
        for path, digest in started[key].items():
            require(hash_matches(ROOT/path, digest), f'Input/prior artifact changed: {path}')
    for path, digest in summary['output_sha256'].items():
        require(sha(OUT/path) == digest, f'New artifact changed: {path}')
    cfg, cv_cfg, features = started['configuration'], started['cv_configuration'], started['features']
    require(json.loads(CONFIG.read_text(encoding='utf-8')) == cfg, 'Config changed')
    require(selected_candidates(cfg, cv_cfg) == started['candidates'], 'Selection changed')
    frame = load_development_data(SOURCE, features, cfg)
    require(frame_digest(frame) == started['filtered_development_sha256'], 'Development data changed')
    masks = sample_masks(frame, cfg)
    pred = pd.read_parquet(OUT/'development_predictions.parquet')
    old = pd.read_parquet(ROOT/'outputs/baselines_v1/development_predictions.parquet')
    keys = ['PERMNO', 'signal_date', 'split', 'eligible_main_sample']
    pd.testing.assert_frame_equal(pred[keys], frame[keys])
    pd.testing.assert_frame_equal(pred[keys], old[keys])
    np.testing.assert_allclose(pred.y_excess_5d, frame.y_excess_5d.where(masks['train'] | masks['validation']), equal_nan=True)
    np.testing.assert_allclose(pred.momentum20, old.momentum20, rtol=0, atol=0)
    ranked = cross_sectional_rank(frame, features)
    independent = np.full(ranked.shape, np.nan)
    for _, group in frame.groupby('signal_date'):
        for j, feature in enumerate(features):
            ok = group.signal_eligible & np.isfinite(group[feature])
            x = group.loc[ok, feature]
            if len(x):
                independent[x.index, j] = (rankdata(x, method='average')-.5)/len(x)-.5
    np.testing.assert_allclose(ranked, independent, atol=1e-14, equal_nan=True)
    checks = 0
    with threadpool_limits(limits=4):
        for representation in cv_cfg['representations']:
            x = frame[features] if representation == 'raw' else ranked
            state = json.loads((OUT/'preprocessing'/f'{representation}.json').read_text(encoding='utf-8'))
            a = x.loc[masks['train']].replace([np.inf, -np.inf], np.nan).to_numpy()
            med = np.nanmedian(a, axis=0)
            bounds = np.nanquantile(a, cv_cfg['raw_clip_quantiles'], axis=0) if representation == 'raw' else np.array([[-.5]*len(features), [.5]*len(features)])
            clean = np.clip(a, bounds[0], bounds[1]); clean = np.where(np.isnan(clean), med, clean)
            scale = clean.std(axis=0); scale[scale == 0] = 1
            for k, expected in dict(lower=bounds[0], upper=bounds[1], median=med, mean=clean.mean(axis=0), scale=scale).items():
                np.testing.assert_allclose(state[k], expected, atol=1e-12, rtol=1e-12)
            require(state['fitted_rows'] == int(masks['train'].sum()), 'Wrong fitting population')
            X = transform_features(x, state)
            for c in [c for c in started['candidates'] if c['representation'] == representation]:
                if c['model'] == 'ridge':
                    model = joblib.load(OUT/'models'/f'{c["id"]}.joblib')
                    np.testing.assert_allclose(model.alpha, c['parameters']['alpha'])
                else:
                    model = load_lightgbm(OUT/'models'/f'{c["id"]}.txt')
                np.testing.assert_allclose(model.predict(X), pred[c['id']], atol=1e-12, rtol=1e-10)
                checks += 1
    daily = pd.read_csv(OUT/'daily_rank_ic.csv', parse_dates=['signal_date'])
    metrics = pd.read_csv(OUT/'metrics.csv', dtype={'year': str})
    ic_checks = 0
    for split, mask in masks.items():
        for name in ['momentum20']+[c['id'] for c in started['candidates']]:
            g = pred.loc[mask]
            ic = pd.Series({date: float(spearmanr(rows[name], rows.y_excess_5d).statistic) for date, rows in g.groupby('signal_date')})
            expected = daily.loc[daily.split.eq(split) & daily.configuration.eq(name)].set_index('signal_date').rank_ic
            require(ic.index.equals(expected.index) and np.isfinite(ic).all(), 'IC dates invalid')
            np.testing.assert_allclose(ic, expected, atol=1e-12)
            ic_checks += len(ic)
            for year, rows in [('all', g)]+[(str(y), r) for y, r in g.groupby(g.signal_date.dt.year)]:
                values = ic if year == 'all' else ic.loc[ic.index.year == int(year)]
                row = metrics.loc[metrics.split.eq(split) & metrics.year.eq(year) & metrics.configuration.eq(name)].iloc[0]
                require(row.rows == len(rows) and row.dates == len(values), 'Metric coverage mismatch')
                for key, value in dict(mean_rank_ic=values.mean(), positive_fraction=values.gt(0).mean(), ic_std=values.std()).items():
                    np.testing.assert_allclose(row[key], value, atol=1e-12)
                if name != 'momentum20':
                    np.testing.assert_allclose(row.rmse, np.sqrt(np.mean((rows[name]-rows.y_excess_5d)**2)), atol=1e-12)
    pairs = json.loads((OUT/'paired_bootstrap.json').read_text(encoding='utf-8'))
    pivot = daily.loc[daily.split.eq('validation')].pivot(index='signal_date', columns='configuration', values='rank_ic')
    for p in pairs:
        values = (pivot[p['ranked']]-pivot[p['raw']]).to_numpy()
        rng = np.random.default_rng(cfg['bootstrap']['seed'])
        draws = []
        for _ in range(cfg['bootstrap']['replications']):
            starts = rng.integers(0, len(values), size=int(np.ceil(len(values)/p['block_signals'])))
            indexes = np.concatenate([(s+np.arange(p['block_signals'])) % len(values) for s in starts])[:len(values)]
            draws.append(values[indexes].mean())
        np.testing.assert_allclose(np.quantile(draws, [.025, .975]), [p['ci_low'], p['ci_high']], atol=1e-12)
        np.testing.assert_allclose(values.mean(), p['mean_difference'], atol=1e-12)
    selected=json.loads((OUT/'selected.json').read_text(encoding='utf-8'))
    scores=metrics.loc[metrics.split.eq('validation')&metrics.year.eq('all')].set_index('configuration')
    for c in selected:
        group=[g for g in started['candidates'] if g['model']==c['model'] and g['representation']==c['representation']]
        require(max(group,key=lambda g:scores.loc[g['id'],'mean_rank_ic'])==c,'Wrong validation selection')
    slim=pd.read_parquet(OUT/'selected_predictions.parquet')
    pd.testing.assert_frame_equal(slim,pred[slim.columns])
    require(checks==80 and len(selected)==4,'Model count mismatch')
    result = dict(validation_passed=True, models_reloaded=checks, independent_daily_ic_checks=ic_checks,
        paired_intervals_checked=len(pairs), main_sample_matches_original_baseline=True,
        original_artifacts_unchanged=True, new_fit_calls=0, test_rows_returned=0, test_performance_evaluated=False)
    write_json(OUT/'validation.json', result)
    report(started, result)
    print(json.dumps(result), flush=True)
