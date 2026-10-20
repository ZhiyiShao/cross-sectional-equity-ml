# 特征与未来五日收益数据字典

参数：`config/features_labels.json`；处理规则：[异常收益与标签政策](RETURN_POLICY.md)。数据只在本地，代码和本说明可以独立审阅。没有使用最终测试期表现选择特征。

## 21 个输入特征

以下所有窗口均按市场交易日计算、包含信号日 t，除非另有说明。`r` 为有效单交易期间的 CRSP 总收益；MP、P*、MV 和退市行不作为日收益观测。滚动窗口要求完整有效观测，缺失保留 NaN。

| 特征列 | 定义与解释 |
|---|---|
| `momentum_5/10/20/60/120`（5 列） | 过去相应交易日的 `prod(1+r)-1`。保留 −100% 收益的零财富结果。 |
| `momentum_60_skip5` | 60 日范围去掉最近 5 日，即 t−59 至 t−5 的 55 个收益连乘。 |
| `reversal_1` | `−r_t`，前一交易期间收益的反向信号。 |
| `volatility_5/20/60`（3 列） | 日收益样本标准差（ddof=1）× √252。 |
| `downside_volatility_20` | `sqrt(mean(min(r,0)^2) × 252)`，零阈值下行波动。 |
| `log_dollar_volume_20/60`（2 列） | `log1p(mean(原始收盘价 × 原始成交股数))`。这是日成交金额代理值。 |
| `dollar_volume_ratio_5_20` | 5 日平均成交金额代理值 / 20 日均值 − 1。 |
| `share_volume_ratio_5_20` | 同一股数基准下的 5 日 / 20 日平均成交量 − 1。用累计股数因子的相对比例，终点基准在比值中抵消；不输出绝对复权成交量。 |
| `share_turnover_20` | `mean(DlyVol / (ShrOut × 1000))`，股数单位一致的平均日换手。 |
| `amihud_20` | `mean(abs(r) / 日成交金额代理值 × 1,000,000)`；零成交金额的日子为缺失。 |
| `intraday_range_20` | `mean((High−Low)/Close)`，日内振幅。 |
| `close_location_5` | `mean((Close−Low)/(High−Low))`；High=Low 且价格有效时定义为 0.5。 |
| `log_market_cap` | `ln(DlyCap × 1000)`，证券层面美元市值，不冒充公司多股份合计市值。 |
| `market_beta_60` | 60 日 `cov(r,vwretd) / var(vwretd)`。两者同日对齐，市场方差必须为正。 |

成交量、价位与日内范围只在 RW/A、非退市、TR、有效正收盘价及非负成交量的日期计算；换手还要求正的总股数（CRSP 总股数，**不是自由流通股数**）。规模特征直接采用供应商市值。第一版不添加行业编码、未来收益质量状态、股票代码或 PERMNO 作为模型特征。

不在此阶段做标准化、填补、缩尾、降维或筛特征。必须让后续训练过程拟合这些步骤；Ridge 和 LightGBM 使用同一白名单与相同样本规则。`features_only.parquet` 刻意不含标签和未来执行信息。

## 目标与日期

- `signal_date`：t，收盘后形成信号。
- `entry_date`：t+1，假设买入收盘。
- `label_end_date`：t+6，计划卖出收盘。
- `stock_return_5d`：普通可执行路径在 t+2 至 t+6 的总收益；经核实的完整 P* 区间可计入一次。
- `market_return_5d`：同五个交易日期的 CRSP 市场收益连乘。
- `y_excess_5d`：股票累计收益减市场累计收益，使用小数单位；不含交易费用，费用属于后续组合回测。
- `*_event_proxy`：有效 CIZ 退市事件下冻结对价至终点的辅助近似，见政策文件。主标签不自动采用。

`label_status` 解释缺失原因及是否经过完整区间桥接。`bridged_spans` 是使用的 P* 整段次数；没有将原始缺失日填零，也没有生成虚假的逐日收益。

## 样本资格与泄漏边界

每个信号日保留原历史股票池 500 行。`signal_eligible` 只根据 t 当日的普通股/交易状态决定是否能成为新买入候选；它不读取未来收益或执行状态。

`eligible_main_sample` 表示在已知目标后，可用于监督学习/预测评价的严格主标签样本，还要求标签终点不越过所属训练、验证或测试阶段。它是**事后目标可用性标记**，不是预测前的股票池筛选规则。回测必须先为所有事前可选股票打分，不能只给未来标签有效的股票打分。缺失结果非随机，主标签评价须披露删失偏差，并比较退市近似情形。

`feature_missing_count` 不自动剔除样本；允许训练集拟合的缺失处理。`eligible_refit_through_2021` 单独表示调参后用 2015–2021 年重训的资格。`purged_boundary` 表示持有期跨过当前阶段末尾；`right_censored` 表示文件末尾无法观察完整持有期。所有这些行仍保留。

训练阶段只允许配置中的 21 个特征列进入 X。日期、证券标识用于分组对齐；标签、入场/退出日期、代理标志、label_status、样本资格等都不能进入模型输入。

## 复现

在 Project 目录依次执行：

```powershell
.\.venv\Scripts\python.exe scripts/review_returns.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/features_labels.py
.\.venv\Scripts\python.exe scripts/validate_features_labels.py
```

生成路径为 `data/processed/research_v1/`。已有完整数据集时生成脚本会拒绝覆盖，需明确归档或版本化后重建。源数据与规则哈希记录在 `dataset_summary.json`。本阶段不训练模型，也不报告任何策略表现。
