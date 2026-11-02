# 无需 WRDS 的完整演示

演示用合成数据串联项目的核心研究代码。克隆仓库、安装依赖后即可运行，全程无需网络、账户、行情下载或本地历史产物。

```bash
python scripts/check_environment.py --demo
```

默认输出到 `outputs/demo/`。已存在非空目录时拒绝覆盖；再次运行可指定新目录：

```bash
python scripts/check_environment.py --demo --output outputs/demo_run2
```

## 运行了什么

| 步骤 | 演示内容 | 复用的核心代码 |
|---|---|---|
| 生成输入 | 60 只虚构证券、随机日行情、一个现金退市事件 | 演示自行生成规范化输入，跳过供应商下载和导入 |
| 历史股票池 | 上月末信息筛选，252 日历史与 60 日流动性，同公司股份去重，每月最多 40 只 | `universe.monthly_snapshots / select_month` |
| 特征与标签 | 原有 21 个特征；t+1 收盘至 t+6 收盘超额收益；排除跨阶段标签 | `features.make_features / label_one / split_masks` |
| 训练和选参 | 原始 / 排名特征 × Ridge / LightGBM，每组 2 个候选，共 8 次拟合 | `preprocessing.fit_state / transform_features`、`common.model_grid / new_model / daily_rank_ic` |
| 最终拟合与预测 | 固定选中配置，重新拟合 2 份预处理和 4 个模型；保存并重新加载模型预测 | 同一预处理、模型构造和模型保存函数 |
| 交易与结算 | 四组模型、动量、等权共六组；前 10% 等权，每 5 日信号，次日收盘交易；买卖各 10 bps | `market.make_targets / build_market`、`engine.simulate`、`Ledger / SettlementCalendar` |
| 核验与评价 | 买卖应收应付、现金退市到账、净值对账；Rank IC、扣费收益、风险与换手 | `engine.verify`、`common.daily_rank_ic`、`market.performance` |

日期仅服务于演示：**2019 预热 → 2020 训练 → 2021 验证选参 → 2020–2021 最终拟合 → 2022 合成留出集评价**。生成到 2023 年初是为了识别并排除跨年标签，回测不延伸到这段数据。

训练入口只接收开发期样本；排名只在同一天的合格股票之间计算；预处理只拟合训练样本，最终拟合时再用开发期合格样本更新。预测函数接收特征与当日资格信息，选股不要求未来标签有效，评价时才关联标签。现金退市窗口的严格标签缺失，但持仓仍由交易引擎处理。

## 先看哪些输出

| 文件 | 用途 |
|---|---|
| `demo_overview.png` | 六组组合净值与五组信号的平均 Rank IC，图中标有合成数据说明 |
| `summary.csv` | 收益、Sharpe、最大回撤、换手、费用和标签覆盖率 |
| `universe_funnel.csv` / `universe.csv` | 每月筛选过程与最终成员 |
| `features_labels.parquet` | 特征、标签状态、训练验证划分和边界排除标记 |
| `validation_scores.csv` / `selected.json` | 8 次候选结果及每组选中的配置 |
| `preprocessing/` / `models/` / `fit_ledger.json` | 预处理状态、4 个最终模型、拟合次数与日期边界 |
| `predictions.parquet` / `schedule.csv` | 留出期分数与信号、成交、持有期日期 |
| `daily.csv` / `orders.csv` / `holdings.parquet` / `events.csv` | 净值、交易、持仓及公司行动，可逐笔追查 |
| `manifest.json` | 固定种子、配置、依赖版本、输入源码及输出哈希、完成状态 |

数据生成和串联代码集中在 [`demo.py`](../src/equity_research/demo.py)，独立配置位于 [`config/demo.json`](../config/demo.json)。原有六个命令入口保持不变，环境检查入口新增 `--demo` 选项。

## 演示与正式研究的关系

这里的行情、证券和公司行动全部虚构；日历把周一至周五视作交易与结算日，使用同一个结算实现计算该时期的 T+2，不模拟真实美国节假日。演示缩小了股票数量、时间跨度和参数预算，包含一个现金事件；真实研究的换股、OTC、CVR 和历史日历等分支继续由原有测试覆盖。

**演示验证的是流程能否运行、时间边界是否正确、资金能否对上；合成收益不能作为策略有效性的证据，也不会复现 README 中的真实测试数值。** 正式研究仍使用原来的数据、冻结参数、执行日历及结果目录。

整个演示纳入 `python -m pytest -q` 和 GitHub Actions。固定随机种子保证输入可复现；相同依赖版本下可对比输出，跨平台浮点结果允许微小差异。输出目录被 Git 忽略，无需提交生成数据或模型。
