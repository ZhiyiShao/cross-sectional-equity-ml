# 两层代码阅读路线

先读核心研究逻辑，辅助实现按问题查阅。完整研究流程和必要核验全部保留。

## 核心精读

```mermaid
flowchart LR
  root["核心精读<br>按研究流程阅读"]
  root --> a["01 历史股票池"]
  a --> a1["data/universe.py<br>历史快照 → 筛选 → 公司去重 → 排名"]
  root --> b["02 特征与标签"]
  b --> b1["data/features.py<br>21 特征 → 未来收益 → 日期边界"]
  root --> c["03 模型与训练"]
  c --> c1["models/preprocessing.py<br>排名 → 拟合状态 → 应用变换"]
  c --> c2["models/selection.py + final_fit.py<br>验证选参 → 冻结后拟合"]
  root --> d["04 组合与回测"]
  d --> d1["backtest/market.py + engine.py<br>分数 → 目标持仓 → 交易与净值"]
  d --> d2["backtest/ledger.py + settlement.py<br>购买力 → 应收应付 → 历史结算"]
  root --> e["05 评价与解释"]
  e --> e1["models/common.py + evaluation/frozen.py<br>Rank IC → 配对区间 → 测试结论"]
  e --> e2["evaluation/exposures.py<br>风格、行业与收益贡献"]
  classDef core fill:#eff6ff,stroke:#64748b,color:#0f172a,font-size:14px;
  class root,a,a1,b,b1,c,c1,c2,d,d1,d2,e,e1,e2 core;
```

### 01 历史股票池
精读 `data/universe.py` 的 `filter_codes()`、`regular_trade()`、`monthly_snapshots()`、`select_month()`。先看筛选规则与信息时间，再看 `build()` 如何串起来。验收：能解释每月为什么选这些股票，以及未来记录为什么不能改变过去股票池。

### 02 特征与标签
精读 `data/features.py` 的 `make_features()`、`compound_past()`、`return_segments()`、`label_one()`、`split_masks()`。验收：能解释特征窗口、t+1 到 t+6 标签、缺失与跨分割边界处理。

### 03 模型与训练
从 `models/preprocessing.py` 读 `cross_sectional_rank()` → `fit_state()` / `fit_preprocessing()` → `transform_features()`。再读 `selection.py` 的候选配置与训练循环，以及 `final_fit.py` 的合格样本边界。数据掩码与模型构造按需查 `common.py`。验收：能解释四组方案、训练期拟合预处理、验证选参和冻结后拟合。

### 04 组合与回测
先读 `market.make_targets()`，再读 `engine.simulate()`；遇到资金约束时进入 `ledger.Ledger`，遇到结算日期时进入 `settlement.SettlementCalendar`，最后看 `engine.verify()`。验收：能解释信号如何变成买卖、现金和净值，为什么标签收益不等于组合收益。

### 05 评价与解释
读 `common.daily_rank_ic()`、`market.performance()` 与 `evaluation/frozen.py` 的配对区间和结果组织；再看 `exposures.py` 的风格、行业与贡献统计。验收：能区分排序能力、扣费收益、风险暴露和统计不确定性。

每轮只精读上述核心函数；输出文件、哈希、绘图等先掌握输入输出即可。核心精读和辅助查阅是学习优先级，运行时仍使用完整流程。

## 辅助查阅

```mermaid
flowchart LR
  root["辅助查阅<br>理解职责，遇到问题再展开"]
  root --> a["数据可信度"]
  a --> a1["data/ingest.py + audit.py + returns.py<br>导入、口径和异常收益"]
  a --> a2["data/validation.py + date_gate.py<br>独立核验与日期隔离"]
  root --> b["模型核验"]
  b --> b1["models/validation.py + verification.py<br>复算预测、检查训练边界和保存一致性"]
  root --> c["公司行动与运行"]
  c --> c1["data/validation_events.py + event_audit.py + test_events.py<br>事件资料与可执行规则"]
  c --> c2["backtest/validation.py + frozen.py<br>输入 → 固定情景 → 输出核验"]
  c --> c3["backtest/calendar.py + validation_calendar.py<br>绑定各阶段日历"]
  root --> d["诊断与复现"]
  d --> d1["evaluation/groups.py<br>分组收益与覆盖率"]
  d --> d2["scripts + config + tests + provenance.py + paths.py<br>入口、规则、独立测试与历史溯源"]
  d --> d3["docs + results + 本地数据/产物<br>研究协议与证据"]
  classDef aux fill:#f1f5f9,stroke:#64748b,color:#0f172a,font-size:14px;
  class root,a,a1,a2,b,b1,c,c1,c2,c3,d,d1,d2,d3 aux;
```

验证期运行组织已拆为 `prepare_inputs()` → `run_scenarios()` → `save_results()`；`main()` 只串联步骤。原始特征预处理已集中到 `models/preprocessing.py`，`models/common.py` 保留兼容导出，原调用方式仍然有效。

完整函数清单见 [导图大纲](README.md)。图中的函数关系表示阅读顺序，不代表完整调用依赖。

旧三折配置与 `tests/reference/cv.py` 属于历史对照；当前选参仍使用 2015–2019 训练、2020–2021 验证。`tests/reference/` 中旧引擎是独立回归对照，不应作为冗余删除。
