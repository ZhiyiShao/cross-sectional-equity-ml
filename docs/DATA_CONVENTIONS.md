# CRSP CIZ 数据口径与股票池规则

核对日期：2026-10-14。适用范围：本项目下载的 2014–2025 年 CRSP Annual Update / CIZ 日频导出。以下区分供应商定义、项目处理规则和实际数据检验；统计结果见数据审计报告。

## 字段与单位

| 字段 | 用途与处理 |
|---|---|
| `PERMNO`、`PERMCO` | 分别为证券和公司标识；证券日主键是 PERMNO + DlyCalDt，公司层面去重用 PERMCO。 |
| `DlyCalDt` | 市场交易日期。日历由全体证券数据汇总，并验证同日市场收益一致；不是自然日，也不是单只股票有数据的日期序列。 |
| `SecInfoStartDt/EndDt` | 分类有效区间，逐行验证交易日期落在闭区间内。 |
| `PrimaryExch`、`ConditionalType`、`TradingStatusFlg` | 交易所及交易状态；月末要求 N/A/Q、RW、A。[交易所代码](https://www.crsp.org/wp-content/uploads/appendix/FlagType_EC.html)、[交易条件代码](https://www.crsp.org/wp-content/uploads/appendix/FlagType_CT.html)。 |
| `SecurityType/SubType`、`ShareType`、`USIncFlg`、`IssuerType` | 普通股要求 EQTY/COM、NS、Y、CORP 或 ACOR，等价于传统 share code 10/11；REIT 和特殊股份不进入第一版。[CRSP 映射](https://www.crsp.org/wp-content/uploads/ShareCode.html)。 |
| `Ticker`、`HdrCUSIP`、`ShareClass`、`SICCD` | 标识和分类辅助信息。Ticker 不作为主键；HdrCUSIP 为 header 信息，不当作历史时点标识。第一版不使用 SICCD 做行业中性化，行业来源需另行核实。 |
| `DlyPrc`、`DlyPrcFlg`、`DlyClose/Open/High/Low` | 美元价格。Prc 允许采用报价均值，Close 是收盘成交价。TR 表示成交价，BA 表示报价均值；DA/DP/DM 是退市相关价格状态，不能当作日常成交价格。[价格标记](https://www.crsp.org/wp-content/uploads/appendix/FlagType_PC.html)。 |
| `DlyVol`、`ShrOut`、`DlyCap` | 分别为股、千股、千美元；内部需要美元市值时使用 DlyCap × 1000。以 DlyCap ≈ DlyPrc × ShrOut 验证量纲，保留供应商市值。收盘价 × 成交股数仅是成交金额代理值，并非逐笔成交额之和。 |
| `DlyRet/Retx` | 小数收益率，0.01 表示 1%；Ret 为总收益，Retx 排除普通现金股息，不能把 Retx 理解为完全不涉及其他分配。直接保留，不重复复权。 |
| `DlyRetMissFlg`、`DlyRetDurFlg` | `NA` 是合法的“不缺失”标记，不能由 CSV 默认 NA 解析吞掉。D1/D2/D3/D4/DU 是一个交易期间，周末 D3 不是跨三个交易日；P1–P9 为多期收益，DD 为特殊退市期间。[缺失原因](https://www.crsp.org/wp-content/uploads/appendix/FlagType_RM.html)、[收益期间](https://www.crsp.org/wp-content/uploads/appendix/FlagType_RD.html)。 |
| `DlyDelFlg` | Y 行为退市收益事件，保留在收益账本，不作为买入候选。 |
| `DlyFacPrc`、`DlyCumFacPr/Shr` | 日调整因子与累计价格/股数调整因子，不相互替代。全样本检验日因子与累计因子相邻比值及拆股收益的关系。 |
| `vwretd` | 导出附带的 CRSP 市场总收益，小数单位。同一日期应只有一个值；它是市场参照，不是本项目 500 股股票池收益。 |

数量、价格、收益和累计因子的供应商定义见 [CRSP CIZ Guide，Daily Security Data / Cumulative Adjustment Factors](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-guide-flat-file-format-2-0/)。官方页面迁移时可查阅[同一 CRSP 指南的学校存档](https://iimidr.ac.in/wp-content/uploads/2025/10/User-Guide-CRSP.pdf)。

## 复权的实际使用

本文件中的因子方向通过本地公司行动数据交叉验证：价格除以 `DlyCumFacPr`，股数/成交量乘以 `DlyCumFacShr`。审计同时计算相反方向，报告与 Retx 的吻合数量；不能把不同 CRSP 工具中的同名近似因子公式直接移植过来。价格因子和股数因子可能不同，非普通分配也可能使价格比值不能完整重建收益；相关差异只记异常，不篡改供应商收益。

用于信号日 t 的历史窗口时，以 t 当日为基准：`price_s_on_t_basis = price_s × factor_price_t / factor_price_s`；`volume_s_on_t_basis = volume_s × factor_share_s / factor_share_t`。只能使用 s ≤ t；不把基于样本终点的复权绝对价格、终点因子或未来事件本身放入模型。收益类特征优先使用已核验的 DlyRet/Retx 连乘。

交易价格、5 美元门槛及流动性排序使用当日未复权价格和原始股数；不能将复权价格乘以原始成交量。累计因子缺失或非正时不自行补值。原始数据始终保留。

## 退市和收益缺失

WRDS 的 [CIZ 说明，第 13–15 页](https://wrds-www.wharton.upenn.edu/documents/2084/Webinar.pdf)将退市收益放在主日频序列，通常在退市后的下一交易日出现 Y 行。因此本项目**不再额外合并并乘上 legacy DLRET**；也不在筛完活跃普通股之后才保存收益，否则会丢失退市事件。

保留 Y 行的原始收益、价格标记和缺失原因。普通日有缺失或多期收益时，不能无声填零，也不能当作一个完整单日标签。未来标签生成须先标记其实际持有路径，再判断能否计算；不能依据未来缺失或退市从当期股票池中删股。

实际数据中发现 `MV`（公司行动金额缺失）同时带有数值收益的记录。数值存在不等于质量标记正常：原值保留并隔离为待核查收益，不自动视为可用、不改成零；该情况不妨碍仅依赖历史价格与流动性的股票池生成。

现有导出不含 `DelPaymentType/DelStatusType`、完整支付日期或非普通分配金额。因此可以核实“收益已在序列内”的口径，但不能独立重建每笔公司行动、假定所有退市都立即收到可交易现金。审计列出缺失退市收益及持有期相关异常；回测前须针对实际触及的事件补查，或明确标注结算近似与敏感性分析。

## 月度股票池的可执行定义

规则参数集中在 `config/universe.json`。月末形成，下月第一个交易日生效；只使用形成日或更早的信息。

1. 以月末当日历史分类筛选美国注册普通股，交易所 N/A/Q；当日 RW/A、非退市、真实成交收盘价 ≥ 5 美元。
2. “252 日历史”具体定义为自 2014 年数据起点起累计至少 252 个 RW/A、非退市且有正的有效成交价或报价均值的观测日。它是保守的数据可用性要求，不宣称 IPO 日期。早于 2014 年的历史不计入。
3. 在全市场最近 60 个交易日（含形成日）内，至少 50 日具有 RW/A、非退市、TR、正收盘价和非负成交量；对这些有效日的 `Close × Vol` 求平均。零成交量保留为零，缺失日不填零，不用更早的观测凑足 60 日。
4. 按平均成交额降序、PERMNO 升序打破并列；每个 PERMCO 只保留第一只，再取前 500。保留全部月末候选与逐层筛选数量以便追溯。
5. 月内不因事后表现更换名单；实际信号日和成交日仍须重新检查交易状态。股票池文件不是成交指令，也不是已经筛除所有未来缺失收益的模型样本。

这是当前 CRSP 数据版本中的历史有效日期重建，不是逐日存档的原始发布版本；不能声称完全消除了供应商事后修订。累计因子只用于相对窗口处理，不影响股票池筛选。

## 数据质量与重现

原件 SHA-256 必须与下载验证一致；先删除全字段完全相同的重复导出行，冲突主键则隔离并中止。所有证券和退市收益保留于 `data/interim/crsp_audit_v1/part-*.parquet`；这是类型规范化后的来源层，不代表每条记录均适用于模型。

依次执行 `scripts/audit_crsp.py`、`scripts/build_universe.py`；合成边界测试使用 `python -m pytest tests/test_crsp_pipeline.py -q`。数据、股票名单和异常行均留在 Git 忽略的 `data/`，代码与不含原始记录的报告可独立审阅。
