# 2026 全年未见评价交接

组合采用 V3，主要对照为 V1 原始 Ridge。两者的配对差单列；原协议中的其他基线、事后最佳固定基线比较、主辅训练截止日和评价指标继续报告。历史研究冻结文件见 [final_v3.json](../../config/final_v3.json)，前瞻入口及依赖另见 [unseen_entry_v3.json](../../config/unseen_entry_v3.json)。

## 当前验收范围

2025 年实测从规范化 CRSP 日表重建股票池、原始及背景特征、严格五日标签。按旧档至 2024 年底、同一档案的 2025 行模拟新分片，12 个月的 6,000 条成员记录（含公司、形成日与流动性名次）、48 个完整信号日的 24,000 个键、资格与行业均吻合。33 个特征最大绝对差为 0，含缺失的标签状态、终点和收益逐行精确一致。最后标签终点为 2025-12-24。此核验不生成策略表现，也无法预判真实新旧版本的修订情况。复算入口为 `python scripts/verify_unseen_inputs_2025.py`。

## 发布与预约

1. 在仓库根目录用项目虚拟环境运行 `python scripts/unseen_guard_v3.py`，只检查本机封存文件。封存覆盖三个入口、输入构建器、审计记录、相关模块和配置、已有模型清单等。原始模型与证券数据保留本地。
2. 正式发布时，先把已审查的入口源码、当前封存 JSON 和依赖代码提交到代码仓库。以该提交创建标签 `git tag v3-unseen-entry-v1 <已审查提交>`，再各执行一次 `git branch audit/v3-unseen-2026-prediction v3-unseen-entry-v1` 和 `git branch audit/v3-unseen-2026-portfolio v3-unseen-entry-v1`。只在首次部署时建立，保留既有引用和历史。运行 `python scripts/unseen_guard_v3.py --released` 可检查发布状态，无需预约。本轮只在临时测试仓库演练发布。
3. 正式入口先检查本机封存与 Git 发布标签一致，然后在固定审计分支追加预约提交，同时创建首次预约标签，使用原子引用更新与独占锁。预测和组合各有一份记录。删除输出目录不会恢复额度，删除审计分支会使入口关闭，将分支重置到起始提交仍会被预约标签拦截。单机 Git 无法抵抗管理员同时改写或移除所有证据；第三方时间证明需要独立保管这些引用。

## 数据与评价

1. 用同一 CRSP 产品取得新档及 2025 重叠段。构建器输入为与现有 ingest 档案相同的规范化、键唯一的 CRSP Parquet 分片，日期列为时间戳，日历列为 ISO 日期字符串和 `vwretd`。应先获得可独立投影的 2025 重叠分区。运行 `python scripts/check_v3_vintage.py --new-parquet <仅含行情分片的文件或数据集>`，该入口在读取层限定 2025 年。核对键、收益与标志差异，记录处理决定。若复权、股票池或收益口径尚未对齐，先完成版本核验。
2. 建立本地预测清单。字段为 `cohort`（`CRSP_US_equities_2026`）、`data_through`（`2026-12-31`）、`overlap_sha256`、`vintage_decision`（`old_through_2025_new_from_2026`）、`decision_reason`、`parts` 及 `new_calendar`。分片数组和日历项均以相对清单的 `path` 和 `sha256` 指定。分片合计须包含与已审阅重叠段一致的 2025 投影和完整的 2026。入口检查重叠投影指纹，历史至 2025 年使用原档。
3. 运行 `python scripts/unseen_v3.py --audited-manifest <预测清单>`。入口构造月末股票池和次月信号截面，评分使用信号日及之前的特征；随后把两套模型的分数、特征、股票池和日历的哈希写入 Git 记录。只有在记录开标签事件之后，程序才调用严格五日标签构建器并计算 IC。评估输入不再接受由调用者预制的特征表和标签表。
4. 2026 年事件和结算核验通过后，运行 `python scripts/unseen_portfolios_v3.py --audited-manifest <执行清单>`。执行清单含 `prediction_manifest` 和 `execution_audit`（路径/哈希）。核验文件含 `unresolved_events`、`settlement_verified`、`market_verified` 和 `execution_market`、`settlement_calendar`、`event_registry` 三份路径/哈希；按原规则完成实质核验并保留证据。组合端先预约，再读取行情，校验预测阶段的 Git 完成记录与文件哈希。V3 与 V1 Ridge 的年化净收益配对差另存为 `v3_minus_raw_ridge.json`。公司行动或日历未能核实时，仅发布预测评价。

## 故障恢复

使用 `git log --format=%B audit/v3-unseen-2026-prediction` 或相应 portfolio 分支查看事件。系统记录失败阶段、异常类型和代码位置；详细诊断留在本机。进程硬退出时最后一次已持久化事件仍在。完成标签前的预测检查点后，恢复只读原预测；已成功核验的组合路径也按哈希复用。对同一清单显式加 `--resume`，任何输入、封存或检查点哈希冲突都会阻断。文件损坏时只可恢复字节一致的备份。若需要改变已封存的算法或输入定义，保留原记录，单独审查修订，不覆盖原版本、不当作同一评价。完整状态以审计分支为准，证券级文件留在 `outputs/`。

未见数据到达之前，只有代码和历史/合成输入得到核验。真正的版本差异、全年股票池、公司行动、结算和策略表现仍待验收。
