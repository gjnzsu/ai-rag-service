# Graph RAG PoC 用户验收与正式收尾

日期：2026-09-10。状态：用户已确认全部测试完成，并授权正式收尾。

## 验收依据

用户按已约定的六类 BA/PM 场景进行本地测试：产品全局、Epic 工作进展、明确依赖、历史关系、语义定位、解释边界。随后明确表示“我的测试已经全部完成了”和“可以正式收尾了”。这是用户对本次 PoC 的整体验收确认，不是独立逐题量化评分。

会话中有截图直接佐证的代表案例：

1. 在 Business description 输入 `Publish approved model to platform catalog`，没有输入 ticket key；Hybrid search 的起点包含 AIPLAT-23。
2. 将 Direction 从 Outbound 改为 Both 后，返回 AIPLAT-37 → BLOCKS → AIPLAT-23，两端均为 To Do。
3. Generate grounded answer 返回 supported，回答同时引用事项事实、BLOCKS 关系和正文，并提供 Open source 链接。

其他场景依据用户整体验收确认记录，不补造未提供的逐题回答、耗时、评分或截图。截图中的 supported 是程序校验状态，不单独等同于语义正确率。

## 验收基线

- 项目：AIPLAT；snapshot_id：`7b9d816aa2c445c9919635aff149f705`。
- 快照包含 35 项：8 Epic、26 Story、1 Bug；27 CHILD_OF、2 BLOCKS。采集时间为 2026-09-09，验收期间未因查询自动刷新 Jira。
- 回答生成默认模型：`gpt-5.5-2026-04-23`。OpenAI reranker 默认模型也已统一，但 reranker 默认关闭，Graph 的 SnapshotSearch 不调用旧路径的可选 reranker。
- 实现与模型变更已进入 main：`159811a`（Graph PoC）、`070ad48`（统一架构与指南）、`0d1b636`（reranker 模型统一）。本验收记录属于后续文档同步，不表示它已自动提交。
- 两项 OpenSpec change 均已归档：原始 Graph PoC 24/24、Network enhancement 7/7。原始产品目标不变。

## 从用户测试补充的操作说明

从 AIPLAT-23 找“谁阻塞我”应使用 Inbound；从 AIPLAT-37 找“我阻塞谁”使用 Outbound。初次探索可使用 Both，结果中的箭头仍保持 Jira 记录的方向。

E1、E4、E5 是本次生成请求中分配的证据编号，可以分别指向图节点、边或文本片段。它们不是向量库 key，也不是跨请求稳定 ID。GraphAnswerer 在检索完成、调用 LLM 前分配编号；跨请求追踪应结合 snapshot_id 与事项 / chunk / 关系身份。

## 正式收尾后仍保留的限制

- 依赖结果尚未自动补齐双方 Epic 归属，相关复合问题可能证据不足。
- 尚无整套生成实验的独立量化人工评分、复杂真实多跳覆盖、负载测试或完整账单统计。
- 图与默认文本索引仍独立构建，同一 Jira 内容可能重复存储和计算 embedding。
- 生产部署、权限体系、增量同步和多轮业务闭环不在已验收范围。
- 回答偏长、包含额外关联内容，属于后续表达优化候选；没有在本轮增加实现任务。

## 文档审查处理

已同步 README 的统一架构图、模型默认值、Graph 入口与本地验收状态；更新交接清单、本地指南和端到端说明；为旧评估报告添加最新验收入口；更新可复用评估模板的模型默认值。

历史评估 JSON、旧模型探针、checkpoint 过程记录和已归档 OpenSpec 不改写成新结果。保留其原始实验时间、模型、快照与失败记录，通过本文件说明后续进展。

相关资料：[交接清单](jira-graph-poc-handoff.md) · [端到端说明](../guides/hybrid-graph-rag-end-to-end.md) · [生成探针](jira-graph-gpt55-validation.md) · [历史检索对照](jira-graph-final-report.md)。
