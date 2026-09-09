## Context

FastAPI 当前使用 Chroma、SQLite FTS、RRF、可选 reranker 和独立 grounding 模块，提供 /retrieve 与 /query。Jira connector 尚未保留 parent/issue links，默认单次取 100 项。生命周期 API 的结果结构没有关系路径。

本次服务消费者是上层应用；BA/PM 是演示目标用户。AIPLAT 的实测结构见 proposal，不能将该时点数量作为每次同步的硬编码断言。

## Goals / Non-Goals

**Goals:** 单项目只读快照、准确层级统计、明确关系查询、关系与文本共同取证、三个无会话演示操作，以及可复现实验。

**Non-Goals:** Jira 写回、生产上线、多用户权限系统、多轮 agent、任意 Cypher、LLM 抽关系、全局社区分析和自动影响预测。PoC 仅在本地受控环境使用同一服务账号可见范围；不宣称提供逐用户 Jira 权限隔离。

## Decisions

### 1. 独立图存储与索引

提议采用本地 Neo4j，通过 GraphStore 接口隔离驱动，版本和镜像在实施时核实并固定。SQLite 邻接表更轻，但独立 Neo4j 能直接展示图存储和遍历，也便于后续扩展；本轮不同时实现两个后端。Chroma 和 SQLite FTS 继续负责语义、关键词定位，不保存关系真相。

图能力与演示分别由 GRAPH_ENABLED、GRAPH_DEMO_ENABLED 控制，默认 false；关闭时无需连接或导入可选图驱动。使用新增 /graph 路由，保留现有 /retrieve、/query 合约。后续统一检索模式另行演进。

### 2. 快照模型与一致性

Issue 业务标识为 site_id + Jira issue ID，key 为可变显示字段。存储节点键额外包含 snapshot_id；字段包含 key、type、title、status、status_category、project、source_url、updated_at 及 document_id。保存 CHILD_OF、BLOCKS 和显式 RELATED_TO；未知 link 类型保留源类型但不推导语义。每边记录 source_issue_id、源字段、link ID（若有）、方向、snapshot_id。双方返回的同一 link 去重。

先分页采集并保存本地受控原始快照，再规范化节点和边，解析 Jira 富文本为可检索文本。跨项目或不可见端点不继续抓取，仅记录边界诊断；不返回其正文或将它计入范围统计。Epic 无 parent 正常，其他无归属项单独计数，未知类型和子任务不静默丢弃。

manifest 包括 site、project、采集起止时间、可见账号范围标识（不含凭据）、源校验和、规范化版本、三个索引版本及阶段状态。分页失败不发布，不能把已获取部分称为完整；项目采集不等于 Jira 全站权限完整。API 非事务快照可能在采集间变化，记录窗口，发现版本变化时报告或重试。

graph、Chroma 和 FTS 使用同一 snapshot_id 的独立命名空间。全部就绪后原子切换本地 manifest 的 active 指针；每次请求固定解析一次 snapshot_id，旧快照保留到无请求使用。该方式避免假设三库分布式事务。首次失败无 active 快照则报 unavailable；重建失败继续使用上一份并显示时间。重建替换整个可见快照，不对缺失项推断真实删除原因；不采用第一版增量同步。现有 lifecycle 写入/删除不操作这些隔离快照。

### 3. 平台接口草案

所有请求使用配置允许的 site/project，返回 snapshot_id、captured_at、scope、completeness 和 diagnostics；不允许浏览器传任意 Jira URL。输入不拼接为 Cypher。

| 接口 | 输入 | 输出 |
|---|---|---|
| GET /graph/projects/{project_key}/overview | 可选 snapshot_id | Epic、按类型及状态分类的计数、未归属/其他事项统计 |
| GET /graph/projects/{project_key}/epics/{epic_key} | snapshot_id、分页参数 | Epic、子事项、全量计数、分页状态 |
| GET /graph/projects/{project_key}/issues/{issue_key}/dependencies | snapshot_id、direction、hops=1..2、limit<=100 | 节点、类型化边、路径、截断标识 |
| POST /graph/retrieve | project_key、snapshot_id、query、可选 epic_key/issue_key、top_k<=20 | seeds、paths、原文 evidence、关系来源、覆盖状态 |
| POST /graph/query | 同 retrieve | 上述证据及可选生成回答、引用、拒答/降级状态 |

同步为本地运维命令，不提供公开写接口。未知范围/事项为 404，非法输入 422，图不可用时结构接口 503；空关系为成功空列表。层级分页不影响总数统计，依赖结果达限时明确 truncated。

### 4. 关系检索和生成

明确 key 直接定位；无 key 使用同一快照的现有 hybrid 作为种子检索。第一版使用模板化意图（overview、epic_detail、dependencies）及显式锚点，不做自由对话规划。未支持的意图返回说明，不猜测执行任意查询。

从锚点按关系类型、方向、项目边界作至多两跳有界扩展；Epic 依赖场景先取得其子事项，再计算这些事项的依赖，层级和依赖步数分开计量。cycles 使用 visited 集合终止。关系候选映射到同快照文档/chunks，再复用排序机制；保留路径中每条边的结构证据，不能被普通 top-k 文本排序丢失。文本预算不足以解释全部路径时报告部分覆盖。

统计和状态直接从结构字段确定，不让 LLM 计算。关系证据可以来自 Jira 字段，文本证据来自描述；字段事实与文字解释分别引用来源。层级接口不调用 LLM；演示摘要通过 /graph/query 复用现有生成器和引用校验，但扩展它们以接受结构事实证据。缺少正文仍可返回结构事实，不能编造原因。

BLOCKS 表示源数据记录的关系。PoC 的 unresolved_pair 定义为边两端 status_category 都不是 Done，只作为显式状态筛选，不能称为经过业务验证的交付风险；UI 同时显示双方状态。Done/Done 的 AIPLAT-46 → AIPLAT-16 仍显示记录关系，不报告当前阻塞。

图故障时仅 /graph/retrieve 和 /graph/query 可返回同快照 hybrid 文本降级，标记 graph_unavailable、无路径、非完整依赖回答；结构接口不使用文本猜图。生成失败仍返回证据并标记 answer_unavailable。无 active 快照不得混用 default collection。

### 5. 薄演示与评估

本地静态单页由 FastAPI 提供，默认项目 AIPLAT。三个操作独立：Epic 表格/树、详情面板、局部关系图及证据列表。无登录产品、会话历史、自动追问或修改 Jira。快照时间、状态、来源链接、空态、错误与截断都可见。页面数据来自平台 API；展示文本作安全转义。

真实快照固定后准备约 15–20 个问题，覆盖统计、归属、状态、现有阻塞和无答案；两跳、环、跨项目、权限边界与分页失败使用独立合成 fixture，并在报告中明确区分。三组固定语料、模型和上下文预算：H=现有 hybrid+exact；S=parent/filter 结构查询；G=图扩展+文本证据。

结构统计及已记录边与快照精确一致；Done/Done 误报当前阻塞为零；引用均可追溯。报告完整证据链召回、回答人工正确性、拒答、P50/P95、生成 token、建图耗时与成本。小数据只有一条依赖，不预设质量提升百分比，也不声称优于简单 parent 查询。运行前建议参考预算：本地热态结构 API P95<=1 秒，生成回答 P95<=10 秒；记录硬件/并发/重复次数，超预算报告原因，不伪造观测。

## Risks / Trade-offs

- 稀疏依赖 → 接受能力演示，复杂场景用标明合成的 fixture 测试。
- 三索引不一致 → 暂存快照全部就绪才发布，失败保留上次版本。
- 账号只见部分事项 → 明示可见范围，不等同全组织全量；上线权限另行设计。
- 关联语义误读 → 保留原始类型和方向，不将 RELATED_TO 当 BLOCKS。
- 新增数据库运维成本 → 本地独立实例、可选依赖、默认关闭，实施时记录启动与清理方法。
- GraphRAG 收益其实来自结构查询 → 增加 S 对照，分别报告全景展示和取证收益。

## Migration Plan

实现阶段固定依赖并启动本地 Neo4j，创建隔离快照，完成 API 与演示后运行验收。不开启 GKE 或公网部署。回滚关闭两个开关，现有 API 继续运行；仅清理显式选中的 PoC 命名空间，不能删除默认 Chroma/FTS 数据。本文仅方案，未执行迁移。

## Open Questions

- 后续生产消费者的权限、同步频率和运维预算未定义，均不阻塞本地单账号 PoC。
- 真正的两跳、跨 Epic 依赖需未来真实数据验证；本轮不人为修改 AIPLAT 补关系。
