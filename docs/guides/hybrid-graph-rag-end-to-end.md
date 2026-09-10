# Hybrid + Graph RAG：端到端流程说明

本文面向第一次使用服务的 BA/PM、平台调用方和新加入的开发人员，说明当前 PoC 如何从源数据构建索引、检索证据并生成回答。以仓库当前实现为准；页面是轻量演示入口，能力由平台 API 提供。

![统一架构图](../superpowers/specs/assets/hybrid-grounded-rag-poc-architecture.drawio.png)

[可编辑 Draw.io 源文件](../superpowers/specs/assets/hybrid-grounded-rag-poc-architecture.drawio) · [本地准备与运行](jira-graph-poc-local.md) · [BA/PM 验收与交接清单](../evaluation/jira-graph-poc-handoff.md)

## 1. 先理解三个阶段

| 阶段 | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| Indexing | Jira、Confluence、PDF；Graph PoC 当前采集 AIPLAT | 规范化、文本切块、向量化、建立文本与关系索引 | 可查询的数据版本 |
| Retrieval | 问题、过滤条件，或项目 / 意图 / 事项 key | Hybrid 找相关文本；Graph 进一步沿已记录关系找证据 | 文本片段、图路径、来源和覆盖情况 |
| Grounded answers | 问题及检索证据 | 生成带证据 ID 的回答，再校验引用 | 回答、来源链接和回答状态 |

索引构建发生在查询之前。查询读取已有索引，不会每次重新抓取 Jira。Jira 更新后，当前 PoC 需要显式重建快照才能看到新内容。

## 2. 新用户如何选择入口

| 想做的事 | 使用入口 | 是否调用回答模型 |
| --- | --- | --- |
| 看产品全景、Epic 明细、依赖结构 | `/graph/demo` 中对应操作，或 Graph GET API | 否 |
| 只拿普通文档检索证据，交给自己的 AI 应用回答 | `POST /retrieve` | 否 |
| 让服务根据普通文档证据回答 | `POST /query` | 是 |
| 获取图路径和相关文本，自行组织回答或可视化 | `POST /graph/retrieve` | 否 |
| 让服务根据图与文本证据回答 | `POST /graph/query` | 是 |

“没有调用回答模型”不等于“没有任何模型调用”：用自然语言做向量检索仍可能调用 embedding 模型。Graph 的显式 key 和确定性结构查询无需依赖语义匹配。

推荐第一次按顺序尝试：项目全景 → AIPLAT-17 下钻 → AIPLAT-37 的出向依赖 → 基于证据生成回答。Graph 请求的 `intent` 由页面或调用方选择，当前服务不使用 LLM 自动规划任意查询意图，也不自动在 Hybrid API 与 Graph API 之间路由。

## 3. Indexing：数据怎样进入服务

### 3.1 原有文档索引路径

Connector adapters 将源内容规范化为 `Document`，包括正文、标题、文档 ID、来源类型和元数据。Shared chunker 负责将长文档切成可检索片段，并生成对应的 chunk ID。

同一批 chunks 分别用于建立 Chroma 向量索引和 SQLite FTS5 文本索引。向量用于语义匹配，FTS5 用于词项匹配；BM25 排名在查询时计算。两条路径保留一致的 document ID、chunk ID 和来源，便于融合与引用。

### 3.2 Graph snapshot builder 为什么也连接 Shared chunker？

因为一条 Jira issue 同时包含两类信息，Graph PoC 需要保留两者。

| 信息 | Jira 来源 | 去向 | 用途 |
| --- | --- | --- | --- |
| 结构事实 | issue key、类型、状态、parent、issuelinks | Neo4j 节点与有向关系 | 回答属于谁、阻塞谁、路径方向是什么 |
| 文本证据 | summary 和 description | Shared chunker → Chroma / SQLite FTS5 | 根据业务描述找到事项，并提供解释所需的正文 |

图中标为 `text` 的箭头表示 **Jira 标题和描述正文进入文本索引流程**。它不是用户提问，不是 LLM 生成的图摘要，也不是把 Neo4j 的节点或边切碎。

实际调用链是：

```text
build_snapshot()
  → normalize()：产生 issues、edges、texts
  → SnapshotTextIndex.write(issues, texts)
  → Document(content=标题和描述, metadata=事项与快照身份)
  → chunk_documents([document])
  → embed(chunks) / Chroma / SQLite FTS5
```

架构图省略了 `SnapshotTextIndex` 这个适配层，以突出复用关系。Graph snapshot builder 是整个快照构建的协调者，因此也负责安排文本索引的构建。

例如，Neo4j 可以证明 `AIPLAT-37 BLOCKS AIPLAT-23`；但要根据“修改模型路由需要重新部署”这类业务描述找到 AIPLAT-37，或解释相关背景，就需要该 issue 的标题与描述。仅有 BLOCKS 边，不能证明阻塞原因或交付影响。

**Shared 表示复用切块代码，不表示共用数据库内容。** Graph 的 Chroma 数据目录和 SQLite 文件按快照隔离，Neo4j 通过 site / project / snapshot 作用域隔离；不会混入普通服务的默认集合。稳定 ID 用于关联事项与 chunks，快照 ID 则区分其版本；正文或切块配置变化后，不应假设片段内容仍然相同。

两条路径都调用 `chunk_documents()`，底层使用 `RecursiveCharacterTextSplitter`，按 `chunk_size` 和 `chunk_overlap` 配置切块。Graph 没有为节点或关系引入另一套文本切块算法。

如果同一条 Jira issue 已被原有索引采集，Graph 快照仍会独立切块、向量化和存储；目前没有跨两条链路的去重，因此可能产生重复存储和 embedding 费用。单次 Graph 查询只读自己的快照，不会因为这两份存储而自动返回双份结果。当前 Graph CLI 不采集 Confluence。后续若统一采集与 chunks，必须同时保留文档版本与图关系的一致性；这属于未来演进，尚未实施。

### 3.3 构建与发布顺序

1. 只读分页采集账号可见的 AIPLAT issues，记录采集起止时间。
2. 规范化节点、parent 与 issue links；保存原始记录和规范化记录。
3. 写入并核验该快照的 Neo4j 节点和关系。
4. 调用文本索引适配层切块，建立并核验 Chroma 与 SQLite FTS5。
5. 三个索引全部就绪后发布 manifest，并切换 active 快照。

当前实现按上述顺序执行；图中的分支表示数据去向，不表示三个索引一定并行构建。Active manifest 是完整构建的发布入口，不能理解为只要 Neo4j 写入成功就发布。构建失败保留旧 active 版本，失败构建不成为新查询的默认版本。

这里的 snapshot 是一次采集及索引发布的版本边界，不是 Jira 的事务级一致性快照。来源端在采集期间仍可能发生变化。默认 CLI 仅允许 AIPLAT，尚不是任意 Jira 项目的通用采集配置。

## 4. Retrieval：两条路径如何工作

### 4.1 Hybrid 检索

原有 Hybrid 路径提取问题中的事项 key 和过滤提示，获得精确匹配、向量相似度与关键词候选，再通过 RRF 按排名融合。之后可经过可选 reranker，以及证据筛选，再返回结果或进入回答生成。OpenAI reranker 与回答生成器的默认模型统一为 `gpt-5.5-2026-04-23`，但两者是独立请求、独立配置。设置 `RERANKER_PROVIDER=openai` 才启用该 reranker；默认仍为 `none`。已有环境若显式设置了旧 `RERANKER_OPENAI_MODEL`，需更新该变量并重启服务。

适合的例子是：“哪些文档提到了检索权限问题？”此时主要目标是找到相关内容，不要求输出类型化图路径。

### 4.2 Graph 检索

Graph API 首先解析项目及 manifest，将本次请求固定在一个 snapshot 上，再按请求意图处理：

- `overview`：按项目生成整体结构与统计，不需要先找某个语义起点。
- `epic_detail`：定位 Epic，读取其子事项；未直接给出 Epic 时，可以尝试从定位到的事项找到 parent Epic。
- `dependencies`：从起点沿指定方向和关系类型查找依赖，再补充关联事项的文本。

起点可以来自显式 `issue_key` / `epic_key`，也可以来自问题中的 key。没有明确 key 时，`SnapshotSearch` 在同一快照内执行语义和关键词检索，融合候选并映射回 issue。显式事项身份优先于语义猜测。

`SnapshotSearch` 复用 FTS5、RRF、chunk 等基础组件，但不会通过 HTTP 调用旧 `/retrieve`，也不经过旧路径的可选 reranker。Graph retrieval 会保留关系方向、来源和路径，而不仅是返回相似文本。

当前依赖查询最多两跳、100 个节点，并支持截断与覆盖诊断。`top_k` 控制种子文本检索数量，不是项目总事项数；确定性全景统计不能当作 top-k 检索结果来解释。

### 4.3 从业务描述到图证据

以“检索返回了未经授权的知识内容”为例：

1. SnapshotSearch 用问题文本定位相关事项，已有演示数据中目标为 AIPLAT-46。
2. Graph retrieval 以事项身份为起点，查询其记录的 BLOCKS 关系。
3. 返回 AIPLAT-46 → AIPLAT-16 的路径，以及两端状态和相关文本。
4. 调用方可以直接展示证据，也可以继续请求生成回答。

自然语言检索结果需要实际检查，以上是已有数据的演示路径，不保证任何改写或新快照都返回同一结果。

## 5. Grounded answers：证据怎样变成回答

对 `/graph/query`，GraphAnswerer 将图节点事实、边及来源、相关文本和适用的结构统计转换为带 ID 的证据。共享生成器将问题和证据发送给配置的回答模型，当前默认固定版本是 `gpt-5.5-2026-04-23`。

生成器要求严格 JSON 输出；CitationValidator 再核验引用是否属于本次提供的证据，并用服务端持有的来源 URL 构造引用。Graph API 返回检索结果和回答，调用方仍可检查原始证据。

| Graph 回答状态 | 含义 |
| --- | --- |
| `supported` | 回答通过当前引用校验，且检索未标记相关覆盖截断；不等同人工确认全部语义正确 |
| `partially_supported` | 引用通过，但证据存在部分覆盖或截断 |
| `insufficient_evidence` | 证据不足，不能据此作答 |
| `answer_unavailable` | 生成或引用校验失败，回答不可用；返回已有检索证据 |

不同 API 的响应结构和状态枚举并不完全相同。例如普通 `/query` 使用 `question`，Graph 请求使用 `query`，不要直接复用同一份请求体。

## 6. 可直接参考的 API 操作

以下示例假定已按本地指南启动服务并发布 AIPLAT 快照。新 clone 不包含凭据、数据库或采集数据；安装依赖与配置开关后仍需构建快照。

```powershell
$base = 'http://127.0.0.1:8001'
$overview = Invoke-RestMethod "$base/graph/projects/AIPLAT/overview"
$snapshot = $overview.snapshot_id

# 沿用全景的版本，避免切换快照造成前后不一致。
Invoke-RestMethod "$base/graph/projects/AIPLAT/epics/AIPLAT-17?snapshot_id=$snapshot"

$body = @{
    project_key = 'AIPLAT'
    query = 'Which issue does AIPLAT-37 block, and what are their statuses?'
    intent = 'dependencies'
    issue_key = 'AIPLAT-37'
    direction = 'outbound'
    hops = 1
    snapshot_id = $snapshot
} | ConvertTo-Json

# 仅检索证据。
$evidence = Invoke-RestMethod "$base/graph/retrieve" -Method Post -ContentType 'application/json' -Body $body

# 独立请求：再次检索，然后生成并校验回答。
$result = Invoke-RestMethod "$base/graph/query" -Method Post -ContentType 'application/json' -Body $body
$result.answer
```

检查 `snapshot_id`、`seed_method`、`data.nodes`、`data.paths`、`data.text_evidence`、`diagnostics` 和回答的引用。Graph 的生成请求不是把上一步 response 直接提交给模型的接口，而是按请求体重新执行检索。

如果想测试语义起点，移除 `issue_key`，将 `query` 改为不含 ticket key 的业务描述。`outbound` 表示从起点沿已规范化的关系方向查询；若查询“谁阻塞我”，应选 `inbound`。

## 7. 新开发人员从哪里读代码

| 组件 | 代码入口 | 重点 |
| --- | --- | --- |
| Graph 路由 | [app/api/graph.py](../../app/api/graph.py) | GET 结构查询、retrieve / query 的边界 |
| 请求合约 | [app/graph/api_models.py](../../app/graph/api_models.py) | intent、锚点、预算、快照参数 |
| Jira 采集与规范化 | [app/graph/jira.py](../../app/graph/jira.py) | 标题 / 描述提取与 link 方向 |
| 构建协调 | [app/graph/build.py](../../app/graph/build.py) | 三索引核验后发布 |
| 文本适配 | [app/graph/text_index.py](../../app/graph/text_index.py) | Document → 共用 chunker → 同快照索引 |
| Shared chunker | [app/pipeline/chunker.py](../../app/pipeline/chunker.py) | 切块、ID 与元数据 |
| 快照仓库 | [app/graph/snapshots.py](../../app/graph/snapshots.py) | manifest、active 版本及构建锁 |
| Graph 检索 | [app/graph/retrieval.py](../../app/graph/retrieval.py) | 起点、图扩展、正文证据、覆盖诊断 |
| 快照文本检索 | [app/graph/search.py](../../app/graph/search.py) | exact / vector / FTS / RRF |
| 图存储 | [app/graph/store.py](../../app/graph/store.py) | Neo4j 作用域、遍历和关系 |
| Graph 回答适配 | [app/graph/answer.py](../../app/graph/answer.py) | 图与文本证据转为引用输入 |
| 共享生成组件 | [app/grounding/generator.py](../../app/grounding/generator.py) | 严格输出及生成失败处理 |

建议阅读顺序：API 合约 → Graph retrieval → SnapshotSearch / GraphStore → GraphAnswerer；需要修改采集时再读 builder 和 text index。测试入口为 `tests/test_graph/` 与 `tests/test_grounding/`。不要把合成向量、模拟回答或浏览器注入响应的测试当作真实模型质量验证。

## 8. 理解结果时的边界

- 全景的“无 Epic 归属”不等于 Jira 的“未分配负责人”；两者是不同字段。
- Done → Done 的 BLOCKS 关系可能仍保留在 Jira。关系存在不代表当前还有未完成阻塞，需结合双方状态。
- 依赖检索当前不会自动附带双方 Epic 归属。复合问题可能返回证据不足，不能让模型补猜。
- 关系不存在于当前快照，只能说明未检索到该范围内的记录，不能证明真实世界没有依赖。
- 当前是只读、本地可演示的平台 PoC；生产权限、增量同步、高并发和多轮业务闭环不在已交付范围。
- 页面打不开先检查 API；结构查询失败检查 Neo4j、manifest 和 scope；语义定位失败检查同快照文本索引及 embedding；只有回答失败时再检查生成模型配置。完整启动与重建步骤见本地指南。

本说明依据源码与架构图整理，不构成新增一次实时 Jira / LLM 验收。真实生成样例与限制见 [GPT-5.5 验证报告](../evaluation/jira-graph-gpt55-validation.md)。


## 9. 用户验收与证据编号

本轮 PoC 已于 2026-09-10 获用户整体验收确认，代表案例与保留限制见 [验收记录](../evaluation/jira-graph-user-acceptance.md)。

E1、E2 等编号在每次 GraphAnswerer 整理证据时按顺序分配，发生在检索之后、生成回答之前。节点事实、关系事实和正文片段都可以获得编号。它不是向量库 key；相同证据在不同请求中可能获得不同编号。跨请求追踪需要使用 snapshot_id 以及事项、chunk 或关系身份。模型只引用证据编号，不创建原始事实。

不知道依赖方向时，先使用 Both；从 AIPLAT-23 找到阻塞它的 AIPLAT-37，使用 Inbound 或 Both。Both 不改变实际 BLOCKS 箭头方向。
