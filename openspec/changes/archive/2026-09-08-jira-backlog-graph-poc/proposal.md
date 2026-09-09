## Why

BA/PM 需要理解产品 backlog 的 Epic 全景和明确依赖；现有关键词、向量及 Jira 精确关联检索不能保证完整收集层级和关系链。本 PoC 以 AIPLAT 验证平台级图检索能力，用轻量演示页展示效果，供上层应用复用。

## What Changes

- 从 Jira 只读分页采集项目快照，建立 Issue 节点、父子关系和明确的 issue links，记录来源与可见范围。
- 新增项目概览、Epic 下钻、最多两跳依赖查询和关系证据检索接口。
- 保留 Chroma 与 SQLite FTS；按需要定位起始事项，再沿图取回关联文本。结构统计由程序计算。
- 新增默认关闭的单页演示：项目全景 → Epic 下钻 → 关系取证；无多轮会话及业务编排。
- 建立真实快照验收及结构化查询、现有 hybrid、图扩展三组对照。
- 不包含 LLM 关系抽取、社区摘要、自动优先级、延期预测、Jira 写回或生产部署。

## Capabilities

### New Capabilities

- `jira-graph-snapshot`: 项目快照、图模型、版本一致性和只读重建。
- `backlog-graph-retrieval`: 层级统计、依赖遍历、文本证据及平台 API。
- `backlog-graph-demo`: 无会话的轻量可视化演示与能力验收。

### Modified Capabilities

无。当前仓库无已有 OpenSpec specs；现有接口默认行为保持兼容。

## Impact

- 预期涉及 Jira connector、独立 graph 模块、新 API router、配置、评估与演示静态资源。
- 建议 PoC 使用独立本地 Neo4j，作为可替换 GraphStore 的实现；新增依赖仅在启用图能力时使用。此项是提议中的技术选择，尚未部署。
- 使用现有 Jira 凭据；不将凭据传给页面或记录到快照。
- 2026-09-08 只读探索观察到当前账号可见 35 个事项：8 Epic、26 Story、1 Bug，27 个子事项均有 Epic parent；唯一已观察到的 Blocks 边为 AIPLAT-46 → AIPLAT-16，两者均 Done。该观察不是持久化评估快照，也不保证未来数量不变。
- 数据足以演示层级及一条真实依赖，不支持声称已验证复杂依赖网络或检索质量提升。
