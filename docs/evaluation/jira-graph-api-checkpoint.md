# 结构 API checkpoint — 2026-09-08

冻结快照：`7cc2704c483c423c9bc1d749040bbd64`。本阶段查询只使用本地已发布快照，不调用 Jira、embedding 或回答模型。

## 真实 HTTP 验证

- overview：35 事项、8 Epic；Epic 8、Story 26、Bug 1；Done 18、To Do 10、In Progress 7。
- AIPLAT-13：直接子事项 4；limit=2 时返回 2 条且 total=4、has_more=true。
- AIPLAT-46 outbound：保留指向 AIPLAT-16 的 BLOCKS 路径，双方 Done；unresolved_pair=true 时无路径。
- 未知项目与事项 404；hops=3 返回 422。
- 每请求固定快照；单元测试覆盖显式快照、缺失图数据 503、方向、两跳、环、节点和路径预算、作用域与默认关闭路由。

## 验证证据

完整质量门禁：lint passed，471 passed，1 个已有 python_multipart 弃用提示，59.30 秒；启用真实 Neo4j 集成测试。独立只读审查无待修复问题。

本机 HTTP 小样本：每端点预热 1 次、串行 10 次。overview p50/p95 40.12/65.42 ms，epic_detail 33.31/54.29 ms，dependencies 60.84/99.49 ms。p95 采用 nearest rank；原始汇总见 jira-graph-api-timings.json。这不是压力测试或性能承诺。

已实现三个 GET 端点：overview、epics/{epic_key}、issues/{issue_key}/dependencies，均位于 /graph/projects/{project_key} 下。尚无 /graph/retrieve、/graph/query 和演示页；稀疏真实数据的两跳/环覆盖来自合成 fixture，不冒充真实关系。
