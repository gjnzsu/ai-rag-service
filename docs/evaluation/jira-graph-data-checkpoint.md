# Jira Graph PoC 数据 checkpoint

日期：2026-09-08。仅验证数据构建及发布，尚未实现图查询 API、回答生成或可视化页面。

## 真实数据

当前快照 `7cc2704c483c423c9bc1d749040bbd64`，采集结束 12:33:14 UTC（北京时间 20:33:14），作用域为配置账号可见的 AIPLAT。

| 指标 | 结果 |
|---|---:|
| Issue | 35 |
| Epic / Story / Bug | 8 / 26 / 1 |
| CHILD_OF / BLOCKS | 27 / 1 |
| Chroma 与 SQLite FTS chunks | 各 99 |
| 文本关联文档 | 35 |
| 本次 embedding tokens | 6,415 |
| 本次采集时间 | 1.96 秒 |
| 本次索引构建（发布前） | 5.22 秒 |

本轮共执行两次真实构建验证完整重建，embedding 共 12,830 tokens；未调用回答模型，未计算货币成本。延迟是单次观测，不能作为 P95 或性能基准。

## AIPLAT-13 示例核验

| 子事项 | 类型 | 状态 |
|---|---|---|
| AIPLAT-14 | Story | Done |
| AIPLAT-15 | Story | To Do |
| AIPLAT-16 | Story | Done |
| AIPLAT-46 | Bug | Done |

记录的唯一阻塞边：AIPLAT-46 → BLOCKS → AIPLAT-16。两端均为 Done，不代表当前未解决阻塞。

## 验证证据

- 两次真实采集的 source_checksum 一致；分别生成独立 snapshot_id。
- 第二次发布后第一份 `42311f0317264cf88b831e502264d3ba` 仍可按版本读取。
- 对两份快照分别重读 Neo4j，逐条比较规范化节点及边；重读 Chroma/SQLite FTS，比较 chunk ID、document ID、正文及来源 URL，一致。
- TDD 覆盖分页失败、越界、关系方向/去重、未知关系、图事务回滚、索引缺失、向量响应异常、发布失败保留旧版本、旧采集不得覆盖新快照，以及发布后的报告/资源清理失败不能误报未发布。
- 完整仓库质量命令：虚拟环境 Scripts 加入 PATH，OPENAI_API_KEY=test-key、GRAPH_INTEGRATION=1，执行 `powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1`。
- 最终结果：lint 通过；424 passed，1 个已有 python_multipart 弃用提示，53.22 秒；包含本地 Neo4j 合成 scope 集成测试。

原始及规范化数据在被忽略的 data/graph-poc 下，不提交 Jira 正文或凭据。尚未测量语义起点召回、图 RAG 回答质量或多跳真实业务效果；只有一条真实 BLOCKS 边。
