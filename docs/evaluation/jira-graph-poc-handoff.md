# Graph RAG PoC 收尾与 BA/PM 验收交接

状态：实现完成，等待用户最终业务验收。日期：2026-09-09。

## 本版交付

- 平台公共服务：Jira 只读采集、Neo4j 图存储、同快照 Chroma 与 SQLite FTS 索引；提供 overview、Epic detail、dependencies、`/graph/retrieve` 与 `/graph/query`。
- 检索链路：显式 issue key 或混合检索定位起点，沿明确关系扩展，汇集图与文本证据，并校验回答引用。
- 本地演示：多 Epic 全景、Grouped/Network 切换、拖动、缩放、邻接高亮、下钻及跨 Epic 依赖 AIPLAT-37 → AIPLAT-23。
- 回答模型：`gpt-5.5-2026-04-23`，真实生成与引用样例已通过。Embedding 和 reranker 配置保持原值。
- 原始 PoC 24/24 与 Network enhancement 7/7 任务均已归档；本轮不扩展产品目标。

## 演示基线

页面：http://127.0.0.1:8001/graph/demo

AIPLAT 当前演示快照：`7b9d816aa2c445c9919635aff149f705`，35 项（8 Epic、26 Story、1 Bug），27 CHILD_OF、2 BLOCKS。数量是该账号在快照时可见的数据；重建后以新快照为准。

## BA/PM 验收问题

以下为待执行的用户验收，不视为已通过。建议主演示顺序 1 → 3 → 5。

| # | 问题 | 验收重点 | 用户结果 |
| --- | --- | --- | --- |
| 1 | AIPLAT 有哪些 Epic？每个 Epic 下有多少 Story 和 Bug？ | 全景与 Jira 一致；统计不受 top-k 影响 | 待验证 |
| 2 | AIPLAT-17 下有哪些事项？哪些完成、哪些未完成？ | 下钻、状态、来源可追溯 | 待验证 |
| 3 | AIPLAT-37 阻塞了谁？双方目前什么状态？ | 37 BLOCKS 23；双方 To Do；引用真实关系 | 待验证 |
| 4 | AIPLAT-46 与 AIPLAT-16 有什么依赖？是否仍未完成？ | 46 BLOCKS 16；双方 Done；不误报为当前未完成阻塞 | 待验证 |
| 5 | 与“检索返回未经授权的知识内容”有关的 backlog 是哪项？有哪些依赖事项？ | 不输入 key 定位 46，并扩展到 16；核对检索证据 | 待验证 |
| 6 | 根据 backlog，为什么 AIPLAT-37 阻塞 AIPLAT-23？哪些明确记录、哪些无法确认？ | 区分正文事实、关系记录与推测；不足时克制回答 | 待验证 |

每题记录：实际答案、来源引用、事实是否正确、证据边界是否清楚、问题备注。支持的结构查询与预设意图需要在页面选择相应入口；本版不包含任意自然语言意图规划。

## 已知限制与后续候选

- 依赖检索尚未附带 Epic 归属证据，同时问依赖与双方 Epic 的复合问题可能拒答；可在全景观察跨 Epic 连接。自动补齐归属是后续候选，不是本次收尾新增任务。
- GPT-5.5 已完成三次真实问题调用，其中两次 supported，一次因缺少 Epic 归属证据返回 insufficient_evidence；尚未完成完整问题集生成对照、BA/PM 人工评分、负载测试或实际账单统计。
- 历史检索覆盖率不等于回答正确率。稀疏真实图不能证明复杂多跳效果或交付影响预测能力。
- 生产权限、增量同步、高并发、部署、多轮业务闭环不在本版范围。

## 验证与资料

- 最新生成验证：[GPT-5.5 报告](jira-graph-gpt55-validation.md)。
- 历史冻结检索评估：[原始最终报告](jira-graph-final-report.md)。其中 404 是旧模型当时结果，已被新模型验证补充；保留历史实验数据。
- 网状图：[Network checkpoint](jira-graph-network-checkpoint.md)。
- 启动、重建与回滚：[本地指南](../guides/jira-graph-poc-local.md)。

## 交付位置

工作分支：`codex/jira-backlog-graph-poc`。
工作目录：`C:/SourceCode/ai-rag-service/.worktrees/jira-backlog-graph-poc`。
用户已授权将本版提交并推送到 main；实际提交标识以 Git 历史为准。为保留本地演示数据和运行环境，worktree 暂时保留。部署不在本轮操作范围。


## 本轮收尾验证

启用 `GRAPH_INTEGRATION=1` 与 `GRAPH_DEMO_BROWSER=1`，执行 `powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1`：lint 通过，561 passed，1 个既有 multipart 弃用提示，143.02 秒。

另执行两个前端文件的 `node --check`、`python -m ruff check scripts/graph-acceptance.py scripts/evaluate-graph-poc.py`、`openspec validate --all --strict`（3 passed）和 `git diff --check`，均通过。

首次完整检查因 Docker/Neo4j 与演示 API 未运行，出现 7 failed、18 errors；恢复本地依赖后完整重跑通过。`/graph/demo` 与项目 overview 均返回 HTTP 200。未修改功能代码以绕过失败；本地服务保持运行供用户验收。
