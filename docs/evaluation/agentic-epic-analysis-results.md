# Agentic Epic analysis：首轮受控实验

## 实验问题与方法

验证有界单 Agent 能否基于自然语言问题选择必要的 Graph 补查，同时保留准确统计、方向、状态和证据。对照组是已经知道问题类型的固定流程，它也会跳过不必要查询；本实验没有使用每题都查全部依赖的弱基线。

- 固定快照：`f1ef094abab04f7a9a2c658839337a23`，可见项目 AIPLAT。
- [预先冻结的 12 个问题与 oracle](agentic-epic-questions.json)：AIPLAT-1、13、17、25，各三类问题。
- Agent 只收到自然语言问题和工具观察；oracle 仅用于基线路由和独立核对。
- 两组使用相同本地工具、快照、最终生成器、引用规则和预算。固定模型 `gpt-5.5-2026-04-23`，`reasoning_effort=low`，SDK 自动重试关闭。
- 每题每组 3 次，配对运行，第二轮调换两组顺序；最多三个配对任务并发。该结果反映此测试负载，不是独占性能基准。
- 结构查询没有 embedding、检索结果或答案缓存；提供方 prompt cache 未控制。最终测量在真实探针及首轮运行之后进行，数据库与运行环境已经预热，不能据此声称冷启动表现。

## 测量结果

| 指标（每组 36 次） | Agentic | 固定流程 |
| --- | ---: | ---: |
| 自动检查全通过 | 36/36 | 36/36 |
| 逻辑工具操作 | 57 | 57 |
| 底层 GraphService 调用 | 81 | 81 |
| 数据库查询 | 252 | 252 |
| 决策模型调用 | 57 | 0 |
| 最终生成调用 | 36 | 36 |
| 提供方报告 token 总量 | 127,239 | 68,913 |
| 用量未知的调用 | 0 | 0 |
| 端到端中位数 | 9.156 秒 | 6.976 秒 |
| 端到端 P95 | 15.796 秒 | 10.484 秒 |

完整的逐次执行计数、耗时、token、动作、检查结果及实现 SHA-256 见 [机器可读结果](agentic-epic-results.json)。token 为提供方报告值，不是美元费用；如果缺失，单独标记未知，不视为零成本。

## 逐题核对

| 问题 | Agentic / 基线通过 | 底层调用/请求 A / B | 耗时中位数 A / B（秒） |
| --- | --- | --- | --- |
| AIPLAT-1-completion | 3/3 / 3/3 | 1 / 1 | 7.203 / 5.500 |
| AIPLAT-1-blockers | 3/3 / 3/3 | 4 / 4 | 10.578 / 7.594 |
| AIPLAT-1-combined | 3/3 / 3/3 | 1 / 1 | 8.812 / 7.625 |
| AIPLAT-13-completion | 3/3 / 3/3 | 1 / 1 | 8.422 / 6.047 |
| AIPLAT-13-blockers | 3/3 / 3/3 | 4 / 4 | 10.547 / 8.593 |
| AIPLAT-13-combined | 3/3 / 3/3 | 2 / 2 | 10.109 / 8.984 |
| AIPLAT-17-completion | 3/3 / 3/3 | 1 / 1 | 8.328 / 8.078 |
| AIPLAT-17-blockers | 3/3 / 3/3 | 4 / 4 | 10.391 / 7.094 |
| AIPLAT-17-combined | 3/3 / 3/3 | 2 / 2 | 10.828 / 6.812 |
| AIPLAT-25-completion | 3/3 / 3/3 | 1 / 1 | 6.953 / 5.938 |
| AIPLAT-25-blockers | 3/3 / 3/3 | 4 / 4 | 9.609 / 6.906 |
| AIPLAT-25-combined | 3/3 / 3/3 | 2 / 2 | 8.453 / 6.703 |

核对内容包括 Story/Bug 口径、请求分析范围、必查起点、BLOCKS 方向、双方状态、关系分类、完整覆盖及必要底层调用数。引用检查核对 ID 与来源绑定；自然语言的语义支持另行逐份审阅，不能将 JSON Schema 或引用 ID 合法视为事实证明。

重要业务差异：

- AIPLAT-1：3/3 Story 已完成，综合问题的未完成 Story 集合为空，无需补查；“所有 Story 含已完成项”的问题仍须查询三项。
- AIPLAT-13：2/3 Story 已完成；Bug AIPLAT-46 与 Story AIPLAT-16 都为 Done，其 BLOCKS 保留为历史关系。只问未完成 Story 时不应查询已完成的 AIPLAT-16。
- AIPLAT-17：2/3 Story 已完成，没有直属 Bug，但 AIPLAT-37 → AIPLAT-23 的跨 Epic 同项目阻塞存在，不能因直属 Bug 为零而跳过依赖。
- AIPLAT-25：2/3 Story 已完成，Bug AIPLAT-47（To Do）→ Story AIPLAT-31（In Progress）属于双方已知未完成的已记录 BLOCKS。

## 实现与异常验证

采用原生 Python 协调循环及 OpenAI 严格结构化动作，无额外 Agent 框架。初始 Epic 清单强制读取，最多两次补查；工具代码固定范围、记录覆盖并核验 finding 谓词。检索结束后统一编号证据并仅生成一次报告。实现说明和架构图见 [使用指南](../guides/agentic-epic-analysis.md)。

新增 30 项测试覆盖分页与快照固定、无 Story、截断、未知状态、Done/Done 历史关系、跨 Epic 来源、重复关系合并、方向补查、禁止目标范围收缩、无效动作、越界、工具文本注入、模型/工具/生成故障、超时、可选 API、SDK 真实传输参数及未知 token 统计。

独立代码审查发现并已修复：部分重复方向不应阻止新方向补查；缺失 usage 不能作为已知零 token。另补充 Bug 数量聚合证据，使“零 Bug”也有直接引用依据。模型解释原始问题和最终文字语义验证仍属于明确记录的限制。

最终验证命令：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1
python -m ruff check app tests scripts/evaluate-agentic-epic.py
openspec validate agentic-rag-epic-analysis-poc --strict
```

质量门禁：**585 passed，25 skipped**；lint 和 OpenSpec strict 校验通过。保留一个既有 python_multipart 弃用提示。跳过项是未在单元门禁启用的集成/浏览器或可选组件测试；本次另用真实 Neo4j 和模型执行上述配对评测，不能声称全部可选集成测试都已执行。

2026-09-21 提交前复查曾出现旧启动测试的 30 秒子进程超时。调用栈定位到切块库在应用导入阶段连带加载 PyTorch；已将该库延迟到实际切块时导入，并加强启动测试，禁止提前加载切块依赖。该修正不改变 Agentic 决策、检索或生成逻辑，先前实验指标仍保留为当时的测量结果。

## 结论与边界

本次 72 份报告均通过自动核对；其中 62 份由独立审查助手、其余 10 份由实现助手逐份复核，未发现错误计数、颠倒关系、虚假完整性、主观评级或无依据预测；这不代替用户验收。两组必要检索次数相同，Agent 额外消耗 57 次决策调用，token 总量增加 84.6%，端到端中位数增加 31.2%。因此本轮没有观察到优于固定流程的质量或检索效率收益。

该 POC 证明了服务内部按自然语言选择补查的可行性。对于本实验的固定、已知三类问题，保留确定性流程更容易控制成本；Agentic 作为默认关闭的实验入口继续保留，用于后续更开放的问题研究。不能将四个小型 Epic 的结果外推为大型、复杂或跨权限项目的优势。

本实现使用部署级项目白名单，不提供用户级 Jira ACL。它只分析快照中的直属事项及单跳已记录关系，不能预测延期、给健康评级或证明实际因果。覆盖账本只能验证模型已解释的分析范围，不能自动证明模型没有漏解用户意图。网络/SDK 超时不是进程级硬截止。

原始响应保留于本地忽略提交目录 `data/graph-poc/agentic-evaluation-final/`；早期探针及修正前首轮位于 `data/graph-poc/agentic-evaluation/`，没有混入正式指标。探针期间遇到未运行的 Neo4j、旧 SDK 参数兼容和 Graph 测试凭据覆盖问题，均已修正后重跑；正式结果不将这些调试尝试当作成功调用。

2026-09-21：用户在收到本地启动与验收步骤后确认“我已经测试通过了”。据此记录本轮用户验收通过；用户未提供逐项测试日志，不额外推断测试覆盖范围。OpenSpec 任务已全部完成，2026-09-21 已[归档](../../openspec/changes/archive/2026-09-21-agentic-rag-epic-analysis-poc/design.md)并同步[正式规格](../../openspec/specs/agentic-epic-analysis/spec.md)；提交与推送尚未执行。
