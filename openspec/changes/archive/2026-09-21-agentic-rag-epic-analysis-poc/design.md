## Context

本提案承接 explore 中确认的目标：ai-rag-service 仍为公共知识查询服务，Agentic RAG 是内部检索策略。首个场景为允许项目、已发布快照内任意 Epic 的 backlog 状态和已记录 Bug/阻塞分析。输入 Epic key，暂不让模型猜测目标 Epic。

GraphService 已提供分页 Epic 明细和单事项依赖查询；structure.py 提供全范围计数，但 children_counts 包含 Bug 等子类型，不能直接当 Story 完成率。GraphAnswerer 已提供来源适配及引用校验；GroundedAnswerGenerator 是一次性、无工具的模型调用，不能直接充当 Agent 循环。

现有快照可见范围基于配置账号和项目 allowlist，不等同于生产环境的逐用户 ACL。所有能力维持本机 PoC 边界，不扩大数据权限。

## Goals / Non-Goals

**Goals:**

- 同一公共入口处理任意有效 Epic；模型按问题和证据选择是否补查、查询对象和方向。
- 完整 Story 统计由代码计算，工具及模型执行均受限，结果可追溯至同一快照。
- 输出事实、证据支持的结论、分项覆盖情况和安全的执行摘要，部分失败保留已核实结果。
- 通过竞争力相当的固定工作流对照，验证 Agentic 决策是否值得额外成本。

**Non-Goals:**

- 主观健康评价、风险等级、延期预测、人员绩效判断、Jira 写回。
- 任意自然语言发现 Epic、跨项目查询、子任务递归统计、完整项目级 Defect 报告。
- 多 Agent 并行分析、完整计划先行的 Plan-and-Execute、跨会话记忆、MCP 或独立部署。
- 新增 Redis、语义缓存、回答缓存或改变已有接口的默认行为。

## Decisions

### 1. Public API and request scope

已实现 `POST /agentic/query`，通过默认关闭的 `AGENTIC_RAG_ENABLED` 控制，开启时要求 Graph 能力已启用。第一轮此入口只支持 Epic 分析，不宣称任意知识域。原 /query、/retrieve、/graph/* 保持原样。

请求包含 `project_key`、`epic_key`、`question` 和可选 `snapshot_id`。省略 snapshot_id 时入口解析一次 active，后续工具使用服务端固定的 scope。调用方不能通过工具参数重写 site、project、account scope 或 snapshot。未知 Epic、非 Epic 事项或不允许的项目在模型调用前拒绝；缺少可用快照时返回明确不可用。

内部上下文含站点及账号范围；对外只返回必要的项目、Epic、快照和时间，不暴露凭证、账号指纹。第一轮仅分析 Epic 的直接 Story 子事项；所有统计必须声明这一口径。

选择可选新入口而非接管旧接口，是为了保留确定性检索能力并单独测量 Agentic 路径。API 路径和配置名称已按本设计实现，并于 2026-09-21 通过本地用户验收；这不代表已经部署到生产环境。

### 2. A deterministic workflow around a bounded ReAct loop

顺序为：校验与固定快照；强制获取 backlog 和统计；模型选择补查或完成；协调层校验并执行；合并证据；生成并校验报告。

第一步 `get_epic_backlog` 固定执行，不消耗模型来猜是否需要基础数据。它仍计入总工具操作预算。模型只能选择 `get_issue_dependencies` 或 `finish`；不能重复触发初始读取、编写 Cypher、执行 Python 或选择任意 URL。

决策适配器以结构化动作或原生 tool calling 表达工具名和参数，独立于最终生成器。Jira 标题、正文和工具文本均作为不可信数据，不得覆盖系统规则或允许工具集合。每轮模型可以识别 completion、bugs、blockers 等请求分析项，并选择当前动作；unsupported 请求项必须明确记录，不得静默将复杂问题缩小为简单统计。

协调层维护请求、固定 scope、统计、证据集合、请求分析项、查询覆盖账本、已执行动作、预算和终止原因。每个已声明请求项必须有输出或限制；最终报告同时受原始问题约束，不能仅依据模型自行缩减的目标。语义目标提取仍需问题集与人工验收，schema 校验不代表理解一定正确。

每轮补查都会累积到同一次请求的状态中。后续决策调用继续基于原始问题、此前已取得的证据、最新工具结果和覆盖情况判断下一步，不是每轮重新开始。协调层保留完整的已取得证据；传给模型的上下文可以受预算限制，但必须保留来源标识和未覆盖范围，不能将上下文省略误认为已完成查询。

中间轮次用于选择工具、观察结果和决定是否继续，不逐轮生成独立的最终报告。模型提出 finish 只是请求结束；协调层根据实际覆盖账本计算 partial/complete，不接受模型声称“已查全”作为依据。

结束补查后，协调层合并、去重多次工具调用的证据，统一分配报告引用编号，再进行一次最终回答生成。这里禁止的是直接拼接多份独立答案及其各自的 E1/E2 编号，而不是禁止复用前一轮的结果。例如，一份独立答案的 E1 可能指 Story 统计，另一份的 E1 可能指 Bug 阻塞；直接拼接会造成引用冲突。统一证据集合和编号后，最终报告中的每个引用才能对应明确的来源。

不单独生成可执行长计划、不引入 Agent 框架或多个 Agent：当前工具集合较小，显式 Python 状态循环便于理解和验证。将来工具数量和恢复需求增加时，再评估框架。

### 3. Tool contracts

| 工具 | 输入 | 返回 | 约束 |
| --- | --- | --- | --- |
| `get_epic_backlog` | 入口校验的 Epic 和服务端 scope | Epic、分页收集的直接子事项、Story 专属统计、Epic 下 Bug、来源、覆盖信息 | 自动翻页；模型不能改 scope；空 Epic 完成率为 null |
| `get_issue_dependencies` | issue_keys、direction=inbound/outbound/both | 有向 BLOCKS 路径、两端状态和类型、关系来源、每个起点的完成/失败/截断状态 | 起点仅限本次 backlog 或已验证工具结果中的事项；一跳；只读；同项目同快照 |

业务工具是当前进程内的函数包装，复用 GraphService，不经 HTTP 调用自身。批量依赖工具内部可顺序调用现有单事项查询，不引入并行 Agent。单个起点失败仍返回其他成功结果，失败起点不得标为“无关系”。

Story 分母仅含直接子事项中的 Story。Done 使用 status_category=done；原始 status 和 category 同时保留。完整列表已获得时，完成率为 done_count/story_count；零 Story 用 null 并注明不适用。若列表未收全，返回观察到的数量及 partial，不输出冒充完整的完成率。

Bug 清单限定 Epic 直接子 Bug；在阻塞补查中出现的外部 Bug 单独标为“通过已记录关系关联”，不混入直属数量。工具优先保留原始 BLOCKS（包括 Done/Done），代码派生双方是否均为已知非 Done 状态。不能以 Epic 内没有 Bug 推断无外部阻塞。RELATED_TO 等其他关联和严重程度、负责人、历史趋势不在当前工具范围内。

### 4. Coverage and bounded execution

以下默认值于 2026-09-20 实施开始时冻结；首次对照按这些值运行，后续调整必须记录。

| 项目 | 首轮建议 |
| --- | --- |
| 逻辑工具操作 | 最多 3 次：1 次强制 backlog + 最多 2 次依赖补查 |
| 决策模型调用 | 最多 3 次；无效动作也计入，无隐式无限修复 |
| 最终回答调用 | 最多 1 次，独立计数 |
| 单次依赖批量 | 最多 20 个去重起点 |
| backlog 明细 | 页大小 100，最多读取 500 个直接子事项，触顶标 partial |
| 图结果预算 | 整个请求最多 100 个不同关系节点、1000 条路径 |
| 总执行时间 | 60 秒；为最终生成预留最多 15 秒 |

外部调用采用不超过剩余预算的超时，禁止预算外自动重试；底层数据库调用也应有有界事务/查询超时。去重动作按起点、方向、关系类型、scope 识别；无新证据的重复调用被拒绝，不增加覆盖。预算不足、无效动作、重复动作、决策模型失败均能终止循环并保留已验证数据。

覆盖账本按请求分析项记录 expected、checked、failed、truncated 和 not_requested。依赖项还记录起点集合、方向、关系类型和跳数。“未发现符合条件的阻塞”要求目标范围全部成功查询且未截断；未请求查询的状态是 not_requested，不是 complete 或 empty。

查询失败/截断时，已找到的一条合格 BLOCKS 足以支持“存在已记录阻塞”的局部正向结论；不能支持整个 Epic 的否定结论。Done/Done 保留为历史关系。unknown 状态不能当作已知未完成。

### 5. Public result contract

| 字段 | 含义 |
| --- | --- |
| subject / snapshot | 项目、Epic、数据版本、采集时间及统计口径 |
| facts | Story 状态分布、数量与完成率、未完成列表、分离的直属及关联 Bug |
| findings | `code`、`statement`、`scope`、`evidence_ids`、`limitations`；允许客观结论 |
| evidence / citations | 事项、关系及代码计算的聚合证据；由服务绑定来源 |
| coverage | completion、bugs、blockers 等分项的 complete/partial/unavailable/not_requested 及覆盖账本摘要 |
| execution | completed/partial/failed、stop_reason、工具和模型调用数、耗时、token 用量（不可得为 null） |
| answer | 基于事实和 findings 的引用报告，可为 null；另有独立 generation_status |

支持的 findings 例子包括 stories_not_all_done、recorded_unresolved_blocker_found、no_recorded_unresolved_blocker_in_checked_scope。统计/关系类 finding 的谓词由代码核验，模型负责表述；未能通过核验的结论不发布为事实。没有风险评分、置信度伪百分比或业务健康评级。

completed 表示用户请求的受支持分析项已覆盖，并不表示事项全部完成。回答生成失败不会抹去 facts/findings/evidence，generation_status 明确失败；执行标 partial。主体不可用时使用原有 404/422/503 语义，不制造空报告表示成功。查询途中预算到期仍可返回结构化结果，即使剩余时间不足以生成文字。

执行记录仅含工具、范围、参数摘要、状态和耗时，不返回模型隐藏推理、原始提示词、密钥或完整敏感正文。引用存在校验只能验证可追溯性，语义是否被证据支持仍需验收。

### 6. Evaluation against a competitive baseline

固定基线执行同样的完整 backlog 工具，并按预先标注的问题类型应用确定性规则：completion-only 不查依赖，blockers 查询相关未完成 Story，采用同样预算、工具和最终生成器。问题标签仅供基线路由和评分，不给 Agent 当检索提示。不能将无条件做所有查询的低效流程作为唯一对照。

两组使用相同快照、输入、模型版本、生成提示、引用规则、工具预算和缓存条件；Agent 的额外决策调用单独计费。先运行 12 个真实问题：AIPLAT-1、13、17、25 各包含完成统计、已记录阻塞、完成统计与 Bug 阻塞组合。人工冻结预期证据和所需工具范围后，每题每组至少运行 3 次，交替组顺序，记录首次/热调用条件。小样本报告逐题结果，不宣称统计显著性；真实费用不可得时仅报 token 和调用数。

- AIPLAT-1：3 个 Story 全 Done；完成率问题不需要补查。
- AIPLAT-13：包含 AIPLAT-46 → AIPLAT-16 的 Done/Done 关系；不能当作双方未完成阻塞。
- AIPLAT-17：AIPLAT-37 → AIPLAT-23 是跨 Epic 但同项目的关系；起点 Epic 内无 Bug 不妨碍发现阻塞。
- AIPLAT-25：3 个 Story 中 2 Done、1 In Progress；AIPLAT-47（Bug，To Do）BLOCKS AIPLAT-31。

另用合成 fixture 验证空 Epic、无关系、多页、预算截断、数据库/模型超时、非法或重复动作、未知状态、提示注入、生成失败、发布中途 active 变化。

正确性、证据支持、覆盖诚实是门槛：确定性测试全过；真实标注集不能接受错误计数、反向关系、无依据结论或虚假完整性。决策价值检查必要补查是否执行及无关动作是否减少；成本记录工具操作、底层查询数、模型请求、token、端到端耗时。报告必须允许“同等质量但更贵，暂无采用理由”，不预先承诺成功提升。

## Risks / Trade-offs

实施记录（2026-09-20）：采用 OpenAI SDK `chat.completions` 的严格 JSON Schema；在仓库现有 SDK 1.51.0 中通过 `extra_body` 传入 reasoning_effort=low。实际探针验证固定模型 gpt-5.5-2026-04-23 可返回受约束动作。代码位于 `app/agentic/`，默认关闭的入口为 `POST /agentic/query`，最终生成器拥有独立提示并复用 CitationValidator，不修改已有回答生成器。

覆盖去重按 `(issue_key, direction)` 进行：已查 inbound 后请求 both，只补查 outbound；完全重复才拒绝。执行摘要区分 decision_rounds 与真正的 decision_calls，使无决策模型的固定基线不被计为模型请求。provider 未返回的 token 保持未知。聚合证据包含完整/已观察 Bug 数量，使零 Bug 结论也可引用直接证据。具体配置、代码入口、预算的网络超时边界及评测复现见 `docs/guides/agentic-epic-analysis.md`。

- 模型漏掉用户子问题 → 返回请求项与覆盖清单，以原始问题和人工标注验收；不能只相信模型自报完成。
- 多轮调用变慢、变贵 → 分开统计决策和生成，冻结预算，与合理固定路由比较。
- 有关系被误写成根因或延期 → findings 限于可验证状态/关系谓词；无正文支持不推断原因。
- 图截断或分页上限导致错误否定 → 服务端覆盖账本限制结论，超限明确 partial。
- 工具结果中的指令污染 → 数据与系统指令隔离、严格工具 allowlist、服务端 scope 和参数校验。
- 当前样本稀疏 → 真实快照加故障/边界合成样例，避免推广为复杂项目效果。
- 当前生产 ACL 不完备 → 维持已有本机 PoC/账号可见范围，不能当成生产多租户能力。

## Migration Plan

无需数据库迁移或重新采集。新增默认关闭开关与路由；通过单元/集成测试和对照实验后，再在本地启用。关闭开关即可回到既有服务路径，不变更 active 快照、不删除任何索引。部署与提交推送不属于本次提案生成。

## Resolved Implementation Decisions（2026-09-21）

- API、预算与批量上限按上述冻结配置实现；真实模型探针与 72 次正式对照请求完成，结果见 [实验报告](../../../../docs/evaluation/agentic-epic-analysis-results.md)。
- 固定模型 gpt-5.5-2026-04-23 已验证支持严格 JSON Schema 动作；现有 SDK 通过 extra_body 传递 reasoning_effort=low。
- 独立 ReportGenerator 支持局部报告，复用 CitationValidator；旧 GroundedAnswerGenerator 提示和行为保持不变。
- 用户已于 2026-09-21 确认本地测试通过，21/21 任务完成。OpenSpec 已于 2026-09-21 归档并同步正式规格；尚未提交或推送本次变更。
