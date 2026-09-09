# Jira Graph PoC 执行控制与准备记录

## 执行约定

- 产品范围以 openspec/changes/jira-backlog-graph-poc 为准，不更改目标文档。
- P0 准备阶段仅检查环境、基线与分段安排；后续按用户逐个批准的关键 checkpoint 实施。
- 每段最多 50 分钟，到时收尾；1 小时为绝对上限。提前完成立即交付，不消耗剩余预算。
- 每段记录开始/结束时间、变更文件、TDD 红绿证据、验证结果和未完成项。剩余不足以完成验证时不接新任务。
- 可委派明确文件边界的独立子任务；子代理不能转派、扩范围或越过当前截止时间。主代理负责整合与审查。
- 每个确认点全部停止，用户明确确认后才继续；环境排障超出当前段时先报告再重估。
- 用户于 P1 后调整：普通执行段完成后自主继续，仅在关键 checkpoint（数据就绪、结构 API、组合检索、演示、最终验收）停止确认。单段时间上限保持不变。
- 实施采用 TDD；页面以交互验收为主。提交、部署与 Jira 写回不自动进行。

## 分段安排

| 段 | 预算分钟 | 对应 OpenSpec tasks | 可审阅交付物 |
|---|---:|---|---|
| P0 当前准备 | 45 | 不勾选实施任务 | 环境、基线、控制记录；硬停止 |
| P1 环境与合约 | 45 | 1.1–1.3 | 隔离工作区/测试环境、Neo4j 固定版本启动配置、模型与开关测试；硬停止 |
| P2 采集与规范化 | 45 | 2.1、2.2、2.5部分 | 分页、方向、去重、边界的红绿测试 |
| P3 图与文本索引 | 50 | 2.3、2.5部分 | 三个隔离索引的集成测试 |
| P4 快照发布 | 45 | 2.4、2.5、6.1 | 原子发布及失败测试、真实数据核验；硬停止 |
| P5 结构接口 | 45 | 3.1、3.4部分 | overview、分页 drilldown、确定性计数 |
| P6 依赖接口 | 45 | 3.2–3.4 | 两跳、环、截断、状态语义验证；硬停止 |
| P7 起点与扩展 | 50 | 4.1、4.2 | 显式 key 与 hybrid 种子串联 |
| P8 回答与故障 | 50 | 4.3、4.4 | 路径取证、引用、降级及真实问句验证；硬停止 |
| P9 演示页面 | 50 | 5.1–5.3 | 全景、下钻、依赖三个独立操作；硬停止 |
| P10 对照评估 | 50 | 6.2–6.4 | H/S/G 对照，真实与合成分开 |
| P11 最终验证 | 40 | 6.5 | lint、回归、操作说明与限制；硬停止 |

P1–P11 上限总计 515 分钟（约 8 小时 35 分钟），不包括确认等待。当前环境问题使原 6–9 小时估算偏向上沿；依赖安装/下载/修复超预算不顺延，停下重估。此表为执行控制，不代替各段开始前的具体 TDD 测试设计。

## 第一阶段观察

- 仓库 main，HEAD c8e24ea；无已跟踪代码改动，openspec/ 尚未跟踪。现有 worktree 仅主目录。
- 后续建议 codex/jira-backlog-graph-poc 独立 worktree；创建前读取 worktree 技能，妥善带入未跟踪方案，不复制默认索引或暴露 .env。
- Python 3.12 可用；Chroma 1.5.5、pytest 8.3.3 已安装。SQLite FTS 使用 Python 内置 sqlite3。
- Docker CLI 可用，Docker Desktop Linux 引擎不可达；Neo4j Python driver 未安装。
- ruff 0.8.0 的安装记录存在，但 python -m ruff 找不到 Scripts/ruff.exe，仓库质量门禁因此失败。
- 直接 pytest 首次结果 356 passed / 3 failed，失败涉及本地 .env 的 Jira URL 与 API key 污染测试预期。测试断言意外输出凭据，不在本文件记录；建议轮换凭据，后续测试显式使用测试配置。
- 未启动数据库、未下载镜像、未修改业务代码、未勾选任何实施任务。
- 显式测试配置（OPENAI_API_KEY=test-key，Jira/Confluence 连接值为空）复跑：359 passed、1 个 python_multipart 弃用提示，19.44 秒。确认三个初始失败来自环境配置；整体质量门禁仍因 ruff 缺失未通过。

## P1 执行记录（2026-09-08）

- 开始 11:57:40 UTC，验证结束 12:01:26 UTC，约 4 分钟，预算 45 分钟。提前交付并硬停止。
- 工作区：C:/SourceCode/ai-rag-service/.worktrees/jira-backlog-graph-poc，分支 codex/jira-backlog-graph-poc；后续实施以此工作区的任务记录为准。主目录保留原方案副本，未同步勾选。
- 一个子代理限定 app/graph 三个文件及 test_models.py，20 分钟截止、不转派；已完成并由主代理检查。主代理负责配置、环境与全量门禁。
- 新增 compose.graph.yaml、requirements-graph.txt、本地指南、内部 graph models/interfaces、15 项测试；修改 app/config.py 的开关/秘密字段及 .gitignore。未修改业务目标、未实现路由/采集/检索。
- .venv 复用系统应用依赖，单独安装 ruff 0.8.0、neo4j 5.28.2；全局 ruff 没有修改。工作区不含主目录 .env。
- Docker Desktop 已启动；Neo4j 5.26.12 只绑定本机 7474/7687，独立 volume，随机密码存入被忽略的 .env.graph。
- TDD：配置 3 failed/1 existing regression passed → 4 passed；模型 11 failed → 11 passed。红色失败对应缺少字段、校验及模型；默认关闭启动测试为已有行为的回归保护。
- 质量命令：虚拟环境 Scripts 加入当前进程 PATH，OPENAI_API_KEY=test-key 后执行 powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1；lint passed，374 passed / 1 已有弃用提示，37.39 秒。
- Docker compose config --quiet 通过；首次连通测试发生于启动完成前，失败为 ServiceUnavailable；日志 Started 后认证 RETURN 1 返回 1。尚未导入任何 Jira 数据。
- OpenSpec 1.1、1.3 完成。1.2 仅内部协议/模型完成，端点请求响应模型保留待后续 API 段补齐，不虚报完成。
- 下一段 P2：45 分钟，分页采集与关系规范化的 TDD；不能进入 P3 三索引构建。等待用户确认。

## P2–P4 数据 checkpoint（2026-09-08）

- 用户已确认普通段可自主衔接，只停关键 checkpoint；因此本轮覆盖采集、构建与发布，未进入 P5 查询 API。
- 12:22:45 UTC 开始；P2 采集及 P3 存储独立子任务在约 5 分钟内交付；主代理同步处理文本索引和 P4 发布，12:34 UTC 完成最终验证。本轮总计约 12 分钟，所有执行段低于 45–50 分钟上限。
- 三个有界子任务：采集（25 分钟预算）、图存储（25 分钟预算）、只读审查（10 分钟预算）。均已结束，无转派和剩余后台任务。
- 变更集中 app/graph/jira.py、store.py、text_index.py、snapshots.py、embedding.py、build.py、cli.py 及对应测试；更新本地指南/本记录/任务状态，不改产品目标。
- 红绿证据：采集初始缺失及畸形字段失败后 12 passed；图存储初始 8 failed，加入回滚/不同未知关系/超时后 12 passed（含真实 Neo4j）。文本索引初始 7 failed → 7 passed；发布初始 7 failed → 7 passed；协调构建初始 3 failed，解决 Windows 路径限制后通过。嵌入 adapter 与 CLI 各先 3 failed 再通过。最终审查针对发布状态/旧采集覆盖/资源清理各先失败再修复。
- 审查发现并修复：未知关系去重身份不一致、旧采集反向覆盖新快照、发布后报告失败误报；Windows Chroma 长路径问题经实际失败复现后缩短目录名。
- 两次真实构建完成，35 Issue、28 Edge、99 chunks；两份三索引内容独立核验一致，旧版本保留。当前快照 7cc2704c483c423c9bc1d749040bbd64。
- 最终质量门禁：lint passed，424 passed/1 已有提示，53.22 秒；独立索引核验通过。详见 docs/evaluation/jira-graph-data-checkpoint.md。
- OpenSpec 进度 8/24；1.2 API 合约待 P5，3.x 查询、4.x 组合检索、5.x 页面未实施。停在数据 checkpoint，等待确认后继续 P5–P6 结构 API。

## P5–P6 结构 API checkpoint（2026-09-08）

- 13:04:31 UTC 开始，13:13 UTC 收尾，约 9 分钟；各段均低于预算。停在结构 API checkpoint。
- 有界子任务分别实现纯结构统计、图遍历，并进行独立只读审查；全部结束，无转派。主代理整合配置、服务、HTTP 合约并验证真实 API。
- 变更：structure.py、store.py、api_models.py、service.py、app/api/graph.py、config.py、main.py 及对应测试；仅更新实施文档和任务状态。
- TDD：结构缺模块 15 项失败后 15 passed；遍历初始 9 failed，扩展后连同已有存储/集成验证 24 passed；API 初始 fixture 缺模块错误后，独立启用路由断言先 RED，再完成实现后 20 passed。
- 完整门禁 lint passed、471 passed/1 已有提示（59.30 秒），真实 HTTP 与小样本延迟记录见 docs/evaluation/jira-graph-api-checkpoint.md。独立审查无待修复发现。
- 本地 API 隐藏进程本次启动 PID 29916，监听 127.0.0.1:8001；Neo4j 保持运行。PID 仅为本次记录，停止前应核实进程身份。
- OpenSpec 3.1–3.4 完成，累计 12/24。1.2 留待 retrieve/query 合约补齐。下一阶段 P7–P8：同快照 hybrid 起点 → 图关系 → 文本证据与引用，等待关键 checkpoint 确认。

## P7–P8 组合检索 checkpoint（2026-09-08）

- 13:35:21 UTC 开始，13:51 UTC 收尾，约 16 分钟；各段低于 50 分钟预算，没有扩展到页面或最终评估。
- 两个有界实施子任务分别负责快照搜索、回答适配器（各 25 分钟预算），第三个只读审查（15 分钟预算），全部结束，无转派。主代理负责合约、串联、HTTP、真实调用与文档。
- 新增 search.py、retrieval.py、answer.py 及测试；更新 api_models.py、service.py、interfaces.py、app/api/graph.py 和原 API 回归。业务目标文档未修改。
- 关键红绿及独立审查修复记录见 docs/evaluation/jira-graph-retrieval-checkpoint.md；全部审查发现已解决。
- 最终质量命令：将 .venv/Scripts 加入 PATH、OPENAI_API_KEY=test-key、GRAPH_INTEGRATION=1 后运行 powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1。lint passed，516 passed / 1 个已有弃用提示，48.77 秒。OpenSpec strict validate 和 git diff --check 通过。
- 实际无 key 问句经 hybrid 命中 AIPLAT-46 并沿 BLOCKS 到 AIPLAT-16；HTTP retrieve 200，query 模型失败仍 200 保留证据。模型 gpt-5-2025-08-07 返回 404 model_not_found；没有替换模型，真实成功生成未验收。
- 本地 API 继续监听 127.0.0.1:8001；凭据仅从已有源文件读入启动进程内存，不复制或输出。排障中发现本机 HTTP 代理需绕过，以及临时启动脚本需将工作区放在 Python import 路径首位；已修复本地启动并确认 OpenAPI title 为 RAG Service。
- OpenSpec 实施进度 17/24：1.2 与 4.1–4.4 已完成实现/故障测试；真实成功回答受模型可用性限制，6.x 最终质量评估未勾选。下一个 checkpoint 为 P9 轻量可视化，等待用户确认；不因模型问题自动更换产品范围。

## P9 演示 checkpoint（2026-09-08）

- 14:06:39 UTC 开始，14:22 UTC 收尾，约 15 分钟；低于 50 分钟预算。未进入 P10 对照评估。
- 一个 30 分钟预算的前端子任务；两个各 10 分钟预算的只读路由/UI 审查任务。均已完成，无转派。主代理负责静态路由 TDD、浏览器验收、整合修复与完整质量门禁。
- 新增 app/graph/demo.py、static/index.html/demo.css/demo.js、test_demo.py、test_demo_browser.py、可选 requirements-demo.txt；main.py 仅在两个开关都开启时挂载页面。默认配置未改变。
- 路由红绿 2 failed / 3 passed → 5 passed；浏览器发现并修复项目/锚点编辑残留、补齐 hybrid 起点提示，审查发现重试后错误残留并以失败测试复现后修复。最终 14 项浏览器验收全过。详见 docs/evaluation/jira-graph-demo-checkpoint.md。
- 最终命令：.venv/Scripts 加入 PATH，OPENAI_API_KEY=test-key、GRAPH_INTEGRATION=1、GRAPH_DEMO_BROWSER=1，运行 node --check app/graph/static/demo.js 与 powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1。JS syntax / lint passed；535 passed / 1 个已有弃用提示，67.83 秒。
- 真实浏览器业务描述命中 AIPLAT-46 并返回 AIPLAT-16 阻塞路径；该次未调用回答模型。生成页面成功/失败/重试分支以注入响应验证，当前真实模型 404 限制保持如实展示。
- 本地 GRAPH_DEMO_ENABLED 已显式开启，/graph/demo 可访问；Neo4j/API 保持运行。无提交、部署、Jira 写回或目标文档更改。
- 将此前本地启动路径排障产生的 chatgptmock.db（13:46 UTC 创建）移入被忽略的 data/graph-poc/bootstrap-artifacts 保留，不纳入工作成果。
- OpenSpec 5.1–5.3 完成，实施进度 20/24；6.2–6.5 对照评估/最终验收待下一 checkpoint。按用户约定在演示 checkpoint 停止确认。


## P9 panorama refinement checkpoint (2026-09-08)

- 14:52:25–15:02 UTC, approximately 10 minutes, within the 50-minute segment budget. Scope: user-requested multi-Epic panorama; no P10 evaluation work.
- One bounded UI subagent (25-minute maximum) completed; root integrated the projection, navigation, regression fixes and browser acceptance.
- Full real snapshot displayed: 8 Epics, 35 issues, 27 CHILD_OF and 1 BLOCKS. Zoom/Fit, background drag, node selection and same-snapshot drill-down available; original tables retained.
- Fresh validation: both JavaScript syntax checks and lint passed; GRAPH_INTEGRATION=1 and GRAPH_DEMO_BROWSER=1 full quality gate: 542 passed, one existing multipart deprecation warning, 66.53 seconds. Screenshot visually inspected. git diff --check passed.
- Local Neo4j/API remain running. No commit, deployment, Jira write or product goal document change. OpenSpec progress remains 20/24; pause for visual checkpoint feedback before evaluation.


## P10–P11 最终收尾（2026-09-08）

- 用户在全景验收后授权完成所有剩余任务、不再暂停；取代上文关键 checkpoint 等待规则。每段 50 分钟/绝对 1 小时上限不变。
- 本轮完成问题集、评估、审查和最终验证，15:30 UTC 收尾，单轮小于 30 分钟，P10/P11 均未超过各段预算。
- 三个有界子任务分别负责真实/合成标注（20 分钟）、评估 harness（25 分钟）、只读审查（10 分钟）；全部结束，无转派。主代理负责真实 API、模型探针、评估运行、报告和运维说明。
- 18 个真实问题、8 个合成场景说明；H/S/G 同快照、同 query embedding 与文本预算，54 个真实检索观测。计量不等同回答正确率；完整逐题证据和限制见 docs/evaluation/jira-graph-final-report.md。
- TDD：harness 初始 5 RED → GREEN；未返回关系不可补入指标的回归先 RED 后 GREEN；负例判定 2 RED → 12 tests GREEN。审查的负例判定、全节点字段校验、结果校验和、P95 分位计算均已修复。
- 真实 API 全节点/边/层级/状态/方向/失败保留证据 22/22；固定模型再次 404 model_not_found。真实回答人工评分、成功生成延迟及实际账单不可测，明确记录，不以模拟通过替代。
- 最终命令：JS syntax 两文件；python -m ruff check scripts/graph-acceptance.py scripts/evaluate-graph-poc.py；PATH 指向 .venv/Scripts，OPENAI_API_KEY=test-key，GRAPH_INTEGRATION=1，GRAPH_DEMO_BROWSER=1，执行 scripts/quality-check.ps1。全部通过，554 passed / 1 已有 multipart 弃用提示，69.77 秒。
- OpenSpec 24/24；6.4 包含已记录未测限制。完成归档就绪校验后归档并同步三个新增能力规格。无原始产品目标改写、提交、推送、部署或 Jira 写回。本地 API/Neo4j 保留运行。

归档结果：openspec/changes/archive/2026-09-08-jira-backlog-graph-poc；三个主规格新增 10 条要求，无修改/删除已有规格。openspec validate --all --strict：3 passed；active changes 为空。


## 2026-09-09 用户验收前收尾

统一交接文档：docs/evaluation/jira-graph-poc-handoff.md。包含本版范围、6 个待执行 BA/PM 问题、模型最新证据与已知限制；用户业务验收保持待完成。恢复未运行的 Docker/Neo4j/API 后，完整质量门禁 561 passed，143.02 秒；OpenSpec strict 3 passed。无功能扩展、提交、合并、推送或部署。
