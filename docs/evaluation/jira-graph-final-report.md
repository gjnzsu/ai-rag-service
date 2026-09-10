# Jira Graph PoC 最终评估（2026-09-08）

> 最新状态：2026-09-10 用户已确认 PoC 测试完成，见 [用户验收记录](jira-graph-user-acceptance.md)。本文以下保留当时的实验结果。

> 2026-09-09 最新收尾与待办验收见 [PoC 交接记录](jira-graph-poc-handoff.md)；GPT-5.5 已打通真实生成，旧模型 404 为历史验证结果。

结论：已验证平台 API 可以返回 AIPLAT 的完整 Epic 结构，并从混合检索起点沿真实 BLOCKS 关系补齐证据。仅展示全景时，简单 parent/filter 查询已经足够；图扩展的额外价值是明确关系路径及来源。本次不声称已证明生成回答质量提升或生产可用。

## 数据与方法

- 冻结快照：`7cc2704c483c423c9bc1d749040bbd64`，35 项（8 Epic / 26 Story / 1 Bug）、27 CHILD_OF、1 BLOCKS、99 chunks。账号可见范围，不是 Jira 事务快照。
- [真实问题标注](jira-graph-questions.json)：18 题，从 normalized.json 独立建立预期证据；标注由实施代理完成，不冒充 BA/PM 人工评分。校验和随 [逐题结果](jira-graph-comparison.json) 保存。
- [合成场景](jira-graph-synthetic-cases.json)：8 类，覆盖两跳、环、范围边界、不可见关联、故障和预算；附实际测试定位，与真实问题分开。没有把合成关系写入 AIPLAT。
- H：当前同快照 SnapshotSearch 的 vector + SQLite FTS/RRF + exact-key 适配；不是对旧生产 /retrieve 的完整端到端基准，未启用 reranker。S：相同记录上的直接 parent/filter。G：现有 GraphRetrievalService + Neo4j 遍历 + 同快照文本。
- 固定 top_k=5 chunks、最多 100 节点、文本 16000 字符、embedding=text-embedding-3-small。H/S/G 使用完全相同问句，预期标注不进入检索。
- 文本预算相同；S/G 另外返回确定性结构事实。因此这是检索能力对照，不是总 LLM 上下文长度相等的生成实验。生成模型固定 gpt-5-2025-08-07，本次因不可用未启用生成。

## 观测结果

| 指标 | H | S | G |
|---|---:|---:|---:|
| 必需事项证据覆盖（17 题平均） | 45.1% | 88.2% | 100.0% |
| 结构关系覆盖（12 题平均） | 0.0% | 56.7% | 100.0% |
| 精确统计（7 题） | 0/7 | 7/7 | 7/7 |

覆盖率是所需证据是否出现，不是回答正确率，也不测量无关事项比例。H 没有输出类型化边，因此其结构关系覆盖为零不代表无法从正文理解关系。未知 key 题不进入空分母；S/G 的各一次 error 是预期的 not-found。

- 全景/明细：S/G 都正确返回 7/7 题的统计，不支持宣称 Neo4j 比简单父子查询更适合统计。
- R14 业务描述：H/S 只找到 AIPLAT-46；G 返回 AIPLAT-46 → BLOCKS → AIPLAT-16，证明这次种子到路径的串联效果。
- R15 业务描述：H 已找到两端事项；G 的增量是明确方向与关系来源，不能把两端命中都归功于图扩展。
- 四个无记录关系/已完成筛选题，G 的空 BLOCKS 检查 4/4；不存在 key 的拒绝检查 1/1。H/S 未执行 BLOCKS 遍历，不把其无边输出当成成功否定答案。
- R18：H 返回相似内容，S/G 拒绝未知 AIPLAT-999。此为锚点边界表现，不是 LLM 拒答率。
- 全景的正文超过 16000 字符时 S/G 均标记截断；结构计数仍精确。不能用节点覆盖 100% 推断所有解释正文也已覆盖。
- 真实图只有一条 BLOCKS，两端均 Done，且在同一 Epic 内。重复问法不是多个独立依赖样本；没有验证真实跨 Epic、复杂两跳或影响预测。

## 实际验收与时间

[真实 API 验收](jira-graph-final-acceptance.json) 22/22 通过：完整节点内容与边、类型统计、8 个 Epic 的子事项/状态、方向、Done pair 排除及生成失败保留证据。平台 API 访问的是本地真实 Neo4j。

同一 Windows 主机、16 逻辑 CPU、并发 1；各结构端点预热 1 次、连续测量 10 次，P95 使用 nearest-rank。

| 本地 HTTP 端点 | P50 ms | P95 ms |
|---|---:|---:|
| overview | 27.73 | 37.46 |
| detail | 25.11 | 38.25 |
| dependencies | 35.38 | 51.02 |

均低于计划中 1 秒的结构 API 参考预算；样本很小，不是并发/负载保证。H/S/G 的逐题热态计时见 JSON：各题各方法一次，embedding 与索引初始化预热在外；S 以内存过滤，G 调 Neo4j，不能直接用这组时间比较数据库性能。

三步演示：全景（8 Epic/35 节点）→ AIPLAT-13 下钻 → AIPLAT-46 到 AIPLAT-16 的关系/证据。浏览器验收同时覆盖独立入口、缩放、快照固定、键盘、窄屏、错误/空态和内容转义。

## 模型、费用与未测指标

- [实际模型探针](jira-graph-model-probe.json)：固定模型再次返回 404 model_not_found；真实 /graph/query 返回 answer_unavailable 并保留证据。没有自动替换模型。
- 成功生成次数、生成 token、真实回答人工正确率、LLM 拒答正确率、生成成功延迟 P50/P95：未测。合成生成器/浏览器注入响应只验证成功、引用、拒答和失败控制分支。
- 本轮评估每次批量查询 embedding 使用 310 tokens；为纳入审查修复重复运行两次，共 620 tokens。最终一次批量耗时见 comparison.execution。这些数字不包含先前开发探针。
- 冻结 active 快照建索引 embedding 为 6415 tokens；两份历史成功构建报告各 6415，共 12830。active 采集约 1.96 秒、发布前构建 5.22 秒；不是完整生命周期时延。
- 实际账单金额不可从现有接口取得，记录为 null（未取得），不能按零费用报告；未用估算价格冒充实际费用。Neo4j、Chroma、FTS 在本机运行，无本次云数据库部署费用记录。

## 交付与限制

本地可复现的查询/图检索/可视化 PoC 已交付。真实成功生成回答仍受模型可用性限制；后续若需验收生成能力，应配置可用的固定模型并在相同三组证据上重新开展生成与人工评分。生产权限、增量同步、高并发和部署仍在本 PoC 范围外。
操作与重跑命令见 [本地指南](../guides/jira-graph-poc-local.md)。原始目标文档不改动。

最终质量命令：`node --check` 两个 JS 文件；`python -m ruff check scripts/graph-acceptance.py scripts/evaluate-graph-poc.py`；启用 GRAPH_INTEGRATION=1、GRAPH_DEMO_BROWSER=1 后执行 `powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1`。结果：lint / JS syntax 通过，**554 passed**，1 个既有 multipart 弃用提示，69.77 秒。
