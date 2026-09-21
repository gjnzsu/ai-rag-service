## Why

ai-rag-service 继续定位为平台级、基于证据的公共知识查询服务。现有接口需要调用方预先选择检索意图并组织补查；本 PoC 验证受控的 Agentic RAG 能否根据问题和已有证据选择必要的补查，在保证事实准确和覆盖诚实的前提下提供价值。

## What Changes

- 新增默认关闭的只读 Agentic 查询入口，以“Epic key + 自然语言分析要求”为输入；适用于允许项目的已发布快照中的任意 Epic，不写死样例。
- 固定工作流负责范围校验、快照固定、完整 backlog 获取和 Story 统计；单 Agent 采用受预算约束的 ReAct 风格工具循环，选择依赖补查及停止。
- 封装进程内业务工具，复用现有 GraphService，不引入 MCP、独立 Agent 服务或多 Agent 并行架构。
- 返回事实统计、带证据的 findings、引用、分项覆盖状态与简要执行记录；允许客观结论，不提供主观评价、风险评级或延期预测。
- 工具失败或预算耗尽时保留已核实的部分结果；未查全不能声称不存在阻塞。
- 固定相同快照、问题集和模型配置，与确定性查询流程比较正确性、证据支持、覆盖诚实、决策价值及成本；允许得出 Agent 暂无增益的结果。

## Capabilities

### New Capabilities

- `agentic-epic-analysis`: 公共 RAG 服务内的单请求 Agentic Epic 分析、受限工具执行、证据与覆盖契约，以及对照验收。

### Modified Capabilities

无。既有 snapshot、graph retrieval、demo 及普通 /retrieve、/query 的行为要求保持不变。

## Impact

- 预期新增协调层、决策模型适配器、业务工具包装、结果契约和可选 API；复用 app/graph/service.py、structure.py、answer.py 和现有引用校验。
- 决策模型调用与最终回答生成分离。现有 GroundedAnswerGenerator 保持无工具的生成职责。
- 第一轮不增加框架或存储依赖，不做 Jira 写回、实时采集、多轮记忆、项目级完整 Defect 报告或生产部署。
- 实验基线为 2026-09-20 发布的 AIPLAT 快照 f1ef094abab04f7a9a2c658839337a23：8 Epic、26 Story、2 Bug；数据与凭据保留本地忽略目录。四个代表 Epic 为 AIPLAT-1、13、17、25，另用合成数据覆盖空 Epic、截断和故障。
- 本次交付是可审阅提案；应用功能尚未实现。详细工程默认值属于设计建议，见 design.md。
