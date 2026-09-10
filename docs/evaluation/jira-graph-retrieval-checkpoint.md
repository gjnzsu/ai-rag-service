# 组合检索 checkpoint — 2026-09-08

> 最新状态：2026-09-10 用户已确认 PoC 测试完成，见 [用户验收记录](jira-graph-user-acceptance.md)。本文以下保留当时的实验结果。

## 实现范围

完成固定意图的显式锚点/同快照 hybrid 起点、图扩展、关系与正文分开取证、POST /graph/retrieve 和 /graph/query。旧 /retrieve、/query 保留。没有实现演示页、自由查询 agent 或 Jira 写回。

同一请求只解析一次 active manifest；所有存储适配器使用该作用域。Chroma/SQLite 缺失或内容不一致不访问默认索引。多种降级均有明确标记；无回答模型时仍返回证据。

## 真实观察

固定快照 `7cc2704c483c423c9bc1d749040bbd64`。

- 不带 key 的问句 `retrieval security unauthorized data access` 经 hybrid 命中 AIPLAT-46，再得到指向 AIPLAT-16 的 BLOCKS 路径与双方 9 个正文 chunks。
- AIPLAT-13 显式锚点解析 4 个直接子事项；返回同一 BLOCKS 路径、16 个正文 chunks，并保留 CHILD_OF 归属事实。两个关系端点均为 Done，不据此称为当前阻塞风险。
- 实际 HTTP：hybrid retrieve 200 / 4533.01 ms；Epic retrieve 200 / 108.17 ms；query 200 / 2693.71 ms，回答状态 answer_unavailable、路径保留。单次端到端样本包括连接/模型请求，不能解读为 P95。
- 直接服务探针两次 embedding 调用各观测 7 tokens，合计 14。另有 HTTP embedding 调用未记录 usage，不把 14 称为全部成本。生成没有返回 usage，不宣称零费用。

详见 jira-graph-retrieval-live.json 和 jira-graph-retrieval-http.json。前者保留诊断时的实际失败状态（当时诊断名 citation_validation_failed）；最终代码进一步区分 generation_unavailable 与引用格式错误。

## 当前外部限制

配置的回答模型 `gpt-5-2025-08-07` 可出现在账号模型列表，但实际调用返回 HTTP 404、model_not_found。未替换或放宽原服务模型约束。真实成功回答和语义正确率尚未验证；成功、拒答、错误引用、超预算与生成故障分支用合成/注入生成器测试，不能当作真实模型质量证据。

演示检索链路已经可用。成功的自然语言回答还需要确认可调用的固定模型配置，或恢复当前模型访问。

## TDD 与审查

- 主串联最初 14 个失败（缺失合约/实现），实现后与原结构 API 合计 34 passed；新增 HTTP 路由先 RED，再通过。
- 搜索适配器使用真实临时 Chroma/SQLite、合成向量；先缺实现失败，故障诊断重置另有行为 RED。回答适配器先缺实现失败，再补方向/引用/截断/清理测试。
- 独立审查发现并修复 Epic 归属证据缺失、多个无子事项 Epic 的静默截断、图故障时忽略问句明确 key。修复均有失败复现；进一步加入稳定 issue ID 到回答证据以保证归属可连接。审查确认全部解决。
- API 构造/清理异常也保留证据；最终无生成内容的诊断与坏引用分开。

最终全量门禁：lint passed，516 passed / 1 个已有弃用提示，48.77 秒；OpenSpec strict validate 通过。真实 Neo4j 用独立合成 scope 验证两跳/环等，未人为修改 AIPLAT。
