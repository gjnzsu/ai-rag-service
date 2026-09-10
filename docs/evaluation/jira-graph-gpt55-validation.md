# GPT-5.5 live generation validation

> 最新状态：2026-09-10 用户已确认 PoC 测试完成，见 [用户验收记录](jira-graph-user-acceptance.md)。本文以下保留当时的实验结果。

Date: 2026-09-09. Model: `gpt-5.5-2026-04-23` using existing OpenAI credentials and Chat Completions with strict JSON schema. Snapshot: `7b9d816aa2c445c9919635aff149f705`. Embedding and retrieval indices unchanged.

The old GPT-5 snapshot returned an organization-verification 404. A direct GPT-5.5 probe succeeded; three subsequent requests exercised the local `/graph/query` endpoint, real generation, and citation validation.

| Question | Result | Latency |
| --- | --- | --- |
| What does AIPLAT-37 block, and which Epics own them? | insufficient_evidence; dependency retrieval supplied Story nodes and BLOCKS edge but no Epic membership evidence | 8.83 s |
| What does AIPLAT-46 block, and are they unresolved? | supported; AIPLAT-16, both Done, three valid citations | 6.22 s |
| What does AIPLAT-37 block, and what are their statuses? | supported; AIPLAT-23, both To Do, valid edge citation | 4.47 s |

The two supported answers were inspected against the returned graph evidence. This establishes real generation and citation compatibility for these examples, not comprehensive answer-quality acceptance or BA/PM review. The compound Epic-membership question remains a retrieval coverage limitation; model migration does not expand retrieval scope. Historical evaluation reports are preserved as records of their original runs.

Machine-readable answers and citations: `jira-graph-gpt55-validation.json`.
