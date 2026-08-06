# Hybrid RAG PoC evaluation report

> **Template status:** `NOT YET MEASURED`. Complete this report only after a
> live run against a versioned, manually labelled corpus. Do not replace an
> unmeasured value with a zero, estimate, or result from the synthetic example.

## Run record

| Field | Value |
| --- | --- |
| Report owner | `NOT YET RECORDED` |
| Evaluation date/time (UTC) | `NOT YET RECORDED` |
| Service revision/commit | `NOT YET RECORDED` |
| Corpus name and version | `NOT YET RECORDED` |
| Corpus source snapshot/ingestion revision | `NOT YET RECORDED` |
| Label set version and label owner/reviewer | `NOT YET RECORDED` |
| Cases supplied / complete cases by configuration | `NOT YET MEASURED` |
| Exact-Jira-key case count | `NOT YET MEASURED` |
| Lexical/identifier-heavy group definition and case IDs | `NOT YET RECORDED` |
| Environment, region, hardware, and load conditions | `NOT YET RECORDED` |
| Evaluation command | `python -m app.evaluation.runner --cases <live-labelled-cases.jsonl> --output <report.json> --corpus-revision <corpus-version> --index-revision <index-version>` |
| Raw JSON report location and checksum | `NOT YET RECORDED` |

The repository example, [cases.example.jsonl](../../evaluation/cases.example.jsonl),
is synthetic placeholder data. It demonstrates the JSONL schema only; it is
**unmeasured** and is not a Jira, Confluence, or PDF-labelled corpus.

## Effective configuration recorded for this run

Record the exact effective values below (including environment overrides), not
only the defaults. The values in the middle column are repository defaults as of
the template revision and must be replaced only when the live run used a
different recorded value.

| Setting | Default / pinned value | Effective run value |
| --- | --- | --- |
| Retrieval mode | `hybrid` | `NOT YET RECORDED` |
| Vector candidate top-k | `30` | `NOT YET RECORDED` |
| Final retrieval top-k | `10` (evaluation retains/ranks top 20) | `NOT YET RECORDED` |
| RRF k | `60` | `NOT YET RECORDED` |
| Lexical DB path and reindex/corpus revision | `./lexical.db` | `NOT YET RECORDED` |
| Lexical weights (Jira key / title / content) | `10 / 5 / 1` | `NOT YET RECORDED` |
| Vector store path and embedding model/revision | `./chroma_db`; `NOT YET RECORDED` | `NOT YET RECORDED` |
| OpenAI reranker model / timeout | `gpt-5-2025-08-07` / `5 s` | `NOT YET RECORDED` |
| Qwen reranker model / revision | `Qwen/Qwen3-Reranker-0.6B` / `e61197ed45024b0ed8a2d74b80b4d909f1255473` | `NOT YET RECORDED` |
| Qwen max candidates / max length / batch / timeout / circuit breaker | `20 / 512 / 4 / 5 s / 30 s` | `NOT YET RECORDED` |
| Answer model / timeout | `gpt-5-2025-08-07` / `15 s` | `NOT YET RECORDED` |
| Grounding evidence top-k / prompt cap / excerpt cap | `5 / 4000 chars / 200 chars` | `NOT YET RECORDED` |

## Experiment matrix

| ID | Configuration | Candidate-pool rule | Run status |
| --- | --- | --- | --- |
| A | Vector only | Retrieve and rank the comparable top-20 pool. | `NOT YET MEASURED` |
| B | BM25 + vector + RRF (no reranker) | Retrieve once per case; cache the B top-20 pool. Exact Jira lookup is disabled and either primary-backend degradation fails the case. | `NOT YET MEASURED` |
| C1 | B + pinned GPT-5 reranker | Consume an independent deep copy of the same cached B top-20 candidates. Do not retrieve again. | `NOT YET MEASURED` |
| C2 | B + pinned Qwen3-Reranker-0.6B reranker | Consume an independent deep copy of the same cached B top-20 candidates. Do not retrieve again. | `NOT YET MEASURED` |

C1 and C2 must use the same B candidate order, labels, cases, and metrics. A
reranker fallback/error is a recorded configuration/case failure, not a
successful unchanged ranking.

## Aggregate measurements

All cells are deliberately blank of numeric defaults. Populate from the raw
JSON report and human labels; retain `NOT YET MEASURED` where an observation is
not available.

| Metric | A | B | C1 | C2 |
| --- | --- | --- | --- | --- |
| Complete successful case population (case indexes) | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Recall@20 | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Hit@5 | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| MRR@10 | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Context Precision | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Citation Validity | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Citation Correctness (human labelled) | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Abstention Accuracy | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Generated-answer latency P50 / P95 | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Reranker / answer token usage and cost inputs | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |
| Local CPU / memory observations (optional) | `NOT APPLICABLE` | `NOT APPLICABLE` | `NOT APPLICABLE` | `NOT YET MEASURED` |
| Failed cases and safe failure types | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |

### Required comparisons

| Check | Evidence/result |
| --- | --- |
| Exact Jira-key Hit@5: numerator / denominator / result | `NOT YET MEASURED` |
| B Recall@20 compared with A: delta and non-regression result | `NOT YET MEASURED` |
| B vs A for the pre-defined lexical/identifier-heavy group: per-case and aggregate change | `NOT YET MEASURED` |
| C1 vs B: MRR@10 and Context Precision change; Recall@20 check | `NOT YET MEASURED` |
| C2 vs B: MRR@10 and Context Precision change; Recall@20 check | `NOT YET MEASURED` |
| C1/C2 complete case populations identical to B | `NOT YET MEASURED` |
| C1/C2 generated-answer P95 is at most 10 seconds | `NOT YET MEASURED` |

## Per-question results and changes

Use case indexes/IDs, not question text, in any sharable report. Add one row
per input case for each configuration; do not summarize a small corpus only by
aggregates.

| Case index/ID | Group / exact-Jira flag | A: ranked IDs & metrics | B: ranked IDs & metrics | C1: ranked IDs & metrics | C2: ranked IDs & metrics | B→C1 change | B→C2 change | Citation/abstention label | Failure/status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NOT YET RECORDED` | `NOT YET RECORDED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` | `NOT YET MEASURED` |

For each case, preserve the complete per-question deltas for Recall@20,
Hit@5, MRR@10, Context Precision, citation observations, abstention, latency,
tokens/cost inputs, and resource observations when captured.

## Failure-path and automated verification evidence

Record automated verification separately from live measurement. After the
quality command passes for the implementation revision under review, record
the exact command, date, test count, and any warning here:

| Evidence | Recorded result |
| --- | --- |
| `powershell -ExecutionPolicy Bypass -File .\scripts\quality-check.ps1` | `NOT YET RECORDED FOR THIS REPORT` |
| `git diff --check` | `NOT YET RECORDED FOR THIS REPORT` |
| Vector-only fallback when lexical retrieval is unavailable | Automated test evidence; live operational rate `NOT YET MEASURED` |
| Lexical-only fallback when vector retrieval is unavailable | Automated test evidence; live operational rate `NOT YET MEASURED` |
| No-reranker/RRF-order fallback on reranker failure | Automated test evidence; live operational rate `NOT YET MEASURED` |
| Both primary retrievers unavailable returns no generated answer | Automated test evidence; live operational rate `NOT YET MEASURED` |
| Invalid citation containment and unanswerable refusal/no citations | Automated test evidence; live correctness rate `NOT YET MEASURED` |

## Acceptance checklist

| # | Approved acceptance criterion | Current evidence and final-report status |
| --- | --- | --- |
| 1 | All existing API tests remain compatible. | **Automated evidence:** repository API/integration coverage and quality gate. Record the fresh gate result above. |
| 2 | Exact Jira-key evaluation cases achieve 100% Hit@5. | **Requires live labelled evaluation:** report exact numerator/denominator; currently `NOT YET MEASURED`. |
| 3 | Hybrid retrieval does not reduce overall Recall@20 relative to vector-only retrieval. | **Requires live labelled evaluation:** compare B against A at top 20; currently `NOT YET MEASURED`. |
| 4 | Hybrid retrieval materially improves at least one lexical or identifier-heavy evaluation group. | **Requires live labelled evaluation:** predefine the group and show aggregate plus per-case deltas; currently `NOT YET MEASURED`. |
| 5 | Reranker configurations are compared using the same candidates, labels, and metrics. | **Automated evidence:** evaluation-runner tests enforce cached B candidates and independent copies for C1/C2. A live run must record the complete case population. |
| 6 | A reranker is enabled by default only when it improves ranking quality without exceeding the 5–10 second generated-answer latency target. | **Requires live labelled evaluation:** apply the recommendation gate below; currently `NOT YET MEASURED`. |
| 7 | 100% of returned citation IDs and URLs are mapped from server-known evidence. | **Automated evidence:** grounding/API validation tests. A live citation-rate measurement remains `NOT YET MEASURED`. |
| 8 | Unanswerable cases return no fabricated citations. | **Automated evidence:** grounded-query/API refusal tests. A live abstention/citation measurement remains `NOT YET MEASURED`. |
| 9 | Failure-path tests demonstrate documented vector-only, lexical-only, and no-reranker fallbacks. | **Automated evidence:** retrieval, API, and query-engine fallback tests. Live failure frequency is `NOT YET MEASURED`. |
| 10 | The final evaluation report states which configuration is recommended and why. | **Requires live labelled evaluation before final PoC recommendation.** Until then, the provisional default below applies. |

## Recommendation and decision gate

**Current default recommendation: B — RRF/no reranker.** This is a default,
not a measured quality claim.

A reranker may be recommended only if it shows a **MRR@10 or Context Precision
gain**, **no Recall@20 regression**, a **complete identical case population**
relative to B, and **generated-answer P95 ≤10 seconds**. Otherwise B/RRF
remains the default.

| Decision field | Result |
| --- | --- |
| Candidate configuration | `B / RRF-no-reranker (default until gate is satisfied)` |
| MRR@10 or Context Precision gain demonstrated | `NOT YET MEASURED` |
| Recall@20 non-regression demonstrated | `NOT YET MEASURED` |
| Complete identical case population demonstrated | `NOT YET MEASURED` |
| Generated-answer P95 ≤10 seconds demonstrated | `NOT YET MEASURED` |
| Final recommendation and rationale | `NOT YET MEASURED — retain B/RRF-no-reranker` |

## Limitations and sign-off

- The automated suite verifies implementation behavior; it does not measure
  live retrieval quality, Jira Hit@5, latency, token cost, or local resources.
- The synthetic example is not a valid source of live-corpus conclusions.
- Initial document-level labels can be used; record whether chunk-level labels
  were added and any resulting limitation.
- Retrieval and reranker scores are not calibrated answer-confidence values.

| Role | Name | Date | Decision / signature |
| --- | --- | --- | --- |
| Evaluation owner | `NOT YET RECORDED` | `NOT YET RECORDED` | `NOT YET RECORDED` |
| Label reviewer | `NOT YET RECORDED` | `NOT YET RECORDED` | `NOT YET RECORDED` |
| Engineering owner | `NOT YET RECORDED` | `NOT YET RECORDED` | `NOT YET RECORDED` |
| Product/security approver (if required) | `NOT YET RECORDED` | `NOT YET RECORDED` | `NOT YET RECORDED` |
