# Hybrid Retrieval and Grounded Answers PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved hybrid retrieval and grounded-answer PoC while preserving the existing `/retrieve` and `/query` contracts.

**Architecture:** The shared indexing path normalizes every source into canonical document/chunk identities and writes the same chunks to ChromaDB and a collection-aware SQLite FTS5 index. A retrieval pipeline runs deterministic Jira lookup, BM25, and vector search, fuses candidates with RRF, optionally reranks them, selects evidence, and exposes the same pipeline to retrieval-only and generated-answer APIs. Server-owned evidence IDs are validated after generation, while an offline harness compares the four approved experiment configurations.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, ChromaDB, SQLite FTS5, OpenAI Python SDK, optional Transformers/PyTorch for Qwen3, pytest, Ruff.

## Global Constraints

- Preserve all existing `/retrieve` and `/query` request and response fields; additions are optional and backward compatible.
- Keep ChromaDB with OpenAI `text-embedding-3-small`; do not add PostgreSQL or pgvector.
- Canonical identities are `jira:<issue-key>`, `confluence:<page-id>`, `pdf:<stable-content-or-file-identity>`, and `<document-id>:chunk:<zero-based-index>`.
- Preserve legacy Jira metadata access through both `issue_key` and lifecycle `key` while using one canonical accessor internally.
- Lexical and vector retrieval each default to 30 candidates; RRF uses `k = 60` and returns at most 20 candidates.
- Reranking is optional and every reranker failure returns the unchanged RRF order.
- URLs and evidence mappings are created only from server-known connector/index metadata; model output may not create or override them.
- Initial evidence thresholds remain disabled until evaluation calibration exists.
- Both retrieval paths failing is an upstream error; one path failing must degrade to the other.
- No new LLM query-planning call, tenant authorization, agentic retrieval, or fine-tuning.
- Normal generated-answer latency target remains 5–10 seconds.

---

### Task 1: Canonical chunks and collection-aware SQLite FTS5 dual indexing

**Files:**
- Create: `app/retrieval/models.py`
- Create: `app/retrieval/lexical.py`
- Create: `app/retrieval/__init__.py`
- Modify: `app/config.py`
- Modify: `app/connectors/jira.py`
- Modify: `app/connectors/confluence.py`
- Modify: `app/connectors/pdf.py`
- Modify: `app/pipeline/chunker.py`
- Modify: `app/pipeline/indexer.py`
- Modify: `app/pipeline/store.py`
- Modify: `app/api/ingest.py`
- Modify: `app/api/lifecycle.py`
- Test: `tests/test_retrieval/test_lexical.py`
- Test: `tests/test_pipeline/test_indexer.py`
- Test: `tests/test_pipeline/test_lifecycle_store.py`
- Test: `tests/test_connectors/test_canonical_metadata.py`

**Interfaces:**
- Produces: `RetrievalCandidate`, `SQLiteFTSIndex.upsert_document(chunks, collection_name)`, `SQLiteFTSIndex.delete_document(document_id, collection_name)`, and canonical chunk dictionaries shared by both stores.
- Consumes: existing `Document`, `chunk_documents`, `embed_chunks`, and Chroma storage functions.

- [ ] **Step 1: Write failing canonical identity tests**

```python
def test_chunk_documents_uses_canonical_chunk_identity():
    chunks = chunk_documents([Document(id="jira:PROJ-7", content="body", source_type="jira", title="Title")])
    assert chunks[0]["document_id"] == "jira:PROJ-7"
    assert chunks[0]["chunk_id"] == "jira:PROJ-7:chunk:0"
    assert chunks[0]["id"] == "jira:PROJ-7:chunk:0"

def test_jira_connector_preserves_legacy_and_canonical_metadata():
    document = fetched_documents[0]
    assert document.id == "jira:PROJ-7"
    assert document.metadata["issue_key"] == "PROJ-7"
    assert document.metadata["key"] == "PROJ-7"
    assert document.metadata["source_url"].endswith("/browse/PROJ-7")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_connectors/test_canonical_metadata.py tests/test_pipeline/test_indexer.py -q`

Expected: FAIL because current connectors use hashed IDs and the chunker uses `_chunk_` IDs.

- [ ] **Step 3: Implement canonical connector and chunk metadata**

Use connector-owned IDs and URLs, set `chunk_index` inside the per-document loop, and preserve legacy metadata names. `chunk_documents()` must emit `id == chunk_id == f"{document_id}:chunk:{index}"` and include `source_url` as a trusted top-level flattened field.

- [ ] **Step 4: Write failing FTS5 tests**

```python
def test_fts_upsert_replaces_all_chunks_for_document(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    index.upsert_document(old_chunks, "alpha")
    index.upsert_document(new_chunks, "alpha")
    assert [item.chunk_id for item in index.search("new", 10, None, "alpha")] == ["jira:PROJ-7:chunk:0"]
    assert index.search("old", 10, None, "alpha") == []

def test_fts_search_is_collection_and_filter_aware(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    index.upsert_document(chunks_in_two_collections, "alpha")
    results = index.search("login", 10, {"source_type": "jira"}, "alpha")
    assert all(result.collection_name == "alpha" for result in results)
    assert all(result.source_type == "jira" for result in results)
```

- [ ] **Step 5: Run FTS5 tests and verify RED**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_retrieval/test_lexical.py -q`

Expected: FAIL because `app.retrieval.lexical` does not exist.

- [ ] **Step 6: Implement candidate model and SQLite FTS5 index**

`RetrievalCandidate` carries `content`, `document_id`, `chunk_id`, `source_type`, `source_url`, `title`, `metadata`, `score`, `retrieval_methods`, `rank_by_method`, `fused_rank`, `rerank_score`, and `exact_match`. Use a normal `lexical_chunks` table for trusted metadata plus an FTS5 virtual table over `issue_key`, `title`, and `content`. Parameterize collection/filter SQL, construct safe MATCH terms from Unicode word tokens, and call `bm25(..., issue_key_weight, title_weight, content_weight)` with settings defaults that favor identifiers and titles.

- [ ] **Step 7: Write failing dual-write and delete tests**

```python
def test_index_documents_writes_identical_chunks_to_both_indexes(mock_embed, mock_vector, mock_lexical):
    index_documents([document], lexical_index=mock_lexical)
    assert mock_vector.call_args.kwargs["chunks"] == mock_lexical.upsert_document.call_args.args[0]

def test_delete_document_removes_vector_and_lexical_chunks(tmp_indexes):
    assert delete_document("jira:PROJ-7", "alpha") is True
    assert lexical_index.search("PROJ-7", 10, None, "alpha") == []
```

- [ ] **Step 8: Implement idempotent dual writes and shared ingestion**

Make `index_documents` write Chroma then FTS with the same chunks and structured failure logging. Route PDF, Jira, Confluence, FX, and lifecycle upserts through it. Extend document deletion to both stores. A partial write must raise after logging collection/document ID so repeating the document upsert repairs both indexes.

- [ ] **Step 9: Verify Task 1 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_connectors tests/test_pipeline tests/test_retrieval/test_lexical.py tests/test_api/test_ingest.py tests/test_api/test_lifecycle.py -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add app tests
git commit -m "feat: add canonical dual indexing with SQLite FTS5"
```

### Task 2: Hybrid retrieval, Jira exact lookup, RRF, and safe degradation

**Files:**
- Create: `app/retrieval/interfaces.py`
- Create: `app/retrieval/vector.py`
- Create: `app/retrieval/query_hints.py`
- Create: `app/retrieval/fusion.py`
- Create: `app/retrieval/pipeline.py`
- Modify: `app/config.py`
- Modify: `app/pipeline/store.py`
- Test: `tests/test_retrieval/test_query_hints.py`
- Test: `tests/test_retrieval/test_fusion.py`
- Test: `tests/test_retrieval/test_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalCandidate` and `SQLiteFTSIndex` from Task 1.
- Produces: `ChromaVectorRetriever.search(query, top_k, filters, collection_name)`, `extract_query_hints(query)`, `ReciprocalRankFusion.fuse(result_sets, top_k)`, and `HybridRetrievalPipeline.retrieve(...)`.

- [ ] **Step 1: Write failing deterministic Jira-key tests**

```python
@pytest.mark.parametrize("query,expected", [("status of PROJ-42?", ["PROJ-42"]), ("proj-42 and AUTH-9", ["PROJ-42", "AUTH-9"]), ("no key", [])])
def test_extract_query_hints(query, expected):
    assert extract_query_hints(query).jira_keys == expected
```

- [ ] **Step 2: Verify query-hint RED, then implement**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_retrieval/test_query_hints.py -q`

Expected: FAIL because the module is absent. Implement a configurable compiled regex defaulting to `\b[A-Z][A-Z0-9]+-\d+\b`, normalize keys to uppercase, and preserve encounter order.

- [ ] **Step 3: Write failing RRF tests**

```python
def test_rrf_fuses_and_deduplicates_by_chunk_id():
    fused = ReciprocalRankFusion(k=60).fuse([[vector_a, shared], [shared, lexical_b]], top_k=20)
    assert [item.chunk_id for item in fused] == [shared.chunk_id, vector_a.chunk_id, lexical_b.chunk_id]
    assert shared.retrieval_methods == ["vector", "bm25"]

def test_exact_match_has_visible_deterministic_priority():
    fused = ReciprocalRankFusion(k=60).fuse([[semantic], [exact]], top_k=20)
    assert fused[0].chunk_id == exact.chunk_id
    assert fused[0].exact_match is True
```

- [ ] **Step 4: Verify fusion RED, implement RRF, then verify GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_retrieval/test_fusion.py -q`

Expected RED: module absent. Implement `sum(1 / (k + one_based_rank))`, merged method/rank diagnostics, stable tie-breaking, exact-match priority, dedupe by `chunk_id`, and `fused_rank` assignment. Rerun and expect PASS.

- [ ] **Step 5: Write failing pipeline fallback tests**

Cover hybrid success, vector-only fallback, lexical-only fallback, one empty result, both retrievers raising, filters passed to both, exact Jira candidates included, and candidate limits 30/30/20.

- [ ] **Step 6: Implement vector adapter and retrieval orchestration**

The vector adapter embeds with `text-embedding-3-small`, maps Chroma distance to a diagnostic score, and returns canonical candidates. The pipeline calls lexical and vector independently, records structured diagnostics without content, adds exact Jira results, continues after one error, and raises `RetrievalUnavailableError` only when both primary retrievers fail.

- [ ] **Step 7: Verify Task 2 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_retrieval -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add app tests
git commit -m "feat: add hybrid retrieval and reciprocal rank fusion"
```

### Task 3: Optional no-op, GPT-5, and Qwen3 rerankers

**Files:**
- Create: `app/reranking/__init__.py`
- Create: `app/reranking/base.py`
- Create: `app/reranking/noop.py`
- Create: `app/reranking/openai.py`
- Create: `app/reranking/qwen.py`
- Create: `requirements-qwen.txt`
- Modify: `app/config.py`
- Modify: `app/retrieval/pipeline.py`
- Test: `tests/test_reranking/test_noop.py`
- Test: `tests/test_reranking/test_openai.py`
- Test: `tests/test_reranking/test_qwen.py`
- Test: `tests/test_retrieval/test_pipeline.py`

**Interfaces:**
- Consumes: the top 20 `RetrievalCandidate` values produced by Task 2.
- Produces: `Reranker.rerank(query, candidates, top_k)`, `NoOpReranker`, `GPT5Reranker`, `Qwen3LocalReranker`, and `build_reranker(settings)`.

- [ ] **Step 1: Write and verify failing no-op/order tests**

Assert that no-op preserves input order and truncates to `top_k`; run the test and observe missing modules.

- [ ] **Step 2: Implement the protocol, no-op reranker, and provider factory**

The factory accepts exactly `none`, `openai`, and `qwen_local`; unknown values fail configuration validation. No-op is the default until evaluation recommends otherwise.

- [ ] **Step 3: Write failing GPT-5 strict-output tests**

Test valid discrete grades, unknown IDs, duplicate IDs, malformed JSON, timeout/API error, prompt delimiters, no URLs/tools, and stable handling of tied grades. Every invalid/error response must equal the original candidate order.

- [ ] **Step 4: Implement GPT-5 listwise reranking**

Send at most 20 delimited untrusted passages through a configurable pinned model and timeout. Request strict JSON containing only `{chunk_id, relevance_grade}` with grades 0–3. Reject unknown/duplicate IDs and never accept URLs. Set `rerank_score` to the grade and preserve original rank for ties/missing valid IDs.

- [ ] **Step 5: Write failing Qwen lifecycle and failure tests**

Inject fake tokenizer/model/clock objects to prove one model load per reranker, bounded batch/input size, descending scores, and unchanged-order fallback on load/inference/timeout errors.

- [ ] **Step 6: Implement lazy optional Qwen3 adapter**

Keep `torch` and `transformers` out of the default runtime import path. Pin `Qwen/Qwen3-Reranker-0.6B` and its revision in settings, load once under a lock, bound batch/input length, run inference without gradients, and fail open. Document installable pinned packages in `requirements-qwen.txt`.

- [ ] **Step 7: Integrate reranking and verify Task 3 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_reranking tests/test_retrieval/test_pipeline.py -q`

Expected: PASS, including reranker-error preservation of RRF order.

- [ ] **Step 8: Commit**

```powershell
git add app tests requirements-qwen.txt
git commit -m "feat: add optional OpenAI and Qwen rerankers"
```

### Task 4: Evidence selection, grounded generation, and citation validation

**Files:**
- Create: `app/grounding/__init__.py`
- Create: `app/grounding/evidence.py`
- Create: `app/grounding/generator.py`
- Create: `app/grounding/citations.py`
- Modify: `app/rag/query_engine.py`
- Test: `tests/test_grounding/test_evidence.py`
- Test: `tests/test_grounding/test_generator.py`
- Test: `tests/test_grounding/test_citations.py`
- Test: `tests/test_rag/test_query_engine.py`

**Interfaces:**
- Consumes: reranked `RetrievalCandidate` values from Task 3.
- Produces: `EvidenceSelector.select`, evidence IDs `E1..En`, `GroundedAnswerGenerator.generate`, `CitationValidator.validate`, citations mapped only from supplied evidence, and grounding statuses.

- [ ] **Step 1: Write failing evidence-selection tests**

Test final count 5–10, deduplication, preference for source diversity on cross-document candidates, no threshold-based refusal by default, and empty candidates returning insufficient evidence.

- [ ] **Step 2: Implement evidence selection**

Use rank/reranker signals without pretending scores are calibrated probabilities. Deduplicate exact chunk IDs and near-duplicate normalized content, apply a bounded per-document selection rule where alternatives exist, then assign server evidence IDs.

- [ ] **Step 3: Write failing generation-contract tests**

Assert the prompt delimiters label retrieved content as untrusted, require citations for material claims, require refusal when unsupported, and send no tools. Test structured output with answer plus citation IDs.

- [ ] **Step 4: Implement grounded generation**

Use the configured answer model and strict JSON schema. Include only server-assigned evidence IDs/chunk content in the prompt; never ask the model to output URLs. Return the exact existing refusal text when evidence is empty or output cannot be validated.

- [ ] **Step 5: Write failing citation validation tests**

Test known IDs, invented IDs, duplicate references, bounded excerpts, trusted document/chunk/URL mapping, unanswerable output with no citations, and status values `supported`, `partially_supported`, `insufficient_evidence`, `validation_failed`.

- [ ] **Step 6: Implement citation validation and query orchestration**

Map citations only through the selected-evidence dictionary, drop invented IDs, cap excerpts using a configured character count, and set operational status without claiming semantic proof. Refactor `query_engine.query` to call retrieval → reranking → selection → generation → validation while preserving `answer`, `sources`, and `model`.

- [ ] **Step 7: Verify Task 4 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_grounding tests/test_rag/test_query_engine.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add app tests
git commit -m "feat: add evidence-aware grounded answer generation"
```

### Task 5: Backward-compatible `/retrieve` and `/query` diagnostics

**Files:**
- Modify: `app/api/lifecycle.py`
- Modify: `app/api/query.py`
- Modify: `app/rag/query_engine.py`
- Test: `tests/test_api/test_lifecycle.py`
- Test: `tests/test_api/test_query.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: the hybrid pipeline and grounding result from Tasks 2–4.
- Produces: additive optional retrieval diagnostics, citations, grounding, and retrieval metadata without changing existing required request fields.

- [ ] **Step 1: Write failing API compatibility/additive-field tests**

Existing response fixtures without new fields must still validate. New fixtures must expose `retrieval_methods`, `fused_rank`, optional `rerank_score`, trusted `source_url`, `citations`, `grounding.status`, and `retrieval_metadata.mode/reranker`.

- [ ] **Step 2: Write failing API fallback tests**

At API level test vector-only and lexical-only success, both retrieval paths failing with 502, reranker failure with a successful response, insufficient evidence with exact refusal/no citations, and invalid citation IDs with `validation_failed`.

- [ ] **Step 3: Implement additive Pydantic models and shared pipeline wiring**

All added response fields use optional/default values. `/retrieve` uses the same pipeline without answer generation. `/query` retains existing request parameters and maps legacy `document_type` into the filter contract. Diagnostics must never include content or secrets in logs.

- [ ] **Step 4: Verify Task 5 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_api tests/test_integration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app tests
git commit -m "feat: expose hybrid retrieval and grounding diagnostics"
```

### Task 6: Repeatable evaluation harness and operator documentation

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/models.py`
- Create: `app/evaluation/metrics.py`
- Create: `app/evaluation/runner.py`
- Create: `evaluation/cases.example.jsonl`
- Create: `tests/test_evaluation/test_metrics.py`
- Create: `tests/test_evaluation/test_runner.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `pytest.ini`
- Create: `scripts/quality-check.ps1`

**Interfaces:**
- Consumes: the same retrieval pipeline with modes `vector`, `hybrid`, and rerankers `none`, `openai`, `qwen_local`.
- Produces: per-question and aggregate JSON reports for configurations A, B, C1, C2 containing Recall@20, Hit@5, MRR@10, context precision, citation validity/correctness labels, abstention accuracy, P50/P95 latency, token usage, and optional local CPU/memory observations.

- [ ] **Step 1: Write failing pure metric tests**

Use fixed ranked IDs/labels to assert exact Recall@20, Hit@5, MRR@10, context precision, citation validity, abstention accuracy, and percentile results, including empty/unanswerable cases.

- [ ] **Step 2: Implement typed evaluation cases and pure metrics**

Accept the approved JSONL case shape, validate query type and required labels, and keep metric calculations independent of external services.

- [ ] **Step 3: Write failing runner tests**

Inject a fake pipeline/generator and assert the same case/candidates are used across A/B/C1/C2, per-question deltas are reported, aggregate metrics are separate, and failed cases are recorded without aborting the run.

- [ ] **Step 4: Implement CLI runner and example corpus**

Expose `python -m app.evaluation.runner --cases <jsonl> --output <json>`. Record configuration, model names/revisions, latency, token/local resource observations, individual results, aggregate metrics, and a recommendation only when the approved material-improvement/no-regression rule is satisfied.

- [ ] **Step 5: Document configuration, reindexing, experiments, and optional local model setup**

Document every new environment variable/default, FTS database persistence, the need to reindex existing Chroma-only data, safe fallbacks, evaluation commands, Qwen optional installation, and the fact that grounding status is operational rather than proof of truth.

- [ ] **Step 6: Add repository-owned quality command**

`scripts/quality-check.ps1` must set only a process-local dummy key when absent, then run `python -m ruff check app tests` and `python -m pytest -q`, exiting on either failure.

- [ ] **Step 7: Verify Task 6 GREEN**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_evaluation -q`

Expected: PASS.

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\quality-check.ps1`

Expected: Ruff PASS and all pytest tests PASS.

- [ ] **Step 8: Commit**

```powershell
git add app evaluation tests README.md .env.example pytest.ini scripts
git commit -m "feat: add hybrid RAG evaluation harness and quality gate"
```

### Task 7: Whole-PoC acceptance and final report

**Files:**
- Create: `docs/evaluation/hybrid-rag-poc-report-template.md`
- Modify as findings require: files from Tasks 1–6

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: a verified branch and a report template that explicitly captures the acceptance criteria without inventing unevaluated quality claims.

- [ ] **Step 1: Run full compatibility and failure-path suite**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\quality-check.ps1`

Expected: Ruff PASS and all tests PASS, including vector-only, lexical-only, reranker-failure, invalid-citation, and insufficient-evidence tests.

- [ ] **Step 2: Run focused acceptance tests**

Run: `$env:OPENAI_API_KEY='test-key'; python -m pytest tests/test_retrieval/test_pipeline.py tests/test_api/test_lifecycle.py tests/test_api/test_query.py tests/test_grounding tests/test_evaluation -q`

Expected: PASS.

- [ ] **Step 3: Add the evaluation report template**

Include corpus/version metadata, A/B/C1/C2 configuration table, aggregate metrics, per-question changes, exact Jira Hit@5, Recall@20 regression check, latency/cost/resources, fallback results, default recommendation, and explicit `not yet measured` entries where the real corpus has not been run.

- [ ] **Step 4: Commit final verification artifacts**

```powershell
git add docs
git commit -m "docs: add hybrid RAG evaluation report template"
```

- [ ] **Step 5: Request final whole-branch review**

Generate a review package from the branch merge-base through HEAD and verify complete design coverage, backward compatibility, security boundaries, test quality, and absence of unevaluated quality claims.

