# zvec-grep: initial AI RAG service PoC assessment

**Assessment date:** 4 October 2026 (Asia/Shanghai).  
**Repository:** https://github.com/zvec-ai/zvec-grep  
**Reviewed snapshot:** `30c316052f4298ff6fac1e71f45ea40606e24723` (2 October 2026).  
**Scope:** documentation and TypeScript source inspection, dependency/build checks, and selected repository tests. This is an initial feasibility assessment, not a production certification or a retrieval-quality benchmark on your data.

## 1. Recommendation

**Proceed with a bounded, single-tenant PoC.** zvec-grep is a credible retrieval component for a service answering questions over a curated local knowledge corpus, especially Markdown, text, and source code. It offers BM25 full-text retrieval, semantic vectors, reciprocal rank fusion (RRF), source locations, incremental indexing, and MCP integration without operating a separate search database server.

Its strongest potential value is shortening evidence discovery and reducing the context an agent consumes. It does not provide the complete RAG service: ingestion connectors, PDF/Office conversion, identity and document permissions, answer generation, evaluation, and service operations remain your responsibility.

**Do not adopt it as the default platform for a shared enterprise knowledge service yet.** The current design centers on filesystem workspaces and a local daemon. Multi-tenant isolation, arbitrary business metadata filtering, distributed operation, availability, and production scale need separate design and evidence.

Assumption: “ai-rag-service” means an application that ingests internal knowledge and returns grounded answers with citations, potentially accessed by agents through MCP. No existing service architecture, corpus, SLA, budget, or user count was supplied. The conclusions below are conditional on that assumption.

| Intended PoC | Feasibility | Reason |
|---|---|---|
| Developer assistant over repositories and Markdown | High | Structural extraction, hybrid discovery, exact verification, file/line evidence |
| Internal Q&A over curated English/Chinese text exports | Medium–high | Straightforward local indexing; model and Chinese retrieval quality must be measured |
| Knowledge Q&A dominated by PDF, Word, and slides | Medium | Requires an external conversion pipeline and preservation of page/source citations |
| Existing RAG service replacing its retriever | Medium | Exported engine API can support an adapter; benefit must beat the current hybrid baseline |
| Shared enterprise search with per-document ACLs | Low as a turnkey solution | Requires permission enforcement and strong isolation outside zg |
| Large, highly available, distributed search service | Unproven | No validated workload limits, distributed architecture, or HA evidence from this review |

### Repository context added when saving this report

The initial assessment above was prepared before inspecting this service repository. Its current README documents FastAPI, PyMuPDF PDF ingestion, ChromaDB vectors, SQLite FTS5 lexical retrieval, and hybrid RRF (`k=60`), with remote model calls routed through AI Gateway. This narrows the recommendation: evaluate zg as an optional retrieval adapter or agent-facing MCP interface against the existing hybrid baseline, rather than assuming hybrid retrieval or PDF ingestion must be built from scratch. Existing PDF extraction may be reused to produce a versioned text/Markdown corpus with page/source mappings. zg’s remote embedding backend is not established here as compatible with this service’s required AI Gateway contract; keep the initial experiment local or validate that integration separately.

Service references: [README](../../README.md) and [AI Gateway migration guide](ai-gateway-migration.md). This context is based on repository documentation, not a fresh runtime validation of the service.

## 2. What the project actually is

The repository is **zvec-grep (`zg`)**, a search application built on **Alibaba’s zvec** engine. The README links the underlying engine to `alibaba/zvec`; the application repository is under `zvec-ai`. This relationship does not establish commercial support or a service SLA.

There are two implementations in the snapshot:

- **TypeScript/Node.js:** the root package is `@zvec/zvec-grep`, version `0.2.1`, requires Node.js ≥22, and exports `createZvecGrep`. The root documentation describes this implementation.
- **Rust:** a separate workspace under `rust/`, with its own CLI behavior, native packaging, dependencies, and compatibility registry. Some recent benchmark workflows build Rust candidates. Rust behavior and published-package availability must be verified separately.

Use the **reviewed TypeScript snapshot as the first integration candidate**, unless your team specifically wants Rust. Pin the implementation, commit/package, embedding model, and corpus in the evaluation. Do not mix Node requirements, Rust packaging claims, or benchmark results across implementations.

License: Apache 2.0 at the application root. Preserve required notices when redistributing; verify dependency and embedding-model licenses independently. A permissive application license does not cover every model artifact.

## 3. Retrieval capabilities and their implications

| Capability | Evidence in the reviewed snapshot | Implication for the RAG service |
|---|---|---|
| Ranked lexical search | zvec FTS collection; docs describe BM25; schema uses `jieba` and lowercase filtering | Useful for names, product terms, and identifiers; test Chinese segmentation and mixed identifiers |
| Semantic search | Model-generated vectors; storage schema uses FP32 HNSW; catalog currently uses cosine similarity | Finds paraphrases and concepts absent from literal query wording |
| Hybrid fusion | Search pipeline has `RRF_K = 60` and sums `1 / (60 + rank)` across recall routes | Combines rankings without assuming lexical and vector scores are comparable |
| Candidate expansion | Node search starts at recall depth 200, can grow to 2,000, then materializes the requested limit | Useful implementation detail; not a throughput or recall guarantee |
| Multi-query retrieval | `query`, `queries`, `fts`, `vector`, and `fuse` in MCP schema | Supports intent plus lexical anchors; route planning still matters |
| Exact/regex search | Managed ripgrep scans files without a vector index | Useful for verification and exhaustive occurrences; ranked FTS is not exhaustive grep |
| Structural extraction | Tree-sitter code entities and Markdown heading sections | Better evidence boundaries for repositories and structured documentation |
| Scope controls | Path globs, file types, symbols, modification-time bounds | Helpful retrieval filters; not equivalent to authenticated document ACLs |
| Evidence output | File locations, matched ranges, snippets, freshness state | Good citation building blocks; normalized exports need mapping back to original sources |
| Index lifecycle | Initial/incremental builds, watcher refresh, reconciliation, explicit rebuild | Reduces custom indexing work; refresh and deletion behavior still need operational tests |

The inspected ranking path implements rank fusion and deduplication. Documentation uses “reranked” in places, but **this review did not establish a neural cross-encoder reranker**. Treat a learned reranker as an optional external experiment, not a confirmed built-in feature. Graph retrieval is roadmap work.

A semantic search returning neighbors does not prove the knowledge base contains an answer. The RAG layer needs an explicit insufficient-evidence policy. RRF scores are ranking signals, not calibrated confidence probabilities.

### MCP contract: an important correction to the blog-level picture

The default endpoint is `http://127.0.0.1:7999/mcp`, using Streamable HTTP. The default `agent` toolset exposes **one tool, `zvec_grep_search`**. This tool accepts hybrid, lexical-only, and vector-only routes.

The optional `full` toolset exposes six tools: search, managed rg, index, index drop, index status, and server status. Thus keyword and semantic retrieval are available through MCP, but separate regex and administrative tools are **not all exposed by default**. Keep administrative tools outside the answering agent’s toolset.

Example retrieval request:

```json
{
  "root": "/srv/rag/corpus/team-a",
  "query": "How do we recover after a failed release?",
  "fts": ["rollback", "recovery runbook"],
  "fuse": true,
  "limit": 8,
  "preview": "short",
  "freshness": "wait_for_fresh"
}
```

`fts` adds ranked lexical routes; it is not an AND constraint. `preview: "full"` returns available retrieved-item content, not the entire source document. The MCP limit is at most 50 items per group. Use a curated context budget and controlled expansion when snippets are insufficient.

## 4. Ingestion and model feasibility

### Supported inputs

Code, Markdown, plain text, HTML/XML, and text representations of JSON/YAML/CSV/TOML are recognized. CSV and other structured formats largely take the plain-text extraction path, so this is not a structured analytics engine. HTML support does not establish clean DOM-aware extraction, table understanding, or layout preservation.

**PDF, DOC/DOCX, PPT/PPTX, and XLS/XLSX are classified as binary and skipped before extraction.** These formats being present in the roadmap does not mean they work now. For a typical business knowledge base, this is likely the largest integration dependency.

Convert these sources to clean Markdown/text before zg indexing. Retain document ID, canonical URL, version, permission scope, and page/section mappings outside the index. Verify OCR, tables, repeated headers, and page references. Measure extraction coverage separately: a “successful index” can still omit unsupported files or files over size limits.

Images have a separate, opt-in path in Node and require an image-capable embedding model; they are excluded by default. The Rust README says multimodal content is unsupported. Neither path establishes turnkey scanned-PDF OCR.

### Model selection

| Corpus | First local candidate | Follow-up comparison |
|---|---|---|
| English documents | `local/potion-retrieval-32m` | A stronger local Transformer if recall is insufficient |
| Chinese + English documents | `local/potion-multilingual-128m` | `local/multilingual-e5-small` or `local/qwen3-embedding-0.6b`, subject to hardware |
| Mostly source code | `local/potion-code-16m-v2` | `local/jina-embeddings-v2-base-code` |

These names and specifications come from the reviewed catalog, not a quality recommendation validated here. The built-in code-oriented default should not silently become the business-document default. Model2Vec is CPU-oriented; GPU selection does not improve its lookup runtime. Model input limits and fragment truncation can disproportionately affect Chinese and long documents.

Local models require an initial artifact download/cache; documented sources include Hugging Face and a pinned ModelScope fallback. Bake approved artifacts into the deployment or provision an allowed cache. Changing embedding model/vector space requires a rebuild even when dimensions match.

Remote Qwen embedding can be evaluated later if policy allows. It sends corpus fragments during indexing and query text during semantic retrieval, requires separate remote-transfer authorization, and incurs provider cost. Do not assume arbitrary OpenAI-compatible models are plug-and-play: the reviewed catalog/backend is provider-specific even though endpoint overrides exist.

## 5. Proposed PoC architecture

```text
Approved knowledge sources
    ↓ export / parse / OCR / normalize
Versioned Markdown/text corpus + source/permission mapping
    ↓ controlled indexing job
zg workspace index + pinned local embedding model
    ↑ retrieval requests over same-host MCP or engine API
AI RAG service: authentication → corpus selection → query planning
    ↓ evidence validation + context budget + answer model
Answer with citations / insufficient-evidence response
    ↓ evaluation traces, latency, costs, freshness metrics
```

Two integration paths are feasible:

1. **MCP adapter:** run the RAG application and zg daemon on the same host/network namespace; use the existing search tool. This tests agent interoperability with minimal search-specific glue, but requires careful MCP text parsing and client compatibility checks.
2. **In-process Node engine:** wrap exported `createZvecGrep` and typed results. This can simplify a conventional service adapter and citation handling, but inherits embedding runtimes/native dependencies and needs your own lifecycle coordination.

For a service that already has an agent/MCP client, start with path 1. For a conventional API RAG service, consider path 2 after a small integration spike. Avoid CLI text parsing as the long-term service contract.

The daemon rejects non-loopback listen hosts in source. A separate container’s `127.0.0.1` does not reach another container. Co-locate for the PoC, or implement an authenticated gateway with deliberate networking. Simply changing zg to `0.0.0.0` is not a supported deployment plan.

### Access and data boundaries

- Resolve authenticated users to approved corpus roots on the server. Never pass an arbitrary user/LLM-supplied filesystem `root` through to zg.
- Initially use a single approved corpus. If multiple permission groups are required, physically segregate corpora and worker access; glob filtering alone is not an ACL system.
- The optional daemon Bearer token is shared endpoint protection, not per-document authorization. Authentication is disabled by default because the endpoint is local.
- Protect source files, indexes, model cache, logs, and backups through filesystem/deployment controls. The manifest can contain an API key if explicitly persisted; prefer secret injection rather than committing manifests or passing secrets on command lines.
- A local embedding model keeps retrieval processing local. Sending retrieved context to a remote answer LLM is a separate data transfer controlled by your RAG application.
- Treat document content as untrusted evidence. Answer generation must resist instructions embedded in retrieved documents and avoid exposing internal file paths to end users.
- With remote embeddings, authorization revocation affects new operations; source deletion and permission removal also require corpus/index/context-cache invalidation. Use fresh/waited retrieval for sensitive deletion checks.

## 6. Where the value could come from

| Value hypothesis | Why zg might help | What would prove it |
|---|---|---|
| Faster first prototype | Indexer, hybrid retrieval, MCP and evidence formatting already exist | Time to a cited answer versus the team’s normal stack |
| Better recall on ambiguous wording | Semantic candidates complement keyword anchors | Recall@10 by query category; inspect failures |
| Lower agent cost | Focused evidence replaces repeated broad file scans | Actual billed cost and token breakdown per successful answer |
| Easier local operation | Embedded storage and local models avoid a separate database service | Reproducible deployment, restart recovery and resource measurements |
| More useful repository Q&A | Symbols, signatures and headings preserve context | Multi-file questions with correct source citations |

The incremental value will be smaller if your service already has tuned BM25 + vector fusion, clean ingestion, and good citations. MCP is an interoperability benefit; it does not itself improve retrieval quality. zg’s packaging and workspace workflow may then matter more than its ranking algorithm.

### Published evidence: promising, with limits

The checked-in BrowseComp-Plus report describes 100 cases and 300 paired trials:

| Metric | Baseline | zg treatment | Reported change |
|---|---:|---:|---:|
| Answer accuracy | 98.67% | 99.00% | +0.33 percentage points |
| Average input tokens | 1,675,379.50 | 1,046,115.41 | −37.56% |
| Average tool calls | 25.42 | 14.36 | −43.52% |
| Average agent time | 259.39 s | 159.31 s | −38.58% |

These are **maintainer-published results, not independently reproduced here**. The configured embedding is remote `qwen/qwen3.7-text-embedding`; they do not validate local Potion quality. Index preparation is excluded from agent time. Most input tokens in the report are cached, so input-token reduction cannot be translated directly into a billing reduction. Accuracy is already near ceiling; two trials favored baseline only and three favored treatment only, which is weak evidence for a material accuracy improvement.

The report’s 100-case sequence is fixed rather than a random sample of your workload. The root README and benchmark-specific documentation also disagree on BrowseComp reasoning effort (medium versus high); use the pinned benchmark-specific report/configuration for any reproduction.

SWE-QA has separate historical and current CI protocols, different trial counts/agents, and documented aggregate exclusions for large score/token deltas. Treat filtered aggregates as sensitivity analysis and inspect all-case results and failed-attempt costs. These studies support trying zg, not forecasting a universal 38% saving.

## 7. A practical two-week PoC

Estimate: **8–12 engineer-days plus part-time corpus-owner review**, assuming text exports are readily available and an answer model is already accessible. PDF/OCR connectors, enterprise ACLs, or platform approval can extend this substantially. This is a planning estimate, not a measured delivery commitment.

| Stage | Work | Deliverable |
|---|---|---|
| Days 1–2 | Pin Node snapshot and model; create corpus inventory; provision cache; test MCP client | Reproducible image/setup and successful real retrieval call |
| Days 3–4 | Normalize 500–2,000 representative documents; preserve source mappings; build index | Coverage report, build time, peak RSS and disk usage |
| Days 5–6 | Add retrieval adapter, bounded context, citations and no-answer policy | End-to-end RAG API for the approved corpus |
| Days 7–9 | Run retrieval and answer comparisons; test updates/deletion/restart/concurrency | Per-category quality, latency and cost results |
| Day 10 | Review failures and integration effort; decide continue/change/stop | Evidence-based go/no-go and next architecture |

Suggested first host: Linux x86-64, 4–8 vCPU, 16 GB RAM, SSD, CPU-only small local model. This is a starting test configuration, not a documented minimum or scale guarantee. Measure before sizing. FP32 raw vectors alone cost approximately `fragment_count × dimensions × 4 bytes`; one million 512-dimensional vectors occupy about 2.05 GB before HNSW, text, metadata, and runtime overhead. Document count is not vector count.

### Evaluation design

Create 100 reviewed questions before tuning. Include exact terms/IDs, paraphrases, Chinese and mixed-language queries where relevant, multi-document synthesis, table-derived facts, stale/deleted evidence, and questions with no answer. Keep gold sources and answer rubrics outside the agent-visible corpus. Use a frozen development/held-out split.

Compare:

- **A:** current RAG service, if available; otherwise a straightforward lexical baseline.
- **B:** zg FTS only.
- **C:** zg vectors only with the selected local model.
- **D:** zg hybrid using that same model.

Run a separate optional remote-model comparison only after authorization. Keep corpus, answer LLM, prompt, context budget, and timeouts fixed. First compare retrieval alone; then run paired end-to-end answer trials, preferably three repetitions for agentic flows. Do not use the held-out set for tuning.

Measure Recall@10, nDCG@10, grounded answer correctness, citation correctness, abstention on unanswerable queries, warm/cold p50/p95 latency, peak memory/disk, index duration, refresh/deletion lag, concurrent-query errors, and billed cost per successful answer. Include extraction, embedding, index amortization, cache rates, and failed attempts in cost accounting.

### Proposed acceptance gates — targets to agree, not observed results

| Gate | Initial target |
|---|---|
| Eligible ingestion coverage | ≥98%; every skipped/failed input explained |
| Retrieval | Recall@10 ≥85%; hybrid improves ≥5 percentage points over lexical on paraphrase subset |
| Answer quality | ≥85% correct and grounded; no >2 pp regression versus existing service |
| Citations | ≥95% accurately support the associated claim |
| Unanswerable questions | ≥90% correctly abstain or request clarification |
| Warm retrieval | p95 ≤1 s at concurrency 5 on the chosen host/corpus; report cold starts separately |
| Freshness | Small approved updates searchable within 60 s; fresh/waited reads honor deletion |
| Access boundary | Zero unauthorized evidence in designed cross-corpus tests |
| Economics | ≥20% lower billed answer cost at comparable quality, **or** documented implementation/operations savings justify adoption |

If retrieval quality misses targets, inspect extraction, fragment truncation, model language coverage, and query routes before adding orchestration complexity. Stop using zg for a shared service if isolation cannot be reliably enforced or if measured latency/resource behavior conflicts with requirements.

## 8. Alternatives and decision logic

| Option | Prefer it when | Main tradeoff |
|---|---|---|
| zg | Local corpus, agent tools, code/docs, rapid single-tenant trial | Service controls and rich ingestion remain external |
| Direct zvec SDK | Embedded search is useful but the product needs custom records/schema/ranking | More implementation work; finer control |
| PostgreSQL + pgvector + text search | Existing relational platform and transactional business metadata/permissions | You implement fusion and ingestion; language/ranking needs evaluation |
| Dedicated search/vector platform | Shared deployments, rich filters, independent scaling and operations dominate | More infrastructure; capabilities vary and must be checked |

The key decision is whether zg’s ready-made workspace retrieval meaningfully reduces your delivery and operating cost while meeting your corpus and permissions needs. A two-week trial is justified; committing to a production platform before these measurements is premature.

## 9. Sources and evidence trail

All repository links below are pinned to the reviewed commit:

- [README and project relationship](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/README.md)
- [Package version, runtime, dependencies and exports](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/package.json)
- [MCP tools and schema](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/docs/03-mcp.md)
- [Pipeline, extraction, skipped formats and index behavior](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/docs/04-pipeline.md)
- [Server transport, refresh and authentication](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/docs/06-server.md)
- [Models and remote authorization](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/docs/07-embedding.md)
- [Roadmap and preview status](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/docs/08-roadmap.md)
- [Source: RRF and candidate recall](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/src/engine/pipeline/search/index.ts)
- [Source: FTS tokenizer and HNSW schema](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/src/engine/storage/zvec.ts)
- [Source: binary format exclusions](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/src/engine/file-type.ts)
- [Source: exported engine types](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/src/engine/service/types.ts)
- [Source: loopback-only server](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/src/daemon/http-server.ts)
- [BrowseComp-Plus full report](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/benchmarks/browse-comp-plus/LATEST_REPORT.md)
- [SWE-QA protocol and exclusions](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/benchmarks/swe-qa-bench/README.md)
- [Rust implementation and compatibility boundaries](https://github.com/zvec-ai/zvec-grep/blob/30c316052f4298ff6fac1e71f45ea40606e24723/rust/README.md)

## 10. Local verification

Verification results are recorded below after the selected checks complete. They distinguish functional tests from real-model retrieval-quality validation.

- Repository cloned successfully; reviewed commit and root package version verified locally.
- Host runtime: Node.js `v24.19.0`, npm `11.9.0`.
- Standard `npm ci --no-audit --no-fund` failed in `onnxruntime-node` while attempting to download CUDA binaries from GitHub (`EAI_AGAIN`). This dependency installer did not successfully use the environment’s proxy path. This is an environment/install constraint, not proof that zg cannot run on a normal host.
- `npm ci --ignore-scripts --no-audit --no-fund` succeeded (388 packages). This workaround disables all lifecycle scripts; it is **not a fully verified normal installation**. The ONNX installer source documents `ONNXRUNTIME_NODE_INSTALL_CUDA=skip` for a future CPU-only normal-install retry.
- `npm run build` succeeded.
- Selected upstream tests completed: **70 passed, 0 failed, 0 skipped** in approximately 65.5 seconds. Command:

```sh
node --test --test-concurrency=1 \
  test/unit/search.test.mjs \
  test/mcp-contract.test.mjs \
  test/unit/zvec-storage.test.mjs \
  test/integration/service.test.mjs
```

These exercise ranking/routes and filters, MCP schemas/tool behavior with an in-memory backend, real native zvec storage behavior, and indexing/service behavior with controlled fixtures. They are useful functional evidence; deterministic/fake embedding fixtures do not validate semantic quality. This did not constitute an external HTTP MCP client end-to-end test, a real catalog-model download/inference trial, or a load test.

No private corpus or credentials were used, no workspace content was sent to a remote embedding provider, and no agent configuration was installed. The full CI gate, Rust build, production deployment, and published benchmark reproduction were not run.

Local evidence: `/workspace/zvec-grep-validation.log`. Repository checkout: `/workspace/zvec-grep`. The analysis is saved separately from the checkout so it is not accidentally added to the retrieval corpus.
