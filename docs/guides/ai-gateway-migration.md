# RAG model access through AI Gateway

## Configuration

Set `AI_GATEWAY_BASE_URL` to the deployed Gateway's OpenAI-compatible `/v1`
endpoint and `AI_GATEWAY_API_KEY` to its service token. Set
`AI_GATEWAY_APPLICATION_ID` for batch jobs (default `ai-rag-service`).
The runtime and evaluation scripts no longer consume provider credentials.
Missing configuration rejects model calls; structural graph queries can still run.

Local example: `http://localhost:4000/v1`. The existing Kubernetes Gateway Service
is `http://ai-gateway.ai-gateway.svc.cluster.local/v1`. If Kong must enforce quotas,
use the verified Kong service address instead; calling the Gateway Service directly
does not exercise Kong's rate limiting. No cluster address is silently selected.

In Gateway, configure `AI_GATEWAY_SERVICE_TOKEN` with the matching token. It gates
all `/v1/*` routes when set, so update other consumers before enabling it. This
shared token is an initial POC mechanism, not per-application authentication.
Application attribution headers are caller-provided labels, not verified identities.

## Compatibility

Gateway must serve `/v1/embeddings` for `text-embedding-3-small` and
`/v1/chat/completions` for `gpt-5.5-2026-04-23` (and any configured pinned overrides).
Embeddings accept a nonempty string or batch of nonempty strings, with float/base64
encoding and optional positive dimensions. No token-array input contract is provided.
No embedding model fallback or input masking is performed: either would change
vector semantics. Input/output content is not written to the embedding audit log.
Gateway's existing chat security and reliability policies still apply to generation.

Keep model, dimensions and preprocessing unchanged. No reindex is required solely
for changing the endpoint. A model change requires a separate index migration.

RAG forwards request ID, W3C trace headers and application/project/team/use-case/feature
labels at HTTP send time, including pooled clients. Missing request IDs are generated
at the RAG boundary. Consumer service remains `ai-rag-service`. CLI calls retain the
configured application label; Gateway generates their request ID. RAG telemetry is
still independent and should retain its ingestion/retrieval stage instrumentation.

## Rollout sequence and acceptance

1. Deploy Gateway with the new embedding route. Coordinate service-token activation
   across all consumers. Keep provider secrets only in Gateway.
2. Test embeddings and structured chat responses through the actual ingress/service
   using a non-sensitive fixture and the RAG service token. Check the pinned model,
   vector dimension, batch ordering, usage and error response.
3. Configure RAG's Gateway URL/token in its environment or `rag-secrets`; deploy RAG.
   Graph CLI credential files and live evaluation scripts need the same new keys.
4. Exercise ingestion, retrieval, grounded query and enabled Graph/Agentic paths;
   correlate RAG request IDs with Gateway logs and Observability ingestion. Invalid
   tokens must fail, and Gateway outage must never trigger provider-direct calls.
5. Remove obsolete provider keys from RAG's environment/secret and restrict egress
   using the cluster's approved policy. Keep Jira, storage, DNS and telemetry reachable.

This code change does not deploy workloads, edit existing local secrets, remove
cluster credentials, or install a NetworkPolicy. Gateway consumer-model policies
remain log-only; quotas depend on Kong. Cost amounts and complete distributed spans
are not added by this change. Run live acceptance before closing the P0 deployment
item; local mocked tests establish contracts, not cluster connectivity or model quality.

## Local verification (2026-09-21)

From the RAG repository, with `.venv-test/Scripts` prepended to `PATH`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/quality-check.ps1
```

Ruff passed; pytest: 590 passed, 25 skipped. Graph/browser integrations requiring
external infrastructure remain skipped. Test configuration uses synthetic Gateway
credentials and clears the local Jira URL to avoid developer `.env` contamination.

From the Gateway repository:

```powershell
C:/SourceCode/ai-rag-service/.venv-test/Scripts/python.exe -m ruff check app tests
.venv-test/Scripts/python.exe -m pytest -q
```

Ruff passed; pytest: 67 passed. This includes an OpenAI SDK request through the
Gateway ASGI application with a mocked provider. The isolated Gateway test overlay
uses OpenAI SDK 1.109.1 with the existing LiteLLM 1.52.3; RAG retains its existing
OpenAI SDK. Ruff 0.11.13 was used. No provider API calls or cluster deployment were
performed, and the existing dependency pins were not changed.
