# Open Code Review: normal vs delegation mode

Summary insight report · 8 October 2026 · ai-rag-service pilot · OCR 1.12.13

**Recommendation: use delegation for routine, interactive security reviews; use normal mode selectively for a second opinion or future automation.** Both workflows produced useful findings in this pilot. Normal mode did not demonstrate enough additional value or reliable coverage to justify making its current configuration the default.

## What the two modes actually do

Both require the OCR CLI. Delegation avoids configuring an additional model endpoint; it does not avoid model use or the host agent's cost.

| Dimension | Delegation | Normal mode |
|---|---|---|
| Review engine | Host coding agent, Codex in this pilot | OCR-controlled agent calling DeepSeek |
| OCR's role | File selection preview and resolved review rules | File selection, rules, context tools, review execution and result generation |
| Model configuration | Existing host agent session | Separate provider, endpoint, key and model |
| Interaction | Host agent can inspect code, run probes and qualify findings | Produces review comments; validation still requires follow-up |
| Automation potential | Depends on the host agent integration | CLI is suited to standalone execution; CI was not tested here |
| Review data path | Existing host agent session | Reviewed source and retrieved context sent to configured DeepSeek endpoint |
| Cost visibility in this pilot | Host tokens and elapsed time not measured | API token usage recorded; dollar billing not measured |

Our delegation trial was manual: `ocr scan --preview` and `ocr delegate rule` supplied selection and rules, after which Codex inspected code and ran probes. It was not a fully packaged delegation integration. Normal mode used `ocr scan` with security-focused background text; we did not implement a reusable custom security-rule pack.

## What we observed

| Result | Delegation pilot | Normal-mode pilot |
|---|---|---|
| Scope | Targeted inspection of seven rule targets plus surrounding code and deployment configuration | Eight-file attempt, then four-file retry and three-file follow-up |
| Completed coverage | Human-directed targeted review; no automated completion metric | Only ingestion had a completed review; other selected files were failed or skipped |
| Findings | Four reported risk categories | Six comments, mostly splitting those same categories into narrower issues |
| Verification | Three local mocked security probes | OCR generated comments; Codex reran the same three probes and assessed wording |
| Execution issues | Old Git warning did not prevent selection/rules | Old Git broke code search; corrected with bundled Git 2.53 |
| Resource usage | Not measured | 327,351 reported tokens across three attempts, including failed work |

The successful ingestion-file review used 50,548 tokens in 28 seconds. That is not the elapsed time or cost of a complete project review. The 327,351 total includes repeated prompt tokens, and caching changes billing; it cannot be converted into a dollar cost without provider billing details.

## Security findings: agreement and differences

| Risk | Delegation result | Normal-mode result | Assessment |
|---|---|---|---|
| API authentication and collection permissions | Identified together; credential-free deletion dispatch reproduced with mocked storage | Separate authentication and collection-write comments | Strong agreement on missing application controls; public exposure and upstream controls remain unverified |
| Jira query injection | Mocked input-to-JQL path reproduced | Identified via ingestion input and connector context | Agreement; actual impact remains bounded by the Jira service account's permissions |
| Resource limits | Unbounded PDF processing; recommended connector batch bounds | PDF upload limit plus explicit max_results/max_pages comment | Useful specificity, rather than a demonstrated new vulnerability category |
| Error disclosure | Synthetic backend detail reproduced in an HTTP response | Raw exception response pattern identified | Agreement; no actual leaked credential was demonstrated |

Six comments do not mean six independent vulnerabilities or better recall. Authentication and collection authorization are related, and PDF/batch limits share the resource-exhaustion category. No new critical issue was demonstrated by normal mode.

Some normal-mode wording needed correction. Ingestion's collection parameter establishes a write path; that path alone does not prove read access. Severity must consider who can reach the service. Suggested remediation also needs checking: enforce a streaming upload cap rather than trusting Content-Length alone, and do not assume the Jira client exposes parameterized JQL.

## Main insights

**1. Review orchestration and evidence validation are separate jobs.** OCR normal mode can produce credible security leads. The stronger evidence came from tracing input to backend calls and exercising local probes. Neither mode replaces that step.

**2. Coverage matters more than comment count.** The normal result reported four files reviewed while session evidence showed only ingestion completed. A review process should reconcile selected, completed, failed and skipped files before accepting its results.

**3. A token budget is not a hard spending cap.** The 80K and 120K limits overshot during execution/finalization. Even concurrency 1 did not eliminate the problem. Budget headroom and completion checks are necessary for unattended runs.

**4. Configuration materially affects the result.** Git compatibility, model behavior, batching and review passes affected execution. The retry disabled planning, deduplication and summary, so it does not establish the performance of OCR's default configuration.

**5. Security context helps, but generic rules have gaps.** We explicitly named relevant threat categories. The delegated YAML rule only checked key spelling, which is insufficient for Kubernetes security. Deployment review needs tailored checks or a separate scanner.

## Recommended workflow for your PoCs

Use delegation during development to trace sensitive flows, validate findings and implement fixes. Add normal mode as a bounded second opinion on a small diff or file when it is useful. Keep dependency and secret scanning alongside both workflows.

For ai-rag-service, prioritize authenticated and authorized corpus operations, restricted Jira projects and validated keys, bounded ingestion work, and generic client error responses. Confirm the actual network and upstream authentication boundary before assigning deployment severity.

Before adopting normal mode for CI, require working tools, trustworthy coverage accounting, actionable findings and acceptable measured billing on a small repeatable test. Broaden scope only after those checks pass.

## Limits of this comparison

This was a practical pilot, not a controlled benchmark. Different models, unequal completed scopes, modified execution settings and unmeasured delegation costs prevent claims that either mode is intrinsically more accurate, faster or cheaper. Security-focused background was informed by the earlier investigation, so category overlap is not an independent recall measurement. Normal mode was not a whole-project audit, dependency scan or live penetration test.

Existing quality checks passed during the pilots: lint clean, 590 tests passed and 25 skipped. These checks do not prove security. Project code was not modified and no live corpus writes or deployment changes were performed.

Supporting artifacts from the local pilot were retained outside this repository: the detailed delegation report, normal-mode run analysis, and normalized findings JSON.
