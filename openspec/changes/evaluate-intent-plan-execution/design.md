## Context

The current Agentic Epic coordinator always reads the scoped Epic backlog, then runs up to three decision rounds. In each round the decider receives the user question, current observations, requested analyses, coverage, remaining budget, and the previous action error. It may request a dependency lookup or finish. Its action schema already limits analysis to `completion`, `bugs`, `blockers`, and `unsupported`.

The workflow is therefore bounded ReAct rather than an open agent. For the accepted evaluation questions, execution dependencies are already predictable:

```text
completion -> epic backlog
bugs       -> epic backlog
blockers   -> epic backlog -> select story population -> dependency lookup
```

The existing experiment showed equal necessary retrieval counts for Agentic and fixed workflows while the Agentic arm added decision calls, tokens, and latency. This change tests a narrower architectural hypothesis: understand all explicit parallel intents once, then let trusted code schedule their known dependencies.

“Parallel intents” describes answer obligations, not necessarily concurrent database calls. A question requesting completion, bugs, and blockers has three peer obligations, while blocker execution still depends on first reading the backlog and selecting the requested Story population.

## Goals / Non-Goals

**Goals:**

- Extract the complete supported intent list once before optional dependency retrieval.
- Separate model-owned semantic interpretation from code-owned execution ordering.
- Share prerequisite reads across intents and avoid repeated model decisions for known paths.
- Track and report coverage independently for every requested intent.
- Compare Plan-and-Execute with the current bounded ReAct workflow using controlled, repeatable evidence.
- Preserve current scope, direction, pagination, budget, provenance, citation, and failure controls.

**Non-Goals:**

- Replace or enable a new default workflow without evaluation.
- Generate rewritten natural-language retrieval queries or fan out hybrid/vector searches.
- Support arbitrary new analysis types, business-defined health/risk judgements, or conversational references.
- Support compound intersection queries beyond the existing blocker population/direction constraints.
- Follow unknown cross-Epic paths, dynamically discover an unbounded tool chain, or generalize into an open agent framework.
- Change the existing API request schema, Graph snapshot model, authorization model, or final report contract.

## Decisions

### 1. Plan answer obligations, not raw tool calls

The model produces a strict `IntentPlan`, not a sequence of model-authored tool invocations. A plan declares the requested intent set and the supported constraints needed to interpret it:

```json
{
  "intents": ["completion", "bugs", "blockers"],
  "blocker_population": "unfinished_stories",
  "blocker_direction": "inbound"
}
```

The subject remains the trusted request `project_key` and `epic_key`; the model cannot replace it. Duplicate intents are normalized and unsupported content is represented explicitly rather than coerced into a supported analysis.

Alternative considered: ask the model to emit ordered tool calls. Rejected because the model could duplicate the mandatory backlog read, mishandle pagination, select out-of-scope issues, or bypass coverage rules.

### 2. Compile the plan into a deterministic dependency graph

Trusted code maps each supported intent to known prerequisites. The executor reads the Epic backlog once, derives completion and direct-bug facts from it, selects `all_stories` or `unfinished_stories` for blocker analysis, and performs the required direction lookups using the existing scoped tools.

The initial experiment does not execute intents as independent pipelines and does not require concurrency. Sharing the backlog observation is more important than parallel wall-clock execution, and the existing budgets remain authoritative.

### 3. Create the coverage ledger from the accepted plan

Every planned intent receives a ledger entry before dependent execution begins. Each entry ends in one of:

- `complete`: required data was retrieved without completeness limitations;
- `partial`: useful evidence exists but pagination, truncation, budget, or partial traversal prevents completeness;
- `unavailable`: the required retrieval failed or could not run;
- `unsupported`: the request contains an analysis obligation outside the experiment taxonomy.

An empty, complete lookup may support “none found.” A failed, skipped, or partial lookup must never be interpreted as an empty fact. Completed peer intents remain reportable when another intent is unavailable.

### 4. Keep backlog content out of intent planning

The system retains the mandatory backlog read before the model call so invalid or out-of-scope subjects fail early and the result can be reused by execution. However, the planner receives only the original question and trusted `project_key`/`epic_key`; it does not receive Story, Bug, status, count, or relationship observations from the backlog.

The planner determines what the user requested independently of whether the resulting population is empty. For example, an explicit Bug-analysis request remains in the plan even when the backlog contains no direct Bugs; the executor then records a complete empty result. The executor—not the model—uses the previously loaded backlog to derive concrete issue keys and schedule dependency calls.

Alternative considered: provide backlog observations to the planner. Rejected because observed data can bias the model to omit an explicitly requested intent when its current result appears empty, mixes interpretation with execution, increases prompt size, and makes intent accuracy harder to evaluate independently.

### 5. Use one planning call and no normal-path replan

The experimental arm makes one strict-schema planning call after subject validation and the internal backlog read, without exposing backlog content to the planner.

Malformed, unavailable, out-of-scope, or internally inconsistent plans do not trigger an open retry loop. They fall back once to the existing bounded ReAct coordinator. Evaluation marks the result as fallback-assisted rather than native Plan-and-Execute success and includes both planning and fallback overhead. Whether a single bounded replan is valuable remains a later question and is deliberately excluded from the first comparison.

Alternative considered: Plan–Execute–Replan. Deferred because the current intent taxonomy and execution graph do not yet demonstrate a need for dynamic replanning.

### 6. Preserve current evidence and generation boundaries

Both experimental arms use the same `AgentTools`, Graph service, evidence consolidation, citation validation, final generator, budgets, and frozen snapshot. The final generator receives the original question plus the verified structured result; it never treats a plan declaration as evidence.

### 7. Evaluate against a fixed baseline and oracle

The comparison includes:

- A: existing bounded ReAct coordinator;
- B: one-shot Intent Plan plus deterministic executor;
- optionally, the existing fixed/oracle workflow as a diagnostic lower-bound on decision overhead, not as a deployable natural-language solution.

Each arm receives the same natural-language question. The oracle is used only for evaluation assertions and fixed-baseline routing; it is never supplied to A or B. Results report intent identification, constraint identification, per-intent coverage, evidence/finding correctness, logical and backend tool calls, decision/generation calls, latency, and provider-reported token usage.

Promotion requires no regression in automatic checks or reviewed answer semantics, complete recognition of every supported requested intent in the frozen set, no false `complete` or false negative blocker claim, and a measured reduction in decision overhead or latency. Passing gates does not automatically change the default.

### 8. Keep diagnostics content-safe and auditable

Execution traces record bounded intent names, constraints, coverage states, tool names, issue keys already permitted by the existing response boundary, timing, and usage. Logs do not add raw prompts, model reasoning, or retrieved text. Unknown usage remains unknown rather than zero.

## Risks / Trade-offs

- **The initial plan omits an intent** → Compare against frozen oracle labels, initialize coverage only after schema validation, and fail the experimental quality gate on any supported-intent omission.
- **A rigid plan loses ReAct adaptability** → Limit the experiment to the current predictable intent taxonomy; classify out-of-scope requests as unsupported rather than inventing a path.
- **Plan says blockers but executor queries the wrong population/direction** → Validate enum constraints and derive issue keys exclusively from trusted backlog facts.
- **One failed intent contaminates the whole report** → Maintain per-intent coverage and preserve evidence for completed peers.
- **Architecture improves but user quality does not** → Require measured overhead reduction with no quality regression; do not promote based on code elegance.
- **Test questions are too narrow** → Add paraphrases and mixed parallel-intent combinations while keeping independent oracle labels; state representativeness limits.
- **The fixed executor becomes a growing rules engine** → Keep this proposal limited to existing intents and require a new design decision before adding materially different analysis dependencies.

## Migration Plan

1. Extend the frozen Agentic question set with explicit intent and constraint labels plus additional parallel-intent paraphrases.
2. Capture a fresh bounded ReAct baseline using the current implementation and pinned environment.
3. Add strict plan models and one-shot planner behind an experimental workflow selector, with one-way fallback to the existing bounded ReAct coordinator on planning failure.
4. Add the deterministic executor and per-intent ledger using existing tools and reporting boundaries.
5. Run paired comparisons and review every disagreement and partial result.
6. Publish a recommendation and leave the current workflow as default unless a separate promotion decision is approved.

Rollback is configuration-only because the existing coordinator remains intact during the experiment.

## Open Questions

- How many new paraphrases and intent combinations are sufficient to reduce overfitting to the existing 12-question set?
- If future queries require result-dependent discovery, should that become one bounded replan or remain a separate ReAct-only route?
