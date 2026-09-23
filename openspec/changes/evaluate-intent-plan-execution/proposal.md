## Why

The accepted Agentic Epic PoC uses a bounded ReAct loop to interpret natural-language questions and choose follow-up Graph dependency lookups. Its current query space is narrow and predictable—completion, direct bugs, and blockers—and the controlled experiment found the Agent used the same necessary retrieval operations as the fixed workflow while adding 84.6% more reported tokens and 31.2% median latency. We need to test whether one structured multi-intent plan followed by deterministic execution can preserve completeness and correctness with fewer decision calls.

## What Changes

- Add an experimental one-shot intent planner for explicit, parallel Epic-analysis intents.
- Represent requested analysis as a bounded structured plan covering the existing `completion`, `bugs`, and `blockers` capabilities and their supported constraints.
- Compile the intent plan into a deterministic execution graph that shares the mandatory Epic backlog read and performs dependency lookups only where required.
- Create a per-intent coverage ledger before execution and require every requested intent to end as complete, partial, unavailable, or unsupported.
- Keep the existing bounded ReAct workflow as the comparison baseline and evaluate both workflows against the same frozen questions, tools, snapshot, evidence rules, generator, and budgets.
- Preserve evidence and partial-result behavior: an unavailable blocker lookup must not erase completed completion/bug results or be reported as “no blockers.”
- Keep the Plan-and-Execute workflow experimental and disabled by default until measured results justify a separate promotion decision.
- Exclude free-text query rewriting, vocabulary expansion, conversational memory, abstract business concepts such as Epic health, and arbitrary multi-step investigation.

## Capabilities

### New Capabilities

- `intent-plan-execution-experiment`: Structured parallel-intent planning, deterministic dependency-aware execution, per-intent coverage verification, bounded fallback, and comparison with the existing ReAct workflow.

### Modified Capabilities

None. The experiment does not change the archived `agentic-epic-analysis` requirements or default runtime behavior.

## Impact

- `app/agentic` gains an alternative planner/executor orchestration path using the existing scoped tools, evidence builder, report generator, and request boundary.
- The Agentic evaluation harness gains a Plan-and-Execute arm and per-intent plan/coverage assertions.
- Configuration gains a default-off experimental workflow selector without changing the public Agentic request schema.
- No changes are required to hybrid text retrieval, Graph storage, snapshots, public `/retrieve` behavior, or persistent data.
