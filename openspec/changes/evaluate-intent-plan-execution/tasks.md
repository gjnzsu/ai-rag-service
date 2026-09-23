## 1. Freeze the Comparison Baseline

- [ ] 1.1 Extend the existing Agentic evaluation questions with independent intent and blocker-constraint labels while preserving the current oracle facts.
- [ ] 1.2 Add reviewed paraphrases and parallel combinations for completion, direct Bugs, and blocker population/direction without adding abstract health or open investigation cases.
- [ ] 1.3 Run and save a fresh bounded ReAct baseline with frozen snapshot, implementation hashes, model settings, budgets, ordering controls, and per-run usage.

## 2. Define Intent Planning Contracts

- [ ] 2.1 Add strict models for supported intents, unsupported obligations, blocker population/direction, planner status, and content-safe usage diagnostics.
- [ ] 2.2 Implement one AI Gateway planning call with a timeout, no tools, one accepted response, and no normal-path replan.
- [ ] 2.3 Validate that trusted project/Epic scope and concrete issue keys cannot be supplied or changed by planner output.
- [ ] 2.4 Enforce and test the planner input boundary: original question and trusted subject only, with no backlog content or observations.
- [ ] 2.5 Add planner tests for single and parallel intents, paraphrases, duplicates, empty-result categories, partially unsupported questions, invalid enums, malformed output, timeout, and missing usage.

## 3. Build Deterministic Execution

- [ ] 3.1 Compile each accepted intent plan into known prerequisites while sharing one Epic backlog read across completion, Bugs, Story selection, and blocker analysis.
- [ ] 3.2 Derive all/all-unfinished Story populations and dependency directions from the validated plan plus trusted backlog observations.
- [ ] 3.3 Execute required dependency lookups through existing scoped tools while preserving pagination, traversal, time, node, path, and logical-call budgets.
- [ ] 3.4 Add executor tests for all-Done populations, no direct Bugs, inbound/outbound/both directions, lookup failure, partial traversal, truncation, and budget exhaustion.

## 4. Make Coverage Explicit

- [ ] 4.1 Initialize a ledger entry for every planned supported or unsupported obligation before dependent execution.
- [ ] 4.2 Derive complete, partial, unavailable, and unsupported states from verified observations and existing completeness diagnostics.
- [ ] 4.3 Ensure complete empty results are distinguishable from failures and prevent unavailable or partial blockers from becoming “no blockers” claims.
- [ ] 4.4 Preserve completed peer intents when another obligation fails, and add report/generation tests for mixed coverage states.

## 5. Integrate the Experimental Workflow

- [ ] 5.1 Add a default-off workflow selector that preserves the existing bounded ReAct coordinator and public Agentic request contract.
- [ ] 5.2 Reuse existing evidence consolidation, provenance, citation validation, final generation, execution accounting, and original-question handling.
- [ ] 5.3 Implement and test one-way planner-failure fallback to the existing bounded ReAct coordinator, explicitly marking native versus fallback-assisted results and accounting for both stages' overhead.
- [ ] 5.4 Add integration and API tests proving unchanged default behavior, identical scope enforcement, and content-safe diagnostics.

## 6. Evaluate and Decide

- [ ] 6.1 Add paired bounded-ReAct and Intent Plan-and-Execute arms with identical questions, snapshots, tools, budgets, final generator, model pins, and run-order controls.
- [ ] 6.2 Report per-case intent/constraint match, coverage, findings, factual and citation checks, logical/backend calls, model calls, latency, and provider-reported usage with unknown values kept null.
- [ ] 6.3 Review every disagreement, fallback, partial result, and generated answer for omitted intent, false completeness, incorrect blocker direction/population, or unsupported claims.
- [ ] 6.4 Publish a recommendation against the quality and efficiency gates, state dataset limitations, and leave the existing workflow as default pending a separate promotion decision.

## 7. Verification

- [ ] 7.1 Run focused planner, executor, coverage, report, API, and evaluation tests plus lint on every changed Python file.
- [ ] 7.2 Run the repository quality gate and strict OpenSpec validation, recording intentionally skipped optional integrations and the final implementation revision.
