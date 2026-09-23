## ADDED Requirements

### Requirement: Experimental workflow is opt-in
The system SHALL keep the existing bounded ReAct Agentic workflow as the default. The Intent Plan-and-Execute workflow SHALL be selectable only through an explicit experimental configuration and SHALL preserve the existing public Agentic request contract.

#### Scenario: Default request
- **WHEN** the experimental workflow is not selected
- **THEN** the system executes the existing bounded ReAct coordinator without an Intent Plan call

#### Scenario: Experimental request
- **WHEN** the Intent Plan-and-Execute workflow is explicitly selected
- **THEN** the system executes the experimental planner and deterministic executor within the existing request scope and budgets

### Requirement: Planner declares all supported parallel intents once
The planner SHALL return one strict structured plan containing every supported analysis obligation expressed by the question. Supported obligations SHALL be limited to `completion`, `bugs`, `blockers`, and explicit `unsupported` content for this experiment. The planner SHALL receive the original question and trusted project/Epic subject but MUST NOT receive backlog content or facts.

#### Scenario: Three parallel intents
- **WHEN** a question asks for Story completion, direct Bugs, and blockers
- **THEN** the accepted plan contains `completion`, `bugs`, and `blockers` exactly once

#### Scenario: Rephrased parallel intents
- **WHEN** a question expresses supported obligations using a labelled paraphrase
- **THEN** the plan identifies the same obligations without generating rewritten retrieval queries

#### Scenario: Partially unsupported question
- **WHEN** a question asks for a supported completion analysis and an unsupported health rating
- **THEN** the plan preserves the completion obligation and explicitly represents the unsupported obligation

#### Scenario: Requested category is empty in the backlog
- **WHEN** the question explicitly requests direct Bug analysis but the internally loaded backlog contains no direct Bugs
- **THEN** the plan still contains the `bugs` obligation because backlog content is not used to decide what the user requested

#### Scenario: Planner input boundary
- **WHEN** the system invokes the Intent Planner after validating the subject and loading the backlog internally
- **THEN** planner input contains the original question and trusted project/Epic subject but no Story, Bug, status, count, relationship, or other backlog observations

### Requirement: Planner captures supported blocker constraints
For blocker analysis, the plan MUST identify whether the population is `all_stories` or `unfinished_stories` and whether the direction is `inbound`, `outbound`, or `both`. The model MUST NOT supply database queries, arbitrary scope, or trusted subject identifiers.

#### Scenario: Unfinished inbound blockers
- **WHEN** the question asks which unfinished Stories are blocked by other issues
- **THEN** the plan selects `unfinished_stories` and `inbound`

#### Scenario: Invalid blocker constraint
- **WHEN** planner output contains a population or direction outside the supported enums
- **THEN** the system rejects the plan without executing model-selected issue keys or database queries

### Requirement: Executor resolves known dependencies deterministically
The executor SHALL compile the accepted intent plan into known prerequisite operations. It SHALL read the Epic backlog once, reuse that observation for completion, direct Bugs, and Story selection, and derive dependency lookup issue keys from the trusted backlog rather than planner output.

#### Scenario: Shared backlog prerequisite
- **WHEN** the plan requests completion, Bugs, and blockers
- **THEN** the executor performs one logical Epic backlog read and reuses it for all three obligations

#### Scenario: Blockers for unfinished Stories
- **WHEN** the plan requests blockers for `unfinished_stories`
- **THEN** the executor derives the non-Done Story keys from the complete backlog and queries only those keys in the requested direction

#### Scenario: Empty unfinished population
- **WHEN** the complete backlog contains no unfinished Stories and the plan requests blockers for `unfinished_stories`
- **THEN** the executor performs no dependency lookup and marks blocker coverage complete with an empty applicable population

### Requirement: Coverage is tracked per planned intent
The system SHALL create a coverage ledger for all planned obligations and SHALL assign each obligation a final state of `complete`, `partial`, `unavailable`, or `unsupported`. It MUST distinguish a complete empty result from an incomplete or failed lookup.

#### Scenario: All obligations complete
- **WHEN** backlog and all required dependency lookups complete without limitations
- **THEN** every supported planned intent is marked complete

#### Scenario: Dependency lookup fails
- **WHEN** completion and Bug facts are complete but the required dependency lookup fails
- **THEN** completion and Bug coverage remain complete while blocker coverage is unavailable

#### Scenario: Dependency traversal is truncated
- **WHEN** a required dependency result is truncated or partial
- **THEN** blocker coverage is partial and the result MUST NOT claim that no additional blockers exist

#### Scenario: Unsupported peer obligation
- **WHEN** one planned obligation is unsupported and another supported obligation completes
- **THEN** the system reports unsupported and complete independently rather than failing or completing the whole question uniformly

### Requirement: Planner and executor failures remain bounded
The experimental workflow SHALL use a strict-schema planning call through the configured AI Gateway with a timeout and no tools. It SHALL NOT enter an unbounded decision or replan loop. Planner or executor failure SHALL preserve safe partial results and explicit coverage limitations.

#### Scenario: Planner response is malformed
- **WHEN** the planner response violates the strict Intent Plan schema
- **THEN** the system records planning failure and falls back once to the existing bounded ReAct coordinator without treating the failed plan as complete

#### Scenario: Planner times out
- **WHEN** the single planning call exceeds its configured timeout
- **THEN** the system does not make another planning call, falls back once to the existing bounded ReAct coordinator, and records the fallback and usage outcome

#### Scenario: Tool budget is exhausted
- **WHEN** the deterministic execution graph requires an operation after the request tool budget is exhausted
- **THEN** the affected intent is partial or unavailable and no out-of-budget operation runs

### Requirement: Existing evidence and generation trust boundaries are preserved
The experimental workflow SHALL use existing scoped Graph tools, provenance validation, evidence consolidation, citation validation, and final generation boundaries. Planner declarations SHALL NOT be treated as factual evidence, and the final generator SHALL receive the original question.

#### Scenario: Planned fact lacks retrieved evidence
- **WHEN** the plan requests an analysis but its required fact is not present in verified tool observations
- **THEN** the final result does not present the planned analysis as a supported fact

#### Scenario: Final generation runs
- **WHEN** verified results are passed to the final generator
- **THEN** the generator receives the original user question and only server-known evidence identifiers

### Requirement: Comparison uses fixed inputs and independent assertions
The evaluation SHALL compare bounded ReAct and Intent Plan-and-Execute using the same natural-language questions, frozen snapshot, scoped tools, budgets, evidence rules, final generator, pinned models, and run conditions. Oracle intent and constraint labels SHALL be used only for independent evaluation and fixed-baseline routing.

#### Scenario: Paired evaluation run
- **WHEN** both workflows are evaluated
- **THEN** the report records per-case plan or action decisions, requested-intent match, constraint match, coverage, findings, logical and backend calls, decision and generation calls, latency, and provider-reported token usage

#### Scenario: Provider omits usage
- **WHEN** a planning, decision, or generation call has no provider-reported usage
- **THEN** the report records usage as unknown rather than zero

#### Scenario: Fallback occurs
- **WHEN** the experimental workflow falls back to bounded ReAct or another safe path
- **THEN** the report distinguishes native Plan-and-Execute success from fallback-assisted success

### Requirement: Promotion requires equal quality and measured efficiency benefit
The experiment SHALL NOT recommend Intent Plan-and-Execute unless it recognizes every supported requested intent and constraint in the frozen evaluation set, produces no false complete coverage or false negative blocker claim, passes the existing factual and citation checks, and demonstrates a measured decision-call, token, or latency benefit. A passing result SHALL NOT automatically change the default workflow.

#### Scenario: Quality is equal and overhead improves
- **WHEN** Intent Plan-and-Execute satisfies every quality gate and shows a repeatable efficiency improvement
- **THEN** the report marks it eligible for a separate default-workflow decision

#### Scenario: Intent is omitted
- **WHEN** the plan omits any supported requested intent in the evaluation oracle
- **THEN** the experiment fails its promotion gate regardless of efficiency improvements

#### Scenario: No efficiency improvement
- **WHEN** quality is preserved but no decision-call, token, or latency benefit is measured
- **THEN** the report recommends retaining the existing workflow pending another justified benefit
