## ADDED Requirements

### Requirement: Optional public Epic analysis
The service SHALL expose a default-disabled, read-only Agentic query capability accepting project key, Epic key, question and optional snapshot ID. It MUST preserve existing retrieval/query behavior and bind one allowed snapshot for the entire request.

#### Scenario: Analyze any valid Epic
- **WHEN** a caller selects any Epic in an allowed published project snapshot
- **THEN** the same workflow operates without hardcoded sample keys and every tool result belongs to the bound scope

#### Scenario: Invalid subject or changing active snapshot
- **WHEN** the subject is missing, is not an Epic or is outside the allowed scope
- **THEN** the request is rejected before model execution
- **AND** a valid request already in progress MUST retain its initial snapshot if active changes

#### Scenario: Feature disabled
- **WHEN** the Agentic capability is disabled
- **THEN** its endpoint is unavailable and existing APIs retain their previous behavior

### Requirement: Deterministic backlog statistics
The service MUST retrieve paginated direct children within a finite budget, separate Stories from Bugs, and calculate Story completion using status category rather than model arithmetic. It MUST distinguish full statistics from partial observations.

#### Scenario: Mixed children across pages
- **WHEN** an Epic has Stories and Bugs across multiple pages
- **THEN** the Story denominator excludes Bugs, every available page within budget is processed and only category done counts as completed

#### Scenario: Empty or incomplete Story population
- **WHEN** no Stories exist or the complete population cannot be obtained
- **THEN** completion rate is null with not-applicable or partial explanation respectively, never a fabricated full-population percentage

### Requirement: Bounded single-agent retrieval decisions
The coordinator SHALL perform an initial backlog read and then accept validated model decisions to retrieve dependencies or finish. It MUST enforce finite model/tool/time budgets independently of model output and retain no cross-request conversational memory.

#### Scenario: Simple completion question
- **WHEN** complete backlog evidence answers a completion-only question
- **THEN** the model can finish without a dependency lookup and the service records the stop reason

#### Scenario: Invalid or repeated action
- **WHEN** the model requests an unauthorized tool, scope change, invalid argument or duplicate lookup
- **THEN** the action is not executed, the decision consumes its bounded round and repeated correction cannot create an unbounded loop

#### Scenario: Budget or decision-provider failure
- **WHEN** the next action exceeds the remaining budget or the decision model fails
- **THEN** the loop terminates with an explicit reason, retains validated evidence and reports incomplete requested coverage

### Requirement: Scoped dependency evidence
The dependency tool SHALL return one-hop directed BLOCKS evidence, both endpoint statuses/types, provenance and per-seed coverage for a bounded batch. The service MUST restrict seeds to request-validated issues and maintain one project/snapshot scope.

#### Scenario: Cross-Epic blocker within the project
- **WHEN** a Story is blocked by an issue in another Epic in the same bound project
- **THEN** inbound lookup returns the recorded edge with source and target intact without requiring a Bug to exist under the selected Epic

#### Scenario: Historical or unknown statuses
- **WHEN** a recorded edge has Done/Done endpoints or an unknown status category
- **THEN** the original relationship remains evidence but MUST NOT be classified as both endpoints known unfinished

#### Scenario: Untrusted issue content
- **WHEN** an issue description instructs the model to access another project or execute a non-allowlisted action
- **THEN** the content is treated as data and service-side validation prevents that action

### Requirement: Findings without subjective ratings
The response SHALL include facts, evidence-supported findings, citations, snapshot metadata, per-analysis coverage and an execution summary. Findings MUST state scope, reference evidence and retain limitations. The service MUST NOT emit health ratings, risk scores, personnel evaluations or schedule predictions.

#### Scenario: Supported partial positive conclusion
- **WHEN** one qualifying recorded blocker is found but another required lookup fails
- **THEN** the response can state that the specific blocker exists with citations while overall blocker coverage remains partial

#### Scenario: No evidence is not evidence of absence
- **WHEN** no qualifying blocker is found and any required seed/direction remains unchecked, failed or truncated
- **THEN** the service reports unknown/incomplete coverage and MUST NOT assert that the requested scope has no blockers

#### Scenario: Fully checked negative result
- **WHEN** all required seeds and directions are checked successfully without truncation and none has a qualifying recorded blocker
- **THEN** a negative finding is limited to the checked scope, relationship type and bound snapshot

### Requirement: Evidence-preserving partial failures
The coordinator MUST distinguish execution status, requested evidence coverage and generation status. It SHALL preserve validated structured facts/findings/evidence when later tools or answer generation fail and SHALL validate references after evidence merge.

#### Scenario: Dependency lookup fails after statistics succeed
- **WHEN** Story statistics are complete but dependencies time out
- **THEN** statistics remain available, blocker coverage is unavailable or partial, and no unavailable lookup is presented as an empty successful result

#### Scenario: Final generation fails
- **WHEN** the final model call or citation validation fails
- **THEN** answer is unavailable, verified structured output remains accessible and the failure is explicitly reported

#### Scenario: Safe audit information
- **WHEN** execution metadata is returned
- **THEN** it includes safe tool actions, coverage, elapsed time, stop reason and usage if available, but excludes credentials, private model reasoning and raw prompts

### Requirement: Competitive controlled evaluation
The PoC MUST compare Agentic execution against a deterministic baseline using equivalent tools, fixed snapshot, questions, final generator, evidence rules, budgets and cache conditions. It SHALL measure correctness, evidence support, coverage honesty, decision value and cost without assuming an Agentic advantage.

#### Scenario: Real and synthetic evaluation
- **WHEN** evaluation is performed
- **THEN** it includes multiple real Epics with complete, partial, historical-blocker and current recorded-blocker cases plus synthetic pagination, empty, failure, truncation and unsafe-action cases

#### Scenario: Cost and benefit reporting
- **WHEN** results are summarized
- **THEN** per-question findings and failures, tool/backend/model call counts, token usage and end-to-end timing are reported separately, unavailable usage remains unknown and an equal-quality but more-expensive Agent is a valid outcome
