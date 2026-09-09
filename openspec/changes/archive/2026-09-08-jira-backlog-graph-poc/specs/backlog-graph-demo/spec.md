## ADDED Requirements

### Requirement: Three independent demonstration operations
The demo SHALL provide project overview, Epic drilldown and dependency evidence views via platform APIs, without conversational state, automatic follow-ups or Jira mutation. It SHALL be disabled by default and intended for local controlled use.

#### Scenario: Independent dependency demonstration
- **WHEN** a viewer selects an issue and requests dependencies
- **THEN** a local relation view and evidence list are displayed without requiring a previous conversation

### Requirement: Honest presentation
The demo SHALL show snapshot scope and time, item statuses, source links, truncation, empty results and errors. It MUST distinguish recorded relations from unresolved pairs and avoid presenting missing relations as proof of no real-world dependency.

#### Scenario: No recorded cross-Epic links
- **WHEN** the snapshot has no cross-Epic dependency edges
- **THEN** the demo reports no such links recorded in this snapshot without claiming the product has no dependencies

### Requirement: Reproducible capability evaluation
The PoC SHALL compare hybrid plus existing exact lookup, simple parent/filter queries, and graph-assisted retrieval on a fixed snapshot with matched generation settings. Real and synthetic cases SHALL be reported separately, including exact structural checks, evidence coverage, answer correctness, timing and observed costs.

#### Scenario: Sparse real graph
- **WHEN** the real snapshot has only one dependency link
- **THEN** the report limits real-data conclusions accordingly and labels cycle and multi-hop fixtures as synthetic

#### Scenario: Acceptance demonstration
- **WHEN** the three demonstration steps are evaluated against a frozen snapshot
- **THEN** hierarchy, counts and explicit edge directions match that snapshot, citations resolve to their sources, and completed pairs are never presented as current unresolved blockers
