## ADDED Requirements

### Requirement: Exact backlog overview and drilldown
The service SHALL expose reusable graph APIs for Epic overview and drilldown with deterministic full-snapshot counts, item statuses, source links and version metadata. It MUST distinguish an Epic's own status from child completion.

#### Scenario: Limited displayed children
- **WHEN** child items are paginated
- **THEN** total counts include all scoped children while the response identifies the displayed page

### Requirement: Bounded directed dependency traversal
The service SHALL support one or two dependency hops with direction, relation type, visited-node cycle control, project scope and at most 100 returned nodes. It SHALL expose truncation and relationship provenance.

#### Scenario: Completed linked issues
- **WHEN** AIPLAT-46 blocks AIPLAT-16 and both have Done status category
- **THEN** the recorded relationship is returned with both statuses and is excluded from unresolved_pair results

#### Scenario: Cycle or result limit
- **WHEN** traversal encounters a cycle or exceeds its limit
- **THEN** it terminates without duplicate expansion and marks any omitted results as truncated

### Requirement: Grounded relationship retrieval
The service SHALL expose stateless /graph/retrieve and /graph/query operations using explicit anchors or same-snapshot hybrid seeds, return paths and supporting source evidence, and preserve edge evidence independently from text ranking. It MUST NOT require an answer-model call for retrieval alone.

#### Scenario: Epic bug question
- **WHEN** a caller asks which Bugs under AIPLAT-13 block Stories with that Epic as anchor
- **THEN** the response includes the recorded AIPLAT-46 to AIPLAT-16 path if present in the selected snapshot, both statuses and source evidence

#### Scenario: Missing explanation text
- **WHEN** an edge exists but source text does not explain its cause
- **THEN** the response states the structural fact without inventing a cause

#### Scenario: Evidence budget exceeded
- **WHEN** all supporting text cannot fit the configured budget
- **THEN** the response marks partial coverage and does not claim an exhaustive generated explanation

### Requirement: Explicit failure and backward compatibility
The graph capability SHALL be disabled by default and preserve existing /retrieve and /query behavior. Structural endpoints SHALL report unavailable graph state rather than fabricate results; graph retrieval MAY degrade only with explicit diagnostics and same-snapshot text.

#### Scenario: Graph backend unavailable
- **WHEN** graph access fails with a valid text snapshot available
- **THEN** structural endpoints return 503 and retrieval can return text-only results marked graph_unavailable without claiming complete dependencies

#### Scenario: Unknown scope or unsupported input
- **WHEN** the caller requests an unknown project or invalid traversal parameters
- **THEN** the service returns 404 or 422 respectively without executing unrestricted graph queries

#### Scenario: Generation fails
- **WHEN** answer generation fails after successful retrieval
- **THEN** the response retains evidence and explicitly marks answer_unavailable
