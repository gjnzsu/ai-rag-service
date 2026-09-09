# backlog-graph-demo Specification

## Purpose
TBD - created by archiving change jira-backlog-graph-poc. Update Purpose after archive.
## Requirements
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

### Requirement: Switchable snapshot panorama layouts
The demo SHALL provide Grouped and Network layouts over the same overview nodes and recorded edges without additional retrieval requests. It MUST preserve canonical direction, statuses, source details, Epic membership and snapshot-pinned drilldown.

#### Scenario: Switch to network and back
- **WHEN** a viewer switches layouts on a loaded overview
- **THEN** the node and edge identities remain identical, no relationships are inferred, and Grouped remains available

### Requirement: Stable interactive network exploration
The Network layout SHALL settle after bounded computation, support node dragging with connected-edge updates, background panning, zoom, Fit, and Reset layout. It SHALL support keyboard node selection and clearable highlighting of direct neighbors and recorded incident edges.

#### Scenario: Select a cross-Epic endpoint
- **WHEN** AIPLAT-37 is selected in a snapshot recording AIPLAT-37 BLOCKS AIPLAT-23
- **THEN** that canonical edge and its endpoints are highlighted while unrelated items are de-emphasized without being removed

#### Scenario: Drag under zoom
- **WHEN** a viewer drags a network node after zooming or panning
- **THEN** its position follows the pointer, connected edge endpoints update, and the graph data and snapshot remain unchanged

#### Scenario: Reset and lifecycle
- **WHEN** the viewer resets the layout or leaves the overview
- **THEN** reset restores a stable initial layout and no ongoing simulation continues after removal

#### Scenario: Missing or bounded graph
- **WHEN** a graph is empty, disconnected, or truncated at its existing budget
- **THEN** the view preserves empty/truncated notices, keeps all supplied nodes accessible and does not fabricate connections
