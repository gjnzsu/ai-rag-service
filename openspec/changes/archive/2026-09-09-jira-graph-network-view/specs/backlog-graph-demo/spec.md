## ADDED Requirements

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
