## ADDED Requirements

### Requirement: Complete visible project snapshot
The service SHALL paginate all issues visible to the configured account in an allowed project and retain identity, type, parent, issue links, status category, source URL, update time and searchable text with a snapshot manifest.

#### Scenario: More than one page
- **WHEN** a project contains more issues than a page limit
- **THEN** all pages are consumed before publication and counts reflect the visible snapshot, not a top-k subset

#### Scenario: Partial acquisition
- **WHEN** any page fails or a pagination token repeats
- **THEN** the build fails without replacing the active snapshot or claiming completeness

### Requirement: Faithful graph mapping
The service SHALL preserve stable site/issue IDs, actual parent relationships and explicit link types and directions, with source provenance and deduplication. It MUST NOT infer blocking from shared parent or text mentions.

#### Scenario: Bug belongs directly to Epic
- **WHEN** a Bug has an Epic parent
- **THEN** it has CHILD_OF to that Epic and no invented Story parent

#### Scenario: Reciprocal link records
- **WHEN** both linked issues expose the same Blocks link
- **THEN** exactly one canonical directed edge is stored

#### Scenario: Boundary or missing parent
- **WHEN** an endpoint is outside the visible project or a non-Epic has no parent
- **THEN** the service records a boundary diagnostic or unassigned item without fetching out-of-scope content or inventing a relationship

### Requirement: Isolated publication and rebuild
The service SHALL publish one snapshot only after its graph and two text indexes are ready, pin each request to one version, and preserve the previous active snapshot on failure. Credentials MUST NOT appear in artifacts or responses.

#### Scenario: Text index build fails
- **WHEN** graph construction succeeds but FTS or vector indexing fails
- **THEN** the new version stays inactive and readers continue with the previous snapshot if available

#### Scenario: Relationship removed at source
- **WHEN** a successful subsequent full snapshot no longer contains an earlier relation
- **THEN** new active requests do not return that relation and existing default retrieval collections remain unchanged
