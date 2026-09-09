# Network panorama implementation checkpoint

Date: 2026-09-09. Follow-up change: jira-graph-network-view. Scope authorized: switchable Grouped/Network presentation using existing snapshot data; no retrieval or database behavior changes.

## Plan and time control

Two segments, each at most 50 minutes (absolute user limit one hour): implementation/TDD, then browser acceptance/quality/documentation. One bounded UI subtask owns panorama.js and panorama CSS; root owns specs, browser tests, review and verification. No delegation by subtask, commits, deployment or Jira mutation.

## Red phase

Before implementation, two browser cases failed at the missing Network button (2 failed, 17 deselected, 69.54 seconds). Cases verify identity-preserving layout switch/no network requests and drag-connected-edge/reset behavior. Added acceptance for keyboard selection, direct-neighbor emphasis, 100 disconnected synthetic nodes, empty/truncated data, zoom-aware drag and snapshot-pinned dependency drilldown.

## Data separation

Real active snapshot 7b9d816aa2c445c9919635aff149f705 has 35 nodes/29 edges, including AIPLAT-37 (Epic AIPLAT-36) BLOCKS AIPLAT-23 (Epic AIPLAT-17). The synthetic 100-node browser payload is only injected into a test response; no synthetic records are published to Jira or databases.


## Implementation and review

The UI subtask implemented a fixed 240-step deterministic layout, then stops; no simulation timers or external libraries. Network/Grouped preserve data attributes and callbacks. Drag uses the inverse SVG screen matrix. Direct neighbors and incident edges are emphasized without removing evidence.

Root review identified pointer-capture selection interaction; implementation now selects on pointerup only below the drag threshold and has no global click-suppression state. Real mouse selection after drag is covered. Another visual review found 100 disconnected nodes clipped because Network reused tall Grouped proportions: viewport containment regression observed RED (1 failed / 21 deselected / 7.73s), then fixed with independent landscape aspect. All five new Network cases passed (15.42s).

The initial Fit test compared untransformed SVG getBBox with viewBox; corrected to compare rendered client coordinates. This was a test-coordinate bug, not evidence of missing nodes.

Manual Chromium checks at 1440px and 390px found no page errors or document overflow. Network computation measured 54ms and 38.7ms respectively (one observation each, not a performance guarantee). Positions were unchanged across two subsequent frames; source review confirms no ongoing timers. Actual measurements: jira-graph-network-browser.json.

Screenshots inspected: data/graph-poc/demo-artifacts/network.png, network-selected.png, network-mobile.png and network-100-synthetic.png. Desktop Fit contains all 100 synthetic nodes. Narrow screens use the bounded graph viewport and panning/scrolling. Fixed-count layout does not guarantee absence of all crossings; users can drag nodes to clarify paths.

## Final verification

Final JS syntax checks passed; full repo quality command with GRAPH_INTEGRATION=1 and GRAPH_DEMO_BROWSER=1: **559 passed**, one existing multipart deprecation warning, 128.03 seconds. Lint passed. OpenSpec strict validation and git diff --check passed.

Document scaffold created 01:38:56 UTC; verification completed 01:49:33 UTC (approximately 10.6 minutes). Both planned segments completed within their time limits; no checkpoint pause.

Archive complete: openspec/changes/archive/2026-09-09-jira-graph-network-view. Two requirements added to backlog-graph-demo; all three main specs pass strict validation. Local API stays running for preview.
