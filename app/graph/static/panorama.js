"use strict";

(function exposePanorama() {
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
    return node;
  };
  const htmlEl = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const identity = issue => String(issue?.issue_id ?? issue?.id ?? issue?.key ?? "");
  const kind = issue => String(issue?.issue_type || "Issue");
  const isEpic = issue => kind(issue).toLowerCase() === "epic";
  const safeSource = (raw, site) => {
    try {
      const url = new URL(raw);
      if (url.protocol !== "https:" || url.username || url.password) return null;
      if (site) {
        const base = new URL(site);
        if (base.origin !== url.origin) return null;
      }
      return url.href;
    } catch {
      return null;
    }
  };
  const short = (value, max) => {
    const text = String(value || "Untitled");
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  };
  const statusClass = issue => {
    const value = String(issue?.status_category || issue?.status || "").toLowerCase();
    if (value.includes("done")) return "is-done";
    if (value.includes("progress") || value.includes("indeterminate")) return "is-progress";
    return "is-todo";
  };

  function layout(nodes, edges) {
    const epicNodes = nodes.filter(isEpic).sort((a, b) => String(a.key).localeCompare(String(b.key), undefined, { numeric: true }));
    const byId = new Map(nodes.map(node => [identity(node), node]));
    const epicIds = new Set(epicNodes.map(identity));
    const parentByChild = new Map();
    edges.filter(edge => edge.relation_type === "CHILD_OF").forEach(edge => {
      if (byId.has(String(edge.source_issue_id)) && epicIds.has(String(edge.target_issue_id))) {
        parentByChild.set(String(edge.source_issue_id), String(edge.target_issue_id));
      }
    });
    const positions = new Map();
    const boxes = [];
    const cellW = 330, cellH = 390, left = 35, top = 35;
    epicNodes.forEach((epic, index) => {
      const col = index % 4, row = Math.floor(index / 4);
      const x = left + col * cellW, y = top + row * cellH;
      const children = nodes.filter(node => parentByChild.get(identity(node)) === identity(epic))
        .sort((a, b) => String(a.key).localeCompare(String(b.key), undefined, { numeric: true }));
      boxes.push({ x, y, width: 305, height: 350, epic });
      positions.set(identity(epic), { x: x + 152, y: y + 171, epic: true });
      children.forEach((child, childIndex) => {
        const angle = -Math.PI / 2 + (Math.PI * 2 * childIndex / Math.max(1, children.length));
        const ring = children.length > 8 && childIndex % 2 ? 128 : 103;
        positions.set(identity(child), {
          x: x + 152 + Math.cos(angle) * (ring + 18),
          y: y + 177 + Math.sin(angle) * ring,
          epic: false
        });
      });
    });
    const remaining = nodes.filter(node => !positions.has(identity(node)))
      .sort((a, b) => String(a.key).localeCompare(String(b.key), undefined, { numeric: true }));
    const extraY = top + Math.max(2, Math.ceil(epicNodes.length / 4)) * cellH;
    remaining.forEach((node, index) => positions.set(identity(node), {
      x: 100 + (index % 10) * 130,
      y: extraY + 75 + Math.floor(index / 10) * 92,
      epic: false,
      unassigned: true
    }));
    return { positions, boxes, width: 1400, height: extraY + (remaining.length ? Math.ceil(remaining.length / 10) * 92 + 120 : 0) };
  }

  // Fixed deterministic computation; the layout never creates business relationships.
  function networkLayout(nodes, edges) {
    const ordered = [...nodes].sort((a, b) => String(a.key).localeCompare(String(b.key), undefined, { numeric: true }));
    const epics = ordered.filter(isEpic), positions = new Map(), anchors = new Map();
    const parent = new Map(edges.filter(edge => edge.relation_type === "CHILD_OF")
      .map(edge => [String(edge.source_issue_id), String(edge.target_issue_id)]));
    const radius = Math.max(260, epics.length * 51);
    epics.forEach((epic, i) => {
      const angle = -Math.PI / 2 + i * Math.PI * 2 / Math.max(1, epics.length);
      anchors.set(identity(epic), { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius * .72 });
    });
    ordered.forEach((node, i) => {
      const id = identity(node), anchor = anchors.get(id) || anchors.get(parent.get(id));
      const angle = i * 2.3999632297;
      positions.set(id, { x: (anchor?.x || 0) + (isEpic(node) ? 0 : Math.cos(angle) * 125),
        y: (anchor?.y || 0) + (isEpic(node) ? 0 : Math.sin(angle) * 125), epic: isEpic(node) });
    });
    const points = ordered.map(node => positions.get(identity(node)));
    const links = edges.map(edge => ({ a: positions.get(String(edge.source_issue_id)),
      b: positions.get(String(edge.target_issue_id)), child: edge.relation_type === "CHILD_OF" }))
      .filter(link => link.a && link.b && link.a !== link.b);
    for (let step = 0; step < 240; step++) {
      points.forEach(point => { point.fx = 0; point.fy = 0; });
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const a = points[i], b = points[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          if (Math.hypot(dx, dy) < .01) { dx = .1 * (j + 1); dy = .1; }
          const distance = Math.hypot(dx, dy), minimum = (a.epic ? 90 : 49) + (b.epic ? 90 : 49);
          const force = 900 / Math.max(100, distance * distance) + Math.max(0, minimum - distance) * .18;
          const fx = dx / distance * force, fy = dy / distance * force;
          a.fx -= fx; a.fy -= fy; b.fx += fx; b.fy += fy;
        }
      }
      links.forEach(({ a, b, child }) => {
        const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
        const force = (distance - (child ? 154 : 235)) * (child ? .022 : .012);
        a.fx += dx / distance * force; a.fy += dy / distance * force;
        b.fx -= dx / distance * force; b.fy -= dy / distance * force;
      });
      ordered.forEach(node => {
        const point = positions.get(identity(node)), anchor = anchors.get(identity(node));
        point.fx += ((anchor?.x || 0) - point.x) * (anchor ? .018 : .0008);
        point.fy += ((anchor?.y || 0) - point.y) * (anchor ? .018 : .0008);
        point.x += Math.max(-12, Math.min(12, point.fx));
        point.y += Math.max(-12, Math.min(12, point.fy));
      });
    }
    return positions;
  }

  window.renderPanorama = function renderPanorama(payload, onSelectEpic, onSelectIssue) {
    const graph = payload?.data?.graph || {};
    const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const edges = Array.isArray(graph.edges) ? graph.edges : [];
    const byId = new Map(nodes.map(node => [identity(node), node]));
    const root = htmlEl("section", "panorama", undefined);
    root.id = "project-panorama";
    root.dataset.layout = "grouped";
    root.setAttribute("aria-label", "Project dependency panorama");

    const heading = htmlEl("div", "panorama-heading");
    const titleBox = htmlEl("div");
    titleBox.append(htmlEl("p", "kicker", "Whole snapshot"), htmlEl("h3", null, "Project panorama"));
    const countText = `${graph.total_nodes ?? nodes.length} issues · ${graph.total_edges ?? edges.length} relationships`;
    titleBox.appendChild(htmlEl("p", "panorama-summary", `${countText}${graph.truncated ? " · Display limited by graph budget" : ""}`));
    const controls = htmlEl("div", "panorama-controls");
    const zoomIn = htmlEl("button", "secondary", "+");
    const zoomOut = htmlEl("button", "secondary", "−");
    const fit = htmlEl("button", "secondary", "Fit");
    zoomIn.type = zoomOut.type = fit.type = "button";
    zoomIn.setAttribute("aria-label", "Zoom in");
    zoomOut.setAttribute("aria-label", "Zoom out");
    fit.setAttribute("aria-label", "Fit graph");
    const grouped = htmlEl("button", "secondary", "Grouped");
    const network = htmlEl("button", "secondary", "Network");
    const reset = htmlEl("button", "secondary", "Reset layout");
    const clear = htmlEl("button", "secondary", "Clear selection");
    [grouped, network, reset, clear].forEach(button => { button.type = "button"; });
    grouped.setAttribute("aria-pressed", "true");
    network.setAttribute("aria-pressed", "false");
    controls.append(grouped, network, zoomOut, fit, zoomIn, reset, clear);
    heading.append(titleBox, controls);
    root.appendChild(heading);

    if (!nodes.length) {
      controls.querySelectorAll("button").forEach(button => { button.disabled = true; });
      root.appendChild(htmlEl("p", "empty", "No graph nodes are recorded in this snapshot."));
      return root;
    }

    const legend = htmlEl("div", "panorama-legend");
    [["legend-epic", "Epic"], ["legend-issue", "Story / bug"], ["legend-todo", "To do"], ["legend-progress", "In progress"], ["legend-done", "Done"], ["legend-child", "CHILD_OF"], ["legend-blocks", "BLOCKS →"]].forEach(([cls, label]) => {
      const item = htmlEl("span", `legend-item ${cls}`);
      item.append(htmlEl("i"), document.createTextNode(label));
      legend.appendChild(item);
    });
    root.appendChild(legend);
    root.appendChild(htmlEl("p", "panorama-summary", "Clusters show recorded Epic membership. Dependencies are not inferred between Epics. In Network, drag nodes to explore. Lines alone represent recorded relations; distance does not imply a dependency. Drag the background to pan."));

    const viewport = htmlEl("div", "panorama-viewport");
    const initial = layout(nodes, edges);
    const { boxes, width, height } = initial;
    let positions = new Map([...initial.positions].map(([id, point]) => [id, { ...point }]));
    let networkInitial = null;
    const nodeElements = new Map(), edgeElements = [];
    const parentById = new Map(edges.filter(edge => edge.relation_type === "CHILD_OF")
      .map(edge => [String(edge.source_issue_id), String(edge.target_issue_id)]));
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "All project Epics, issues, and recorded relationships" });
    const defs = svgEl("defs");
    [["panorama-arrow-child", "#6f9e98"], ["panorama-arrow-blocks", "#c65332"]].forEach(([id, color]) => {
      const marker = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
      const path = svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color });
      marker.appendChild(path); defs.appendChild(marker);
    });
    svg.appendChild(defs);
    const canvas = svgEl("g", { class: "panorama-canvas" });
    boxes.forEach(box => {
      const boundary = svgEl("rect", { x: box.x, y: box.y, width: box.width, height: box.height, rx: 22, class: "panorama-cluster" });
      canvas.appendChild(boundary);
    });
    if ([...positions.values()].some(point => point.unassigned)) {
      const label = svgEl("text", { x: 35, y: height - Math.max(145, Math.ceil(nodes.length / 10) * 92 + 85), class: "panorama-orphan-label" });
      label.textContent = "No Epic grouping / other hierarchy";
      canvas.appendChild(label);
    }
    const updateEdge = (line, edge) => {
      const a = positions.get(String(edge.source_issue_id)), b = positions.get(String(edge.target_issue_id));
      if (!a || !b) return;
      const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
      const boundary = point => point.epic
        ? Math.min(dx ? 74 / Math.abs(dx) : Infinity, dy ? 38 / Math.abs(dy) : Infinity) * distance + 5 : 36;
      const start = Math.min(boundary(a), distance / 2), end = Math.min(boundary(b), distance / 2);
      Object.entries({ x1: a.x + dx / distance * start, y1: a.y + dy / distance * start,
        x2: b.x - dx / distance * end, y2: b.y - dy / distance * end })
        .forEach(([key, value]) => line.setAttribute(key, value));
    };
    edges.forEach(edge => {
      const sourceId = String(edge.source_issue_id), targetId = String(edge.target_issue_id);
      const a = positions.get(sourceId), b = positions.get(targetId);
      if (!a || !b) return;
      const relation = String(edge.relation_type || "UNKNOWN");
      const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
      const boundary = point => point.epic
        ? Math.min(dx ? 74 / Math.abs(dx) : Infinity, dy ? 38 / Math.abs(dy) : Infinity) * distance + 5 : 36;
      const start = boundary(a), end = boundary(b);
      const line = svgEl("line", {
        x1: a.x + dx / distance * start, y1: a.y + dy / distance * start,
        x2: b.x - dx / distance * end, y2: b.y - dy / distance * end,
        class: `panorama-edge relation-${relation.toLowerCase()}`,
        "data-relation": relation, "data-source": sourceId, "data-target": targetId,
        "marker-end": relation === "BLOCKS" ? "url(#panorama-arrow-blocks)" : relation === "CHILD_OF" ? "url(#panorama-arrow-child)" : ""
      });
      const edgeTitle = svgEl("title");
      edgeTitle.textContent = `${byId.get(sourceId)?.key || sourceId} ${relation} ${byId.get(targetId)?.key || targetId}`;
      line.appendChild(edgeTitle);
      canvas.appendChild(line);
      edgeElements.push({ line, edge });
    });

    const info = htmlEl("aside", "panorama-info");
    info.setAttribute("aria-live", "polite");
    const showInfo = issue => {
      info.replaceChildren();
      info.append(htmlEl("strong", null, `${issue.key || identity(issue)} · ${kind(issue)}`), htmlEl("span", "panorama-info-title", issue.title || "Untitled"), htmlEl("span", `status ${statusClass(issue) === "is-done" ? "done" : ""}`, `${issue.status || "Unknown"} · ${issue.status_category || "Unknown category"}`));
      const parent = byId.get(parentById.get(identity(issue)));
      if (parent) info.appendChild(htmlEl("span", "panorama-membership", `Parent: ${parent.key} / ${parent.title || "Untitled"}`));
      const url = safeSource(issue.source_url, payload?.scope?.site_id);
      if (url) {
        const link = htmlEl("a", "source", "Open Jira source");
        link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
        info.appendChild(link);
      }
      const action = htmlEl("button", "text-button", isEpic(issue) ? "Inspect Epic" : "Inspect dependencies");
      action.type = "button";
      action.addEventListener("click", () => isEpic(issue) ? onSelectEpic?.(issue) : onSelectIssue?.(issue));
      info.appendChild(action);
    };
    info.appendChild(htmlEl("span", null, "Select a node to see its snapshot details."));

    let selectedId = null;
    const highlight = id => {
      selectedId = id;
      const neighbors = new Set(id ? [id] : []);
      edges.forEach(edge => {
        const a = String(edge.source_issue_id), b = String(edge.target_issue_id);
        if (a === id || b === id) { neighbors.add(a); neighbors.add(b); }
      });
      nodeElements.forEach(({ group }, nodeId) => {
        group.classList.toggle("is-selected", nodeId === id);
        group.classList.toggle("is-neighbor", nodeId !== id && neighbors.has(nodeId));
        group.classList.toggle("is-dimmed", !!id && !neighbors.has(nodeId));
        group.setAttribute("aria-pressed", String(nodeId === id));
      });
      edgeElements.forEach(({ line, edge }) => {
        const incident = String(edge.source_issue_id) === id || String(edge.target_issue_id) === id;
        line.classList.toggle("is-highlighted", !!id && incident);
        line.classList.toggle("is-dimmed", !!id && !incident);
      });
    };
    nodes.forEach(issue => {
      const point = positions.get(identity(issue));
      if (!point) return;
      const epic = isEpic(issue);
      const group = svgEl("g", {
        class: `panorama-node ${epic ? "epic" : "issue"} ${statusClass(issue)}`,
        "data-key": issue.key || identity(issue), "data-kind": epic ? "epic" : "issue",
        tabindex: 0, role: "button", "aria-label": `${issue.key || identity(issue)}, ${kind(issue)}, ${issue.status || "unknown status"}`
      });
      const shape = epic
        ? svgEl("rect", { x: point.x - 74, y: point.y - 38, width: 148, height: 76, rx: 14 })
        : svgEl("circle", { cx: point.x, cy: point.y, r: 31 });
      group.appendChild(shape);
      const key = svgEl("text", { x: point.x, y: point.y + (epic ? -5 : -3), class: "panorama-key", "text-anchor": "middle" });
      key.textContent = issue.key || identity(issue);
      const detail = svgEl("text", { x: point.x, y: point.y + (epic ? 15 : 14), class: "panorama-node-sub", "text-anchor": "middle" });
      detail.textContent = epic ? short(issue.title, 21) : short(kind(issue), 11);
      group.append(key, detail);
      const parent = byId.get(parentById.get(identity(issue)));
      const membership = svgEl("text", { x: point.x, y: point.y + 47, class: "panorama-parent", "text-anchor": "middle" });
      membership.textContent = parent ? `${isEpic(parent) ? "Epic" : "Parent"}: ${parent.key}` : (!epic ? "No Epic parent" : "");
      group.appendChild(membership);
      nodeElements.set(identity(issue), { group, origin: { ...point } });
      const select = () => { highlight(identity(issue)); showInfo(issue); };
      group.addEventListener("click", () => { if (root.dataset.layout === "grouped") select(); });
      group.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
      canvas.appendChild(group);
    });
    svg.appendChild(canvas);
    viewport.appendChild(svg);
    root.append(viewport, info);

    let scale = 1, centerX = width / 2, centerY = height / 2, drag = null;
    let frameWidth = width, frameHeight = height;
    const applyScale = () => svg.setAttribute("viewBox", `${centerX - frameWidth / scale / 2} ${centerY - frameHeight / scale / 2} ${frameWidth / scale} ${frameHeight / scale}`);
    const updateGeometry = () => {
      nodeElements.forEach(({ group, origin }, id) => {
        const point = positions.get(id);
        group.setAttribute("transform", `translate(${point.x - origin.x} ${point.y - origin.y})`);
      });
      edgeElements.forEach(({ line, edge }) => updateEdge(line, edge));
    };
    const fitAll = () => {
      scale = 1;
      if (root.dataset.layout === "grouped") {
        frameWidth = width; frameHeight = height; centerX = width / 2; centerY = height / 2;
      } else {
        const values = [...positions.values()];
        const minX = Math.min(...values.map(point => point.x - (point.epic ? 80 : 65))) - 35;
        const maxX = Math.max(...values.map(point => point.x + (point.epic ? 80 : 65))) + 35;
        const minY = Math.min(...values.map(point => point.y - 45)) - 35;
        const maxY = Math.max(...values.map(point => point.y + 55)) + 35;
        centerX = (minX + maxX) / 2; centerY = (minY + maxY) / 2;
        frameWidth = Math.max(maxX - minX, (maxY - minY) * 1400 / 815);
        frameHeight = frameWidth * 815 / 1400;
      }
      applyScale(); viewport.scrollTo(0, 0);
    };
    const switchLayout = mode => {
      root.dataset.layout = mode;
      if (mode === "network" && !networkInitial) networkInitial = networkLayout(nodes, edges);
      positions = new Map([...(mode === "network" ? networkInitial : initial.positions)]
        .map(([id, point]) => [id, { ...point }]));
      grouped.setAttribute("aria-pressed", String(mode === "grouped"));
      network.setAttribute("aria-pressed", String(mode === "network"));
      updateGeometry(); fitAll(); highlight(selectedId);
    };
    grouped.addEventListener("click", () => switchLayout("grouped"));
    network.addEventListener("click", () => switchLayout("network"));
    reset.addEventListener("click", () => switchLayout(root.dataset.layout));
    clear.addEventListener("click", () => {
      highlight(null); info.replaceChildren(htmlEl("span", null, "Select a node to see its snapshot details."));
    });
    zoomIn.addEventListener("click", () => { scale = Math.min(2.2, scale + .2); applyScale(); });
    zoomOut.addEventListener("click", () => { scale = Math.max(.55, scale - .2); applyScale(); });
    fit.addEventListener("click", fitAll);
    const svgPoint = event => {
      const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const matrix = svg.getScreenCTM();
      return matrix ? point.matrixTransform(matrix.inverse()) : null;
    };
    svg.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      const node = event.target.closest(".panorama-node");
      if (node && root.dataset.layout !== "network") return;
      const pointer = svgPoint(event);
      if (!pointer) return;
      const id = node ? [...nodeElements].find(([, value]) => value.group === node)?.[0] : null;
      drag = { id, pointer, startX: event.clientX, startY: event.clientY, centerX, centerY,
        point: id ? { ...positions.get(id) } : null, inverse: svg.getScreenCTM().inverse() };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener("pointermove", event => {
      if (!drag) return;
      const screen = svg.createSVGPoint(); screen.x = event.clientX; screen.y = event.clientY;
      const pointer = screen.matrixTransform(drag.inverse);
      const dx = pointer.x - drag.pointer.x, dy = pointer.y - drag.pointer.y;
      if (drag.id) {
        const point = positions.get(drag.id);
        point.x = drag.point.x + dx; point.y = drag.point.y + dy;
        drag.moved = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 3;
        updateGeometry();
      } else {
        centerX = drag.centerX - dx; centerY = drag.centerY - dy; applyScale();
      }
    });
    const release = event => {
      if (event.type === "pointerup" && drag?.id && !drag.moved) {
        highlight(drag.id); showInfo(byId.get(drag.id));
      }
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      drag = null;
    };
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    svg.addEventListener("lostpointercapture", () => { drag = null; });
    return root;
  };
})();
