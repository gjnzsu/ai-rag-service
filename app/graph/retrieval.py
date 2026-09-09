"""Stateless, snapshot-pinned hybrid anchors and bounded graph evidence."""

import json

from app.graph import structure
from app.graph.api_models import GraphRetrieveRequest, GraphRetrieveResponse
from app.graph.models import GraphEvidenceResult, Issue, TraversalRequest
from app.graph.service import GraphNotFound, GraphService, GraphUnavailable
from app.retrieval.query_hints import extract_query_hints


class GraphRetrievalService(GraphService):
    def __init__(self, *args, search_provider, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_provider = search_provider

    def retrieve(self, request: GraphRetrieveRequest) -> GraphRetrieveResponse:
        manifest = self._resolve(request.project_key, request.snapshot_id)
        scope = manifest.scope
        search = None
        diagnostics = []
        try:
            search = self.search_provider(manifest)
        except Exception:
            diagnostics.append('text_index_unavailable')
        try:
            _, issues, edges = self._records(manifest)
            structure.overview(scope, issues, edges)  # Validate before using scoped identities.
        except Exception:
            return self._fallback(request, manifest, search)
        by_id = {issue.issue_id: issue for issue in issues}
        keys = [request.epic_key or request.issue_key] if request.epic_key or request.issue_key else []
        if not keys and request.intent != 'overview':
            keys = extract_query_hints(request.query).jira_keys
        method = 'explicit' if keys else 'hybrid'
        anchors = []
        if request.intent == 'overview':
            method = 'project'
        elif keys:
            for key in keys:
                issue = next((x for x in issues if x.key == key), None)
                if issue is None:
                    raise GraphNotFound('Issue not found in snapshot')
                anchors.append(issue)
            if request.epic_key and anchors[0].issue_type.casefold() != 'epic':
                raise GraphNotFound('Epic not found in snapshot')
        else:
            if search is None:
                raise GraphUnavailable('Seed retrieval unavailable')
            try:
                hits = search.search(request.query, top_k=request.top_k)
                self._validate_text(hits, scope, by_id)
            except Exception:
                raise GraphUnavailable('Seed retrieval unavailable') from None
            anchors = list({hit.issue_id: by_id[hit.issue_id] for hit in hits}.values())
        result = GraphEvidenceResult(scope=scope, diagnostics=diagnostics)
        payload = None
        if request.intent == 'overview':
            payload = structure.overview(scope, issues, edges).model_dump(mode='json', exclude={'graph'})
            result.nodes = issues[:request.limit]
            if len(issues) > request.limit:
                self._partial(result, 'node_budget_exceeded')
        elif request.intent == 'epic_detail':
            epic = next((x for x in anchors if x.issue_type.casefold() == 'epic'), None)
            if epic is None:
                parents = {e.target_issue_id for e in edges if e.relation_type == 'CHILD_OF'
                           and e.source_issue_id in {x.issue_id for x in anchors}}
                epic = next((x for x in issues if x.issue_id in parents and x.issue_type.casefold() == 'epic'), None)
            if epic is not None:
                detail = structure.epic_detail(scope, issues, edges, epic.key, limit=max(1, request.limit-1))
                payload = detail.model_dump(mode='json')
                result.nodes = [epic, *detail.children][:request.limit]
                result.seeds = [epic.issue_id]
                if detail.has_more or detail.total + 1 > request.limit:
                    self._partial(result, 'node_budget_exceeded')
            else:
                result.diagnostics.append('no_matching_epic')
        else:
            seeds = []
            for anchor in anchors:
                if anchor.issue_type.casefold() == 'epic':
                    if len(result.nodes) < request.limit and anchor not in result.nodes:
                        result.nodes.append(anchor)
                    elif anchor not in result.nodes:
                        self._partial(result, 'node_budget_exceeded')
                    children = structure._children(issues, edges, anchor.issue_id)
                    seeds.extend(children)
                else:
                    seeds.append(anchor)
            seeds = list({x.issue_id: x for x in seeds}.values())
            if len(seeds) > request.limit:
                self._partial(result, 'seed_budget_exceeded')
            for seed in seeds[:request.limit]:
                current_ids = {x.issue_id for x in result.nodes}
                if seed.issue_id not in current_ids:
                    if len(current_ids) >= request.limit:
                        self._partial(result, 'node_budget_exceeded')
                        continue
                    result.nodes.append(seed)
                result.seeds.append(seed.issue_id)
                try:
                    part = self.store.traverse(TraversalRequest(
                        scope=scope, issue_id=seed.issue_id, direction=request.direction, hops=request.hops,
                        limit=request.limit, relation_types=(request.relation_type,), unresolved_pair=request.unresolved_pair))
                    if part.scope != scope or any(by_id.get(x.issue_id) != x for x in part.nodes):
                        raise ValueError('invalid traversal scope or node identity')
                    self._merge(result, part, request.limit, by_id)
                except Exception:
                    return self._fallback(request, manifest, search)
        if request.intent == 'dependencies':
            returned = {x.issue_id for x in result.nodes}
            memberships = []
            for anchor in anchors:
                if anchor.issue_type.casefold() != 'epic' or anchor.issue_id not in returned:
                    continue
                links = [e for e in edges if e.relation_type == 'CHILD_OF'
                         and e.target_issue_id == anchor.issue_id and e.source_issue_id in returned]
                memberships.append({'epic':anchor.model_dump(mode='json'),
                                    'edges':[e.model_dump(mode='json') for e in links]})
            if memberships:
                payload = {'epic_membership':memberships}
        if not anchors and request.intent != 'overview':
            result.diagnostics.append('no_matching_seeds')
        if search is not None:
            try:
                text = search.for_issues([x.issue_id for x in result.nodes])
                self._validate_text(text, scope, by_id)
                result.text_evidence = self._budget_text(text, request.text_budget, result)
                if result.nodes and set(x.issue_id for x in text) != set(x.issue_id for x in result.nodes):
                    self._partial(result, 'missing_issue_text', truncated=False)
            except Exception:
                self._partial(result, 'text_evidence_unavailable', truncated=False)
            for diagnostic in getattr(search, 'diagnostics', []):
                self._partial(result, diagnostic, truncated=False)
        elif result.nodes:
            self._partial(result, 'text_evidence_unavailable', truncated=False)
        return self._response(request, manifest, result, method, payload)

    @staticmethod
    def _validate_text(text, scope, by_id):
        for item in text:
            node = by_id.get(item.issue_id)
            if item.scope != scope or node is None or (item.document_id, item.source_url) != (node.document_id, node.source_url):
                raise ValueError('text provenance mismatch')

    @staticmethod
    def _partial(result, diagnostic, truncated=True):
        result.coverage = 'partial'
        result.truncated = result.truncated or truncated
        if diagnostic not in result.diagnostics:
            result.diagnostics.append(diagnostic)

    def _merge(self, result, part, limit, by_id):
        known_paths = {path.model_dump_json() for path in result.paths}
        for path in part.paths:
            if path.model_dump_json() in known_paths:
                continue
            ids = {x.issue_id for x in result.nodes}
            missing = [key for key in dict.fromkeys(path.issue_ids) if key not in ids]
            if len(ids) + len(missing) > limit or len(result.paths) >= 1000:
                self._partial(result, 'graph_budget_exceeded')
                continue
            result.nodes.extend(by_id[key] for key in missing)
            result.paths.append(path)
            known_paths.add(path.model_dump_json())
        if part.truncated or part.coverage != 'complete':
            self._partial(result, 'traversal_partial', truncated=part.truncated)
        result.diagnostics.extend(x for x in part.diagnostics if x not in result.diagnostics)

    def _budget_text(self, text, budget, result):
        selected = []
        remaining = budget
        for item in text:
            if remaining <= 0:
                self._partial(result, 'text_budget_exceeded')
                break
            if len(item.text) > remaining:
                selected.append(item.model_copy(update={'text':item.text[:remaining]}))
                self._partial(result, 'text_budget_exceeded')
                break
            selected.append(item)
            remaining -= len(item.text)
        return selected

    def _fallback(self, request, manifest, search):
        if search is None:
            raise GraphUnavailable('Graph and text unavailable')
        try:
            raw = json.loads((self.repository.directory(manifest.scope)/'normalized.json').read_text(encoding='utf-8'))
            issues = [Issue.model_validate(item) for item in raw['issues']]
            if any(x.scope != manifest.scope for x in issues):
                raise ValueError('snapshot mismatch')
            by_id = {x.issue_id:x for x in issues}
            key = request.issue_key or request.epic_key
            keys = [key] if key else extract_query_hints(request.query).jira_keys
            if keys:
                nodes = []
                for anchor in keys:
                    node = next((x for x in issues if x.key == anchor), None)
                    if node is None:
                        raise GraphNotFound('Issue not found in snapshot')
                    if request.epic_key and node.issue_type.casefold() != 'epic':
                        raise GraphNotFound('Epic not found in snapshot')
                    nodes.append(node)
                text = search.for_issues([node.issue_id for node in nodes])
            else:
                text = search.search(request.query, top_k=request.top_k)
            self._validate_text(text, manifest.scope, by_id)
        except GraphNotFound:
            raise
        except Exception:
            raise GraphUnavailable('Same-snapshot text unavailable') from None
        result = GraphEvidenceResult(scope=manifest.scope, coverage='graph_unavailable',
                                     diagnostics=['graph_unavailable', 'text_only_not_complete_dependencies'])
        result.text_evidence = self._budget_text(text, request.text_budget, result)
        result.coverage = 'graph_unavailable'
        result.diagnostics.extend(getattr(search, 'diagnostics', []))
        return self._response(request, manifest, result, 'explicit' if keys else 'hybrid', None)

    def _response(self, request, manifest, result, method, payload):
        metadata = self._metadata(manifest)
        metadata['diagnostics'] = [*manifest.diagnostics, *result.diagnostics]
        if result.coverage != 'complete':
            metadata['completeness'] = 'partial'
        return GraphRetrieveResponse(**metadata, data=result, intent=request.intent, seed_method=method, structure=payload)
