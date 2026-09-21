"""Scoped Python tools. Model input never supplies database queries or scope."""

from collections import Counter
from time import monotonic

from app.graph.service import GraphUnavailable
from app.graph.store import Neo4jGraphStore


def public_issue(issue):
    return issue.model_dump(mode='json', exclude={'scope'})


def directions(direction):
    if direction not in {'inbound', 'outbound', 'both'}:
        raise ValueError('Invalid direction')
    return ('inbound', 'outbound') if direction == 'both' else (direction,)


class ReadBudget:
    def __init__(self, seconds=60, clock=monotonic):
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + seconds
        self.reserve = min(15, seconds / 4)

    def remaining(self, *, generation=False):
        return max(0., self.deadline - self.clock() - (0 if generation else self.reserve))

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError('Request budget exhausted')


class TimedGraphStore(Neo4jGraphStore):
    """Read-only adapter with per-query server timeout and request accounting."""

    def __init__(self, driver, budget):
        super().__init__(driver)
        self.budget = budget
        self.query_count = 0

    def _read(self, query, scope, **params):
        from neo4j import Query
        self.budget.check()
        self.query_count += 1
        with self.driver.session(database=self.database, default_access_mode='READ') as session:
            result = session.run(Query(query, timeout=min(5., self.budget.remaining())),
                                 **scope.model_dump(), **params)
            return [row['payload'] for row in result]


class EpicTools:
    def __init__(self, service, request, *, budget=None, page_size=100, max_children=500,
                 max_nodes=100, max_paths=1000):
        self.service, self.request = service, request
        self.budget = budget or ReadBudget()
        self.page_size, self.max_children = page_size, max_children
        self.max_nodes, self.max_paths = max_nodes, max_paths
        self.scope = None
        self.epic = None
        self.captured_at = None
        self.children = {}
        self.known = {}
        self.edges = {}
        self.relation_nodes = set()
        self.ledger = {}
        self.backlog_complete = False
        self.backlog_limitations = []
        self.backend_calls = 0
        self.child_total = 0

    def get_epic_backlog(self):
        if self.scope is not None:
            raise ValueError('Backlog already read')
        offset = 0
        snapshot = self.request.snapshot_id
        while offset < self.max_children:
            try:
                self.budget.check()
                self.backend_calls += 1
                response = self.service.epic_detail(
                    self.request.project_key, self.request.epic_key, snapshot,
                    offset=offset, limit=min(self.page_size, self.max_children-offset))
                if self.scope is None:
                    scope, epic = response.scope, response.data.epic
                    if (scope.project_key != self.request.project_key
                            or epic.key != self.request.epic_key
                            or epic.issue_type.casefold() != 'epic'
                            or epic.scope != scope
                            or response.snapshot_id != scope.snapshot_id
                            or (snapshot is not None and response.snapshot_id != snapshot)):
                        raise GraphUnavailable('Invalid Epic evidence')
                    self.scope, self.epic = scope, epic
                    self.captured_at = response.captured_at
                    self.child_total = response.data.total
                    snapshot = response.snapshot_id
                    self.known[self.epic.key] = self.epic
                if (response.scope != self.scope or response.snapshot_id != snapshot
                        or response.data.total != self.child_total):
                    raise ValueError('Changed snapshot or population')
                page = response.data.children
                if any(i.scope != self.scope or i.key in self.children for i in page):
                    raise ValueError('Invalid or repeated child evidence')
                for child in page:
                    self.children[child.key] = child
                    self.known[child.key] = child
                offset += len(page)
                if not response.data.has_more:
                    self.backlog_complete = len(self.children) == self.child_total
                    break
                if not page:
                    raise ValueError('Non-progressing page')
            except Exception:
                if self.epic is None:
                    raise
                self.backlog_limitations.append('backlog_read_failed')
                break
        if not self.backlog_complete:
            self.backlog_limitations.append('backlog_incomplete')
        return self.facts()

    @property
    def stories(self):
        return [i for i in self.children.values() if i.issue_type.casefold() == 'story']

    def get_issue_dependencies(self, issue_keys, direction):
        sides = directions(direction)
        if (not issue_keys or len(issue_keys) > 20 or len(set(issue_keys)) != len(issue_keys)
                or any(k not in self.known or k == self.epic.key for k in issue_keys)):
            raise ValueError('Invalid dependency seeds')
        pending = {key: tuple(side for side in sides if (key, side) not in self.ledger)
                   for key in issue_keys}
        if not any(pending.values()):
            raise ValueError('Repeated dependency action')
        for key in issue_keys:
            fresh_sides = pending[key]
            if not fresh_sides:
                continue
            query_direction = 'both' if len(fresh_sides) == 2 else fresh_sides[0]
            state = 'failed'
            try:
                self.budget.check()
                self.backend_calls += 1
                response = self.service.dependencies(
                    self.request.project_key, key, self.scope.snapshot_id,
                    direction=query_direction, hops=1, limit=self.max_nodes,
                    relation_type='BLOCKS', unresolved_pair=False)
                result = response.data
                if (response.scope != self.scope or result.scope != self.scope
                        or response.snapshot_id != self.scope.snapshot_id):
                    raise ValueError('Invalid dependency scope')
                nodes = {i.issue_id: i for i in result.nodes}
                if any(i.scope != self.scope or (i.key in self.known and self.known[i.key] != i)
                       for i in result.nodes):
                    raise ValueError('Invalid dependency identity')
                staged = {}
                for path in result.paths:
                    if len(path.edges) != 1:
                        raise ValueError('Only one-hop evidence is allowed')
                    edge = path.edges[0]
                    if (edge.relation_type != 'BLOCKS' or edge.scope != self.scope
                            or any(i not in nodes for i in (edge.source_issue_id, edge.target_issue_id,
                                                           edge.origin_issue_id))):
                        raise ValueError('Invalid relationship evidence')
                    seed = self.known[key].issue_id
                    if ((query_direction == 'inbound' and edge.target_issue_id != seed)
                            or (query_direction == 'outbound' and edge.source_issue_id != seed)
                            or seed not in (edge.source_issue_id, edge.target_issue_id)):
                        raise ValueError('Invalid relationship direction')
                    staged[edge.model_dump_json()] = edge
                state = 'partial' if result.truncated or result.coverage != 'complete' else 'complete'
                for identity, edge in staged.items():
                    involved = {edge.source_issue_id, edge.target_issue_id, edge.origin_issue_id}
                    if (len(self.relation_nodes | involved) > self.max_nodes
                            or (identity not in self.edges and len(self.edges) >= self.max_paths)):
                        state = 'partial'
                        continue
                    self.relation_nodes.update(involved)
                    self.edges[identity] = edge
                    for identifier in involved:
                        self.known[nodes[identifier].key] = nodes[identifier]
            except Exception:
                state = 'failed'
            for side in fresh_sides:
                self.ledger[(key, side)] = state
        return self.observations()

    def facts(self):
        stories = self.stories
        done = sum(i.status_category.casefold() == 'done' for i in stories)
        bugs = [i for i in self.children.values() if i.issue_type.casefold() == 'bug']
        return {
            'population': 'direct_story_children', 'child_total': self.child_total,
            'observed_story_count': len(stories), 'observed_done_count': done,
            'story_count': len(stories) if self.backlog_complete else None,
            'done_count': done if self.backlog_complete else None,
            'direct_bug_count': len(bugs) if self.backlog_complete else None,
            'observed_direct_bug_count': len(bugs),
            'completion_rate': done / len(stories) if self.backlog_complete and stories else None,
            'by_status': dict(Counter(i.status for i in stories)),
            'by_status_category': dict(Counter(i.status_category for i in stories)),
            'stories': [public_issue(i) for i in stories],
            'direct_bugs': [public_issue(i) for i in bugs],
            'associated_bugs': [public_issue(i) for k, i in self.known.items()
                                if k not in self.children and i.issue_type.casefold() == 'bug'],
        }

    def observations(self):
        return {'facts': self.facts(), 'backlog_complete': self.backlog_complete,
                'edges': [e.model_dump(mode='json', exclude={'scope'}) for e in self.edges.values()],
                'related_issues': [public_issue(i) for i in self.known.values() if i.issue_id in self.relation_nodes],
                'coverage': [{'key': key, 'direction': side, 'status': status}
                             for (key, side), status in sorted(self.ledger.items())]}
