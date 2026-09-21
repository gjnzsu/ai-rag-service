"""Deterministic workflow around a single, bounded ReAct decision loop."""

from app.agentic.models import Decision, Execution
from app.agentic.report import build_report
from app.agentic.tools import directions


class Coordinator:
    def __init__(self, tools, decider, generator=None):
        self.tools, self.decider, self.generator = tools, decider, generator

    def run(self):
        tools = self.tools
        execution = Execution(tool_calls=1)
        tools.get_epic_backlog()  # Invalid subjects fail before any model call.
        execution.trace.append({'tool': 'get_epic_backlog',
                                'status': 'complete' if tools.backlog_complete else 'partial'})
        requested, required = set(), set()
        last_error = None
        execution.stop_reason = 'decision_budget'
        for _ in range(3):
            if tools.budget.remaining() <= 0:
                execution.stop_reason = 'time_budget'
                break
            context = {
                'question': tools.request.question,
                'subject': {'project_key': tools.request.project_key, 'epic_key': tools.epic.key},
                'observations': tools.observations(),
                'requested_so_far': sorted(requested),
                'required_dependencies': [list(pair) for pair in sorted(required)],
                'remaining_tool_calls': 3-execution.tool_calls,
                'remaining_decisions': 3-execution.decision_rounds,
                'last_action_error': last_error,
            }
            execution.decision_rounds += 1
            execution.decision_calls += int(getattr(self.decider, 'uses_model', True))
            try:
                raw = self.decider.decide(context, timeout=min(15., tools.budget.remaining()))
                action = raw if isinstance(raw, Decision) else Decision.model_validate(raw)
            except ValueError:
                last_error = 'invalid_action'
                execution.trace.append({'action': 'rejected', 'status': last_error})
                continue
            except Exception:
                execution.stop_reason = 'decision_unavailable'
                break
            requested.update(action.requested)
            if 'blockers' in action.requested:
                seeds = [i.key for i in tools.stories if action.blocker_population == 'all_stories'
                         or i.status_category.casefold() != 'done']
                required.update((key, side) for key in seeds for side in directions(action.blocker_direction))
            if tools.budget.remaining() <= 0:
                execution.stop_reason = 'time_budget'
                break
            if action.action == 'finish':
                execution.stop_reason = 'finished'
                break
            if execution.tool_calls >= 3:
                execution.stop_reason = 'tool_budget'
                break
            if tools.budget.remaining() <= 0:
                execution.stop_reason = 'time_budget'
                break
            start = tools.budget.clock()
            try:
                tools.get_issue_dependencies(action.issue_keys, action.direction)
            except ValueError:
                last_error = 'invalid_or_repeated_lookup'
                execution.trace.append({'action': 'rejected', 'status': last_error})
                continue
            execution.tool_calls += 1
            last_error = None
            execution.trace.append({'tool': action.action, 'issue_keys': action.issue_keys,
                                    'direction': action.direction, 'status': 'executed',
                                    'elapsed_ms': round((tools.budget.clock()-start)*1000, 3)})
        execution.backend_calls = tools.backend_calls
        store = getattr(tools.service, '_store', None)
        execution.database_queries = getattr(store, 'query_count', 0)
        execution.model_usage = list(getattr(self.decider, 'usage', []))
        result = build_report(tools, requested, required, execution)
        if self.generator is not None:
            result.generation_status = 'unavailable'
            if tools.budget.remaining(generation=True) > 0:
                execution.generation_calls = 1
                try:
                    self.generator.generate(result, tools.request.question,
                                            timeout=min(15., tools.budget.remaining(generation=True)))
                except Exception:
                    result.answer = None
                    result.citations = []
                    result.generation_status = 'unavailable'
                execution.model_usage.extend(getattr(self.generator, 'usage', []))
            if result.generation_status != 'supported':
                execution.status = 'partial'
        execution.elapsed_ms = round((tools.budget.clock()-tools.budget.started)*1000, 3)
        return result


class FixedDecider:
    """Evaluation baseline. Oracle routing tags are never given to the Agent."""

    uses_model = False

    def __init__(self, requested, population='unfinished_stories', direction='inbound'):
        self.requested, self.population, self.direction = requested, population, direction

    def decide(self, context, timeout):
        del timeout
        observations = context['observations']
        checked = {(r['key'], r['direction']) for r in observations['coverage']}
        seeds = [i['key'] for i in observations['facts']['stories']
                 if self.population == 'all_stories' or i['status_category'].casefold() != 'done']
        pending = [key for key in seeds if any((key, side) not in checked for side in directions(self.direction))]
        lookup = 'blockers' in self.requested and pending and context['remaining_tool_calls'] > 0
        return Decision(action='get_issue_dependencies' if lookup else 'finish', requested=self.requested,
                        blocker_population=self.population, blocker_direction=self.direction,
                        issue_keys=pending[:20] if lookup else [], direction=self.direction)
