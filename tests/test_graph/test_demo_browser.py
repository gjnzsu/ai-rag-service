"""Opt-in real-browser checks. Run against the local graph demo server only."""

import copy
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.getenv('GRAPH_DEMO_BROWSER') != '1', reason='Opt-in local browser acceptance')
BASE = 'http://127.0.0.1:8001'
ARTIFACTS = Path(__file__).resolve().parents[2] / 'data/graph-poc/demo-artifacts'


@pytest.fixture(scope='module')
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        instance=playwright.chromium.launch(headless=True,args=['--no-proxy-server'])
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    instance=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    instance.on('pageerror',lambda error:errors.append(str(error)))
    yield instance
    instance.close()
    assert not errors


def open_demo(page):
    response=page.goto(BASE+'/graph/demo')
    page.wait_for_load_state('networkidle')
    assert response.status == 200
    assert page.title() == 'Jira backlog graph'


def submit(page,form):
    page.locator(form).locator('button[type=submit]').click()


def wait_finished(page,form):
    from playwright.sync_api import expect
    expect(page.locator(form).locator('button[type=submit]')).to_be_enabled(timeout=15000)


@pytest.fixture(scope='module')
def payloads():
    with httpx.Client(base_url=BASE,trust_env=False,timeout=15) as client:
        overview=client.get('/graph/projects/AIPLAT/overview').json()
        detail=client.get('/graph/projects/AIPLAT/epics/AIPLAT-13').json()
        dependency=client.post('/graph/retrieve',json={'project_key':'AIPLAT','query':'AIPLAT-46','issue_key':'AIPLAT-46'}).json()
    assert 'data' in overview and 'data' in detail and 'data' in dependency
    return overview,detail,dependency


def test_live_three_operations_and_recorded_done_pair(page):
    from playwright.sync_api import expect
    posts=[]
    page.on('request',lambda req: posts.append(req.url) if req.method=='POST' else None)
    open_demo(page)
    submit(page,'#overview-form')
    expect(page.locator('#overview-results tbody tr').first).to_be_visible()
    assert not posts
    expect(page.locator('#overview-results')).to_contain_text('AIPLAT-13')
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    page.screenshot(path=str(ARTIFACTS/'overview.png'),full_page=True)
    page.locator('#tab-detail').click()
    submit(page,'#detail-form')
    expect(page.locator('#detail-results')).to_contain_text('AIPLAT-46')
    expect(page.locator('#detail-results .pager')).to_contain_text('of 4')
    page.screenshot(path=str(ARTIFACTS/'detail.png'),full_page=True)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results .graph-shell svg')).to_be_visible()
    expect(page.locator('#dependency-results')).to_contain_text('AIPLAT-46 → AIPLAT-16')
    expect(page.locator('#dependency-results')).to_contain_text('not a current blocker')
    assert all('/graph/query' not in url for url in posts)
    page.screenshot(path=str(ARTIFACTS/'dependencies.png'),full_page=True)
    page.locator('#dependency-unresolved').check()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results')).to_contain_text('No matching recorded relationship')
    expect(page.locator('#dependency-results .graph-shell')).to_have_count(0)


def test_standalone_detail_pagination_pins_first_snapshot(page,payloads):
    from playwright.sync_api import expect
    detail=copy.deepcopy(payloads[1])
    detail['data'].update({'offset':0,'limit':1,'total':4,'has_more':True,'children':detail['data']['children'][:1]})
    seen=[]
    def route_request(route):
        seen.append(route.request.url)
        route.fulfill(json=detail)
    page.route('**/graph/projects/AIPLAT/epics/**',route_request)
    open_demo(page)
    page.locator('#tab-detail').click()
    submit(page,'#detail-form')
    expect(page.locator('#detail-results .pager')).to_be_visible()
    page.get_by_role('button',name='Next',exact=True).click()
    wait_finished(page,'#detail-form')
    assert len(seen)==2
    assert 'snapshot_id='+detail['snapshot_id'] in seen[1]


def test_generation_is_pinned_and_failure_retains_evidence(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    seen=[]
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    def generation(route):
        seen.append(route.request.post_data_json)
        answer={**payload,'answer':{'answer':None,'status':'answer_unavailable','citations':[],'diagnostics':['generation_unavailable']}}
        route.fulfill(json=answer)
    page.route('**/graph/query',generation)
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    page.get_by_role('button',name='Generate grounded answer').click()
    expect(page.locator('#dependency-results')).to_contain_text('answer_unavailable')
    assert seen[0].get('snapshot_id') == payload['snapshot_id']
    expect(page.locator('#dependency-results .graph-shell')).to_be_visible()


def test_controls_clear_stale_results(page,payloads):
    from playwright.sync_api import expect
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payloads[2]))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results .graph-shell')).to_be_visible()
    page.locator('#project-key').fill('OTHER')
    expect(page.locator('#dependency-results .graph-shell')).to_have_count(0)
    expect(page.locator('#dependency-results')).not_to_contain_text(payloads[2]['snapshot_id'])


def test_unavailable_graph_text_has_safe_source_and_escaping(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    payload['data'].update({'nodes':[],'paths':[],'seeds':[],'coverage':'graph_unavailable'})
    text=payload['data']['text_evidence'][0]
    text['text']='<img src=x onerror="window.demoInjected=true">'
    payload['data']['text_evidence']=[text]
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results')).to_contain_text('Graph traversal is unavailable')
    expect(page.locator('.evidence-list')).to_contain_text('<img src=x')
    expect(page.locator('.evidence-list img')).to_have_count(0)
    expect(page.locator('.evidence-list a')).to_have_attribute('href',text['source_url'])
    assert not page.evaluate('Boolean(window.demoInjected)')


def test_error_then_independent_dependency_recovery(page,payloads):
    from playwright.sync_api import expect
    page.route('**/graph/projects/*/overview',lambda route:route.fulfill(status=503,json={'detail':'Graph snapshot unavailable'}))
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payloads[2]))
    open_demo(page)
    submit(page,'#overview-form')
    expect(page.locator('#overview-results .notice.error')).to_be_visible()
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results .graph-shell')).to_be_visible()


def test_partial_coverage_is_visible(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    payload['data'].update({'coverage':'partial','truncated':True})
    payload['completeness']='partial'
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results .notice.warning')).to_contain_text('truncated')


def test_mobile_layout_has_no_document_overflow(page,payloads):
    from playwright.sync_api import expect
    page.set_viewport_size({'width':390,'height':844})
    page.route('**/graph/projects/*/overview',lambda route:route.fulfill(json=payloads[0]))
    open_demo(page)
    submit(page,'#overview-form')
    expect(page.locator('#overview-results tbody tr').first).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(ARTIFACTS/'mobile.png'),full_page=True)



def test_editing_anchor_clears_old_evidence_immediately(page,payloads):
    from playwright.sync_api import expect
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payloads[2]))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results .graph-shell')).to_be_visible()
    page.locator('#dependency-anchor').fill('AIPLAT-16')
    expect(page.locator('#dependency-results .graph-shell')).to_have_count(0)


def test_late_response_does_not_restore_old_project(page,payloads):
    from playwright.sync_api import expect
    open_demo(page)
    page.evaluate("""payload => {
        const original = window.fetch;
        window.fetch = (url, options) => String(url) === '/graph/retrieve'
            ? new Promise(resolve => { window.releaseOldResult = () => resolve(new Response(JSON.stringify(payload),
                {status:200, headers:{'Content-Type':'application/json'}})); })
            : original(url, options);
    }""",payloads[2])
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    page.locator('#project-key').fill('OTHER')
    page.evaluate('window.releaseOldResult()')
    wait_finished(page,'#dependencies-form')
    expect(page.locator('#dependency-results .graph-shell')).to_have_count(0)
    expect(page.locator('#dependency-results')).not_to_contain_text(payloads[2]['snapshot_id'])


def test_unsafe_source_urls_are_not_links(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    payload['data'].update({'nodes':[],'paths':[],'seeds':[],'coverage':'graph_unavailable'})
    text=payload['data']['text_evidence'][0]
    payload['data']['text_evidence']=[]
    for url in ['javascript:alert(1)','https://evil.test/issue','https://user:password@30156758.atlassian.net/browse/AIPLAT-46']:
        record=copy.deepcopy(text)
        record['source_url']=url
        payload['data']['text_evidence'].append(record)
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    expect(page.locator('.evidence-list article')).to_have_count(3)
    expect(page.locator('.evidence-list a')).to_have_count(0)


def test_tab_keyboard_and_in_app_epic_drilldown(page,payloads):
    from playwright.sync_api import expect
    page.route('**/graph/projects/*/overview',lambda route:route.fulfill(json=payloads[0]))
    page.route('**/graph/projects/AIPLAT/epics/**',lambda route:route.fulfill(json=payloads[1]))
    open_demo(page)
    page.locator('#tab-overview').focus()
    page.keyboard.press('ArrowRight')
    expect(page.locator('#tab-detail')).to_have_attribute('aria-selected','true')
    page.keyboard.press('Home')
    submit(page,'#overview-form')
    row=page.locator('#overview-results tbody tr').filter(has_text='AIPLAT-13')
    row.get_by_role('button',name='Inspect Epic').click()
    expect(page.locator('#detail-results')).to_contain_text('AIPLAT-46')
    assert page.url == BASE+'/graph/demo'


def test_hybrid_result_identifies_starting_issues(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    payload['seed_method']='hybrid'
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    open_demo(page)
    page.locator('#tab-dependencies').click()
    page.locator('#use-preset').click()
    submit(page,'#dependencies-form')
    expect(page.locator('#dependency-results')).to_contain_text('Hybrid search')
    expect(page.locator('#dependency-results')).to_contain_text('Starting issues: AIPLAT-46')


def test_answer_retry_replaces_previous_error(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[2])
    page.route('**/graph/retrieve',lambda route:route.fulfill(json=payload))
    attempts=[]
    def generation(route):
        attempts.append(True)
        if len(attempts)==1:
            route.fulfill(status=503,json={'detail':'Temporary model failure'})
        else:
            route.fulfill(json={**payload,'answer':{'answer':'Recorded relation [E1].','status':'supported','citations':[],'diagnostics':[]}})
    page.route('**/graph/query',generation)
    open_demo(page)
    page.locator('#tab-dependencies').click()
    submit(page,'#dependencies-form')
    button=page.get_by_role('button',name='Generate grounded answer')
    button.click()
    expect(page.locator('#dependency-results .notice.error')).to_be_visible()
    button.click()
    expect(page.locator('#dependency-results .answer')).to_contain_text('Recorded relation [E1]')
    expect(page.locator('#dependency-results .notice.error')).to_have_count(0)


def test_panorama_shows_all_epics_and_real_relationships(page,payloads):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    expect(graph).to_be_visible()
    expect(graph.locator('.panorama-node[data-kind="epic"]')).to_have_count(8)
    expect(graph.locator('.panorama-node')).to_have_count(35)
    expect(graph.locator('.panorama-edge[data-relation="CHILD_OF"]')).to_have_count(27)
    expect(graph.locator('.panorama-edge[data-relation="BLOCKS"]')).to_have_count(sum(e["relation_type"] == "BLOCKS" for e in payloads[0]["data"]["graph"]["edges"]))
    graph.screenshot(path=str(ARTIFACTS/'panorama.png'))


def test_panorama_select_and_zoom(page):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    expect(graph.locator('svg')).to_be_visible()
    initial=graph.locator('svg').get_attribute('viewBox')
    graph.get_by_role('button',name='Zoom in',exact=True).click()
    assert graph.locator('svg').get_attribute('viewBox') != initial
    graph.get_by_role('button',name='Fit graph',exact=True).click()
    assert graph.locator('svg').get_attribute('viewBox') == initial
    graph.locator('.panorama-node[data-key="AIPLAT-13"]').click()
    expect(graph).to_contain_text('ai-rag-service 1.0 MVP for rag pipeline setup')
    graph.get_by_role('button',name='Inspect Epic',exact=True).click()
    expect(page.locator('#detail-results')).to_contain_text('AIPLAT-46')


def test_panorama_issue_uses_displayed_snapshot(page,payloads):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    expect(page.locator('#project-panorama')).to_be_visible()
    page.evaluate("state.detail = {scope: {project_key: 'AIPLAT'}, snapshot_id: 'older-snapshot'}")
    captured=[]
    def retrieve(route):
        captured.append(route.request.post_data_json)
        route.fulfill(json=payloads[2])
    page.route('**/graph/retrieve',retrieve)
    page.locator('.panorama-node[data-key="AIPLAT-46"]').click()
    page.get_by_role('button',name='Inspect dependencies',exact=True).click()
    expect(page.locator('#dependency-results')).to_contain_text('AIPLAT-46')
    assert captured[0]['snapshot_id']==payloads[0]['snapshot_id']


def test_network_switch_preserves_graph_and_highlights_neighbors(page,payloads):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    expect(graph).to_be_visible()
    keys=graph.locator('.panorama-node').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.key).sort()')
    edges=graph.locator('.panorama-edge').evaluate_all('(edges)=>edges.map(e=>[e.dataset.source,e.dataset.target,e.dataset.relation].join(":")).sort()')
    requests=[]
    page.on('request',lambda request:requests.append(request.url))
    graph.get_by_role('button',name='Network',exact=True).click()
    expect(graph).to_have_attribute('data-layout','network')
    assert graph.locator('.panorama-node').evaluate_all('(nodes)=>nodes.map(n=>n.dataset.key).sort()')==keys
    assert graph.locator('.panorama-edge').evaluate_all('(edges)=>edges.map(e=>[e.dataset.source,e.dataset.target,e.dataset.relation].join(":")).sort()')==edges
    node=graph.locator('.panorama-node[data-key="AIPLAT-37"]')
    node.focus()
    page.keyboard.press('Enter')
    expect(node).to_have_class(__import__('re').compile('is-selected'))
    expect(graph.locator('.panorama-node.is-dimmed')).not_to_have_count(0)
    expect(graph.locator('.panorama-node[data-key="AIPLAT-23"]')).not_to_have_class(__import__('re').compile('is-dimmed'))
    selected_id=next(n['issue_id'] for n in payloads[0]['data']['graph']['nodes'] if n['key']=='AIPLAT-37')
    highlighted=graph.locator('.panorama-edge:not(.is-dimmed)').evaluate_all('(edges)=>edges.map(e=>[e.dataset.source,e.dataset.target])')
    assert highlighted and all(selected_id in ends for ends in highlighted)
    graph.get_by_role('button',name='Clear selection',exact=True).click()
    expect(graph.locator('.is-dimmed')).to_have_count(0)
    graph.get_by_role('button',name='Grouped',exact=True).click()
    expect(graph).to_have_attribute('data-layout','grouped')
    assert requests==[]


def test_network_drag_updates_edges_and_reset(page):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    graph.get_by_role('button',name='Network',exact=True).click()
    node=graph.locator('.panorama-node[data-key="AIPLAT-37"]')
    graph.locator('svg').scroll_into_view_if_needed()
    initial=node.bounding_box()
    edge_before=graph.locator('.panorama-edge').evaluate_all('(edges)=>edges.map(e=>e.outerHTML)')
    page.mouse.move(initial['x']+initial['width']/2,initial['y']+initial['height']/2)
    page.mouse.down()
    page.mouse.move(initial['x']+initial['width']/2+55,initial['y']+initial['height']/2+35,steps=8)
    page.mouse.up()
    moved=node.bounding_box()
    assert moved['x']>initial['x']+35
    assert graph.locator('.panorama-edge').evaluate_all('(edges)=>edges.map(e=>e.outerHTML)')!=edge_before
    graph.get_by_role('button',name='Reset layout',exact=True).click()
    graph.locator('svg').scroll_into_view_if_needed()
    restored=node.bounding_box()
    assert abs(restored['x']-initial['x'])<2
    assert abs(restored['y']-initial['y'])<2
    expect(graph.locator('.panorama-node')).to_have_count(35)
    graph.screenshot(path=str(ARTIFACTS/'network.png'))


def test_network_handles_disconnected_budget_and_empty_graph(page,payloads):
    from playwright.sync_api import expect
    payload=copy.deepcopy(payloads[0])
    template=payload['data']['graph']['nodes'][0]
    nodes=[{**template,'issue_id':str(9000+i),'key':f'SYN-{i+1}',
            'issue_type':'Story','title':'Synthetic disconnected item'} for i in range(100)]
    payload['data']['graph']={'nodes':nodes,'edges':[],'total_nodes':120,'total_edges':0,'truncated':True}
    page.route('**/graph/projects/AIPLAT/overview',lambda route:route.fulfill(json=payload))
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    graph.get_by_role('button',name='Network',exact=True).click()
    expect(graph.locator('.panorama-node')).to_have_count(100)
    expect(graph.locator('.panorama-edge')).to_have_count(0)
    expect(graph).to_contain_text('Display limited')
    assert graph.locator('svg').evaluate("svg => [...svg.querySelectorAll('.panorama-node')].every(n => {const b=n.getBBox();return Number.isFinite(b.x)&&Number.isFinite(b.y)&&b.width>0})")
    graph.get_by_role('button',name='Fit graph',exact=True).click()
    assert graph.locator('.panorama-viewport').evaluate("viewport => {const v=viewport.getBoundingClientRect();return [...viewport.querySelectorAll('.panorama-node')].every(n => {const b=n.getBoundingClientRect();return b.left>=v.left&&b.top>=v.top&&b.right<=v.right&&b.bottom<=v.bottom})}")
    graph.screenshot(path=str(ARTIFACTS/'network-100-synthetic.png'))
    payload['data']['graph']={'nodes':[],'edges':[],'total_nodes':0,'total_edges':0,'truncated':False}
    submit(page,'#overview-form')
    expect(page.locator('#project-panorama')).to_contain_text('No graph nodes')


def test_network_zoomed_drag_and_snapshot_drilldown(page,payloads):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    graph.get_by_role('button',name='Network',exact=True).click()
    graph.get_by_role('button',name='Zoom in',exact=True).click()
    node=graph.locator('.panorama-node[data-key="AIPLAT-37"]')
    node.scroll_into_view_if_needed()
    before=node.bounding_box()
    x,y=before['x']+before['width']/2,before['y']+before['height']/2
    page.mouse.move(x,y)
    page.mouse.down()
    page.mouse.move(x+40,y+25,steps=8)
    page.mouse.up()
    after=node.bounding_box()
    assert abs((after['x']-before['x'])-40)<3
    assert abs((after['y']-before['y'])-25)<3
    graph.get_by_role('button',name='Fit graph',exact=True).click()
    assert graph.locator('svg').evaluate("svg => {const v=svg.getBoundingClientRect();return [...svg.querySelectorAll('.panorama-node')].every(n => {const b=n.getBoundingClientRect();return b.left>=v.left&&b.top>=v.top&&b.right<=v.right&&b.bottom<=v.bottom})}")
    node.focus()
    page.keyboard.press('Space')
    captured=[]
    page.on('request',lambda request:captured.append(request.post_data_json) if request.url.endswith('/graph/retrieve') else None)
    graph.get_by_role('button',name='Inspect dependencies',exact=True).click()
    expect(page.locator('#dependency-results')).to_contain_text('AIPLAT-23')
    assert captured[0]['snapshot_id']==payloads[0]['snapshot_id']


def test_network_mouse_selection_after_drag(page):
    from playwright.sync_api import expect
    open_demo(page)
    submit(page,'#overview-form')
    graph=page.locator('#project-panorama')
    graph.get_by_role('button',name='Network',exact=True).click()
    node=graph.locator('.panorama-node[data-key="AIPLAT-37"]')
    node.click()
    expect(graph.locator('.panorama-info')).to_contain_text('AIPLAT-37')
    bounds=node.bounding_box()
    x,y=bounds['x']+bounds['width']/2,bounds['y']+bounds['height']/2
    page.mouse.move(x,y)
    page.mouse.down()
    page.mouse.move(x+25,y+15,steps=5)
    page.mouse.up()
    graph.locator('.panorama-node[data-key="AIPLAT-23"]').click()
    expect(graph.locator('.panorama-info')).to_contain_text('AIPLAT-23')
