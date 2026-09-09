import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


@pytest.mark.parametrize('graph_enabled', [False, True])
def test_demo_is_absent_unless_explicitly_enabled(monkeypatch, graph_enabled):
    monkeypatch.setattr(settings, 'graph_enabled', graph_enabled)
    monkeypatch.setattr(settings, 'graph_demo_enabled', False)
    with TestClient(create_app()) as client:
        assert client.get('/graph/demo').status_code == 404
        assert client.get('/graph/demo/assets/demo.js').status_code == 404


def test_enabled_demo_has_a_local_html_entrypoint(monkeypatch):
    monkeypatch.setattr(settings, 'graph_enabled', True)
    monkeypatch.setattr(settings, 'graph_demo_enabled', True)
    with TestClient(create_app()) as client:
        response=client.get('/graph/demo')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/html')
        assert 'script-src' in response.headers['content-security-policy']
        assert 'no-store' in response.headers['cache-control']


def test_demo_assets_cannot_read_other_workspace_files(monkeypatch):
    monkeypatch.setattr(settings, 'graph_enabled', True)
    monkeypatch.setattr(settings, 'graph_demo_enabled', True)
    with TestClient(create_app()) as client:
        for path in ['/graph/demo/assets/.env.graph','/graph/demo/assets/models.py','/graph/demo/assets/%2e%2e%2fmodels.py']:
            assert client.get(path).status_code == 404


def test_demo_assets_are_served_with_expected_types(monkeypatch):
    monkeypatch.setattr(settings, 'graph_enabled', True)
    monkeypatch.setattr(settings, 'graph_demo_enabled', True)
    with TestClient(create_app()) as client:
        for name, media in [('demo.js','javascript'),('demo.css','text/css')]:
            response=client.get('/graph/demo/assets/'+name)
            assert response.status_code == 200
            assert media in response.headers['content-type']
            assert response.headers['x-content-type-options'] == 'nosniff'
