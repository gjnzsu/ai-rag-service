import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_graph_is_disabled_and_uses_separate_storage_by_default():
    settings = Settings(_env_file=None, openai_api_key="test-key")
    assert settings.graph_enabled is False
    assert settings.graph_demo_enabled is False
    assert settings.graph_data_dir == "./data/graph-poc"
    assert settings.graph_neo4j_uri == "bolt://127.0.0.1:7687"


def test_demo_requires_graph_capability():
    with pytest.raises(ValidationError, match="requires graph_enabled"):
        Settings(_env_file=None, openai_api_key="test-key", graph_demo_enabled=True)


def test_graph_password_is_not_exposed_in_repr():
    settings = Settings(_env_file=None, openai_api_key="test-key", graph_neo4j_password="local-only-secret")
    assert "local-only-secret" not in repr(settings)
    assert settings.graph_neo4j_password.get_secret_value() == "local-only-secret"


def test_disabled_app_starts_without_importing_graph_driver():
    code = '''
import sys
class NoNeo4j:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "neo4j" or fullname.startswith("neo4j."):
            raise AssertionError("disabled app imported neo4j")
        if fullname == "langchain_text_splitters" or fullname.startswith("langchain_text_splitters."):
            raise AssertionError("app startup imported ingestion-only text splitters")
sys.meta_path.insert(0, NoNeo4j())
from fastapi.testclient import TestClient
from app.main import create_app
with TestClient(create_app()) as client:
    assert client.get("/health").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/retrieve" in paths and "/query" in paths
    assert not any(path.startswith("/graph") for path in paths)
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        env={**os.environ, "OPENAI_API_KEY": "test-key", "GRAPH_ENABLED": "false", "GRAPH_DEMO_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
