import builtins
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.reranking import build_reranker
from app.reranking.noop import NoOpReranker


def _settings(**overrides):
    return Settings(openai_api_key="test-key", _env_file=None, **overrides)


def test_factory_defaults_to_none_without_importing_optional_backends(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("openai", "torch", "transformers")):
            pytest.fail(f"unselected backend imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert isinstance(build_reranker(settings=_settings()), NoOpReranker)


@pytest.mark.parametrize("provider", ["OPENAI", "qwen", "", " none", "unknown"])
def test_factory_rejects_every_provider_except_the_three_exact_values(provider):
    with pytest.raises(ValueError, match="provider"):
        build_reranker(provider, settings=_settings())


def test_settings_validate_the_provider_and_default_to_none():
    assert _settings().reranker_provider == "none"
    with pytest.raises(ValidationError):
        _settings(reranker_provider="OPENAI")


def test_settings_default_to_immutable_openai_and_qwen_model_versions():
    configured = _settings()

    assert configured.reranker_openai_model == "gpt-5-2025-08-07"
    assert configured.reranker_qwen_model == "Qwen/Qwen3-Reranker-0.6B"
    assert configured.reranker_qwen_revision == "e61197ed45024b0ed8a2d74b80b4d909f1255473"


@pytest.mark.parametrize(
    "overrides",
    [
        {"reranker_openai_model": "gpt-5"},
        {"reranker_openai_model": "gpt-5-latest"},
        {"reranker_qwen_model": "Qwen/Qwen3-Reranker-4B"},
        {"reranker_qwen_revision": "main"},
        {"reranker_qwen_revision": "e61197e"},
        {"reranker_qwen_revision": "z" * 40},
    ],
)
def test_settings_reject_floating_or_non_immutable_reranker_models(overrides):
    with pytest.raises(ValidationError):
        _settings(**overrides)


def test_factory_constructs_only_the_selected_openai_backend():
    client = object()

    reranker = build_reranker("openai", settings=_settings(), client=client)

    from app.reranking.openai import GPT5Reranker

    assert isinstance(reranker, GPT5Reranker)
    assert reranker.client is client


def test_factory_constructs_qwen_without_loading_optional_packages():
    def fail_loader(*args, **kwargs):
        pytest.fail("factory eagerly loaded Qwen")

    reranker = build_reranker(
        "qwen_local",
        settings=_settings(),
        tokenizer_loader=fail_loader,
        model_loader=fail_loader,
    )

    from app.reranking.qwen import Qwen3LocalReranker

    assert isinstance(reranker, Qwen3LocalReranker)


def test_default_requirements_pin_an_sdk_httpx_compatibility_pair():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "openai==1.51.0" in requirements
    assert "httpx>=0.27.0,<0.28.0" in requirements
