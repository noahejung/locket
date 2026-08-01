"""Backend-selection tests -- all network-free (object construction only, no `.invoke()`).
`get_chat_model` never makes an outbound call by itself; `ChatAnthropic(...)`/`ChatOllama(...)`
both construct successfully with no key/server present (confirmed live), so these run in the
default suite."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from locket.config import Settings
from locket.llm import DEFAULT_LOCAL_TEXT_MODEL, get_chat_model, local_model_name, resolve_backend


def _settings(*, api_key: str | None) -> Settings:
    return Settings(corpus_dir=None, db_url="postgresql://x/y", anthropic_api_key=api_key, ollama_model="qwen3-vl:8b")


def test_resolve_backend_defaults_to_ollama_when_no_key(monkeypatch):
    monkeypatch.delenv("LOCKET_LLM_BACKEND", raising=False)
    assert resolve_backend(_settings(api_key=None)) == "ollama"


def test_resolve_backend_defaults_to_anthropic_when_key_present(monkeypatch):
    monkeypatch.delenv("LOCKET_LLM_BACKEND", raising=False)
    assert resolve_backend(_settings(api_key="sk-ant-fake")) == "anthropic"


def test_resolve_backend_explicit_ollama_wins_even_with_a_key(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "ollama")
    assert resolve_backend(_settings(api_key="sk-ant-fake")) == "ollama"


def test_resolve_backend_explicit_anthropic_wins_even_without_a_key(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    assert resolve_backend(_settings(api_key=None)) == "anthropic"


def test_resolve_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "OLLAMA")
    assert resolve_backend(_settings(api_key=None)) == "ollama"


def test_resolve_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "openai")
    with pytest.raises(ValueError, match="LOCKET_LLM_BACKEND"):
        resolve_backend(_settings(api_key=None))


def test_local_model_name_defaults_to_gemma(monkeypatch):
    monkeypatch.delenv("LOCKET_LOCAL_MODEL", raising=False)
    assert local_model_name() == DEFAULT_LOCAL_TEXT_MODEL == "gemma3:12b"


def test_local_model_name_env_override(monkeypatch):
    monkeypatch.setenv("LOCKET_LOCAL_MODEL", "qwen2.5:3b-instruct")
    assert local_model_name() == "qwen2.5:3b-instruct"


def test_get_chat_model_ollama_backend_returns_chat_ollama(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "ollama")
    monkeypatch.setenv("LOCKET_LOCAL_MODEL", "qwen2.5:3b-instruct")
    model = get_chat_model("extraction_default", _settings(api_key=None))
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen2.5:3b-instruct"
    assert model.temperature == 0


def test_get_chat_model_anthropic_backend_returns_chat_anthropic_with_correct_model_id(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    model = get_chat_model("extraction_default", _settings(api_key=None))
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-4-5"


def test_get_chat_model_escalation_role_uses_sonnet_on_anthropic_backend(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    model = get_chat_model("extraction_escalation", _settings(api_key=None))
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5"


def test_get_chat_model_unknown_role_on_anthropic_backend_raises(monkeypatch):
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    with pytest.raises(ValueError, match="unknown role"):
        get_chat_model("not_a_real_role", _settings(api_key=None))


def test_get_chat_model_anthropic_backend_uses_temperature_zero_for_determinism(monkeypatch):
    """Parity with the ollama backend (test_get_chat_model_ollama_backend_returns_chat_ollama
    above already asserts ChatOllama(..., temperature=0)) -- an unpinned Anthropic
    temperature (defaults to None, i.e. the API's own non-zero default) made
    extraction/resolution/profile non-deterministic across identical pipeline runs,
    undermining the point of the statement-hash dedup and the extracted-windows watermark
    (fix-wave-1 item 8): a re-run over unchanged input could still produce different facts."""
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    model = get_chat_model("extraction_default", _settings(api_key=None))
    assert model.temperature == 0
