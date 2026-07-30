from locket.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("LOCKET_DB_URL", raising=False)
    monkeypatch.delenv("LOCKET_CORPUS_DIR", raising=False)
    monkeypatch.delenv("LOCKET_OLLAMA_MODEL", raising=False)
    s = Settings.load()
    assert s.db_url == "postgresql://locket:locket@localhost:5432/locket"
    assert s.ollama_model == "qwen3-vl:8b"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("LOCKET_DB_URL", "postgresql://x/y")
    monkeypatch.setenv("LOCKET_CORPUS_DIR", r"C:\somewhere")
    s = Settings.load()
    assert s.db_url == "postgresql://x/y"
    assert s.corpus_dir is not None and s.corpus_dir.name == "somewhere"
