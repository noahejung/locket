from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path | None
    db_url: str
    anthropic_api_key: str | None
    ollama_model: str

    @classmethod
    def load(cls) -> Settings:
        cd = os.environ.get("LOCKET_CORPUS_DIR")
        return cls(
            corpus_dir=Path(cd) if cd else None,
            db_url=os.environ.get(
                # 127.0.0.1, not "localhost" -- docker-compose.yml now binds Postgres to
                # IPv4 loopback only (fix-wave-1 HIGH: was 0.0.0.0/LAN-reachable). On a
                # dual-stack host "localhost" resolves to ::1 first; nothing listens there
                # anymore, so every connection would eat a multi-second IPv6-timeout
                # fallback before retrying on 127.0.0.1. Measured live 2026-07-31: ~5s per
                # connection via "localhost" vs ~0.02s via "127.0.0.1" against this same
                # loopback-only binding.
                "LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket"
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            ollama_model=os.environ.get("LOCKET_OLLAMA_MODEL", "qwen3-vl:8b"),
        )
