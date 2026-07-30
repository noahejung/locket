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
                "LOCKET_DB_URL", "postgresql://locket:locket@localhost:5432/locket"
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            ollama_model=os.environ.get("LOCKET_OLLAMA_MODEL", "qwen3-vl:8b"),
        )
