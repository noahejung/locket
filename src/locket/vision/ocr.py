"""RapidOCR wrapper. 3.9.x defaults to Chinese-first models (config.yaml's Det/Cls/Rec all
ship `lang_type: "ch"`) — the demo corpus and Noah's real photos are English, so the English
recognition + detection models are pinned explicitly at construction time rather than trusted
to a default that would silently mis-recognize everything."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from rapidocr import RapidOCR


@cache
def _get_engine() -> RapidOCR:
    return RapidOCR(params={"Det.lang_type": "en", "Rec.lang_type": "en"})


def ocr_image(path: Path) -> list[tuple[str, float]]:
    """Returns (line, confidence) pairs. Empty list if no text was recognized."""
    engine = _get_engine()
    result = engine(str(path))
    if result is None or result.txts is None or result.scores is None:
        return []
    return list(zip(result.txts, result.scores, strict=True))
