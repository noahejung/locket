"""Adapter registry: maps a corpus path to the parser function that reads it.

Adapters are pure parsers — no LLM, no DB. Each module registers itself here
by calling `register()` at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from locket.models import RawItem

Adapter = Callable[[Path], Iterator["RawItem"]]

_REGISTRY: dict[str, Adapter] = {}


def register(matcher: str, fn: Adapter) -> None:
    """Register an adapter under a matcher key (glob suffix or directory-name hint)."""
    _REGISTRY[matcher] = fn


def adapter_for(path: Path) -> Adapter:
    """Return the adapter function that knows how to parse `path`.

    Matching is by suffix first (".txt" → whatsapp, ".xml" → sms_xml), then by
    directory-name hints for adapters that own a directory tree (instagram
    thread dirs, photo roots) rather than a single file.
    """
    suffix = path.suffix.lower()
    if suffix in _REGISTRY:
        return _REGISTRY[suffix]
    name = path.name.lower()
    for key, fn in _REGISTRY.items():
        if key.startswith("dir:") and key[4:] in name:
            return fn
    raise ValueError(f"No adapter registered for {path}")
