"""Instagram DM export adapter — Meta's exports carry latin-1/utf-8 mojibake:
every non-ASCII string got UTF-8-encoded then mis-decoded as Latin-1 before
being written into the JSON. `fix_mojibake` reverses that, recursively, over
whatever JSON-shaped structure it's handed. Exported so the corpus generator
can apply the corruption in reverse when synthesizing fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ftfy

from locket.models import RawItem, SourceKind


def fix_mojibake(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return obj.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                return ftfy.fix_text(obj)
            except Exception:
                return obj
    if isinstance(obj, dict):
        return {k: fix_mojibake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_mojibake(v) for v in obj]
    return obj


def parse_instagram_thread(thread_dir: Path) -> Iterator[RawItem]:
    all_msgs: list[dict] = []
    for f in sorted(thread_dir.glob("message_*.json")):
        data = json.loads(f.read_bytes())
        data = fix_mojibake(data)
        all_msgs.extend(data.get("messages", []))

    # File order is not a documented guarantee — always re-sort ascending.
    all_msgs.sort(key=lambda m: m["timestamp_ms"])

    thread_name = thread_dir.name
    for m in all_msgs:
        ts = datetime.fromtimestamp(m["timestamp_ms"] / 1000, tz=UTC)
        sender = m.get("sender_name")
        text = m.get("content")
        media_path = None
        photos = m.get("photos")
        if photos:
            media_path = photos[0].get("uri")
        yield RawItem.make(
            source=SourceKind.instagram,
            ts=ts,
            sender=sender,
            text=text,
            media_path=media_path,
            thread=thread_name,
        )
