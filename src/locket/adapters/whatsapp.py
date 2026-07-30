"""WhatsApp chat-export adapter — hand-rolled (see Task 3 decision note: whatstk
is GPL-3.0, drags in plotly/seaborn/pandas, and silently discards colon-less
system messages that a context engine needs to keep).

Handles both export shapes:
  - Android/US dash form:    "1/15/25, 10:32 AM - John: text"   (month/day)
  - iOS bracket form:        "[15/01/25, 22:41:03] John: text"  (day/month)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from locket.adapters.base import register
from locket.models import RawItem, SourceKind

_LRM = re.compile(r"^[‎‏]+")

# Android/US: "1/15/25, 10:32 AM - "
_DASH = re.compile(
    r"^(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2,4}), "
    r"(?P<h>\d{1,2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?\s?(?P<ampm>[AaPp][Mm])? - "
)
# iOS: "[15/01/25, 22:41:03] "
_BRACKET = re.compile(
    r"^\[(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{2,4}), "
    r"(?P<h>\d{1,2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?\]\s"
)
_ATTACH = re.compile(
    r"^[‎‏]*(?:<attached:\s*(?P<ios>[^>]+)>|(?P<android>[\w.\- ]+\.\w+)\s\(file attached\))"
)


def _ts(g: dict) -> datetime:
    y = int(g["y"])
    y += 2000 if y < 100 else 0
    h = int(g["h"])
    if g.get("ampm"):
        ap = g["ampm"].lower()
        h = (h % 12) + (12 if ap == "pm" else 0)
    return datetime(y, int(g["m"]), int(g["d"]), h, int(g["min"]), int(g.get("s") or 0), tzinfo=UTC)


def _match_header(line: str) -> tuple[datetime, str] | None:
    """Return (ts, remainder-after-header) if `line` starts a new message, else None."""
    m = _BRACKET.match(line)
    if m:
        return _ts(m.groupdict()), line[m.end() :]
    m = _DASH.match(line)
    if m:
        return _ts(m.groupdict()), line[m.end() :]
    return None


def _split_pending(pending: dict) -> RawItem:
    remainder = pending["text"]
    idx = remainder.find(": ")
    if idx == -1:
        # No "sender: " prefix at all — a system/group-management line.
        return RawItem.make(
            source=SourceKind.whatsapp,
            ts=pending["ts"],
            sender=None,
            text=remainder,
            is_system=True,
            thread=pending["thread"],
        )
    sender = remainder[:idx]
    body = remainder[idx + 2 :]
    media_path = None
    am = _ATTACH.match(body)
    if am:
        media_path = am.group("ios") or am.group("android")
        body = (body[: am.start()] + body[am.end() :]).strip() or None
    return RawItem.make(
        source=SourceKind.whatsapp,
        ts=pending["ts"],
        sender=sender,
        text=body,
        media_path=media_path,
        is_system=False,
        thread=pending["thread"],
    )


def parse_whatsapp(path: Path, thread: str | None = None) -> Iterator[RawItem]:
    pending: dict | None = None
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").rstrip("\r")
            line = _LRM.sub("", line)
            if not line:
                continue
            header = _match_header(line)
            if header is not None:
                if pending is not None:
                    yield _split_pending(pending)
                ts, remainder = header
                pending = {"ts": ts, "text": remainder, "thread": thread}
            else:
                # Continuation of the previous message (multiline text).
                if pending is None:
                    continue
                pending["text"] += "\n" + line
        if pending is not None:
            yield _split_pending(pending)


register(".txt", parse_whatsapp)
