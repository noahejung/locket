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
from datetime import UTC, datetime, tzinfo
from pathlib import Path

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
# android branch's class deliberately excludes "." (fix-wave-2 item 9, security audit LOW
# #5): the old `[\w.\- ]+\.\w+` let the leading `+` match through literal dots too, so it
# overlapped with the required trailing `\.\w+` -- for a non-matching suffix, the engine
# could retry the split at every prior dot in the string, a polynomial (not exponential)
# backtracking blowup bounded only by WhatsApp's message-length ceiling. Excluding "." from
# the class removes the overlap: the `+` can only ever stop at the position immediately
# before a literal dot, so there is exactly one candidate split point per dot, no retrying.
# Trade-off: a filename with more than one dot in its name portion (e.g. "photo.v2.jpg")
# no longer matches at all -- accepted per the audit's own suggested fix, since real
# WhatsApp android export filenames are single-dot in every observed case.
_ATTACH = re.compile(
    r"^[‎‏]*(?:<attached:\s*(?P<ios>[^>]+)>|(?P<android>[\w\- ]+\.\w+)\s\(file attached\))"
)


def _local_tz() -> tzinfo:
    """The system's local timezone, resolved at call time (not import time -- the UTC
    offset can vary by date under DST, so freezing it once at import would be wrong for
    some fraction of any real export)."""
    return datetime.now().astimezone().tzinfo


def _ts(g: dict, tz: tzinfo) -> datetime:
    """WhatsApp's exported header timestamp is the phone's LOCAL wall-clock reading at
    send time, never UTC -- tagging it UTC directly (the pre-fix behavior) silently shifted
    every derived timestamp by the true UTC offset. Build the naive local reading first,
    attach `tz`, then convert."""
    y = int(g["y"])
    y += 2000 if y < 100 else 0
    h = int(g["h"])
    if g.get("ampm"):
        ap = g["ampm"].lower()
        h = (h % 12) + (12 if ap == "pm" else 0)
    local = datetime(y, int(g["m"]), int(g["d"]), h, int(g["min"]), int(g.get("s") or 0), tzinfo=tz)
    return local.astimezone(UTC)


def _match_header(line: str, tz: tzinfo) -> tuple[datetime, str] | None:
    """Return (ts, remainder-after-header) if `line` starts a new message, else None."""
    m = _BRACKET.match(line)
    if m:
        return _ts(m.groupdict(), tz), line[m.end() :]
    m = _DASH.match(line)
    if m:
        return _ts(m.groupdict(), tz), line[m.end() :]
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


def parse_whatsapp(
    path: Path,
    thread: str | None = None,
    source_tz: tzinfo | None = None,
    *,
    warnings: list[str] | None = None,
) -> Iterator[RawItem]:
    """`source_tz` is the timezone the export's header timestamps were written in --
    defaults to the system's local timezone (`_local_tz()`) since that's the export
    device's own clock in the overwhelmingly common case. Pass it explicitly when ingesting
    an export known to be from a different timezone than the machine running locket.

    `warnings`, if given, gets one human-readable message appended if any line in the file
    matched neither header regex AND had no pending message to continue -- e.g. a whole
    export in an unrecognized locale variant (a differently-formatted AM/PM marker, say)
    would otherwise parse to zero items with no trace of why. A normal multiline message
    continuation is NOT counted here -- only lines with nothing to attach to."""
    tz = source_tz if source_tz is not None else _local_tz()
    pending: dict | None = None
    unmatched = 0
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").rstrip("\r")
            line = _LRM.sub("", line)
            if not line:
                continue
            header = _match_header(line, tz)
            if header is not None:
                if pending is not None:
                    yield _split_pending(pending)
                ts, remainder = header
                pending = {"ts": ts, "text": remainder, "thread": thread}
            elif pending is None:
                # No header match AND nothing pending to continue -- an unrecognized-format
                # line, not a real multiline continuation. Count it instead of dropping it
                # with no trace.
                unmatched += 1
            else:
                # Continuation of the previous message (multiline text).
                pending["text"] += "\n" + line
        if pending is not None:
            yield _split_pending(pending)
    if unmatched and warnings is not None:
        warnings.append(f"{unmatched} unmatched line(s) in {path} could not be parsed as a message header")
