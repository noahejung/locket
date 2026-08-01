"""SMS/MMS backup XML adapter (SyncTech "SMS Backup & Restore" schema).

Streams via `lxml.etree.iterparse` and never loads the full tree — real
backups reach GB scale from base64-encoded MMS attachment blobs. Every
attribute read goes through `_val`, which treats the literal string
`"null"` (a placeholder Android's export tooling writes, not XML null) the
same as a genuinely missing attribute.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree

from locket.adapters.base import register
from locket.models import RawItem, SourceKind

_SMS_DIRECTION = {"1": "received", "2": "sent"}
_MMS_DIRECTION = {"1": "received", "2": "sent"}


def _val(elem, key: str, default: str | None = None) -> str | None:
    v = elem.get(key)
    return default if v in (None, "null") else v


def _ts_from_millis(ms: str | None) -> datetime | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)
    except (ValueError, TypeError, OSError):
        return None


def _parse_sms(elem, *, warnings: list[str] | None) -> RawItem | None:
    type_ = _val(elem, "type")
    if type_ == "3":  # draft — never sent, not part of the record
        return None
    direction = _SMS_DIRECTION.get(type_)
    if direction is None:
        # An unrecognized `type` code (e.g. "0", SMS Backup & Restore's "all messages"
        # QUERY code, which should never appear on a real per-message row) must never fall
        # through to a guessed sender -- the old code's `sender = ... if direction ==
        # "received" else "me"` silently misattributed every such row to "me".
        if warnings is not None:
            warnings.append(f"sms with unrecognized type={type_!r} skipped, sender not guessed")
        return None
    ts = _ts_from_millis(_val(elem, "date"))
    address = _val(elem, "address")
    contact_name = _val(elem, "contact_name")
    sender = contact_name if direction == "received" else "me"
    return RawItem.make(
        source=SourceKind.sms,
        ts=ts,
        sender=sender,
        text=_val(elem, "body"),
        thread=address,
        meta={"direction": direction, "address": address},
    )


def _parse_mms(elem, *, warnings: list[str] | None) -> RawItem | None:
    msg_box = _val(elem, "msg_box")
    direction = _MMS_DIRECTION.get(msg_box)
    if direction is None:
        # Same guess-if-not-recognized shape as _parse_sms above -- symmetric fix.
        if warnings is not None:
            warnings.append(f"mms with unrecognized msg_box={msg_box!r} skipped, sender not guessed")
        return None
    ts = _ts_from_millis(_val(elem, "date"))

    text_parts: list[str] = []
    has_media = False
    for part in elem.findall("parts/part"):
        ct = _val(part, "ct")
        if ct == "text/plain":
            t = _val(part, "text")
            if t:
                text_parts.append(t)
        elif ct and ct.startswith("application/smil"):
            continue  # layout metadata, not conversational content
        elif _val(part, "data") is not None:
            has_media = True

    address = None
    for addr in elem.findall("addrs/addr"):
        if _val(addr, "type") == "137":  # From
            address = _val(addr, "address")

    meta = {"direction": direction, "address": address}
    if has_media:
        meta["mms_media"] = True  # decoding is the photos adapter's job

    return RawItem.make(
        source=SourceKind.mms,
        ts=ts,
        sender=address if direction == "received" else "me",
        text="\n".join(text_parts) or None,
        thread=address,
        meta=meta,
    )


def parse_sms_xml(path: Path, *, warnings: list[str] | None = None) -> Iterator[RawItem]:
    """`warnings`, if given, gets one message appended per skipped row whose `type`/
    `msg_box` code wasn't recognized as sent/received/draft (see _parse_sms/_parse_mms)."""
    for _, elem in etree.iterparse(str(path), tag=("sms", "mms"), resolve_entities=False):
        item = _parse_sms(elem, warnings=warnings) if elem.tag == "sms" else _parse_mms(elem, warnings=warnings)
        elem.clear()
        if item is not None:
            yield item


register(".xml", parse_sms_xml)
