"""Shared vocabulary: RawItem and friends. No imports from sibling locket modules."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceKind(StrEnum):
    whatsapp = "whatsapp"
    instagram = "instagram"
    sms = "sms"
    mms = "mms"
    photo = "photo"


class FactKind(StrEnum):
    """Canonical fact-kind vocabulary. Defined here (not in extraction/schemas.py) so both
    the extraction-time schema and the store-time Fact model share one enum without a
    siblings-import — extraction/schemas.py imports this rather than redefining it."""

    person = "person"
    place = "place"
    event = "event"
    relationship = "relationship"
    habit = "habit"
    preference = "preference"


def _is_unsafe_media_path(path: str) -> bool:
    """Absolute paths and `..` traversal components are never a legitimate media_path --
    every adapter's contract is "a relative path inside the corpus dir" (see photos.py's
    `str(path.relative_to(root))`). A chat adapter's media_path is attacker-influenceable
    text (e.g. an instruction-shaped `<attached: ../../x>` message); rejecting both shapes
    here, at the single choke point every adapter's RawItem construction funnels through,
    is defense-in-depth for a future "open the original attachment" feature that would
    otherwise inherit path traversal for free (`Path("/safe") / "/abs"` silently discards
    the left operand). Checked as plain strings, not via pathlib, since the value can use
    either slash style regardless of which OS locket itself runs on."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("~"):
        return True
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():  # Windows drive letter, e.g. "C:\x"
        return True
    return ".." in normalized.split("/")


class RawItem(BaseModel):
    id: str  # deterministic: sha256 of (source, native identity fields)[:16]
    source: SourceKind
    ts: datetime | None = None  # timezone-aware UTC when known
    sender: str | None = None  # display name as the export gives it — resolution comes later
    text: str | None = None
    media_path: str | None = None  # relative path inside the corpus dir
    is_system: bool = False  # group-management / encryption notices etc.
    meta: dict[str, Any] = Field(default_factory=dict)  # adapter-specific extras

    @field_validator("ts")
    @classmethod
    def _ts_must_be_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("RawItem.ts must be timezone-aware (naive datetimes are rejected)")
        return v

    @classmethod
    def make(
        cls,
        *,
        source: SourceKind,
        ts: datetime | None,
        sender: str | None,
        text: str | None = None,
        media_path: str | None = None,
        thread: str | None = None,
        is_system: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> RawItem:
        meta = dict(meta or {})
        if thread is not None:
            meta["thread"] = thread
        if media_path is not None and _is_unsafe_media_path(media_path):
            # Never silently discard the mention -- fold it back into the visible text as
            # inert content (matching fix-wave-1 item 2's "untrusted data, not
            # instructions" framing) rather than a media_path a future filesystem-join
            # could traverse with.
            note = f"[unsafe media path rejected: {media_path!r}]"
            text = f"{text}\n{note}" if text else note
            media_path = None
        identity = "|".join(
            [
                str(source),
                thread or "",
                ts.isoformat() if ts is not None else "",
                sender or "",
                text or media_path or "",
            ]
        )
        item_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return cls(
            id=item_id,
            source=source,
            ts=ts,
            sender=sender,
            text=text,
            media_path=media_path,
            is_system=is_system,
            meta=meta,
        )


class Fact(BaseModel):
    """The store-time fact shape: an ExtractedFact (extraction/schemas.py, Task 11) plus the
    entity_ids and provenance that entity resolution / the extraction graph attach before
    Store.add_fact persists it. Superset of ExtractedFact's fields by design so a later stage
    can build one from the other with `Fact(**extracted.model_dump(), entity_ids=..., provenance=...)`.
    """

    kind: FactKind
    statement: str
    confidence: float = Field(ge=0, le=1)
    subjects: list[str] = Field(default_factory=list)
    place: str | None = None
    happened_at: str | None = None  # ISO date or date range, own facts.happened_at column
    entity_ids: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)  # raw_items.id values
