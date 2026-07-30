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
