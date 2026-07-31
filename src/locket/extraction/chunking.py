"""Conversation windowing — pure function, no LLM/DB involved. Splits a chronological
RawItem list into windows that get sent to the extraction graph as one unit each."""

from __future__ import annotations

from datetime import timedelta

from locket.models import RawItem, SourceKind


def windows(items: list[RawItem], *, max_msgs: int = 40, gap_minutes: int = 180) -> list[list[RawItem]]:
    """A new window starts when consecutive messages are more than `gap_minutes` apart, or
    the current window reaches `max_msgs`. Photo items always form their own singleton
    window — there's no conversational continuity to chunk a lone photo against. System
    messages are not special-cased: they ride along in whatever window their timestamp
    lands them in, same as any other message."""
    if not items:
        return []

    result: list[list[RawItem]] = []
    current: list[RawItem] = []

    def flush() -> None:
        nonlocal current
        if current:
            result.append(current)
            current = []

    for item in items:
        if item.source == SourceKind.photo:
            flush()
            result.append([item])
            continue

        if current:
            prev = current[-1]
            gap_too_big = (
                prev.ts is not None
                and item.ts is not None
                and (item.ts - prev.ts) > timedelta(minutes=gap_minutes)
            )
            if gap_too_big or len(current) >= max_msgs:
                flush()

        current.append(item)

    flush()
    return result
