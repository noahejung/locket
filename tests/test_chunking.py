from __future__ import annotations

from datetime import UTC, datetime, timedelta

from locket.extraction.chunking import windows
from locket.models import RawItem, SourceKind


def _msg(i: int, ts: datetime, *, is_system: bool = False, source: SourceKind = SourceKind.whatsapp) -> RawItem:
    return RawItem.make(source=source, ts=ts, sender="John", text=f"msg {i}", is_system=is_system, thread="t")


def test_splits_on_time_gap():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [
        _msg(0, base),
        _msg(1, base + timedelta(minutes=10)),
        _msg(2, base + timedelta(hours=4)),  # >3h gap from the previous message
    ]
    ws = windows(items)
    assert len(ws) == 2
    assert [i.text for i in ws[0]] == ["msg 0", "msg 1"]
    assert [i.text for i in ws[1]] == ["msg 2"]


def test_exactly_gap_minutes_does_not_split():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [_msg(0, base), _msg(1, base + timedelta(minutes=180))]
    ws = windows(items, gap_minutes=180)
    assert len(ws) == 1  # not strictly greater than the threshold


def test_splits_at_max_msgs():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [_msg(i, base + timedelta(minutes=i)) for i in range(45)]
    ws = windows(items, max_msgs=40)
    assert len(ws) == 2
    assert len(ws[0]) == 40
    assert len(ws[1]) == 5


def test_system_messages_ride_along_in_their_window():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [
        _msg(0, base),
        _msg(1, base + timedelta(minutes=1), is_system=True),
        _msg(2, base + timedelta(minutes=2)),
    ]
    ws = windows(items)
    assert len(ws) == 1
    assert len(ws[0]) == 3
    assert ws[0][1].is_system


def test_photo_items_form_singleton_windows():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [
        _msg(0, base),
        _msg(1, base + timedelta(minutes=1), source=SourceKind.photo),
        _msg(2, base + timedelta(minutes=2)),
    ]
    ws = windows(items)
    assert len(ws) == 3
    assert ws[1][0].source == SourceKind.photo
    assert len(ws[1]) == 1


def test_two_consecutive_photos_each_get_their_own_window():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [
        _msg(0, base, source=SourceKind.photo),
        _msg(1, base + timedelta(seconds=1), source=SourceKind.photo),
    ]
    ws = windows(items)
    assert len(ws) == 2
    assert all(len(w) == 1 for w in ws)


def test_empty_input():
    assert windows([]) == []
