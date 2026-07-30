from datetime import UTC, datetime

import pytest

from locket.models import RawItem, SourceKind


def test_rawitem_id_is_deterministic():
    a = RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="John",
        text="hello",
        thread="john-thread",
    )
    b = RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="John",
        text="hello",
        thread="john-thread",
    )
    assert a.id == b.id and len(a.id) == 16


def test_rawitem_naive_ts_rejected():
    with pytest.raises(ValueError):
        RawItem.make(
            source=SourceKind.sms,
            ts=datetime(2025, 1, 1),
            sender="x",
            text="y",
            thread="t",
        )
