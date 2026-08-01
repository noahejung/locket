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


# ---------------------------------------------------------------------------
# media_path validation (fix-wave-1 item 11) -- a chat adapter's media_path is
# attacker-influenceable text (e.g. an instruction-shaped `<attached: ../../x>` message);
# `Path("/safe") / "/abs"` silently discards the left operand, so any future "open the
# original attachment" feature would inherit path traversal for free unless RawItem.make
# itself refuses to carry an absolute or `..`-containing media_path.
# ---------------------------------------------------------------------------


def test_media_path_absolute_posix_is_rejected_and_folded_into_text():
    item = RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="John",
        text="check this out",
        media_path="/etc/passwd",
        thread="t",
    )
    assert item.media_path is None
    assert "/etc/passwd" in item.text  # not silently discarded -- kept as inert text


def test_media_path_dotdot_traversal_is_rejected_and_folded_into_text():
    item = RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="John",
        text=None,
        media_path="../../secrets.txt",
        thread="t",
    )
    assert item.media_path is None
    assert "secrets.txt" in item.text


def test_media_path_windows_drive_absolute_is_rejected():
    item = RawItem.make(
        source=SourceKind.sms,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="x",
        text=None,
        media_path=r"C:\Windows\System32\config",
        thread="t",
    )
    assert item.media_path is None


def test_media_path_home_tilde_is_rejected():
    item = RawItem.make(
        source=SourceKind.sms,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender="x",
        text=None,
        media_path="~/.ssh/id_rsa",
        thread="t",
    )
    assert item.media_path is None


def test_media_path_ordinary_relative_path_is_kept():
    item = RawItem.make(
        source=SourceKind.photo,
        ts=datetime(2025, 1, 15, 10, 32, tzinfo=UTC),
        sender=None,
        text=None,
        media_path="Album1/photo.jpg",
        thread=None,
    )
    assert item.media_path == "Album1/photo.jpg"
