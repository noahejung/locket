from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from locket.adapters.whatsapp import parse_whatsapp
from locket.models import SourceKind

FIX = Path(__file__).parent / "fixtures"


def test_android_parse_counts_and_multiline():
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team"))
    assert len(items) == 8
    assert "Monday instead" in items[1].text  # continuation merged
    assert items[1].ts == items[0].ts


def test_system_messages_are_kept_and_flagged():
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team"))
    system = [i for i in items if i.is_system]
    assert any("changed the group description" in i.text for i in system)
    assert any("end-to-end encrypted" in i.text for i in system)
    assert all(i.sender is None for i in system)


def test_ios_brackets_lrm_and_attachment():
    items = list(parse_whatsapp(FIX / "whatsapp_ios.txt", thread="john"))
    assert len(items) == 3
    att = items[1]
    assert att.media_path == "00000042-PHOTO-2025-01-15-22-41-20.jpg"
    assert att.ts.second == 30


def test_all_items_are_whatsapp_kind():
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team"))
    assert all(i.source == SourceKind.whatsapp for i in items)


# ---------------------------------------------------------------------------
# source_tz -- WhatsApp header timestamps are LOCAL wall-clock, not UTC
# (fix-wave-1 item 4). A fixed, non-UTC offset -- never the machine's own tz --
# so these don't depend on where they run.
# ---------------------------------------------------------------------------

_EST = timezone(timedelta(hours=-5))


def test_explicit_source_tz_converts_local_wall_clock_to_utc():
    # "1/15/25, 10:32 AM" in the fixture is a local wall-clock reading. Interpreted as
    # EST (UTC-5), it must convert to 15:32 UTC -- not be tagged UTC verbatim (the bug).
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team", source_tz=_EST))
    assert items[0].ts == datetime(2025, 1, 15, 15, 32, tzinfo=UTC)


def test_no_source_tz_given_defaults_to_the_resolved_local_timezone(monkeypatch):
    # Don't depend on the machine's real tz -- monkeypatch the module's own local-tz
    # resolution to a fixed, known-non-UTC offset and prove the default path picks it up.
    monkeypatch.setattr("locket.adapters.whatsapp._local_tz", lambda: _EST)
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team"))
    assert items[0].ts == datetime(2025, 1, 15, 15, 32, tzinfo=UTC)


def test_source_tz_utc_is_a_no_op():
    items = list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team", source_tz=UTC))
    assert items[0].ts == datetime(2025, 1, 15, 10, 32, tzinfo=UTC)
