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


# ---------------------------------------------------------------------------
# unmatched-line counting (fix-wave-1 item 5a) -- a line with no header match AND no
# pending message to continue was silently dropped with no signal at all.
# ---------------------------------------------------------------------------


def test_well_formed_file_reports_zero_unmatched_lines():
    warnings: list[str] = []
    list(parse_whatsapp(FIX / "whatsapp_android.txt", thread="team", warnings=warnings))
    assert warnings == []


def test_orphan_line_with_no_pending_message_is_counted_not_silently_dropped(tmp_path):
    # The very first line of the file is in a header shape neither regex recognizes (a
    # locale variant with a narrow no-break space before AM/PM, say) -- there is no
    # "pending" message yet for it to continue, so pre-fix this line vanished with zero
    # trace. It should now be counted and surfaced via `warnings`.
    p = tmp_path / "orphan.txt"
    p.write_text(
        "this line matches no header shape at all\n"
        "1/15/25, 10:32 AM - John: real message\n",
        encoding="utf-8",
    )
    warnings: list[str] = []
    items = list(parse_whatsapp(p, thread="t", warnings=warnings))
    assert len(items) == 1  # the one real message still parses
    assert len(warnings) == 1
    assert "1" in warnings[0]  # names the count
    assert str(p) in warnings[0] or p.name in warnings[0]  # names the file


def test_fully_unparseable_file_is_loud_zero_items_and_a_warning(tmp_path):
    # A whole file in a format neither regex recognizes at all -- zero items parsed, and
    # this must be LOUD (a warning naming every dropped line), never a silent empty result
    # indistinguishable from "this file genuinely had nothing in it."
    p = tmp_path / "unparseable.txt"
    p.write_text(
        "totally unrecognized line one\ntotally unrecognized line two\n",
        encoding="utf-8",
    )
    warnings: list[str] = []
    items = list(parse_whatsapp(p, thread="t", warnings=warnings))
    assert items == []
    assert len(warnings) == 1
    assert "2" in warnings[0]  # both orphan lines counted


# ---------------------------------------------------------------------------
# media_path validation, layer 2 (fix-wave-1 item 11) -- proves RawItem.make's rejection
# (layer 1, tests/test_models.py) actually fires end-to-end when a real adapter parses a
# crafted export file, not just when constructed directly.
# ---------------------------------------------------------------------------


def test_android_file_attached_marker_sets_media_path(tmp_path):
    # The android "(file attached)" branch had zero coverage before this test -- added
    # alongside fix-wave-2 item 9's character-class tightening (see _ATTACH's comment) to
    # lock in the legitimate-filename case the tightened regex must keep matching.
    p = tmp_path / "attach.txt"
    p.write_text(
        "1/15/25, 3:15 PM - John: IMG-20250115-WA0002.jpg (file attached)\n",
        encoding="utf-8",
    )

    items = list(parse_whatsapp(p, thread="t"))

    assert len(items) == 1
    assert items[0].media_path == "IMG-20250115-WA0002.jpg"
    assert items[0].text is None


def test_traversal_attachment_path_becomes_plain_text_not_a_media_item(tmp_path):
    p = tmp_path / "malicious.txt"
    p.write_text("[15/01/25, 22:41:03] John: <attached: ../../etc/passwd>\n", encoding="utf-8")

    items = list(parse_whatsapp(p, thread="t"))

    assert len(items) == 1
    assert items[0].media_path is None
    assert "etc/passwd" in items[0].text  # kept as inert, describable text -- not discarded
