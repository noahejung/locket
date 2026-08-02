"""Tests for the ios_backup adapter (Phase 1, 2026-08-02 spec).

tests/fixtures/ios_backup/sms.db is a synthetic, schema-verified fixture (see
tests/fixtures/gen_ios_backup_fixture.py for the exact scenario + rerun instructions);
tests/fixtures/ios_backup/typedstream/{AttributedBodyTextOnly,URL,MultiPart,Blank} are
REAL binary attributedBody blobs from imessage-exporter's own test suite (used here only
as test input data for this independently-written parser, per the dispatch -- not vendored
source; imessage-exporter itself is GPL-3.0, see the adapter module's docstring and
THIRD_PARTY_NOTICES.md).

All tests here are offline/db-free -- no Postgres, no network.
"""

from __future__ import annotations

import plistlib
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from locket.adapters.ios_backup import (
    BackupInfo,
    EncryptedBackupPassphraseRequired,
    IosBackupError,
    NoBackupFoundError,
    _tier1_typedstream,  # private, but directly unit-tested per the dispatch's explicit ask
    _tier3_byte_scan,
    apple_ts_to_unix,
    compute_file_id,
    discover_backups,
    extract_text,
    find_backup_roots,
    group_by_thread,
    is_ios_backup_dir,
    iter_messages,
    read_backup_info,
)
from locket.models import SourceKind

FIX_DIR = Path(__file__).parent / "fixtures" / "ios_backup"
TYPEDSTREAM_DIR = FIX_DIR / "typedstream"
SMS_DB = FIX_DIR / "sms.db"


def _typedstream_blob(name: str) -> bytes:
    return (TYPEDSTREAM_DIR / name).read_bytes()


def _make_unencrypted_backup(tmp_path: Path, *, sms_db_source: Path = SMS_DB) -> Path:
    """Builds a minimal but structurally-real backup directory: Manifest.plist
    (IsEncrypted=False), Info.plist (a device name), and sms.db staged at its correctly-
    derived fanout path -- so the adapter's own discovery/staging code is exercised
    end-to-end, not bypassed."""
    backup_dir = tmp_path / "9F1A2B3C-DEADBEEF00112233"
    file_id = compute_file_id("HomeDomain", "Library/SMS/sms.db")
    fanout_dir = backup_dir / file_id[:2]
    fanout_dir.mkdir(parents=True)
    shutil.copyfile(sms_db_source, fanout_dir / file_id)
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": False}, fh)
    with (backup_dir / "Info.plist").open("wb") as fh:
        plistlib.dump({"Device Name": "Noah's iPhone"}, fh)
    (backup_dir / "Manifest.db").write_bytes(b"")  # presence-only for is_ios_backup_dir
    return backup_dir


# ---------------------------------------------------------------------------
# compute_file_id -- the research's verified constant
# ---------------------------------------------------------------------------


def test_compute_file_id_matches_the_verified_sms_db_hash():
    assert compute_file_id("HomeDomain", "Library/SMS/sms.db") == "3d0d7e5fb2ce288813306e4d4636395e047a3d28"


def test_compute_file_id_is_pure_and_deterministic():
    a = compute_file_id("HomeDomain", "Library/SMS/sms.db")
    b = compute_file_id("HomeDomain", "Library/SMS/sms.db")
    assert a == b
    assert compute_file_id("MediaDomain", "b/c.png") != a


# ---------------------------------------------------------------------------
# apple_ts_to_unix -- magnitude-based ns-vs-seconds branch
# ---------------------------------------------------------------------------


def test_apple_ts_modern_nanoseconds_branch():
    # 2025-06-01T10:00:00Z is 771,411,600s after the Apple epoch -> ns encoding multiplies by 1e9.
    target = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
    raw_ns = int((target - datetime(2001, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000_000)
    assert apple_ts_to_unix(raw_ns) == target


def test_apple_ts_legacy_seconds_branch():
    target = datetime(2013, 9, 4, 12, 0, 0, tzinfo=UTC)
    raw_seconds = int((target - datetime(2001, 1, 1, tzinfo=UTC)).total_seconds())
    assert raw_seconds < 1_000_000_000_000  # stays under the ns-magnitude threshold
    assert apple_ts_to_unix(raw_seconds) == target


def test_apple_ts_zero_and_none_both_mean_no_timestamp():
    assert apple_ts_to_unix(0) is None
    assert apple_ts_to_unix(None) is None


# ---------------------------------------------------------------------------
# Three-tier text extraction -- direct unit tests against the real fixture blobs
# ---------------------------------------------------------------------------


def test_tier1_typedstream_text_only_fixture():
    assert _tier1_typedstream(_typedstream_blob("AttributedBodyTextOnly")) == "Noter test"


def test_tier1_typedstream_url_fixture():
    assert _tier1_typedstream(_typedstream_blob("URL")) == "https://github.com/ReagentX/Logria"


def test_tier1_typedstream_multipart_fixture_recovers_attachment_placeholders():
    assert _tier1_typedstream(_typedstream_blob("MultiPart")) == "\ufffctest 1\ufffctest 2 \ufffctest 3"


def test_tier1_typedstream_blank_fixture_returns_empty_string_not_none():
    # Tier 1 succeeds but finds no string content -- "" is the correct verified result, and
    # callers (extract_text) must apply a truthy check, not `is not None`, on this.
    assert _tier1_typedstream(_typedstream_blob("Blank")) == ""


def test_tier1_typedstream_garbage_bytes_return_none_not_raise():
    assert _tier1_typedstream(b"not a real typedstream blob at all") is None


def test_tier3_byte_scan_text_only_fixture():
    assert _tier3_byte_scan(_typedstream_blob("AttributedBodyTextOnly")) == "Noter test"


def test_tier3_byte_scan_url_fixture():
    assert _tier3_byte_scan(_typedstream_blob("URL")) == "https://github.com/ReagentX/Logria"


def test_tier3_byte_scan_multipart_fixture():
    assert _tier3_byte_scan(_typedstream_blob("MultiPart")) == "\ufffctest 1\ufffctest 2 \ufffctest 3"


def test_tier3_byte_scan_blank_fixture_raises_value_error():
    with pytest.raises(ValueError, match="NoEndPattern|end marker"):
        _tier3_byte_scan(_typedstream_blob("Blank"))


# ---------------------------------------------------------------------------
# extract_text -- the full three-tier fallback, incl. tier ordering + genuinely-
# unrecoverable case
# ---------------------------------------------------------------------------


def test_extract_text_tier1_wins_over_populated_text_column():
    blob = _typedstream_blob("AttributedBodyTextOnly")
    assert extract_text(blob, "should never be used") == "Noter test"


def test_extract_text_falls_back_to_text_column_when_no_blob():
    assert extract_text(None, "plain text column value") == "plain text column value"


def test_extract_text_falls_through_tier1_empty_to_text_column():
    blob = _typedstream_blob("Blank")  # tier 1 -> "" (falsy) -> must fall through
    assert extract_text(blob, "fallback text") == "fallback text"


def test_extract_text_blank_blob_and_no_text_column_falls_to_tier3_which_also_fails():
    blob = _typedstream_blob("Blank")
    assert extract_text(blob, None) is None


def test_extract_text_both_none_is_the_genuinely_unrecoverable_unsent_case():
    assert extract_text(None, None) is None


def test_extract_text_empty_string_text_column_is_falsy_falls_through_to_tier3():
    blob = _typedstream_blob("URL")
    # An empty (not None) text column must not win over a real tier-1 result, and if tier 1
    # is unavailable, must not short-circuit tier 3 either.
    assert extract_text(blob, "") == "https://github.com/ReagentX/Logria"


# ---------------------------------------------------------------------------
# Backup discovery -- both roots probed, opaque UDID names never trusted
# ---------------------------------------------------------------------------


def test_find_backup_roots_returns_both_windows_roots(monkeypatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\testuser")
    monkeypatch.setenv("APPDATA", r"C:\Users\testuser\AppData\Roaming")
    roots = find_backup_roots()
    assert Path(r"C:\Users\testuser\Apple\MobileSync\Backup") in roots
    assert Path(r"C:\Users\testuser\AppData\Roaming\Apple Computer\MobileSync\Backup") in roots
    assert len(roots) == 2


def test_is_ios_backup_dir_true_when_both_files_present(tmp_path):
    (tmp_path / "Manifest.plist").write_bytes(b"")
    (tmp_path / "Manifest.db").write_bytes(b"")
    assert is_ios_backup_dir(tmp_path) is True


def test_is_ios_backup_dir_false_when_manifest_db_missing(tmp_path):
    (tmp_path / "Manifest.plist").write_bytes(b"")
    assert is_ios_backup_dir(tmp_path) is False


def test_is_ios_backup_dir_false_for_an_ordinary_directory(tmp_path):
    (tmp_path / "some_photo.jpg").write_bytes(b"")
    assert is_ios_backup_dir(tmp_path) is False


def test_discover_backups_raises_naming_both_searched_roots_when_none_found(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    with pytest.raises(NoBackupFoundError) as exc_info:
        discover_backups(roots=[root_a, root_b])
    assert str(root_a) in str(exc_info.value)
    assert str(root_b) in str(exc_info.value)


def test_discover_backups_tolerates_a_nonexistent_root(tmp_path):
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    backup_dir = _make_unencrypted_backup(real_root)
    nonexistent_root = tmp_path / "does_not_exist_at_all"

    found = discover_backups(roots=[nonexistent_root, real_root])

    assert found == [backup_dir]


def test_discover_backups_ignores_decoy_directories_lacking_manifest_plist(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "not_a_backup").mkdir()
    backup_dir = _make_unencrypted_backup(root)

    found = discover_backups(roots=[root])

    assert found == [backup_dir]


def test_discover_backups_finds_backups_under_both_roots_at_once(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    backup_a = _make_unencrypted_backup(root_a)
    backup_b = _make_unencrypted_backup(root_b)

    found = discover_backups(roots=[root_a, root_b])

    assert set(found) == {backup_a, backup_b}


def test_read_backup_info_reads_device_name_from_info_plist_not_folder_name(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    info = read_backup_info(backup_dir)
    assert isinstance(info, BackupInfo)
    assert info.device_name == "Noah's iPhone"
    assert info.is_encrypted is False


def test_read_backup_info_reports_encrypted_flag_from_manifest_plist(tmp_path):
    backup_dir = tmp_path / "encrypted-backup"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": True}, fh)
    info = read_backup_info(backup_dir)
    assert info.is_encrypted is True


def test_read_backup_info_handles_missing_plists_gracefully(tmp_path):
    info = read_backup_info(tmp_path)
    assert info.device_name is None
    assert info.is_encrypted is False


# ---------------------------------------------------------------------------
# iter_messages -- end-to-end against the real synthetic sms.db fixture
# ---------------------------------------------------------------------------


def test_iter_messages_parses_every_row_in_the_fixture(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    assert len(items) == 12
    assert all(i.source == SourceKind.imessage for i in items)


def test_iter_messages_id_is_the_raw_message_guid(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    ids = {i.id for i in items}
    assert "msg-1-me-modern" in ids
    assert "msg-10-group-url" in ids


def test_iter_messages_sender_is_me_or_the_handle(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-1-me-modern"].sender == "me"
    assert items["msg-2-friend-attributedbody"].sender == "+15551234567"
    assert items["msg-10-group-url"].sender == "friend1@example.com"


def test_iter_messages_recovers_text_from_attributedbody_when_text_column_is_null(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-2-friend-attributedbody"].text == "Noter test"
    assert items["msg-10-group-url"].text == "https://github.com/ReagentX/Logria"
    assert items["msg-11-group-multipart"].text == "\ufffctest 1\ufffctest 2 \ufffctest 3"


def test_iter_messages_blank_attributedbody_yields_none_text_not_a_crash(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-12-group-blank"].text is None


def test_iter_messages_unsent_message_has_none_text_and_does_not_crash(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-5-unsent"].text is None


def test_iter_messages_tapback_is_flagged_is_system(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    tapback = items["msg-3-tapback"]
    assert tapback.is_system is True
    assert tapback.meta["associated_message_type"] == 2000
    assert tapback.meta["associated_message_guid"] == "p:0/msg-2-friend-attributedbody"


def test_iter_messages_ordinary_message_is_not_flagged_is_system(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-1-me-modern"].is_system is False


def test_iter_messages_legacy_seconds_row_converts_to_the_correct_chronological_date(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    legacy = items["msg-4-legacy-seconds"]
    assert legacy.ts == datetime(2013, 9, 4, 12, 0, 0, tzinfo=UTC)


def test_iter_messages_results_are_sorted_by_converted_timestamp_not_raw_date(tmp_path):
    # The legacy (seconds-epoch, 2013) row must sort chronologically BEFORE every modern
    # (ns-epoch, 2025) row, despite being message ROWID 4 (inserted after ROWID 1-3).
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    timestamps = [i.ts for i in items]
    assert timestamps == sorted(timestamps)
    assert items[0].id == "msg-4-legacy-seconds"


def test_iter_messages_group_chat_does_not_fan_out_per_participant(tmp_path):
    # chat 2 has 2 participants (chat_handle_join) but only 3 messages -- proves
    # chat_handle_join is correctly excluded from the canonical per-message join.
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    group_items = [i for i in items if i.meta["chat_rowid"] == 2]
    assert len(group_items) == 3


def test_iter_messages_thread_display_name_prefers_group_name(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-10-group-url"].meta["thread_display_name"] == "Trip Planning"


def test_iter_messages_media_path_is_always_none_v1_scope(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    assert all(i.media_path is None for i in items)


def test_iter_messages_service_is_passed_through_opaquely(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["msg-1-me-modern"].meta["service"] == "iMessage"
    assert items["msg-4-legacy-seconds"].meta["service"] == "SMS"


def test_iter_messages_no_warning_when_count_clears_the_low_threshold(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    warnings: list[str] = []
    list(iter_messages(backup_dir, warnings=warnings))
    assert warnings == []


# ---------------------------------------------------------------------------
# group_by_thread -- pure grouping over already-parsed RawItems
# ---------------------------------------------------------------------------


def test_group_by_thread_splits_into_one_group_per_chat_rowid(tmp_path):
    backup_dir = _make_unencrypted_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    groups = group_by_thread(items)
    assert len(groups) == 2
    sizes = {label: len(g) for label, g in groups}
    assert sizes["imessage:+15551234567"] == 9
    assert sizes["imessage:Trip Planning"] == 3


def test_group_by_thread_on_empty_input_returns_empty_list():
    assert group_by_thread([]) == []


# ---------------------------------------------------------------------------
# Preflight, FAIL LOUD -- no backup / low count / encrypted-no-passphrase
# ---------------------------------------------------------------------------


def _minimal_sms_db(tmp_path: Path, *, message_count: int) -> Path:
    """A tiny standalone sms.db (not the rich fixture) for the low-message-count warning
    test, whose row count needs to be independently controllable."""
    db_path = tmp_path / "tiny.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT NOT NULL, text TEXT,
            handle_id INTEGER, service TEXT, date INTEGER, is_from_me INTEGER);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT NOT NULL, service TEXT NOT NULL);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT NOT NULL, chat_identifier TEXT,
            room_name TEXT, display_name TEXT);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER, PRIMARY KEY(chat_id, message_id));
        """
    )
    conn.execute("INSERT INTO handle VALUES (1, '+15550001111', 'iMessage')")
    conn.execute("INSERT INTO chat VALUES (1, 'g', '+15550001111', NULL, NULL)")
    for i in range(message_count):
        conn.execute(
            "INSERT INTO message (ROWID, guid, text, handle_id, service, date, is_from_me) "
            "VALUES (?, ?, ?, 1, 'iMessage', ?, 0)",
            (i + 1, f"guid-{i}", f"message {i}", 700000000 + i),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, ?)", (i + 1,))
    conn.commit()
    conn.close()
    return db_path


def test_iter_messages_warns_about_icloud_messages_when_count_is_suspiciously_low(tmp_path):
    tiny_db = _minimal_sms_db(tmp_path, message_count=2)
    backup_dir = _make_unencrypted_backup(tmp_path / "backup", sms_db_source=tiny_db)
    warnings: list[str] = []

    items = list(iter_messages(backup_dir, warnings=warnings))

    assert len(items) == 2
    assert len(warnings) == 1
    assert "iCloud" in warnings[0]
    assert "Messages" in warnings[0]


def test_iter_messages_missing_sms_db_raises_clear_error_not_empty_result(tmp_path):
    backup_dir = tmp_path / "backup-with-no-sms-db"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": False}, fh)

    with pytest.raises(IosBackupError, match="sms.db"):
        list(iter_messages(backup_dir))


def test_iter_messages_encrypted_backup_without_passphrase_raises_clear_error(tmp_path):
    backup_dir = tmp_path / "encrypted-backup"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": True}, fh)

    with pytest.raises(EncryptedBackupPassphraseRequired, match="passphrase"):
        list(iter_messages(backup_dir))


def test_iter_messages_not_a_backup_directory_at_all_raises_clear_error(tmp_path):
    with pytest.raises(IosBackupError, match="Manifest.plist"):
        list(iter_messages(tmp_path))


# ---------------------------------------------------------------------------
# PRAGMA-branch: schema is version-gated -- an older backup missing several columns must
# degrade gracefully (NULL-substituted), never sqlite3.OperationalError.
# ---------------------------------------------------------------------------


def _old_schema_sms_db(tmp_path: Path) -> Path:
    """Missing attributedBody, date_edited, associated_message_type,
    associated_message_guid, cache_has_attachments -- exercises _available_message_columns/
    _build_query's NULL-substitution for every one of them at once."""
    db_path = tmp_path / "old_schema.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT NOT NULL, text TEXT,
            handle_id INTEGER, service TEXT, date INTEGER, is_from_me INTEGER);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT NOT NULL, service TEXT NOT NULL);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT NOT NULL, chat_identifier TEXT,
            room_name TEXT, display_name TEXT);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER, PRIMARY KEY(chat_id, message_id));
        """
    )
    conn.execute("INSERT INTO handle VALUES (1, '+15550001111', 'SMS')")
    conn.execute("INSERT INTO chat VALUES (1, 'g', '+15550001111', NULL, NULL)")
    conn.execute(
        "INSERT INTO message (ROWID, guid, text, handle_id, service, date, is_from_me) "
        "VALUES (1, 'old-1', 'hello from an old schema', 1, 'SMS', 700000000, 0)"
    )
    conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    conn.commit()
    conn.close()
    return db_path


def test_iter_messages_old_schema_missing_columns_degrades_gracefully(tmp_path):
    old_db = _old_schema_sms_db(tmp_path)
    backup_dir = _make_unencrypted_backup(tmp_path / "backup", sms_db_source=old_db)

    items = list(iter_messages(backup_dir))

    assert len(items) == 1
    item = items[0]
    assert item.text == "hello from an old schema"
    assert item.meta["date_edited"] is None
    assert item.meta["associated_message_type"] is None
    assert item.meta["associated_message_guid"] is None
    assert item.meta["has_attachments"] is False
    assert item.is_system is False  # None assoc_type must not be misread as a tapback


# ---------------------------------------------------------------------------
# Live-gated: only runs against a REAL backup on THIS machine, self-skips otherwise.
# Never prints/logs actual message content (privacy; also the reproduced Windows cp1252
# crash on real content, attributedbody.md).
# ---------------------------------------------------------------------------


def _real_unencrypted_backup() -> Path | None:
    try:
        backups = discover_backups()
    except NoBackupFoundError:
        return None
    for backup_dir in backups:
        info = read_backup_info(backup_dir)
        if not info.is_encrypted:
            return backup_dir
    return None


@pytest.mark.skipif(
    _real_unencrypted_backup() is None,
    reason="no real unencrypted iOS backup found on this machine at either Windows root",
)
def test_live_real_backup_parses_without_raising():
    backup_dir = _real_unencrypted_backup()
    assert backup_dir is not None
    warnings: list[str] = []

    items = list(iter_messages(backup_dir, warnings=warnings))

    assert all(i.source == SourceKind.imessage for i in items)
    assert all(isinstance(i.id, str) and i.id for i in items)
    # Never assert/print message content here -- this may be Noah's real corpus.
