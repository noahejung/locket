"""Tests for the whatsapp_backup adapter (Phase 2, 2026-08-02 spec).

tests/fixtures/whatsapp_backup/ChatStorage-{personal,business}.sqlite are synthetic,
schema-verified fixtures (see tests/fixtures/gen_whatsapp_backup_fixture.py for the exact
scenario + rerun instructions) built to the schema documented in the Phase 2 spec.

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
    EncryptedBackupPassphraseRequired,
    IosBackupError,
    compute_file_id,
)
from locket.adapters.whatsapp_backup import (
    CHAT_STORAGE_RELATIVE_PATH,
    WA_MESSAGE_TYPE_DELETED,
    WA_MESSAGE_TYPE_STICKER,
    WA_MESSAGE_TYPE_SYSTEM,
    WHATSAPP_DOMAINS,
    group_by_thread,
    iter_messages,
    whatsapp_ts_to_unix,
)
from locket.models import SourceKind

FIX_DIR = Path(__file__).parent / "fixtures" / "whatsapp_backup"
PERSONAL_DB = FIX_DIR / "ChatStorage-personal.sqlite"
BUSINESS_DB = FIX_DIR / "ChatStorage-business.sqlite"


def _write_manifest_db(backup_dir: Path, entries: list[tuple[str, str, str]]) -> None:
    """entries: (domain, relativePath, fileID) rows for Manifest.db's own Files table --
    the exact shape _manifest_db_file_id queries against."""
    conn = sqlite3.connect(backup_dir / "Manifest.db")
    conn.execute("CREATE TABLE Files (fileID TEXT, domain TEXT, relativePath TEXT, flags INTEGER, file BLOB)")
    for domain, relative_path, file_id in entries:
        conn.execute(
            "INSERT INTO Files (fileID, domain, relativePath, flags) VALUES (?, ?, ?, 1)",
            (file_id, domain, relative_path),
        )
    conn.commit()
    conn.close()


def _stage_chat_storage(backup_dir: Path, domain: str, source_db: Path) -> str:
    file_id = compute_file_id(domain, CHAT_STORAGE_RELATIVE_PATH)
    fanout_dir = backup_dir / file_id[:2]
    fanout_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_db, fanout_dir / file_id)
    return file_id


def _make_backup(
    tmp_path: Path,
    *,
    personal_db: Path | None = PERSONAL_DB,
    business_db: Path | None = None,
    encrypted: bool = False,
    write_manifest_db: bool = True,
) -> Path:
    backup_dir = tmp_path / "9F1A2B3C-WHATSAPP0011223344"
    backup_dir.mkdir(parents=True, exist_ok=True)
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": encrypted}, fh)

    entries: list[tuple[str, str, str]] = []
    if personal_db is not None:
        file_id = _stage_chat_storage(backup_dir, WHATSAPP_DOMAINS["personal"], personal_db)
        entries.append((WHATSAPP_DOMAINS["personal"], CHAT_STORAGE_RELATIVE_PATH, file_id))
    if business_db is not None:
        file_id = _stage_chat_storage(backup_dir, WHATSAPP_DOMAINS["business"], business_db)
        entries.append((WHATSAPP_DOMAINS["business"], CHAT_STORAGE_RELATIVE_PATH, file_id))
    if write_manifest_db:
        _write_manifest_db(backup_dir, entries)
    return backup_dir


# ---------------------------------------------------------------------------
# Domain constants -- both personal AND Business, per the spec's explicit "probe both" ask
# ---------------------------------------------------------------------------


def test_both_whatsapp_domains_are_probed():
    assert WHATSAPP_DOMAINS["personal"] == "AppDomainGroup-group.net.whatsapp.WhatsApp.shared"
    assert WHATSAPP_DOMAINS["business"] == "AppDomainGroup-group.net.whatsapp.WhatsAppSMB.shared"
    assert WHATSAPP_DOMAINS["personal"] != WHATSAPP_DOMAINS["business"]


# ---------------------------------------------------------------------------
# whatsapp_ts_to_unix -- Core Data seconds, no ns-magnitude branch
# ---------------------------------------------------------------------------


def test_whatsapp_ts_seconds_conversion_matches_the_verified_epoch_offset():
    target = datetime(2025, 6, 1, 9, 0, 0, tzinfo=UTC)
    raw_seconds = (target - datetime(2001, 1, 1, tzinfo=UTC)).total_seconds()
    assert whatsapp_ts_to_unix(raw_seconds) == target


def test_whatsapp_ts_zero_and_none_both_mean_no_timestamp():
    assert whatsapp_ts_to_unix(0) is None
    assert whatsapp_ts_to_unix(0.0) is None
    assert whatsapp_ts_to_unix(None) is None


def test_whatsapp_ts_treats_a_large_raw_value_as_plain_seconds_not_nanoseconds():
    # A raw value well below ios_backup's own ns-magnitude threshold (1e12) but large enough
    # that dividing it by 1e9 (the wrong, iMessage-only interpretation) would land on a
    # different YEAR than treating it as plain seconds does (spec: "do not reuse the
    # magnitude-detection branch blindly"). Values at/above 1e12 aren't usable for this
    # check at all -- interpreted correctly as plain seconds, they'd land outside
    # datetime's representable range (year 9999), which is itself further confirmation
    # real WhatsApp timestamps never reach that magnitude.
    raw_seconds = 500_000_000.0  # ~15.8 years after the epoch -> year 2016
    result = whatsapp_ts_to_unix(raw_seconds)
    assert result.year == 2016
    wrongly_divided_by_ns_factor = whatsapp_ts_to_unix(raw_seconds / 1_000_000_000)
    assert wrongly_divided_by_ns_factor.year == 2001  # what an (incorrect) ns-interpretation would give
    assert result != wrongly_divided_by_ns_factor


# ---------------------------------------------------------------------------
# iter_messages -- personal-only, business-only, and combined domain probing
# ---------------------------------------------------------------------------


def test_iter_messages_personal_only_parses_every_row(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=PERSONAL_DB, business_db=None)
    items = list(iter_messages(backup_dir))
    assert len(items) == 8
    assert all(i.source == SourceKind.whatsapp for i in items)
    assert all(i.meta["whatsapp_variant"] == "personal" for i in items)


def test_iter_messages_business_only_parses_every_row(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=None, business_db=BUSINESS_DB)
    items = list(iter_messages(backup_dir))
    assert len(items) == 1
    assert items[0].meta["whatsapp_variant"] == "business"


def test_iter_messages_probes_both_domains_at_once(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=PERSONAL_DB, business_db=BUSINESS_DB)
    items = list(iter_messages(backup_dir))
    assert len(items) == 9  # 8 personal + 1 business
    variants = {i.meta["whatsapp_variant"] for i in items}
    assert variants == {"personal", "business"}


def test_iter_messages_neither_domain_present_yields_no_items_and_warns_not_errors(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=None, business_db=None)
    warnings: list[str] = []

    items = list(iter_messages(backup_dir, warnings=warnings))

    assert items == []
    assert len(warnings) == 1
    assert "personal" in warnings[0]
    assert "Business" in warnings[0] or "business" in warnings[0]


def test_iter_messages_no_warning_when_whatsapp_data_is_found(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=PERSONAL_DB)
    warnings: list[str] = []
    list(iter_messages(backup_dir, warnings=warnings))
    assert warnings == []


# ---------------------------------------------------------------------------
# Sender + name resolution -- "me" | group-member JID | thread JID, push-name priority
# ---------------------------------------------------------------------------


def test_iter_messages_from_me_sender_is_me(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-1-me"].sender == "me"


def test_iter_messages_1to1_sender_resolves_to_thread_jid(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-2-friend"].sender == "15551234567@s.whatsapp.net"


def test_iter_messages_push_name_wins_over_partner_name(tmp_path):
    # ZWAPROFILEPUSHNAME has "Jamie R (push)" for this contact; ZWACHATSESSION.ZPARTNERNAME
    # says "Jamie Rivera" -- push name must win, per the spec's explicit priority order.
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-2-friend"].meta["sender_display_name"] == "Jamie R (push)"


def test_iter_messages_group_member_sender_resolves_to_member_jid(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-4-group-sam"].sender == "15559876543@s.whatsapp.net"
    assert items["wa-msg-4-group-sam"].meta["sender_display_name"] == "Sam"


def test_iter_messages_group_member_without_push_name_falls_back_to_bare_jid(tmp_path):
    # A real, spec-documented limit -- a group member who never broadcast a push name and
    # isn't in iOS Contacts resolves only to a phone number.
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    msg = items["wa-msg-5-group-nopush"]
    assert msg.sender == "15551112222@s.whatsapp.net"
    assert msg.meta["sender_display_name"] == "15551112222@s.whatsapp.net"


def test_iter_messages_thread_display_name_is_the_chat_subject_or_partner_name(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-4-group-sam"].meta["thread_display_name"] == "Trip Crew"
    assert items["wa-msg-1-me"].meta["thread_display_name"] == "Jamie Rivera"


# ---------------------------------------------------------------------------
# Message-type gotchas: deleted tombstones, system messages, un-enumerated types, vCards
# ---------------------------------------------------------------------------


def test_iter_messages_deleted_message_text_is_none_never_a_substituted_literal(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    deleted = items["wa-msg-3-deleted"]
    assert deleted.text is None
    assert deleted.meta["deleted"] is True
    assert deleted.meta["message_type"] == WA_MESSAGE_TYPE_DELETED


def test_iter_messages_system_message_is_flagged_is_system(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    system_msg = items["wa-msg-6-system"]
    assert system_msg.is_system is True
    assert system_msg.meta["message_type"] == WA_MESSAGE_TYPE_SYSTEM


def test_iter_messages_ordinary_message_is_not_flagged_is_system(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    assert items["wa-msg-1-me"].is_system is False


def test_iter_messages_sticker_type_passes_through_without_crashing(tmp_path):
    # ZMESSAGETYPE beyond {6, 14, 15} is un-enumerated upstream (spec GOTCHA) -- sticker
    # (15) is one of the three the spec DOES name, and must degrade to "just carry the type
    # through", not crash or misclassify as system/deleted.
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    sticker = items["wa-msg-7-sticker"]
    assert sticker.meta["message_type"] == WA_MESSAGE_TYPE_STICKER
    assert sticker.is_system is False
    assert sticker.meta["deleted"] is False


def test_iter_messages_vcard_string_carried_in_meta_never_promoted_into_text(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = {i.id: i for i in iter_messages(backup_dir)}
    vcard_msg = items["wa-msg-8-vcard"]
    assert vcard_msg.text is None
    assert vcard_msg.meta["vcard_string"] == "BEGIN:VCARD\nFN:Contact Name\nEND:VCARD"


# ---------------------------------------------------------------------------
# id = ZSTANZAID (spec's WhatsApp analog to ios_backup.py's guid-as-id decision)
# ---------------------------------------------------------------------------


def test_iter_messages_id_is_the_raw_stanza_id(tmp_path):
    backup_dir = _make_backup(tmp_path)
    ids = {i.id for i in iter_messages(backup_dir)}
    assert "wa-msg-1-me" in ids
    assert "wa-msg-4-group-sam" in ids


def test_iter_messages_media_path_is_always_none_v1_scope(tmp_path):
    backup_dir = _make_backup(tmp_path)
    items = list(iter_messages(backup_dir))
    assert all(i.media_path is None for i in items)


# ---------------------------------------------------------------------------
# group_by_thread -- variant-qualified grouping key (personal chat_rowid=1 must not merge
# with business chat_rowid=1 -- each ChatStorage.sqlite numbers its own Z_PK independently)
# ---------------------------------------------------------------------------


def test_group_by_thread_splits_into_one_group_per_chat(tmp_path):
    backup_dir = _make_backup(tmp_path, personal_db=PERSONAL_DB, business_db=None)
    items = list(iter_messages(backup_dir))
    groups = group_by_thread(items)
    assert len(groups) == 2
    sizes = {label: len(g) for label, g in groups}
    assert sizes["whatsapp:Jamie Rivera"] == 4  # msg1, msg2, msg3, msg8
    assert sizes["whatsapp:Trip Crew"] == 4  # msg4, msg5, msg6, msg7


def test_group_by_thread_does_not_merge_colliding_chat_rowids_across_variants(tmp_path):
    # Personal's chat 1 ("Jamie Rivera") and Business's chat 1 ("Business Client") both have
    # raw chat_rowid=1 -- a variant-unaware grouping key would wrongly merge them into one
    # group of 5 (4 personal + 1 business) instead of two groups of 4 and 1.
    backup_dir = _make_backup(tmp_path, personal_db=PERSONAL_DB, business_db=BUSINESS_DB)
    items = list(iter_messages(backup_dir))
    groups = group_by_thread(items)
    assert len(groups) == 3
    sizes = {label: len(g) for label, g in groups}
    assert sizes["whatsapp:Jamie Rivera"] == 4
    assert sizes["whatsapp:Trip Crew"] == 4
    assert sizes["whatsapp:Business Client"] == 1


def test_group_by_thread_on_empty_input_returns_empty_list():
    assert group_by_thread([]) == []


# ---------------------------------------------------------------------------
# PRAGMA-branch degrade: an old/minimal schema missing ZSTANZAID/ZVCARDSTRING/ZFROMJID/
# ZTOJID must degrade gracefully, never sqlite3.OperationalError -- mirrors
# ios_backup.py's exact resilience mechanism.
# ---------------------------------------------------------------------------


def _old_schema_chat_storage(tmp_path: Path) -> Path:
    """Missing ZFROMJID, ZTOJID, ZSTANZAID, ZVCARDSTRING from ZWAMESSAGE -- exercises
    _available_message_columns/_build_query's NULL-substitution for all four at once,
    mirroring ios_backup.py's own old-schema test scope exactly: only ZWAMESSAGE's OWN
    column list is PRAGMA-branched (per the spec's literal ask); the join tables/keys
    (ZWAGROUPMEMBER, ZWAPROFILEPUSHNAME, ZCHATSESSION, ZGROUPMEMBER) stay fixed/present,
    same confidence level ios_backup.py's own fixed join structure was given."""
    db_path = tmp_path / "old_schema_ChatStorage.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE ZWACHATSESSION (Z_PK INTEGER PRIMARY KEY, ZCONTACTJID TEXT, ZPARTNERNAME TEXT);
        CREATE TABLE ZWAGROUPMEMBER (Z_PK INTEGER PRIMARY KEY, ZCHATSESSION INTEGER, ZMEMBERJID TEXT);
        CREATE TABLE ZWAPROFILEPUSHNAME (Z_PK INTEGER PRIMARY KEY, ZJID TEXT, ZPUSHNAME TEXT);
        CREATE TABLE ZWAMESSAGE (Z_PK INTEGER PRIMARY KEY, ZCHATSESSION INTEGER, ZGROUPMEMBER INTEGER,
            ZISFROMME INTEGER, ZMESSAGEDATE REAL, ZTEXT TEXT, ZMESSAGETYPE INTEGER);
        """
    )
    conn.execute(
        "INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME) VALUES (1, '15550001111@s.whatsapp.net', 'Old Contact')"
    )
    conn.execute(
        "INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZISFROMME, ZMESSAGEDATE, ZTEXT, ZMESSAGETYPE) "
        "VALUES (1, 1, 0, ?, 'hello from an old schema', 0)",
        ((datetime(2020, 1, 1, tzinfo=UTC) - datetime(2001, 1, 1, tzinfo=UTC)).total_seconds(),),
    )
    conn.commit()
    conn.close()
    return db_path


def test_iter_messages_old_schema_missing_columns_degrades_gracefully(tmp_path):
    old_db = _old_schema_chat_storage(tmp_path)
    backup_dir = _make_backup(tmp_path, personal_db=old_db)

    items = list(iter_messages(backup_dir))

    assert len(items) == 1
    item = items[0]
    assert item.text == "hello from an old schema"
    assert item.meta["vcard_string"] is None
    assert item.meta["from_jid"] is None
    assert item.meta["to_jid"] is None
    # ZSTANZAID absent entirely -- id must fall back to RawItem.make()'s derived hash, not a
    # bare None/empty id.
    assert item.id and item.id != "None"


# ---------------------------------------------------------------------------
# Preflight, FAIL LOUD only for real backup problems -- WhatsApp being absent is NOT one.
# ---------------------------------------------------------------------------


def test_iter_messages_not_a_backup_directory_at_all_raises_clear_error(tmp_path):
    with pytest.raises(IosBackupError, match="Manifest.plist"):
        list(iter_messages(tmp_path))


def test_iter_messages_encrypted_backup_without_passphrase_raises_clear_error(tmp_path):
    backup_dir = tmp_path / "encrypted-backup"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": True}, fh)

    with pytest.raises(EncryptedBackupPassphraseRequired, match="passphrase"):
        list(iter_messages(backup_dir))


def test_iter_messages_missing_manifest_db_yields_no_items_not_an_error(tmp_path):
    # Unlike ios_backup.py's sms.db (present on nearly every real backup, so ITS absence is
    # worth raising loudly), WhatsApp is an optional third-party app -- a completely missing
    # Manifest.db must degrade to "found nothing", never crash.
    backup_dir = tmp_path / "backup-no-manifest-db"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": False}, fh)

    warnings: list[str] = []
    items = list(iter_messages(backup_dir, warnings=warnings))

    assert items == []
    assert len(warnings) == 1


def test_iter_messages_placeholder_empty_manifest_db_does_not_crash(tmp_path):
    # Some test fixtures elsewhere in this repo (test_cli.py's _make_ios_backup_dir) use a
    # 0-byte Manifest.db purely for ios_backup.is_ios_backup_dir's presence-only check --
    # this adapter must tolerate that gracefully too, since discover_corpus_sources/
    # _ingest_source now call this adapter for EVERY ios backup directory unconditionally.
    backup_dir = tmp_path / "backup-with-placeholder-manifest-db"
    backup_dir.mkdir()
    with (backup_dir / "Manifest.plist").open("wb") as fh:
        plistlib.dump({"IsEncrypted": False}, fh)
    (backup_dir / "Manifest.db").write_bytes(b"")

    warnings: list[str] = []
    items = list(iter_messages(backup_dir, warnings=warnings))

    assert items == []
    assert len(warnings) == 1
