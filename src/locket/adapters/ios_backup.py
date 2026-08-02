"""iOS backup adapter — reads Messages (`sms.db`) directly out of an on-disk iPhone
backup (Finder / Apple Devices app / iTunes format). This is the same mechanism
iMazing/Decipher use: the backup format is public, and every hard part has an OSS
reference implementation (imessage-exporter/imessage-database) that this module reads
for facts (schema, hashing scheme, timestamp epoch) without vendoring its GPL-3.0 code.

Phase 1 scope only (spec: Claude/specs/2026-08-02-locket-ios-backup-adapter.md) —
Messages via sms.db. WhatsApp's ChatStorage.sqlite (Phase 2) and pymobiledevice3
auto-trigger (Phase 3) are NOT built here.

Explicitly v1, shipped without (see the spec's "Explicitly v1 scope" section):
  - Attachment file resolution (media_path is always None on every RawItem here).
  - Cross-service thread dedup: one RawItem thread per raw `chat.ROWID` — a contact
    reached over both iMessage and SMS/RCS at different times shows as separate threads
    until v2 (`chat_lookup`/`person_centric_id`, both under-documented outside a single
    source per the research). Callers are told this explicitly (see cli.py/pipeline.py),
    never silently merged.
  - Unsent/retracted messages (`text` AND `attributedBody` both NULL) are genuinely
    unrecoverable by Apple's own design, not a parser bug — `extract_text` returns None
    for them and nothing here tries harder.

Zero non-stdlib dependencies in the common (unencrypted) path — `pytypedstream` (tier-1
text extraction) and `iphone_backup_decrypt`/`pycryptodome` (the encrypted-backup branch)
are imported lazily, only inside the functions that actually need them.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import plistlib
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from locket.models import RawItem, SourceKind

# A real phone's Messages history is hundreds to tens of thousands of rows; a backup that
# yields fewer than this is far more likely to be "Messages in iCloud" silently pruning
# local content (spec's #3 research finding -- the one failure mode that looks exactly
# like a working adapter that just happens to find nothing) than a phone that genuinely
# never used Messages. Purely a heuristic -- no authoritative source publishes a real
# threshold (attributedbody.md), so this is deliberately conservative, not load-bearing.
LOW_MESSAGE_COUNT_THRESHOLD = 10

APPLE_EPOCH_OFFSET = 978307200  # 2001-01-01 00:00:00 UTC, in Unix seconds
_NS_MAGNITUDE_THRESHOLD = 1_000_000_000_000  # raw values at/above this are nanoseconds, not seconds


class IosBackupError(Exception):
    """Base for every ios_backup adapter error. Deliberately loud everywhere it's raised
    (locket's adapter convention, restated by the Phase 1 spec) -- a caller must never see
    a silent empty result where a clear error was possible instead."""


class NoBackupFoundError(IosBackupError):
    """No backup located at either Windows root."""


class EncryptedBackupPassphraseRequired(IosBackupError):
    """`Manifest.plist` declares the backup encrypted and no passphrase was supplied."""


# ---------------------------------------------------------------------------
# Backup discovery -- probes both Windows roots, identifies devices by Info.plist only
# ---------------------------------------------------------------------------


def find_backup_roots() -> list[Path]:
    """Both possible Windows backup roots. Apple Devices app / Microsoft-Store iTunes
    writes to `%USERPROFILE%\\Apple\\MobileSync\\Backup`; legacy desktop (apple.com
    installer) iTunes writes to `%APPDATA%\\Apple Computer\\MobileSync\\Backup`. A
    machine that has run both historically can have real backups under both roots at
    once (backup-layout.md GOTCHA), so both are always returned -- never just the first
    one that happens to exist."""
    return [
        Path(os.path.expandvars(r"%USERPROFILE%\Apple\MobileSync\Backup")),
        Path(os.path.expandvars(r"%APPDATA%\Apple Computer\MobileSync\Backup")),
    ]


def is_ios_backup_dir(path: Path) -> bool:
    """The exact shape both `locket ingest <path>` and `pipeline run --corpus-dir <path>`
    use to auto-detect a raw backup directory (dispatch's literal rule): Manifest.plist
    AND Manifest.db both present. UDID folder names are opaque and never inspected."""
    return (path / "Manifest.plist").is_file() and (path / "Manifest.db").is_file()


def discover_backups(roots: list[Path] | None = None) -> list[Path]:
    """Every UDID subfolder under both Windows roots that looks like a real backup
    (Manifest.plist + Manifest.db present -- shape, not folder name, is the signal, since
    UDIDs are opaque). Raises NoBackupFoundError naming every searched root when nothing
    is found, rather than returning an empty list a caller could mistake for "a backup
    with zero messages" instead of "no backup at all"."""
    search_roots = roots if roots is not None else find_backup_roots()
    found: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and is_ios_backup_dir(child):
                found.append(child)
    if not found:
        searched = ", ".join(str(r) for r in search_roots)
        raise NoBackupFoundError(f"no iOS backup found -- searched: {searched}")
    return found


@dataclass
class BackupInfo:
    path: Path
    device_name: str | None
    is_encrypted: bool
    last_backup_date: datetime | None


def read_backup_info(backup_dir: Path) -> BackupInfo:
    """Device identity always comes from Info.plist's `Device Name` -- never the UDID
    folder name, which is an opaque identifier carrying no human-readable information
    (spec: "identify devices via Info.plist, never folder name")."""
    device_name = None
    info_plist = backup_dir / "Info.plist"
    if info_plist.is_file():
        with info_plist.open("rb") as fh:
            info = plistlib.load(fh)
        device_name = info.get("Device Name")

    is_encrypted = False
    last_backup_date: datetime | None = None
    manifest_plist = backup_dir / "Manifest.plist"
    if manifest_plist.is_file():
        with manifest_plist.open("rb") as fh:
            manifest = plistlib.load(fh)
        is_encrypted = bool(manifest.get("IsEncrypted", False))
        last_backup_date = manifest.get("Date")
        if last_backup_date is not None and last_backup_date.tzinfo is None:
            last_backup_date = last_backup_date.replace(tzinfo=UTC)

    return BackupInfo(
        path=backup_dir, device_name=device_name, is_encrypted=is_encrypted, last_backup_date=last_backup_date
    )


# ---------------------------------------------------------------------------
# File location -- SHA1(domain + "-" + relativePath), never a hardcoded lookup table
# ---------------------------------------------------------------------------


def compute_file_id(domain: str, relative_path: str) -> str:
    """The on-disk filename (and Manifest.db's `Files.fileID`) for any file in an iOS
    backup: `SHA1(domain + "-" + relativePath)`, hex-encoded. Identical on every
    device/backup since it depends only on domain+relativePath -- this one helper covers
    sms.db here and every other backup file (attachments, WhatsApp's ChatStorage.sqlite,
    contacts, ...) a future adapter needs, with no hardcoded hash table anywhere.

    Verified 4 independent ways during research (backup-layout.md):
    ``compute_file_id("HomeDomain", "Library/SMS/sms.db") ==
    "3d0d7e5fb2ce288813306e4d4636395e047a3d28"``.
    """
    return hashlib.sha1(f"{domain}-{relative_path}".encode()).hexdigest()


def _unencrypted_sms_db_path(backup_dir: Path) -> Path:
    file_id = compute_file_id("HomeDomain", "Library/SMS/sms.db")
    return backup_dir / file_id[:2] / file_id


# ---------------------------------------------------------------------------
# Staging: produce one queryable, private temp-copy Path to sms.db, regardless of branch.
# Unencrypted: "copy the blob + open read-only" (dispatch's exact instruction) -- never
# query the file in place inside Apple's own backup directory. Encrypted: delegate to
# iphone_backup_decrypt, which decrypts straight to a file of our choosing.
# ---------------------------------------------------------------------------


def _decrypt_sms_db(backup_dir: Path, passphrase: str, output_path: Path) -> None:
    try:
        from iphone_backup_decrypt import EncryptedBackup, RelativePath
    except ImportError as exc:  # pragma: no cover - dependency is always installed per pyproject.toml
        raise IosBackupError("backup is encrypted but iphone-backup-decrypt is not installed") from exc

    backup = None
    try:
        backup = EncryptedBackup(backup_directory=str(backup_dir), passphrase=passphrase)
        backup.extract_file(relative_path=RelativePath.TEXT_MESSAGES, output_filename=str(output_path))
    except ValueError as exc:
        # iphone_backup_decrypt raises bare ValueError for a wrong passphrase -- no custom
        # exception hierarchy exists in that library (encrypted-backups.md GOTCHA).
        raise IosBackupError(f"failed to decrypt backup at {backup_dir} (wrong passphrase?): {exc}") from exc
    finally:
        # iphone_backup_decrypt decrypts Manifest.db into a fresh tempfile.mkdtemp() dir
        # for the EncryptedBackup object's lifetime and only best-effort cleans it up in
        # __del__, which it explicitly documents may not run on crash (encrypted-
        # backups.md GOTCHA). Proactively call its own _cleanup if present, then drop the
        # last reference (CPython's refcounting triggers __del__ synchronously once this
        # is the only reference) -- try/finally, not "trust eventual GC", per the dispatch.
        if backup is not None:
            cleanup = getattr(backup, "_cleanup", None)
            if callable(cleanup):
                with contextlib.suppress(Exception):
                    cleanup()
            del backup

    if not output_path.is_file():
        raise IosBackupError(f"decryption did not produce sms.db for backup at {backup_dir}")


@contextlib.contextmanager
def _staged_sms_db(backup_dir: Path, *, passphrase: str | None) -> Iterator[Path]:
    """Yields a Path to a private, temp-directory copy of sms.db, staged one of two ways
    depending on `Manifest.plist`'s `IsEncrypted` (auto-detected, never a precondition):
    a plain byte-copy for the common unencrypted case, or iphone_backup_decrypt's
    decrypt-to-file output for the encrypted case. Either way, the whole temp directory is
    removed on the way out, success or exception."""
    manifest_plist = backup_dir / "Manifest.plist"
    if not manifest_plist.is_file():
        raise IosBackupError(f"{backup_dir} does not look like an iOS backup (no Manifest.plist found)")
    with manifest_plist.open("rb") as fh:
        manifest = plistlib.load(fh)
    encrypted = bool(manifest.get("IsEncrypted", False))

    with tempfile.TemporaryDirectory(prefix="locket-ios-backup-") as tmp_dir:
        staged_path = Path(tmp_dir) / "sms.db"
        if not encrypted:
            source_path = _unencrypted_sms_db_path(backup_dir)
            if not source_path.is_file():
                raise IosBackupError(
                    f"sms.db not found in backup at {backup_dir} (expected {source_path}) -- "
                    "Messages may never have been backed up on this device"
                )
            shutil.copyfile(source_path, staged_path)
        else:
            if not passphrase:
                raise EncryptedBackupPassphraseRequired(
                    f"backup at {backup_dir} is encrypted -- pass a backup passphrase "
                    "(`locket ingest <path> --passphrase ...`, or set the "
                    "LOCKET_IOS_BACKUP_PASSPHRASE environment variable) to read it; "
                    "refusing to silently return zero messages"
                )
            _decrypt_sms_db(backup_dir, passphrase, staged_path)
        yield staged_path


# ---------------------------------------------------------------------------
# Timestamps -- Apple epoch (2001-01-01 UTC), magnitude-detected ns (iOS 11+) vs s (legacy)
# ---------------------------------------------------------------------------


def apple_ts_to_unix(raw: int | None) -> datetime | None:
    """iOS 11+ (essentially every 2026 backup) stores `date`/`date_edited`/... as
    NANOSECONDS since the Apple epoch (2001-01-01 UTC); legacy pre-iOS-11 rows (2017 and
    older) store plain SECONDS. Detected by magnitude, not schema/iOS version, verified
    against imessage-database's own `get_local_time` (sms-db-schema.md). `raw` falsy (0 or
    None -- 0 is Apple's own "not set" sentinel for date_edited/date_retracted/etc.) means
    "no timestamp", not epoch-zero 2001-01-01."""
    if not raw:
        return None
    seconds_since_2001 = raw / 1_000_000_000 if raw >= _NS_MAGNITUDE_THRESHOLD else raw
    return datetime.fromtimestamp(seconds_since_2001 + APPLE_EPOCH_OFFSET, tz=UTC)


# ---------------------------------------------------------------------------
# Three-tier text extraction, mirroring imessage-exporter's own Message::parse_body():
# tier 1 full typedstream deserialize -> tier 2 the `text` column -> tier 3 legacy byte-
# scan. Every tier's exact algorithm here was independently verified against all 4 real
# fixtures in tests/fixtures/ios_backup/typedstream/ during research -- copied, not
# re-derived.
# ---------------------------------------------------------------------------


def _tier1_typedstream(blob: bytes) -> str | None:
    """Full typedstream deserialization via pytypedstream (pip name "pytypedstream",
    import name "typedstream") plus a recursive walk to the first NSString/
    NSMutableString.value -- NSAttributedString/NSMutableAttributedString aren't in the
    library's registered-class table, so the high-level API degrades to a generic
    GenericArchivedObject/TypedGroup wrapper that must be walked by hand (attributedbody.md).

    Broad `except Exception` throughout: the format is Apple's undocumented, reverse-
    engineered `streamtyped`, and any exception here means "try the next tier", never
    "this message has no text" (attributedbody.md GOTCHAS). Returns "" (not None) when the
    deserializer succeeds but finds no string content (the `Blank` fixture) -- callers
    must use a truthy check, not `is not None`, or this "successfully found nothing"
    result short-circuits the tier-2/tier-3 fallback.
    """
    try:
        import typedstream
        from typedstream.archiving import GenericArchivedObject, TypedGroup
        from typedstream.types.foundation import NSMutableString, NSString
    except ImportError:
        return None

    def find_first_string(obj: object, depth: int = 0, seen: set[int] | None = None) -> str | None:
        seen = seen if seen is not None else set()
        if depth > 50 or id(obj) in seen:
            return None
        seen.add(id(obj))
        if isinstance(obj, (NSString, NSMutableString)):
            return obj.value
        if isinstance(obj, GenericArchivedObject):
            if obj.super_object is not None:
                found = find_first_string(obj.super_object, depth + 1, seen)
                if found is not None:
                    return found
            for item in obj.contents:
                found = find_first_string(item, depth + 1, seen)
                if found is not None:
                    return found
        elif isinstance(obj, TypedGroup):
            for value in obj.values:
                found = find_first_string(value, depth + 1, seen)
                if found is not None:
                    return found
        return None

    try:
        root = typedstream.unarchive_from_data(blob)
        return find_first_string(root)
    except Exception:
        return None


_TIER3_START = bytes([0x01, 0x2B])
_TIER3_END = bytes([0x86, 0x84])


def _tier3_byte_scan(stream: bytes) -> str:
    """Ported from imessage-exporter's `util/streamtyped.rs` legacy byte-scan parser
    (facts only, not vendored source -- GPL-3.0 on the reference project, see the module
    docstring): find `[0x01, 0x2B]`, drain up to and including it; find `[0x86, 0x84]`,
    truncate there; decode UTF-8, drop the first 1 char on a strict decode or the first 3
    chars if it had to lossy-decode (replacement chars shift the offset). Verified byte-
    for-byte against all 4 real fixtures during research, including the `Blank` fixture's
    expected-failure case. Raises ValueError (not caught here) when neither marker is
    found -- the caller (`extract_text`) decides how to treat that."""
    n = len(stream)
    start_at = None
    for idx in range(0, max(n - 1, 0)):
        if stream[idx : idx + 2] == _TIER3_START:
            start_at = idx + 2
            break
    if start_at is None:
        raise ValueError("no start marker found in attributedBody stream")
    stream = stream[start_at:]
    n2 = len(stream)
    end_at = None
    for idx in range(1, max(n2 - 2, 1)):
        if stream[idx : idx + 2] == _TIER3_END:
            end_at = idx
            break
    if end_at is None:
        raise ValueError("no end marker found in attributedBody stream")
    stream = stream[:end_at]
    try:
        return stream.decode("utf-8")[1:]
    except UnicodeDecodeError:
        return stream.decode("utf-8", errors="replace")[3:]


def extract_text(attributed_body: bytes | None, text_column: str | None) -> str | None:
    """Three-tier fallback, in order: (1) full typedstream deserialize, when it recovers a
    non-empty string; (2) the message's own already-populated `text` column; (3) the
    legacy byte-scan, as a last resort. Returns None when every tier comes up empty --
    including the genuinely-unrecoverable unsent/retracted case, where both
    `attributed_body` and `text_column` are None by Apple's own design (not a parser bug;
    nothing here tries to recover it)."""
    if attributed_body:
        tier1 = _tier1_typedstream(attributed_body)
        if tier1:  # truthy, not `is not None` -- "" is tier 1's own "found nothing" result
            return tier1
    if text_column:
        return text_column
    if attributed_body:
        try:
            recovered = _tier3_byte_scan(attributed_body)
        except ValueError:
            return None
        return recovered or None
    return None


# ---------------------------------------------------------------------------
# The canonical JOIN -- message ⋈ chat_message_join ⋈ chat ⋈ handle, deliberately
# excluding chat_handle_join (participant membership, not per-message sender -- joining
# it here would fan out one row per participant per group message).
# ---------------------------------------------------------------------------

# column name -> its SELECT expression. PRAGMA table_info(message) is queried at runtime
# (_available_message_columns) and any column NOT present in THIS backup's schema version
# gets `NULL AS ...` substituted instead (_build_query) -- the schema is version-gated
# (sms-db-schema.md GOTCHA: "never hardcode a column list, or an older backup throws
# sqlite3.OperationalError: no such column"), so every wanted column goes through this
# same defensive substitution, not just the ones research happened to flag as version-
# specific.
_MESSAGE_COLUMN_EXPRESSIONS: dict[str, str] = {
    "guid": "m.guid",
    "text": "m.text",
    "attributedBody": "m.attributedBody",
    "is_from_me": "m.is_from_me",
    "service": "m.service",
    "date": "m.date",
    "date_edited": "m.date_edited",
    "associated_message_type": "m.associated_message_type",
    "associated_message_guid": "m.associated_message_guid",
    "cache_has_attachments": "m.cache_has_attachments",
}


def _available_message_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(message)").fetchall()}


def _build_query(available_columns: set[str]) -> str:
    select_parts = [
        (f"{expr} AS m_{column}" if column in available_columns else f"NULL AS m_{column}")
        for column, expr in _MESSAGE_COLUMN_EXPRESSIONS.items()
    ]
    select_clause = ",\n            ".join(select_parts)
    return f"""
        SELECT
            {select_clause},
            c.ROWID AS chat_rowid, c.chat_identifier AS thread_identifier,
            c.display_name AS thread_display_name, c.room_name AS thread_room_name,
            h.id AS sender_handle, h.service AS sender_service
        FROM message m
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN chat c ON c.ROWID = cmj.chat_id
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        ORDER BY m.date
    """


def _row_to_raw_item(row: sqlite3.Row) -> RawItem:
    is_from_me = bool(row["m_is_from_me"])
    sender = "me" if is_from_me else row["sender_handle"]
    text = extract_text(row["m_attributedBody"], row["m_text"])
    assoc_type = row["m_associated_message_type"]
    is_system = assoc_type not in (0, None)  # tapback/poll-vote rows (sms-db-schema.md)
    ts = apple_ts_to_unix(row["m_date"])
    date_edited = apple_ts_to_unix(row["m_date_edited"])
    chat_rowid = row["chat_rowid"]
    thread_identifier = row["thread_identifier"]
    thread_label = thread_identifier or (f"chat-{chat_rowid}" if chat_rowid is not None else None)

    # id = the message's own guid (spec's explicit mapping), NOT RawItem.make()'s derived
    # sha256 -- Apple already assigns a stable, globally-unique identity per message, so
    # re-ingesting the same backup is idempotent via raw_items' own ON CONFLICT (id) DO
    # NOTHING with no extra work here. media_path is always None (v1 scope: attachment
    # resolution deferred), so RawItem.make()'s unsafe-media-path guard has nothing to do
    # here either -- direct construction is the correct, not merely convenient, choice.
    return RawItem(
        id=row["m_guid"],
        source=SourceKind.imessage,
        ts=ts,
        sender=sender,
        text=text,
        media_path=None,
        is_system=is_system,
        meta={
            "thread": thread_label,
            "chat_rowid": chat_rowid,
            "thread_identifier": thread_identifier,
            "thread_display_name": row["thread_display_name"] or row["thread_room_name"],
            "service": row["m_service"],
            "associated_message_type": assoc_type,
            "associated_message_guid": row["m_associated_message_guid"],
            "date_edited": date_edited.isoformat() if date_edited is not None else None,
            "has_attachments": bool(row["m_cache_has_attachments"]),
        },
    )


def iter_messages(
    backup_dir: Path,
    *,
    passphrase: str | None = None,
    warnings: list[str] | None = None,
) -> Iterator[RawItem]:
    """The adapter's core entry point: every message in `backup_dir`'s sms.db, as
    RawItems, oldest first. Raises IosBackupError/EncryptedBackupPassphraseRequired
    per the module's fail-loud preflight convention rather than ever returning an empty
    result for a real problem (no sms.db, wrong/missing passphrase).

    `warnings`, if given, gets one message appended if the total message count is
    suspiciously low (LOW_MESSAGE_COUNT_THRESHOLD) -- the "Messages in iCloud" trap
    (spec's #3 research finding): when that setting is on, the device stops keeping
    message content locally, so ANY local backup (encrypted or not) silently captures
    only stubs, with no error, looking exactly like a working adapter that just found
    nothing.

    Results are re-sorted by CONVERTED timestamp after the query (not trusted from the
    raw `ORDER BY m.date`) -- raw `date` values aren't safely comparable across the
    seconds/nanoseconds epoch-magnitude boundary if both kinds of rows ever coexist
    (sms-db-schema.md GOTCHA); items with no timestamp sort last.
    """
    items: list[RawItem] = []
    with _staged_sms_db(backup_dir, passphrase=passphrase) as staged_path:
        uri = f"file:{staged_path.as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            available = _available_message_columns(conn)
            query = _build_query(available)
            items = [_row_to_raw_item(row) for row in conn.execute(query)]
        finally:
            conn.close()

    items.sort(key=lambda item: (item.ts is None, item.ts))

    if warnings is not None and len(items) < LOW_MESSAGE_COUNT_THRESHOLD:
        warnings.append(
            f"only {len(items)} message(s) found in backup at {backup_dir} -- if this "
            'seems low, check Settings -> [Apple ID] -> iCloud -> Messages on the '
            'device: when "Messages in iCloud" is on, message content is no longer '
            "kept locally, so any local backup (encrypted or not) captures only stubs, "
            "silently, with no error"
        )

    yield from items


def group_by_thread(items: Iterable[RawItem]) -> list[tuple[str, list[RawItem]]]:
    """Groups already-parsed RawItems by their raw `chat_rowid` -- v1 scope, per the spec:
    one group per raw chat, no cross-service dedup, so the same contact reached over both
    iMessage and SMS/RCS at different times produces two separate groups here. Order-
    preserving (first-seen chat order); mirrors pipeline.discover_corpus_sources's
    (label, items) contract so its groups can be extended directly with this function's
    output."""
    buckets: dict[str, list[RawItem]] = {}
    order: list[str] = []
    for item in items:
        key = str(item.meta.get("chat_rowid", "unknown"))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)

    labeled: list[tuple[str, list[RawItem]]] = []
    for key in order:
        group_items = buckets[key]
        label_source = (
            group_items[0].meta.get("thread_display_name") or group_items[0].meta.get("thread_identifier") or key
        )
        labeled.append((f"imessage:{label_source}", group_items))
    return labeled


__all__ = [
    "LOW_MESSAGE_COUNT_THRESHOLD",
    "BackupInfo",
    "EncryptedBackupPassphraseRequired",
    "IosBackupError",
    "NoBackupFoundError",
    "apple_ts_to_unix",
    "compute_file_id",
    "discover_backups",
    "extract_text",
    "find_backup_roots",
    "group_by_thread",
    "is_ios_backup_dir",
    "iter_messages",
    "read_backup_info",
]
