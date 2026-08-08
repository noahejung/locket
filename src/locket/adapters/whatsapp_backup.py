"""WhatsApp backup adapter — reads `ChatStorage.sqlite` directly out of the same on-disk
iPhone backup `ios_backup.py` reads `sms.db` from. Kills the manual per-thread "Export Chat"
chore WhatsApp otherwise requires, the same way Phase 1 killed it for iMessage/SMS/RCS.

Phase 2 scope only (spec: Claude/specs/2026-08-02-locket-ios-backup-adapter.md §Phase 2).
Hand-rolled stdlib `sqlite3`, matching both whatsapp.py's ("avoid the heavy GPL dep") and
ios_backup.py's ("read facts from a reference implementation, never vendor/shell out to it")
ethos -- does NOT shell out to WhatsApp-Chat-Exporter (emits HTML/JSON for humans, not
RawItems, adding a translation layer with its own drift risk).

Probes BOTH the personal WhatsApp domain (`AppDomainGroup-group.net.whatsapp.WhatsApp.shared`)
and the Business variant (`...WhatsAppSMB.shared`) -- silently missing Business data is
exactly the failure mode the spec names explicitly.

Explicitly v1, shipped without (mirrors ios_backup.py's own v1 list):
  - Media resolution: `ZWAMEDIAITEM` / `media_path` is always None on every RawItem here.
  - vCard *parsing*: `ZVCARDSTRING` is carried through opaquely in `meta`, never promoted
    into `text` -- it "double-duties" as vCard text AND a MIME-type override (spec GOTCHA),
    so folding it into visible text risks injecting arbitrary vCard-format content as if it
    were a real message.
  - Deleted-message *reconstruction*: tombstone rows (`ZMESSAGETYPE=14`) keep `text=None` and
    a `meta["deleted"]=True` flag -- this adapter does NOT substitute exporter-style literal
    "Message deleted" text, preserving the real/deleted distinction for downstream consumers
    rather than baking WhatsApp-Chat-Exporter's own UI convention into the fact record.

Zero non-stdlib dependencies in the common (unencrypted) path, same as ios_backup.py --
`iphone_backup_decrypt` is imported lazily, only inside the function that needs it, and only
reached on the (auto-detected, never assumed) encrypted branch.
"""

from __future__ import annotations

import contextlib
import plistlib
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from locket.adapters.ios_backup import (
    APPLE_EPOCH_OFFSET,
    EncryptedBackupPassphraseRequired,
    IosBackupError,
)
from locket.models import RawItem, SourceKind

# Personal WhatsApp and WhatsApp Business are DIFFERENT iOS app-group domains with their own
# independent ChatStorage.sqlite -- probing only "personal" silently misses every Business
# WhatsApp conversation, with no error (spec's explicit Phase 2 warning). Both are always
# probed, mirroring ios_backup.find_backup_roots's "always both roots" convention.
WHATSAPP_DOMAINS: dict[str, str] = {
    "personal": "AppDomainGroup-group.net.whatsapp.WhatsApp.shared",
    "business": "AppDomainGroup-group.net.whatsapp.WhatsAppSMB.shared",
}
CHAT_STORAGE_RELATIVE_PATH = "ChatStorage.sqlite"

# ZWAMESSAGE.ZMESSAGETYPE codes named explicitly by the spec -- every other value is passed
# through opaquely in meta["message_type"], never errored on (spec: "ZMESSAGETYPE codes
# beyond {6=system, 14=deleted, 15=sticker} are un-enumerated upstream").
WA_MESSAGE_TYPE_SYSTEM = 6
WA_MESSAGE_TYPE_DELETED = 14
WA_MESSAGE_TYPE_STICKER = 15


# ---------------------------------------------------------------------------
# Timestamps -- Core Data reference date (2001-01-01 UTC), SECONDS ONLY. Same epoch offset
# as iMessage's sms.db, but WhatsApp's ZMESSAGEDATE (a REAL, not an INTEGER) never switches
# to nanoseconds the way sms.db's modern rows do -- reusing ios_backup.apple_ts_to_unix's
# magnitude-detection branch here would be wrong (spec: "do not reuse the magnitude-detection
# branch blindly").
# ---------------------------------------------------------------------------


def whatsapp_ts_to_unix(raw: float | int | None) -> datetime | None:
    """`raw` falsy (0/0.0/None) means "no timestamp", matching apple_ts_to_unix's own
    falsy-is-unset convention in the sibling adapter."""
    if not raw:
        return None
    return datetime.fromtimestamp(raw + APPLE_EPOCH_OFFSET, tz=UTC)


# ---------------------------------------------------------------------------
# File location -- via Manifest.db's own Files table (spec's explicitly preferred generic
# lookup, "survives future layout changes"), NOT a re-derived compute_file_id hash the way
# ios_backup.py's sms.db lookup does it. Both approaches produce the same fileID for a real
# backup (fileID IS SHA1(domain-relativePath) by construction) -- this module follows the
# spec's literal instruction for Phase 2 rather than copying Phase 1's approach verbatim.
# ---------------------------------------------------------------------------


def _manifest_db_file_id(backup_dir: Path, domain: str, relative_path: str) -> str | None:
    """Returns the on-disk fileID for (domain, relativePath) per Manifest.db's Files table,
    or None for every "can't tell" case -- Manifest.db missing, present but not a real SQLite
    database (some test fixtures use an empty placeholder file purely for
    ios_backup.is_ios_backup_dir's presence-only check), or no matching row. Never raises:
    unlike sms.db (present on nearly every real backup, so ITS absence is worth raising
    loudly in ios_backup.py), WhatsApp is an optional third-party app -- "can't confirm it's
    here" is a legitimate, non-error outcome for this adapter (see iter_messages's own
    warning-not-exception design below)."""
    manifest_db_path = backup_dir / "Manifest.db"
    if not manifest_db_path.is_file():
        return None
    uri = f"file:{manifest_db_path.as_posix()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute(
                "SELECT fileID FROM Files WHERE domain = ? AND relativePath = ? AND flags = 1",
                (domain, relative_path),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Staging: produce a Path per found variant ("personal"/"business") to a queryable, private
# temp-copy of that variant's ChatStorage.sqlite. Mirrors ios_backup._staged_sms_db's two-
# branch shape (byte-copy unencrypted / iphone_backup_decrypt encrypted), generalized to
# "zero, one, or two variants found" instead of always-exactly-one.
# ---------------------------------------------------------------------------


def _decrypt_chat_storage(backup_dir: Path, passphrase: str, domain: str, output_path: Path) -> bool:
    """Returns True if `domain` had a ChatStorage.sqlite to decrypt, False if that domain
    simply isn't present in this backup (not installed) -- distinct from a real decrypt
    failure (wrong passphrase), which raises IosBackupError same as ios_backup.py's
    equivalent helper."""
    try:
        from iphone_backup_decrypt import EncryptedBackup
    except ImportError as exc:  # pragma: no cover - dependency is always installed per pyproject.toml
        raise IosBackupError("backup is encrypted but iphone-backup-decrypt is not installed") from exc

    backup = None
    try:
        backup = EncryptedBackup(backup_directory=str(backup_dir), passphrase=passphrase)
        backup.extract_file(
            relative_path=CHAT_STORAGE_RELATIVE_PATH, domain_like=domain, output_filename=str(output_path)
        )
    except FileNotFoundError:
        return False
    except ValueError as exc:
        # Same bare-ValueError-for-wrong-passphrase behavior ios_backup.py's equivalent
        # helper already documents (encrypted-backups.md GOTCHA) -- no custom exception
        # hierarchy in iphone_backup_decrypt for this.
        raise IosBackupError(f"failed to decrypt backup at {backup_dir} (wrong passphrase?): {exc}") from exc
    finally:
        # Same proactive-cleanup-then-drop-reference pattern as ios_backup.py's
        # _decrypt_sms_db -- iphone_backup_decrypt's own __del__-based cleanup is
        # documented best-effort only.
        if backup is not None:
            cleanup = getattr(backup, "_cleanup", None)
            if callable(cleanup):
                with contextlib.suppress(Exception):
                    cleanup()
            del backup

    return output_path.is_file()


@contextlib.contextmanager
def _staged_chat_storage_dbs(backup_dir: Path, *, passphrase: str | None) -> Iterator[dict[str, Path]]:
    """Yields {variant_label: staged_path} for every WhatsApp variant ("personal"/"business")
    actually found in this backup -- zero, one, or two entries. Whole temp directory removed
    on the way out, success or exception, same as ios_backup.py's equivalent."""
    manifest_plist = backup_dir / "Manifest.plist"
    if not manifest_plist.is_file():
        raise IosBackupError(f"{backup_dir} does not look like an iOS backup (no Manifest.plist found)")
    with manifest_plist.open("rb") as fh:
        manifest = plistlib.load(fh)
    encrypted = bool(manifest.get("IsEncrypted", False))

    if encrypted and not passphrase:
        # Mirrors ios_backup.py's own EncryptedBackupPassphraseRequired exactly: Manifest.db
        # itself is ciphertext when encrypted, so there is no way to even check whether
        # WhatsApp is present without a passphrase -- fail loud before touching anything,
        # same as the sms.db case, rather than silently reporting "WhatsApp not found".
        raise EncryptedBackupPassphraseRequired(
            f"backup at {backup_dir} is encrypted -- pass a backup passphrase "
            "(`locket ingest <path> --passphrase ...`, or set the "
            "LOCKET_IOS_BACKUP_PASSPHRASE environment variable) to read it; "
            "refusing to silently return zero messages"
        )

    with tempfile.TemporaryDirectory(prefix="locket-whatsapp-backup-") as tmp_dir:
        staged: dict[str, Path] = {}
        for variant, domain in WHATSAPP_DOMAINS.items():
            staged_path = Path(tmp_dir) / f"ChatStorage-{variant}.sqlite"
            if not encrypted:
                file_id = _manifest_db_file_id(backup_dir, domain, CHAT_STORAGE_RELATIVE_PATH)
                if file_id is None:
                    continue
                source_path = backup_dir / file_id[:2] / file_id
                if not source_path.is_file():
                    continue
                shutil.copyfile(source_path, staged_path)
                staged[variant] = staged_path
            else:
                if _decrypt_chat_storage(backup_dir, passphrase, domain, staged_path):
                    staged[variant] = staged_path
        yield staged


# ---------------------------------------------------------------------------
# The canonical JOIN -- ZWAMESSAGE ⋈ ZWACHATSESSION (thread identity ZCONTACTJID, name
# ZPARTNERNAME) ⋈ LEFT ZWAGROUPMEMBER (per-message group sender ZMEMBERJID, legitimately
# NULL for a 1:1 message) ⋈ LEFT ZWAPROFILEPUSHNAME (ZPUSHNAME, looked up by JID). Join keys
# (ZCHATSESSION, ZGROUPMEMBER) are fixed/unbranched -- same confidence level ios_backup.py's
# own JOIN structure was given, since the spec flags only ZWAMESSAGE's OWN column list as
# version-uncertain, not the join shape. The message-level column list below IS
# PRAGMA-branched, mirroring ios_backup.py's exact mechanism 1:1.
# ---------------------------------------------------------------------------

_MESSAGE_COLUMN_EXPRESSIONS: dict[str, str] = {
    "ZTEXT": "m.ZTEXT",
    "ZMESSAGETYPE": "m.ZMESSAGETYPE",
    "ZFROMJID": "m.ZFROMJID",
    "ZTOJID": "m.ZTOJID",
    "ZISFROMME": "m.ZISFROMME",
    "ZMESSAGEDATE": "m.ZMESSAGEDATE",
    "ZSTANZAID": "m.ZSTANZAID",
    "ZVCARDSTRING": "m.ZVCARDSTRING",
}


def _available_message_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(ZWAMESSAGE)").fetchall()}


def _build_query(available_columns: set[str]) -> str:
    select_parts = [
        (f"{expr} AS m_{column}" if column in available_columns else f"NULL AS m_{column}")
        for column, expr in _MESSAGE_COLUMN_EXPRESSIONS.items()
    ]
    select_clause = ",\n            ".join(select_parts)
    return f"""
        SELECT
            {select_clause},
            cs.Z_PK AS chat_rowid, cs.ZCONTACTJID AS thread_identifier,
            cs.ZPARTNERNAME AS thread_display_name,
            gm.ZMEMBERJID AS group_member_jid,
            pn_group.ZPUSHNAME AS group_member_push_name,
            pn_chat.ZPUSHNAME AS chat_contact_push_name
        FROM ZWAMESSAGE m
        LEFT JOIN ZWACHATSESSION cs ON cs.Z_PK = m.ZCHATSESSION
        LEFT JOIN ZWAGROUPMEMBER gm ON gm.Z_PK = m.ZGROUPMEMBER
        LEFT JOIN ZWAPROFILEPUSHNAME pn_group ON pn_group.ZJID = gm.ZMEMBERJID
        LEFT JOIN ZWAPROFILEPUSHNAME pn_chat ON pn_chat.ZJID = cs.ZCONTACTJID
        ORDER BY m.ZMESSAGEDATE
    """


def _row_to_raw_item(row: sqlite3.Row, *, variant: str) -> RawItem:
    is_from_me = bool(row["m_ZISFROMME"])
    message_type = row["m_ZMESSAGETYPE"]
    is_system = message_type == WA_MESSAGE_TYPE_SYSTEM
    is_deleted = message_type == WA_MESSAGE_TYPE_DELETED

    chat_rowid = row["chat_rowid"]
    thread_identifier = row["thread_identifier"]
    thread_display_name = row["thread_display_name"]
    group_member_jid = row["group_member_jid"]

    # Sender = "me" if ZISFROMME else group-member JID (group message) or thread JID (1:1) --
    # the spec's exact mapping; ZFROMJID/ZTOJID are NOT part of this primary rule (a 2017-era
    # schema the 2026 reference exporter no longer reads for sender identity -- spec GOTCHA),
    # so they're only consulted as a last-resort fallback if thread_identifier itself is
    # unexpectedly absent, and otherwise carried through opaquely in meta only. Name-
    # resolution priority (also spec's exact order): push-name -> partner-name-if-1:1 ->
    # bare phone JID. A group member who never broadcast a push-name and isn't in iOS
    # Contacts resolves only to a phone number -- a real limit of ChatStorage.sqlite alone,
    # not a bug (spec's explicit framing).
    if is_from_me:
        sender_jid: str | None = None
        display_name: str | None = None
    elif group_member_jid:
        sender_jid = group_member_jid
        display_name = row["group_member_push_name"] or group_member_jid
    else:
        sender_jid = thread_identifier or row["m_ZFROMJID"]
        display_name = row["chat_contact_push_name"] or thread_display_name or sender_jid
    sender = "me" if is_from_me else sender_jid

    # Deleted-message tombstones (ZMESSAGETYPE=14) keep text=None -- see the module
    # docstring's "explicitly v1" section for why this adapter does not substitute
    # exporter-style literal "Message deleted" text.
    text = None if is_deleted else row["m_ZTEXT"]
    ts = whatsapp_ts_to_unix(row["m_ZMESSAGEDATE"])
    stanza_id = row["m_ZSTANZAID"]
    thread_label = thread_identifier or (f"chat-{chat_rowid}" if chat_rowid is not None else None)

    meta = {
        "thread": thread_label,
        "chat_rowid": chat_rowid,
        "thread_identifier": thread_identifier,
        "thread_display_name": thread_display_name,
        "sender_display_name": display_name,
        "message_type": message_type,
        "deleted": is_deleted,
        "vcard_string": row["m_ZVCARDSTRING"],  # carried opaquely, never promoted into text
        "from_jid": row["m_ZFROMJID"],  # legacy column, not load-bearing for sender identity
        "to_jid": row["m_ZTOJID"],  # legacy column, not load-bearing for sender identity
        "whatsapp_variant": variant,  # "personal" | "business"
    }

    if stanza_id:
        # id = the message's own ZSTANZAID (spec analog to ios_backup.py's guid-as-id
        # decision) -- WhatsApp already assigns a stable, globally-unique stanza id per
        # message, so re-ingesting the same backup is idempotent via raw_items' own
        # ON CONFLICT (id) DO NOTHING with no extra work here. Bypasses RawItem.make()'s
        # derived-sha256 id, same deliberate deviation ios_backup.py documents at its own
        # equivalent call site.
        return RawItem(
            id=stanza_id,
            source=SourceKind.whatsapp,
            ts=ts,
            sender=sender,
            text=text,
            media_path=None,
            is_system=is_system,
            meta=meta,
        )
    # ZSTANZAID missing from this backup's schema (PRAGMA-branch degrade case) or NULL on
    # this specific row -- fall back to RawItem.make()'s derived-hash id instead of a bare
    # None id.
    return RawItem.make(
        source=SourceKind.whatsapp,
        ts=ts,
        sender=sender,
        text=text,
        is_system=is_system,
        thread=thread_label,
        meta=meta,
    )


def iter_messages(
    backup_dir: Path,
    *,
    passphrase: str | None = None,
    warnings: list[str] | None = None,
) -> Iterator[RawItem]:
    """Every WhatsApp message across BOTH the personal and Business variants found in
    `backup_dir`, as RawItems, oldest first. Raises IosBackupError/
    EncryptedBackupPassphraseRequired only for genuine backup-directory problems (not a
    backup at all, wrong passphrase) -- WhatsApp being entirely absent (not installed on this
    device) is NOT an error: `warnings`, if given, gets one message appended explaining both
    domains were checked and neither was found, so a caller can distinguish "genuinely no
    WhatsApp" from a silent bug."""
    items: list[RawItem] = []
    with _staged_chat_storage_dbs(backup_dir, passphrase=passphrase) as staged:
        if not staged and warnings is not None:
            warnings.append(
                "WhatsApp not found in this backup -- checked both the personal "
                f"({WHATSAPP_DOMAINS['personal']}) and Business "
                f"({WHATSAPP_DOMAINS['business']}) app domains; most likely WhatsApp was "
                "never installed on this device"
            )
        for variant, staged_path in staged.items():
            uri = f"file:{staged_path.as_posix()}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            try:
                available = _available_message_columns(conn)
                query = _build_query(available)
                items.extend(_row_to_raw_item(row, variant=variant) for row in conn.execute(query))
            finally:
                conn.close()

    items.sort(key=lambda item: (item.ts is None, item.ts))
    yield from items


def group_by_thread(items: Iterable[RawItem]) -> list[tuple[str, list[RawItem]]]:
    """Groups already-parsed RawItems by (whatsapp_variant, chat_rowid) -- variant is part of
    the key because personal and Business ChatStorage.sqlite each number their own
    ZWACHATSESSION.Z_PK independently starting from 1, so the same raw chat_rowid can refer
    to two entirely different chats across the two variants; grouping by chat_rowid alone
    would silently merge them. Mirrors ios_backup.group_by_thread's shape (order-preserving,
    (label, items) pairs) with a `whatsapp:` label prefix instead of `imessage:`."""
    buckets: dict[str, list[RawItem]] = {}
    order: list[str] = []
    for item in items:
        key = f"{item.meta.get('whatsapp_variant', '?')}:{item.meta.get('chat_rowid', 'unknown')}"
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
        labeled.append((f"whatsapp:{label_source}", group_items))
    return labeled


__all__ = [
    "CHAT_STORAGE_RELATIVE_PATH",
    "WA_MESSAGE_TYPE_DELETED",
    "WA_MESSAGE_TYPE_STICKER",
    "WA_MESSAGE_TYPE_SYSTEM",
    "WHATSAPP_DOMAINS",
    "group_by_thread",
    "iter_messages",
    "whatsapp_ts_to_unix",
]
