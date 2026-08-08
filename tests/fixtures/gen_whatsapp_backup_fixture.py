"""Regenerates tests/fixtures/whatsapp_backup/ChatStorage-{personal,business}.sqlite --
synthetic ChatStorage.sqlite databases built to the schema described in the Phase 2 spec
(Claude/specs/2026-08-02-locket-ios-backup-adapter.md §Phase 2), covering every branch the
whatsapp_backup adapter has to handle in one pair of files.

Rerunnable: `python tests/fixtures/gen_whatsapp_backup_fixture.py`

Core Data timestamp conversion here is deliberately re-derived independently from
locket.adapters.whatsapp_backup.whatsapp_ts_to_unix (a different-shaped formula: timedelta
from the epoch datetime, vs. the adapter's unix-epoch-plus-offset) rather than importing and
calling it -- otherwise a round-trip test would only prove encode/decode are inverses of
each other, not that either is actually correct. Mirrors gen_ios_backup_fixture.py's own
stated rationale for the identical choice.

Scenario -- ChatStorage-personal.sqlite:
  chat 1 (1:1, "15551234567@s.whatsapp.net", partner name "Jamie Rivera", 4 messages):
    msg1  me, plain text "on my way"
    msg2  friend, plain text "see you soon" -- sender resolves via thread identity (no
          ZFROMJID needed), display name resolves to the PUSH NAME ("Jamie R (push)"),
          proving push-name wins over ZPARTNERNAME ("Jamie Rivera") per the spec's priority
    msg3  friend, DELETED (ZMESSAGETYPE=14) -- text must come back None regardless of
          whatever garbage sits in ZTEXT on a real deleted row
    msg8  friend, ZVCARDSTRING set + ZTEXT NULL -- vcard_string must land in meta, and text
          must stay None (never promoted from the vcard blob)

  chat 2 (group, "120363012345678901@g.us", subject "Trip Crew", 4 messages, 2 members):
    msg4  group member 1 ("15559876543@s.whatsapp.net", HAS a push name "Sam") -- proves
          group-member sender resolution + push name
    msg5  group member 2 ("15551112222@s.whatsapp.net", NO push name registered) -- proves
          the "resolves only to a bare phone JID" real limit the spec names explicitly
    msg6  SYSTEM message (ZMESSAGETYPE=6) -- proves is_system=True
    msg7  STICKER (ZMESSAGETYPE=15) from group member 1 -- proves un-enumerated-beyond-
          {6,14,15} types still degrade to "just pass the type through", no crash

Scenario -- ChatStorage-business.sqlite (a SEPARATE, independent Z_PK numbering space, on
purpose -- proves whatsapp_backup.group_by_thread's variant-qualified grouping key, since
this file's chat_rowid=1 would otherwise collide with the personal file's chat_rowid=1):
  chat 1 (1:1, "15550009999@s.whatsapp.net", partner name "Business Client", 1 message):
    msg1  me, plain text "thanks for reaching out"
"""

import pathlib
import sqlite3
from datetime import UTC, datetime

OUT_DIR = pathlib.Path(__file__).with_name("whatsapp_backup")
OUT_DIR.mkdir(exist_ok=True)
PERSONAL_DB_PATH = OUT_DIR / "ChatStorage-personal.sqlite"
BUSINESS_DB_PATH = OUT_DIR / "ChatStorage-business.sqlite"

_CORE_DATA_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)


def _core_data_seconds(dt: datetime) -> float:
    """SECONDS since the Core Data reference date -- WhatsApp's ZMESSAGEDATE never uses
    nanoseconds the way sms.db's modern rows do."""
    return (dt - _CORE_DATA_EPOCH).total_seconds()


SCHEMA = """
CREATE TABLE ZWACHATSESSION (
    Z_PK INTEGER PRIMARY KEY,
    ZCONTACTJID TEXT,
    ZPARTNERNAME TEXT
);

CREATE TABLE ZWAGROUPMEMBER (
    Z_PK INTEGER PRIMARY KEY,
    ZCHATSESSION INTEGER,
    ZMEMBERJID TEXT
);

CREATE TABLE ZWAPROFILEPUSHNAME (
    Z_PK INTEGER PRIMARY KEY,
    ZJID TEXT,
    ZPUSHNAME TEXT
);

CREATE TABLE ZWAMESSAGE (
    Z_PK INTEGER PRIMARY KEY,
    ZCHATSESSION INTEGER,
    ZGROUPMEMBER INTEGER,
    ZFROMJID TEXT,
    ZTOJID TEXT,
    ZISFROMME INTEGER,
    ZMESSAGEDATE REAL,
    ZTEXT TEXT,
    ZMESSAGETYPE INTEGER,
    ZSTANZAID TEXT,
    ZVCARDSTRING TEXT
);
"""


def _build_personal() -> None:
    if PERSONAL_DB_PATH.exists():
        PERSONAL_DB_PATH.unlink()
    conn = sqlite3.connect(PERSONAL_DB_PATH)
    conn.executescript(SCHEMA)

    conn.execute(
        "INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME) "
        "VALUES (1, '15551234567@s.whatsapp.net', 'Jamie Rivera')"
    )
    conn.execute(
        "INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME) "
        "VALUES (2, '120363012345678901@g.us', 'Trip Crew')"
    )

    conn.execute(
        "INSERT INTO ZWAGROUPMEMBER (Z_PK, ZCHATSESSION, ZMEMBERJID) VALUES (1, 2, '15559876543@s.whatsapp.net')"
    )
    conn.execute(
        "INSERT INTO ZWAGROUPMEMBER (Z_PK, ZCHATSESSION, ZMEMBERJID) VALUES (2, 2, '15551112222@s.whatsapp.net')"
    )

    # Push name registered for chat 1's own 1:1 contact -- proves push-name wins over
    # ZPARTNERNAME. Deliberately NO push name row for group member 2 ("15551112222...").
    conn.execute(
        "INSERT INTO ZWAPROFILEPUSHNAME (Z_PK, ZJID, ZPUSHNAME) VALUES (1, '15551234567@s.whatsapp.net', 'Jamie R (push)')"
    )
    conn.execute(
        "INSERT INTO ZWAPROFILEPUSHNAME (Z_PK, ZJID, ZPUSHNAME) VALUES (2, '15559876543@s.whatsapp.net', 'Sam')"
    )

    messages = [
        # (Z_PK, ZCHATSESSION, ZGROUPMEMBER, ZFROMJID, ZISFROMME, ts, text, type, stanza, vcard)
        (1, 1, None, None, 1, datetime(2025, 6, 1, 9, 0, 0, tzinfo=UTC), "on my way", 0, "wa-msg-1-me", None),
        (
            2,
            1,
            None,
            None,
            0,
            datetime(2025, 6, 1, 9, 1, 0, tzinfo=UTC),
            "see you soon",
            0,
            "wa-msg-2-friend",
            None,
        ),
        (
            3,
            1,
            None,
            None,
            0,
            datetime(2025, 6, 1, 9, 2, 0, tzinfo=UTC),
            "this text should be ignored -- message is deleted",
            14,
            "wa-msg-3-deleted",
            None,
        ),
        (
            4,
            2,
            1,
            None,
            0,
            datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
            "let's go",
            0,
            "wa-msg-4-group-sam",
            None,
        ),
        (
            5,
            2,
            2,
            None,
            0,
            datetime(2025, 6, 1, 10, 1, 0, tzinfo=UTC),
            "same",
            0,
            "wa-msg-5-group-nopush",
            None,
        ),
        (
            6,
            2,
            None,
            None,
            0,
            datetime(2025, 6, 1, 10, 2, 0, tzinfo=UTC),
            "Trip Crew added Alex",
            6,
            "wa-msg-6-system",
            None,
        ),
        (7, 2, 1, None, 0, datetime(2025, 6, 1, 10, 3, 0, tzinfo=UTC), None, 15, "wa-msg-7-sticker", None),
        (
            8,
            1,
            None,
            None,
            0,
            datetime(2025, 6, 1, 9, 3, 0, tzinfo=UTC),
            None,
            0,
            "wa-msg-8-vcard",
            "BEGIN:VCARD\nFN:Contact Name\nEND:VCARD",
        ),
    ]
    for pk, chat, group_member, from_jid, is_from_me, ts, text, msg_type, stanza, vcard in messages:
        conn.execute(
            "INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZGROUPMEMBER, ZFROMJID, ZISFROMME, ZMESSAGEDATE, "
            "ZTEXT, ZMESSAGETYPE, ZSTANZAID, ZVCARDSTRING) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pk, chat, group_member, from_jid, is_from_me, _core_data_seconds(ts), text, msg_type, stanza, vcard),
        )

    conn.commit()
    conn.close()


def _build_business() -> None:
    if BUSINESS_DB_PATH.exists():
        BUSINESS_DB_PATH.unlink()
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    conn.executescript(SCHEMA)

    conn.execute(
        "INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME) "
        "VALUES (1, '15550009999@s.whatsapp.net', 'Business Client')"
    )
    conn.execute(
        "INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZGROUPMEMBER, ZFROMJID, ZISFROMME, ZMESSAGEDATE, "
        "ZTEXT, ZMESSAGETYPE, ZSTANZAID, ZVCARDSTRING) VALUES (1, 1, NULL, NULL, 1, ?, ?, 0, ?, NULL)",
        (_core_data_seconds(datetime(2025, 6, 2, 8, 0, 0, tzinfo=UTC)), "thanks for reaching out", "wa-biz-msg-1"),
    )

    conn.commit()
    conn.close()


def build() -> None:
    _build_personal()
    _build_business()


if __name__ == "__main__":
    build()
    print(f"wrote fixtures to {PERSONAL_DB_PATH} and {BUSINESS_DB_PATH}")
