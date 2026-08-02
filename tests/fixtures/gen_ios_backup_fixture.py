"""Regenerates tests/fixtures/ios_backup/sms.db — a synthetic sms.db built to the
verified schema (sms-db-schema.md), covering every branch the ios_backup adapter has to
handle in one file.

Rerunnable: `python tests/fixtures/gen_ios_backup_fixture.py`

Apple-epoch conversion here is deliberately re-derived independently from
locket.adapters.ios_backup.apple_ts_to_unix (a different-shaped formula: timedelta from
the epoch datetime, vs. the adapter's unix-epoch-plus-offset) rather than importing and
calling it — otherwise a round-trip test would only prove encode/decode are inverses of
each other, not that either is actually correct.

Scenario (12 messages total, comfortably over LOW_MESSAGE_COUNT_THRESHOLD so the "happy
path" test doesn't also trip the low-message-count warning):

  chat 1 (1:1, "+15551234567", 9 messages):
    msg1  me, plain text, modern (ns-epoch)
    msg2  friend, text=NULL, attributedBody=AttributedBodyTextOnly fixture ("Noter test")
    msg3  friend, tapback (associated_message_type=2000, "Loved") targeting msg2
    msg4  me, plain text, LEGACY seconds-epoch (2013) — must sort chronologically FIRST
          within this chat despite being inserted between modern rows
    msg5  me, text=NULL, attributedBody=NULL — the genuinely-unrecoverable unsent/
          retracted case
    msg6-9  ordinary back-and-forth plain-text messages (padding to clear the low-count
            threshold, and proving multi-message windows unaffected by the above)

  chat 2 (group, "chat-group-nyc-trip", room_name="Trip Planning", 2 participants via
  chat_handle_join, 3 messages — proves a group message does NOT fan out per participant):
    msg10  friend1, attributedBody=URL fixture ("https://github.com/ReagentX/Logria")
    msg11  friend2, attributedBody=MultiPart fixture (multi-part U+FFFC placeholders)
    msg12  me, attributedBody=Blank fixture — tier 1 empty + tier 3 raises -> text=None
"""

import pathlib
import sqlite3
from datetime import UTC, datetime

OUT_DIR = pathlib.Path(__file__).with_name("ios_backup")
OUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUT_DIR / "sms.db"
TYPEDSTREAM_DIR = OUT_DIR / "typedstream"

_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)


def _apple_ns(dt: datetime) -> int:
    """Nanoseconds since the Apple epoch -- the iOS 11+ (modern) encoding."""
    return int((dt - _APPLE_EPOCH).total_seconds() * 1_000_000_000)


def _apple_seconds(dt: datetime) -> int:
    """Seconds since the Apple epoch -- the legacy pre-iOS-11 encoding."""
    return int((dt - _APPLE_EPOCH).total_seconds())


def _blob(name: str) -> bytes:
    return (TYPEDSTREAM_DIR / name).read_bytes()


SCHEMA = """
CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT NOT NULL,
    text TEXT,
    attributedBody BLOB,
    handle_id INTEGER,
    service TEXT,
    date INTEGER,
    date_read INTEGER,
    date_delivered INTEGER,
    date_edited INTEGER,
    date_retracted INTEGER,
    is_from_me INTEGER,
    associated_message_guid TEXT,
    associated_message_type INTEGER,
    associated_message_emoji TEXT,
    cache_has_attachments INTEGER,
    item_type INTEGER
);

CREATE TABLE handle (
    ROWID INTEGER PRIMARY KEY,
    id TEXT NOT NULL,
    service TEXT NOT NULL,
    person_centric_id TEXT
);

CREATE TABLE chat (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT NOT NULL,
    style INTEGER,
    chat_identifier TEXT,
    service_name TEXT,
    room_name TEXT,
    display_name TEXT
);

CREATE TABLE chat_message_join (
    chat_id INTEGER,
    message_id INTEGER,
    message_date INTEGER,
    PRIMARY KEY (chat_id, message_id)
);

CREATE TABLE chat_handle_join (
    chat_id INTEGER,
    handle_id INTEGER,
    UNIQUE(chat_id, handle_id)
);
"""


def build() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # -- handles --------------------------------------------------------------------
    conn.execute("INSERT INTO handle (ROWID, id, service) VALUES (1, '+15551234567', 'iMessage')")
    conn.execute("INSERT INTO handle (ROWID, id, service) VALUES (2, 'friend1@example.com', 'iMessage')")
    conn.execute("INSERT INTO handle (ROWID, id, service) VALUES (3, 'friend2@example.com', 'iMessage')")

    # -- chats ------------------------------------------------------------------------
    conn.execute(
        "INSERT INTO chat (ROWID, guid, chat_identifier, room_name, display_name) "
        "VALUES (1, 'iMessage;-;+15551234567', '+15551234567', NULL, NULL)"
    )
    conn.execute(
        "INSERT INTO chat (ROWID, guid, chat_identifier, room_name, display_name) "
        "VALUES (2, 'iMessage;+;chat-group-nyc-trip', 'chat-group-nyc-trip', "
        "'chat-group-nyc-trip', 'Trip Planning')"
    )
    conn.execute("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (1, 1)")
    conn.execute("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (2, 2)")
    conn.execute("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (2, 3)")

    # -- chat 1 (1:1) messages --------------------------------------------------------
    messages = [
        # (ROWID, guid, text, attributedBody, handle_id, service, date, date_edited,
        #  is_from_me, assoc_guid, assoc_type)
        (
            1,
            "msg-1-me-modern",
            "On my way!",
            None,
            None,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
        (
            2,
            "msg-2-friend-attributedbody",
            None,
            _blob("AttributedBodyTextOnly"),
            1,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 1, 0, tzinfo=UTC)),
            0,
            0,
            None,
            0,
        ),
        (
            3,
            "msg-3-tapback",
            None,
            None,
            1,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 2, 0, tzinfo=UTC)),
            0,
            0,
            "p:0/msg-2-friend-attributedbody",
            2000,  # "Loved" tapback added
        ),
        (
            4,
            "msg-4-legacy-seconds",
            "legacy row from the actual early era",
            None,
            None,
            "SMS",
            _apple_seconds(datetime(2013, 9, 4, 12, 0, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
        (
            5,
            "msg-5-unsent",
            None,
            None,
            None,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 3, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
        (
            6,
            "msg-6-friend-reply",
            "Sounds good, see you at 6",
            None,
            1,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 4, 0, tzinfo=UTC)),
            0,
            0,
            None,
            0,
        ),
        (
            7,
            "msg-7-me-reply",
            "actually let's do 6:30",
            None,
            None,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 5, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
        (
            8,
            "msg-8-friend-reply",
            "works for me",
            None,
            1,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 6, 0, tzinfo=UTC)),
            0,
            0,
            None,
            0,
        ),
        (
            9,
            "msg-9-me-reply",
            "see you then",
            None,
            None,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 10, 7, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
        # -- chat 2 (group) messages ---------------------------------------------------
        (
            10,
            "msg-10-group-url",
            None,
            _blob("URL"),
            2,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 11, 0, 0, tzinfo=UTC)),
            0,
            0,
            None,
            0,
        ),
        (
            11,
            "msg-11-group-multipart",
            None,
            _blob("MultiPart"),
            3,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 11, 1, 0, tzinfo=UTC)),
            0,
            0,
            None,
            0,
        ),
        (
            12,
            "msg-12-group-blank",
            None,
            _blob("Blank"),
            None,
            "iMessage",
            _apple_ns(datetime(2025, 6, 1, 11, 2, 0, tzinfo=UTC)),
            0,
            1,
            None,
            0,
        ),
    ]

    for (
        rowid,
        guid,
        text,
        blob,
        handle_id,
        service,
        date,
        date_edited,
        is_from_me,
        assoc_guid,
        assoc_type,
    ) in messages:
        conn.execute(
            "INSERT INTO message (ROWID, guid, text, attributedBody, handle_id, service, date, "
            "date_edited, is_from_me, associated_message_guid, associated_message_type, "
            "cache_has_attachments) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (rowid, guid, text, blob, handle_id, service, date, date_edited, is_from_me, assoc_guid, assoc_type),
        )

    chat1_message_ids = range(1, 10)
    chat2_message_ids = range(10, 13)
    for message_id in chat1_message_ids:
        conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)", (message_id,))
    for message_id in chat2_message_ids:
        conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (2, ?)", (message_id,))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    build()
    print(f"wrote fixture to {DB_PATH}")
