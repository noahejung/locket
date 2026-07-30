"""Regenerates tests/fixtures/instagram_thread/ — Meta's byte-accurate mojibake
corruption (every non-ASCII string is UTF-8 bytes mis-decoded as Latin-1).

Rerunnable: `python tests/fixtures/gen_instagram_fixture.py`
"""

import json
import pathlib


def corrupt(s: str) -> str:  # what Meta does to strings
    return s.encode("utf-8").decode("latin-1")


thread = {
    "participants": [{"name": corrupt("Sarah Kovács")}, {"name": "Noah Jung"}],
    "messages": [
        {
            "sender_name": corrupt("Sarah Kovács"),
            "timestamp_ms": 1737000000000,
            "content": corrupt("see you saturday 😊"),
        },
        {
            "sender_name": "Noah Jung",
            "timestamp_ms": 1736990000000,
            "photos": [{"uri": "messages/inbox/t/photos/1.jpg", "creation_timestamp": 1736990000}],
        },
    ],
}

out = pathlib.Path(__file__).with_name("instagram_thread")
out.mkdir(exist_ok=True)
(out / "message_1.json").write_text(json.dumps(thread, ensure_ascii=False), encoding="utf-8")

# message_2.json — plain-ASCII, timestamps interleave with message_1 to prove the re-sort.
thread2 = {
    "participants": [{"name": corrupt("Sarah Kovács")}, {"name": "Noah Jung"}],
    "messages": [
        {
            "sender_name": "Noah Jung",
            "timestamp_ms": 1736995000000,  # between message_1's two timestamps
            "content": "sounds good, see you then",
        },
        {
            "sender_name": corrupt("Sarah Kovács"),
            "timestamp_ms": 1736985000000,  # before both message_1 timestamps
            "content": "hey are we still on for saturday?",
        },
    ],
}
(out / "message_2.json").write_text(json.dumps(thread2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    print(f"wrote fixtures to {out}")
