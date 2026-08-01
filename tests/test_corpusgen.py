"""The corpus generator's own test: render -> parse with Tasks 3-6's real
adapters -> assert every message round-trips. This is the whole point of the
synthetic corpus — if this test is green, the adapters actually work against
export shapes indistinguishable from the real ones, and demo_corpus/ can
stand in for a real export in every other test/eval in this repo.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from locket.adapters.instagram import parse_instagram_thread
from locket.adapters.photos import parse_photos
from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.models import SourceKind

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "demo_corpus"
CONVERSATIONS = json.loads((REPO_ROOT / "corpusgen" / "conversations.json").read_text(encoding="utf-8"))


def _canonical(thread: str) -> list[dict]:
    return [m for m in CONVERSATIONS if m["thread"] == thread]


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _find_canonical(thread: str, needle: str) -> dict:
    matches = [m for m in _canonical(thread) if m["text"] and needle in m["text"]]
    assert len(matches) == 1, f"expected exactly one canonical match for {needle!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# personas
# ---------------------------------------------------------------------------


def test_personas_have_five_distinct_canonical_names():
    from corpusgen.personas import PERSONAS

    assert len(PERSONAS) == 5
    assert len({p.canonical for p in PERSONAS}) == 5
    assert "Jeffrey Williams" in {p.canonical for p in PERSONAS}  # protagonist / device owner


# ---------------------------------------------------------------------------
# WhatsApp round trip (Android dash convention)
# ---------------------------------------------------------------------------


def _check_whatsapp_thread(filename: str, thread: str):
    canonical = _canonical(thread)
    # source_tz=UTC pinned: corpusgen renders WhatsApp header text straight from each
    # canonical message's UTC timestamp (no timezone reasoning at generation time -- see
    # corpusgen/renderers.py), so by construction the rendered wall-clock numbers ARE the
    # canonical UTC numbers. This round-trip test is about content/format survival, not
    # timezone-conversion correctness (locket.adapters.whatsapp's own tests cover that) --
    # without pinning UTC here, parse_whatsapp's real fix (fix-wave-1 item 4: interpret
    # header timestamps via the system's local timezone by default) would shift every
    # parsed ts by this machine's UTC offset and break the ts-survival assertions below for
    # a reason unrelated to what this test is actually checking.
    items = list(parse_whatsapp(DEMO / "whatsapp" / filename, thread=thread, source_tz=UTC))

    assert len(items) == len(canonical)
    assert sum(1 for i in items if i.is_system) == sum(1 for m in canonical if m.get("system"))
    assert all(i.source == SourceKind.whatsapp for i in items)

    # Content survives exactly as a multiset — timestamps get minute-truncated
    # by the real Android export format (no seconds field), so we don't
    # compare full (speaker, text, ts) triples here; see the spot-checks below
    # for explicit timestamp survival.
    parsed_content = Counter((i.sender, i.text) for i in items)
    canonical_content = Counter(
        (m["speaker"] if not m.get("system") else None, m["text"]) for m in canonical
    )
    assert parsed_content == canonical_content

    # At least one multiline message survives with its newline intact.
    assert any(i.text and "\n" in i.text for i in items)
    # <Media omitted> passes through as ordinary text (that's what real
    # WhatsApp exports do — no special-casing needed).
    assert any(i.text == "<Media omitted>" for i in items)

    return items


def test_whatsapp_team_round_trips():
    items = _check_whatsapp_thread("team.txt", "team")

    # Spot-check: the Lisbon flight-booking fact survives with its date intact
    # at minute granularity (the format has no seconds field).
    canon = _find_canonical("team", "locking in the lisbon flights")
    parsed = next(i for i in items if i.text == canon["text"])
    assert parsed.ts.replace(second=0) == _ts(canon["ts"]).replace(second=0)
    assert parsed.sender == "Jeffrey Williams"


def test_whatsapp_sarah_round_trips():
    items = _check_whatsapp_thread("sarah.txt", "sarah")

    canon = _find_canonical("sarah", "second week of june")
    parsed = next(i for i in items if i.text == canon["text"])
    assert parsed.ts.replace(second=0) == _ts(canon["ts"]).replace(second=0)


# ---------------------------------------------------------------------------
# Instagram DM round trip (mojibake recovery)
# ---------------------------------------------------------------------------


def test_instagram_round_trips_with_exact_mojibake_recovery():
    canonical = _canonical("kathryn")
    items = list(parse_instagram_thread(DEMO / "instagram" / "inbox" / "kathryn"))

    assert len(items) == len(canonical)
    assert all(i.source == SourceKind.instagram for i in items)

    # Instagram timestamps carry full millisecond precision in both the
    # canonical data and the real export format — no truncation anywhere in
    # this pipeline, so we can assert exact (sender, text, ts) triples.
    from corpusgen.personas import BY_CANONICAL

    def _to_millis(dt):
        return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)

    parsed_triples = Counter((i.sender, i.text, _to_millis(i.ts)) for i in items)
    canonical_triples = Counter(
        (
            BY_CANONICAL[m["speaker"]].instagram_handle,
            None if m.get("media") == "photo" else m["text"],
            _to_millis(_ts(m["ts"])),
        )
        for m in canonical
    )
    assert parsed_triples == canonical_triples

    # The accented sender name round-trips exactly through the mojibake fix.
    assert "Kathryn Petrović" in {i.sender for i in items}
    photo_items = [i for i in items if i.media_path is not None]
    assert len(photo_items) == sum(1 for m in canonical if m.get("media") == "photo")
    assert all(i.text is None for i in photo_items)


# ---------------------------------------------------------------------------
# SMS/MMS round trip
# ---------------------------------------------------------------------------


def test_sms_round_trips():
    canonical = _canonical("sms")
    items = list(parse_sms_xml(DEMO / "sms" / "backup.xml"))

    assert len(items) == len(canonical)

    mms_canonical = [m for m in canonical if m.get("media") == "mms_receipt"]
    mms_parsed = [i for i in items if i.source == SourceKind.mms]
    assert len(mms_parsed) == len(mms_canonical)
    assert mms_parsed[0].meta["mms_media"] is True
    assert mms_parsed[0].text == mms_canonical[0]["text"]

    # Every sent (Jeffrey-authored) message round-trips to sender "me".
    sent_canonical = sum(1 for m in canonical if m["speaker"] == "Jeffrey Williams")
    sent_parsed = sum(1 for i in items if i.meta.get("direction") == "sent")
    assert sent_parsed == sent_canonical

    # Received messages carry their contact's address for entity resolution.
    for i in items:
        if i.meta.get("direction") == "received":
            assert i.meta["address"] is not None


# ---------------------------------------------------------------------------
# Photos round trip — EXIF written by piexif, read back through Task 6's
# Pillow-native getexif()/get_ifd() reader (piexif is unmaintained and has
# documented round-trip issues against modern Pillow, so this is verified,
# never trusted).
# ---------------------------------------------------------------------------


def test_photos_round_trip_exif_and_sidecar():
    items = list(parse_photos(DEMO / "photos"))

    # 30 portraits + 3 receipts + 5 screenshots, all content-distinct.
    assert len(items) == 38
    assert all(i.source == SourceKind.photo for i in items)
    assert all(i.ts is not None for i in items)

    sidecar_items = [i for i in items if i.meta["taken_source"] == "sidecar"]
    exif_items = [i for i in items if i.meta["taken_source"] == "exif"]
    assert len(sidecar_items) == 15  # half of the 30 staged portraits
    assert len(exif_items) == 23  # the other 15 portraits + 3 receipts + 5 screenshots

    # Sidecar-derived GPS is exact (plain JSON float, no DMS round trip).
    sidecar_with_gps = [i for i in sidecar_items if i.meta["lat"] is not None]
    assert sidecar_with_gps
    # EXIF-derived GPS survives through the degrees/minutes/seconds
    # round trip within a small tolerance (rational-arithmetic rounding).
    exif_with_gps = [i for i in exif_items if i.meta["lat"] is not None]
    assert exif_with_gps
    for i in exif_with_gps:
        assert -90 <= i.meta["lat"] <= 90
        assert -180 <= i.meta["lon"] <= 180


def test_photos_content_dedupe_finds_no_accidental_duplicates():
    # Every staged composition is generated from a different (persona, seed)
    # pair, so — unlike Task 6's dedupe fixture — none of these 38 files are
    # byte-identical; the adapter's sha256 dedupe should therefore keep all
    # of them.
    items = list(parse_photos(DEMO / "photos"))
    media_paths = [i.media_path for i in items]
    assert len(media_paths) == len(set(media_paths))


# ---------------------------------------------------------------------------
# Cross-thread narrative consistency — the same real-world facts referenced
# from more than one thread must agree on their calendar date. This is a
# regression test for a real bug caught during authoring: the Lisbon trip's
# "leaving in three days" messages initially landed two months before the
# June 9-16 dates that were actually booked.
# ---------------------------------------------------------------------------


def test_lisbon_trip_dates_agree_across_team_and_sarah_threads():
    team_leaving = _find_canonical("team", "leaving for lisbon in three days")
    sarah_leaving = _find_canonical("sarah", "I haven't started and we leave in three days")
    assert _ts(team_leaving["ts"]).date() == _ts(sarah_leaving["ts"]).date()

    team_back = _find_canonical("team", "back from lisbon")
    sarah_back = _find_canonical("sarah", "still can't believe we actually did the whole itinerary")
    assert _ts(team_back["ts"]).date() == _ts(sarah_back["ts"]).date()


def test_northwind_start_date_agrees_across_threads():
    team_msg = _find_canonical("team", "may 5th. giving my two weeks")
    sms_msg = _find_canonical("sms", "start date is officially may 5th")
    assert "may 5th" in team_msg["text"] and "may 5th" in sms_msg["text"]

    sms_first_week = _find_canonical("sms", "first week at northwind done")
    started = datetime(2025, 5, 5, tzinfo=UTC).date()
    assert (_ts(sms_first_week["ts"]).date() - started).days < 14  # within ~a week of starting
