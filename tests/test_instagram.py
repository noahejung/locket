from pathlib import Path

from locket.adapters.instagram import fix_mojibake, parse_instagram_thread
from locket.models import SourceKind

FIX = Path(__file__).parent / "fixtures" / "instagram_thread"


def _corrupt(s: str) -> str:
    return s.encode("utf-8").decode("latin-1")


def test_fix_mojibake_round_trips_accented_name_and_emoji():
    assert fix_mojibake(_corrupt("Sarah Kovács")) == "Sarah Kovács"
    assert fix_mojibake(_corrupt("see you saturday 😊")) == "see you saturday 😊"


def test_fix_mojibake_leaves_ascii_untouched():
    assert fix_mojibake("Noah Jung") == "Noah Jung"
    assert fix_mojibake({"a": "plain text", "b": [1, "more plain text", None]}) == {
        "a": "plain text",
        "b": [1, "more plain text", None],
    }


def test_parse_merges_and_resorts_ascending_by_timestamp():
    items = list(parse_instagram_thread(FIX))
    timestamps = [i.ts for i in items]
    assert timestamps == sorted(timestamps)
    assert len(items) == 4


def test_photo_only_message_has_media_path_and_no_text():
    items = list(parse_instagram_thread(FIX))
    photo_items = [i for i in items if i.media_path is not None]
    assert len(photo_items) == 1
    assert photo_items[0].text is None
    assert photo_items[0].media_path == "messages/inbox/t/photos/1.jpg"


def test_senders_preserved_and_mojibake_fixed():
    items = list(parse_instagram_thread(FIX))
    senders = {i.sender for i in items}
    assert "Sarah Kovács" in senders
    assert "Noah Jung" in senders
    assert all(i.source == SourceKind.instagram for i in items)
