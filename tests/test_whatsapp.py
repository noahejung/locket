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
