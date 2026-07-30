from pathlib import Path

from locket.adapters.sms_xml import parse_sms_xml
from locket.models import SourceKind

FIX = Path(__file__).parent / "fixtures" / "sms_backup.xml"


def test_drafts_are_excluded():
    items = list(parse_sms_xml(FIX))
    assert all("draft not sent yet" not in (i.text or "") for i in items)


def test_received_and_sent_sms_directions_and_ts():
    items = list(parse_sms_xml(FIX))
    sms_items = [i for i in items if i.source == SourceKind.sms]
    assert len(sms_items) == 2
    received = next(i for i in sms_items if i.meta["direction"] == "received")
    sent = next(i for i in sms_items if i.meta["direction"] == "sent")
    assert received.sender == "Sarah Kovacs"
    assert received.ts is not None and received.ts.tzinfo is not None
    assert received.text == "Hey, are we still on for tomorrow?"
    assert sent.text == "Yes! See you at 6"


def test_address_preserved_for_entity_resolution():
    items = list(parse_sms_xml(FIX))
    sms_items = [i for i in items if i.source == SourceKind.sms]
    assert all(i.meta["address"] == "+15551234567" for i in sms_items)


def test_mms_text_part_surfaces_and_smil_ignored():
    items = list(parse_sms_xml(FIX))
    mms_items = [i for i in items if i.source == SourceKind.mms]
    assert len(mms_items) == 1
    mms = mms_items[0]
    assert mms.text == "Where is Santa Clause?"
    assert "smil" not in (mms.text or "").lower()
    assert mms.meta["direction"] == "received"
    assert mms.meta["address"] == "+15551112222"


def test_null_attributes_never_crash_int_conversion(tmp_path):
    # A "date" attribute of literal "null" must degrade to ts=None, not raise.
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<smses count="1">'
        '<sms protocol="0" address="+15550001111" date="null" type="1" '
        'subject="null" body="whenever" contact_name="null" />'
        "</smses>"
    )
    p = tmp_path / "null_date.xml"
    p.write_text(xml, encoding="utf-8")
    items = list(parse_sms_xml(p))
    assert len(items) == 1
    assert items[0].ts is None
    assert items[0].sender is None  # contact_name was "null" too
