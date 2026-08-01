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


# ---------------------------------------------------------------------------
# Unrecognized type/msg_box codes (fix-wave-1 item 5b) -- must be skipped with a counted
# warning, never silently defaulted to sender="me" (misattributed authorship).
# ---------------------------------------------------------------------------


def test_unrecognized_sms_type_is_skipped_not_defaulted_to_me(tmp_path):
    # type="0" is SMS Backup & Restore's "all messages" QUERY code -- it should never appear
    # on a real per-message row, but if it (or any other stray value) does, the old code
    # silently attributed it to "me" (direction lookup missed -> `direction == "received"`
    # is False -> falls through to the "sent" branch's sender="me").
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<smses count="1">'
        '<sms protocol="0" address="+15550001111" date="1700000000000" type="0" '
        'body="weird type code" contact_name="Someone Else" />'
        "</smses>"
    )
    p = tmp_path / "weird_type.xml"
    p.write_text(xml, encoding="utf-8")
    warnings: list[str] = []

    items = list(parse_sms_xml(p, warnings=warnings))

    assert items == []  # not silently kept with a guessed sender
    assert len(warnings) == 1
    assert "0" in warnings[0]
    assert "type" in warnings[0]


def test_unrecognized_mms_msg_box_is_skipped_not_defaulted_to_me(tmp_path):
    # Same guess-if-not-recognized shape exists in the MMS path via msg_box -- symmetric fix.
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<smses count="1">'
        '<mms date="1700000000000" msg_box="99" sub_id="1">'
        "<parts><part seq=\"0\" ct=\"text/plain\" text=\"weird msg_box\" /></parts>"
        '<addrs><addr address="+15550009999" type="137" /></addrs>'
        "</mms>"
        "</smses>"
    )
    p = tmp_path / "weird_msgbox.xml"
    p.write_text(xml, encoding="utf-8")
    warnings: list[str] = []

    items = list(parse_sms_xml(p, warnings=warnings))

    assert items == []
    assert len(warnings) == 1
    assert "99" in warnings[0]


def test_well_formed_file_reports_zero_type_warnings():
    warnings: list[str] = []
    list(parse_sms_xml(FIX, warnings=warnings))
    assert warnings == []


# ---------------------------------------------------------------------------
# XXE regression (fix-wave-2 item 8) -- parse_sms_xml's etree.iterparse call passes
# resolve_entities=False specifically so a crafted export can't read local files via a
# DOCTYPE-declared external entity. Pins the CURRENT, live-verified behavior (2026-07-31):
# lxml refuses to resolve the reference and raises XMLSyntaxError rather than substituting
# file content. "Neutralized" here means the referenced file's content never reaches a
# RawItem or an exception message -- not that parsing degrades gracefully (a real `ingest`
# run over a file shaped like this would still surface the raised exception, same as any
# other corrupt-file error).
# ---------------------------------------------------------------------------


def test_xxe_external_entity_never_leaks_file_contents(tmp_path):
    from lxml import etree

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET SENTINEL VALUE", encoding="utf-8")
    xml = (
        '<?xml version="1.0"?>'
        f'<!DOCTYPE smses [<!ENTITY xxe SYSTEM "file:///{secret.as_posix()}">]>'
        '<smses><sms protocol="0" address="+15550001111" date="1700000000000" type="1" '
        'body="leaked: &xxe;" contact_name="Attacker" /></smses>'
    )
    p = tmp_path / "xxe.xml"
    p.write_text(xml, encoding="utf-8")

    try:
        items = list(parse_sms_xml(p))
    except etree.XMLSyntaxError as exc:
        # Neutralized by refusal -- the secret must not even leak into the error message.
        assert "TOP SECRET SENTINEL VALUE" not in str(exc)
    else:
        # If a future lxml version degrades gracefully instead of raising, the secret must
        # still never appear anywhere in the parsed output.
        assert all("TOP SECRET SENTINEL VALUE" not in (i.text or "") for i in items)
