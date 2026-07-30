"""Pure functions that turn corpusgen/conversations.json's canonical message
records into the exact on-disk shapes Tasks 3-5's adapters parse. Each
renderer is the deliberate mirror image of its adapter:

  render_whatsapp   <-> locket.adapters.whatsapp.parse_whatsapp
  render_instagram  <-> locket.adapters.instagram.parse_instagram_thread
  render_sms_xml    <-> locket.adapters.sms_xml.parse_sms_xml

`corrupt_for_export` is the literal inverse of
`locket.adapters.instagram.fix_mojibake`'s primary path
(`s.encode("latin-1").decode("utf-8")`): it re-introduces the exact
byte-mangling Meta's own export tooling produces, so the committed demo
corpus is corrupted the same way a real Instagram export is, and the
round-trip test proves the adapter's fix actually recovers it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from lxml import etree

from corpusgen.personas import BY_CANONICAL, Persona

OWNER = "Jeffrey Williams"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# WhatsApp (Android dash convention, held fixed corpus-wide)
# ---------------------------------------------------------------------------


def render_whatsapp(msgs: list[dict[str, Any]]) -> str:
    """Real WhatsApp Android exports carry no seconds field, so any
    second-level precision in the canonical `ts` is deliberately lost here —
    that's authentic behavior, not a bug (see the round-trip test's
    minute-granularity comparison for whatsapp-sourced items)."""
    lines: list[str] = []
    for m in sorted(msgs, key=lambda x: x["ts"]):
        ts = _parse_ts(m["ts"])
        h12 = ts.hour % 12 or 12
        ampm = "AM" if ts.hour < 12 else "PM"
        header = f"{ts.month}/{ts.day}/{ts.year % 100:02d}, {h12}:{ts.minute:02d} {ampm} - "
        if m.get("system"):
            lines.append(header + (m["text"] or ""))
            continue
        text = m["text"] or ""
        text_lines = text.split("\n")
        lines.append(header + f"{m['speaker']}: {text_lines[0]}")
        lines.extend(text_lines[1:])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Instagram DM export (Meta's on-disk JSON shape)
# ---------------------------------------------------------------------------


def render_instagram(msgs: list[dict[str, Any]], other: Persona, thread_slug: str) -> dict[str, Any]:
    """Returns the CLEAN (uncorrupted) export dict — call `corrupt_for_export`
    on the result before writing, to reproduce Meta's mojibake bug."""
    messages: list[dict[str, Any]] = []
    photo_idx = 0
    for m in sorted(msgs, key=lambda x: x["ts"]):
        ts = _parse_ts(m["ts"])
        sender = BY_CANONICAL[m["speaker"]].instagram_handle
        entry: dict[str, Any] = {
            "sender_name": sender,
            "timestamp_ms": int(ts.timestamp() * 1000),
        }
        if m.get("media") == "photo":
            photo_idx += 1
            entry["photos"] = [
                {
                    "uri": f"messages/inbox/{thread_slug}/photos/{photo_idx}.jpg",
                    "creation_timestamp": int(ts.timestamp()),
                }
            ]
        else:
            entry["content"] = m["text"]
        messages.append(entry)
    return {
        "participants": [
            {"name": BY_CANONICAL[OWNER].instagram_handle},
            {"name": other.instagram_handle},
        ],
        "messages": messages,
        "title": other.instagram_handle,
        "is_still_participant": True,
        "thread_type": "Regular",
        "thread_path": f"inbox/{thread_slug}",
    }


def corrupt_for_export(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return obj.encode("utf-8").decode("latin-1")
        except UnicodeDecodeError:
            return obj
    if isinstance(obj, dict):
        return {k: corrupt_for_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [corrupt_for_export(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# SMS/MMS backup XML (SyncTech "SMS Backup & Restore" schema)
# ---------------------------------------------------------------------------

# A short, syntactically-valid (but not a real photo) base64 stub — the
# adapter only checks for the *presence* of a `data` attribute to flag
# `mms_media`, it never decodes it.
_FAKE_MMS_IMAGE_DATA = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkI"


def _nearest_counterparty(sorted_msgs: list[dict[str, Any]], idx: int) -> str:
    """Fallback ONLY for messages with no explicit `with` tag. Nearest-in-time
    inference is fragile once filler from a different contact can land inside
    what was meant to be an uninterrupted conversation with someone else —
    that misattributed a real reply ("that's huge man, congrats again", about
    Joshua's job) to Cory's phone number during authoring. conversations.json
    now tags every sms-thread message with an explicit `with` contact for
    exactly this reason; this function only exists as a defensive fallback."""
    speaker = sorted_msgs[idx]["speaker"]
    if speaker != OWNER:
        return speaker
    cur = _parse_ts(sorted_msgs[idx]["ts"])
    best_speaker, best_delta = None, None
    for m in sorted_msgs:
        if m["speaker"] == OWNER:
            continue
        delta = abs((_parse_ts(m["ts"]) - cur).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta, best_speaker = delta, m["speaker"]
    return best_speaker


def render_sms_xml(msgs: list[dict[str, Any]]) -> bytes:
    sorted_msgs = sorted(msgs, key=lambda x: x["ts"])
    root = etree.Element("smses")

    for idx, m in enumerate(sorted_msgs):
        ts = _parse_ts(m["ts"])
        millis = str(int(ts.timestamp() * 1000))
        readable = ts.strftime("%b %d, %Y %I:%M:%S %p")
        is_sent = m["speaker"] == OWNER
        contact_name = m.get("with") or _nearest_counterparty(sorted_msgs, idx)
        counterparty = BY_CANONICAL[contact_name]

        if m.get("media") == "mms_receipt":
            mms = etree.SubElement(root, "mms")
            mms.set("date", millis)
            mms.set("msg_box", "2" if is_sent else "1")
            mms.set("m_type", "128" if is_sent else "132")
            mms.set("sub_id", "1")
            mms.set("read", "1")
            mms.set("sub", "null")
            mms.set("readable_date", readable)
            parts = etree.SubElement(mms, "parts")
            text_part = etree.SubElement(parts, "part")
            for k, v in {
                "seq": "0", "ct": "text/plain", "name": "null", "chset": "106",
                "cd": "null", "fn": "null", "cid": "<text.txt>", "cl": "text.txt",
                "text": m["text"] or "",
            }.items():
                text_part.set(k, v)
            img_part = etree.SubElement(parts, "part")
            for k, v in {
                "seq": "1", "ct": "image/jpeg", "name": "receipt.jpg", "chset": "null",
                "cd": "null", "fn": "null", "cid": "<receipt.jpg>", "cl": "receipt.jpg",
                "data": _FAKE_MMS_IMAGE_DATA,
            }.items():
                img_part.set(k, v)
            addrs = etree.SubElement(mms, "addrs")
            from_addr = etree.SubElement(addrs, "addr")
            to_addr = etree.SubElement(addrs, "addr")
            if is_sent:
                from_addr.set("address", "insert-address-token")
                from_addr.set("type", "137")
                to_addr.set("address", counterparty.phone)
                to_addr.set("type", "151")
            else:
                from_addr.set("address", counterparty.phone)
                from_addr.set("type", "137")
                to_addr.set("address", "insert-address-token")
                to_addr.set("type", "151")
        else:
            sms = etree.SubElement(root, "sms")
            sms.set("protocol", "0")
            sms.set("address", counterparty.phone)
            sms.set("date", millis)
            sms.set("type", "2" if is_sent else "1")
            sms.set("subject", "null")
            sms.set("body", m["text"] or "")
            sms.set("toa", "null")
            sms.set("sc_toa", "null")
            sms.set("service_center", "null")
            sms.set("read", "1")
            sms.set("status", "-1")
            sms.set("locked", "0")
            sms.set("date_sent", millis)
            sms.set("sub_id", "1")
            sms.set("readable_date", readable)
            sms.set("contact_name", "null" if is_sent else counterparty.sms_contact_name)

    root.set("count", str(len(sorted_msgs)))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
