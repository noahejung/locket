"""Ties the whole synthetic corpus together: reads conversations.json, renders
it through every platform's renderer, generates the standalone photos/, and
writes everything under demo_corpus/.

Run as: `python -m corpusgen.generate`
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from corpusgen import photos as photo_gen
from corpusgen.personas import BY_CANONICAL
from corpusgen.renderers import (
    corrupt_for_export,
    render_instagram,
    render_sms_xml,
    render_whatsapp,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_PATH = Path(__file__).with_name("conversations.json")
DEMO_CORPUS = REPO_ROOT / "demo_corpus"


def _load_conversations() -> list[dict]:
    return json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))


def _by_thread(msgs: list[dict], thread: str) -> list[dict]:
    return [m for m in msgs if m["thread"] == thread]


def generate(out_dir: Path = DEMO_CORPUS) -> None:
    msgs = _load_conversations()

    # --- WhatsApp: "team" group + "sarah" 1:1 ---
    wa_dir = out_dir / "whatsapp"
    wa_dir.mkdir(parents=True, exist_ok=True)
    (wa_dir / "team.txt").write_text(render_whatsapp(_by_thread(msgs, "team")), encoding="utf-8")
    (wa_dir / "sarah.txt").write_text(render_whatsapp(_by_thread(msgs, "sarah")), encoding="utf-8")

    # --- Instagram DM: "kathryn" 1:1 ---
    ig_dir = out_dir / "instagram" / "inbox" / "kathryn"
    ig_dir.mkdir(parents=True, exist_ok=True)
    ig_data = render_instagram(_by_thread(msgs, "kathryn"), BY_CANONICAL["Kathryn Petrović"], "kathryn")
    corrupted = corrupt_for_export(ig_data)
    (ig_dir / "message_1.json").write_text(json.dumps(corrupted, ensure_ascii=False), encoding="utf-8")

    # The 2 photo-only IG messages reference messages/inbox/kathryn/photos/{1,2}.jpg —
    # write matching (tiny, plain) placeholder images so the corpus has no
    # dangling media references.
    ig_photos_dir = ig_dir / "photos"
    ig_photos_dir.mkdir(parents=True, exist_ok=True)
    for i, color in enumerate(((40, 40, 45), (200, 195, 180)), start=1):
        Image.new("RGB", (300, 300), color=color).save(ig_photos_dir / f"{i}.jpg", format="JPEG")

    # --- SMS/MMS backup ---
    sms_dir = out_dir / "sms"
    sms_dir.mkdir(parents=True, exist_ok=True)
    (sms_dir / "backup.xml").write_bytes(render_sms_xml(_by_thread(msgs, "sms")))

    # --- Standalone photos/ (portraits, receipts, screenshots) ---
    photo_gen.generate(out_dir / "photos")


if __name__ == "__main__":
    generate()
    print(f"wrote demo_corpus to {DEMO_CORPUS}")
