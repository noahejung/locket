"""Generates demo_corpus/photos/: staged portraits (SFHQ synthetic faces
pasted onto plain backgrounds), rendered chat-screenshot and receipt PNGs,
with EXIF/GPS injected via piexif and Takeout-style JSON sidecars written
for half the portraits — the other half exercises Task 6's EXIF fallback.

SFHQ faces (corpusgen/assets/face_0N.jpg) are entirely AI-generated —
Synthetic Faces High Quality, MIT licensed, no real person depicted. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import piexif
from PIL import Image, ImageDraw, ImageFont

from corpusgen.personas import PERSONAS

ASSETS = Path(__file__).parent / "assets"

_BG_COLORS = [
    (222, 214, 199),  # warm beige
    (198, 214, 222),  # pale blue
    (214, 222, 198),  # sage
    (222, 198, 210),  # dusty rose
    (205, 205, 214),  # cool grey
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def staged_portrait(face_path: Path, bg_color: tuple[int, int, int], seed: int) -> Image.Image:
    canvas = Image.new("RGB", (480, 640), color=bg_color)
    face = Image.open(face_path).convert("RGB")
    face_size = 260 + (seed % 3) * 20
    face = face.resize((face_size, face_size))
    x = (canvas.width - face_size) // 2 + ((seed % 5) - 2) * 15
    y = 140 + (seed % 4) * 20
    canvas.paste(face, (x, y))
    return canvas


def render_receipt(store: str, items: list[tuple[str, float]], total: float) -> Image.Image:
    img = Image.new("RGB", (380, 90 + 24 * len(items) + 60), color=(250, 250, 245))
    draw = ImageDraw.Draw(img)
    f_title = _font(18)
    f_body = _font(14)
    draw.text((20, 16), store, font=f_title, fill=(20, 20, 20))
    draw.text((20, 42), "-" * 30, font=f_body, fill=(80, 80, 80))
    y = 62
    for name, price in items:
        draw.text((20, y), name, font=f_body, fill=(30, 30, 30))
        draw.text((280, y), f"${price:.2f}", font=f_body, fill=(30, 30, 30))
        y += 24
    draw.text((20, y + 6), "-" * 30, font=f_body, fill=(80, 80, 80))
    draw.text((20, y + 28), "TOTAL", font=f_title, fill=(20, 20, 20))
    draw.text((280, y + 28), f"${total:.2f}", font=f_title, fill=(20, 20, 20))
    return img


def render_chat_screenshot(contact_name: str, messages: list[tuple[bool, str]]) -> Image.Image:
    """`messages` is a list of (is_outgoing, text) tuples."""
    img = Image.new("RGB", (420, 720), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)
    f_header = _font(18)
    f_body = _font(14)
    draw.rectangle([(0, 0), (420, 50)], fill=(20, 110, 200))
    draw.text((16, 14), contact_name, font=f_header, fill=(255, 255, 255))
    y = 70
    for is_outgoing, text in messages:
        bubble_w = min(300, 40 + len(text) * 7)
        x0 = 420 - bubble_w - 16 if is_outgoing else 16
        color = (30, 140, 90) if is_outgoing else (225, 225, 230)
        text_color = (255, 255, 255) if is_outgoing else (20, 20, 20)
        draw.rounded_rectangle([(x0, y), (x0 + bubble_w, y + 32)], radius=10, fill=color)
        draw.text((x0 + 10, y + 8), text[:40], font=f_body, fill=text_color)
        y += 46
    return img


def inject_exif(path: Path, when: datetime, lat: float | None, lon: float | None) -> None:
    exif_dict: dict = {
        "0th": {},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: when.strftime("%Y:%m:%d %H:%M:%S").encode()},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    if lat is not None and lon is not None:
        def _to_dms(value: float) -> tuple:
            deg = int(abs(value))
            minutes_full = (abs(value) - deg) * 60
            minutes = int(minutes_full)
            seconds = round((minutes_full - minutes) * 60 * 100)
            return ((deg, 1), (minutes, 1), (seconds, 100))

        exif_dict["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _to_dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _to_dms(lon),
        }
    exif_bytes = piexif.dump(exif_dict)
    img = Image.open(path)
    img.save(path, exif=exif_bytes)


def write_sidecar(media_path: Path, when: datetime, lat: float | None, lon: float | None) -> None:
    data = {
        "title": media_path.name,
        "photoTakenTime": {
            "timestamp": str(int(when.timestamp())),
            "formatted": when.strftime("%b %d, %Y, %I:%M:%S %p UTC"),
        },
        "geoData": {
            "latitude": lat or 0.0,
            "longitude": lon or 0.0,
            "altitude": 0.0,
        },
    }
    sidecar = media_path.with_name(media_path.name + ".supplemental-metadata.json")
    sidecar.write_text(json.dumps(data), encoding="utf-8")


# A handful of real-world-ish (lat, lon) pairs the corpus's photos are
# scattered across, loosely matching the conversation's US-city setting.
_LOCATIONS = [
    (40.7128, -74.0060),  # NYC
    (37.7749, -122.4194),  # SF
    (41.8781, -87.6298),  # Chicago
    (47.6062, -122.3321),  # Seattle
    (38.9072, -77.0369),  # DC
]


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base = datetime(2025, 1, 10, 12, 0, 0)

    # --- staged portraits: 6 compositions per persona ---
    counter = 0
    for p_idx, persona in enumerate(PERSONAS):
        face_path = ASSETS / f"{persona.face_id}.jpg"
        for comp in range(6):
            counter += 1
            img = staged_portrait(face_path, _BG_COLORS[p_idx % len(_BG_COLORS)], seed=counter)
            fname = f"portrait_{persona.face_id}_{comp:02d}.jpg"
            path = out_dir / fname
            img.save(path, format="JPEG", quality=90)
            when = _shift_days(base, comp * 27 + p_idx * 5)
            lat, lon = _LOCATIONS[(p_idx + comp) % len(_LOCATIONS)]
            if counter % 2 == 0:
                # sidecar-driven half
                write_sidecar(path, when, lat, lon)
            else:
                # EXIF-fallback half
                inject_exif(path, when, lat, lon)
            written.append(path)

    # --- receipts ---
    receipts = [
        ("Bertucci's Trattoria", [("Carbonara", 22.0), ("Tiramisu", 9.0), ("Sparkling water", 4.0)], 35.0),
        ("Radiant Yoga Studio", [("Monthly membership", 89.0)], 89.0),
        ("Frame & Field Gallery Cafe", [("Espresso", 4.5), ("Croissant", 5.0)], 9.5),
    ]
    for i, (store, items, total) in enumerate(receipts, start=1):
        img = render_receipt(store, items, total)
        path = out_dir / f"receipt_{i:02d}.png"
        img.save(path, format="PNG")
        when = _shift_days(base, 40 + i * 15)
        inject_exif(path, when, None, None)
        written.append(path)

    # --- chat screenshots ---
    screenshots = [
        ("Sarah Mendes", [(False, "made a rough itinerary for lisbon"), (True, "sending it to the group"), (False, "day 3 sintra!!"), (True, "already excited")]),
        ("Cory Davis", [(True, "biscuit ate a sock again"), (False, "of course he did"), (True, "vet bill was $340")]),
        ("Kathryn Petrović", [(False, "frame & field emailed back"), (True, "AND"), (False, "small room, april 26th")]),
        ("Joshua Vega", [(False, "got the northwind offer"), (True, "letsgo!!"), (False, "start date may 5th")]),
        ("team", [(False, "movie night saturday"), (True, "in"), (False, "bringing popcorn seasoning")]),
    ]
    for i, (contact, msgs) in enumerate(screenshots, start=1):
        img = render_chat_screenshot(contact, msgs)
        path = out_dir / f"screenshot_{i:02d}.png"
        img.save(path, format="PNG")
        when = _shift_days(base, 60 + i * 11)
        inject_exif(path, when, None, None)
        written.append(path)

    return written


def _shift_days(base: datetime, days: int) -> datetime:
    return base + timedelta(days=days)
