"""Regenerates tests/fixtures/photos/ — tiny synthetic JPEGs exercising every
sidecar/EXIF branch the photos adapter has to handle.

Rerunnable: `python tests/fixtures/gen_photos_fixture.py`

  IMG_a.jpg   EXIF DateTimeOriginal + GPS IFD rationals, no sidecar.
  IMG_b.jpg   no EXIF; full-name sidecar with photoTakenTime + real geoData.
  IMG_c.jpg   no EXIF; ONLY a truncated-name sidecar (Takeout path-length
              truncation), found via find_sidecar's glob fallback.
  IMG_d.jpg   no EXIF; sidecar geoData is (0.0, 0.0) — the "no location" trap.
  Album1/IMG_a_dup.jpg   byte-identical copy of IMG_a.jpg — proves content dedupe.
"""

import json
import pathlib
import shutil

import piexif
from PIL import Image

OUT = pathlib.Path(__file__).with_name("photos")
OUT.mkdir(exist_ok=True)
(OUT / "Album1").mkdir(exist_ok=True)


def _plain_jpeg(path: pathlib.Path, color: str) -> None:
    Image.new("RGB", (50, 50), color=color).save(path, format="JPEG")


def _exif_jpeg(path: pathlib.Path, color: str) -> None:
    exif_dict = {
        "0th": {},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2025:01:15 10:30:00"},
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: ((37, 1), (46, 1), (1, 2)),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: ((122, 1), (25, 1), (3, 4)),
        },
        "1st": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    Image.new("RGB", (50, 50), color=color).save(path, format="JPEG", exif=exif_bytes)


# (a) EXIF-only.
_exif_jpeg(OUT / "IMG_a.jpg", "red")

# (b) sidecar-only, full-name suffix, real geoData.
_plain_jpeg(OUT / "IMG_b.jpg", "green")
(OUT / "IMG_b.jpg.supplemental-metadata.json").write_text(
    json.dumps(
        {
            "title": "IMG_b.jpg",
            "photoTakenTime": {"timestamp": "1736938800", "formatted": "Jan 15, 2025, 11:00:00 AM UTC"},
            "creationTime": {"timestamp": "1800000000", "formatted": "wrong — this is upload time"},
            "geoData": {"latitude": 40.7128, "longitude": -74.0060, "altitude": 10.0},
        }
    ),
    encoding="utf-8",
)

# (c) sidecar-only, TRUNCATED suffix (Takeout path-length truncation).
_plain_jpeg(OUT / "IMG_c.jpg", "blue")
(OUT / "IMG_c.jpg.supplemental-me.json").write_text(
    json.dumps(
        {
            "title": "IMG_c.jpg",
            "photoTakenTime": {"timestamp": "1736942400", "formatted": "Jan 15, 2025, 12:00:00 PM UTC"},
            "geoData": {"latitude": 51.5074, "longitude": -0.1278, "altitude": 5.0},
        }
    ),
    encoding="utf-8",
)

# (d) sidecar geoData all-zero — the "no location" sentinel trap.
_plain_jpeg(OUT / "IMG_d.jpg", "yellow")
(OUT / "IMG_d.jpg.supplemental-metadata.json").write_text(
    json.dumps(
        {
            "title": "IMG_d.jpg",
            "photoTakenTime": {"timestamp": "1736946000", "formatted": "Jan 15, 2025, 01:00:00 PM UTC"},
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
        }
    ),
    encoding="utf-8",
)

# Duplicate content across a second "album" — proves sha256 content dedupe.
shutil.copyfile(OUT / "IMG_a.jpg", OUT / "Album1" / "IMG_a_dup.jpg")

if __name__ == "__main__":
    print(f"wrote fixtures to {OUT}")
