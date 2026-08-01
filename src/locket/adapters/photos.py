"""Photos adapter — Google Takeout sidecar JSON first, EXIF fallback second.

`register_heif_opener()` runs once here, at module import, before any
`Image.open` happens anywhere else in the process (mandatory per Pillow-HEIF's
own docs — opener registration is process-global and order-sensitive).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, tzinfo
from pathlib import Path

from PIL import ExifTags, Image
from pillow_heif import register_heif_opener

from locket.adapters.base import register
from locket.models import RawItem, SourceKind

register_heif_opener()

_MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

_DT_ORIGINAL = 36867  # Exif.DateTimeOriginal
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4


def find_sidecar(media_path: Path) -> Path | None:
    """Locate a Takeout-style JSON sidecar for `media_path`.

    Google Takeout truncates the ".supplemental-metadata.json" suffix on long
    filenames to keep the total path under its limit, so we try, in order:
    the exact suffix, a bare ".json", then a prefix glob that catches any
    truncated variant regardless of exact length.
    """
    exact = media_path.with_name(media_path.name + ".supplemental-metadata.json")
    if exact.exists():
        return exact
    bare = media_path.with_name(media_path.name + ".json")
    if bare.exists():
        return bare
    matches = sorted(media_path.parent.glob(f"{media_path.name}.supplement*.json"))
    return matches[0] if matches else None


def _dms_to_decimal(dms, ref: str | None) -> float | None:
    if not dms or ref is None:
        return None
    d, m, s = (float(x) for x in dms)
    val = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        val = -val
    return val


def _local_tz() -> tzinfo:
    """The system's local timezone, resolved at call time (not import time -- the UTC
    offset can vary by date under DST, so freezing it once at import would be wrong for
    some fraction of any real library)."""
    return datetime.now().astimezone().tzinfo


def _from_exif(path: Path, tz: tzinfo) -> tuple[datetime | None, float | None, float | None]:
    """EXIF DateTimeOriginal is the camera's LOCAL wall-clock reading at capture time --
    baseline EXIF carries no offset (the optional OffsetTimeOriginal tag isn't read here),
    so it's never UTC. Tagging it UTC directly (the pre-fix behavior) silently shifted every
    derived timestamp by the true UTC offset."""
    with Image.open(path) as img:
        exif = img.getexif()
        if not exif:
            return None, None, None
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)  # plain getexif() lacks DateTimeOriginal
        dt_str = exif_ifd.get(_DT_ORIGINAL)
        ts = None
        if dt_str:
            try:
                local = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").replace(tzinfo=tz)
                ts = local.astimezone(UTC)
            except ValueError:
                ts = None
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        lat = _dms_to_decimal(gps_ifd.get(_GPS_LAT), gps_ifd.get(_GPS_LAT_REF)) if gps_ifd else None
        lon = _dms_to_decimal(gps_ifd.get(_GPS_LON), gps_ifd.get(_GPS_LON_REF)) if gps_ifd else None
        return ts, lat, lon


def _from_sidecar(sidecar: Path) -> tuple[datetime | None, float | None, float | None]:
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    ts = None
    photo_taken = data.get("photoTakenTime") or {}
    # photoTakenTime, NEVER creationTime — creationTime is upload time, wrong
    # by years for old scans backfilled into Takeout.
    if photo_taken.get("timestamp"):
        ts = datetime.fromtimestamp(int(photo_taken["timestamp"]), tz=UTC)
    geo = data.get("geoData") or {}
    lat, lon = geo.get("latitude"), geo.get("longitude")
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        lat = lon = None  # (0.0, 0.0) is Google's "no location" sentinel, not real
    return ts, lat, lon


def parse_photos(
    root: Path, source_tz: tzinfo | None = None, *, warnings: list[str] | None = None
) -> Iterator[RawItem]:
    """`source_tz` is the timezone EXIF DateTimeOriginal timestamps were written in --
    defaults to the system's local timezone (`_local_tz()`) since that's the capturing
    device's own clock in the overwhelmingly common case. Sidecar (Google Takeout)
    timestamps are unaffected -- they're already epoch/UTC, not wall-clock text.

    `warnings`, if given, gets one message appended per file this pass couldn't fully
    process -- mirrors cli.py's vision-LLM-tail per-item catch-and-degrade pattern (one bad
    file logs and is skipped; the run continues). Two distinct failure shapes:
      - the image itself is unreadable (0-byte, truncated, not actually an image) -- nothing
        usable can be extracted at all, so the whole file is skipped (no RawItem yielded).
      - only its Takeout sidecar JSON is corrupt/truncated -- the photo itself is still
        real, so it's kept, falling through to EXIF (or ts=None, exactly as if no sidecar
        existed) rather than losing the photo over a metadata-file problem.
    """
    tz = source_tz if source_tz is not None else _local_tz()
    seen_hashes: set[str] = set()
    # Shallower paths win dedup ties — Takeout nests album *copies* one level
    # deeper (e.g. "Photos from 2025/x.jpg" vs "Album/x.jpg"), so the
    # top-level occurrence is the more likely canonical one.
    media_files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _MEDIA_SUFFIXES),
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )
    for path in media_files:
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            _warn(warnings, f"unreadable file {path.name}: {exc}")
            continue
        if content_hash in seen_hashes:
            continue  # Takeout duplicates the same photo across album exports
        seen_hashes.add(content_hash)

        ts: datetime | None = None
        lat: float | None = None
        lon: float | None = None
        taken_source: str | None = None

        sidecar = find_sidecar(path)
        if sidecar is not None:
            try:
                ts, lat, lon = _from_sidecar(sidecar)
            except (OSError, ValueError) as exc:
                _warn(warnings, f"corrupt sidecar {sidecar.name}: {exc}")
            else:
                if ts is not None:
                    taken_source = "sidecar"

        if ts is None:
            try:
                exif_ts, exif_lat, exif_lon = _from_exif(path, tz)
            except (OSError, ValueError) as exc:
                _warn(warnings, f"corrupt/unreadable image {path.name}: {exc}")
                continue  # nothing usable to extract from this file -- skip it, run continues
            if exif_ts is not None:
                ts = exif_ts
                taken_source = "exif"
            if lat is None and exif_lat is not None:
                lat, lon = exif_lat, exif_lon

        yield RawItem.make(
            source=SourceKind.photo,
            ts=ts,
            sender=None,
            text=None,
            media_path=str(path.relative_to(root)),
            meta={"lat": lat, "lon": lon, "taken_source": taken_source},
        )


def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(message)


register("dir:photos", parse_photos)
