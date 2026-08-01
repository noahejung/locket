from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from locket.adapters.photos import find_sidecar, parse_photos
from locket.models import SourceKind

FIX = Path(__file__).parent / "fixtures" / "photos"


def _by_stem(items, stem):
    return next(i for i in items if Path(i.media_path).stem == stem)


def test_sidecar_preferred_over_exif_when_both_exist(tmp_path):
    # IMG_b has sidecar only in the base fixture; here we also drop EXIF onto
    # the same file to prove sidecar wins even when EXIF is present too.
    import piexif
    from PIL import Image

    d = tmp_path / "photos"
    d.mkdir()
    exif_dict = {
        "0th": {},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"1999:01:01 00:00:00"},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    Image.new("RGB", (10, 10)).save(d / "both.jpg", format="JPEG", exif=piexif.dump(exif_dict))
    (d / "both.jpg.supplemental-metadata.json").write_text(
        '{"photoTakenTime": {"timestamp": "1736938800"}, "geoData": {"latitude": 1.0, "longitude": 2.0}}',
        encoding="utf-8",
    )
    items = list(parse_photos(d))
    assert len(items) == 1
    item = items[0]
    assert item.meta["taken_source"] == "sidecar"
    assert item.ts == datetime.fromtimestamp(1736938800, tz=UTC)


def test_truncated_sidecar_found_by_prefix_glob():
    sidecar = find_sidecar(FIX / "IMG_c.jpg")
    assert sidecar is not None
    assert sidecar.name == "IMG_c.jpg.supplemental-me.json"


def test_zero_geodata_yields_no_coordinate():
    items = list(parse_photos(FIX))
    d = _by_stem(items, "IMG_d")
    assert d.meta["lat"] is None
    assert d.meta["lon"] is None


_EST = timezone(timedelta(hours=-5))


def test_exif_only_photo_gets_ts_and_decimal_gps():
    # Fixture's EXIF DateTimeOriginal is "2025:01:15 23:30:00" local wall-clock -- passed
    # as EST (UTC-5) explicitly (never the machine's real tz), it converts to 04:30 UTC on
    # the FOLLOWING day. A pre-fix "tag it UTC directly" bug would instead yield
    # 2025-01-15 23:30 UTC -- wrong hour AND wrong date, so this can't accidentally pass.
    items = list(parse_photos(FIX, source_tz=_EST))
    a = _by_stem(items, "IMG_a")
    assert a.meta["taken_source"] == "exif"
    assert a.ts == datetime(2025, 1, 16, 4, 30, 0, tzinfo=UTC)
    assert a.meta["lat"] is not None and a.meta["lon"] is not None
    assert 37 < a.meta["lat"] < 38
    assert -123 < a.meta["lon"] < -122


def test_no_source_tz_given_defaults_to_the_resolved_local_timezone(monkeypatch):
    # Don't depend on the machine's real tz -- monkeypatch the module's own local-tz
    # resolution to a fixed, known-non-UTC offset and prove the default path picks it up.
    monkeypatch.setattr("locket.adapters.photos._local_tz", lambda: _EST)
    items = list(parse_photos(FIX))
    a = _by_stem(items, "IMG_a")
    assert a.ts == datetime(2025, 1, 16, 4, 30, 0, tzinfo=UTC)


def test_source_tz_utc_is_a_no_op():
    items = list(parse_photos(FIX, source_tz=UTC))
    a = _by_stem(items, "IMG_a")
    assert a.ts == datetime(2025, 1, 15, 23, 30, 0, tzinfo=UTC)


def test_duplicate_content_dedupes_to_one_raw_item():
    items = list(parse_photos(FIX))
    # IMG_a.jpg and Album1/IMG_a_dup.jpg are byte-identical.
    a_like = [i for i in items if i.media_path in ("IMG_a.jpg", str(Path("Album1") / "IMG_a_dup.jpg"))]
    assert len(a_like) == 1


def test_all_items_are_photo_kind():
    items = list(parse_photos(FIX))
    assert all(i.source == SourceKind.photo for i in items)
