"""Local vision pre-pass tests, run against demo_corpus/photos. Marked `vision` — downloads
SigLIP2 (~400MB), RapidOCR's English det+rec models, and InsightFace buffalo_l (~326MB) on
first run. Excluded from the default `-x -q` sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locket.vision.faces import cluster, embed_faces
from locket.vision.ocr import ocr_image
from locket.vision.tagger import LABELS, tag_images

pytestmark = pytest.mark.vision

PHOTOS = Path(__file__).parent.parent / "demo_corpus" / "photos"


def test_tagger_ranks_correct_label_top_for_each_image_type():
    """Absolute >0.5 is NOT asserted — measured against these real images, SigLIP2-base's
    sigmoid scores for a simple single-sentence template stay well under 0.5 even for the
    unambiguously-correct label (e.g. screenshot_01.png -> screenshot 0.034, next-highest
    document 0.003). Independent-yes/no sigmoid scoring is documented to not need to sum to
    1 like softmax; what's verified true and load-bearing for the tagger's actual job
    (bucketing) is that the correct label wins by a wide margin. See agent-report for the
    full measured table across all 10 labels x 3 images."""
    screenshot = PHOTOS / "screenshot_01.png"
    receipt = PHOTOS / "receipt_01.png"
    portrait = PHOTOS / "portrait_face_01_00.jpg"

    tags = tag_images([screenshot, receipt, portrait])

    assert set(tags[screenshot].keys()) == set(LABELS)
    assert max(tags[screenshot], key=tags[screenshot].get) == "screenshot"
    assert max(tags[receipt], key=tags[receipt].get) == "receipt"
    assert max(tags[portrait], key=tags[portrait].get) in ("people", "selfie")

    # Screenshot label should score far higher on the screenshot than on the portrait.
    assert tags[screenshot]["screenshot"] > tags[portrait]["screenshot"] * 5


def test_ocr_recovers_receipt_text():
    lines = ocr_image(PHOTOS / "receipt_01.png")
    assert lines
    joined = " ".join(line for line, _score in lines).lower()
    assert "bertucci" in joined or "trattoria" in joined
    assert all(0.0 <= score <= 1.0 for _line, score in lines)


def test_embed_faces_finds_a_face_per_portrait():
    paths = [PHOTOS / "portrait_face_01_00.jpg", PHOTOS / "portrait_face_02_00.jpg"]
    hits = embed_faces(paths)
    assert len(hits) >= 2
    found = {h.path for h in hits}
    assert set(paths) <= found
    for h in hits:
        assert len(h.embedding) == 512
        assert len(h.bbox) == 4


def test_cluster_groups_same_face_and_isolates_singleton():
    same_persona = [PHOTOS / f"portrait_face_01_0{i}.jpg" for i in range(3)]
    other_persona = [PHOTOS / f"portrait_face_02_0{i}.jpg" for i in range(3)]
    singleton = [PHOTOS / "portrait_face_05_00.jpg"]

    hits = embed_faces(same_persona + other_persona + singleton)
    clusters = cluster(hits)

    noise = clusters.get(-1, [])
    assert any(h.path == PHOTOS / "portrait_face_05_00.jpg" for h in noise)

    non_noise = {k: v for k, v in clusters.items() if k != -1}
    assert len(non_noise) >= 2
    for members in non_noise.values():
        # Each real cluster is pure — every member shares the same "portrait_face_NN" stem
        # prefix, i.e. the clusterer never mixes two different personas' photos together.
        prefixes = {p.path.stem.rsplit("_", 1)[0] for p in members}
        assert len(prefixes) == 1
