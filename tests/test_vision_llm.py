"""Vision-LLM tail tests.

`select_for_vision` is a pure function over synthetic tag dicts -- no marks needed, runs in
the default suite (PLAN.md Task 13 Step 1 is explicit about this, even though the Files
section's blanket "(all @pytest.mark.vision)" note would suggest otherwise -- Step 1's
per-function instruction is the more specific one and is what's followed here).

`describe_photo` hits a real local Ollama server + qwen3-vl:8b -- marked `vision` and
self-skipping if the server isn't reachable or the model isn't pulled, per the dispatch's
explicit instruction.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from locket.vision.vision_llm import DEFAULT_MODEL, PhotoFacts, describe_photo, select_for_vision

PHOTOS = Path(__file__).parent.parent / "demo_corpus" / "photos"


def _ollama_model_available(model: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and model.split(":")[0] in result.stdout


# ---------------------------------------------------------------------------
# select_for_vision -- pure, synthetic tag dicts, no marks
# ---------------------------------------------------------------------------


def _tags(**labels: float) -> dict[str, float]:
    """Fill every SigLIP2 LABELS entry with a low floor, then override -- mirrors the real
    calibration (Build notes: correct label wins by margin, absolute values stay low)."""
    base = {
        "screenshot": 0.01,
        "receipt": 0.01,
        "document": 0.01,
        "food": 0.01,
        "people": 0.01,
        "selfie": 0.01,
        "landscape": 0.01,
        "pet": 0.01,
        "event": 0.01,
        "indoor scene": 0.01,
    }
    base.update(labels)
    return base


def test_select_for_vision_excludes_screenshot_receipt_document_by_relative_ranking():
    """Exclusion is argmax-based, not an absolute >0.5 cutoff -- these synthetic scores stay
    well under 0.5 (matching the real SigLIP2 calibration measured in Week 2), and the
    screenshot/receipt/document images must still be excluded because their own argmax
    label is in the excluded set."""
    screenshot = Path("shot.png")
    receipt = Path("receipt.png")
    document = Path("doc.png")
    people_photo = Path("people.jpg")

    tags = {
        screenshot: _tags(screenshot=0.04, document=0.01),
        receipt: _tags(receipt=0.03, document=0.02),
        document: _tags(document=0.05, screenshot=0.01),
        people_photo: _tags(people=0.06, event=0.02),
    }

    selected = select_for_vision(tags, cap=10)

    assert screenshot not in selected
    assert receipt not in selected
    assert document not in selected
    assert people_photo in selected


def test_select_for_vision_ranks_remainder_by_people_plus_event_and_caps():
    low = Path("low.jpg")
    mid = Path("mid.jpg")
    high = Path("high.jpg")
    tags = {
        low: _tags(people=0.02, event=0.01),
        mid: _tags(people=0.05, event=0.01),
        high: _tags(people=0.03, event=0.06),  # people+event = 0.09, highest combined
    }

    selected = select_for_vision(tags, cap=2)

    assert len(selected) == 2
    assert high in selected
    assert low not in selected  # lowest combined score, dropped by the cap


def test_select_for_vision_spreads_across_months_when_dates_given():
    """Without month-spreading, a single busy month could crowd out the cap entirely --
    `dates` (path -> ISO date string) is an additive optional keyword (the plan's stated
    signature carries no timestamp parameter) that round-robins the cap across months."""
    jan = [Path(f"jan_{i}.jpg") for i in range(4)]
    feb = [Path(f"feb_{i}.jpg") for i in range(1)]

    tags = {}
    dates = {}
    for i, p in enumerate(jan):
        tags[p] = _tags(people=0.09 - i * 0.01)  # jan photos rank highest by raw score
        dates[p] = f"2025-01-{10 + i:02d}"
    for p in feb:
        tags[p] = _tags(people=0.02)
        dates[p] = "2025-02-05"

    selected = select_for_vision(tags, cap=2, dates=dates)

    assert len(selected) == 2
    # Without spreading, both slots would go to January (it dominates raw score). With
    # month-spreading, February's lone photo must still get a slot.
    assert feb[0] in selected


def test_select_for_vision_never_exceeds_cap_even_with_few_candidates():
    p = Path("only.jpg")
    selected = select_for_vision({p: _tags(people=0.05)}, cap=300)
    assert selected == [p]


# ---------------------------------------------------------------------------
# describe_photo -- live Ollama, marked vision, self-skips if unavailable
# ---------------------------------------------------------------------------


@pytest.mark.vision
def test_describe_photo_returns_validated_photo_facts():
    if not _ollama_model_available(DEFAULT_MODEL):
        pytest.skip(f"ollama server/model {DEFAULT_MODEL!r} not available -- skipping live vision-LLM call")

    result = describe_photo(PHOTOS / "portrait_face_01_00.jpg", model=DEFAULT_MODEL)

    assert isinstance(result, PhotoFacts)
    assert result.scene
    assert result.people_count >= 0
