"""Vision-LLM tail: curated selection + structured photo description via local Ollama.

Privacy rule (hard, PLAN.md Task 13 / design spec 3.2 + 6): real photos go to LOCAL Ollama
`qwen3-vl:8b` only (~11GB RAM; do NOT substitute qwen2.5vl:7b -- confirmed 15-17GB memory
bug, ollama#14312). Gemini free tier trains on and human-reviews inputs per its own ToS --
permitted only for demo_corpus/ images, opt-in. Claude API is a paid real-photo fallback
behind an explicit --cloud-ok flag. Neither the Gemini nor the Claude path is wired here --
this module is the local-Ollama default only; the CLI flags that gate the alternates land
with Task 19's `locket pipeline run`.

select_for_vision curation policy -- AMENDED from the plan's literal text per the Week 2
Task 10 finding recorded in PLAN.md's "Build notes" section: SigLIP2-base sigmoid scores
for this model + prompt template never approach 0.5 even for an unambiguously-correct
label (measured: an unambiguous screenshot scored ~0.03), so bucket routing here uses
relative (argmax) ranking, never an absolute cutoff:
  1. A photo whose argmax tag is screenshot/receipt/document is excluded from the
     vision-LLM tail -- those route to OCR-augmented extraction instead (never to an LLM).
  2. The remainder is ranked by (people + event) score, descending.
  3. `dates` (path -> ISO date string) is an ADDITIVE optional keyword, not in the plan's
     literal signature (which carries no timestamp parameter at all -- "spread across
     months" is unimplementable without one). When given, selection round-robins across
     calendar months before taking each month's top scorers, so one busy month can't crowd
     out the whole cap. Without `dates`, selection is a straight top-`cap` by score.

CPU latency measured live against demo_corpus (see evals/BASELINE.md): ~140s/image on this
CPU-only box -- well above the plan's own explicitly-flagged-as-unverified 10-40s estimate.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ollama import chat
from pydantic import BaseModel

from locket.vision.tagger import LABELS

DEFAULT_MODEL = "qwen3-vl:8b"

_OCR_ROUTED_LABELS = {"screenshot", "receipt", "document"}


class PhotoFacts(BaseModel):
    scene: str
    activity: str | None = None
    people_count: int
    notable: list[str] = []


def _argmax_label(tags: dict[str, float]) -> str:
    return max(tags, key=tags.get)


def select_for_vision(
    tags: dict[Path, dict[str, float]],
    *,
    cap: int,
    dates: dict[Path, str] | None = None,
) -> list[Path]:
    candidates = [p for p, t in tags.items() if _argmax_label(t) not in _OCR_ROUTED_LABELS]

    def combined_score(p: Path) -> float:
        t = tags[p]
        return t.get("people", 0.0) + t.get("event", 0.0)

    if not dates:
        ranked = sorted(candidates, key=combined_score, reverse=True)
        return ranked[:cap]

    by_month: dict[str, list[Path]] = defaultdict(list)
    for p in candidates:
        month = dates.get(p, "")[:7]  # "YYYY-MM"; unknown dates group under ""
        by_month[month].append(p)
    for month_paths in by_month.values():
        month_paths.sort(key=combined_score, reverse=True)

    queues = {month: list(paths) for month, paths in by_month.items()}
    months = sorted(queues)
    selected: list[Path] = []
    while len(selected) < cap and any(queues.values()):
        for month in months:
            if len(selected) >= cap:
                break
            if queues[month]:
                selected.append(queues[month].pop(0))

    return selected


def describe_photo(path: Path, *, model: str = DEFAULT_MODEL) -> PhotoFacts:
    resp = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Describe this photo for a personal-memory index. JSON only.",
                "images": [str(path)],
            }
        ],
        format=PhotoFacts.model_json_schema(),  # schema-constrained decoding
        options={"temperature": 0},
    )
    return PhotoFacts.model_validate_json(resp.message.content)


__all__ = ["DEFAULT_MODEL", "LABELS", "PhotoFacts", "describe_photo", "select_for_vision"]
