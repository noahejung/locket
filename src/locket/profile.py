"""Profile synthesis: renders the fact store into the "profile of you" living document.

Deterministic scaffold (which facts go in which section, timeline ordering, whether a new
version is worth writing) plus one haiku call per section to turn each fact's raw statement
into a polished sentence. Citations are injected MECHANICALLY after the model call -- the
code zips the model's rendered sentences back to their source fact ids by position and
appends `[fact:<id8>]` itself, rather than trusting the model to keep id markers intact
through a rewrite (per PLAN.md Task 18 Step 2-3's explicit instruction).

Section <-> FactKind mapping is deterministic and total (every FactKind lands in exactly one
of the five sections):
  Identity     <- person, place    (who the user is, general biographical/location facts)
  People       <- relationship     (the user's relationships to other people)
  Timeline     <- event            (chronological, sorted by happened_at)
  Habits       <- habit
  Preferences  <- preference
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel, Field

from locket.llm import get_default_chat_model
from locket.models import FactKind
from locket.store import FactRow, Store

SECTION_ORDER = ["Identity", "People", "Timeline", "Habits", "Preferences"]

SECTION_KIND_MAP: dict[str, list[FactKind]] = {
    "Identity": [FactKind.person, FactKind.place],
    "People": [FactKind.relationship],
    "Timeline": [FactKind.event],
    "Habits": [FactKind.habit],
    "Preferences": [FactKind.preference],
}

_NO_FACTS_PLACEHOLDER = "_Nothing extracted for this section yet._"


class SectionRendering(BaseModel):
    """One rendered sentence per input fact, same order, same count -- the code, not the
    model, is responsible for attaching each sentence back to its source fact's citation."""

    sentences: list[str] = Field(description="One polished sentence per input fact, in order")


def _build_section_prompt(name: str, facts: list[FactRow]) -> str:
    numbered = "\n".join(f"{i + 1}. {f.statement}" for i, f in enumerate(facts))
    return (
        f"Rewrite each of the following {len(facts)} facts about the user's life (profile "
        f"section: {name!r}) into one polished, natural sentence each. Return exactly "
        f"{len(facts)} sentences, in the same order, one per input fact -- do not merge, "
        "drop, reorder, or add any.\n\n"
        f"Facts:\n{numbered}"
    )


@cache
def _default_render_model() -> Any:
    return get_default_chat_model("profile_render").with_structured_output(SectionRendering)


def _render_sentences(name: str, facts: list[FactRow], *, model: Any | None) -> list[str]:
    active = model if model is not None else _default_render_model()
    result = active.invoke(_build_section_prompt(name, facts))
    return result.sentences


def _render_section(name: str, facts: list[FactRow], *, model: Any | None) -> str:
    lines = [f"## {name}", ""]
    if not facts:
        lines.append(_NO_FACTS_PLACEHOLDER)
        return "\n".join(lines) + "\n"

    rendered = _render_sentences(name, facts, model=model)
    for i, fact in enumerate(facts):
        # Defensive: never trust the model to preserve count -- a fact with no matching
        # rendered sentence falls back to its own raw statement rather than being dropped.
        sentence = rendered[i].strip() if i < len(rendered) and rendered[i].strip() else fact.statement
        lines.append(f"- {sentence.rstrip('.')} [fact:{fact.id[:8]}]")
    return "\n".join(lines) + "\n"


def _group_by_section(facts: list[FactRow]) -> dict[str, list[FactRow]]:
    by_kind: dict[str, list[FactRow]] = {}
    for fact in facts:
        by_kind.setdefault(fact.kind, []).append(fact)

    grouped: dict[str, list[FactRow]] = {}
    for section, kinds in SECTION_KIND_MAP.items():
        section_facts = [f for kind in kinds for f in by_kind.get(str(kind), [])]
        if section == "Timeline":
            section_facts = sorted(section_facts, key=lambda f: f.happened_at or "")
        grouped[section] = section_facts
    return grouped


def synthesize(store: Store, *, model: Any | None = None) -> str:
    """Render the fact store into a five-section markdown profile and persist a new
    versioned row -- unless the fact count is unchanged since the latest saved profile, in
    which case this is a no-op and the existing body is returned as-is (no duplicate
    version row, no wasted model calls)."""
    all_facts = store.list_facts(limit=100_000)
    fact_count = len(all_facts)

    latest = store.get_latest_profile()
    if latest is not None and latest.fact_count == fact_count:
        return latest.body

    grouped = _group_by_section(all_facts)
    sections = [_render_section(name, grouped[name], model=model) for name in SECTION_ORDER]
    body = "# Life Profile\n\n" + "\n".join(sections)

    store.save_profile(body, fact_count)
    return body


__all__ = [
    "SECTION_KIND_MAP",
    "SECTION_ORDER",
    "SectionRendering",
    "synthesize",
]
