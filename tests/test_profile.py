"""Profile synthesis tests.

Per the dispatch's stated premise: profile synthesis runs keylessly only in its
deterministic scaffold (grouping facts into sections, sorting the timeline, persisting a
versioned row). The per-section haiku rendering is key-gated -- these tests prove the
scaffold + citation mechanics with a stubbed renderer (mirrors resolution.py/graph.py's
`model=` seam); one `@pytest.mark.llm` test exercises the real haiku call and self-skips
without ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

import pytest

from locket.embeddings import get_backend
from locket.models import Fact, FactKind
from locket.profile import SECTION_ORDER, SectionRendering, synthesize
from locket.store import Store

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")

pytestmark = pytest.mark.db

_CITE_RE = re.compile(r"\[fact:([0-9a-f]{8})\]")


@pytest.fixture
def store():
    s = Store(DB_URL)
    with s._conn.cursor() as cur:
        cur.execute(
            "TRUNCATE raw_items, facts, entities, fact_history, merge_proposals, profiles, "
            "extracted_windows RESTART IDENTITY CASCADE"
        )
    s._conn.commit()
    yield s
    s._conn.close()


def _make_seeded_store(store: Store) -> dict[str, str]:
    """Seeds one fact per FactKind, returns {kind: fact_id}."""
    backend = get_backend()
    ids: dict[str, str] = {}
    seeds = [
        (FactKind.person, "Noah Jung is a software engineer", None),
        (FactKind.place, "Noah went on exchange to Budapest", "2024-08-01/2024-12-15"),
        (FactKind.relationship, "Sarah Kovacs is Noah's dance teammate", None),
        (FactKind.event, "The group had Saturday dinner at Bertucci's", "2025-01-18"),
        (FactKind.habit, "Noah runs every morning", None),
        (FactKind.preference, "Noah prefers tea over coffee", None),
    ]
    for kind, statement, happened_at in seeds:
        fact = Fact(kind=kind, statement=statement, confidence=0.9, happened_at=happened_at, provenance=["r1"])
        fact_id = store.add_fact(fact, backend.embed_docs([statement])[0])
        ids[str(kind)] = fact_id
    return ids


class _EchoRenderModel:
    """Returns each fact's own statement, unmodified, as its "rendered" sentence -- proves
    the scaffold zips facts back to citations correctly without depending on prose
    quality/content, which is exactly what a stubbed renderer should prove per the
    dispatch's stated premise."""

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SectionRendering:
        self.prompts.append(prompt)
        # The prompt embeds a numbered list "1. <statement>\n2. <statement>..." -- recover
        # the statements in order so the echoed sentence count always matches input count.
        lines = [
            line.split(". ", 1)[1]
            for line in prompt.splitlines()
            if line[:1].isdigit() and ". " in line
        ]
        return SectionRendering(sentences=lines)


def test_synthesize_produces_all_five_sections(store):
    _make_seeded_store(store)
    model = _EchoRenderModel()

    body = synthesize(store, model=model)

    for section in SECTION_ORDER:
        assert f"## {section}" in body


def test_every_citation_id_exists_in_facts(store):
    fact_ids = _make_seeded_store(store)
    model = _EchoRenderModel()

    body = synthesize(store, model=model)

    cited_id8s = set(_CITE_RE.findall(body))
    assert cited_id8s  # at least one citation exists
    known_id8s = {fid[:8] for fid in fact_ids.values()}
    assert cited_id8s <= known_id8s


def test_facts_land_in_their_mapped_section(store):
    fact_ids = _make_seeded_store(store)
    model = _EchoRenderModel()

    body = synthesize(store, model=model)

    relationship_id8 = fact_ids[str(FactKind.relationship)][:8]
    habit_id8 = fact_ids[str(FactKind.habit)][:8]

    people_section = body.split("## People", 1)[1].split("## ", 1)[0]
    assert relationship_id8 in people_section
    assert habit_id8 not in people_section

    habits_section = body.split("## Habits", 1)[1].split("## ", 1)[0]
    assert habit_id8 in habits_section


def test_timeline_section_sorted_chronologically(store):
    backend = get_backend()
    later = Fact(kind=FactKind.event, statement="Later event", confidence=0.9, happened_at="2025-06-01", provenance=["r1"])
    earlier = Fact(kind=FactKind.event, statement="Earlier event", confidence=0.9, happened_at="2024-01-01", provenance=["r2"])
    store.add_fact(later, backend.embed_docs([later.statement])[0])
    store.add_fact(earlier, backend.embed_docs([earlier.statement])[0])
    model = _EchoRenderModel()

    body = synthesize(store, model=model)

    timeline_section = body.split("## Timeline", 1)[1].split("## ", 1)[0]
    assert timeline_section.index("Earlier event") < timeline_section.index("Later event")


def test_second_build_with_no_new_facts_is_a_noop(store):
    _make_seeded_store(store)
    model = _EchoRenderModel()

    first_body = synthesize(store, model=model)
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM profiles")
        assert cur.fetchone()[0] == 1

    second_body = synthesize(store, model=model)
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM profiles")
        assert cur.fetchone()[0] == 1  # no duplicate version row
    assert second_body == first_body


def test_third_build_after_new_fact_creates_a_new_version(store):
    _make_seeded_store(store)
    model = _EchoRenderModel()
    synthesize(store, model=model)

    backend = get_backend()
    extra = Fact(kind=FactKind.habit, statement="Noah does yoga on Tuesdays", confidence=0.8, provenance=["r9"])
    store.add_fact(extra, backend.embed_docs([extra.statement])[0])

    synthesize(store, model=model)
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM profiles")
        assert cur.fetchone()[0] == 2


def test_empty_store_still_produces_all_sections_with_placeholder_text(store):
    model = _EchoRenderModel()
    body = synthesize(store, model=model)
    for section in SECTION_ORDER:
        assert f"## {section}" in body


def test_expired_fact_is_excluded_from_the_synthesized_profile(store):
    """Bi-temporal validity wired into profile synthesis (fix-wave-1 item 9): an expired
    fact must not appear in the rendered profile -- no include_expired escape hatch here
    per the dispatch (profile synthesis is always as-of now)."""
    fact_ids = _make_seeded_store(store)
    habit_id = fact_ids[str(FactKind.habit)]
    store.update_fact(habit_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    model = _EchoRenderModel()

    body = synthesize(store, model=model)

    habits_section = body.split("## Habits", 1)[1].split("## ", 1)[0]
    assert habit_id[:8] not in habits_section
    assert "_Nothing extracted for this section yet._" in habits_section


# ---------------------------------------------------------------------------
# Live smoke: real haiku rendering. Blocked on ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------


@pytest.mark.llm
def test_live_synthesize_renders_real_prose(store):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live profile synthesis")

    _make_seeded_store(store)
    body = synthesize(store)

    assert "## Identity" in body
    assert _CITE_RE.search(body) is not None
