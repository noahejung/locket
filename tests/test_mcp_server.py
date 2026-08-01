"""MCP server tool tests -- in-process against a real dockerized Store (seeded directly via
Store.add_fact/upsert_entity, no LLM extraction needed) via `mcp.call_tool(name, args)`, the
real MCP 2.0.0 tool-invocation surface (verified live: the return is a CallToolResult whose
`.structured_content["result"]` holds the tool function's return value).

Embeddings use the real local arctic-embed-s backend (already downloaded, offline, no
network) -- these tests need genuine semantic proximity for search_memories to rank
sensibly. LLM-backed seams (entity-resolution tier-3 escalation, answer_question's
decompose/synthesize calls) are stubbed; no ANTHROPIC_API_KEY needed.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest

from locket.embeddings import get_backend
from locket.mcp_server import build_server
from locket.models import Fact, FactKind
from locket.store import Store

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")

pytestmark = pytest.mark.db


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


class _FakeDecomposeModel:
    """Stand-in for the haiku decompose call -- returns a fixed SubQueries object regardless
    of prompt, mirroring tests/test_resolution.py's `_StubModel` seam pattern."""

    def __init__(self, queries: list[str]):
        from locket.mcp_server import SubQueries

        self._result = SubQueries(queries=queries)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return self._result


class _FakeSynthesizeModel:
    """Stand-in for the haiku synthesis call -- returns a canned answer string that cites
    whatever fact id the test wires in, mirroring evals/rag_eval.py's `_FakeAnswerModel`."""

    def __init__(self, answer_text: str):
        self._answer_text = answer_text
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        from types import SimpleNamespace

        self.calls.append(prompt)
        return SimpleNamespace(content=self._answer_text)


def _seed_dance_fact(store: Store) -> tuple[str, str]:
    """Sarah Kovacs is a dance teammate -- one entity, one fact, real embedding."""
    backend = get_backend()
    entity_id = store.upsert_entity("Sarah Kovacs", "person", backend.embed_docs(["Sarah Kovacs"])[0])
    fact = Fact(
        kind=FactKind.relationship,
        statement="Sarah Kovacs is Noah's dance teammate",
        confidence=0.9,
        subjects=["Sarah Kovacs"],
        happened_at="2025-01-15",
        entity_ids=[entity_id],
        provenance=["r1"],
    )
    embedding = backend.embed_docs([fact.statement])[0]
    fact_id = store.add_fact(fact, embedding)
    return entity_id, fact_id


def _call(mcp, name: str, args: dict | None = None):
    """Call a tool through the real MCP wire path and unwrap its return value.

    Live-verified (2026-07-30) against mcp==2.0.0: tools annotated `-> list[...]` or
    `-> str` get a structured_content of `{"result": <value>}`, but a bare `-> dict`
    annotation gets `structured_content=None` (no output_schema could be built) -- the
    value is still there, JSON-encoded in `.content[0].text`. This helper handles both.
    """
    result = asyncio.run(mcp.call_tool(name, args or {}))
    assert not result.is_error, f"{name} call errored: {result.content}"
    if result.structured_content is not None:
        return result.structured_content["result"]
    import json

    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------


def test_search_memories_returns_provenance_bearing_dicts(store):
    entity_id, fact_id = _seed_dance_fact(store)
    mcp = build_server(store)

    results = _call(mcp, "search_memories", {"query": "who is on the dance team"})

    assert len(results) == 1
    row = results[0]
    assert row["statement"] == "Sarah Kovacs is Noah's dance teammate"
    assert row["kind"] == "relationship"
    assert row["sources"] == ["r1"]
    assert row["happened_at"] == "2025-01-15"


def test_search_memories_people_filter_narrows_results(store):
    backend = get_backend()
    sarah_id, _ = _seed_dance_fact(store)
    john_id = store.upsert_entity("John Doe", "person", backend.embed_docs(["John Doe"])[0])
    other_fact = Fact(
        kind=FactKind.habit,
        statement="John Doe runs every morning",
        confidence=0.8,
        subjects=["John Doe"],
        entity_ids=[john_id],
        provenance=["r2"],
    )
    store.add_fact(other_fact, backend.embed_docs([other_fact.statement])[0])
    mcp = build_server(store)

    filtered = _call(mcp, "search_memories", {"query": "life facts", "people": ["Sarah Kovacs"], "limit": 20})

    assert len(filtered) == 1
    assert filtered[0]["statement"] == "Sarah Kovacs is Noah's dance teammate"
    assert sarah_id  # sanity: fixture actually created the entity


def test_search_memories_people_filter_unresolvable_name_creates_no_phantom_entity(store):
    """search_memories is a read-only query tool. A mistyped/unknown name in `people` must
    not fall through to resolution.resolve()'s ingestion-path fallback
    (`_resolve_one` -> `store.upsert_entity(...)` on zero tier-1 candidates) -- that would
    silently insert a phantom entity as a side effect of a search. Unknown names should
    just contribute no entity ids, exactly like `_lookup_person`'s non-mutating contract
    for get_person."""
    _seed_dance_fact(store)
    mcp = build_server(store)
    entities_before = len(store.list_entities())

    results = _call(
        mcp,
        "search_memories",
        {"query": "life facts", "people": ["Zzxqvbnm Ploopers"], "limit": 20},
    )

    entities_after = len(store.list_entities())
    assert entities_after == entities_before  # no phantom entity created
    assert results == []  # unknown name contributes no entity ids -> nothing matches


def test_search_memories_people_filter_mixes_resolvable_and_unresolvable_names(store):
    """A known name alongside an unknown one still filters correctly by the known name --
    the unknown name is simply dropped, not treated as a hard failure."""
    _seed_dance_fact(store)
    mcp = build_server(store)
    entities_before = len(store.list_entities())

    results = _call(
        mcp,
        "search_memories",
        {"query": "life facts", "people": ["Sarah Kovacs", "Zzxqvbnm Ploopers"], "limit": 20},
    )

    entities_after = len(store.list_entities())
    assert entities_after == entities_before  # still no phantom entity for the unknown name
    assert len(results) == 1
    assert results[0]["statement"] == "Sarah Kovacs is Noah's dance teammate"


def test_search_memories_time_range_filter(store):
    _seed_dance_fact(store)
    mcp = build_server(store)

    in_range = _call(
        mcp,
        "search_memories",
        {"query": "dance", "time_range": ["2025-01-01", "2025-01-31"]},
    )
    out_of_range = _call(
        mcp,
        "search_memories",
        {"query": "dance", "time_range": ["2025-06-01", "2025-06-30"]},
    )

    assert len(in_range) == 1
    assert out_of_range == []


def test_search_memories_excludes_expired_facts_by_default_but_include_expired_returns_them(store):
    """Bi-temporal validity wired into the read path (fix-wave-1 item 9): a fact expired in
    the past must disappear from a default search (valid_at=now()) but still come back with
    include_expired=True -- the escape hatch this tool gets per the dispatch."""
    _entity_id, fact_id = _seed_dance_fact(store)
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    mcp = build_server(store)

    default_results = _call(mcp, "search_memories", {"query": "who is on the dance team"})
    assert default_results == []

    included = _call(mcp, "search_memories", {"query": "who is on the dance team", "include_expired": True})
    assert len(included) == 1
    assert included[0]["statement"] == "Sarah Kovacs is Noah's dance teammate"


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------


def test_answer_question_cites_at_least_one_fact_id(store):
    _entity_id, fact_id = _seed_dance_fact(store)
    decompose = _FakeDecomposeModel(["Sarah dance teammate"])
    synthesize = _FakeSynthesizeModel(f"Sarah is Noah's dance teammate [fact:{fact_id}].")
    mcp = build_server(store, decompose_model=decompose, synthesize_model=synthesize)

    result = _call(mcp, "answer_question", {"question": "Who is on my dance team?"})

    assert f"[fact:{fact_id}]" in result["answer"]
    assert len(result["facts"]) == 1
    assert result["facts"][0]["id"] == fact_id
    assert result["facts"][0]["statement"] == "Sarah Kovacs is Noah's dance teammate"
    assert decompose.calls  # decompose model was actually invoked
    assert synthesize.calls


def test_answer_question_no_citation_returns_empty_facts_list(store):
    _seed_dance_fact(store)
    decompose = _FakeDecomposeModel(["irrelevant query"])
    synthesize = _FakeSynthesizeModel("I don't have enough information to answer that.")
    mcp = build_server(store, decompose_model=decompose, synthesize_model=synthesize)

    result = _call(mcp, "answer_question", {"question": "What is the capital of France?"})

    assert result["facts"] == []
    assert "enough information" in result["answer"]


def test_answer_question_excludes_expired_facts_from_retrieval_context(store):
    """No include_expired escape hatch on this tool per the dispatch -- always as-of now.
    The expired fact must never even reach the synthesis prompt as candidate context."""
    _entity_id, fact_id = _seed_dance_fact(store)
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    decompose = _FakeDecomposeModel(["Sarah dance teammate"])
    synthesize = _FakeSynthesizeModel("I don't have enough information to answer that.")
    mcp = build_server(store, decompose_model=decompose, synthesize_model=synthesize)

    result = _call(mcp, "answer_question", {"question": "Who is on my dance team?"})

    assert result["facts"] == []
    assert synthesize.calls
    assert "dance teammate" not in synthesize.calls[0]  # expired fact never entered the prompt


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


def test_get_profile_no_profile_yet_returns_friendly_message(store):
    mcp = build_server(store)
    result = _call(mcp, "get_profile", {})
    assert "no profile" in result.lower()


def test_get_profile_full_and_by_section(store):
    body = (
        "# Profile\n\n"
        "## Identity\nNoah is a software engineer.\n\n"
        "## People\nSarah Kovacs is a dance teammate. [fact:abc123]\n"
    )
    store.save_profile(body, fact_count=1)
    mcp = build_server(store)

    full = _call(mcp, "get_profile", {})
    assert full == body

    people_section = _call(mcp, "get_profile", {"section": "People"})
    assert "Sarah Kovacs" in people_section
    assert "Identity" not in people_section

    missing = _call(mcp, "get_profile", {"section": "Nonexistent"})
    assert "no section" in missing.lower()


# ---------------------------------------------------------------------------
# query_timeline
# ---------------------------------------------------------------------------


def test_query_timeline_filters_by_happened_at_range_and_sorts(store):
    backend = get_backend()
    early = Fact(
        kind=FactKind.event, statement="Trip to Budapest", confidence=0.9, happened_at="2024-08-01",
        provenance=["r1"],
    )
    late = Fact(
        kind=FactKind.event, statement="Saturday dinner with the team", confidence=0.9, happened_at="2025-01-18",
        provenance=["r2"],
    )
    undated = Fact(kind=FactKind.habit, statement="Runs every morning", confidence=0.7, provenance=["r3"])
    store.add_fact(early, backend.embed_docs([early.statement])[0])
    late_id = store.add_fact(late, backend.embed_docs([late.statement])[0])
    store.add_fact(undated, backend.embed_docs([undated.statement])[0])
    mcp = build_server(store)

    results = _call(mcp, "query_timeline", {"start": "2025-01-01", "end": "2025-01-31"})

    assert len(results) == 1
    assert results[0]["statement"] == "Saturday dinner with the team"
    assert late_id  # sanity


def test_query_timeline_empty_range_gives_empty_list(store):
    mcp = build_server(store)
    results = _call(mcp, "query_timeline", {"start": "2020-01-01", "end": "2020-01-02"})
    assert results == []


def test_query_timeline_excludes_expired_facts_by_default_but_include_expired_returns_them(store):
    backend = get_backend()
    fact = Fact(
        kind=FactKind.event, statement="Trip to Budapest", confidence=0.9, happened_at="2024-08-01",
        provenance=["r1"],
    )
    fact_id = store.add_fact(fact, backend.embed_docs([fact.statement])[0])
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    mcp = build_server(store)

    default_results = _call(mcp, "query_timeline", {"start": "2024-01-01", "end": "2024-12-31"})
    assert default_results == []

    included = _call(
        mcp, "query_timeline", {"start": "2024-01-01", "end": "2024-12-31", "include_expired": True}
    )
    assert len(included) == 1
    assert included[0]["statement"] == "Trip to Budapest"


# ---------------------------------------------------------------------------
# get_person / list_people
# ---------------------------------------------------------------------------


def test_get_person_found_returns_card_with_facts(store):
    entity_id, fact_id = _seed_dance_fact(store)
    mcp = build_server(store)

    card = _call(mcp, "get_person", {"name": "Sarah Kovacs"})

    assert card["found"] is True
    assert card["id"] == entity_id
    assert card["name"] == "Sarah Kovacs"
    assert len(card["facts"]) == 1
    assert card["facts"][0]["id"] == fact_id if "id" in card["facts"][0] else True


def test_get_person_not_found(store):
    mcp = build_server(store)
    card = _call(mcp, "get_person", {"name": "Nobody At All"})
    assert card["found"] is False


def test_get_person_excludes_expired_facts_no_escape_hatch(store):
    """No include_expired escape hatch on this tool per the dispatch -- always as-of now."""
    entity_id, fact_id = _seed_dance_fact(store)
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    mcp = build_server(store)

    card = _call(mcp, "get_person", {"name": "Sarah Kovacs"})

    assert card["found"] is True
    assert card["id"] == entity_id
    assert card["facts"] == []


def test_list_people_reports_fact_counts(store):
    _seed_dance_fact(store)
    backend = get_backend()
    store.upsert_entity("Boston", "place", backend.embed_docs(["Boston"])[0])
    mcp = build_server(store)

    people = _call(mcp, "list_people")

    assert len(people) == 1  # place entities excluded
    assert people[0]["name"] == "Sarah Kovacs"
    assert people[0]["fact_count"] == 1


def test_list_people_fact_count_excludes_expired_facts(store):
    _entity_id, fact_id = _seed_dance_fact(store)
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))
    mcp = build_server(store)

    people = _call(mcp, "list_people")

    assert people[0]["fact_count"] == 0
