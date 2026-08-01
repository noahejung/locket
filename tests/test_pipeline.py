"""Tests for the shared corpus-walk + extract-and-persist pipeline core (src/locket/pipeline.py,
fix-wave-2 items 4-5). discover_corpus_sources is a pure filesystem walk -- already exercised
indirectly by tests/test_cli.py's pipeline-run tests and tests/test_extraction_eval.py's live
run via run_extraction_pipeline. This file's own focus is extract_and_persist's batching
contract, which neither of those exercised directly before this module existed.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from locket.extraction.schemas import ExtractedFact, ExtractionResult
from locket.models import FactKind, RawItem, SourceKind
from locket.pipeline import extract_and_persist
from locket.store import Store

pytestmark = pytest.mark.db

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")


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


class _ThreeFactsPerWindowModel:
    """Returns three distinct facts for every window it's invoked on, regardless of prompt
    content -- lets a test assert an exact resulting fact count without needing real
    extraction content, mirroring test_cli.py's _AlwaysOneFactModel pattern."""

    def __init__(self):
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> dict:
        self.calls.append(prompt)
        facts = [
            ExtractedFact(
                kind=FactKind.event,
                statement=f"fact {i} from call {len(self.calls)}",
                subjects=["Noah"],
                confidence=0.9,
            )
            for i in range(3)
        ]
        return {"raw": None, "parsed": ExtractionResult(facts=facts), "parsing_error": None}


class _CountingBackend:
    """Fake embedding backend that only records how it was called -- isolates the batching
    contract from the real (network-free but non-trivial) local embedding model."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


def _item(suffix: str) -> RawItem:
    return RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 1, tzinfo=UTC),
        sender="Noah",
        text=f"message {suffix}",
        thread="t",
    )


def test_extract_and_persist_batches_all_fact_embeddings_into_one_call(store, monkeypatch):
    """Fix-wave-2 item 5: embedding was one embed_docs([statement]) call PER extracted fact
    -- with W windows producing F facts each, that was W*F separate model calls where one
    batched call over every statement in the whole run does the same job. Two explicit
    windows_override windows x 3 facts/window (the scripted model above) = 6 facts total."""
    backend = _CountingBackend()
    monkeypatch.setattr("locket.pipeline.get_backend", lambda: backend)

    item_a, item_b = _item("a"), _item("b")
    model = _ThreeFactsPerWindowModel()

    result = extract_and_persist(store, [item_a, item_b], model=model, windows_override=[[item_a], [item_b]])

    assert len(result.rows) == 6  # 2 windows x 3 facts/window
    assert len(model.calls) == 2  # one model call per window -- extraction itself unbatched
    assert len(backend.calls) == 1  # but exactly ONE embed_docs call for every statement
    assert len(backend.calls[0]) == 6
    # Both windows succeeded on their first call -- zero retries/give-ups/escalations.
    assert result.retries == 0
    assert result.give_ups == 0
    assert result.escalations == 0


def test_extract_and_persist_returns_fact_rows_paired_with_extracted_subjects(store, monkeypatch):
    """Companion correctness check: the batching change must not scramble which embedding/
    row belongs to which extracted fact, nor drop the subjects each row is paired with
    (cli.py's entity-resolution step reads them from here)."""
    backend = _CountingBackend()
    monkeypatch.setattr("locket.pipeline.get_backend", lambda: backend)
    item_a = _item("a")

    result = extract_and_persist(store, [item_a], model=_ThreeFactsPerWindowModel(), windows_override=[[item_a]])

    assert len(result.rows) == 3
    statements = {row.statement for row, _subjects in result.rows}
    assert statements == {"fact 0 from call 1", "fact 1 from call 1", "fact 2 from call 1"}
    for row, subjects in result.rows:
        assert subjects == ["Noah"]
        assert row.id  # a real persisted fact id
        assert row.provenance == [item_a.id]


def test_extract_and_persist_no_facts_makes_no_embed_call(store, monkeypatch):
    """A window that produces zero facts (e.g. give-up-after-retries) must not call
    embed_docs at all -- embed_docs([]) is a wasted round-trip, and the pre-fix per-fact
    loop already skipped it naturally by never entering its body for zero facts."""
    backend = _CountingBackend()
    monkeypatch.setattr("locket.pipeline.get_backend", lambda: backend)

    class _NoFactsModel:
        def invoke(self, prompt: str) -> dict:
            return {"raw": None, "parsed": ExtractionResult(facts=[]), "parsing_error": None}

    item_a = _item("a")
    result = extract_and_persist(store, [item_a], model=_NoFactsModel(), windows_override=[[item_a]])

    assert result.rows == []
    assert result.give_ups == 0  # a validly-empty extraction, not a give-up
    assert backend.calls == []


def test_extract_and_persist_bubbles_up_extract_batch_counters_for_a_give_up_window(store, monkeypatch):
    """Companion to extract_batch's own counter tests (test_extraction_graph.py): this
    function is extract_batch's one caller in the whole codebase, so it must pass retries/
    give_ups/escalations through unchanged even on the empty-rows early-return path -- a
    window that gives up entirely still has counters worth reporting in the per-run JSONL
    capture (locket stats), not just windows that produced at least one fact."""
    backend = _CountingBackend()
    monkeypatch.setattr("locket.pipeline.get_backend", lambda: backend)

    class _AlwaysInvalidModel:
        def invoke(self, prompt: str) -> dict:
            return {"raw": None, "parsed": None, "parsing_error": ValueError("bad")}

    item_a = _item("a")
    result = extract_and_persist(store, [item_a], model=_AlwaysInvalidModel(), windows_override=[[item_a]])

    assert result.rows == []
    assert result.give_ups == 1
    assert result.retries == 2  # MAX_ATTEMPTS=3 calls -> 2 retries before giving up
    assert result.escalations == 1  # 3rd (final) call escalated
    assert backend.calls == []
