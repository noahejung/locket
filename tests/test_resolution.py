"""Entity resolution tests.

Pure helper functions (normalization, prefix matching) run unmarked -- fast, no DB, no
model. `resolve()`'s tiers 1-2 exercise the real dockerized Postgres + the real local
embedding backend (already downloaded from Week 2 -- no new network needed) via the `db`
mark. Tier-3 escalation is exercised with a stubbed model (predicate-based, mirroring
tests/test_extraction_graph.py's `_ScriptedModel`) -- no live LLM call, still `db`-marked
because it still needs a real Store for tier 1's candidate search and the confirm queue.
"""

from __future__ import annotations

import os

import pytest

from locket.resolution import (
    MatchVerdict,
    _exact_match,
    _first_token,
    _normalize,
    _prefix_related,
    label_face_cluster,
    pending_confirmations,
    resolve,
)
from locket.store import EntityRow, Store

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@localhost:5432/locket")


# ---------------------------------------------------------------------------
# Pure helpers -- no marks
# ---------------------------------------------------------------------------


def test_normalize_strips_emoji_and_treats_separators_as_whitespace():
    assert _normalize("Sarah M ⭐") == "sarah m"
    assert _normalize("sarah.mendes") == "sarah mendes"
    assert _normalize("Sarah Mendes") == "sarah mendes"
    assert _normalize("Cory_Davis-photo") == "cory davis photo"


def test_first_token_uses_normalized_form():
    assert _first_token("Josh V") == "josh"
    assert _first_token("Joshua Vega") == "joshua"
    assert _first_token("") == ""


def test_prefix_related_matches_either_direction():
    assert _prefix_related("josh", "joshua")
    assert _prefix_related("joshua", "josh")
    assert not _prefix_related("josh", "jane")
    assert not _prefix_related("", "josh")


def test_exact_match_checks_name_and_aliases():
    entity = EntityRow(id="e1", name="Sarah Mendes", kind="person", similarity=1.0, aliases=["sarah.mendes"])
    assert _exact_match("sarah.mendes", entity)  # via name normalization
    assert _exact_match("Sarah Mendes", entity)  # via name
    assert not _exact_match("Sarah M", entity)  # partial, not exact


# ---------------------------------------------------------------------------
# resolve() tiers 1-2 -- real db + real embeddings, no LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    s = Store(DB_URL)
    with s._conn.cursor() as cur:
        cur.execute(
            "TRUNCATE raw_items, facts, entities, fact_history, merge_proposals RESTART IDENTITY CASCADE"
        )
    s._conn.commit()
    yield s
    s._conn.close()


@pytest.mark.db
def test_noisy_platform_variants_resolve_to_one_entity_via_tiers_1_2_only(store):
    """Real persona noise from demo_corpus (corpusgen/personas.py): "Sarah Mendes"
    (WhatsApp/canonical), "sarah.mendes" (Instagram handle), "Sarah M ⭐" (SMS contact
    name). Measured live (2026-07-30) against the real arctic-embed-s backend: 0.8954 and
    0.6631 cosine similarity respectively against the canonical name -- both clear
    SIMILARITY_FLOOR=0.6, so tier 1 surfaces both as candidates without any LLM call."""
    resolved = resolve(store, ["Sarah Mendes", "sarah.mendes", "Sarah M ⭐"])

    assert len(resolved) == 3
    ids = set(resolved.values())
    assert len(ids) == 1  # all three converged on the same entity

    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities")
        assert cur.fetchone()[0] == 1  # no duplicate entity created
    assert pending_confirmations(store) == []  # resolved deterministically, nothing queued


@pytest.mark.db
def test_prefix_tier_resolves_josh_v_to_joshua_vega(store):
    """"Joshua Vega" vs SMS contact name "Josh V" -- measured 0.6119 cosine similarity,
    clears the floor; normalized full strings differ ("joshua vega" != "josh v") so the
    exact-match tier doesn't fire, but "josh" is an unambiguous prefix of "joshua" among
    the tier-1 candidate set."""
    resolved = resolve(store, ["Joshua Vega", "Josh V"])

    assert len(resolved) == 2
    assert resolved["Joshua Vega"] == resolved["Josh V"]


@pytest.mark.db
def test_genuinely_new_name_creates_a_new_entity(store):
    resolved = resolve(store, ["Someone Entirely New"])

    assert "Someone Entirely New" in resolved
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities")
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# resolve() tier 3 -- stubbed model, ambiguous case queues instead of merging
# ---------------------------------------------------------------------------


class _StubModel:
    """Predicate-based canned verdicts, keyed on substrings of the rendered prompt."""

    def __init__(self, rules: list[tuple[str, MatchVerdict]]):
        self._rules = rules
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> MatchVerdict:
        self.calls.append(prompt)
        for needle, verdict in self._rules:
            if needle in prompt:
                return verdict
        raise AssertionError(f"no scripted verdict for prompt:\n{prompt}")


@pytest.mark.db
def test_ambiguous_alex_produces_a_queue_item_not_a_silent_merge(store):
    """Two real entities both named "Alex ___". A bare "Alex" mention clears the tier-1
    floor against both (measured live: 0.615 and 0.651) but the exact-match and
    first-name-prefix tiers can't disambiguate -- both candidates' first token IS "alex".
    A stubbed tier-3 model returns plausible-but-not-confident verdicts for both, so
    neither should auto-merge (LLM_CONFIRM_THRESHOLD=0.85) and both should land in the
    confirm queue instead."""
    from locket.embeddings import get_backend

    backend = get_backend()
    store.upsert_entity("Alex Rivera", "person", backend.embed_docs(["Alex Rivera"])[0])
    store.upsert_entity("Alex Chen", "person", backend.embed_docs(["Alex Chen"])[0])

    model = _StubModel(
        [
            ("Alex Rivera", MatchVerdict(same=True, confidence=0.55)),
            ("Alex Chen", MatchVerdict(same=True, confidence=0.45)),
        ]
    )

    resolved = resolve(store, ["Alex"], model=model)

    assert "Alex" not in resolved  # no silent merge

    proposals = pending_confirmations(store)
    assert len(proposals) == 2
    assert {p.candidate_entity_name for p in proposals} == {"Alex Rivera", "Alex Chen"}
    assert all(p.mention == "Alex" for p in proposals)
    assert all(0.0 <= p.score <= 1.0 for p in proposals)


@pytest.mark.db
def test_high_confidence_tier3_verdict_auto_merges(store):
    from locket.embeddings import get_backend

    backend = get_backend()
    store.upsert_entity("Alexandra Rivera", "person", backend.embed_docs(["Alexandra Rivera"])[0])

    model = _StubModel([("Alexandra Rivera", MatchVerdict(same=True, confidence=0.95))])

    resolved = resolve(store, ["Alexandra"], model=model)

    assert "Alexandra" in resolved
    assert pending_confirmations(store) == []


# ---------------------------------------------------------------------------
# Face-cluster labeling -- separate path, exact alias lookup
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_label_face_cluster_then_resolve_maps_cluster_to_entity(store):
    entity_id = label_face_cluster(store, cluster_id=3, entity_name="Kathryn Petrovic")

    resolved = resolve(store, ["face:3"])

    assert resolved["face:3"] == entity_id


@pytest.mark.db
def test_unlabeled_face_cluster_mention_does_not_resolve(store):
    resolved = resolve(store, ["face:999"])
    assert "face:999" not in resolved
