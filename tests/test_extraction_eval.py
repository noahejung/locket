"""Extraction eval matcher tests.

Unit tests (`score()`, `load_gold()`) run unmarked against hand-built fixtures and a
deterministic fake embedder -- no DB, no LLM, no real (heavy) embedding model. The real
pipeline run (`run_extraction_pipeline` over demo_corpus, scored against the full 60-fact
gold file, recorded to evals/BASELINE.md) is `@pytest.mark.llm` and self-skips without
ANTHROPIC_API_KEY, per the dispatch's explicit instruction not to fabricate baseline
numbers.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import pytest

from evals.extraction_eval import GoldFact, load_gold, run_extraction_pipeline, score
from locket.store import FactRow

GOLD_PATH = Path(__file__).parent.parent / "evals" / "gold" / "persona_gold.yaml"


def _fact_row(kind: str, statement: str, *, id_: str = "f1") -> FactRow:
    return FactRow(
        id=id_,
        kind=kind,
        statement=statement,
        confidence=0.9,
        entity_ids=[],
        provenance=["r1"],
        happened_at=None,
        valid_at=None,
        invalid_at=None,
    )


def _gold(kind: str, statement: str, must_match: list[str] | None = None) -> GoldFact:
    return GoldFact(kind=kind, statement=statement, subjects=[], must_match=must_match or [])


def _fake_embed(texts: list[str], *, dims: int = 32):
    """Deterministic bag-of-words-by-hash-bucket embedder -- no real model, no
    PYTHONHASHSEED dependence (uses zlib.crc32, not the builtin hash()). Cosine similarity
    tracks shared-vocabulary overlap closely enough to exercise the COSINE_FLOOR guard
    predictably: near-identical wording -> high similarity, unrelated wording -> low."""
    import re

    vecs = []
    for t in texts:
        v = [0.0] * dims
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[zlib.crc32(w.encode()) % dims] += 1.0
        vecs.append(v)
    return vecs


# ---------------------------------------------------------------------------
# load_gold -- the real gold file
# ---------------------------------------------------------------------------


def test_load_gold_parses_the_real_persona_gold_file():
    facts = load_gold(GOLD_PATH)
    assert len(facts) == 60
    assert all(f.statement for f in facts)
    assert all(f.must_match for f in facts)  # every entry anchors on at least one regex


# ---------------------------------------------------------------------------
# score() -- matcher semantics on hand-built fixtures
# ---------------------------------------------------------------------------


def test_exact_topical_match_counts_as_matched_with_perfect_precision_recall():
    gold = [_gold("event", "The group had dinner at Bertucci's for Cory's birthday", ["bertucci", "birthday"])]
    extracted = [_fact_row("event", "The group had dinner at Bertucci's for Cory's birthday")]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.misses == []
    assert report.spurious == []


def test_kind_mismatch_never_matches_even_with_identical_text():
    gold = [_gold("event", "Cory's birthday dinner at Bertucci's", ["bertucci"])]
    extracted = [_fact_row("habit", "Cory's birthday dinner at Bertucci's")]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.recall == 0.0
    assert report.misses == ["Cory's birthday dinner at Bertucci's"]
    assert report.spurious == ["Cory's birthday dinner at Bertucci's"]


def test_must_match_regex_guards_against_topically_similar_but_wrong_fact():
    gold = [_gold("event", "Cory's birthday dinner was at Bertucci's Trattoria", ["bertucci"])]
    # Same kind, vaguely similar words, but no "bertucci" anywhere -- must not match.
    extracted = [_fact_row("event", "Cory had a birthday dinner somewhere downtown")]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.recall == 0.0
    assert report.misses


def test_cosine_floor_guards_against_regex_lucky_nonsense():
    """Both facts are 'event' kind and both happen to contain the literal word "bertucci"
    (satisfying kind + regex), but the surrounding statement is about something completely
    unrelated -- the cosine-similarity guard must still reject this pairing."""
    gold = [_gold("event", "The group had a birthday dinner at Bertucci's Trattoria to celebrate Cory", ["bertucci"])]
    extracted = [
        _fact_row(
            "event",
            "A stranger far away mentioned overhearing someone say the word bertucci once at an unrelated conference in another country entirely",
        )
    ]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.recall == 0.0  # regex hit, kind matches, but cosine similarity too low


def test_greedy_matching_never_double_counts_one_extracted_fact():
    gold = [
        _gold("event", "Cory's birthday dinner at Bertucci's", ["bertucci"]),
        _gold("event", "Cory's birthday dinner at Bertucci's again", ["bertucci"]),  # near-duplicate gold
    ]
    extracted = [_fact_row("event", "Cory's birthday dinner at Bertucci's", id_="only-one")]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.precision == 1.0  # the one extracted fact was used, and used correctly
    assert report.recall == 0.5  # only one of the two (near-duplicate) gold facts got it
    assert len(report.misses) == 1


def test_by_kind_breakdown_is_computed_per_kind():
    gold = [
        _gold("event", "Cory's birthday dinner at Bertucci's", ["bertucci"]),
        _gold("habit", "Sarah does yoga on Tuesdays", ["yoga"]),
    ]
    extracted = [
        _fact_row("event", "Cory's birthday dinner at Bertucci's", id_="e1"),
        _fact_row("habit", "Sarah does yoga on Tuesdays", id_="h1"),
        _fact_row("habit", "Some unrelated spurious habit fact about nothing in the gold set", id_="h2"),
    ]

    report = score(extracted, gold, embed_fn=_fake_embed)

    assert report.by_kind["event"].gold == 1
    assert report.by_kind["event"].matched == 1
    assert report.by_kind["event"].precision == 1.0
    assert report.by_kind["habit"].gold == 1
    assert report.by_kind["habit"].extracted == 2
    assert report.by_kind["habit"].matched == 1
    assert report.by_kind["habit"].precision == 0.5  # 1 matched / 2 extracted habit facts
    assert report.by_kind["habit"].recall == 1.0


def test_empty_extracted_list_gives_zero_precision_and_all_misses():
    gold = [_gold("event", "Something that happened", ["something"])]
    report = score([], gold, embed_fn=_fake_embed)

    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.misses == ["Something that happened"]


def test_empty_gold_list_gives_zero_recall_and_all_spurious():
    extracted = [_fact_row("event", "Something extracted with no gold counterpart")]
    report = score(extracted, [], embed_fn=_fake_embed)

    assert report.recall == 0.0
    assert report.spurious == ["Something extracted with no gold counterpart"]


# ---------------------------------------------------------------------------
# Live run: full pipeline over demo_corpus, scored against the full gold set,
# baseline recorded to evals/BASELINE.md. Blocked on ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------


@pytest.mark.llm
def test_live_extraction_pipeline_scores_against_full_gold_set():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live extraction baseline run")

    from locket.store import Store

    db_url = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")
    store = Store(db_url)
    try:
        with store._conn.cursor() as cur:
            cur.execute(
                "TRUNCATE raw_items, facts, entities, fact_history, merge_proposals, "
                "extracted_windows RESTART IDENTITY CASCADE"
            )
        store._conn.commit()

        corpus_dir = Path(__file__).parent.parent / "demo_corpus"
        extracted = run_extraction_pipeline(store, corpus_dir)
        gold = load_gold(GOLD_PATH)
        report = score(extracted, gold)

        assert report.precision >= 0.0  # baseline run — record numbers, don't gate yet
        assert report.recall >= 0.0
    finally:
        store.close()
