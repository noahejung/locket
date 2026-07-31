"""RAG eval harness tests.

The harness test (`test_run_rag_eval_...`) stubs retrieval, answering, AND the three ragas
metrics -- no DB, no embedding model, no LLM, no network -- verifying only the harness's
own plumbing (looping the question set, calling each metric with the kwargs shape its real
`ascore()` signature actually takes, averaging into RagEvalResult). The live run against
real ragas + the real Claude judge is `@pytest.mark.llm` and self-skips without
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.rag_eval import (
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXT_PRECISION_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    RagQuestion,
    load_questions,
    run_rag_eval_async,
)
from locket.store import FactRow

QUESTIONS_PATH = Path(__file__).parent.parent / "evals" / "questions.yaml"


class _StubMetric:
    """Duck-types ragas.metrics.collections.{Faithfulness,AnswerRelevancy,
    ContextPrecisionWithReference} -- only the async ascore(**kwargs) -> obj-with-.value
    surface the harness actually calls."""

    def __init__(self, value: float):
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class _FakeAnswerModel:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content="a fake synthesized answer")


def _fact_row(statement: str) -> FactRow:
    return FactRow(
        id="f1",
        kind="event",
        statement=statement,
        confidence=0.9,
        entity_ids=[],
        provenance=["r1"],
        happened_at=None,
        valid_at=None,
        invalid_at=None,
    )


# ---------------------------------------------------------------------------
# load_questions -- the real question file
# ---------------------------------------------------------------------------


def test_load_questions_parses_the_real_questions_file():
    questions = load_questions(QUESTIONS_PATH)
    assert len(questions) == 25
    assert all(q.question and q.reference for q in questions)


# ---------------------------------------------------------------------------
# Harness -- fully stubbed, no DB/model/network
# ---------------------------------------------------------------------------


def test_run_rag_eval_computes_means_across_questions():
    questions = [
        RagQuestion(question="Where was the birthday dinner?", reference="Bertucci's"),
        RagQuestion(question="What is Sarah's yoga studio?", reference="Radiant Yoga"),
    ]
    fake_rows = [_fact_row("The birthday dinner was at Bertucci's")]

    def fake_retrieve(store, question_text: str) -> list[FactRow]:
        return fake_rows

    metrics = {
        "faithfulness": _StubMetric(0.9),
        "answer_relevancy": _StubMetric(0.8),
        "context_precision": _StubMetric(0.7),
    }

    result = asyncio.run(
        run_rag_eval_async(
            store=None,
            questions=questions,
            retrieve_fn=fake_retrieve,
            answer_model=_FakeAnswerModel(),
            metrics=metrics,
        )
    )

    assert result.faithfulness == pytest.approx(0.9)
    assert result.answer_relevancy == pytest.approx(0.8)
    assert result.context_precision == pytest.approx(0.7)
    assert len(result.per_question) == 2
    assert result.per_question[0].retrieved_contexts == ["The birthday dinner was at Bertucci's"]
    assert result.per_question[0].response == "a fake synthesized answer"


def test_run_rag_eval_calls_each_metric_with_its_own_kwarg_shape():
    """faithfulness needs response+retrieved_contexts; answer_relevancy needs response but
    NOT reference (it measures response-to-question relevance, not response-to-reference);
    context_precision needs reference+retrieved_contexts but NOT response. Getting these
    kwarg shapes wrong is exactly the kind of mismatch the plan flags as needing live
    verification against the actual installed ragas 0.4.3 API, not the (stale) docs."""
    questions = [RagQuestion(question="Q?", reference="R")]
    fake_rows = [_fact_row("some fact")]

    faithfulness = _StubMetric(1.0)
    answer_relevancy = _StubMetric(1.0)
    context_precision = _StubMetric(1.0)

    asyncio.run(
        run_rag_eval_async(
            store=None,
            questions=questions,
            retrieve_fn=lambda store, q: fake_rows,
            answer_model=_FakeAnswerModel(),
            metrics={
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
            },
        )
    )

    assert faithfulness.calls[0].keys() == {"user_input", "response", "retrieved_contexts"}
    assert answer_relevancy.calls[0].keys() == {"user_input", "response"}
    assert context_precision.calls[0].keys() == {"user_input", "reference", "retrieved_contexts"}


def test_run_rag_eval_empty_question_set_gives_zero_means_not_an_error():
    result = asyncio.run(
        run_rag_eval_async(
            store=None,
            questions=[],
            retrieve_fn=lambda store, q: [],
            answer_model=_FakeAnswerModel(),
            metrics={
                "faithfulness": _StubMetric(1.0),
                "answer_relevancy": _StubMetric(1.0),
                "context_precision": _StubMetric(1.0),
            },
        )
    )
    assert result.faithfulness == 0.0
    assert result.answer_relevancy == 0.0
    assert result.context_precision == 0.0
    assert result.per_question == []


# ---------------------------------------------------------------------------
# Live run: real ragas judge + embeddings, real demo_corpus pipeline.
# Blocked on ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------


@pytest.mark.llm
def test_live_rag_eval_meets_thresholds():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live RAG eval run")

    from evals.extraction_eval import run_extraction_pipeline
    from evals.rag_eval import run_rag_eval
    from locket.store import Store

    db_url = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@localhost:5432/locket")
    store = Store(db_url)
    try:
        with store._conn.cursor() as cur:
            cur.execute(
                "TRUNCATE raw_items, facts, entities, fact_history, merge_proposals RESTART IDENTITY CASCADE"
            )
        store._conn.commit()

        corpus_dir = Path(__file__).parent.parent / "demo_corpus"
        run_extraction_pipeline(store, corpus_dir)

        questions = load_questions(QUESTIONS_PATH)
        result = run_rag_eval(store, questions)

        assert result.faithfulness >= FAITHFULNESS_THRESHOLD
        assert result.answer_relevancy >= ANSWER_RELEVANCY_THRESHOLD
        assert result.context_precision >= CONTEXT_PRECISION_THRESHOLD
    finally:
        store.close()
