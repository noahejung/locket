"""RAG evals: Ragas over a thin retrieve+answer path built directly against
Store.search_facts (Task 8) + embeddings (Task 9) + one haiku synthesis call.

Interim wiring note (mirrors Task 15's extraction_eval.py note): `search_memories` doesn't
exist until Task 17's MCP server. `retrieve()`/`answer()` here are the thin stand-in this
task's own Interfaces section calls for; swap to the shared function when Task 17 lands.

Research-locked API facts (ragas 0.4.3 — verified live 2026-07-30 against the actually
installed package via `inspect.signature`, not the docs' own CI page, which the plan
explicitly flags as stale):
  - The MODERN track (`ragas.metrics.collections.{Faithfulness,AnswerRelevancy,
    ContextPrecisionWithReference}`) hard-rejects LangChain-wrapped LLMs. The judge is
    built with the plain `anthropic` SDK via
    `llm_factory("claude-haiku-4-5", provider="anthropic", client=Anthropic())`.
  - `evaluate()` is deprecated; score via `await metric.ascore(...)` directly (this module
    uses `asyncio.gather` across the question set, per the plan's explicit instruction).
  - Each collections metric's `ascore()` takes PLAIN KEYWORD ARGUMENTS
    (user_input/response/retrieved_contexts/reference, varying per metric) — NOT a
    `SingleTurnSample` object, despite `SingleTurnSample` still existing at
    `ragas.SingleTurnSample`. Verified via `inspect.signature(Faithfulness.ascore)` etc.
    before writing this module; the plan's own worked example text describes building a
    SingleTurnSample, which does not match the installed 0.4.3 collections API.
  - `reference` is a single string — a v0.3-era `ground_truths` list fails validation.
  - Embeddings for AnswerRelevancy: `ragas.embeddings.HuggingFaceEmbeddings(model=...,
    use_api=False)` — local, free.

Dependency note: ragas 0.4.3's own `import ragas` chain unconditionally imports
`langchain_community.chat_models.vertexai`, which the latest langchain-community (0.4.2)
dropped entirely. Fixed by pinning `langchain-community==0.4.1` in pyproject.toml (see that
commit) — without it, even `from ragas.metrics.collections import ...` raises
ModuleNotFoundError before any of the above ever runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import Field as PydField

from locket.embeddings import get_backend
from locket.store import FactRow, Store

FAITHFULNESS_THRESHOLD = 0.85
ANSWER_RELEVANCY_THRESHOLD = 0.80
CONTEXT_PRECISION_THRESHOLD = 0.70

JUDGE_MODEL_ID = "claude-haiku-4-5"
RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_ANSWER_SYSTEM = (
    "Answer the question using ONLY the facts given below. Be concise, one or two "
    "sentences. If the facts don't cover the question, say so plainly."
)


class RagQuestion(BaseModel):
    question: str
    reference: str
    kinds: list[str] = PydField(default_factory=list)


@dataclass
class QuestionResult:
    question: str
    response: str
    retrieved_contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float


@dataclass
class RagEvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    per_question: list[QuestionResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "per_question": [
                {
                    "question": q.question,
                    "response": q.response,
                    "retrieved_contexts": q.retrieved_contexts,
                    "faithfulness": q.faithfulness,
                    "answer_relevancy": q.answer_relevancy,
                    "context_precision": q.context_precision,
                }
                for q in self.per_question
            ],
        }


def load_questions(path: Path) -> list[RagQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [RagQuestion.model_validate(d) for d in data]


def retrieve(store: Store, backend: Any, question: str, *, limit: int = 5) -> list[FactRow]:
    embedding = backend.embed_query(question)
    return store.search_facts(embedding, limit=limit)


@cache
def _default_answer_model() -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=JUDGE_MODEL_ID)


def answer(question: str, contexts: list[FactRow], *, model: Any | None = None) -> str:
    active_model = model if model is not None else _default_answer_model()
    facts_block = "\n".join(f"- {c.statement}" for c in contexts) or "(no facts retrieved)"
    prompt = f"{_ANSWER_SYSTEM}\n\nFacts:\n{facts_block}\n\nQuestion: {question}"
    response = active_model.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _build_judge() -> Any:
    from anthropic import Anthropic
    from ragas.llms import llm_factory

    return llm_factory(JUDGE_MODEL_ID, provider="anthropic", client=Anthropic())


def _build_ragas_embeddings() -> Any:
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=RAGAS_EMBEDDING_MODEL, use_api=False)


def _default_metrics() -> dict[str, Any]:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        Faithfulness,
    )

    judge = _build_judge()
    embeddings = _build_ragas_embeddings()
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=judge),
    }


async def _score_question(
    q: RagQuestion,
    store: Store,
    *,
    retrieve_fn: Any,
    answer_model: Any | None,
    metrics: dict[str, Any],
) -> QuestionResult:
    context_rows = retrieve_fn(store, q.question)
    contexts = [c.statement for c in context_rows]
    response_text = answer(q.question, context_rows, model=answer_model)

    faithfulness_result = await metrics["faithfulness"].ascore(
        user_input=q.question, response=response_text, retrieved_contexts=contexts
    )
    answer_relevancy_result = await metrics["answer_relevancy"].ascore(
        user_input=q.question, response=response_text
    )
    context_precision_result = await metrics["context_precision"].ascore(
        user_input=q.question, reference=q.reference, retrieved_contexts=contexts
    )

    return QuestionResult(
        question=q.question,
        response=response_text,
        retrieved_contexts=contexts,
        faithfulness=faithfulness_result.value,
        answer_relevancy=answer_relevancy_result.value,
        context_precision=context_precision_result.value,
    )


async def run_rag_eval_async(
    store: Store,
    questions: list[RagQuestion],
    *,
    retrieve_fn: Any = None,
    answer_model: Any | None = None,
    metrics: dict[str, Any] | None = None,
) -> RagEvalResult:
    """`retrieve_fn(store, question_text) -> list[FactRow]` defaults to
    `retrieve(store, get_backend(), question_text)`. `metrics` defaults to the real ragas
    collections metrics (needs ANTHROPIC_API_KEY + network); both are injectable test seams
    (mirrors extraction_eval.score's `embed_fn` and graph.py/resolution.py's `model=`)."""
    if retrieve_fn is None:
        backend = get_backend()

        def retrieve_fn(s: Store, question_text: str) -> list[FactRow]:
            return retrieve(s, backend, question_text)

    active_metrics = metrics if metrics is not None else _default_metrics()

    results = await asyncio.gather(
        *[
            _score_question(q, store, retrieve_fn=retrieve_fn, answer_model=answer_model, metrics=active_metrics)
            for q in questions
        ]
    )

    return RagEvalResult(
        faithfulness=mean(r.faithfulness for r in results) if results else 0.0,
        answer_relevancy=mean(r.answer_relevancy for r in results) if results else 0.0,
        context_precision=mean(r.context_precision for r in results) if results else 0.0,
        per_question=list(results),
    )


def run_rag_eval(store: Store, questions: list[RagQuestion], **kwargs: Any) -> RagEvalResult:
    return asyncio.run(run_rag_eval_async(store, questions, **kwargs))


__all__ = [
    "ANSWER_RELEVANCY_THRESHOLD",
    "CONTEXT_PRECISION_THRESHOLD",
    "FAITHFULNESS_THRESHOLD",
    "RagEvalResult",
    "RagQuestion",
    "answer",
    "load_questions",
    "retrieve",
    "run_rag_eval",
    "run_rag_eval_async",
]
