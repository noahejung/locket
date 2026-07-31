"""Unit tests run entirely against a hand-stubbed model — no network. One @pytest.mark.llm
live smoke exercises the real Claude API and is skipped if ANTHROPIC_API_KEY is unset."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from locket.extraction.graph import _should_escalate, extract_batch, run_batch
from locket.extraction.schemas import ExtractedFact, ExtractionResult
from locket.models import FactKind, RawItem, SourceKind


class _ScriptedModel:
    """Test double for a with_structured_output(..., include_raw=True) runnable.

    `rules` maps a predicate over the rendered prompt to a queue of canned responses
    returned in order. Predicate-based (not call-order-based) on purpose: LangGraph's Send
    fan-out does not guarantee which window's node body runs first, so a single shared
    call-order queue would be flaky across windows.
    """

    def __init__(self, rules: list[tuple[Callable[[str], bool], list[dict]]]):
        self._rules = [(pred, list(queue)) for pred, queue in rules]
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> dict:
        self.calls.append(prompt)
        for pred, queue in self._rules:
            if pred(prompt) and queue:
                return queue.pop(0)
        raise AssertionError(f"no scripted response left for prompt:\n{prompt}")


def _item(ts: datetime, text: str, sender: str = "John") -> RawItem:
    return RawItem.make(source=SourceKind.whatsapp, ts=ts, sender=sender, text=text, thread="t")


def _ok(statement: str) -> dict:
    fact = ExtractedFact(kind=FactKind.event, statement=statement, subjects=["John"], confidence=0.9)
    return {"raw": None, "parsed": ExtractionResult(facts=[fact]), "parsing_error": None}


def _err(message: str) -> dict:
    return {"raw": None, "parsed": None, "parsing_error": ValueError(message)}


def test_should_escalate_is_pure_and_network_free():
    assert _should_escalate(0) is False
    assert _should_escalate(1) is False
    assert _should_escalate(2) is True
    assert _should_escalate(3) is True


def test_fan_out_returns_facts_from_both_windows_with_correct_provenance():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    window_a = [_item(base, "windowA hello")]
    window_b = [_item(base + timedelta(hours=10), "windowB hello")]

    model = _ScriptedModel(
        [
            (lambda p: "windowA" in p, [_ok("Fact from window A")]),
            (lambda p: "windowB" in p, [_ok("Fact from window B")]),
        ]
    )

    results = extract_batch(window_a + window_b, model=model)

    assert len(results) == 2
    statements = {fact.statement for fact, _prov in results}
    assert statements == {"Fact from window A", "Fact from window B"}
    for fact, provenance in results:
        if fact.statement == "Fact from window A":
            assert provenance == [window_a[0].id]
        else:
            assert provenance == [window_b[0].id]


def test_corrective_retry_feeds_error_text_into_the_retry_prompt():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [_item(base, "retryme hello")]

    model = _ScriptedModel(
        [(lambda p: "retryme" in p, [_err("kind must be one of the enum values"), _ok("Recovered fact")])]
    )

    results = extract_batch(items, model=model)

    assert len(results) == 1
    assert results[0][0].statement == "Recovered fact"
    assert len(model.calls) == 2
    assert "kind must be one of the enum values" not in model.calls[0]
    assert "kind must be one of the enum values" in model.calls[1]


def test_three_failures_give_up_with_notes_marker_not_exception():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    failing = [_item(base, "failwin hello")]

    model = _ScriptedModel(
        [(lambda p: "failwin" in p, [_err("bad 1"), _err("bad 2"), _err("bad 3")])]
    )

    raw_results = run_batch(failing, model=model)  # no exception raised

    assert len(raw_results) == 1
    result = raw_results[0]
    assert result["facts"] is None
    assert result["attempt"] == 3
    assert result.get("notes")
    assert "bad 3" in result["notes"]
    assert len(model.calls) == 3


def test_give_up_window_does_not_block_a_sibling_windows_facts():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    failing = [_item(base, "failwin hello")]
    succeeding = [_item(base + timedelta(hours=10), "okwin hello")]

    model = _ScriptedModel(
        [
            (lambda p: "failwin" in p, [_err("bad 1"), _err("bad 2"), _err("bad 3")]),
            (lambda p: "okwin" in p, [_ok("A fine fact")]),
        ]
    )

    results = extract_batch(failing + succeeding, model=model)  # no exception raised

    assert len(results) == 1
    assert results[0][0].statement == "A fine fact"
    assert results[0][1] == [succeeding[0].id]


@pytest.mark.llm
def test_live_smoke_extracts_at_least_one_plausible_fact():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live smoke")

    from locket.adapters.whatsapp import parse_whatsapp

    demo = Path(__file__).parent.parent / "demo_corpus" / "whatsapp" / "team.txt"
    items = list(parse_whatsapp(demo, thread="team"))[:16]  # one window's worth

    results = extract_batch(items)

    assert len(results) >= 1
    for fact, provenance in results:
        assert fact.kind in set(FactKind)
        assert provenance
        assert all(p in {i.id for i in items} for p in provenance)
