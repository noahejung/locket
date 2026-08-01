"""Unit tests run entirely against a hand-stubbed model — no network. One @pytest.mark.llm
live smoke exercises the real Claude API and is skipped if ANTHROPIC_API_KEY is unset. One
@pytest.mark.vision live smoke exercises the local Ollama backend (locket.llm) and is
skipped if the local server/model isn't available -- same self-skip contract as
test_vision_llm.py's describe_photo test, since both need a real local Ollama server."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from locket.extraction.graph import (
    _should_escalate,
    extract_batch,
    run_batch,
    window_hash,
    window_hash_from_provenance,
)
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

    result = extract_batch(window_a + window_b, model=model)

    assert len(result.facts) == 2
    statements = {fact.statement for fact, _prov in result.facts}
    assert statements == {"Fact from window A", "Fact from window B"}
    for fact, provenance in result.facts:
        if fact.statement == "Fact from window A":
            assert provenance == [window_a[0].id]
        else:
            assert provenance == [window_b[0].id]
    # Both windows succeeded on their first call -- zero retries/give-ups/escalations.
    assert result.retries == 0
    assert result.give_ups == 0
    assert result.escalations == 0


def test_corrective_retry_feeds_error_text_into_the_retry_prompt():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items = [_item(base, "retryme hello")]

    model = _ScriptedModel(
        [(lambda p: "retryme" in p, [_err("kind must be one of the enum values"), _ok("Recovered fact")])]
    )

    result = extract_batch(items, model=model)

    assert len(result.facts) == 1
    assert result.facts[0][0].statement == "Recovered fact"
    assert len(model.calls) == 2
    assert "kind must be one of the enum values" not in model.calls[0]
    assert "kind must be one of the enum values" in model.calls[1]
    # Succeeded on the 2nd call -- 1 retry, not yet escalated (ESCALATE_AFTER=2 means
    # escalation only kicks in on the 3rd call), no give-up.
    assert result.retries == 1
    assert result.escalations == 0
    assert result.give_ups == 0


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

    result = extract_batch(failing + succeeding, model=model)  # no exception raised

    assert len(result.facts) == 1
    assert result.facts[0][0].statement == "A fine fact"
    assert result.facts[0][1] == [succeeding[0].id]
    # failing window: 3 attempts -> 2 retries, escalated on its final (3rd) call, gave up.
    # succeeding window: 1 attempt -> 0 retries, no escalation, no give-up. Summed:
    assert result.retries == 2
    assert result.give_ups == 1
    assert result.escalations == 1


def test_window_hash_from_provenance_matches_window_hash_for_the_same_window():
    """window_hash_from_provenance (fix-wave-3 follow-up: cli.py's `_run_pipeline` needs to
    map extract_batch's given_up_window_hashes back to the pending_hashes it computed via
    window_hash BEFORE extraction ran) must be the exact same identity as window_hash, just
    computed from a provenance id list instead of a list[RawItem]."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    window = [_item(base, "hello"), _item(base + timedelta(minutes=1), "world", sender="Sarah")]

    assert window_hash_from_provenance([item.id for item in window]) == window_hash(window)


def test_window_hash_from_provenance_is_order_sensitive():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    window = [_item(base, "hello"), _item(base + timedelta(minutes=1), "world")]
    ids = [item.id for item in window]

    assert window_hash_from_provenance(ids) != window_hash_from_provenance(list(reversed(ids)))


def test_extract_batch_surfaces_given_up_window_hashes_for_only_the_windows_that_gave_up():
    """given_up_window_hashes (fix-wave-3 follow-up to the 2026-08-01 catch-up review's
    MEDIUM finding) lets a caller (cli.py's `_run_pipeline`) record ONLY the windows that
    actually gave up with Store.mark_windows_given_up, distinct from the ones that
    succeeded -- before this field existed, every attempted window was marked identically
    regardless of outcome, making a give-up permanently and silently unretryable."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    failing = [_item(base, "failwin hello")]
    succeeding = [_item(base + timedelta(hours=10), "okwin hello")]

    model = _ScriptedModel(
        [
            (lambda p: "failwin" in p, [_err("bad 1"), _err("bad 2"), _err("bad 3")]),
            (lambda p: "okwin" in p, [_ok("A fine fact")]),
        ]
    )

    result = extract_batch(failing + succeeding, model=model)

    assert result.given_up_window_hashes == [window_hash(failing)]
    assert window_hash(succeeding) not in result.given_up_window_hashes


def test_extract_batch_given_up_window_hashes_is_empty_when_nothing_gives_up():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    clean = [_item(base, "cleanwin hello")]
    model = _ScriptedModel([(lambda p: "cleanwin" in p, [_ok("Clean fact")])])

    result = extract_batch(clean, model=model)

    assert result.given_up_window_hashes == []


def test_extract_batch_surfaces_retry_give_up_and_escalation_counters_across_a_mixed_batch():
    """Dedicated counter-surfacing test (locket stats / per-run JSONL capture task): three
    windows, each exercising a different terminal state, prove BatchExtractionResult's
    retries/give_ups/escalations are summed correctly across a batch, not just correct for
    a single window in isolation (the two tests above each only ever have one non-trivial
    window)."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    retried_then_ok = [_item(base, "retrywin hello")]
    gave_up = [_item(base + timedelta(hours=10), "givewin hello")]
    clean = [_item(base + timedelta(hours=20), "cleanwin hello")]

    model = _ScriptedModel(
        [
            (lambda p: "retrywin" in p, [_err("bad"), _ok("Recovered fact")]),
            (lambda p: "givewin" in p, [_err("bad 1"), _err("bad 2"), _err("bad 3")]),
            (lambda p: "cleanwin" in p, [_ok("Clean fact")]),
        ]
    )

    result = extract_batch(retried_then_ok + gave_up + clean, model=model)

    assert len(result.facts) == 2  # the give-up window contributes nothing
    statements = {fact.statement for fact, _prov in result.facts}
    assert statements == {"Recovered fact", "Clean fact"}
    # retried_then_ok: 2 calls -> 1 retry, not escalated (ESCALATE_AFTER=2 only bites on the
    # 3rd call). gave_up: 3 calls -> 2 retries, escalated on its final call, and gave up.
    # clean: 1 call -> 0 retries, not escalated. Summed: retries = 1 + 2 + 0 = 3.
    assert result.retries == 3
    assert result.give_ups == 1
    assert result.escalations == 1


def test_hard_model_error_gives_up_that_window_without_aborting_the_batch():
    """Regression (fix-wave-1 item 7, MUST-FIX #6 of the code-quality review): a
    ConnectionError from .invoke() used to propagate straight out of extract_node, uncaught
    -- LangGraph's own RetryPolicy(max_attempts=3) retries the node a few times but
    re-raises once exhausted, and nothing above the node caught it, so the WHOLE batch
    graph's .invoke() call raised and every sibling window's already-extracted facts were
    lost too. Two windows: one whose model call always raises ConnectionError, one that
    succeeds normally -- the batch must not raise, the failing window must reach the same
    give-up-with-notes-marker terminal state validation failures already do, and the
    succeeding window's fact must still come back."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    failing = [_item(base, "hardfail hello")]
    succeeding = [_item(base + timedelta(hours=10), "hardok hello")]

    class _MixedModel:
        def __init__(self):
            self.calls: list[str] = []

        def invoke(self, prompt: str):
            self.calls.append(prompt)
            if "hardfail" in prompt:
                raise ConnectionError("simulated network outage")
            return _ok("A fine fact despite the sibling's outage")

    model = _MixedModel()

    raw_results = run_batch(failing + succeeding, model=model)  # must not raise

    by_provenance = {tuple(r["provenance"]): r for r in raw_results}
    failing_result = by_provenance[(failing[0].id,)]
    succeeding_result = by_provenance[(succeeding[0].id,)]

    assert failing_result["facts"] is None
    assert failing_result["attempt"] == 3  # exhausted the same MAX_ATTEMPTS as validation failures
    assert "ConnectionError" in failing_result["notes"]
    assert len(model.calls) == 4  # 3 attempts on the failing window + 1 on the succeeding one

    assert succeeding_result["facts"] is not None
    assert succeeding_result["facts"][0]["statement"] == "A fine fact despite the sibling's outage"

    # extract_batch's flattening wrapper over the same run must also stay upright end-to-end.
    flattened = extract_batch(failing + succeeding, model=model)
    assert len(flattened.facts) == 1
    assert flattened.facts[0][0].statement == "A fine fact despite the sibling's outage"
    assert flattened.give_ups == 1  # the hard-error window, same terminal state as validation give-ups


@pytest.mark.llm
def test_live_smoke_extracts_at_least_one_plausible_fact():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live smoke")

    from locket.adapters.whatsapp import parse_whatsapp

    demo = Path(__file__).parent.parent / "demo_corpus" / "whatsapp" / "team.txt"
    items = list(parse_whatsapp(demo, thread="team"))[:16]  # one window's worth

    result = extract_batch(items)

    assert len(result.facts) >= 1
    for fact, provenance in result.facts:
        assert fact.kind in set(FactKind)
        assert provenance
        assert all(p in {i.id for i in items} for p in provenance)


def _ollama_model_available(model: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and model.split(":")[0] in result.stdout


@pytest.mark.vision
def test_live_local_backend_extracts_a_validated_result_from_one_window():
    """Same corpus window as the Anthropic live smoke above, but forced onto the local
    Ollama backend via locket.llm -- the keyless path this task adds. Marked `vision`
    (not `llm`) because it needs the local Ollama server/model, not ANTHROPIC_API_KEY."""
    from locket.config import Settings
    from locket.extraction.schemas import ExtractionResult
    from locket.llm import DEFAULT_LOCAL_TEXT_MODEL, get_chat_model

    if not _ollama_model_available(DEFAULT_LOCAL_TEXT_MODEL):
        pytest.skip(f"ollama server/model {DEFAULT_LOCAL_TEXT_MODEL!r} not available — skipping live local smoke")

    from locket.adapters.whatsapp import parse_whatsapp

    demo = Path(__file__).parent.parent / "demo_corpus" / "whatsapp" / "team.txt"
    items = list(parse_whatsapp(demo, thread="team"))[:16]  # one window's worth

    settings = Settings(corpus_dir=None, db_url="postgresql://x/y", anthropic_api_key=None, ollama_model="qwen3-vl:8b")
    model = get_chat_model("extraction_default", settings).with_structured_output(
        ExtractionResult, method="json_schema", include_raw=True
    )

    result = extract_batch(items, model=model)

    assert len(result.facts) >= 1
    for fact, provenance in result.facts:
        assert fact.kind in set(FactKind)
        assert provenance
        assert all(p in {i.id for i in items} for p in provenance)
