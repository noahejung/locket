"""CLI tests against the real dockerized Postgres (Store injected via `main(..., store=...)`,
bypassing the real Settings.load()-constructed connection so tests control isolation the same
way every other db-marked test suite in this repo does).

The genuinely new logic here is source discovery (_ingest_source/_discover_corpus_sources)
and pipeline orchestration (_run_pipeline) -- everything else (eval/profile/resolve
subcommands) is a thin argument-parsing wrapper over already-unit-tested library functions
(evals/extraction_eval.py, evals/rag_eval.py, profile.py, resolution.py), so those get a
lighter "does it run and print sane output" smoke rather than re-deriving their own test
suites here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.cli import build_parser, main
from locket.extraction.schemas import ExtractedFact, ExtractionResult
from locket.models import FactKind
from locket.store import Store

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")
DEMO_CORPUS = Path(__file__).parent.parent / "demo_corpus"

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


class _AlwaysOneFactModel:
    """Test double for the extraction graph's structured-output runnable -- returns one
    plausible fact for any window, regardless of prompt content. Mirrors
    tests/test_extraction_graph.py's _ScriptedModel but with a single catch-all rule, since
    CLI-level pipeline tests care about orchestration wiring, not per-window content."""

    def __init__(self, kind: FactKind = FactKind.event, statement: str = "Something happened"):
        self._kind = kind
        self._statement = statement
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> dict:
        self.calls.append(prompt)
        fact = ExtractedFact(kind=self._kind, statement=self._statement, subjects=["Noah"], confidence=0.9)
        return {"raw": None, "parsed": ExtractionResult(facts=[fact]), "parsing_error": None}


class _EchoProfileModel:
    """Echoes each input fact's own statement back -- mirrors tests/test_profile.py's
    _EchoRenderModel."""

    def invoke(self, prompt: str):
        from locket.profile import SectionRendering

        lines = [line.split(". ", 1)[1] for line in prompt.splitlines() if line[:1].isdigit() and ". " in line]
        return SectionRendering(sentences=lines)


# ---------------------------------------------------------------------------
# ingest -- the new source-discovery logic
# ---------------------------------------------------------------------------


def test_ingest_whatsapp_count_matches_adapter(store):
    team_path = DEMO_CORPUS / "whatsapp" / "team.txt"
    expected = len(list(parse_whatsapp(team_path, thread="team")))

    exit_code = main(["ingest", str(team_path)], store=store)

    assert exit_code == 0
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_items")
        assert cur.fetchone()[0] == expected


def test_ingest_whatsapp_then_sms_counts_accumulate(store):
    team_path = DEMO_CORPUS / "whatsapp" / "team.txt"
    sms_path = DEMO_CORPUS / "sms" / "backup.xml"
    expected_wa = len(list(parse_whatsapp(team_path, thread="team")))
    expected_sms = len(list(parse_sms_xml(sms_path)))

    main(["ingest", str(team_path)], store=store)
    main(["ingest", str(sms_path)], store=store)

    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_items")
        assert cur.fetchone()[0] == expected_wa + expected_sms


def test_ingest_reingest_is_idempotent(store):
    team_path = DEMO_CORPUS / "whatsapp" / "team.txt"
    main(["ingest", str(team_path)], store=store)
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_items")
        first_count = cur.fetchone()[0]

    main(["ingest", str(team_path)], store=store)
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_items")
        assert cur.fetchone()[0] == first_count


def test_ingest_unknown_extension_raises(store, tmp_path):
    bogus = tmp_path / "notes.md"
    bogus.write_text("hello")
    with pytest.raises(ValueError, match="don't know how to ingest"):
        main(["ingest", str(bogus)], store=store)


def test_ingest_prints_a_warning_for_unparseable_whatsapp_lines(store, tmp_path, capsys):
    """Fix-wave-1 item 5a: a whatsapp .txt with unrecognized-format lines must print a
    WARNING to stderr naming the count -- not silently ingest zero items with no trace."""
    bogus = tmp_path / "weird_locale.txt"
    bogus.write_text("totally unrecognized line one\ntotally unrecognized line two\n", encoding="utf-8")

    exit_code = main(["ingest", str(bogus)], store=store)

    assert exit_code == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "2" in err


# ---------------------------------------------------------------------------
# pipeline run -- the new orchestration logic, --skip-vision + stubbed extraction/profile
# ---------------------------------------------------------------------------


def _write_mini_whatsapp_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    wa_dir = corpus_dir / "whatsapp"
    wa_dir.mkdir(parents=True)
    (wa_dir / "team.txt").write_text(
        "1/15/25, 10:32 AM - John: Can we push the deadline to Friday?\n"
        "1/15/25, 10:33 AM - Sarah: sounds good\n",
        encoding="utf-8",
    )
    return corpus_dir


def test_pipeline_run_skip_vision_with_stubbed_llm_populates_facts(store, tmp_path):
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)
    extraction_model = _AlwaysOneFactModel()

    exit_code = main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=extraction_model,
        profile_model=_EchoProfileModel(),
    )

    assert exit_code == 0
    assert extraction_model.calls  # the stub was actually invoked
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_items")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM facts")
        assert cur.fetchone()[0] >= 1
        cur.execute("SELECT count(*) FROM entities")
        assert cur.fetchone()[0] >= 1  # "Noah" subject resolved/created
        cur.execute("SELECT count(*) FROM profiles")
        assert cur.fetchone()[0] == 1  # synthesize() ran at the end


def test_pipeline_run_second_invocation_skips_already_extracted_windows(store, tmp_path):
    """Fix-wave-1 item 8b: `pipeline run` re-extracted every window on every call, even
    unchanged ones -- a real re-bill risk against a real corpus + a real API key. A second
    run over the SAME corpus must make zero NEW model calls (proving the skip happens
    BEFORE the model is invoked, not just that duplicate facts get deduped after the fact)
    and create zero new facts."""
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)
    extraction_model = _AlwaysOneFactModel()

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=extraction_model,
        profile_model=_EchoProfileModel(),
    )
    calls_after_first_run = len(extraction_model.calls)
    assert calls_after_first_run >= 1
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM facts")
        facts_after_first_run = cur.fetchone()[0]

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=extraction_model,
        profile_model=_EchoProfileModel(),
    )

    assert len(extraction_model.calls) == calls_after_first_run  # zero new model calls
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM facts")
        assert cur.fetchone()[0] == facts_after_first_run  # zero new facts
        cur.execute("SELECT count(*) FROM extracted_windows")
        assert cur.fetchone()[0] >= 1


def test_pipeline_run_prints_json_summary(store, tmp_path, capsys):
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["raw_items_inserted"] == 2
    assert summary["facts_created"] >= 1
    assert summary["sources"] == 1


def test_pipeline_run_facts_carry_entity_ids_after_resolution(store, tmp_path):
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(statement="Noah and Sarah made plans"),
        profile_model=_EchoProfileModel(),
    )

    with store._conn.cursor() as cur:
        cur.execute("SELECT entity_ids FROM facts LIMIT 1")
        entity_ids = cur.fetchone()[0]
    assert entity_ids  # non-empty -- "Noah" resolved to a real entity


# ---------------------------------------------------------------------------
# resolve -- thin wrapper over resolution.pending_confirmations
# ---------------------------------------------------------------------------


def test_resolve_no_pending_prints_friendly_message(store, capsys):
    exit_code = main(["resolve"], store=store)
    assert exit_code == 0
    assert "no pending" in capsys.readouterr().out.lower()


def test_resolve_yes_confirms_every_pending_proposal(store, capsys):
    from locket.embeddings import get_backend

    backend = get_backend()
    alex1 = store.upsert_entity("Alex Rivera", "person", backend.embed_docs(["Alex Rivera"])[0])
    alex2 = store.upsert_entity("Alex Chen", "person", backend.embed_docs(["Alex Chen"])[0])
    store.add_merge_proposal("Alex", alex1, evidence="test", score=0.5)
    store.add_merge_proposal("Alex", alex2, evidence="test", score=0.5)

    exit_code = main(["resolve", "--yes"], store=store)

    assert exit_code == 0
    with store._conn.cursor() as cur:
        cur.execute("SELECT status FROM merge_proposals")
        statuses = [r[0] for r in cur.fetchall()]
    assert statuses == ["confirmed", "confirmed"]


# ---------------------------------------------------------------------------
# profile build -- thin wrapper over profile.synthesize
# ---------------------------------------------------------------------------


def test_profile_build_prints_markdown_and_persists(store, capsys):
    from locket.embeddings import get_backend
    from locket.models import Fact

    backend = get_backend()
    fact = Fact(kind=FactKind.habit, statement="Noah runs every morning", confidence=0.9, provenance=["r1"])
    store.add_fact(fact, backend.embed_docs([fact.statement])[0])

    exit_code = main(["profile", "build"], store=store, profile_model=_EchoProfileModel())

    assert exit_code == 0
    assert "## Habits" in capsys.readouterr().out
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM profiles")
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# eval extraction -- thin wrapper, stubbed model, real demo_corpus + real gold file
# ---------------------------------------------------------------------------


def test_eval_extraction_json_output_has_expected_shape(store, capsys):
    """Runs the real evals/extraction_eval.run_extraction_pipeline over the FULL demo
    corpus with a stub model that returns one fact per window (regardless of window
    content) -- proves the CLI wires args -> run_extraction_pipeline -> score -> JSON
    correctly, without needing ANTHROPIC_API_KEY. Extraction quality itself is already
    covered by evals/BASELINE.md's blocked-on-key live run."""
    model = _AlwaysOneFactModel(kind=FactKind.event, statement="Something happened")

    exit_code = main(
        ["eval", "extraction", "--json"],
        store=store,
        extraction_model=model,
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"precision", "recall", "f1", "by_kind", "misses", "spurious"}
    assert isinstance(report["precision"], float)
    assert model.calls  # the stub was actually invoked, not bypassed


# ---------------------------------------------------------------------------
# serve -- must close its Store on every exit path (fix-wave-2 item 10)
# ---------------------------------------------------------------------------


def test_serve_closes_store_even_if_mcp_run_raises(monkeypatch):
    """`serve` is main()'s one command that bypasses the shared owns_store try/finally
    below (it never takes an injectable `store=` -- a real serve blocks on stdio, so tests
    can't pass one in) -- before this fix, _cmd_serve had no try/finally of its own either,
    so its Store/connection leaked on every exit path, including this exception one."""
    closed = []

    class _FakeStore:
        def close(self):
            closed.append(True)

    class _FakeMCP:
        def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("locket.cli.Store", lambda url: _FakeStore())
    monkeypatch.setattr("locket.mcp_server.build_server", lambda store: _FakeMCP())

    with pytest.raises(RuntimeError, match="boom"):
        main(["serve"])

    assert closed == [True]


# ---------------------------------------------------------------------------
# serve-ui -- same close-on-every-exit-path contract as serve, above (Task: phone chat UI)
# ---------------------------------------------------------------------------


def test_serve_ui_closes_store_even_if_uvicorn_run_raises(monkeypatch):
    """Mirrors test_serve_closes_store_even_if_mcp_run_raises above -- serve-ui has the
    identical no-injectable-store-seam / blocks-for-the-process-lifetime shape as serve
    (uvicorn.run(...) instead of mcp.run()), so it needs the same own try/finally."""
    closed = []

    class _FakeStore:
        def close(self):
            closed.append(True)

    def _raising_run(app, host, port):
        raise RuntimeError("boom")

    monkeypatch.setattr("locket.cli.Store", lambda url: _FakeStore())
    monkeypatch.setattr("locket.webui.create_app", lambda store: object())
    monkeypatch.setattr("uvicorn.run", _raising_run)

    with pytest.raises(RuntimeError, match="boom"):
        main(["serve-ui"])

    assert closed == [True]


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def test_unknown_command_raises_systemexit():
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])


def test_pipeline_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main(["pipeline"])


def test_serve_ui_host_and_port_default_to_loopback_and_8765():
    """Security posture (non-negotiable per the dispatch, mirrors the fix-wave-1 docker-
    compose port-binding fix): --host must default to 127.0.0.1, never 0.0.0.0 -- binding
    0.0.0.0 would expose answer_question's citable "profile of you" to the whole LAN, not
    just the intended phone-over-Tailscale path."""
    parser = build_parser()
    args = parser.parse_args(["serve-ui"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_serve_ui_host_is_overridable():
    parser = build_parser()
    args = parser.parse_args(["serve-ui", "--host", "100.102.116.112", "--port", "9000"])
    assert args.host == "100.102.116.112"
    assert args.port == 9000


def test_pipeline_run_forces_anthropic_backend_without_key_fails_clearly(store, tmp_path, monkeypatch, capsys):
    """The guard now only fires for an EXPLICITLY forced anthropic backend with no key --
    since locket.llm defaults to the (keyless) ollama backend when no key is present, a
    bare no-key run no longer refuses to proceed (see the next test)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LOCKET_LLM_BACKEND", "anthropic")
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    exit_code = main(["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)], store=store)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "ANTHROPIC_API_KEY" in err
    assert "LOCKET_LLM_BACKEND" in err


def test_pipeline_run_without_key_defaults_to_ollama_backend_and_proceeds(store, tmp_path, monkeypatch):
    """Keyless default path (this task): no ANTHROPIC_API_KEY, no LOCKET_LLM_BACKEND
    override -- locket.llm.resolve_backend picks "ollama", so the CLI's key guard does not
    fire at all. Stubs extraction_model/profile_model to stay network-free here; the real
    local-Ollama call path is covered live by
    test_extraction_graph.py::test_live_local_backend_extracts_a_validated_result_from_one_window
    (@pytest.mark.vision) and evals/BASELINE.md's local-backend pipeline run."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LOCKET_LLM_BACKEND", raising=False)
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    exit_code = main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )

    assert exit_code == 0


# ---------------------------------------------------------------------------
# pipeline run -- per-run JSONL capture (locket stats, metrics.md §1/§5)
# ---------------------------------------------------------------------------


class _RetryTwiceThenSucceedModel:
    """Fails validation twice, then succeeds on the 3rd call -- exercises the counter-
    surfacing path end-to-end through `pipeline run`: 2 retries, escalated on the 3rd
    (final) call (ESCALATE_AFTER=2), zero give-ups. _write_mini_whatsapp_corpus's two
    messages land in exactly one window, so every call here is for that same window."""

    def __init__(self):
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> dict:
        self.calls.append(prompt)
        if len(self.calls) < 3:
            return {"raw": None, "parsed": None, "parsing_error": ValueError("bad")}
        fact = ExtractedFact(kind=FactKind.event, statement="Something happened", subjects=["Noah"], confidence=0.9)
        return {"raw": None, "parsed": ExtractionResult(facts=[fact]), "parsing_error": None}


def test_pipeline_run_appends_a_jsonl_run_record(store, tmp_path, monkeypatch):
    """`evals/runs.local.jsonl` is resolved relative to the CWD at runtime (same convention
    as the CLI's other relative defaults, e.g. `demo_corpus`) -- chdir into tmp_path so the
    default path lands there instead of the real repo's evals/ directory."""
    monkeypatch.chdir(tmp_path)
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    exit_code = main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )

    assert exit_code == 0
    runs_log = tmp_path / "evals" / "runs.local.jsonl"
    assert runs_log.exists()
    lines = runs_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["windows_processed"] == 1
    assert record["windows_skipped"] == 0
    assert record["facts_added"] >= 1
    assert record["dedup_hits"] == 0
    assert record["wall_seconds"] >= 0
    assert record["backend"] in {"anthropic", "ollama"}
    assert record["model"]
    assert "timestamp" in record


def test_pipeline_run_second_invocation_appends_a_second_jsonl_line_with_zero_new_work(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )
    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )

    runs_log = tmp_path / "evals" / "runs.local.jsonl"
    lines = runs_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # append-only -- the first run's line is still there
    second = json.loads(lines[1])
    assert second["windows_processed"] == 0  # the only window was already extracted
    assert second["windows_skipped"] == 1
    assert second["facts_added"] == 0


def test_pipeline_run_json_summary_and_run_record_surface_retry_and_escalation_counters(
    store, tmp_path, monkeypatch, capsys
):
    """Counter-surfacing test (this task's explicit test requirement): a model that fails
    twice before succeeding must show up as retries=2/escalations=1/give_ups=0 in BOTH the
    printed JSON summary and the appended run record -- proving the counters actually
    travel extract_batch -> extract_and_persist -> _run_pipeline -> the JSONL line, not
    just that extract_batch computes them correctly in isolation (already covered by
    test_extraction_graph.py)."""
    monkeypatch.chdir(tmp_path)
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)
    model = _RetryTwiceThenSucceedModel()

    exit_code = main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=model,
        profile_model=_EchoProfileModel(),
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["retries"] == 2
    assert summary["give_ups"] == 0
    assert summary["escalations"] == 1
    assert len(model.calls) == 3

    runs_log = tmp_path / "evals" / "runs.local.jsonl"
    record = json.loads(runs_log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["retries"] == 2
    assert record["give_ups"] == 0
    assert record["escalations"] == 1


# ---------------------------------------------------------------------------
# stats -- metrics.md §1/§5's DB aggregates + the last captured run record
# ---------------------------------------------------------------------------


def test_stats_json_reports_every_aggregate(store, tmp_path, monkeypatch, capsys):
    from locket.embeddings import get_backend
    from locket.models import Fact

    monkeypatch.chdir(tmp_path)  # no evals/runs.local.jsonl yet in this cwd -> last_run is None

    backend = get_backend()
    raw = list(parse_whatsapp(DEMO_CORPUS / "whatsapp" / "team.txt", thread="team"))[:2]
    store.add_raw_items(raw)

    fact = Fact(kind=FactKind.habit, statement="Noah runs every morning", confidence=0.8, provenance=[raw[0].id])
    fact_id = store.add_fact(fact, backend.embed_docs([fact.statement])[0])
    store.update_fact(fact_id, confidence=0.9)  # ADD + UPDATE fact_history rows

    john = store.upsert_entity("John", "person", backend.embed_docs(["John"])[0])
    sarah = store.upsert_entity("Sarah", "person", backend.embed_docs(["Sarah"])[0])
    store.add_merge_proposal("J", john, evidence="e", score=0.5)

    exit_code = main(["stats", "--json"], store=store)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw_items_by_source"] == {"whatsapp": 2}
    assert payload["facts"]["total"] == 1
    assert payload["facts"]["mean_confidence"] == pytest.approx(0.9)
    assert payload["facts"]["by_kind"]["habit"]["count"] == 1
    assert payload["entities"] == 2
    assert payload["confirm_queue"]["depth"] == 1
    assert payload["confirm_queue"]["oldest_age_seconds"] >= 0
    assert payload["fact_history_events"]["ADD"] == 1
    assert payload["fact_history_events"]["UPDATE"] == 1
    assert payload["last_run"] is None
    assert sarah  # sanity: fixture actually created the entity


def test_stats_json_confirm_queue_is_null_when_empty(store, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["stats", "--json"], store=store)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["confirm_queue"] == {"depth": 0, "oldest_created_at": None, "oldest_age_seconds": None}
    assert payload["fact_history_events"] == {}


def test_stats_human_readable_output_when_store_is_empty(store, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["stats"], store=store)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "raw items by source" in out
    assert "facts: 0 total" in out
    assert "entities: 0" in out
    assert "confirm queue: empty" in out
    assert "last pipeline run: none recorded yet" in out


def test_stats_reports_the_last_pipeline_run_record(store, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    corpus_dir = _write_mini_whatsapp_corpus(tmp_path)

    main(
        ["pipeline", "run", "--skip-vision", "--corpus-dir", str(corpus_dir)],
        store=store,
        extraction_model=_AlwaysOneFactModel(),
        profile_model=_EchoProfileModel(),
    )
    capsys.readouterr()  # discard pipeline run's own stdout

    exit_code = main(["stats", "--json"], store=store)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["last_run"] is not None
    assert payload["last_run"]["windows_processed"] == 1
