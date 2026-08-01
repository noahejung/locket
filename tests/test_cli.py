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
from locket.cli import main
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
# argparse wiring
# ---------------------------------------------------------------------------


def test_unknown_command_raises_systemexit():
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])


def test_pipeline_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main(["pipeline"])


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
