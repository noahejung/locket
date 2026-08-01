"""Unit tests for the per-run JSONL capture (src/locket/stats.py) -- tmp_path only, no DB,
no LLM. DB-aggregate reads that back `locket stats`'s other numbers live on Store and are
tested in tests/test_store.py (db-marked); this file covers only the local-disk JSONL
bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

from locket.stats import RunRecord, append_run_record, read_last_run_record


def _record(**overrides) -> RunRecord:
    base = dict(
        timestamp="2026-07-31T12:00:00+00:00",
        backend="ollama",
        model="gemma3:12b",
        windows_processed=3,
        windows_skipped=1,
        windows_skipped_extracted=1,
        windows_skipped_gave_up=0,
        facts_added=5,
        dedup_hits=2,
        retries=1,
        give_ups=0,
        escalations=0,
        wall_seconds=12.5,
    )
    base.update(overrides)
    return RunRecord(**base)


def test_read_last_run_record_returns_none_when_file_does_not_exist(tmp_path):
    assert read_last_run_record(path=tmp_path / "runs.local.jsonl") is None


def test_append_then_read_round_trips_every_field(tmp_path):
    path = tmp_path / "runs.local.jsonl"
    record = _record()

    append_run_record(record, path=path)
    result = read_last_run_record(path=path)

    assert result == record.as_dict()


def test_append_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "evals" / "runs.local.jsonl"

    append_run_record(_record(), path=path)

    assert path.exists()


def test_append_is_additive_not_overwriting(tmp_path):
    path = tmp_path / "runs.local.jsonl"

    append_run_record(_record(facts_added=1), path=path)
    append_run_record(_record(facts_added=2), path=path)
    append_run_record(_record(facts_added=3), path=path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["facts_added"] for line in lines] == [1, 2, 3]


def test_read_last_run_record_returns_the_most_recently_appended_line(tmp_path):
    path = tmp_path / "runs.local.jsonl"
    append_run_record(_record(facts_added=1), path=path)
    append_run_record(_record(facts_added=2), path=path)

    result = read_last_run_record(path=path)

    assert result["facts_added"] == 2


def test_read_last_run_record_tolerates_a_trailing_blank_line(tmp_path):
    path = tmp_path / "runs.local.jsonl"
    append_run_record(_record(facts_added=7), path=path)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")  # trailing newline noise -- must not become "the last line"

    result = read_last_run_record(path=path)

    assert result["facts_added"] == 7


def test_default_runs_log_path_is_under_evals_and_gitignore_matches_it():
    """Sanity-check the default path matches the repo's existing `*.local.*` gitignore
    pattern -- a per-run capture file that accidentally got committed would leak corpus
    size/shape."""
    import fnmatch

    from locket.stats import DEFAULT_RUNS_LOG_PATH

    assert DEFAULT_RUNS_LOG_PATH == Path("evals/runs.local.jsonl")
    assert fnmatch.fnmatch(DEFAULT_RUNS_LOG_PATH.name, "*.local.*")
