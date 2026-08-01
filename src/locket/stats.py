"""`locket stats` support (metrics.md §1/§5): the per-run JSONL capture file `pipeline run`
appends to, plus the small dataclasses `cli.py`'s stats command renders. DB aggregate reads
themselves live on Store (store.py stays the only module that talks to Postgres); this
module only owns the JSONL file, which is local-disk bookkeeping, not a DB concern.

`evals/runs.local.jsonl` is gitignored via the repo's existing `*.local.*` pattern (same
convention as `evals/gold/real_gold.local.yaml`, `evals/journal.local.md`) -- per-run
numbers can reveal corpus size/shape, so this file stays off-repo like every other
`.local.` artifact, even though (unlike those two) it holds no real personal content itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_RUNS_LOG_PATH = Path("evals/runs.local.jsonl")


@dataclass
class RunRecord:
    """One `pipeline run` invocation's numbers -- exactly metrics.md §1's per-run column
    set plus the timing/backend context needed to make a run reproducible-in-spirit later.
    A plain dict would work too, but an explicit dataclass keeps the JSONL schema
    self-documenting at the one place it's constructed (cli.py's `_cmd_pipeline_run`).

    `windows_skipped_extracted`/`windows_skipped_gave_up` (fix-wave-3 follow-up to the
    2026-08-01 catch-up review's LOW finding): `windows_skipped` alone conflated windows
    skipped because they already succeeded with windows skipped because they'd already
    given up -- indistinguishable in one number, even though only the latter is retryable
    via `pipeline retry-given-up`/`pipeline run --retry-failed`. `windows_skipped` is kept
    as their sum, unchanged, for backward compatibility with any code/tooling reading the
    JSONL shape (read_last_run_record returns a loose dict, not a rehydrated RunRecord, so
    older lines missing the two new fields still read back fine)."""

    timestamp: str  # ISO 8601 UTC, when the run finished
    backend: str  # "anthropic" | "ollama" -- locket.llm.resolve_backend's output
    model: str  # concrete model id/name -- locket.llm.model_name's output
    windows_processed: int
    windows_skipped: int  # == windows_skipped_extracted + windows_skipped_gave_up
    windows_skipped_extracted: int
    windows_skipped_gave_up: int
    facts_added: int  # genuinely NEW fact rows (store's total-count delta across the run)
    dedup_hits: int  # extracted candidates that hit add_fact's statement-hash ON CONFLICT
    retries: int
    give_ups: int
    escalations: int
    wall_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_run_record(record: RunRecord, *, path: Path = DEFAULT_RUNS_LOG_PATH) -> None:
    """Append one JSON line. Creates `path`'s parent dir if missing (a fresh checkout has no
    `evals/runs.local.jsonl` until the first `pipeline run`) -- never truncates or rewrites
    prior lines, so the file is an append-only history of every run, diffable over time
    (metrics.md §5's stated goal)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.as_dict()) + "\n")


def read_last_run_record(*, path: Path = DEFAULT_RUNS_LOG_PATH) -> dict[str, Any] | None:
    """The most recently appended run record, or None if the file doesn't exist yet or is
    empty. Returns the raw parsed dict (not a rehydrated RunRecord) -- `locket stats` only
    ever prints/serializes it, never mutates or type-checks it further, and staying loose
    here means an older JSONL line missing a field this module later adds still reads back
    fine instead of raising on rehydration."""
    if not path.exists():
        return None
    last_line: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return None
    return json.loads(last_line)


__all__ = ["DEFAULT_RUNS_LOG_PATH", "RunRecord", "append_run_record", "read_last_run_record"]
