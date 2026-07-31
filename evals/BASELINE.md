# locket eval baselines

This file is the resume artifact: "improved extraction F1 from X to Y" needs X written down
on day one. Every number below states how it was produced and when.

## Vision-LLM tail latency (Task 13)

Measured live against `demo_corpus/photos/` on this machine (CPU-only, no GPU), local
Ollama `qwen3-vl:8b`, `options={"temperature": 0}`, schema-constrained decoding via
`format=PhotoFacts.model_json_schema()`. The plan's own 10-40s/image figure was explicitly
flagged as an unverified estimate inferred from adjacent benchmarks — this is the real,
measured replacement.

| image | seconds |
|---|---|
| portrait_face_01_00.jpg (cold model load) | 141.5 |
| portrait_face_01_00.jpg (warm, during pytest -m vision run) | 86.3 |
| portrait_face_02_00.jpg | 204.5 |
| portrait_face_03_00.jpg | 94.4 |
| screenshot_02.png | 168.7 |
| receipt_02.png | 117.4 |

**Mean ≈ 135s/image, range 86-205s.** No correlation observed between image content
(portrait vs. screenshot vs. receipt) and latency — the variance looks like host-level
noise (thermal throttling / other processes), not content-dependent. This number sizes
`select_for_vision`'s `cap` parameter and any batch/overnight scheduling: at ~135s/image, a
300-image cap (the plan's CLI example) is ~11 hours serial on this hardware — batch runs
belong overnight or on a machine with more consistent throughput, not in an interactive
session.

## Extraction eval (Task 15) — BASELINE PENDING API KEY

`ANTHROPIC_API_KEY` was absent from the environment (no `.env` file) throughout Weeks 2-3
of this build. The matcher (`evals/extraction_eval.score`) and the pipeline runner
(`evals/extraction_eval.run_extraction_pipeline`) are implemented and unit-tested against
fixtures + a deterministic fake embedder (`tests/test_extraction_eval.py`, 9 tests, all
green). The live run against the real Claude API (`claude-haiku-4-5` extraction, scored
against the full 60-fact `evals/gold/persona_gold.yaml`) is written as
`tests/test_extraction_eval.py::test_live_extraction_pipeline_scores_against_full_gold_set`,
marked `@pytest.mark.llm`, and self-skips cleanly without the key.

**Run once the key exists:**

```bash
# ensure the db container is up (docker compose up -d db) and .env/ANTHROPIC_API_KEY is set
uv run pytest -m llm tests/test_extraction_eval.py -q
```

To capture precision/recall/F1 by kind for this file (not just the pass/fail the pytest
assertion currently checks — the test intentionally only asserts non-negative numbers so
it can't fail on a bad-but-real baseline):

```bash
uv run python -c "
from pathlib import Path
from locket.store import Store
from evals.extraction_eval import run_extraction_pipeline, load_gold, score

store = Store('postgresql://locket:locket@localhost:5432/locket')
with store._conn.cursor() as cur:
    cur.execute('TRUNCATE raw_items, facts, entities, fact_history, merge_proposals RESTART IDENTITY CASCADE')
store._conn.commit()

extracted = run_extraction_pipeline(store, Path('demo_corpus'))
gold = load_gold(Path('evals/gold/persona_gold.yaml'))
report = score(extracted, gold)
print('precision', report.precision)
print('recall', report.recall)
print('f1', report.f1)
for kind, kb in sorted(report.by_kind.items()):
    print(kind, kb)
print('MISSES:')
for m in report.misses:
    print(' -', m)
print('SPURIOUS:')
for s in report.spurious:
    print(' -', s)
"
```

Then paste the P/R/F1-by-kind table and the misses/spurious lists into this file, dated,
below this section — do not overwrite the pending-status note above until that's done.

Once real numbers land here, `tests/test_extraction_eval.py`'s live test should gain an
actual regression-threshold assertion (it currently only asserts non-negative, deliberately
— see the module: "Do not fabricate baseline numbers").

## RAG eval (Task 16) — BASELINE PENDING API KEY

Same blocker: no `ANTHROPIC_API_KEY` in this environment. The harness
(`evals/rag_eval.run_rag_eval`) is implemented and unit-tested with everything stubbed —
retrieval, answer synthesis, and all three ragas metrics — in `tests/test_rag_eval.py` (4
tests, all green, verifying the harness calls each metric with the exact kwargs its real
`ascore()` signature takes: `faithfulness(user_input, response, retrieved_contexts)`,
`answer_relevancy(user_input, response)` — no `reference` — and
`context_precision(user_input, reference, retrieved_contexts)`).

The live run (real ragas judge = `claude-haiku-4-5` via the plain `anthropic` SDK, real
local `sentence-transformers/all-MiniLM-L6-v2` ragas embeddings, all 25
`evals/questions.yaml` questions) is
`tests/test_rag_eval.py::test_live_rag_eval_meets_thresholds`, marked `@pytest.mark.llm`,
self-skips without the key, and — unlike the extraction eval's live test — DOES assert the
plan's starting thresholds directly (`FAITHFULNESS_THRESHOLD=0.85`,
`ANSWER_RELEVANCY_THRESHOLD=0.80`, `CONTEXT_PRECISION_THRESHOLD=0.70`), since those are the
plan's own explicit starting values, not a number this session invented.

**Run once the key exists:**

```bash
# ensure the db container is up (docker compose up -d db) and .env/ANTHROPIC_API_KEY is set
uv run pytest -m llm tests/test_rag_eval.py -q
```

For per-question detail (which question failed which metric, not just the pass/fail the
pytest assertions give):

```bash
uv run python -c "
from pathlib import Path
from locket.store import Store
from evals.extraction_eval import run_extraction_pipeline
from evals.rag_eval import load_questions, run_rag_eval

store = Store('postgresql://locket:locket@localhost:5432/locket')
with store._conn.cursor() as cur:
    cur.execute('TRUNCATE raw_items, facts, entities, fact_history, merge_proposals RESTART IDENTITY CASCADE')
store._conn.commit()

run_extraction_pipeline(store, Path('demo_corpus'))
questions = load_questions(Path('evals/questions.yaml'))
result = run_rag_eval(store, questions)
print('faithfulness', result.faithfulness)
print('answer_relevancy', result.answer_relevancy)
print('context_precision', result.context_precision)
for q in result.per_question:
    print(q.question, '->', q.faithfulness, q.answer_relevancy, q.context_precision)
"
```

**CI:** `.github/workflows/eval.yml` runs both live eval suites (`pytest -m llm
tests/test_extraction_eval.py tests/test_rag_eval.py`) on `workflow_dispatch` and a nightly
cron (07:00 UTC), against a pgvector service container, needing the `ANTHROPIC_API_KEY`
repo secret. It is a SEPARATE workflow file from `.github/workflows/ci.yml` — `ci.yml`'s
`test`/`db` jobs and their `on: [push, pull_request]` trigger are untouched (verified: `git
diff --stat .github/workflows/ci.yml` against this session's start is empty). Add the
`ANTHROPIC_API_KEY` secret in the repo settings before expecting a real signal from this
workflow — until then, both eval tests skip cleanly (not fail) inside it.
