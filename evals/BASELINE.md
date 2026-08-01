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

store = Store('postgresql://locket:locket@127.0.0.1:5432/locket')
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

store = Store('postgresql://locket:locket@127.0.0.1:5432/locket')
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

## End-to-end CLI pipeline run over demo_corpus (Task 19) — INGEST VERIFIED, EXTRACTION BLOCKED ON API KEY

`locket ingest` was run for real (real terminal invocation of `python -m locket.cli`, not a
pytest stub) against every source in `demo_corpus/` on 2026-07-30, against the same running
`locket-db-1` container Weeks 1-3 used:

```bash
uv run python -m locket.cli ingest demo_corpus/whatsapp/team.txt      # 190 raw_items
uv run python -m locket.cli ingest demo_corpus/whatsapp/sarah.txt     # 144 new (1 pre-existing overlap)
uv run python -m locket.cli ingest demo_corpus/sms/backup.xml         # 45 new
uv run python -m locket.cli ingest demo_corpus/instagram/inbox/kathryn  # 90 new
uv run python -m locket.cli ingest demo_corpus/photos                 # 38 new
```

Real counts observed, grouped by `raw_items.source`: whatsapp 334, sms 44, mms 1, instagram
90, photo 38 (`SELECT source, count(*) FROM raw_items GROUP BY source`). All idempotent —
covered by `tests/test_cli.py::test_ingest_reingest_is_idempotent`.

`locket pipeline run --skip-vision --corpus-dir demo_corpus` was then run for real. It fails
fast and clearly, by design (`_cmd_pipeline_run`'s upfront guard, added this session):

```
ANTHROPIC_API_KEY is not set -- extraction needs the Claude API and cannot run without it.
Set it in .env (see .env.example) and retry. (Vision pre-pass does NOT need it; `locket
ingest` alone works fine without a key.)
```

This is expected, not a bug: extraction is the one pipeline stage that always needs the
Claude API (haiku structured-output calls), regardless of `--skip-vision`. Every other
keyless stage was verified for real: `locket resolve` (prints "no pending merge proposals"
against the fresh ingest, correctly — no facts yet means no mentions to resolve) and `locket
profile build` (prints all five section headers with `_Nothing extracted for this section
yet._` placeholders, and persists that as `profiles` row #1 — correct, since an empty facts
table needs no model call at all, and `_render_section` short-circuits before ever building
a prompt when a section's fact list is empty).

`--skip-vision` was chosen for this session's own demo-corpus run given the measured
~135s/image vision-LLM latency above: 46 photos serial would be ~1.7 hours, not a reasonable
interactive-session cost, and the vision pre-pass code path itself (SigLIP2 tags + RapidOCR +
InsightFace clusters + curated Ollama tail) is exercised independently by
`tests/test_vision*.py`'s existing `@pytest.mark.vision` suite (Weeks 2-3) — `pipeline run`'s
own `_run_vision_prepass` wiring is new this session but reuses those same already-verified
functions, so re-running the full latency cost here would re-verify plumbing, not new risk.

**Run once ANTHROPIC_API_KEY exists**, to complete the extraction leg for real (haiku
structured-output extraction over the real windowed demo corpus, entity resolution, and
profile synthesis with real haiku-rendered prose, all through the single CLI entry point):

```bash
# ensure the db container is up (docker compose up -d db) and .env/ANTHROPIC_API_KEY is set
uv run python -m locket.cli pipeline run --skip-vision --corpus-dir demo_corpus
# then inspect the synthesized profile:
uv run python -m locket.cli profile build
```

Drop `--skip-vision` for the full run (also needs local Ollama `qwen3-vl:8b` pulled and
running, per Task 13 — expect roughly `46 * 135s ≈ 1.7 hours` serial for the vision-LLM tail
alone at this machine's measured per-image latency, before extraction even starts; a smaller
`--cap` bounds this).

## Local backend (informal, not the official baseline) — 2026-07-31

Added `locket.llm.get_chat_model`: a backend-selection seam (`LOCKET_LLM_BACKEND=anthropic`
| `ollama`, defaulting to `ollama` when no `ANTHROPIC_API_KEY` is set) so extraction,
resolution, profile rendering, and `answer_question` can all run against a local Ollama
model instead of the Claude API, with zero code changes to the LangGraph extraction graph
itself (`langchain-ollama==1.1.0`'s `ChatOllama.with_structured_output(..., method=
"json_schema", include_raw=True)` returns the identical `{"raw","parsed","parsing_error"}`
shape as `ChatAnthropic`'s — verified live against the installed package, not assumed).
**These numbers are informal** — a different local model, different hardware, or the
official `claude-haiku-4-5` baseline (still pending `ANTHROPIC_API_KEY`, see above) will all
differ from what's recorded here. Kept as a clearly separate section for exactly that reason.

### Model comparison: which local model to default to

Three candidate local text models were tried against real `demo_corpus/whatsapp/team.txt`
extraction windows (this machine, CPU-only Ollama, `format=ExtractionResult.model_json_schema()`,
`temperature=0`):

| model | source | 3-window times | notes |
|---|---|---|---|
| `qwen3-vl:8b` | already pulled (this repo's vision model) | did not return within 5 min on a single 2-message window | Ollama reports a "thinking" capability for this model. Neither `think=False` (raw `ollama.chat()`/HTTP API) nor `reasoning=False` (`ChatOllama`) suppressed its `<think>` reasoning trace — a plain "say hello in one word" prompt still emitted a ~280-token `<think>` block, 32s end-to-end (~7 tok/s on this box, only 2.27/6.2GB of the model in VRAM). Adding a JSON-schema `format` on top of that made even a trivial 2-message window not return in 5 minutes. **Disqualified as a text-extraction model** — it remains `vision_llm.py`'s image-description model (already measured to tolerate ~140s/image there; vision is a fundamentally different, already-slow workload). |
| `qwen2.5:3b-instruct` | already pulled locally, no download | 9.8s / 2.7s / 1.4s (13.8s total) | Valid schema output, no parsing errors, ~10x faster than gemma3:12b below. But facts skewed toward noise: on the largest (26-item) window it produced bare `person | Jeffrey Williams` / `person | Cory Davis` entries with no content beyond a name, mixed in with legitimate facts (9 facts total on that window). |
| `gemma3:12b` | pulled for this comparison (`ollama pull gemma3:12b`, ~8.1GB) | 131.8s / 27.3s / 10.0s (169.1s total) | ~10x slower, but denser/more accurate facts on the *same* largest window (8 facts, no bare-name noise): a `habit` ("ate a bagel alone at 9pm on his birthday last year"), a `preference` ("enjoys carbonara from Bertucci's Trattoria"), and specific event details ("dinner at Bertucci's ... March 3rd at 7pm") that `qwen2.5:3b-instruct` did not surface at all. **Chosen as the default** (`LOCKET_LOCAL_MODEL` env var, `DEFAULT_LOCAL_TEXT_MODEL` in `src/locket/llm.py`) — for a personal-context engine whose facts feed a citable profile and an `answer_question` tool, completeness/accuracy outweighs the latency cost, and 130s for the largest window measured is still inside `vision_llm.py`'s already-accepted ~140s/image local-model tolerance. |

Full side-by-side transcript for the largest window (`Kathryn Petrović created group "team"` ...
Cory Davis's birthday dinner), same input, same schema:

```
=== qwen2.5:3b-instruct (9 facts) ===
 - event | Kathryn Petrović created group 'team'
 - person | Jeffrey Williams
 - event | Cory Davis mentioned his birthday is March 3rd
 - person | Joshua Vega
 - event | Kathryn Petrović reminded Cory Davis about his birthday last year
 - person | Cory Davis
 - event | Joshua Vega agreed with Cory Davis's opinion about the carbonara at Bertucci's Trattoria
 - person | Jeffrey Williams
 - event | Kathryn Petrović booked a table at Bertucci's Trattoria for Cory Davis' birthday dinner
=== gemma3:12b (8 facts) ===
 - event | Kathryn Petrović created group "team"
 - relationship | Cory Davis is part of the "team" group.
 - event | Cory Davis' birthday is on March 3rd.
 - habit | Last year, Cory Davis ate a bagel alone at 9pm on his birthday.
 - preference | Cory Davis enjoys carbonara from Bertucci's Trattoria.
 - event | The group plans to have dinner at Bertucci's Trattoria downtown.
 - event | The dinner at Bertucci's is scheduled for March 3rd at 7pm.
 - person | Biscuit says hi.
```

Set `LOCKET_LOCAL_MODEL=qwen2.5:3b-instruct` for a much faster, lower-quality run (e.g. while
iterating on adapters/chunking, not a real corpus pass).

### A real bug this run surfaced: `store.py`'s `_jsonable` and UUID arrays

The first full end-to-end attempt with the local backend got all the way through extraction
(all 5 sources, 507 raw items) and crashed during entity resolution:
`TypeError: Object of type UUID is not JSON serializable`, inside
`update_fact`'s `json.dumps(_jsonable(prev))` call. Root cause: `_jsonable` only checked the
*container's* type (`isinstance(v, (list, dict, ...))`) and passed matching containers
through unconverted — it never recursed into the container's *elements*. Postgres's
`entity_ids uuid[]` column comes back from psycopg as `list[uuid.UUID]`, not `list[str]`, so
any `update_fact` call whose `prev` row already had a populated `entity_ids` crashed. This is
pre-existing and backend-independent (nothing to do with local vs. Claude-API extraction) —
it just had never been hit by a real full pipeline run before, because `add_fact`'s
statement-hash dedup returning an *existing* fact id (routine on a 268-fact corpus with
several near-duplicate statements) is what makes the resolution loop call `update_fact(...,
entity_ids=...)` a second time on the same fact id, and only the *second* call has a
non-empty `prev.entity_ids` to trip over. Fixed by making `_jsonable` recurse into
list/dict elements (`src/locket/store.py`); regression test added at
`tests/test_store.py::test_update_fact_a_second_time_with_populated_entity_ids_does_not_crash`
(`-m db`, reproduces the exact scenario against the real Postgres container).

### Real full pipeline run

`locket pipeline run --skip-vision --corpus-dir demo_corpus` run for real (terminal
invocation of `python -m locket.cli`, not a pytest stub), keylessly (`ANTHROPIC_API_KEY`
unset throughout — `resolve_backend` picked `ollama`, `LOCKET_LOCAL_MODEL` at its default
`gemma3:12b`), against every source in `demo_corpus/` (507 raw items across whatsapp x2, sms,
instagram, photos — 181 extraction windows total).

**Wall time: ~71 minutes**, end to end (extraction + resolution + profile synthesis, this
third attempt — the first two attempts had already paid extraction's compute cost before
hitting the `store.py` bug above, so 71 minutes is one clean successful pass, not a low
estimate inflated by retries). CLI's own summary:

```json
{
  "sources": 5,
  "raw_items_inserted": 0,
  "facts_created": 280,
  "mentions_seen": 31
}
```

(`raw_items_inserted: 0` because this was re-run against data already ingested by the
earlier crashed attempts — `add_raw_items` is idempotent by design, so this is expected, not
a bug. `facts_created` counts every fact extracted this pass, including ones that
`add_fact`'s hash-dedup folded into an already-existing row.)

Final store state after the run: **268 unique facts** (event 103, relationship 63,
preference 42, person 25, event/place 16, habit 19 — `SELECT kind, count(*) FROM facts GROUP
BY kind`), **19 entities**, and a synthesized profile persisted (`profiles` table, `id`
generated, `fact_count=268`). Spot-checked against `docs/demo.md`'s three scripted demo
questions — the synthesized profile's "Identity"/"People" sections directly contain the
expected answers, e.g. `"Joshua Vega will work at Northwind Robotics"` and `"Jeffrey Williams
and Sarah Mendes are going to Lisbon"`, both correctly cited to `[fact:<id>]` markers.

`locket resolve` and `locket profile build` were already keyless before this task (see the
pending-on-key section above) and are unaffected here — this run is the first time
`pipeline run`'s extraction leg has ever completed for real against the full demo corpus.
The official `claude-haiku-4-5` baseline (extraction P/R/F1 against the gold set, RAG eval)
remains pending `ANTHROPIC_API_KEY` — nothing here substitutes for that; this section exists
only to prove the keyless path works and to record its real, honestly-slower cost.
