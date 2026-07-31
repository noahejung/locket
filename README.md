# locket

A privacy-first personal context engine: it ingests your own photo and
messaging exports (WhatsApp, Instagram DMs, SMS/MMS backups, Google Photos
Takeout), extracts typed, provenance-cited facts about your life with an
LLM pipeline, resolves the people and places those facts mention into
stable entities, stores everything in Postgres+pgvector, and serves the
resulting profile to other tools over MCP (Model Context Protocol) — so you
can ask Claude Code or Claude Desktop things like "when did I last see
Sarah?" and get an answer that cites the exact message it came from.

## Architecture

```mermaid
flowchart LR
    Sources["Your exports\n(WhatsApp/Instagram/SMS/Photos)"] --> Adapters["adapters/\npure parsers"]
    Adapters --> RawItems[(RawItem stream)]
    RawItems -- photos --> Vision["vision/\nSigLIP2 + RapidOCR + InsightFace\n(local, 100% of photos)"]
    Vision -- curated subset --> VisionLLM["local Ollama qwen3-vl:8b\n(curated tail only)"]
    RawItems --> Windowing["extraction/chunking.py"]
    Windowing --> Extraction["extraction/graph.py\nLangGraph: Claude API\nstructured outputs"]
    Extraction --> Resolution["resolution.py\ntiered entity resolution\n+ human confirm queue"]
    Resolution --> Store[(Postgres + pgvector)]
    Store --> Profile["profile.py\nsynthesized, cited profile"]
    Profile --> Store
    Store --> MCP["mcp_server.py\nsix tools, stdio"]
    MCP --> Client["Claude Code /\nClaude Desktop"]
```

Full stage-by-stage breakdown, module boundaries, and the dual-corpus design:
**[`docs/architecture.md`](docs/architecture.md)**. 90-second walkthrough with
scripted questions and MCP registration commands:
**[`docs/demo.md`](docs/demo.md)**.

## Quickstart (against the committed synthetic demo corpus)

```bash
docker compose up -d db          # Postgres + pgvector
uv sync                          # hand-edit pyproject.toml + `uv sync` to add deps — no `uv add`

uv run python -m locket.cli ingest demo_corpus/whatsapp/team.txt
uv run python -m locket.cli ingest demo_corpus/sms/backup.xml
uv run python -m locket.cli ingest demo_corpus/photos

# Needs ANTHROPIC_API_KEY (see .env.example) — extraction is the one stage
# that always calls the Claude API. --skip-vision bypasses the local vision
# pre-pass + Ollama vision-LLM tail (~135s/image measured — see
# evals/BASELINE.md — worth skipping for a quick pass).
uv run python -m locket.cli pipeline run --skip-vision --corpus-dir demo_corpus

uv run python -m locket.cli profile build
claude mcp add --scope user locket -- uv run --directory "$(pwd)" python -m locket.mcp_server
```

Full command reference (every subcommand: `ingest`, `pipeline run`,
`resolve`, `label-faces`, `eval extraction|rag`, `profile build`, `serve`)
and the exact Claude Desktop registration JSON block:
**[`docs/demo.md`](docs/demo.md)**.

To run against your own data instead of the demo corpus, set
`LOCKET_CORPUS_DIR` in a local `.env` (never inside this repo — see
`.env.example`) and point `ingest` / `pipeline run --corpus-dir` at it.

## Privacy posture

Stated plainly, not hand-waved:

- **Storage is fully local.** Postgres+pgvector runs in your own Docker
  container. Nothing about your facts, entities, or profile is sent
  anywhere except the specific API calls described below.
- **Text extraction uses the Claude API.** Message/photo-OCR text is sent to
  Anthropic under their no-training API terms to extract structured facts
  (`claude-haiku-4-5`, escalating to `claude-sonnet-5` on repeated
  validation failures) and to render profile prose and answer questions.
  This is a real network call to a third party — disclosed honestly, not
  claimed as "fully private."
- **Real photos are processed by local models only.** EXIF/GPS, SigLIP2
  zero-shot tagging, RapidOCR, and InsightFace face clustering all run
  locally on 100% of your photo library, for free. The one open-ended
  "describe this photo" step (the vision-LLM tail) runs against a small,
  curated subset using **local Ollama `qwen3-vl:8b`** — never a cloud vision
  model, for real photos.
- **Gemini's free tier is explicitly forbidden for real photos.** Google's
  free-tier terms grant Google the right to train on and have humans review
  submitted content — unacceptable for private photos of your life. Gemini
  is permitted only as an opt-in path for generating the *synthetic* demo
  corpus, where no privacy stakes exist (the faces are AI-generated, MIT-
  licensed SFHQ portraits, and every conversation is invented). A paid
  Claude-API fallback for real photos exists behind an explicit `--cloud-ok`
  flag if local Ollama is unavailable — still Anthropic's no-training terms,
  never Gemini free tier.
- **Your real exports never enter this repository.** They're read from
  `LOCKET_CORPUS_DIR`, an env var pointing outside the repo, declared in a
  local, gitignored `.env`. `.gitignore` also blocks `real_corpus/` and
  `*.local.*` (the pattern the real self-labeled eval gold set uses:
  `evals/gold/real_gold.local.yaml`). Everything under `demo_corpus/` in
  this repo is synthetic — five invented personas, generated conversations,
  and staged photos of AI-generated faces — used for every test, CI run,
  and the public demo. No real data of any kind ships in this repository.

## Eval results

locket ships two eval suites (`evals/extraction_eval.py`,
`evals/rag_eval.py`), both runnable via `locket eval extraction|rag --json`
and both gated in CI (`.github/workflows/eval.yml`, nightly + on-demand —
kept out of the free push/PR lint+test workflow since they cost real money
per run). Full methodology, every number's provenance, and the exact
commands to reproduce or extend each measurement: **`evals/BASELINE.md`**.

| Metric | Value | Status |
|---|---|---|
| Vision-LLM tail latency (`qwen3-vl:8b`, CPU-only, this machine) | **~135s/image mean** (range 86–205s, n=6) | Measured live, Task 13 |
| Entity-resolution similarity floor (`arctic-embed-s`) | Same-person variants 0.57–0.90 cosine; different-person 0.42–0.47 | Measured live, Task 14 |
| Extraction P/R/F1 vs. the 60-fact synthetic gold set | — | **Pending `ANTHROPIC_API_KEY`** — harness implemented + unit-tested, live run recorded as a ready-to-run command in `evals/BASELINE.md` |
| Ragas faithfulness / answer-relevancy / context-precision (25 questions) | — | **Pending `ANTHROPIC_API_KEY`** — same status; starting thresholds (0.85 / 0.80 / 0.70) are asserted directly once it runs |
| Real-corpus self-labeled gold set (100–200 facts, spec §4.1) | — | **Noah-gated** — needs his real exports + the API key, off-repo by design (`evals/gold/real_gold.local.yaml`, gitignored) |

No number above is invented — where a measurement is blocked on a
still-absent API key, the table says so plainly instead of filling in a
plausible-looking placeholder.

## License

MIT (`LICENSE`). Third-party model weights and assets carry their own,
narrower terms — see **`THIRD_PARTY_NOTICES.md`** before distributing or
monetizing anything built on this repo (notably: InsightFace's `buffalo_l`
face-analysis weights are non-commercial/research-personal use only, even
though the InsightFace code itself is MIT).

See `Claude/specs/2026-07-30-locket-design.md` (private planning vault, not
part of this repo) for the full design writeup.
