# locket

A privacy-first personal context engine: it ingests your own photo and
messaging exports (WhatsApp, Instagram DMs, SMS/MMS backups, Google Photos
Takeout), extracts typed, provenance-cited facts about your life with an
LLM pipeline, resolves the people and places those facts mention into
stable entities, stores everything in Postgres+pgvector, and serves the
resulting profile to other tools over MCP (Model Context Protocol).

**Privacy posture:** your real exports never live inside this repository and
are never committed — they're read from a directory outside the repo,
pointed to by `LOCKET_CORPUS_DIR` in a local, gitignored `.env`. Photos are
processed locally by default (EXIF, local vision models, local face
clustering); only extracted text is ever sent to a third-party API, under
that API's no-training terms. Everything under `demo_corpus/` in this repo
is synthetic — five invented personas, generated conversations and staged
photos of AI-generated faces — used for tests, CI, and the public demo.
No real data of any kind ships in this repository.

See `Claude/specs/2026-07-30-locket-design.md` (private planning vault, not
part of this repo) for the full design writeup.

## Usage

```bash
# 1. Start Postgres+pgvector.
docker compose up -d db

# 2. Install dependencies (uv resolves the lockfile; add ANTHROPIC_API_KEY to
#    a local .env first if you want extraction to actually run).
uv sync

# 3. Ingest a single export file/directory. Adapter is auto-detected by shape:
#    .txt -> WhatsApp, .xml -> SMS/MMS, a dir of message_*.json -> Instagram,
#    any other dir -> photos (EXIF + Takeout sidecars).
uv run python -m locket.cli ingest demo_corpus/whatsapp/team.txt
uv run python -m locket.cli ingest demo_corpus/sms/backup.xml
uv run python -m locket.cli ingest demo_corpus/photos

# 4. Run the full pipeline: vision pre-pass -> extraction -> entity
#    resolution -> profile synthesis. Needs ANTHROPIC_API_KEY (extraction is
#    the one stage that always calls the Claude API). --skip-vision bypasses
#    the local SigLIP2/OCR/face pre-pass and the local-Ollama vision-LLM tail
#    (the latter measured at ~135s/image on a CPU-only machine — see
#    evals/BASELINE.md — so it's worth skipping for a quick pass).
uv run python -m locket.cli pipeline run --skip-vision --corpus-dir demo_corpus

# 5. Review anything entity resolution couldn't confidently auto-merge.
uv run python -m locket.cli resolve

# 6. Label detected face clusters with names (needs the local vision models).
uv run python -m locket.cli label-faces

# 7. Read the synthesized profile, or serve it over MCP.
uv run python -m locket.cli profile build
uv run python -m locket.cli serve   # stdio MCP server -- see docs/demo.md for
                                     # the exact `claude mcp add` registration command

# 8. Run the eval suites (extraction P/R/F1 against a self-labeled gold set;
#    Ragas faithfulness/relevancy/precision over a fixed question set).
uv run python -m locket.cli eval extraction --json
uv run python -m locket.cli eval rag --json
```

Every subcommand above works against the committed synthetic `demo_corpus/`
out of the box. To run against your own data, set `LOCKET_CORPUS_DIR` in a
local `.env` (see `.env.example`) to a directory *outside* this repository
and point `ingest`/`pipeline run --corpus-dir` at it instead.
