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
