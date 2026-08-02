# Third-Party Notices

locket's own code is MIT licensed (see `LICENSE`). It depends on and, for the
demo corpus, redistributes assets from the following third parties. Some of
these carry licensing terms narrower than MIT — read before distributing or
monetizing anything built on this repo.

## Model weights

- **InsightFace `buffalo_l` face-analysis weights** — the InsightFace *code* is
  MIT licensed, but the pretrained `buffalo_l` model weights are licensed for
  **non-commercial, research/personal use only**. This project uses them for
  personal context extraction on the operator's own data, which fits that
  license. If locket is ever distributed as a hosted product or monetized,
  the face-clustering feature must either be re-reviewed for a commercially
  licensed weight set or gated behind a bring-your-own-weights setup.
- **SigLIP2** (`google/siglip2-base-patch16-224`) — Apache 2.0 / Google model
  license per the Hugging Face model card. Compatible with commercial use.
- **Ollama `qwen3-vl`** — Qwen model license per the upstream model card
  (Apache 2.0 / Tongyi Qianwen license depending on variant). Verify the
  specific tag's license before redistribution.

## Synthetic corpus assets

- **SFHQ (Synthetic Faces High Quality) portraits** — MIT licensed, entirely
  AI-generated faces with no real people depicted. Used as the portrait
  source for the committed `demo_corpus/` synthetic persona photos.

## Test fixtures

- **`tests/fixtures/ios_backup/typedstream/{AttributedBodyTextOnly,URL,MultiPart,Blank}`**
  — 4 small (76–1040 byte) binary `attributedBody` blobs taken from
  `ReagentX/imessage-exporter`'s own test suite (that project is
  GPL-3.0-or-later). They are used here strictly as **test input data**, to
  verify this repo's independently-written three-tier typedstream text
  extractor (`locket/adapters/ios_backup.py`) against known-good expected
  strings — no source code from that project is vendored or ported (see the
  ios_backup adapter section below). None of the four contain real personal
  data: they decode to a test string ("Noter test"), a public GitHub repo
  URL, and synthetic placeholder text.

## Python dependencies

All pinned dependencies in `pyproject.toml` are used under their respective
upstream licenses (mostly MIT/Apache 2.0/BSD). `pgvector` (PostgreSQL
extension) is PostgreSQL-licensed. No GPL-licensed runtime dependency is used
in this project — see the WhatsApp adapter decision note in the implementation
plan for why `whatstk` (GPL-3.0) was rejected in favor of a hand-rolled parser.
The `ios_backup` adapter (Phase 1, 2026-08-02) follows the same policy: its
reference implementation, `ReagentX/imessage-exporter`, is GPL-3.0-or-later
and is used only as a facts source (schema, hashing scheme, timestamp epoch,
algorithm descriptions) — its Rust source is not ported or vendored; the
Python implementation here is independently written from those facts.

Verified directly against each package's installed metadata (2026-07-30):
the `mcp` SDK (Model Context Protocol server, `mcp_server.py`) is MIT
licensed; `langgraph`/`langchain-core`/`langchain-anthropic`/`anthropic` are
MIT; `ragas` is Apache 2.0. No surprises against the blanket statement above.

Two exceptions to the MIT/Apache/BSD norm, both verified live 2026-08-02 and
both used only in the `ios_backup` adapter:

- **`pytypedstream`** (PyPI name `pytypedstream`, import name `typedstream`)
  — **LGPL-3.0-or-later**. Used as an ordinary imported dependency (tier-1
  `attributedBody` text extraction); LGPL permits this without copyleft
  flowing into locket's own MIT-licensed code, since it's dynamically
  imported rather than statically linked/modified.
- **`iphone-backup-decrypt`** — MIT. **`pycryptodome`** (its own dependency,
  also declared directly here) — BSD/public-domain-equivalent. Both fit the
  blanket MIT/Apache/BSD statement above; called out explicitly here since
  they're new as of the `ios_backup` adapter.
