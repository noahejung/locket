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

## Python dependencies

All pinned dependencies in `pyproject.toml` are used under their respective
upstream licenses (mostly MIT/Apache 2.0/BSD). `pgvector` (PostgreSQL
extension) is PostgreSQL-licensed. No GPL-licensed runtime dependency is used
in this project — see the WhatsApp adapter decision note in the implementation
plan for why `whatstk` (GPL-3.0) was rejected in favor of a hand-rolled parser.

Verified directly against each package's installed metadata (2026-07-30):
the `mcp` SDK (Model Context Protocol server, `mcp_server.py`) is MIT
licensed; `langgraph`/`langchain-core`/`langchain-anthropic`/`anthropic` are
MIT; `ragas` is Apache 2.0. No surprises against the blanket statement above.
