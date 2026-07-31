# Architecture

locket is a staged, deterministic pipeline: each stage is a plain Python
module with a narrow, typed interface, independently unit-testable without
the stages around it. The only agentic behavior in the whole system is on
the query side (`answer_question`'s decompose → retrieve → synthesize loop).

```mermaid
flowchart LR
    subgraph Sources["Your exports (never committed)"]
        WA[WhatsApp .txt]
        IG[Instagram DM JSON]
        SMS[SMS/MMS XML]
        PH[Photos + Takeout sidecars]
    end

    subgraph Adapters["adapters/ — pure parsers, no LLM, no DB"]
        AWA[whatsapp.py]
        AIG[instagram.py]
        ASMS[sms_xml.py]
        APH[photos.py]
    end

    WA --> AWA
    IG --> AIG
    SMS --> ASMS
    PH --> APH

    AWA & AIG & ASMS & APH --> RI[(RawItem stream)]

    subgraph Vision["vision/ — local models, 100% of photos"]
        TAG[SigLIP2 tagger]
        OCR[RapidOCR]
        FACE[InsightFace + DBSCAN]
        VLLM[qwen3-vl:8b via Ollama\ncurated tail only]
    end

    RI -- photo items --> TAG & OCR & FACE
    TAG -- select_for_vision --> VLLM

    RI --> WIN[chunking.windows\nconversation windowing]
    WIN --> EXT[extraction/graph.py\nLangGraph: Send fan-out,\ncorrective retry, haiku→sonnet escalation]
    EXT -- ExtractedFact + provenance --> RES[resolution.py\ntiered entity resolution\n+ human confirm queue]
    RES --> STORE[(Postgres + pgvector\nfacts, entities, raw_items,\nfact_history, merge_proposals, profiles)]

    STORE --> PROF[profile.py\nsection scaffold +\nhaiku prose, [fact:id] citations]
    PROF --> STORE

    STORE --> MCP[mcp_server.py\nsix tools, stdio]
    MCP --> CLIENT[Claude Code / Claude Desktop]

    CLI[cli.py\ningest · pipeline run · resolve\nlabel-faces · eval · profile · serve] -.orchestrates.-> AWA & AIG & ASMS & APH & EXT & RES & PROF & MCP
```

## Stage table

| Stage | Module | Talks to | Notes |
|---|---|---|---|
| 1. Adapters | `adapters/{whatsapp,instagram,sms_xml,photos}.py` | nothing (pure parsers) | Emit `RawItem`, the one shared vocabulary every later stage speaks. No LLM, no DB — fully unit-testable against fixture files. |
| 2a. Vision pre-pass | `vision/{tagger,ocr,faces}.py` | local models only | Runs on 100% of photos: SigLIP2 zero-shot tags, RapidOCR text, InsightFace face embeddings + DBSCAN clustering. Free, local, no API calls. |
| 2b. Vision-LLM tail | `vision/vision_llm.py` | local Ollama `qwen3-vl:8b` | Only the curated subset `select_for_vision` picks (screenshots/receipts/documents route to OCR instead; the rest capped and ranked by people+event score). Real photos never go to a cloud vision model by default — see Privacy below. |
| 3. Extraction | `extraction/{chunking,schemas,graph}.py` | Claude API (`claude-haiku-4-5`, escalates to `claude-sonnet-5`) | The only module that calls the Claude API. LangGraph: `Send` fan-out over conversation windows, `with_structured_output(..., method="json_schema", include_raw=True)`, an explicit corrective-retry loop (not `RetryPolicy` — see PLAN.md's build notes for why), escalates a window to sonnet after two failed haiku attempts. |
| 4. Entity resolution | `resolution.py` | `store.py`, embeddings, one haiku call per ambiguous candidate | Three tiers: embedding nearest-neighbor (floor 0.6), deterministic rules (casefold/alias/prefix match), LLM adjudication. Anything below the auto-merge confidence bar lands in a human confirm queue (`merge_proposals` table) instead of guessing. |
| 5. Store | `store.py` | Postgres + pgvector (Docker) | The *only* module that runs SQL. Bi-temporal facts (`valid_at`/`invalid_at`/`expired_at`, graphiti's pattern), an append-only audit history (`fact_history`, mem0's pattern), HNSW cosine indexes on `facts.embedding` and `entities.embedding`. |
| 6. Profile synthesis | `profile.py` | `store.py`, one haiku call per section | Deterministic scaffold (fact-kind → section grouping, chronological timeline) plus one structured-output call per section to render prose. Citations (`[fact:<id8>]`) are injected mechanically after the model call, never trusted from model output. |
| 7. MCP server | `mcp_server.py` | `store.py`, embeddings, resolution, one haiku call each for `answer_question`'s decompose + synthesize | `mcp==2.0.0`'s `MCPServer`, stdio transport. Six tools: `search_memories`, `answer_question`, `get_profile`, `query_timeline`, `get_person`, `list_people`. The only agentic path in the system. |
| — CLI | `cli.py` | every module above | `locket ingest / pipeline run / label-faces / resolve / eval / profile / serve` — argparse only, no framework dependency. `pipeline run` is the single entry point that chains vision → extraction → resolution → profile over a whole corpus directory. |

## Boundaries (enforced by convention, not tooling)

- `models.py` owns the shared vocabulary (`RawItem`, `Fact`, `FactKind`, `SourceKind`) and imports from no sibling module.
- Adapters are pure parsers: no LLM calls, no DB access, fully testable against fixture files alone.
- `store.py` is the only module that talks to Postgres — every other module that needs persistence goes through it (this is why, e.g., the entity-resolution confirm queue and the versioned profile table both live in `store.py`/`db/init.sql` even though earlier task plans didn't originally list them there).
- `extraction/` is the only module that calls the Claude API for *extraction*; `profile.py` and `mcp_server.py` make their own separate, much smaller haiku calls for prose rendering and query decomposition/synthesis respectively.
- `vision/` is the only place that touches local ML models (SigLIP2, RapidOCR, InsightFace) or the local Ollama vision-LLM.

## Dual corpus

Two corpora exist and never mix:

- **Real corpus** — your actual exports, read from `LOCKET_CORPUS_DIR` (an env var pointing outside the repo). Never committed; `.gitignore` blocks `real_corpus/`, `.env`, and `*.local.*` (the pattern the real self-labeled gold set — `evals/gold/real_gold.local.yaml` — also uses).
- **Synthetic corpus** — `demo_corpus/`, five invented personas with AI-generated (SFHQ, MIT-licensed) faces, generated conversations, and staged photos. Fully committed, drives every test, CI run, and the public demo. See `docs/demo.md`.
