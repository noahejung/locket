# Demo script (90 seconds)

Everything below runs against the committed synthetic `demo_corpus/` — five
invented personas, generated conversations, staged AI-generated-face photos.
No real data is involved. Steps marked **(needs ANTHROPIC_API_KEY)** call the
Claude API specifically; the pipeline run itself needs no key at all — with
none set it runs against a local Ollama model instead (see README's "Running
fully local" section).

## 0. Setup (once)

```bash
cd "C:\Users\nejtu\Desktop\development repos\locket"
docker compose up -d db      # Postgres + pgvector
uv sync
```

## 1. Ingest the synthetic corpus (~5s, no key needed)

```bash
uv run python -m locket.cli ingest demo_corpus/whatsapp/team.txt
uv run python -m locket.cli ingest demo_corpus/whatsapp/sarah.txt
uv run python -m locket.cli ingest demo_corpus/sms/backup.xml
uv run python -m locket.cli ingest demo_corpus/instagram/inbox/kathryn
uv run python -m locket.cli ingest demo_corpus/photos
```

Each line prints how many new `raw_items` rows it added — this alone proves
the WhatsApp dash/bracket parser, the Instagram mojibake fix, the streaming
SMS/MMS XML parser, and the EXIF/Takeout-sidecar photo adapter all work on
real (synthetic) export formats.

## 2. Run the pipeline

```bash
uv run python -m locket.cli pipeline run --skip-vision --corpus-dir demo_corpus
```

Windows the ingested messages and extracts typed facts via the LangGraph
extraction graph, resolves "Sarah Mendes" / `sarah.mendes` / "Sarah M ⭐"
into one entity, and synthesizes the profile. With `ANTHROPIC_API_KEY` set,
extraction uses `claude-haiku-4-5` (escalating to `claude-sonnet-5` on
repeated validation failures); with no key, it runs keylessly against a
local Ollama model instead (`gemma3:12b` by default — much slower and
somewhat lower-quality; see README/`evals/BASELINE.md`). `--skip-vision`
skips the local vision pre-pass + Ollama vision-LLM tail for a fast pass —
drop it for the full run (see `evals/BASELINE.md` for measured latency:
~135s/image on a CPU-only machine).

## 3. Ask it about the (synthetic) group's life

Once registered as an MCP server (see below), ask Claude three questions —
each one is answerable only from facts actually extracted from the corpus,
and every answer should cite `[fact:<id>]` markers that resolve back to real
source lines:

1. **"Where did the group have Cory Davis's birthday dinner?"**
   Expected: *Bertucci's Trattoria*, cited to a fact whose `provenance`
   traces back to a specific WhatsApp message in `demo_corpus/whatsapp/team.txt`.
2. **"What job did Joshua Vega get?"**
   Expected: *Backend engineer at Northwind Robotics*.
3. **"Where did Jeffrey Williams and Sarah Mendes travel together?"**
   Expected: *Lisbon*.

To show provenance resolving all the way back to source text without a
client in the loop, run the same lookup directly:

```bash
uv run python -c "
from locket.store import Store
store = Store('postgresql://locket:locket@localhost:5432/locket')
rows = store.search_facts([0.0]*384, limit=5)   # replace with a real query embedding
for r in rows:
    print(r.statement, '->', r.provenance)
    for raw_id in r.provenance:
        cur = store._conn.cursor()
        cur.execute('SELECT source, sender, body FROM raw_items WHERE id = %s', (raw_id,))
        print('   ', cur.fetchone())
"
```

## 4. Show the eval run

```bash
uv run python -m locket.cli eval extraction --json   # needs ANTHROPIC_API_KEY
uv run python -m locket.cli eval rag --json           # needs ANTHROPIC_API_KEY
```

Both print precision/recall/F1 (extraction, against the 60-fact self-scripted
gold set in `evals/gold/persona_gold.yaml`) and Ragas faithfulness/answer-
relevancy/context-precision (RAG, against the 25 questions in
`evals/questions.yaml`). `evals/BASELINE.md` has the full methodology and,
once a key is available, the real recorded numbers.

## MCP registration (for the coordinator to run — not this session's job)

**Claude Code:**

```bash
claude mcp add --scope user locket -- uv run --directory "C:\Users\nejtu\Desktop\development repos\locket" python -m locket.mcp_server
claude mcp list   # verify it shows up as "locket"
```

**Claude Desktop** — add this block to
`%APPDATA%\Claude\claude_desktop_config.json` (absolute paths only; merge into
the existing `mcpServers` object if one already exists):

```json
{
  "mcpServers": {
    "locket": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\nejtu\\Desktop\\development repos\\locket",
        "python",
        "-m",
        "locket.mcp_server"
      ]
    }
  }
}
```

Restart Claude Desktop (or run `claude mcp list` for Claude Code) after
adding either config to pick up the new server. Once connected, the six
tools (`search_memories`, `answer_question`, `get_profile`, `query_timeline`,
`get_person`, `list_people`) are available to ask about — first against the
demo corpus above, and later against your own data once
`LOCKET_CORPUS_DIR` points at your real exports and `locket pipeline run`
has been run against them.
