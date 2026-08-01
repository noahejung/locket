# Demo script (90 seconds)

Everything below runs against the committed synthetic `demo_corpus/` — five
invented personas, generated conversations, staged AI-generated-face photos.
No real data is involved. Steps marked **(needs ANTHROPIC_API_KEY)** call the
Claude API specifically; the pipeline run itself needs no key at all — with
none set it runs against a local Ollama model instead (see README's "Running
fully local" section).

## 0. Setup (once)

```bash
cd /path/to/your/clone/of/locket   # adjust to wherever you cloned this repo
docker compose up -d db            # Postgres + pgvector
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
store = Store('postgresql://locket:locket@127.0.0.1:5432/locket')
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

## 5. Phone testing (tailnet-only chat UI)

`locket serve-ui` serves a small self-contained chat web page — one question
box, answers with tappable `[fact:...]` citation chips that expand the
source statement inline. It talks to the same `answer_question` logic the
MCP server's tool uses (shared implementation, not a duplicate), so it's a
second, independent frontend onto the same store.

```bash
uv run python -m locket.cli serve-ui
```

**Security posture — read before exposing this to your phone.** The default
bind host is `127.0.0.1` (loopback only, reachable from this machine alone)
and it never binds `0.0.0.0` by default — `answer_question` reads a
citable, provenance-linked "profile of you," and `0.0.0.0` would expose
that to every device on your LAN, not just your own phone over Tailscale.
To actually reach it from a phone, pass your **Tailscale IP** explicitly:

```bash
# find your Tailscale IP first: `tailscale ip -4` (or check the Tailscale app)
uv run python -m locket.cli serve-ui --host <your-tailscale-ip> --port 8765
```

Then, with Tailscale installed and signed in on the phone, open
`http://<your-tailscale-ip>:8765` in the phone's browser. Nothing here is
publicly reachable — only devices on your own tailnet can connect.

**Honest latency note (shown in the UI itself, not just here):** on the
local Ollama backend, an answer can take 30–90 seconds — the page shows a
real elapsed-time counter ("thinking... 12s") while it waits, not a fake
fast-looking spinner. This is also the surface for metrics.md §3's 10-
question self-quiz and trust-incident log — the UX-testing loop this UI
exists for.

## MCP registration (for the coordinator to run — not this session's job)

**Claude Code:**

```bash
claude mcp add --scope user locket -- uv run --directory /path/to/your/clone/of/locket python -m locket.mcp_server
claude mcp list   # verify it shows up as "locket"
```

**Claude Desktop** — add this block to
`%APPDATA%\Claude\claude_desktop_config.json` (absolute paths only; merge into
the existing `mcpServers` object if one already exists; adjust the path below
to wherever you cloned this repo):

```json
{
  "mcpServers": {
    "locket": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\your\\clone\\of\\locket",
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
