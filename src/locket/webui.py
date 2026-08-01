"""FastAPI phone chat UI (Task: tailnet-only web page, setup-guide.md Part 2 Option 2).

Product intent: Noah opens his phone's browser at the laptop's Tailscale address, asks
locket questions, gets answers with tappable citations -- the UX-testing surface for
metrics.md §3's 10-question self-quiz and trust-incident log. This module is a second,
independent frontend onto the same store + the same decompose-retrieve-synthesize answer
logic mcp_server.py's `answer_question` tool uses (`answer_question_impl`, shared rather
than duplicated) -- mcp_server.py's own six-tool surface is untouched by this module.

Security posture (mirrors the fix-wave-1 docker-compose port-binding fix and README's
Privacy posture section): `locket serve-ui` binds 127.0.0.1 by default (see cli.py's
`serve-ui` subcommand). Binding 0.0.0.0 would make `answer_question` -- which reads a
citable, provenance-linked "profile of you" -- reachable from anything else on the LAN, not
just the intended phone-over-Tailscale path (a Tailscale IP is only reachable over the
tailnet; 0.0.0.0 has no such restriction). Pass `--host <your-tailscale-ip>` explicitly for
phone access; this module never chooses that default itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from locket.embeddings import EmbeddingBackend
from locket.mcp_server import answer_question_impl
from locket.store import Store

# src/locket/webui.py -> repo root: parents[0]=src/locket, [1]=src, [2]=repo root.
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


class AskRequest(BaseModel):
    question: str


def create_app(
    store: Store,
    *,
    backend: EmbeddingBackend | None = None,
    decompose_model: Any | None = None,
    synthesize_model: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to `store`. `backend`/`*_model` mirror
    mcp_server.build_server's own injectable test seams (same defaulting contract: None
    means "use the live production default") -- real callers (cli.py's `locket serve-ui`)
    leave them unset; tests inject stubs so `/api/ask` never needs a real model or a real
    embedding backend to exercise the route wiring."""
    app = FastAPI(title="locket")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict:
        return answer_question_impl(
            store,
            body.question,
            backend=backend,
            decompose_model=decompose_model,
            synthesize_model=synthesize_model,
        )

    @app.get("/api/fact/{fact_id}")
    def fact(fact_id: str) -> dict:
        """`fact_id` is typically the 8-hex-char prefix the answer's `[fact:...]` citation
        chips carry client-side (matches profile.py's citation-shorthand convention, fix-
        wave-2 item 12) -- but Store.get_fact_by_prefix accepts any length prefix, including
        a full id, so this doesn't need to enforce exactly 8 characters itself."""
        row = store.get_fact_by_prefix(fact_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no fact matching id prefix {fact_id!r}")
        return {"id": row.id, **row.as_tool_dict()}

    return app


__all__ = ["STATIC_DIR", "AskRequest", "create_app"]
