"""FastAPI phone chat UI tests (src/locket/webui.py) -- TestClient against `create_app`,
entirely stubbed/faked (fake store, fake embedding backend, fake decompose/synthesize
models), no DB and no real embedding model needed. answer_question_impl's own
decompose-retrieve-synthesize CORRECTNESS is already covered against a real dockerized
Store by tests/test_mcp_server.py's answer_question tests (same shared implementation,
different frontend calling it) -- this file's job is proving webui.py's routing/wiring:
request in, JSON out, citations round-trip, `/` serves the static page.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from locket.mcp_server import SubQueries
from locket.store import FactRow
from locket.webui import create_app


class _FakeBackend:
    """`answer_question_impl` only ever calls `.embed_query(text)` on the backend -- the
    fake store below ignores the vector's actual content, so any fixed-length vector works."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


class _FakeDecomposeModel:
    def __init__(self, queries: list[str]):
        self._result = SubQueries(queries=queries)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return self._result


class _FakeSynthesizeModel:
    def __init__(self, answer_text: str):
        self._answer_text = answer_text
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(content=self._answer_text)


class _FakeStore:
    """Duck-types just the two Store methods webui.py's routes actually call --
    search_facts (via answer_question_impl) and get_fact_by_prefix -- against a fixed,
    in-memory list of real FactRow instances. No Postgres involved."""

    def __init__(self, rows: list[FactRow]):
        self._rows = rows

    def search_facts(self, embedding, *, limit=20, kinds=None, valid_at=None):
        return list(self._rows)[:limit]

    def get_fact_by_prefix(self, prefix: str):
        for row in self._rows:
            if row.id.startswith(prefix):
                return row
        return None


def _fact_row(id_: str, statement: str) -> FactRow:
    return FactRow(
        id=id_,
        kind="relationship",
        statement=statement,
        confidence=0.9,
        entity_ids=[],
        provenance=["r1"],
        happened_at="2025-01-15",
        valid_at=None,
        invalid_at=None,
    )


FACT_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def _client(rows: list[FactRow], *, answer_text: str, queries: list[str] | None = None) -> TestClient:
    store = _FakeStore(rows)
    app = create_app(
        store,
        backend=_FakeBackend(),
        decompose_model=_FakeDecomposeModel(queries or ["dance teammate"]),
        synthesize_model=_FakeSynthesizeModel(answer_text),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/ask
# ---------------------------------------------------------------------------


def test_ask_returns_answer_and_cited_facts_json():
    row = _fact_row(FACT_ID, "Sarah Kovacs is Noah's dance teammate")
    client = _client([row], answer_text=f"Sarah is your dance teammate [fact:{FACT_ID}].")

    response = client.post("/api/ask", json={"question": "who is on my dance team?"})

    assert response.status_code == 200
    body = response.json()
    assert f"[fact:{FACT_ID}]" in body["answer"]
    assert len(body["facts"]) == 1
    assert body["facts"][0]["id"] == FACT_ID
    assert body["facts"][0]["statement"] == "Sarah Kovacs is Noah's dance teammate"
    assert body["facts"][0]["kind"] == "relationship"
    assert body["facts"][0]["sources"] == ["r1"]


def test_ask_no_citation_returns_empty_facts_list():
    row = _fact_row(FACT_ID, "Sarah Kovacs is Noah's dance teammate")
    client = _client([row], answer_text="I don't have enough information to answer that.")

    response = client.post("/api/ask", json={"question": "what is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["facts"] == []
    assert "enough information" in body["answer"]


def test_ask_threads_the_stubbed_models_through_not_bypassing_them():
    row = _fact_row(FACT_ID, "Sarah Kovacs is Noah's dance teammate")
    store = _FakeStore([row])
    decompose = _FakeDecomposeModel(["dance teammate"])
    synthesize = _FakeSynthesizeModel(f"Sarah [fact:{FACT_ID}].")
    app = create_app(store, backend=_FakeBackend(), decompose_model=decompose, synthesize_model=synthesize)
    client = TestClient(app)

    response = client.post("/api/ask", json={"question": "who is on my dance team?"})

    assert response.status_code == 200
    assert decompose.calls  # the stub was actually invoked, not bypassed
    assert synthesize.calls


def test_ask_rejects_a_missing_question_field():
    client = _client([], answer_text="unused")

    response = client.post("/api/ask", json={})

    assert response.status_code == 422  # FastAPI/pydantic request validation


# ---------------------------------------------------------------------------
# GET /api/fact/{fact_id}
# ---------------------------------------------------------------------------


def test_fact_expansion_round_trips_by_8_char_prefix():
    row = _fact_row(FACT_ID, "Sarah Kovacs is Noah's dance teammate")
    client = _client([row], answer_text="unused")

    response = client.get(f"/api/fact/{FACT_ID[:8]}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == FACT_ID
    assert body["statement"] == "Sarah Kovacs is Noah's dance teammate"
    assert body["kind"] == "relationship"
    assert body["happened_at"] == "2025-01-15"
    assert body["sources"] == ["r1"]


def test_fact_expansion_accepts_a_full_id_too():
    row = _fact_row(FACT_ID, "Sarah Kovacs is Noah's dance teammate")
    client = _client([row], answer_text="unused")

    response = client.get(f"/api/fact/{FACT_ID}")

    assert response.status_code == 200
    assert response.json()["id"] == FACT_ID


def test_fact_expansion_unknown_prefix_returns_404():
    client = _client([], answer_text="unused")

    response = client.get("/api/fact/00000000")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET / -- serves the static single-page UI
# ---------------------------------------------------------------------------


def test_index_serves_html():
    client = _client([], answer_text="unused")

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>locket</title>" in response.text
    assert "ask locket" in response.text.lower()  # lowercase UI text (Noah's standing preference)
