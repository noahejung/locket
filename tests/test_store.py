"""Store tests against the real dockerized Postgres+pgvector. Requires `docker compose up -d db`.

Run explicitly: `uv run pytest -m db`. Excluded from the default `-x -q` sweep (Task 1's addopts).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from locket.models import Fact, FactKind, RawItem, SourceKind
from locket.store import Store

pytestmark = pytest.mark.db

DB_URL = os.environ.get("LOCKET_DB_URL", "postgresql://locket:locket@127.0.0.1:5432/locket")


def _vec(seed: float, dims: int = 384) -> list[float]:
    """A cheap deterministic unit-ish vector for test data — not a real embedding."""
    v = [0.0] * dims
    v[0] = seed
    v[1] = 1.0
    return v


@pytest.fixture
def store():
    s = Store(DB_URL)
    # Isolate each test — truncate everything, cascading through FKs-by-convention.
    with s._conn.cursor() as cur:
        cur.execute(
            "TRUNCATE raw_items, facts, entities, fact_history, merge_proposals, profiles, "
            "extracted_windows RESTART IDENTITY CASCADE"
        )
    s._conn.commit()
    yield s
    s._conn.close()


def _raw(i: int, text: str = "hello") -> RawItem:
    return RawItem.make(
        source=SourceKind.whatsapp,
        ts=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=i),
        sender="John",
        text=f"{text} {i}",
        thread="t",
    )


def test_add_raw_items_idempotent(store):
    items = [_raw(i) for i in range(3)]
    n1 = store.add_raw_items(items)
    assert n1 == 3
    n2 = store.add_raw_items(items)  # re-ingest same batch
    assert n2 == 0


def test_add_fact_writes_history_and_dedupes(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(
        kind=FactKind.event,
        statement="John and Sarah had dinner in Boston",
        confidence=0.9,
        subjects=["John", "Sarah"],
        place="Boston",
        happened_at="2025-01-01",
        provenance=[raw.id],
    )
    fact_id = store.add_fact(fact, embedding=_vec(1.0))
    assert fact_id

    with store._conn.cursor() as cur:
        cur.execute("SELECT event FROM fact_history WHERE fact_id = %s", (fact_id,))
        events = [r[0] for r in cur.fetchall()]
    assert events == ["ADD"]

    # Identical statement -> dedup, same id returned, no second history row, no second fact row.
    dup_id = store.add_fact(fact, embedding=_vec(1.0))
    assert dup_id == fact_id
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM facts")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM fact_history WHERE fact_id = %s", (fact_id,))
        assert cur.fetchone()[0] == 1


def test_update_fact_writes_history_and_mutates(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(
        kind=FactKind.habit,
        statement="Sarah attends yoga on Tuesdays",
        confidence=0.7,
        subjects=["Sarah"],
        provenance=[raw.id],
    )
    fact_id = store.add_fact(fact, embedding=_vec(2.0))

    store.update_fact(fact_id, confidence=0.95)
    with store._conn.cursor() as cur:
        cur.execute("SELECT confidence FROM facts WHERE id = %s", (fact_id,))
        assert cur.fetchone()[0] == pytest.approx(0.95)
        cur.execute(
            "SELECT event FROM fact_history WHERE fact_id = %s ORDER BY id", (fact_id,)
        )
        events = [r[0] for r in cur.fetchall()]
    assert events == ["ADD", "UPDATE"]

    invalidated_at = datetime(2025, 6, 1, tzinfo=UTC)
    store.update_fact(fact_id, invalid_at=invalidated_at)
    with store._conn.cursor() as cur:
        cur.execute("SELECT invalid_at FROM facts WHERE id = %s", (fact_id,))
        assert cur.fetchone()[0] == invalidated_at
        cur.execute(
            "SELECT event FROM fact_history WHERE fact_id = %s ORDER BY id", (fact_id,)
        )
        events = [r[0] for r in cur.fetchall()]
    assert events == ["ADD", "UPDATE", "EXPIRE"]


def test_update_fact_rejects_a_non_mutable_column(store):
    """Fix-wave-1 item 10 / code-quality review's latent-injection-shape finding:
    update_fact's SET clause interpolates column NAMES straight from kwargs (values stay
    parameterized) -- safe at every current call site, but one allowlist away from
    defensive. `id` is a real column but must never be settable through this generic path."""
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(kind=FactKind.habit, statement="Sarah runs", confidence=0.7, provenance=[raw.id])
    fact_id = store.add_fact(fact, embedding=_vec(2.0))

    with pytest.raises(ValueError, match="non-mutable column"):
        store.update_fact(fact_id, id="00000000-0000-0000-0000-000000000000")

    # Rejected before any write -- the fact is untouched, and the connection stays usable.
    with store._conn.cursor() as cur:
        cur.execute("SELECT confidence FROM facts WHERE id = %s", (fact_id,))
        assert cur.fetchone()[0] == pytest.approx(0.7)


def test_update_fact_a_second_time_with_populated_entity_ids_does_not_crash(store):
    """Regression: psycopg returns a `uuid[]` column as `list[uuid.UUID]`, not `list[str]`.
    `update_fact`'s history write reads the fact's CURRENT row as `prev` before applying the
    new change -- so the second call here has a `prev["entity_ids"]` that is a real populated
    `list[UUID]` (from the first call), which `_jsonable`'s old shallow (container-type-only)
    check let straight through to `json.dumps`, raising `TypeError: Object of type UUID is
    not JSON serializable`. Confirmed live during a real end-to-end `pipeline run`: `add_fact`
    dedupes on statement hash, so the pipeline's resolution step can legitimately call
    `update_fact(fact_id, entity_ids=...)` twice for the same fact id."""
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(
        kind=FactKind.relationship,
        statement="John and Sarah are friends",
        confidence=0.8,
        subjects=["John", "Sarah"],
        provenance=[raw.id],
    )
    fact_id = store.add_fact(fact, embedding=_vec(3.0))
    entity_a = store.upsert_entity("John", "person", _vec(4.0))
    entity_b = store.upsert_entity("Sarah", "person", _vec(5.0))

    store.update_fact(fact_id, entity_ids=[entity_a])  # prev.entity_ids is None here -- fine even pre-fix
    store.update_fact(fact_id, entity_ids=[entity_a, entity_b])  # prev.entity_ids is list[UUID] -- this crashed pre-fix

    with store._conn.cursor() as cur:
        cur.execute("SELECT entity_ids FROM facts WHERE id = %s", (fact_id,))
        stored = cur.fetchone()[0]
        cur.execute("SELECT event FROM fact_history WHERE fact_id = %s ORDER BY id", (fact_id,))
        events = [r[0] for r in cur.fetchall()]
    assert {str(e) for e in stored} == {entity_a, entity_b}
    assert events == ["ADD", "UPDATE", "UPDATE"]


def test_search_facts_cosine_ranked_and_valid_at_filter(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    close = Fact(
        kind=FactKind.preference,
        statement="John prefers tea over coffee",
        confidence=0.8,
        provenance=[raw.id],
    )
    far = Fact(
        kind=FactKind.preference,
        statement="Sarah dislikes cilantro",
        confidence=0.8,
        provenance=[raw.id],
    )
    close_id = store.add_fact(close, embedding=_vec(1.0))
    far_id = store.add_fact(far, embedding=_vec(-1.0))

    results = store.search_facts(_vec(1.0), limit=10)
    ids = [r.id for r in results]
    assert ids.index(close_id) < ids.index(far_id)  # closer vector ranks first

    # Expire `close`, then a query "as of" before the expiry should still surface it.
    before = datetime(2025, 1, 1, tzinfo=UTC)
    after_expiry_point = datetime(2025, 6, 1, tzinfo=UTC)
    store.update_fact(close_id, invalid_at=datetime(2025, 3, 1, tzinfo=UTC))

    still_valid = store.search_facts(_vec(1.0), limit=10, valid_at=before)
    assert close_id in [r.id for r in still_valid]

    now_expired = store.search_facts(_vec(1.0), limit=10, valid_at=after_expiry_point)
    assert close_id not in [r.id for r in now_expired]


def test_search_facts_kinds_filter(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    pref = Fact(kind=FactKind.preference, statement="John likes hiking", confidence=0.8, provenance=[raw.id])
    habit = Fact(kind=FactKind.habit, statement="John runs every morning", confidence=0.8, provenance=[raw.id])
    pref_id = store.add_fact(pref, embedding=_vec(1.0))
    habit_id = store.add_fact(habit, embedding=_vec(1.0))

    only_pref = store.search_facts(_vec(1.0), kinds=["preference"])
    ids = [r.id for r in only_pref]
    assert pref_id in ids
    assert habit_id not in ids


def test_fact_row_as_tool_dict():
    from locket.store import FactRow

    row = FactRow(
        id="abc",
        kind="event",
        statement="stmt",
        confidence=0.5,
        entity_ids=["e1"],
        provenance=["r1"],
        happened_at="2025-01-01",
        valid_at=None,
        invalid_at=None,
    )
    d = row.as_tool_dict()
    assert d["statement"] == "stmt"
    assert d["kind"] == "event"
    assert d["confidence"] == 0.5
    assert d["happened_at"] == "2025-01-01"
    assert d["sources"] == ["r1"]


def test_upsert_entity_and_nearest_entities(store):
    id1 = store.upsert_entity("Sarah Kovacs", "person", _vec(1.0))
    id1_again = store.upsert_entity("Sarah Kovacs", "person", _vec(1.0))
    assert id1 == id1_again  # same name+kind upserts to the same row

    id2 = store.upsert_entity("Boston", "place", _vec(-1.0))
    assert id2 != id1

    nearest = store.nearest_entities(_vec(1.0), k=5)
    ids = [e.id for e in nearest]
    assert ids.index(id1) < ids.index(id2)
    top = nearest[0]
    assert top.id == id1
    assert top.similarity == pytest.approx(1.0, abs=1e-6)


def test_entities_name_kind_has_a_unique_constraint(store):
    """Fix-wave-2 item 11 / code-quality review nice-to-have: upsert_entity's prior
    check-then-insert shape had a race window between its SELECT and its INSERT where two
    concurrent callers could both see "not found" and both insert, producing two rows for
    the same (name, kind) pair -- the exact duplicate upsert_entity's own docstring
    promises never happens. A real UNIQUE constraint (db/init.sql) closes that window at
    the database level; this proves the constraint exists by trying to violate it directly
    via raw SQL, bypassing upsert_entity's own ON CONFLICT handling entirely."""
    import psycopg
    from pgvector import Vector

    store.upsert_entity("Sarah Kovacs", "person", _vec(1.0))

    with pytest.raises(psycopg.errors.UniqueViolation):
        with store._conn.transaction(), store._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO entities (name, kind, embedding) VALUES (%s, %s, %s)",
                ("Sarah Kovacs", "person", Vector(_vec(2.0))),
            )

    # transaction() auto-rolled back on the raised exception -- connection still usable.
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities WHERE name = %s AND kind = %s", ("Sarah Kovacs", "person"))
        assert cur.fetchone()[0] == 1


def test_upsert_entity_is_a_single_atomic_statement_not_check_then_insert(store):
    """Companion to the constraint test above: upsert_entity itself must resolve a
    conflict via ON CONFLICT (one round-trip), not a SELECT-then-INSERT-on-miss shape that
    the constraint above would now turn into a raised UniqueViolation under a genuine race.
    Calling it twice for the same (name, kind) must still return the same id and must not
    raise, even though a real UNIQUE constraint is now in place."""
    id1 = store.upsert_entity("Priya Patel", "person", _vec(6.0))
    id2 = store.upsert_entity("Priya Patel", "person", _vec(7.0))  # different embedding, same key
    assert id1 == id2
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities WHERE name = %s AND kind = %s", ("Priya Patel", "person"))
        assert cur.fetchone()[0] == 1


def test_get_facts_for_entity(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    entity_id = store.upsert_entity("John", "person", _vec(1.0))
    fact = Fact(
        kind=FactKind.person,
        statement="John works at Acme",
        confidence=0.9,
        entity_ids=[entity_id],
        provenance=[raw.id],
    )
    fact_id = store.add_fact(fact, embedding=_vec(1.0))

    rows = store.get_facts_for_entity(entity_id)
    assert len(rows) == 1
    assert rows[0].id == fact_id


def test_list_facts_filters_by_kind_and_orders_by_created_at(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    pref = Fact(kind=FactKind.preference, statement="John likes hiking", confidence=0.8, provenance=[raw.id])
    habit = Fact(kind=FactKind.habit, statement="John runs every morning", confidence=0.8, provenance=[raw.id])
    pref_id = store.add_fact(pref, embedding=_vec(1.0))
    habit_id = store.add_fact(habit, embedding=_vec(1.0))

    all_rows = store.list_facts()
    assert [r.id for r in all_rows] == [pref_id, habit_id]

    only_habit = store.list_facts(kinds=["habit"])
    assert [r.id for r in only_habit] == [habit_id]


def test_list_facts_valid_at_excludes_expired_facts(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(kind=FactKind.habit, statement="John runs every morning", confidence=0.8, provenance=[raw.id])
    fact_id = store.add_fact(fact, embedding=_vec(1.0))
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))

    before_expiry = datetime(2019, 1, 1, tzinfo=UTC)
    after_expiry = datetime(2025, 1, 1, tzinfo=UTC)
    assert fact_id in [r.id for r in store.list_facts(valid_at=before_expiry)]
    assert fact_id not in [r.id for r in store.list_facts(valid_at=after_expiry)]
    assert fact_id in [r.id for r in store.list_facts()]  # no valid_at -> unfiltered


def test_get_facts_for_entity_valid_at_excludes_expired_facts(store):
    raw = _raw(0)
    store.add_raw_items([raw])
    entity_id = store.upsert_entity("John", "person", _vec(1.0))
    fact = Fact(
        kind=FactKind.person, statement="John works at Acme", confidence=0.9,
        entity_ids=[entity_id], provenance=[raw.id],
    )
    fact_id = store.add_fact(fact, embedding=_vec(1.0))
    store.update_fact(fact_id, invalid_at=datetime(2020, 1, 1, tzinfo=UTC))

    after_expiry = datetime(2025, 1, 1, tzinfo=UTC)
    assert store.get_facts_for_entity(entity_id, valid_at=after_expiry) == []
    assert len(store.get_facts_for_entity(entity_id)) == 1  # no valid_at -> unfiltered


def test_extracted_window_hashes_round_trip(store):
    assert store.get_extracted_window_hashes(["h1", "h2"]) == set()

    store.mark_windows_extracted(["h1", "h2"])
    assert store.get_extracted_window_hashes(["h1", "h2", "h3"]) == {"h1", "h2"}

    # Marking an already-marked hash again is a no-op, not a duplicate-key error.
    store.mark_windows_extracted(["h1"])
    assert store.get_extracted_window_hashes(["h1"]) == {"h1"}


def test_extracted_window_hashes_empty_input_short_circuits(store):
    assert store.get_extracted_window_hashes([]) == set()
    store.mark_windows_extracted([])  # must not raise
    with store._conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM extracted_windows")
        assert cur.fetchone()[0] == 0


def test_list_entities_and_get_entity(store):
    person_id = store.upsert_entity("John", "person", _vec(1.0))
    place_id = store.upsert_entity("Boston", "place", _vec(-1.0))
    store.add_entity_alias(person_id, "Johnny")

    people = store.list_entities(kind="person")
    assert [e.id for e in people] == [person_id]

    everyone = store.list_entities()
    assert {e.id for e in everyone} == {person_id, place_id}

    card = store.get_entity(person_id)
    assert card is not None
    assert card.name == "John"
    assert card.aliases == ["Johnny"]

    assert store.get_entity("00000000-0000-0000-0000-000000000000") is None


def test_write_failure_rolls_back_and_leaves_the_connection_usable(store):
    """Regression (fix-wave-1 item 3, MUST-FIX #1 of the code-quality review): before this
    fix, no Store method ever called rollback() -- Postgres aborts the whole transaction on
    any statement error, and every later statement on that same connection fails with
    "current transaction is aborted, commands ignored until end of transaction block" until
    someone calls rollback(). Worst surface: the MCP server holds one Store/connection for
    its entire process lifetime, so a single bad write would poison every subsequent tool
    call -- the exact MCP-server-poisoning scenario the review named. `add_merge_proposal`
    references a non-existent candidate_entity_id, which the schema enforces with a REFERENCES
    entities(id) foreign key -- a real constraint violation, not a contrived error."""
    import psycopg

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.add_merge_proposal(
            "Ghost", "00000000-0000-0000-0000-000000000000", evidence="x", score=0.5
        )

    # The connection must still be usable for a totally unrelated subsequent call -- this is
    # what "poisoned" would mean if the fix were absent: every call below would raise
    # "current transaction is aborted" instead of succeeding.
    entity_id = store.upsert_entity("Sarah", "person", _vec(1.0))
    assert entity_id
    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(kind=FactKind.habit, statement="Sarah runs every morning", confidence=0.8, provenance=[raw.id])
    fact_id = store.add_fact(fact, embedding=_vec(2.0))
    assert fact_id


def test_read_failure_rolls_back_and_leaves_the_connection_usable(store):
    """Same poisoning scenario, but for a READ -- per the review's "reads get
    rollback-on-error protection too" requirement. Must be a genuine SERVER-side failure
    (mid-SELECT, after a real round-trip), not a client-side adaptation error caught before
    any query is sent -- a wrong-dimension embedding is the real thing: pgvector's `<=>`
    operator rejects a query vector whose dimensionality doesn't match the indexed column
    (`vector(384)`) server-side, exactly like a real caller bug would. Needs at least one
    real row present -- against an empty table the `<=>` operator is never actually
    invoked, so the dimension check never fires."""
    import psycopg

    raw = _raw(0)
    store.add_raw_items([raw])
    fact = Fact(kind=FactKind.preference, statement="John prefers tea", confidence=0.8, provenance=[raw.id])
    store.add_fact(fact, embedding=_vec(1.0))

    with pytest.raises(psycopg.Error):
        store.search_facts([0.0, 1.0], limit=5)  # 2 dims, column is vector(384)

    results = store.search_facts(_vec(1.0), limit=5)
    assert len(results) == 1  # the call itself must succeed, not raise


def test_save_profile_and_get_latest_profile(store):
    assert store.get_latest_profile() is None

    first_id = store.save_profile("# Profile v1", fact_count=3)
    latest = store.get_latest_profile()
    assert latest is not None
    assert latest.id == first_id
    assert latest.body == "# Profile v1"
    assert latest.fact_count == 3

    second_id = store.save_profile("# Profile v2", fact_count=5)
    latest = store.get_latest_profile()
    assert latest.id == second_id
    assert latest.body == "# Profile v2"
