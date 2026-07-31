"""Postgres + pgvector data layer. The only module that talks to Postgres.

Schema: db/init.sql (raw_items, entities, facts, fact_history). Bi-temporal facts
(valid_at/invalid_at/expired_at) steal graphiti's pattern; flat-fact-plus-audit-history
steals mem0's. See PLAN.md Task 8.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from locket.models import Fact, RawItem

_FACT_COLUMNS = (
    "id",
    "kind",
    "statement",
    "confidence",
    "entity_ids",
    "provenance",
    "happened_at",
    "valid_at",
    "invalid_at",
)


@dataclass
class FactRow:
    id: str
    kind: str
    statement: str
    confidence: float
    entity_ids: list[str]
    provenance: list[str]
    happened_at: str | None
    valid_at: datetime | None
    invalid_at: datetime | None

    def as_tool_dict(self) -> dict:
        """The MCP wire shape."""
        return {
            "statement": self.statement,
            "kind": self.kind,
            "confidence": self.confidence,
            "happened_at": self.happened_at,
            "sources": self.provenance,
        }


@dataclass
class EntityRow:
    id: str
    name: str
    kind: str
    similarity: float  # 1 - cosine_distance
    aliases: list[str] = field(default_factory=list)


@dataclass
class EntityCard:
    """A plain entity record (no similarity score) -- for listing/lookup by id, distinct
    from EntityRow which specifically carries a nearest-neighbor search's similarity."""

    id: str
    name: str
    kind: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class ProfileRow:
    id: str
    body: str
    fact_count: int
    created_at: datetime


@dataclass
class MergeProposal:
    """A tier-3-escalated entity-resolution verdict that didn't clear the auto-merge
    confidence bar — sits in the confirm queue for a human y/n (Task 19's CLI) instead of
    merging silently. See locket.resolution (Task 14)."""

    id: str
    mention: str
    candidate_entity_id: str
    candidate_entity_name: str
    evidence: str
    score: float
    status: str = "pending"


def _row_to_fact(row: tuple) -> FactRow:
    (id_, kind, statement, confidence, entity_ids, provenance, happened_at, valid_at, invalid_at) = row
    return FactRow(
        id=str(id_),
        kind=kind,
        statement=statement,
        confidence=confidence,
        entity_ids=[str(e) for e in (entity_ids or [])],
        provenance=list(provenance or []),
        happened_at=happened_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


class Store:
    def __init__(self, db_url: str):
        self._conn = psycopg.connect(db_url, autocommit=False)
        register_vector(self._conn)

    def close(self) -> None:
        self._conn.close()

    # ---- raw_items ---------------------------------------------------------------

    def add_raw_items(self, items: Iterable[RawItem]) -> int:
        """Insert a batch of RawItems, ON CONFLICT (id) DO NOTHING. Returns count inserted."""
        items = list(items)
        if not items:
            return 0
        inserted = 0
        with self._conn.cursor() as cur:
            for item in items:
                cur.execute(
                    """
                    INSERT INTO raw_items (id, source, ts, sender, body, media_path, is_system, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        item.id,
                        str(item.source),
                        item.ts,
                        item.sender,
                        item.text,
                        item.media_path,
                        item.is_system,
                        json.dumps(item.meta),
                    ),
                )
                inserted += cur.rowcount
        self._conn.commit()
        return inserted

    # ---- facts --------------------------------------------------------------------

    def add_fact(self, fact: Fact, embedding: list[float]) -> str:
        """Insert a fact, writing a fact_history ADD row. Dedupes on md5(statement) — a
        second call with an identical statement returns the existing fact's id and writes
        no new rows anywhere."""
        stmt_hash = hashlib.md5(fact.statement.encode()).hexdigest()
        body = fact.model_dump(mode="json")
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO facts (kind, body, statement, confidence, happened_at,
                                    entity_ids, provenance, valid_at, hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now(), %s, %s)
                ON CONFLICT (hash) DO NOTHING
                RETURNING id
                """,
                (
                    str(fact.kind),
                    json.dumps(body),
                    fact.statement,
                    fact.confidence,
                    fact.happened_at,
                    fact.entity_ids or None,
                    fact.provenance,
                    stmt_hash,
                    Vector(embedding),
                ),
            )
            row = cur.fetchone()
            if row is not None:
                fact_id = str(row[0])
                cur.execute(
                    """
                    INSERT INTO fact_history (fact_id, event, prev, next)
                    VALUES (%s, 'ADD', NULL, %s)
                    """,
                    (fact_id, json.dumps(body)),
                )
                self._conn.commit()
                return fact_id

            # Conflict: statement already exists — return its id, write nothing new.
            cur.execute("SELECT id FROM facts WHERE hash = %s", (stmt_hash,))
            existing = cur.fetchone()
        self._conn.commit()
        return str(existing[0])

    def update_fact(self, fact_id: str, *, invalid_at: datetime | None = None, **changes: Any) -> None:
        """Mutate a fact's mutable columns and write a fact_history row.

        `invalid_at` set (with or without other `changes`) always logs event EXPIRE — setting
        it is what "expiring" a fact means bi-temporally. Otherwise, if `changes` is non-empty,
        logs UPDATE. A call with neither is a no-op (no history row written).
        """
        all_changes: dict[str, Any] = dict(changes)
        if invalid_at is not None:
            all_changes["invalid_at"] = invalid_at
        if not all_changes:
            return

        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM facts WHERE id = %s", (fact_id,))
            colnames = [d.name for d in cur.description]
            prev_row = cur.fetchone()
            if prev_row is None:
                raise KeyError(f"no fact with id {fact_id}")
            prev = dict(zip(colnames, prev_row, strict=True))

            set_clause = ", ".join(f"{col} = %s" for col in all_changes)
            values = list(all_changes.values())
            cur.execute(
                f"UPDATE facts SET {set_clause}, updated_at = now() WHERE id = %s",
                (*values, fact_id),
            )

            event = "EXPIRE" if invalid_at is not None else "UPDATE"
            cur.execute(
                "INSERT INTO fact_history (fact_id, event, prev, next) VALUES (%s, %s, %s, %s)",
                (fact_id, event, json.dumps(_jsonable(prev)), json.dumps(_jsonable(all_changes))),
            )
        self._conn.commit()

    def search_facts(
        self,
        embedding: list[float],
        *,
        limit: int = 20,
        kinds: list[str] | None = None,
        valid_at: datetime | None = None,
    ) -> list[FactRow]:
        """Cosine-ranked fact search, optionally filtered by kind and by "as-of" validity."""
        clauses = []
        params: list[Any] = []
        if kinds:
            clauses.append("kind = ANY(%s)")
            params.append(kinds)
        if valid_at is not None:
            clauses.append("(invalid_at IS NULL OR invalid_at > %s)")
            params.append(valid_at)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT {", ".join(_FACT_COLUMNS)}
            FROM facts
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params.extend([Vector(embedding), limit])
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    def list_facts(self, *, kinds: list[str] | None = None, limit: int = 1000) -> list[FactRow]:
        """Plain listing (no vector query) -- used by the MCP server's timeline tool and
        profile synthesis, neither of which needs semantic ranking."""
        clauses = []
        params: list[Any] = []
        if kinds:
            clauses.append("kind = ANY(%s)")
            params.append(kinds)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT {", ".join(_FACT_COLUMNS)}
            FROM facts
            {where}
            ORDER BY created_at
            LIMIT %s
        """
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_facts_for_entity(self, entity_id: str) -> list[FactRow]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {", ".join(_FACT_COLUMNS)}
                FROM facts
                WHERE %s = ANY(entity_ids)
                ORDER BY created_at
                """,
                (entity_id,),
            )
            rows = cur.fetchall()
        return [_row_to_fact(r) for r in rows]

    # ---- entities -------------------------------------------------------------------

    def upsert_entity(self, name: str, kind: str, embedding: list[float]) -> str:
        """Insert an entity, or return the existing id for an identical (name, kind) pair.

        No unique DB constraint on (name, kind) — upsert semantics live here in application
        code, matching the plan's `upsert_entity(name, kind, embedding) -> str` signature."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE name = %s AND kind = %s LIMIT 1",
                (name, kind),
            )
            existing = cur.fetchone()
            if existing is not None:
                self._conn.commit()
                return str(existing[0])
            cur.execute(
                "INSERT INTO entities (name, kind, embedding) VALUES (%s, %s, %s) RETURNING id",
                (name, kind, Vector(embedding)),
            )
            new_id = cur.fetchone()[0]
        self._conn.commit()
        return str(new_id)

    def nearest_entities(self, embedding: list[float], k: int = 15) -> list[EntityRow]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, kind, aliases, 1 - (embedding <=> %s) AS similarity
                FROM entities
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(embedding), Vector(embedding), k),
            )
            rows = cur.fetchall()
        return [
            EntityRow(id=str(r[0]), name=r[1], kind=r[2], aliases=list(r[3] or []), similarity=r[4])
            for r in rows
        ]

    def list_entities(self, kind: str | None = None) -> list[EntityCard]:
        query = "SELECT id, name, kind, aliases FROM entities"
        params: list[Any] = []
        if kind is not None:
            query += " WHERE kind = %s"
            params.append(kind)
        query += " ORDER BY name"
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [EntityCard(id=str(r[0]), name=r[1], kind=r[2], aliases=list(r[3] or [])) for r in rows]

    def get_entity(self, entity_id: str) -> EntityCard | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, name, kind, aliases FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return EntityCard(id=str(row[0]), name=row[1], kind=row[2], aliases=list(row[3] or []))

    def add_entity_alias(self, entity_id: str, alias: str) -> None:
        """Append `alias` to an entity's aliases array, idempotently (no duplicate if the
        alias is already present)."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE entities SET aliases = array_append(aliases, %s)
                WHERE id = %s AND NOT (%s = ANY(aliases))
                """,
                (alias, entity_id, alias),
            )
        self._conn.commit()

    def find_entity_by_alias(self, alias: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT id FROM entities WHERE %s = ANY(aliases) LIMIT 1", (alias,))
            row = cur.fetchone()
        return str(row[0]) if row is not None else None

    # ---- merge_proposals (entity-resolution confirm queue, Task 14) -----------------

    def add_merge_proposal(self, mention: str, candidate_entity_id: str, evidence: str, score: float) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merge_proposals (mention, candidate_entity_id, evidence, score)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (mention, candidate_entity_id, evidence, score),
            )
            new_id = cur.fetchone()[0]
        self._conn.commit()
        return str(new_id)

    def pending_merge_proposals(self) -> list[MergeProposal]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT mp.id, mp.mention, mp.candidate_entity_id, e.name, mp.evidence, mp.score, mp.status
                FROM merge_proposals mp
                JOIN entities e ON e.id = mp.candidate_entity_id
                WHERE mp.status = 'pending'
                ORDER BY mp.created_at
                """
            )
            rows = cur.fetchall()
        return [
            MergeProposal(
                id=str(r[0]),
                mention=r[1],
                candidate_entity_id=str(r[2]),
                candidate_entity_name=r[3],
                evidence=r[4],
                score=r[5],
                status=r[6],
            )
            for r in rows
        ]

    def resolve_merge_proposal(self, proposal_id: str, *, accept: bool) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE merge_proposals SET status = %s WHERE id = %s",
                ("confirmed" if accept else "rejected", proposal_id),
            )
        self._conn.commit()

    # ---- profiles (synthesized living profile, Task 18) ------------------------------

    def save_profile(self, body: str, fact_count: int) -> str:
        """Insert a new versioned profile snapshot. Never mutates a prior row -- callers
        (profile.synthesize) decide whether a new snapshot is warranted (e.g. skip when
        fact_count hasn't changed since the latest one)."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO profiles (body, fact_count) VALUES (%s, %s) RETURNING id",
                (body, fact_count),
            )
            new_id = cur.fetchone()[0]
        self._conn.commit()
        return str(new_id)

    def get_latest_profile(self) -> ProfileRow | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, body, fact_count, created_at FROM profiles ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        if row is None:
            return None
        return ProfileRow(id=str(row[0]), body=row[1], fact_count=row[2], created_at=row[3])


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Best-effort make a dict JSON-serializable for fact_history's prev/next jsonb columns
    (datetimes and other psycopg-returned types need coercing to strings)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out
