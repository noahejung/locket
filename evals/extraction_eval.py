"""Extraction eval: self-labeled gold set vs. real pipeline output.

`score()` matches extracted facts against the gold set by kind + regex anchors + a cosine
similarity floor (the plan's explicit "guard against regex-lucky nonsense"). Matching is
greedy bipartite: each gold fact consumes at most one extracted fact, so precision/recall
counts never double-count a single extracted fact against two gold facts.

Interim wiring note (mirrors Task 16's rag_eval.py note): no earlier task produces a
combined end-to-end runner -- that arrives as Task 19's `locket pipeline run`. For now,
`run_extraction_pipeline` assembles adapters -> extract_batch() -> store.add_fact()
manually, per-source-file (never pooling all sources into one windows() call --
chunking.windows() splits purely on time gaps, with no notion of "thread", so merging
distinct conversations before windowing would let unrelated threads bleed into the same
extraction window). Swap to the shared pipeline function when Task 19 lands.

CLI (`locket eval extraction --json`) is NOT wired here -- cli.py doesn't exist until
Task 19; this module's public functions are what that CLI will call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import Field as PydField

from locket.adapters.instagram import parse_instagram_thread
from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.embeddings import get_backend
from locket.extraction.graph import extract_batch
from locket.models import Fact, FactKind
from locket.store import FactRow, Store

COSINE_FLOOR = 0.6


class GoldFact(BaseModel):
    kind: FactKind
    statement: str
    subjects: list[str] = PydField(default_factory=list)
    must_match: list[str] = PydField(default_factory=list)


@dataclass
class KindBreakdown:
    gold: int
    extracted: int
    matched: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    by_kind: dict[str, KindBreakdown]
    misses: list[str] = field(default_factory=list)  # gold statements with no match
    spurious: list[str] = field(default_factory=list)  # extracted statements with no match


def load_gold(path: Path) -> list[GoldFact]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [GoldFact.model_validate(d) for d in data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score(
    extracted: Iterable[FactRow],
    gold: Iterable[GoldFact],
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> EvalReport:
    """A gold fact matches an extracted fact when: kind matches AND every `must_match`
    regex (case-insensitive) hits the extracted statement AND cosine(embed(gold.statement),
    embed(extracted.statement)) >= COSINE_FLOOR. Greedy: each gold fact is matched against
    the first still-unused qualifying extracted fact, in gold-list order.
    """
    embed_fn = embed_fn or get_backend().embed_docs
    gold_list = list(gold)
    extracted_list = list(extracted)

    gold_vecs = embed_fn([g.statement for g in gold_list]) if gold_list else []
    ext_vecs = embed_fn([e.statement for e in extracted_list]) if extracted_list else []

    compiled_patterns = [[re.compile(p, re.IGNORECASE) for p in g.must_match] for g in gold_list]

    used: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    for gi, g in enumerate(gold_list):
        for ei, e in enumerate(extracted_list):
            if ei in used:
                continue
            if str(e.kind) != str(g.kind):
                continue
            if not all(p.search(e.statement) for p in compiled_patterns[gi]):
                continue
            if gold_vecs and ext_vecs and _cosine(gold_vecs[gi], ext_vecs[ei]) < COSINE_FLOOR:
                continue
            used.add(ei)
            matched_pairs.append((gi, ei))
            break

    matched_gold = {gi for gi, _ in matched_pairs}
    misses = [gold_list[gi].statement for gi in range(len(gold_list)) if gi not in matched_gold]
    spurious = [extracted_list[ei].statement for ei in range(len(extracted_list)) if ei not in used]

    matched = len(matched_pairs)
    precision = matched / len(extracted_list) if extracted_list else 0.0
    recall = matched / len(gold_list) if gold_list else 0.0

    by_kind: dict[str, KindBreakdown] = {}
    all_kinds = {str(g.kind) for g in gold_list} | {str(e.kind) for e in extracted_list}
    for kind in sorted(all_kinds):
        k_gold_idx = [gi for gi, g in enumerate(gold_list) if str(g.kind) == kind]
        k_ext_idx = [ei for ei, e in enumerate(extracted_list) if str(e.kind) == kind]
        k_matched = sum(1 for gi, _ei in matched_pairs if gi in k_gold_idx)
        kg, ke = len(k_gold_idx), len(k_ext_idx)
        kp = k_matched / ke if ke else 0.0
        kr = k_matched / kg if kg else 0.0
        by_kind[kind] = KindBreakdown(gold=kg, extracted=ke, matched=k_matched, precision=kp, recall=kr, f1=_f1(kp, kr))

    return EvalReport(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        by_kind=by_kind,
        misses=misses,
        spurious=spurious,
    )


def _demo_sources(corpus_dir: Path) -> list[list[Any]]:
    """Every RawItem group in the demo corpus, kept separate by conversation so windows()
    (called inside extract_batch) never mixes unrelated threads."""
    from locket.models import RawItem  # local import: type hint only, avoids a cycle risk

    groups: list[list[RawItem]] = [
        list(parse_whatsapp(corpus_dir / "whatsapp" / "team.txt", thread="team")),
        list(parse_whatsapp(corpus_dir / "whatsapp" / "sarah.txt", thread="sarah")),
        list(parse_sms_xml(corpus_dir / "sms" / "backup.xml")),
    ]
    ig_inbox = corpus_dir / "instagram" / "inbox"
    if ig_inbox.is_dir():
        for thread_dir in sorted(p for p in ig_inbox.iterdir() if p.is_dir()):
            groups.append(list(parse_instagram_thread(thread_dir)))
    return groups


def run_extraction_pipeline(store: Store, corpus_dir: Path, *, model: Any | None = None) -> list[FactRow]:
    """Ingests every demo_corpus text source, runs extraction per-thread, persists each
    fact via store.add_fact, and returns the resulting FactRows (built locally from the
    known fields rather than re-querying -- add_fact already returns everything needed)."""
    backend = get_backend()
    rows: list[FactRow] = []

    for items in _demo_sources(corpus_dir):
        if not items:
            continue
        store.add_raw_items(items)
        for extracted_fact, provenance in extract_batch(items, model=model):
            fact = Fact(
                kind=extracted_fact.kind,
                statement=extracted_fact.statement,
                confidence=extracted_fact.confidence,
                subjects=extracted_fact.subjects,
                place=extracted_fact.place,
                happened_at=extracted_fact.happened_at,
                provenance=provenance,
            )
            embedding = backend.embed_docs([fact.statement])[0]
            fact_id = store.add_fact(fact, embedding)
            rows.append(
                FactRow(
                    id=fact_id,
                    kind=str(fact.kind),
                    statement=fact.statement,
                    confidence=fact.confidence,
                    entity_ids=fact.entity_ids,
                    provenance=fact.provenance,
                    happened_at=fact.happened_at,
                    valid_at=None,
                    invalid_at=None,
                )
            )
    return rows


__all__ = [
    "COSINE_FLOOR",
    "EvalReport",
    "GoldFact",
    "KindBreakdown",
    "load_gold",
    "run_extraction_pipeline",
    "score",
]
