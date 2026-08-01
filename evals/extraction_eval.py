"""Extraction eval: self-labeled gold set vs. real pipeline output.

`score()` matches extracted facts against the gold set by kind + regex anchors + a cosine
similarity floor (the plan's explicit "guard against regex-lucky nonsense"). Matching is
greedy bipartite: each gold fact consumes at most one extracted fact, so precision/recall
counts never double-count a single extracted fact against two gold facts.

`run_extraction_pipeline` calls locket.pipeline's shared corpus-walk (discover_corpus_sources)
+ extract-and-persist (extract_and_persist) functions -- fix-wave-2 item 4. This module used
to keep its own separate copy of both (a hardcoded whatsapp/sms/instagram-only walker plus a
duplicate extract_batch -> store.add_fact loop) that could silently drift from cli.py's
`pipeline run`; now there is exactly one implementation of that shared core. What stays
unique to this module: no vision pre-pass, no idempotency watermark, no entity resolution,
no profile synthesis -- `pipeline run`-only concerns this scoring harness doesn't need.
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

from locket.embeddings import get_backend
from locket.models import FactKind
from locket.pipeline import discover_corpus_sources, extract_and_persist
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


def run_extraction_pipeline(store: Store, corpus_dir: Path, *, model: Any | None = None) -> list[FactRow]:
    """Ingests every corpus source (locket.pipeline.discover_corpus_sources), runs
    extraction per source group, and persists each fact via locket.pipeline.extract_and_persist
    -- returns the resulting FactRows. See module docstring: this is the shared pipeline
    core, not a separate copy of it."""
    groups, _warnings = discover_corpus_sources(corpus_dir)
    rows: list[FactRow] = []

    for _label, items in groups:
        if not items:
            continue
        store.add_raw_items(items)
        rows.extend(row for row, _subjects in extract_and_persist(store, items, model=model).rows)
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
