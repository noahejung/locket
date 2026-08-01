"""Shared corpus-walk + extract-and-persist pipeline path.

Fix-wave-2 item 4 (code-quality review nice-to-have: "extraction_eval.py never swapped to
the shared pipeline -- two orchestration copies drifting"). Before this module existed,
cli.py's `_run_pipeline` and evals/extraction_eval.py's `run_extraction_pipeline` each had
their own copy of "walk the corpus directory into (label, items) groups, then for each
group's items run extract_batch, build a Fact, embed its statement, and persist it" --
independently maintained, silently able to drift apart. This module is the one
implementation of that shared core; cli.py adds vision/idempotency-watermark/resolution/
profile orchestration around it, and evals/extraction_eval.py calls it directly for
scoring, with no orchestration of its own layered on top.

Boundary: no SQL here -- store.py stays the only module that talks to Postgres; this module
only calls Store's already-public methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locket.adapters.instagram import parse_instagram_thread
from locket.adapters.photos import parse_photos
from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.embeddings import get_backend
from locket.extraction.graph import extract_batch
from locket.models import Fact, RawItem
from locket.store import FactRow, Store


def discover_corpus_sources(corpus_dir: Path) -> tuple[list[tuple[str, list[RawItem]]], list[str]]:
    """(label, items) pairs, one per conversation/source, kept separate so windows() (inside
    extract_batch) never mixes unrelated threads -- chunking.windows() splits purely on
    time gaps, with no notion of "thread", so merging distinct conversations before
    windowing would let unrelated threads bleed into the same extraction window. Walks
    whatever files exist under corpus_dir/{whatsapp,sms,instagram/inbox,photos} rather than
    hardcoding filenames -- a real corpus's whatsapp/ dir won't be named team.txt/sarah.txt.
    Second element: every adapter parse warning collected across the whole corpus."""
    groups: list[tuple[str, list[RawItem]]] = []
    warnings: list[str] = []

    wa_dir = corpus_dir / "whatsapp"
    if wa_dir.is_dir():
        for txt_path in sorted(wa_dir.glob("*.txt")):
            groups.append(
                (
                    f"whatsapp:{txt_path.stem}",
                    list(parse_whatsapp(txt_path, thread=txt_path.stem, warnings=warnings)),
                )
            )

    sms_dir = corpus_dir / "sms"
    if sms_dir.is_dir():
        for xml_path in sorted(sms_dir.glob("*.xml")):
            groups.append((f"sms:{xml_path.stem}", list(parse_sms_xml(xml_path, warnings=warnings))))

    ig_inbox = corpus_dir / "instagram" / "inbox"
    if ig_inbox.is_dir():
        for thread_dir in sorted(p for p in ig_inbox.iterdir() if p.is_dir()):
            groups.append((f"instagram:{thread_dir.name}", list(parse_instagram_thread(thread_dir))))

    photos_dir = corpus_dir / "photos"
    if photos_dir.is_dir():
        groups.append(("photos", list(parse_photos(photos_dir, warnings=warnings))))

    return groups, warnings


@dataclass
class ExtractAndPersistResult:
    """extract_and_persist's full return: every (FactRow, subjects) pair persisted this
    call, plus extract_batch's retry/give-up/escalation counters bubbled through unchanged
    -- this function makes extract_batch's one call in the whole codebase, so it's the only
    place those counters can be picked up on their way to cli.py's per-run JSONL capture
    (`locket stats`, metrics.md §1/§5). `rows`, not `facts` (extract_batch's own field name)
    -- elements here are already-persisted FactRow, not the pre-persistence ExtractedFact.

    `given_up_window_hashes` is extract_batch's own field of the same name, bubbled through
    unchanged (fix-wave-3 follow-up to the 2026-08-01 catch-up review's MEDIUM finding) --
    cli.py's `_run_pipeline` is this function's only caller that cares about it (to call
    Store.mark_windows_given_up instead of mark_windows_extracted for exactly those windows);
    evals/extraction_eval.py's run_extraction_pipeline, this module's other consumer, has no
    idempotency watermark of its own and simply discards it."""

    rows: list[tuple[FactRow, list[str]]]
    retries: int
    give_ups: int
    escalations: int
    given_up_window_hashes: list[str]


def extract_and_persist(
    store: Store,
    items: list[RawItem],
    *,
    model: Any | None = None,
    windows_override: list[list[RawItem]] | None = None,
) -> ExtractAndPersistResult:
    """Runs extract_batch over `items`, persists each resulting fact via store.add_fact,
    and returns (FactRow, subjects) pairs. Subjects ride alongside the row because
    FactRow itself only carries entity_ids (populated later, by resolution) -- callers that
    need the raw extracted subject names for resolution (cli.py's `pipeline run`) read them
    from here instead of re-deriving them; callers that don't (evals/extraction_eval.py's
    scoring harness) just discard the second element.

    `windows_override`, if given, is passed straight through to extract_batch -- lets a
    caller (cli.py's `_run_pipeline`) supply an already-filtered subset (e.g. excluding
    windows a prior run already extracted) without this module needing to know anything
    about idempotency watermarking itself.

    Embeds every fact statement from this call in ONE embed_docs batch (fix-wave-2 item 5)
    instead of one embed_docs([statement]) call per fact -- both prior copies of this logic
    (cli.py's _run_pipeline, evals/extraction_eval.py's run_extraction_pipeline) made a
    separate model call per fact before this module unified them.
    """
    backend = get_backend()
    batch = extract_batch(items, model=model, windows_override=windows_override)
    if not batch.facts:
        return ExtractAndPersistResult(
            rows=[],
            retries=batch.retries,
            give_ups=batch.give_ups,
            escalations=batch.escalations,
            given_up_window_hashes=batch.given_up_window_hashes,
        )

    facts = [
        Fact(
            kind=extracted_fact.kind,
            statement=extracted_fact.statement,
            confidence=extracted_fact.confidence,
            subjects=extracted_fact.subjects,
            place=extracted_fact.place,
            happened_at=extracted_fact.happened_at,
            provenance=provenance,
        )
        for extracted_fact, provenance in batch.facts
    ]
    embeddings = backend.embed_docs([fact.statement for fact in facts])

    out: list[tuple[FactRow, list[str]]] = []
    for (extracted_fact, _provenance), fact, embedding in zip(batch.facts, facts, embeddings, strict=True):
        fact_id = store.add_fact(fact, embedding)
        row = FactRow(
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
        out.append((row, extracted_fact.subjects))
    return ExtractAndPersistResult(
        rows=out,
        retries=batch.retries,
        give_ups=batch.give_ups,
        escalations=batch.escalations,
        given_up_window_hashes=batch.given_up_window_hashes,
    )


__all__ = ["ExtractAndPersistResult", "discover_corpus_sources", "extract_and_persist"]
