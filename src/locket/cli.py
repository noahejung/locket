"""locket CLI: `locket ingest / pipeline run / pipeline retry-given-up / label-faces /
resolve / eval / profile / stats / serve / serve-ui`.

argparse only, no CLI framework dependency, per PLAN.md Task 19. Heavy-dependency imports
(vision models, evals/extraction_eval.py, evals/rag_eval.py) are deferred to the function
that actually needs them, so `locket --help` or `locket ingest <path>` doesn't eagerly import
torch/insightface/ragas -- matches the deferred-import convention already used in
evals/rag_eval.py's `_build_judge`/`_build_ragas_embeddings`.

Every stage function that can call an LLM takes an injectable `model=`-shaped keyword,
threaded down from `main()`'s own keywords -- mirrors the `model=` test seam used throughout
extraction/graph.py, resolution.py, profile.py, evals/rag_eval.py. Production callers
(the real `python -m locket.cli ...` invocation) leave them at their live defaults; tests
inject stubs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from locket.adapters.instagram import parse_instagram_thread
from locket.adapters.photos import parse_photos
from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.config import Settings
from locket.extraction.chunking import windows
from locket.extraction.graph import window_hash
from locket.llm import model_name, resolve_backend
from locket.models import RawItem
from locket.pipeline import discover_corpus_sources, extract_and_persist
from locket.resolution import pending_confirmations, resolve
from locket.stats import RunRecord, append_run_record, read_last_run_record
from locket.store import Store

# ~135s/image measured mean latency for the vision-LLM tail on this CPU-only machine
# (evals/BASELINE.md, Task 13) -- 20 images is ~45 minutes serial, a reasonable default for
# a backgrounded/overnight `pipeline run`; raise --cap explicitly for a fuller pass.
DEFAULT_VISION_CAP = 20

_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


# ---------------------------------------------------------------------------
# Source discovery / ingestion
# ---------------------------------------------------------------------------


def _ingest_source(path: Path) -> tuple[list[RawItem], list[str]]:
    """Adapter auto-detection by shape (plan's exact rule): ".txt" -> whatsapp, ".xml" ->
    sms, a directory containing message_*.json -> instagram, any other directory -> photos.
    Second element is any non-fatal parse warnings the adapter collected (e.g. whatsapp's
    unmatched-line count) -- callers surface these; a silently-empty result is exactly what
    the warnings exist to distinguish from "genuinely nothing here"."""
    warnings: list[str] = []
    if path.is_dir():
        if any(path.glob("message_*.json")):
            return list(parse_instagram_thread(path)), warnings
        return list(parse_photos(path, warnings=warnings)), warnings
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return list(parse_whatsapp(path, thread=path.stem, warnings=warnings)), warnings
    if suffix == ".xml":
        return list(parse_sms_xml(path, warnings=warnings)), warnings
    raise ValueError(f"don't know how to ingest {path} (expected .txt, .xml, or a directory)")


# ---------------------------------------------------------------------------
# Vision pre-pass (only reached when `pipeline run` is NOT given --skip-vision)
# ---------------------------------------------------------------------------


def _run_vision_prepass(photo_items: list[RawItem], photos_root: Path, *, cap: int) -> None:
    """Local pre-pass (SigLIP2 tags + RapidOCR + InsightFace face clusters) over every photo
    item, mutating each RawItem's `meta` in place so extraction's transcript renderer
    (graph.py's `_render_transcript`, which reads `meta["ocr_lines"]`/`meta["vision_tags"]`)
    picks the signal up. Then curates + runs the vision-LLM tail (local Ollama by default,
    per the hard privacy rule in vision_llm.py's module docstring) on a capped subset,
    attaching a `vision_llm` PhotoFacts summary to each selected item's meta.

    Only the local-Ollama default path is wired -- the Gemini (synthetic-corpus-only,
    opt-in) and Claude-API (--cloud-ok, paid fallback) alternates from Task 13's module
    docstring remain deferred, same as Task 13 itself documented ("neither alternate path
    has anywhere to attach a flag yet") -- this session's scope is the demo-corpus run, which
    never needs the real-photo cloud fallback.
    """
    from locket.vision.faces import cluster, embed_faces
    from locket.vision.ocr import ocr_image
    from locket.vision.tagger import tag_images
    from locket.vision.vision_llm import describe_photo, select_for_vision

    by_path: dict[Path, RawItem] = {}
    for item in photo_items:
        if item.media_path:
            by_path[photos_root / item.media_path] = item
    if not by_path:
        return
    paths = list(by_path)

    tags = tag_images(paths)
    for path, item in by_path.items():
        path_tags = tags.get(path, {})
        item.meta["vision_tags"] = sorted(path_tags, key=lambda label: path_tags[label], reverse=True)[:3]

    for path, item in by_path.items():
        ocr_lines = [line for line, _confidence in ocr_image(path)]
        if ocr_lines:
            item.meta["ocr_lines"] = ocr_lines

    hits = embed_faces(paths)
    for cluster_id, cluster_hits in cluster(hits).items():
        if cluster_id == -1:  # DBSCAN noise bucket -- a one-off face, not a real cluster
            continue
        for hit in cluster_hits:
            item = by_path.get(hit.path)
            if item is None:
                continue
            clusters = item.meta.setdefault("face_clusters", [])
            if cluster_id not in clusters:
                clusters.append(cluster_id)

    dates = {path: item.ts.date().isoformat() for path, item in by_path.items() if item.ts is not None}
    for path in select_for_vision(tags, cap=cap, dates=dates):
        item = by_path[path]
        try:
            facts = describe_photo(path)
        except Exception as exc:  # noqa: BLE001 - Ollama server may be down/unreachable; degrade, don't crash the pipeline
            item.meta["vision_llm_error"] = str(exc)
            continue
        item.meta["vision_llm"] = facts.model_dump()


# ---------------------------------------------------------------------------
# Pipeline orchestration: ingest -> (vision) -> extraction -> resolution -> profile
# ---------------------------------------------------------------------------


def _run_pipeline(
    store: Store,
    corpus_dir: Path,
    *,
    skip_vision: bool,
    cap: int,
    extraction_model: Any | None,
    resolve_model: Any | None,
    profile_model: Any | None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    from locket.profile import synthesize

    # --retry-failed (fix-wave-3 follow-up to the 2026-08-01 catch-up review's MEDIUM
    # finding): clear give-up rows BEFORE discovering pending windows below, so this same
    # run re-attempts them like any other not-yet-done window -- same effect as running
    # `pipeline retry-given-up` first, just inline. Global across the whole
    # extracted_windows table, same as the standalone command (not scoped to corpus_dir).
    given_up_cleared = store.clear_given_up_windows() if retry_failed else 0

    groups, warnings = discover_corpus_sources(corpus_dir)

    raw_inserted = 0
    facts_created = 0
    windows_skipped_extracted = 0
    windows_skipped_gave_up = 0
    windows_processed = 0
    retries = 0
    give_ups = 0
    escalations = 0
    fact_subjects: list[tuple[str, list[str]]] = []
    mentions: set[str] = set()

    for label, items in groups:
        if not items:
            continue
        if not skip_vision and label == "photos":
            _run_vision_prepass(items, corpus_dir / "photos", cap=cap)

        raw_inserted += store.add_raw_items(items)

        # Idempotency watermark (fix-wave-1 item 8b, outcome-split fix-wave-3): skip windows
        # a prior `pipeline run` already reached a terminal state for (either outcome) --
        # computed here (not inside extract_batch) so window boundaries stay stable across
        # runs; re-deriving windows() from a pruned item list could reshuffle them
        # (gap/order-sensitive). get_window_outcomes (not the older membership-only
        # get_extracted_window_hashes) so the skip count below can be split by outcome.
        item_windows = windows(items)
        hashes = [window_hash(w) for w in item_windows]
        outcomes = store.get_window_outcomes(hashes) if hashes else {}
        pending_windows = [w for w, h in zip(item_windows, hashes, strict=True) if h not in outcomes]
        pending_hashes = [h for h in hashes if h not in outcomes]
        windows_skipped_extracted += sum(1 for o in outcomes.values() if o == "extracted")
        windows_skipped_gave_up += sum(1 for o in outcomes.values() if o == "gave_up")
        windows_processed += len(pending_windows)

        persisted = extract_and_persist(
            store, items, model=extraction_model, windows_override=pending_windows
        )
        for row, subjects in persisted.rows:
            facts_created += 1
            fact_subjects.append((row.id, subjects))
            mentions.update(subjects)
        retries += persisted.retries
        give_ups += persisted.give_ups
        escalations += persisted.escalations

        # Split pending_hashes by THIS run's outcome -- a give-up must be recorded with
        # mark_windows_given_up (a distinct, later-retryable outcome), not
        # mark_windows_extracted, or it becomes exactly as permanently/silently unretryable
        # as the pre-fix version of this method made every give-up.
        gave_up_now = set(persisted.given_up_window_hashes)
        succeeded_hashes = [h for h in pending_hashes if h not in gave_up_now]
        if succeeded_hashes:
            store.mark_windows_extracted(succeeded_hashes)
        if gave_up_now:
            store.mark_windows_given_up(sorted(gave_up_now))

    if mentions:
        resolved = resolve(store, sorted(mentions), model=resolve_model)
        for fact_id, subjects in fact_subjects:
            entity_ids = sorted({resolved[s] for s in subjects if s in resolved})
            if entity_ids:
                store.update_fact(fact_id, entity_ids=entity_ids)

    synthesize(store, model=profile_model)

    return {
        "sources": len(groups),
        "raw_items_inserted": raw_inserted,
        "facts_created": facts_created,
        "mentions_seen": len(mentions),
        "windows_skipped": windows_skipped_extracted + windows_skipped_gave_up,
        "windows_skipped_extracted": windows_skipped_extracted,
        "windows_skipped_gave_up": windows_skipped_gave_up,
        "windows_processed": windows_processed,
        "given_up_cleared": given_up_cleared,
        "retries": retries,
        "give_ups": give_ups,
        "escalations": escalations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace, store: Store) -> int:
    path = Path(args.path)
    items, warnings = _ingest_source(path)
    inserted = store.add_raw_items(items)
    print(f"ingested {inserted} new raw item(s) from {path} ({len(items)} total in source)")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def _cmd_pipeline_run(
    args: argparse.Namespace,
    store: Store,
    settings: Settings,
    *,
    extraction_model: Any | None,
    resolve_model: Any | None,
    profile_model: Any | None,
) -> int:
    if (
        extraction_model is None
        and resolve_backend(settings) == "anthropic"
        and not settings.anthropic_api_key
    ):
        print(
            "LOCKET_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set -- extraction "
            "needs the Claude API and cannot run without it. Set it in .env (see "
            ".env.example), or unset LOCKET_LLM_BACKEND to run fully local via Ollama "
            "instead (default when no key is set).",
            file=sys.stderr,
        )
        return 1

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else (settings.corpus_dir or Path("demo_corpus"))

    # Per-run capture (locket stats, metrics.md §1/§5): wall time wraps the WHOLE pipeline
    # (vision pre-pass + extraction + resolution + profile synthesis), matching metrics.md
    # §4's "pipeline wall-time per run" definition. facts_added is measured as the facts
    # table's total row-count delta across the run (Store.fact_stats -- one cheap aggregate
    # query before, one after) rather than threaded up from extract_and_persist, because
    # "genuinely a new row" is exactly what add_fact's ON CONFLICT (hash) DO NOTHING
    # decides, and that decision isn't otherwise surfaced to callers; dedup_hits is then the
    # remainder (candidates processed this run that did NOT become a new row).
    facts_before = store.fact_stats().total
    started_at = time.monotonic()
    summary = _run_pipeline(
        store,
        corpus_dir,
        skip_vision=args.skip_vision,
        cap=args.cap,
        extraction_model=extraction_model,
        resolve_model=resolve_model,
        profile_model=profile_model,
        retry_failed=args.retry_failed,
    )
    wall_seconds = time.monotonic() - started_at
    facts_added = store.fact_stats().total - facts_before

    append_run_record(
        RunRecord(
            timestamp=datetime.now(UTC).isoformat(),
            backend=resolve_backend(settings),
            model=model_name("extraction_default", settings),
            windows_processed=summary["windows_processed"],
            windows_skipped=summary["windows_skipped"],
            windows_skipped_extracted=summary["windows_skipped_extracted"],
            windows_skipped_gave_up=summary["windows_skipped_gave_up"],
            facts_added=facts_added,
            dedup_hits=summary["facts_created"] - facts_added,
            retries=summary["retries"],
            give_ups=summary["give_ups"],
            escalations=summary["escalations"],
            wall_seconds=wall_seconds,
        )
    )

    print(json.dumps(summary, indent=2))
    for w in summary.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


def _cmd_pipeline_retry_given_up(store: Store) -> int:
    """The standalone escape hatch (fix-wave-3 follow-up to the 2026-08-01 catch-up
    review's MEDIUM finding): clears every extracted_windows row with outcome='gave_up' --
    successfully-extracted rows are untouched -- so the next `pipeline run` re-attempts
    exactly those windows instead of skipping them forever. `pipeline run --retry-failed`
    does the identical clear inline, for a one-command "retry and run now" workflow; this
    command is for "just clear them, I'll run the pipeline separately"."""
    cleared = store.clear_given_up_windows()
    print(f"cleared {cleared} given-up window(s) -- the next `pipeline run` will re-attempt them")
    return 0


def _cmd_label_faces(args: argparse.Namespace, store: Store) -> int:
    """Interactive: run the local face pre-pass over --photos-dir (default demo_corpus/
    photos), print each real cluster's size, prompt for a name (blank = skip), persist via
    resolution.label_face_cluster. Needs InsightFace's local models (@pytest.mark.vision in
    tests -- not part of the default suite)."""
    from locket.resolution import label_face_cluster
    from locket.vision.faces import cluster, embed_faces

    photos_dir = Path(args.photos_dir) if args.photos_dir else Path("demo_corpus/photos")
    paths = sorted(p for p in photos_dir.glob("*") if p.suffix.lower() in _PHOTO_SUFFIXES)
    hits = embed_faces(paths)
    clusters = cluster(hits)

    labeled = 0
    for cluster_id in sorted(cid for cid in clusters if cid != -1):
        cluster_hits = clusters[cluster_id]
        print(f"cluster {cluster_id}: {len(cluster_hits)} photo(s), e.g. {cluster_hits[0].path.name}")
        name = input("  name (blank to skip): ").strip()
        if name:
            label_face_cluster(store, cluster_id, name)
            labeled += 1
    print(f"labeled {labeled} cluster(s)")
    return 0


def _cmd_resolve(args: argparse.Namespace, store: Store) -> int:
    """Print the entity-resolution confirm queue for a human y/n. --yes/--no batch-decide
    every pending item non-interactively (useful for scripted demos/CI); without either,
    prompts per item."""
    proposals = pending_confirmations(store)
    if not proposals:
        print("no pending merge proposals")
        return 0
    for proposal in proposals:
        print(
            f"[{proposal.id}] {proposal.mention!r} == {proposal.candidate_entity_name!r}? "
            f"(score={proposal.score:.2f}) {proposal.evidence}"
        )
        if args.yes:
            decision = True
        elif args.no:
            decision = False
        else:
            decision = input("  confirm merge? [y/N] ").strip().lower() == "y"
        store.resolve_merge_proposal(proposal.id, accept=decision)
        print("  -> confirmed" if decision else "  -> rejected")
    return 0


def _cmd_eval_extraction(args: argparse.Namespace, store: Store, *, model: Any | None) -> int:
    from evals.extraction_eval import load_gold, run_extraction_pipeline, score

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path("demo_corpus")
    gold_path = Path(args.gold) if args.gold else Path("evals/gold/persona_gold.yaml")

    extracted = run_extraction_pipeline(store, corpus_dir, model=model)
    gold = load_gold(gold_path)
    report = score(extracted, gold)

    if args.json:
        print(
            json.dumps(
                {
                    "precision": report.precision,
                    "recall": report.recall,
                    "f1": report.f1,
                    "by_kind": {
                        kind: {"gold": kb.gold, "extracted": kb.extracted, "matched": kb.matched, "f1": kb.f1}
                        for kind, kb in report.by_kind.items()
                    },
                    "misses": report.misses,
                    "spurious": report.spurious,
                },
                indent=2,
            )
        )
    else:
        print(f"precision={report.precision:.3f} recall={report.recall:.3f} f1={report.f1:.3f}")
    return 0


def _cmd_eval_rag(args: argparse.Namespace, store: Store, *, answer_model: Any | None) -> int:
    from evals.extraction_eval import run_extraction_pipeline
    from evals.rag_eval import load_questions, run_rag_eval

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path("demo_corpus")
    questions_path = Path(args.questions) if args.questions else Path("evals/questions.yaml")

    run_extraction_pipeline(store, corpus_dir)
    questions = load_questions(questions_path)
    result = run_rag_eval(store, questions, answer_model=answer_model)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(
            f"faithfulness={result.faithfulness:.3f} "
            f"answer_relevancy={result.answer_relevancy:.3f} "
            f"context_precision={result.context_precision:.3f}"
        )
    return 0


def _cmd_profile_build(store: Store, *, model: Any | None) -> int:
    from locket.profile import synthesize

    body = synthesize(store, model=model)
    print(body)
    return 0


def _cmd_stats(args: argparse.Namespace, store: Store) -> int:
    """metrics.md §1/§5's `locket stats`: every §1 DB aggregate in one command, plus the
    last `pipeline run`'s captured JSONL line (if any exist yet) -- makes every §1 metric a
    one-command check without hand-querying Postgres."""
    raw_by_source = store.raw_item_counts_by_source()
    facts = store.fact_stats()
    entities = store.entity_count()
    queue = store.confirm_queue_stats()
    history = store.fact_history_event_counts()
    last_run = read_last_run_record()
    queue_age_seconds = (
        (datetime.now(UTC) - queue.oldest_created_at).total_seconds()
        if queue.oldest_created_at is not None
        else None
    )

    if args.json:
        print(
            json.dumps(
                {
                    "raw_items_by_source": raw_by_source,
                    "facts": {
                        "total": facts.total,
                        "mean_confidence": facts.mean_confidence,
                        "by_kind": {
                            kind: {"count": ks.count, "mean_confidence": ks.mean_confidence}
                            for kind, ks in facts.by_kind.items()
                        },
                    },
                    "entities": entities,
                    "confirm_queue": {
                        "depth": queue.depth,
                        "oldest_created_at": (
                            queue.oldest_created_at.isoformat() if queue.oldest_created_at else None
                        ),
                        "oldest_age_seconds": queue_age_seconds,
                    },
                    "fact_history_events": history,
                    "last_run": last_run,
                },
                indent=2,
            )
        )
        return 0

    print("raw items by source:")
    for source, count in sorted(raw_by_source.items()):
        print(f"  {source}: {count}")
    print(f"facts: {facts.total} total (mean confidence {facts.mean_confidence:.2f})")
    for kind, ks in sorted(facts.by_kind.items()):
        print(f"  {kind}: {ks.count} (mean confidence {ks.mean_confidence:.2f})")
    print(f"entities: {entities}")
    if queue.depth:
        print(f"confirm queue: {queue.depth} pending, oldest {queue_age_seconds:,.0f}s old")
    else:
        print("confirm queue: empty")
    print("fact history events:")
    if history:
        for event, count in sorted(history.items()):
            print(f"  {event}: {count}")
    else:
        print("  none yet")
    if last_run:
        print(f"last pipeline run: {json.dumps(last_run)}")
    else:
        print("last pipeline run: none recorded yet (run `locket pipeline run`)")
    return 0


def _cmd_serve(settings: Settings) -> int:
    """Unlike every other subcommand, `serve` bypasses main()'s shared owns_store
    try/finally entirely (it has no injectable `store=` seam -- a real serve blocks on
    stdio for the process lifetime) -- so it needs its own try/finally for the same
    close-on-every-exit-path guarantee, including when `mcp.run()` itself raises."""
    from locket.mcp_server import build_server

    store = Store(settings.db_url)
    try:
        mcp = build_server(store)
        mcp.run()
        return 0
    finally:
        store.close()


def _cmd_serve_ui(args: argparse.Namespace, settings: Settings) -> int:
    """Phone chat UI (Task, setup-guide.md Part 2 Option 2). Mirrors `_cmd_serve` exactly:
    `uvicorn.run(...)` blocks for the process lifetime same as `mcp.run()`, so this bypasses
    main()'s shared owns_store try/finally too, for the identical reason -- no injectable
    `store=` seam, own try/finally for close-on-every-exit-path including a raise from
    uvicorn itself.

    `args.host` defaults to 127.0.0.1 (argparse wiring below) -- never 0.0.0.0 by default.
    See webui.py's module docstring for why: answer_question reads a citable "profile of
    you", and 0.0.0.0 would expose it to the whole LAN, not just the intended
    phone-over-Tailscale path. Pass --host <your-tailscale-ip> explicitly for phone access.
    """
    import uvicorn

    from locket.webui import create_app

    store = Store(settings.db_url)
    try:
        app = create_app(store)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# argparse wiring + entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="locket", description="Privacy-first personal context engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse a single export file/dir into raw_items")
    p_ingest.add_argument("path")

    p_pipeline = sub.add_parser("pipeline", help="Pipeline commands")
    pipeline_sub = p_pipeline.add_subparsers(dest="pipeline_command", required=True)
    p_run = pipeline_sub.add_parser("run", help="vision pre-pass -> extraction -> resolution -> profile")
    p_run.add_argument("--skip-vision", action="store_true", help="skip local vision pre-pass + vision-LLM tail")
    p_run.add_argument("--cap", type=int, default=DEFAULT_VISION_CAP, help="vision-LLM tail image cap")
    p_run.add_argument("--corpus-dir", default=None, help="defaults to LOCKET_CORPUS_DIR, else ./demo_corpus")
    p_run.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "clear previously given-up-on windows before running, so this run re-attempts "
            "them (same effect as running `pipeline retry-given-up` first, inline)"
        ),
    )
    pipeline_sub.add_parser(
        "retry-given-up",
        help=(
            "clear extracted_windows rows recorded as given-up-on (not successfully "
            "extracted ones), so the next `pipeline run` re-attempts exactly those windows"
        ),
    )

    sub.add_parser("label-faces", help="Label detected face clusters interactively").add_argument(
        "--photos-dir", default=None
    )

    p_resolve = sub.add_parser("resolve", help="Review the entity-resolution confirm queue")
    p_resolve.add_argument("--yes", action="store_true", help="auto-confirm every pending proposal")
    p_resolve.add_argument("--no", action="store_true", help="auto-reject every pending proposal")

    p_eval = sub.add_parser("eval", help="Run an eval suite")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)
    p_eval_extraction = eval_sub.add_parser("extraction", help="Extraction P/R/F1 against the gold set")
    p_eval_extraction.add_argument("--json", action="store_true")
    p_eval_extraction.add_argument("--corpus-dir", default=None)
    p_eval_extraction.add_argument("--gold", default=None)
    p_eval_rag = eval_sub.add_parser("rag", help="Ragas faithfulness/relevancy/precision")
    p_eval_rag.add_argument("--json", action="store_true")
    p_eval_rag.add_argument("--corpus-dir", default=None)
    p_eval_rag.add_argument("--questions", default=None)

    p_profile = sub.add_parser("profile", help="Profile commands")
    profile_sub = p_profile.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("build", help="Synthesize and persist the living profile")

    p_stats = sub.add_parser("stats", help="Print pipeline/store health metrics (metrics.md §1)")
    p_stats.add_argument("--json", action="store_true")

    sub.add_parser("serve", help="Run the MCP stdio server")

    p_serve_ui = sub.add_parser("serve-ui", help="Serve the phone chat UI (tailnet-only web page)")
    p_serve_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "bind host -- defaults to loopback (127.0.0.1) and never binds 0.0.0.0 by "
            "default; pass your tailscale IP explicitly for phone access, e.g. "
            "--host 100.102.116.112 (see docs/demo.md)"
        ),
    )
    p_serve_ui.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")

    return parser


def main(
    argv: list[str] | None = None,
    *,
    store: Store | None = None,
    extraction_model: Any | None = None,
    resolve_model: Any | None = None,
    profile_model: Any | None = None,
    rag_answer_model: Any | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load()

    if args.command == "serve":
        return _cmd_serve(settings)
    if args.command == "serve-ui":
        return _cmd_serve_ui(args, settings)

    owns_store = store is None
    active_store = store if store is not None else Store(settings.db_url)
    try:
        if args.command == "ingest":
            return _cmd_ingest(args, active_store)
        if args.command == "pipeline" and args.pipeline_command == "run":
            return _cmd_pipeline_run(
                args,
                active_store,
                settings,
                extraction_model=extraction_model,
                resolve_model=resolve_model,
                profile_model=profile_model,
            )
        if args.command == "pipeline" and args.pipeline_command == "retry-given-up":
            return _cmd_pipeline_retry_given_up(active_store)
        if args.command == "label-faces":
            return _cmd_label_faces(args, active_store)
        if args.command == "resolve":
            return _cmd_resolve(args, active_store)
        if args.command == "eval" and args.eval_command == "extraction":
            return _cmd_eval_extraction(args, active_store, model=extraction_model)
        if args.command == "eval" and args.eval_command == "rag":
            return _cmd_eval_rag(args, active_store, answer_model=rag_answer_model)
        if args.command == "profile" and args.profile_command == "build":
            return _cmd_profile_build(active_store, model=profile_model)
        if args.command == "stats":
            return _cmd_stats(args, active_store)
        parser.error("unrecognized command")
        return 2
    finally:
        if owns_store:
            active_store.close()


if __name__ == "__main__":
    sys.exit(main())
