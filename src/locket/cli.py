"""locket CLI: `locket ingest / pipeline run / label-faces / resolve / eval / profile / serve`.

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
from pathlib import Path
from typing import Any

from locket.adapters.instagram import parse_instagram_thread
from locket.adapters.photos import parse_photos
from locket.adapters.sms_xml import parse_sms_xml
from locket.adapters.whatsapp import parse_whatsapp
from locket.config import Settings
from locket.extraction.chunking import windows
from locket.extraction.graph import window_hash
from locket.llm import resolve_backend
from locket.models import RawItem
from locket.pipeline import discover_corpus_sources, extract_and_persist
from locket.resolution import pending_confirmations, resolve
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
) -> dict[str, Any]:
    from locket.profile import synthesize

    groups, warnings = discover_corpus_sources(corpus_dir)

    raw_inserted = 0
    facts_created = 0
    windows_skipped = 0
    fact_subjects: list[tuple[str, list[str]]] = []
    mentions: set[str] = set()

    for label, items in groups:
        if not items:
            continue
        if not skip_vision and label == "photos":
            _run_vision_prepass(items, corpus_dir / "photos", cap=cap)

        raw_inserted += store.add_raw_items(items)

        # Idempotency watermark (fix-wave-1 item 8b): skip windows a prior `pipeline run`
        # already extracted -- computed here (not inside extract_batch) so window
        # boundaries stay stable across runs; re-deriving windows() from a pruned item list
        # could reshuffle them (gap/order-sensitive).
        item_windows = windows(items)
        hashes = [window_hash(w) for w in item_windows]
        already_done = store.get_extracted_window_hashes(hashes) if hashes else set()
        pending_windows = [w for w, h in zip(item_windows, hashes, strict=True) if h not in already_done]
        pending_hashes = [h for h in hashes if h not in already_done]
        windows_skipped += len(hashes) - len(pending_hashes)

        for row, subjects in extract_and_persist(
            store, items, model=extraction_model, windows_override=pending_windows
        ):
            facts_created += 1
            fact_subjects.append((row.id, subjects))
            mentions.update(subjects)

        if pending_hashes:
            store.mark_windows_extracted(pending_hashes)

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
        "windows_skipped": windows_skipped,
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
    summary = _run_pipeline(
        store,
        corpus_dir,
        skip_vision=args.skip_vision,
        cap=args.cap,
        extraction_model=extraction_model,
        resolve_model=resolve_model,
        profile_model=profile_model,
    )
    print(json.dumps(summary, indent=2))
    for w in summary.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
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

    sub.add_parser("serve", help="Run the MCP stdio server")

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
        parser.error("unrecognized command")
        return 2
    finally:
        if owns_store:
            active_store.close()


if __name__ == "__main__":
    sys.exit(main())
