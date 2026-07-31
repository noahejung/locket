"""Tiered entity resolution + confirm queue (graphiti's pattern, minus the graph db).

Tier 1: embedding top-15 against `entities` (SIMILARITY_FLOOR).
Tier 2: deterministic rules on the tier-1 candidates -- casefold/de-emoji/de-punctuate
        exact match (name or alias), then first-name-prefix match when unambiguous
        within the candidate set.
Tier 3: one haiku structured-output call per still-ambiguous candidate (same/different +
        confidence). A verdict >= LLM_CONFIRM_THRESHOLD auto-merges; every remaining
        "same" verdict below that bar becomes a MergeProposal in the confirm queue instead
        of a silent merge.

Face-cluster labeling (`label_face_cluster` / the `face:<id>` mention prefix) is a
DELIBERATELY separate path from the tiers above: a cluster id has no semantic relationship
to any name string, so embedding similarity can never surface it as a tier-1 candidate.
It resolves by exact alias lookup instead, short-circuiting tiers 1-3 entirely.

Similarity floor calibration note (measured 2026-07-30, mirrors the Week 2 SigLIP2
finding): SIMILARITY_FLOOR=0.6 cleanly separates unrelated names (~0.42-0.47 cosine
similarity between e.g. "Sarah Mendes" and "Kathryn Petrovic") from same-person noisy
variants (~0.57-0.90), but a few same-person pairs measured right at or just under the
floor on this corpus + this embedding model (e.g. "Cory Davis" vs its Instagram handle
"corydavis.photo" measured 0.571; "Kathryn Petrovic" vs its SMS contact name "Kat P" with
an emoji measured 0.595). Those cases don't clear tier 1 as candidates at all and are NOT
a resolution bug -- they're simply unreachable by this mention alone until a stronger
signal (a shared face cluster, a shared thread, or a future mention using different
phrasing) puts them in range. The floor is deliberately conservative: false-negative
(unresolved, ends up creating a duplicate entity a human can merge later) is a cheaper
mistake than false-positive (two different people silently merged).
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from locket.embeddings import get_backend
from locket.store import EntityRow, MergeProposal, Store

SIMILARITY_FLOOR = 0.6
LLM_CONFIRM_THRESHOLD = 0.85
HAIKU_MODEL_ID = "claude-haiku-4-5"

_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # main emoji block: pictographs, transport, supplemental symbols
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002b00-\U00002bff"  # misc symbols and arrows (e.g. ⭐ star)
    "\U0000fe0f"  # variation selector-16 (emoji presentation)
    "\U0000200d"  # zero-width joiner (multi-codepoint emoji)
    "]+",
    flags=re.UNICODE,
)
_SEPARATOR_RE = re.compile(r"[._\-]+")


class MatchVerdict(BaseModel):
    """Tier-3 structured output: is `mention` the same entity as `candidate`?"""

    same: bool
    confidence: float = Field(ge=0, le=1)


def _normalize(name: str) -> str:
    """casefold + strip emoji + treat handle-style separators (. _ -) as whitespace, so
    "sarah.mendes" and "Sarah Mendes" normalize identically."""
    stripped = _EMOJI_RE.sub("", name)
    stripped = _SEPARATOR_RE.sub(" ", stripped)
    stripped = unicodedata.normalize("NFKC", stripped)
    return " ".join(stripped.casefold().split())


def _first_token(name: str) -> str:
    normalized = _normalize(name)
    return normalized.split(" ")[0] if normalized else ""


def _prefix_related(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def _exact_match(mention: str, candidate: EntityRow) -> bool:
    norm_mention = _normalize(mention)
    if not norm_mention:
        return False
    return any(_normalize(n) == norm_mention for n in (candidate.name, *candidate.aliases))


def _verdict_prompt(mention: str, candidate: EntityRow) -> str:
    aliases = ", ".join(candidate.aliases) if candidate.aliases else "(none)"
    return (
        "Two names might refer to the same real person. Judge whether they do.\n"
        f'Mention: "{mention}"\n'
        f'Known entity: "{candidate.name}" (known aliases: {aliases})\n'
        "Respond with `same` (true/false) and a `confidence` in [0, 1]."
    )


@cache
def _default_model() -> Any:
    return ChatAnthropic(model=HAIKU_MODEL_ID).with_structured_output(MatchVerdict)


def label_face_cluster(store: Store, cluster_id: int, entity_name: str, *, kind: str = "person") -> str:
    """Assigns a face cluster to an entity (creating it if it doesn't exist yet), tagging
    the entity with a `face:<cluster_id>` alias. A later `resolve()` call with the mention
    `f"face:{cluster_id}"` resolves straight to this entity, bypassing tiers 1-3 -- see
    module docstring for why embedding similarity can't do this on its own."""
    backend = get_backend()
    entity_id = store.upsert_entity(entity_name, kind, backend.embed_docs([entity_name])[0])
    store.add_entity_alias(entity_id, f"face:{cluster_id}")
    return entity_id


def resolve(store: Store, mentions: list[str], *, model: Any | None = None) -> dict[str, str]:
    """Batched: mention -> entity_id. Ambiguous mentions that land in the confirm queue
    (see MergeProposal / pending_confirmations) are simply absent from the returned dict --
    they are not yet resolved, by design; a silent guess is worse than no answer."""
    backend = get_backend()
    resolved: dict[str, str] = {}
    for mention in mentions:
        entity_id = _resolve_one(store, mention, backend, model=model)
        if entity_id is not None:
            resolved[mention] = entity_id
    return resolved


def _resolve_one(store: Store, mention: str, backend: Any, *, model: Any | None) -> str | None:
    if mention.startswith("face:"):
        return store.find_entity_by_alias(mention)

    embedding = backend.embed_query(mention)
    nearest = store.nearest_entities(embedding, k=15)
    candidates = [e for e in nearest if e.similarity >= SIMILARITY_FLOOR]

    if not candidates:
        return store.upsert_entity(mention, "person", backend.embed_docs([mention])[0])

    exact = [c for c in candidates if _exact_match(mention, c)]
    if len(exact) == 1:
        return exact[0].id

    mention_first = _first_token(mention)
    prefix_matches = [c for c in candidates if _prefix_related(mention_first, _first_token(c.name))]
    if len(prefix_matches) == 1:
        return prefix_matches[0].id

    return _escalate(store, mention, candidates, model=model)


def _escalate(store: Store, mention: str, candidates: list[EntityRow], *, model: Any | None) -> str | None:
    active_model = model if model is not None else _default_model()
    verdicts: list[tuple[EntityRow, MatchVerdict]] = []
    for candidate in candidates:
        verdict = active_model.invoke(_verdict_prompt(mention, candidate))
        verdicts.append((candidate, verdict))

    same_verdicts = [(c, v) for c, v in verdicts if v.same]
    if same_verdicts:
        best_candidate, best_verdict = max(same_verdicts, key=lambda cv: cv[1].confidence)
        if best_verdict.confidence >= LLM_CONFIRM_THRESHOLD:
            return best_candidate.id

    # Not confident enough to auto-merge -- queue every plausible ("same") candidate for a
    # human y/n instead of silently merging or silently dropping the mention.
    for candidate, verdict in same_verdicts:
        store.add_merge_proposal(
            mention,
            candidate.id,
            evidence=f"LLM verdict: same={verdict.same} confidence={verdict.confidence:.2f}",
            score=verdict.confidence,
        )
    return None


def pending_confirmations(store: Store) -> list[MergeProposal]:
    return store.pending_merge_proposals()


__all__ = [
    "HAIKU_MODEL_ID",
    "LLM_CONFIRM_THRESHOLD",
    "SIMILARITY_FLOOR",
    "MatchVerdict",
    "label_face_cluster",
    "pending_confirmations",
    "resolve",
]
