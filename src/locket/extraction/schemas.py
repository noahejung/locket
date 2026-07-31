"""Extraction-time fact schemas. Strict (extra="forbid") — unlike the permissive adapters,
these go straight to the Claude structured-output API and back, so an unexpected key from a
model that's drifted off-contract should fail loudly, not get silently absorbed.

FactKind is defined once in locket.models (models.py owns shared types) and imported here
rather than redefined — extraction/schemas.py's ExtractedFact and the store-time
locket.models.Fact (Task 8) speak the same kind vocabulary.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from locket.models import FactKind

__all__ = ["FactKind", "ExtractedFact", "ExtractionResult"]


class ExtractedFact(BaseModel):
    """One atomic, self-contained statement about the user's life."""

    model_config = ConfigDict(extra="forbid")

    kind: FactKind
    statement: str = Field(description="One-sentence natural-language rendering, standalone")
    subjects: list[str] = Field(description="Display names of people involved, as written in the source")
    place: str | None = None
    happened_at: str | None = Field(None, description="ISO date or date range if the fact is time-bound")
    confidence: float = Field(ge=0, le=1)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedFact]
    notes: str | None = None  # extractor's uncertainty notes — logged, never stored as facts
