from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from locket.extraction.schemas import ExtractedFact, ExtractionResult
from locket.models import FactKind


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ExtractedFact(
            kind=FactKind.event,
            statement="John and Sarah had dinner",
            subjects=["John", "Sarah"],
            confidence=1.5,
        )


def test_extraction_result_schema_is_json_serializable():
    schema = ExtractionResult.model_json_schema()
    # This exact schema is what gets sent to the Claude structured-output API — must
    # round-trip through json.dumps without error.
    dumped = json.dumps(schema)
    assert isinstance(dumped, str)
    assert "facts" in schema["properties"]


def test_unknown_extra_key_rejected():
    """Extraction models are strict (extra='forbid') — unlike the permissive adapters."""
    payload = {
        "kind": "event",
        "statement": "John and Sarah had dinner",
        "subjects": ["John", "Sarah"],
        "confidence": 0.9,
        "unexpected_field": "should not be allowed",
    }
    with pytest.raises(ValidationError):
        ExtractedFact(**payload)


def test_valid_extracted_fact_round_trips():
    fact = ExtractedFact(
        kind=FactKind.event,
        statement="John and Sarah had dinner in Boston",
        subjects=["John", "Sarah"],
        place="Boston",
        happened_at="2025-01-01",
        confidence=0.9,
    )
    result = ExtractionResult(facts=[fact], notes="one clean extraction")
    again = ExtractionResult.model_validate(result.model_dump())
    assert again.facts[0].statement == fact.statement
    assert again.notes == "one clean extraction"


def test_extraction_result_defaults_notes_to_none():
    result = ExtractionResult(facts=[])
    assert result.notes is None
