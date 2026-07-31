"""Embedding backend tests — downloads Snowflake/snowflake-arctic-embed-s on first run.

Marked `vision` per PLAN.md Task 9 (the `vision` marker covers "needs local ML models
downloaded", not literal computer vision). Excluded from the default `-x -q` sweep.
"""

from __future__ import annotations

import math

import pytest

from locket.embeddings import get_backend

pytestmark = pytest.mark.vision


def test_local_backend_dims():
    backend = get_backend("local")
    assert backend.dims == 384


def test_embed_docs_shape_and_l2_normalized():
    backend = get_backend("local")
    vecs = backend.embed_docs(["a", "b"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 384
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_embed_query_is_asymmetric_vs_embed_docs():
    backend = get_backend("local")
    text = "coffee shop in Boston"
    doc_vec = backend.embed_docs([text])[0]
    query_vec = backend.embed_query(text)
    assert doc_vec != query_vec  # asymmetric encoding — the easy-to-miss bug
