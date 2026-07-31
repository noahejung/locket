"""EmbeddingBackend protocol + local sentence-transformers default.

Local default: Snowflake/snowflake-arctic-embed-s, 384 dims, best MTEB retrieval in its size
class (51.98 NDCG@10). Asymmetric by design — docs and queries use different encode paths;
this is the easy-to-miss bug this module exists to wire correctly once.
"""

from __future__ import annotations

from functools import cache
from typing import Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingBackend(Protocol):
    dims: int

    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformersBackend:
    """Local sentence-transformers backend. Docs are encoded plain; queries are encoded with
    the model's `query` prompt — arctic-embed-s is trained asymmetrically and mixing the two
    paths silently degrades retrieval quality."""

    dims = 384
    _MODEL_NAME = "Snowflake/snowflake-arctic-embed-s"

    def __init__(self) -> None:
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vecs = self._model.encode([text], prompt_name="query", normalize_embeddings=True)
        return vecs[0].tolist()


@cache
def _cached_local_backend() -> SentenceTransformersBackend:
    return SentenceTransformersBackend()


def get_backend(name: str = "local") -> EmbeddingBackend:
    if name == "local":
        return _cached_local_backend()
    raise ValueError(f"unknown embedding backend: {name!r}")
