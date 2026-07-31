"""InsightFace face detection, embedding, and DBSCAN clustering.

buffalo_l weights are licensed non-commercial/research-personal use only (code is MIT) —
see THIRD_PARTY_NOTICES.md. Fine for this personal tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


@dataclass
class FaceHit:
    path: Path
    bbox: tuple[float, float, float, float]
    embedding: list[float]  # normed_embedding — NOT raw .embedding (upstream issue #2256)


@cache
def _get_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def embed_faces(paths: list[Path]) -> list[FaceHit]:
    app = _get_app()
    hits: list[FaceHit] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        for face in app.get(img):
            hits.append(
                FaceHit(
                    path=path,
                    bbox=tuple(float(x) for x in face.bbox.tolist()),
                    embedding=face.normed_embedding.tolist(),
                )
            )
    return hits


def cluster(hits: list[FaceHit]) -> dict[int, list[FaceHit]]:
    """DBSCAN over cosine distance. Label -1 is DBSCAN's noise bucket — a face with no
    close match, which is semantics (a one-off / stranger), not a clustering failure.
    eps=0.4 matches InsightFace's own album tool (~0.48 similarity threshold), min_samples=2."""
    if not hits:
        return {}
    x = np.array([h.embedding for h in hits])
    labels = DBSCAN(eps=0.4, min_samples=2, metric="cosine").fit_predict(x)
    out: dict[int, list[FaceHit]] = {}
    for label, hit in zip(labels, hits, strict=True):
        out.setdefault(int(label), []).append(hit)
    return out
