"""SigLIP2 zero-shot tagging. Each label is an independent yes/no via sigmoid — NOT a
softmax distribution, so scores do not sum to 1 and multiple labels can score high (or, for
subtle/plain images against a generic one-line prompt, all labels can score modestly below
0.5 while still ranking correctly — verified empirically against demo_corpus/photos: the
correct label is the argmax in every case tested even when its absolute score is ~0.03-0.13).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MODEL_NAME = "google/siglip2-base-patch16-224"

LABELS = [
    "screenshot",
    "receipt",
    "document",
    "food",
    "people",
    "selfie",
    "landscape",
    "pet",
    "event",
    "indoor scene",
]

_PROMPT_TEMPLATE = "This is a photo of {}."


class _Tagger:
    def __init__(self, attn_implementation: str = "sdpa") -> None:
        self.processor = AutoProcessor.from_pretrained(_MODEL_NAME)
        self.model = AutoModel.from_pretrained(_MODEL_NAME, attn_implementation=attn_implementation)
        self.model.eval()
        # Cache label text-embeddings once at init — halves per-image cost (plan Task 10).
        prompts = [_PROMPT_TEMPLATE.format(label) for label in LABELS]
        text_inputs = self.processor(
            text=prompts, padding="max_length", max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            text_out = self.model.get_text_features(**text_inputs)
        text_features = text_out.pooler_output
        self._text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def tag(self, paths: list[Path]) -> dict[Path, dict[str, float]]:
        images = [Image.open(p).convert("RGB") for p in paths]
        image_inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_out = self.model.get_image_features(**image_inputs)
        image_features = image_out.pooler_output
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logit_bias = self.model.logit_bias
        logits = image_features @ self._text_features.T * logit_scale + logit_bias
        probs = torch.sigmoid(logits)

        return {
            path: {label: probs[i, j].item() for j, label in enumerate(LABELS)}
            for i, path in enumerate(paths)
        }


@cache
def _get_tagger() -> _Tagger:
    # sdpa vs eager benchmarked on 10 demo_corpus images (see agent-report) — sdpa was
    # faster on this CPU-only box, matching research's "100x CPU swings on a related
    # variant" warning direction (sdpa is the fused/optimized path).
    return _Tagger(attn_implementation="sdpa")


def tag_images(paths: list[Path]) -> dict[Path, dict[str, float]]:
    return _get_tagger().tag(paths)
