"""Backend selection for every LLM call in locket -- Claude API (`anthropic`, paid, current
default behavior) or a fully local Ollama server (`ollama`, keyless). One function,
`get_chat_model(role, settings)`, replaces every module-level `ChatAnthropic(...)`
construction that used to be hardcoded directly in extraction/graph.py, resolution.py,
profile.py, and mcp_server.py -- each of those now asks this module for its chat model
instead of importing `ChatAnthropic` itself, then calls `.with_structured_output(...)` (or
not) on the result exactly as before. Vision (locket.vision.vision_llm) is untouched: it
always needs a local multimodal model regardless of which text backend is selected, per its
own hard privacy rule.

Verified live 2026-07-31 against the actually-installed `langchain-ollama==1.1.0` (latest on
PyPI at build time; requires `langchain-core<2.0.0,>=1.2.21` and `ollama<1.0.0,>=0.6.1`,
both satisfied by this repo's existing pins) -- training-data API references for this
package are likely stale, so this was checked against `inspect.signature`/`inspect.getsource`
on the installed class, not assumed:

  `ChatOllama.with_structured_output(schema, *, method="json_schema"|"function_calling"|
  "json_mode", include_raw=False)` has an IDENTICAL contract to `ChatAnthropic`'s: same
  `method=` literal values, and `include_raw=True` returns the same `{"raw", "parsed",
  "parsing_error"}` dict shape (parsing failures land in `"parsing_error"`, not an
  exception). graph.py's `parsed`/`parsing_error` branch in `extract_node` works completely
  unchanged against either backend -- no adapter needed. `method` already defaults to
  `"json_schema"` (changed from `"function_calling"` in langchain-ollama 0.3.0), matching
  every call site in this repo, which passes it explicitly anyway.

Local model choice -- measured live on this machine (CPU-only Ollama, `qwen3-vl:8b` loaded
with only 2.27/6.2GB in VRAM, the rest on CPU):

  `qwen3-vl:8b` (this repo's existing vision model; Ollama reports it with a "thinking"
  capability) could NOT be made to skip its <think> reasoning trace for TEXT calls --
  neither `think=False` on the raw `ollama.chat()`/HTTP API nor `reasoning=False` on
  `ChatOllama` suppressed it (confirmed: a plain "say hello in one word" prompt still
  returned a ~280-token <think> block, taking 32s end-to-end at roughly 7 tok/s on this
  box). Adding `format=<json schema>` on top of that made a single 2-message extraction
  window not return within 5 minutes. qwen3-vl:8b remains vision_llm.py's model for the
  image-description tail (already measured to tolerate ~140s/image there) but is
  UNSUITABLE as the default TEXT-extraction/resolution/profile model -- the repeated
  per-window structured-output calls this backend needs to make would make a full corpus
  pass impractically slow.

  `qwen2.5:3b-instruct` (already present locally on this machine, no download, no
  "thinking" capability) returned a valid, schema-conformant `ExtractionResult` in 6-13s
  per window (3 real demo_corpus/whatsapp/team.txt windows: 9.8s/2.7s/1.4s), but its facts
  skewed toward noise -- e.g. bare `person | Jeffrey Williams` entries with no content
  beyond a name, and a duplicate `person | Cory Davis` -- alongside legitimate ones.

  `gemma3:12b` (pulled for this comparison, ~8.1GB) took roughly 10x longer per window
  (131.8s/27.3s/10.0s on the same 3 windows) but produced denser, more accurate facts on
  identical input: a `habit` ("ate a bagel alone at 9pm on his birthday last year"), a
  `preference` ("enjoys carbonara from Bertucci's Trattoria"), and specific event details
  ("dinner at Bertucci's ... March 3rd at 7pm") that qwen2.5:3b-instruct did not surface at
  all, with zero bare-name noise facts. Full side-by-side transcript in
  evals/BASELINE.md's "local backend" section.

  `gemma3:12b` is the default local TEXT model -- for a personal-context engine whose
  extracted facts feed a citable "profile of you" and an `answer_question` MCP tool,
  completeness/accuracy outweighs the ~10x latency cost, and 130s for the largest window
  measured is still well inside vision_llm.py's already-accepted ~140s/image local-model
  tolerance. Set `LOCKET_LOCAL_MODEL=qwen2.5:3b-instruct` for a much faster, lower-quality
  run (e.g. iterating on adapters/chunking, not a real corpus pass), or to skip gemma3:12b's
  ~8GB download entirely.

Env vars:
  LOCKET_LLM_BACKEND: "anthropic" | "ollama" (case-insensitive). Defaults to "ollama" when
    `settings.anthropic_api_key` is unset, else "anthropic" -- so `locket pipeline run`
    works keylessly out of the box, and setting a real key opts back into the
    higher-quality, paid Claude backend with no flag needed.
  LOCKET_LOCAL_MODEL: local Ollama model name used for every TEXT role below when the
    `ollama` backend is active. Defaults to "gemma3:12b" (requires `ollama pull gemma3:12b`,
    ~8GB). Independent of `locket.config.Settings.ollama_model` / `LOCKET_OLLAMA_MODEL` (an
    existing, currently call-site-less field) and of `locket.vision.vision_llm.DEFAULT_MODEL`
    ("qwen3-vl:8b") -- vision keeps its own multimodal model regardless of this backend
    choice.
  OLLAMA_HOST: read by `langchain-ollama`/the `ollama` client itself (via `ChatOllama`'s
    own `base_url` resolution) -- deliberately NOT read or overridden here, so pointing it
    at box-b's tailnet Ollama (100.70.154.99:11434) would redirect every local-backend call
    there. Untested by this task.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

from locket.config import Settings

DEFAULT_LOCAL_TEXT_MODEL = "gemma3:12b"

# Claude model id per call site -- same ids the pre-existing hardcoded ChatAnthropic(...)
# constructions used (HAIKU_MODEL_ID/SONNET_MODEL_ID constants in graph.py, resolution.py,
# profile.py, mcp_server.py), just centralized here now that construction is centralized.
_ANTHROPIC_MODEL_IDS: dict[str, str] = {
    "extraction_default": "claude-haiku-4-5",
    "extraction_escalation": "claude-sonnet-5",
    "resolution": "claude-haiku-4-5",
    "profile_render": "claude-haiku-4-5",
    "mcp_decompose": "claude-haiku-4-5",
    "mcp_synthesize": "claude-haiku-4-5",
}

_VALID_BACKENDS = ("anthropic", "ollama")


def resolve_backend(settings: Settings) -> str:
    """"anthropic" | "ollama". Explicit LOCKET_LLM_BACKEND always wins; otherwise "ollama"
    unless a real Anthropic key is present -- see module docstring."""
    raw = os.environ.get("LOCKET_LLM_BACKEND")
    if raw:
        backend = raw.strip().lower()
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"LOCKET_LLM_BACKEND={raw!r} is not one of {_VALID_BACKENDS}"
            )
        return backend
    return "anthropic" if settings.anthropic_api_key else "ollama"


def local_model_name() -> str:
    return os.environ.get("LOCKET_LOCAL_MODEL", DEFAULT_LOCAL_TEXT_MODEL)


def get_chat_model(role: str, settings: Settings) -> Any:
    """The bare LangChain chat model for `role` (one of `_ANTHROPIC_MODEL_IDS`'s keys),
    backend-selected per `resolve_backend(settings)`. Callers attach
    `.with_structured_output(...)` (or invoke it directly, e.g. mcp_server's free-text
    synthesize step) exactly as they did against the old hardcoded `ChatAnthropic(...)` --
    this function only replaces *which* model gets constructed, not how it's used."""
    backend = resolve_backend(settings)
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if role not in _ANTHROPIC_MODEL_IDS:
            raise ValueError(f"unknown role {role!r} -- not in {sorted(_ANTHROPIC_MODEL_IDS)}")
        # temperature=0 for parity with the ollama branch below -- an unpinned Anthropic
        # temperature defaults to the API's own non-zero value, undermining the point of
        # the statement-hash dedup and the extracted-windows watermark: identical input
        # should extract identically on a re-run, not just usually.
        return ChatAnthropic(model=_ANTHROPIC_MODEL_IDS[role], temperature=0)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=local_model_name(), temperature=0)


@cache
def _cached_settings() -> Settings:
    """`Settings.load()` re-reads os.environ every call; callers that just want "the live
    default model for this role" (module-level `@cache`d factories in graph.py/
    resolution.py/profile.py/mcp_server.py) go through this instead so repeated calls in the
    same process don't re-parse env each time. Tests that need env changes to take effect
    call `Settings.load()` themselves and pass it to `get_chat_model` directly, bypassing
    this cache entirely."""
    return Settings.load()


def get_default_chat_model(role: str) -> Any:
    """Convenience wrapper: `get_chat_model(role, _cached_settings())`. This is what
    production call sites use; test seams inject `model=` and never reach this function."""
    return get_chat_model(role, _cached_settings())


__all__ = [
    "DEFAULT_LOCAL_TEXT_MODEL",
    "get_chat_model",
    "get_default_chat_model",
    "local_model_name",
    "resolve_backend",
]
