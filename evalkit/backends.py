"""Model backends.

Every backend exposes the same tiny surface -- ``generate(prompt, ...) ->
Generation`` -- so the runner never learns which vendor it is talking to, and
adding a provider means adding a class here and nothing else.

Heavy vendor SDKs are imported inside ``__init__`` rather than at module
import, so that ``import evalkit.backends`` costs nothing and the grading and
analysis modules stay usable without any provider installed.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

__all__ = ["Generation", "Backend", "GeminiBackend", "OllamaBackend",
           "MockBackend", "build_backend", "BACKEND_NAMES", "DEFAULT_MODELS",
           "THINK_MODES"]

DEFAULT_MODELS = {
    "gemini": "gemma-4-31b-it",
    "qwen": "qwen3.6:35b-mlx",
    "mock": "mock-v1",
}
BACKEND_NAMES = tuple(DEFAULT_MODELS)


@dataclass
class Generation:
    """One model response plus its token accounting."""
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    warning: str | None = None

    def __post_init__(self):
        # Providers report a missing count as an explicit null rather than
        # omitting the field: Gemini returns candidates_token_count=None when a
        # response is truncated at MAX_TOKENS having emitted no text. Coercing
        # here keeps every downstream consumer working with plain ints, instead
        # of each one having to guard separately.
        self.prompt_tokens = int(self.prompt_tokens or 0)
        self.output_tokens = int(self.output_tokens or 0)
        self.total_tokens = int(self.total_tokens or 0)


class Backend:
    """Base class holding the shared retry loop."""

    name = "base"

    def __init__(self, model: str, retries: int = 8, on_log=None):
        self.model = model
        self.retries = retries
        self._log = on_log or (lambda msg: None)

    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.0, context: dict | None = None) -> Generation:
        raise NotImplementedError

    def _with_retries(self, fn, transient: tuple[type[BaseException], ...]):
        """Retry ``fn`` with exponential backoff and jitter.

        Jitter matters: without it, parallel runs that hit a rate limit retry
        in lockstep and hit it again together.
        """
        last = None
        for attempt in range(self.retries):
            try:
                return fn()
            except transient as e:
                last = e
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                self._log(
                    f"  [{self.name}] {type(e).__name__}: {e}; "
                    f"retry {attempt + 1}/{self.retries} in {wait:.1f}s..."
                )
                time.sleep(wait)
        raise RuntimeError(
            f"{self.name} backend failed after {self.retries} retries"
        ) from last


class GeminiBackend(Backend):
    """Google Gemini / Gemma via the google-genai SDK."""

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, **kw):
        super().__init__(model or DEFAULT_MODELS["gemini"], **kw)
        from google import genai  # noqa: PLC0415 -- deliberately lazy
        from google.genai import errors, types

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Export it, or use `-m mock` to "
                "exercise the harness without an API key."
            )
        self._types = types
        self._transient = (errors.APIError,)
        self._client = genai.Client(api_key=key)

    def generate(self, prompt, max_tokens=2048, temperature=0.0, context=None):
        def call():
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            text = resp.text or ""
            usage = resp.usage_metadata
            warning = None
            if not text and resp.candidates:
                c = resp.candidates[0]
                warning = f"empty response | finish_reason={c.finish_reason}"
            return Generation(
                text=text,
                prompt_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                total_tokens=usage.total_token_count if usage else 0,
                warning=warning,
            )

        return self._with_retries(call, self._transient)


THINK_MODES = ("merge", "off")


class OllamaBackend(Backend):
    """Any model served by a local Ollama instance.

    Thinking-capable models (e.g. qwen3.5) return their chain-of-thought in a
    separate ``thinking`` field, distinct from ``response`` which holds only
    the terse final line. ``extract_steps`` (in answers.py) needs the actual
    reasoning to produce meaningful PRM step scores, so ``think`` controls how
    that field is handled:

    - "merge" (default): fold ``thinking`` into the graded text, ahead of the
      final response, so step scoring sees the real reasoning chain.
    - "off": ask the model not to think at all (``think: false``), so
      ``response`` already contains everything.
    """

    name = "qwen"

    def __init__(self, model: str | None = None, host: str | None = None,
                 timeout: int = 600, think: str = "merge", **kw):
        super().__init__(model or DEFAULT_MODELS["qwen"], **kw)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        if think not in THINK_MODES:
            raise ValueError(f"think must be one of {THINK_MODES}, got {think!r}")
        self.think = think

    def generate(self, prompt, max_tokens=2048, temperature=0.0, context=None):
        url = f"{self.host}/api/generate"
        request = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if self.think == "off":
            request["think"] = False
        payload = json.dumps(request).encode("utf-8")

        def call():
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            response = body.get("response", "")
            thinking = body.get("thinking") or ""
            text = f"{thinking}\n\n{response}" if thinking else response
            prompt_tokens = body.get("prompt_eval_count", 0)
            output_tokens = body.get("eval_count", 0)
            return Generation(
                text=text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                warning=None if text else f"empty response | done_reason={body.get('done_reason')}",
            )

        return self._with_retries(
            call, (urllib.error.URLError, urllib.error.HTTPError, TimeoutError)
        )


class MockBackend(Backend):
    """A deterministic test double -- **not** a model.

    It exists so the whole pipeline (checkpointing, selection, aggregation,
    reporting) can be exercised in CI and by anyone who clones the repo
    without a GPU or an API key. It reads the gold answer from ``context`` and
    decides whether to emit it based on a hash of the problem, giving a stable
    target accuracy without any randomness across runs.

    Results produced this way are tagged ``synthetic`` so they can never be
    mistaken for a real measurement.
    """

    name = "mock"
    synthetic = True

    def __init__(self, model: str | None = None, accuracy: float = 0.6,
                 seed: int = 0, **kw):
        super().__init__(model or DEFAULT_MODELS["mock"], **kw)
        self.accuracy = accuracy
        self.seed = seed

    def generate(self, prompt, max_tokens=2048, temperature=0.0, context=None):
        context = context or {}
        gold = str(context.get("answer", "42"))
        # At temperature 0 repeated calls must agree; above it they must not,
        # or k>1 self-consistency would be exercised with k identical samples.
        draw = context.get("sample_index", 0) if temperature > 0 else 0
        # Stable per-problem pseudo-randomness: hash() is salted per process,
        # so use a fixed digest instead.
        import hashlib  # noqa: PLC0415
        digest = hashlib.sha256(
            f"{self.seed}|{temperature}|{draw}|{prompt}".encode()
        ).hexdigest()
        roll = int(digest[:8], 16) / 0xFFFFFFFF
        answer = gold if roll < self.accuracy else f"{roll:.3f}"

        text = (
            f"Let me work through this step by step.\n"
            f"First, I identify what the problem is asking for.\n"
            f"\\[\nx = {answer}\n\\]\n"
            f"Checking the arithmetic, this is consistent.\n"
            f"Therefore the final answer is \\boxed{{{answer}}}."
        )
        n_out = len(text.split())
        return Generation(
            text=text,
            prompt_tokens=len(prompt.split()),
            output_tokens=n_out,
            total_tokens=len(prompt.split()) + n_out,
        )


_BACKEND_CLASSES = {
    "gemini": GeminiBackend,
    "qwen": OllamaBackend,
    "mock": MockBackend,
}


def build_backend(name: str, model: str | None = None, on_log=None, **kw) -> Backend:
    if name not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend {name!r} (expected one of {list(_BACKEND_CLASSES)})"
        )
    return _BACKEND_CLASSES[name](model=model, on_log=on_log, **kw)
