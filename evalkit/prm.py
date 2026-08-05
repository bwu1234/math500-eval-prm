"""Reward-model scoring: process-level (PRM) and outcome-level (ORM).

Both scorers answer "how good is this solution?" without seeing the gold
answer, which is what makes them usable for best-of-n reranking -- the one
place in this harness where the ground truth is deliberately withheld. They
differ in what they look at:

``PRMScorer`` (``Qwen2.5-Math-1.5B-Instruct-PRM-0.2``) is a token-classification
model trained (TRL stepwise-reward) on PRM800K. Reasoning steps are concatenated
with a ``\\n\\n`` separator and the reward for a step is ``P(LABEL_1)`` at that
step's trailing separator token. It scores the *reasoning*: no prompting and no
answer parsing is involved, and ``extract_steps`` drops the ``\\boxed`` line, so
the PRM cannot simply be reading off the final answer.

``ORMScorer`` (``Llama3.1-8B-ORM-Deepseek-Data``) is a *generative* reward model
rather than a regression head: the solution is put in a user turn, the assistant
turn is forced to ``"+"``, and the score is the probability the model assigns to
that ``"+"`` over ``"-"``. Same Math-Shepherd-style supervision as the PRM, so
the two scores are on comparable footing -- both are ``P(correct)``. It *does*
see the final answer, so a high ORM score is not evidence about reasoning
quality; treat it as a selection signal only.

Neither is the accuracy metric. MATH-500 ships gold answers and
``answers.final_answer_correct`` decides correctness exactly; these models exist
so that reranking can be measured against that ground truth, not to replace it.

Loading is lazy either way, and runs at full precision by default: nf4 error
lands directly on the score the calibration analysis is computed over, which is
not the same trade as quantizing a generator. ``load_in_4bit=True`` opts in
where the installed bitsandbytes has kernels for the local accelerator -- CUDA
and, since 0.50, MPS -- and that support is queried at runtime rather than
assumed.
"""

from __future__ import annotations

import ctypes
import glob
import math
import os
import sysconfig
from typing import Any

__all__ = ["PRMScorer", "ORMScorer", "NullScorer", "build_scorer",
           "SCORER_KINDS", "SCORER_MODELS", "PRM_MODEL", "ORM_MODEL", "PRM_SEP"]

PRM_MODEL = "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"
ORM_MODEL = "RLHFlow/Llama3.1-8B-ORM-Deepseek-Data"
PRM_SEP = "\n\n"  # must match the separator used during training

# The ORM's verdict token is the second-to-last token of the rendered chat
# (``... "+" <|eot_id|>``), so the distribution that predicts it is the logit
# row three from the end. Both indices are properties of the Llama-3.1 chat
# template rather than of the model, which is why ``ORMScorer`` checks the
# alignment at runtime instead of trusting them.
ORM_VERDICT_POS = -2
ORM_LOGIT_POS = -3

SCORER_KINDS = ("prm", "orm")
SCORER_MODELS = {"prm": PRM_MODEL, "orm": ORM_MODEL}


def _preload_cuda_libs() -> None:
    """Preload CUDA libs that bitsandbytes (built for CUDA 13) needs.

    ``libnvJitLink.so.13`` ships inside PyTorch's pip CUDA wheels but isn't on
    the default loader path, so bitsandbytes' native extension fails to dlopen
    it. Loading it RTLD_GLOBAL here makes its symbols available process-wide
    without requiring LD_LIBRARY_PATH. Best effort; a no-op elsewhere.
    """
    site = sysconfig.get_paths()["purelib"]
    for path in glob.glob(os.path.join(site, "nvidia/cu13/lib/libnvJitLink.so*")):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _plus_probability(plus_logit: float, minus_logit: float) -> float:
    """Two-way softmax over the ``+``/``-`` logits, done in the overflow-safe
    order. Kept in plain Python rather than ``torch.softmax`` so the arithmetic
    behind every reported ORM score is testable without a GPU stack."""
    d = plus_logit - minus_logit
    if d >= 0:
        return 1.0 / (1.0 + math.exp(-d))
    e = math.exp(d)
    return e / (1.0 + e)


def _check_verdict_alignment(token_ids: list[int], plus_id: int,
                             model_name: str) -> None:
    """Fail loudly if the forced ``"+"`` is not where the indices assume.

    ``ORM_LOGIT_POS`` is a positional index into a rendered chat, so it is
    only correct while the template ends the way this model's training data
    did. A tokenizer update that shifts the verdict token would otherwise
    leave the scorer reading the wrong logit row and quietly reporting noise.
    """
    found = token_ids[ORM_VERDICT_POS] if len(token_ids) >= abs(ORM_VERDICT_POS) else None
    if found != plus_id:
        raise RuntimeError(
            f"{model_name}: expected the '+' verdict token (id {plus_id}) at "
            f"position {ORM_VERDICT_POS} of the rendered chat, found {found}. "
            f"The chat template no longer matches the one this scorer targets, "
            f"so the scores it would produce are meaningless."
        )


def _accelerator() -> str:
    """The best device torch can actually reach here."""
    import torch  # noqa: PLC0415 -- deliberately lazy

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bnb_devices() -> frozenset[str]:
    """Devices this bitsandbytes build has 4-bit kernels for.

    Asked at runtime rather than hardcoded: the backend list has grown (MPS
    landed well after CUDA), so a hardcoded "CUDA only" would keep refusing to
    quantize on hardware that gained support in a later release.
    """
    try:
        import bitsandbytes as bnb  # noqa: PLC0415 -- deliberately lazy
    except ImportError:
        return frozenset()
    return frozenset(getattr(bnb, "supported_torch_devices", {"cuda"}))


def _should_quantize(load_in_4bit: bool | None, device: str,
                     supported: frozenset[str]) -> bool:
    """Whether to load in 4-bit, given what bitsandbytes can do here.

    The default is False, and deliberately so. These models produce the score
    the calibration analysis is computed over, and nf4 error perturbs that
    score directly -- unlike a generator, where quantization costs a little
    accuracy and nothing else. Full precision is the honest default; trading it
    for memory should be a decision someone made, not one made for them.

    ``load_in_4bit=None`` opts into "quantize if the installed bitsandbytes can
    do it on this device". CPU is excluded even when listed: there is no kernel
    worth having there, and dequantizing on every forward pass costs more than
    carrying the weights. An explicit request the backend cannot honour is an
    error rather than a silent fallback -- a run that quietly used four times
    the memory it was told to is worth hearing about.
    """
    if load_in_4bit is None:
        return device != "cpu" and device in supported
    if load_in_4bit and device not in supported:
        raise RuntimeError(
            f"load_in_4bit was requested but this bitsandbytes build has no "
            f"4-bit kernels for {device!r} (it supports "
            f"{sorted(supported) or 'nothing -- is bitsandbytes installed?'}). "
            f"Pass load_in_4bit=False to load in bfloat16 instead."
        )
    return bool(load_in_4bit)


def _resolve_placement(load_in_4bit: bool | None):
    """Decide (use_4bit, device, dtype) for the machine this is running on."""
    import torch  # noqa: PLC0415 -- deliberately lazy

    device = _accelerator()
    use_4bit = _should_quantize(load_in_4bit, device, _bnb_devices())
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return use_4bit, device, dtype


def _load_kwargs(use_4bit: bool, device: str, dtype):
    """Shared ``from_pretrained`` kwargs for both scorers."""
    # An explicit device beats device_map="auto" off CUDA: accelerate will
    # happily leave layers on the CPU, and a half-offloaded 8B model scores at
    # a crawl. On CUDA "auto" is still right, since it can shard across GPUs.
    placement = "auto" if device == "cuda" else device
    if use_4bit:
        import torch  # noqa: PLC0415
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        return {
            "device_map": placement,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    return {"dtype": dtype, "device_map": placement}


class NullScorer:
    """Used with ``--no-prm``: keeps the pipeline shape, scores nothing.

    Returns an empty score list rather than zeros -- a zero is a claim about
    reasoning quality, absence of a score is not, and the calibration analysis
    correctly excludes rows whose score is None.
    """

    enabled = False
    kind = "none"
    model_name = None

    def step_scores(self, problem: str, steps: list[str]) -> list[float]:
        return []

    def score(self, problem: str, steps: list[str],
              text: str | None = None) -> float | None:
        return None


class PRMScorer:
    """Wraps the token-classification pipeline, loading it on first use."""

    enabled = True
    kind = "prm"

    def __init__(self, model_name: str = PRM_MODEL,
                 load_in_4bit: bool | None = False, on_log=None):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self._log = on_log or (lambda msg: None)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        _preload_cuda_libs()
        from transformers import (  # noqa: PLC0415 -- deliberately lazy
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        use_4bit, device, dtype = _resolve_placement(self.load_in_4bit)
        precision = "4-bit" if use_4bit else str(dtype).replace("torch.", "")
        self._log(f"    [PRM] loading {self.model_name} ({precision} on {device}) ...")

        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **_load_kwargs(use_4bit, device, dtype)
        ).eval()
        self._pipe = pipeline("token-classification", model=model, tokenizer=tok)
        return self._pipe

    def step_scores(self, problem: str, steps: list[str]) -> list[float]:
        """Score each step on the cumulative prefix (problem + steps so far),
        reading the reward at the trailing separator token.

        Scoring the growing prefix rather than each step in isolation is what
        the model was trained for: a step is judged in the context of the
        reasoning that led to it.
        """
        if not steps:
            return []
        pipe = self._load()
        scores = []
        for idx in range(1, len(steps) + 1):
            text = PRM_SEP.join((problem, *steps[:idx])) + PRM_SEP
            tagged = pipe(text)
            if not tagged:
                scores.append(0.0)
                continue
            last = tagged[-1]
            # The pipeline returns the predicted label and its probability;
            # convert to P(correct) regardless of which label won.
            p_correct = last["score"] if last["entity"] == "LABEL_1" else 1.0 - last["score"]
            scores.append(float(p_correct))
        return scores

    def score(self, problem: str, steps: list[str],
              text: str | None = None) -> float | None:
        """Mean step reward. ``text`` is accepted for interface parity with the
        ORM and deliberately ignored: the PRM scores steps, not the response."""
        scores = self.step_scores(problem, steps)
        if not scores:
            return None
        return sum(scores) / len(scores)


class ORMScorer:
    """Outcome reward model: one ``P(correct)`` for the whole solution.

    This is a generative reward model, not a regression head. Problem and
    solution go in a single user turn, the assistant turn is forced to ``"+"``,
    and the score is that token's probability against ``"-"`` -- the format the
    model was trained on, and the one RLHFlow's own evaluation uses. Scoring a
    bare concatenation, or reading a classification head that isn't there,
    would both silently produce numbers unrelated to the training objective.

    The result is a genuine probability, which is what makes it comparable to
    the PRM's per-step ``P(correct)`` and safe for every downstream consumer:
    ``prm_weighted`` sums scores as vote weights, and a negative weight would
    make a sample argue against its own answer.

    Unlike the PRM this model is shown the response verbatim, boxed answer
    included -- an ORM that never sees the answer it is scoring is crippled --
    which is precisely why its score says nothing about reasoning quality.
    """

    enabled = True
    kind = "orm"

    def __init__(self, model_name: str = ORM_MODEL,
                 load_in_4bit: bool | None = False, on_log=None,
                 max_length: int = 8192):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self._log = on_log or (lambda msg: None)
        self._model: Any = None
        self._tok: Any = None
        self._verdict_ids: Any = None

    def _load(self):
        if self._model is not None:
            return self._model, self._tok

        _preload_cuda_libs()
        from transformers import (  # noqa: PLC0415 -- deliberately lazy
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        use_4bit, device, dtype = _resolve_placement(self.load_in_4bit)
        precision = "4-bit" if use_4bit else str(dtype).replace("torch.", "")
        self._log(f"    [ORM] loading {self.model_name} ({precision} on {device}) ...")

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **_load_kwargs(use_4bit, device, dtype)
        ).eval()
        # ``encode`` may prepend BOS, so take the trailing id of each.
        self._verdict_ids = [self._tok.encode("+")[-1], self._tok.encode("-")[-1]]
        return self._model, self._tok

    def step_scores(self, problem: str, steps: list[str]) -> list[float]:
        """Empty by construction: an outcome model has no per-step signal.

        Returning ``[]`` rather than a broadcast of the outcome score keeps the
        report's per-step shading honest -- it renders nothing instead of
        painting every step with a number that was never about that step.
        """
        return []

    def reward(self, problem: str, solution: str) -> float:
        """``P("+")`` against ``P("-")`` for one (problem, solution) pair."""
        import torch  # noqa: PLC0415 -- deliberately lazy

        model, tok = self._load()
        conversation = [
            {"role": "user", "content": f"{problem} {solution}"},
            {"role": "assistant", "content": "+"},
        ]
        ids = tok.apply_chat_template(
            conversation, tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_length,
        ).to(model.device)

        plus_id, minus_id = self._verdict_ids
        _check_verdict_alignment(ids[0].tolist(), plus_id, self.model_name)

        with torch.no_grad():
            row = model(ids).logits[0, ORM_LOGIT_POS]
            return _plus_probability(float(row[plus_id]), float(row[minus_id]))

    def score(self, problem: str, steps: list[str],
              text: str | None = None) -> float | None:
        """``P(correct)`` in (0, 1) for the solution as a whole.

        Prefers the raw response ``text``; falls back to the reconstructed
        steps, which lose the boxed answer and so are a weaker input.
        """
        solution = (text or "").strip() or PRM_SEP.join(steps).strip()
        if not solution:
            return None
        return self.reward(problem, solution)


def build_scorer(enabled: bool = True, kind: str = "prm", on_log=None, **kw):
    if not enabled:
        return NullScorer()  # scoring is off; loading options are moot
    if kind == "orm":
        return ORMScorer(on_log=on_log, **kw)
    if kind == "prm":
        return PRMScorer(on_log=on_log, **kw)
    raise ValueError(f"unknown scorer kind: {kind!r} (expected one of {SCORER_KINDS})")
