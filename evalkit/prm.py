"""Process Reward Model scoring.

``Qwen2.5-Math-1.5B-Instruct-PRM-0.2`` is a token-classification model trained
(TRL stepwise-reward) on PRM800K. Reasoning steps are concatenated with a
``\\n\\n`` separator and the reward for a step is ``P(LABEL_1)`` at that step's
trailing separator token.

This is the substantive difference between this harness and the usual
"ask GPT-4 to grade the reasoning" approach: the score comes from a model
trained on human step-level correctness labels, so it is a measurement rather
than another language model's opinion. No prompting and no answer parsing is
involved, which also means the PRM cannot simply be reading off the final
answer -- it is scoring the reasoning itself.

Loading is lazy and quantized to 4-bit so it fits alongside a 4 GB GPU budget.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sysconfig

__all__ = ["PRMScorer", "NullScorer", "build_scorer", "PRM_MODEL", "PRM_SEP"]

PRM_MODEL = "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"
PRM_SEP = "\n\n"  # must match the separator used during training


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


class NullScorer:
    """Used with ``--no-prm``: keeps the pipeline shape, scores nothing.

    Returns an empty score list rather than zeros -- a zero is a claim about
    reasoning quality, absence of a score is not, and the calibration analysis
    correctly excludes rows whose score is None.
    """

    enabled = False

    def step_scores(self, problem: str, steps: list[str]) -> list[float]:
        return []

    def score(self, problem: str, steps: list[str]) -> float | None:
        return None


class PRMScorer:
    """Wraps the token-classification pipeline, loading it on first use."""

    enabled = True

    def __init__(self, model_name: str = PRM_MODEL, load_in_4bit: bool = True,
                 on_log=None):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self._log = on_log or (lambda msg: None)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        _preload_cuda_libs()
        import torch  # noqa: PLC0415 -- deliberately lazy
        from transformers import (  # noqa: PLC0415
            AutoModelForTokenClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        quant = "4-bit" if self.load_in_4bit else "full precision"
        self._log(f"    [PRM] loading {self.model_name} ({quant}) ...")

        kwargs = {"device_map": "auto"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **kwargs
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

    def score(self, problem: str, steps: list[str]) -> float | None:
        scores = self.step_scores(problem, steps)
        if not scores:
            return None
        return sum(scores) / len(scores)


def build_scorer(enabled: bool = True, on_log=None, **kw):
    return PRMScorer(on_log=on_log, **kw) if enabled else NullScorer()
