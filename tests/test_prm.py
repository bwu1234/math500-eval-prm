"""Reward-scorer selection and the outcome-model path through the pipeline.

Nothing here loads a model. What is worth testing is the plumbing: that the
runner asks an outcome scorer the right question, that its answer reaches the
same column the PRM writes to, and that the run records *which* model produced
that column -- the one thing a reader of the results file cannot infer.
"""

import json

import pytest

from evalkit.prm import (
    ORM_MODEL,
    PRM_MODEL,
    NullScorer,
    ORMScorer,
    PRMScorer,
    _check_verdict_alignment,
    _load_kwargs,
    _plus_probability,
    _should_quantize,
    build_scorer,
)
from evalkit.runner import EvalConfig, run_eval

# What bitsandbytes 0.50 reports as supported.
CUDA_AND_MPS = frozenset({"cuda", "mps", "cpu", "xpu"})


def make_config(tmp_path, **kw):
    return EvalConfig(**{
        "num_questions": 12, "backend": "mock", "use_prm": True,
        "verbose": False, "out_dir": str(tmp_path), **kw
    })


# ── build_scorer ──────────────────────────────────────────────────────
def test_build_scorer_defaults_to_the_prm():
    scorer = build_scorer()
    assert isinstance(scorer, PRMScorer)
    assert scorer.kind == "prm"
    assert scorer.model_name == PRM_MODEL


def test_build_scorer_returns_the_orm_on_request():
    scorer = build_scorer(kind="orm")
    assert isinstance(scorer, ORMScorer)
    assert scorer.kind == "orm"
    assert scorer.model_name == ORM_MODEL


def test_build_scorer_disabled_beats_kind():
    """``--no-prm`` means no scoring, whichever model was named."""
    assert isinstance(build_scorer(enabled=False, kind="orm"), NullScorer)


def test_build_scorer_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown scorer kind"):
        build_scorer(kind="reward")


# ── The verdict logits -> score mapping ───────────────────────────────
def test_plus_probability_is_a_bounded_two_way_softmax():
    """The ORM's score is ``P("+")`` against ``P("-")``.

    ``prm_weighted`` sums scores as vote weights, so a negative or unbounded
    value would let a sample argue against its own answer; a probability is
    also what makes the number comparable to the PRM's per-step ``P(correct)``.
    """
    assert _plus_probability(0.0, 0.0) == pytest.approx(0.5)
    assert _plus_probability(5.0, -5.0) > 0.99
    assert _plus_probability(-5.0, 5.0) < 0.01
    # Ordering must follow the margin, and extremes must not overflow.
    margins = [(-900.0, 900.0), (-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (900.0, -900.0)]
    scores = [_plus_probability(p, m) for p, m in margins]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores)


# ── Quantization policy ───────────────────────────────────────────────
# The 4-bit decision is separated from the torch import on purpose: CI has
# neither torch nor bitsandbytes, and this is the part with actual branching.
def test_scorers_default_to_full_precision():
    """nf4 error lands directly on the score the calibration is computed over,
    so paying memory for an honest number is the default."""
    assert PRMScorer().load_in_4bit is False
    assert ORMScorer().load_in_4bit is False
    assert EvalConfig().load_in_4bit is False
    assert _should_quantize(False, "cuda", CUDA_AND_MPS) is False


def test_auto_quantizes_on_any_supported_accelerator():
    """bitsandbytes gained MPS after CUDA, so the opt-in policy asks the
    installed build what it supports rather than assuming a device list."""
    assert _should_quantize(None, "cuda", CUDA_AND_MPS) is True
    assert _should_quantize(None, "mps", CUDA_AND_MPS) is True


def test_never_quantizes_on_cpu_even_when_listed():
    """Dequantizing every forward pass costs more than carrying the weights."""
    assert _should_quantize(None, "cpu", CUDA_AND_MPS) is False


def test_falls_back_to_bfloat16_where_there_are_no_kernels():
    assert _should_quantize(None, "mps", frozenset({"cuda"})) is False
    assert _should_quantize(None, "cuda", frozenset()) is False


def test_an_unsupportable_explicit_request_is_an_error():
    """Silently using four times the requested memory is worth hearing about."""
    with pytest.raises(RuntimeError, match="no.*4-bit kernels"):
        _should_quantize(True, "mps", frozenset({"cuda"}))
    with pytest.raises(RuntimeError, match="is bitsandbytes installed"):
        _should_quantize(True, "cuda", frozenset())


def test_quantization_reaches_the_scorer_from_the_config(
        tmp_path, monkeypatch, problems, backend):
    """The flag is useless if it stops at ``EvalConfig``."""
    seen = {}

    def spy(enabled=True, kind="prm", on_log=None, **kw):
        seen.update(kw)
        return NullScorer()

    monkeypatch.setattr("evalkit.runner.build_scorer", spy)
    run_eval(make_config(tmp_path, load_in_4bit=True), problems=problems,
             backend=backend, scorer=None)
    assert seen == {"load_in_4bit": True}


def test_quantized_and_full_precision_runs_do_not_share_a_checkpoint(tmp_path):
    """nf4 moves the scores, so resuming across the two would splice
    incomparable numbers into one column."""
    full = make_config(tmp_path, load_in_4bit=False)
    quantized = make_config(tmp_path, load_in_4bit=True)
    assert full.fingerprint() != quantized.fingerprint()


def test_placement_is_explicit_off_cuda():
    """``device_map="auto"`` can strand layers on the CPU, and a half-offloaded
    8B model scores at a crawl; on CUDA it is still right, since it shards."""
    assert _load_kwargs(False, "mps", "bf16")["device_map"] == "mps"
    assert _load_kwargs(False, "cuda", "bf16")["device_map"] == "auto"


def test_verdict_alignment_accepts_the_expected_layout():
    """The rendered chat ends ``... "+" <|eot_id|>``."""
    _check_verdict_alignment([1, 2, 3, 43, 128009], plus_id=43, model_name="m")


def test_verdict_alignment_rejects_a_shifted_template():
    """``ORM_LOGIT_POS`` indexes into the rendered chat by position, so a
    template change has to be an error rather than a silently wrong score."""
    with pytest.raises(RuntimeError, match="no longer matches"):
        _check_verdict_alignment([1, 2, 3, 128009, 43], plus_id=43, model_name="m")
    with pytest.raises(RuntimeError, match="no longer matches"):
        _check_verdict_alignment([43], plus_id=43, model_name="m")


def test_orm_step_scores_are_empty_by_construction():
    """An outcome model has no per-step signal, and must not invent one."""
    assert ORMScorer().step_scores("problem", ["a", "b", "c"]) == []


def test_orm_scores_the_response_not_the_steps(monkeypatch):
    """The raw text is what gets scored: ``extract_steps`` drops the boxed
    answer, and an ORM blind to the answer is scoring the wrong thing."""
    seen = {}
    scorer = ORMScorer()

    def fake_reward(problem, solution):
        seen["problem"], seen["solution"] = problem, solution
        return 0.75

    monkeypatch.setattr(scorer, "reward", fake_reward)
    score = scorer.score("What is 2+2?", ["some step"], text="reasoning \\boxed{4}")
    assert seen["problem"] == "What is 2+2?"
    assert seen["solution"] == "reasoning \\boxed{4}"
    assert score == pytest.approx(0.75)


def test_orm_falls_back_to_steps_without_text(monkeypatch):
    scorer = ORMScorer()
    monkeypatch.setattr(scorer, "reward", lambda p, s: 0.5)
    assert scorer.score("p", ["a", "b"]) is not None
    assert scorer.score("p", []) is None
    assert scorer.score("p", [], text="   ") is None


# ── Through the pipeline ──────────────────────────────────────────────
def test_orm_scores_land_in_the_same_column(tmp_path, problems, backend,
                                            outcome_scorer):
    cfg = make_config(tmp_path, scorer_kind="orm")
    summary = run_eval(cfg, problems=problems, backend=backend,
                       scorer=outcome_scorer)

    assert all(r["prm_score"] is not None for r in summary["results"])
    # No per-step scores, so the report shades nothing rather than painting
    # every step with a number that was never about that step.
    assert all(r["step_prm_scores"] == [] for r in summary["results"])
    assert summary["avg_prm_score"] is not None
    assert summary["prm_calibration"]["n"] == len(problems)


def test_orm_run_is_shown_the_raw_response(tmp_path, problems, backend,
                                           outcome_scorer):
    run_eval(make_config(tmp_path, scorer_kind="orm"), problems=problems,
             backend=backend, scorer=outcome_scorer)
    assert outcome_scorer.seen_text
    assert all("\\boxed" in (t or "") for t in outcome_scorer.seen_text)


def test_summary_records_which_model_scored(tmp_path, problems, backend,
                                            outcome_scorer, scorer):
    orm = run_eval(make_config(tmp_path / "a", scorer_kind="orm"),
                   problems=problems, backend=backend, scorer=outcome_scorer)
    assert orm["scorer_kind"] == "orm"
    assert orm["scorer_model"] == "fake-orm"

    prm = run_eval(make_config(tmp_path / "b"), problems=problems,
                   backend=backend, scorer=scorer)
    assert prm["scorer_kind"] == "prm"
    assert prm["scorer_model"] == "fake-prm"


def test_disabled_scoring_records_no_scorer(tmp_path, problems, backend):
    summary = run_eval(make_config(tmp_path, use_prm=False, scorer_kind="orm"),
                       problems=problems, backend=backend, scorer=NullScorer())
    assert summary["scorer_kind"] == "none"
    assert summary["scorer_model"] is None
    assert summary["avg_prm_score"] is None


def test_prm_and_orm_checkpoints_are_not_interchangeable(tmp_path):
    """Both write ``prm_score``; resuming one from the other would splice two
    different measurements into a single column."""
    prm = make_config(tmp_path, scorer_kind="prm")
    orm = make_config(tmp_path, scorer_kind="orm")
    assert prm.fingerprint() != orm.fingerprint()


def test_results_file_names_the_scorer(tmp_path, problems, backend,
                                       outcome_scorer):
    cfg = make_config(tmp_path, scorer_kind="orm")
    run_eval(cfg, problems=problems, backend=backend, scorer=outcome_scorer)
    with open(tmp_path / "eval_results.json") as f:
        data = json.load(f)
    assert data["scorer_kind"] == "orm"
    assert data["scorer_model"] == "fake-orm"
