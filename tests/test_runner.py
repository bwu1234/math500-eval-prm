"""End-to-end pipeline tests using the mock backend and a fake PRM.

These cover the parts most likely to lose someone's work: checkpoint resume,
config-change invalidation, and the single-question merge path.
"""

import json
import os

import pytest

from evalkit.backends import MockBackend
from evalkit.runner import (
    EvalConfig,
    aggregate,
    load_checkpoint,
    run_eval,
    solve_question,
)
from evalkit.runner import RunLogger


def make_config(tmp_path, **kw):
    return EvalConfig(**{
        "num_questions": 12, "backend": "mock", "use_prm": True,
        "verbose": False, "out_dir": str(tmp_path), **kw
    })


# ── Happy path ────────────────────────────────────────────────────────
def test_run_eval_end_to_end(tmp_path, problems, backend, scorer, capsys):
    cfg = make_config(tmp_path)
    summary = run_eval(cfg, problems=problems, backend=backend, scorer=scorer)

    assert summary["num_questions"] == 12
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["correct"] == sum(1 for r in summary["results"] if r["answer_accuracy"])
    assert summary["synthetic"] is True
    assert os.path.exists(tmp_path / "eval_results.json")
    assert os.path.exists(tmp_path / "eval_debug.log")


def test_results_file_is_valid_json(tmp_path, problems, backend, scorer):
    run_eval(make_config(tmp_path), problems=problems, backend=backend, scorer=scorer)
    with open(tmp_path / "eval_results.json") as f:
        data = json.load(f)
    assert data["results"] and "prm_calibration" in data


def test_prm_scores_recorded_per_step(tmp_path, problems, backend, scorer):
    summary = run_eval(make_config(tmp_path), problems=problems,
                       backend=backend, scorer=scorer)
    row = summary["results"][0]
    assert len(row["step_prm_scores"]) == row["num_steps"]
    assert row["prm_score"] is not None


def test_no_prm_leaves_scores_none(tmp_path, problems, backend):
    from evalkit.prm import NullScorer
    summary = run_eval(make_config(tmp_path, use_prm=False), problems=problems,
                       backend=backend, scorer=NullScorer())
    assert summary["avg_prm_score"] is None
    assert all(r["prm_score"] is None for r in summary["results"])
    # A missing score must not be recorded as a zero, which would read as
    # "this reasoning is terrible" rather than "not measured".
    assert summary["prm_calibration"]["n"] == 0


# ── Checkpointing ─────────────────────────────────────────────────────
class CountingBackend(MockBackend):
    """Counts generate() calls so tests can prove work was actually skipped."""

    def __init__(self, fail_at=None, **kw):
        super().__init__(**kw)
        self.fail_at = fail_at
        self.calls = 0

    def generate(self, prompt, max_tokens=2048, temperature=0.0, context=None):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("simulated crash")
        return super().generate(prompt, max_tokens, temperature, context)


def FlakyBackend(fail_at, **kw):
    return CountingBackend(fail_at=fail_at, **kw)


def test_checkpoint_resumes_after_crash(tmp_path, problems, scorer):
    cfg = make_config(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_eval(cfg, problems=problems, backend=FlakyBackend(fail_at=5), scorer=scorer)

    saved = load_checkpoint(str(tmp_path / ".eval_checkpoint.jsonl"), cfg.fingerprint())
    assert len(saved) == 4, "completed questions should survive the crash"

    good = CountingBackend(accuracy=0.6)
    summary = run_eval(cfg, problems=problems, backend=good, scorer=scorer)
    assert summary["num_questions"] == 12
    # The 4 checkpointed questions must not be regenerated: that is the whole
    # point of the checkpoint on a multi-hour run.
    assert good.calls == 8


def test_checkpoint_is_removed_after_a_clean_run(tmp_path, problems, backend, scorer):
    run_eval(make_config(tmp_path), problems=problems, backend=backend, scorer=scorer)
    assert not os.path.exists(tmp_path / ".eval_checkpoint.jsonl")


def test_checkpoint_invalidated_by_config_change(tmp_path, problems, scorer):
    cfg = make_config(tmp_path)
    with pytest.raises(RuntimeError):
        run_eval(cfg, problems=problems, backend=FlakyBackend(fail_at=5), scorer=scorer)

    path = str(tmp_path / ".eval_checkpoint.jsonl")
    assert load_checkpoint(path, cfg.fingerprint())
    # Changing the sampling config makes the cached generations non-comparable.
    other = make_config(tmp_path, temperature=0.9)
    assert load_checkpoint(path, other.fingerprint()) == {}


def test_no_resume_flag_ignores_checkpoint(tmp_path, problems, scorer):
    cfg = make_config(tmp_path)
    with pytest.raises(RuntimeError):
        run_eval(cfg, problems=problems, backend=FlakyBackend(fail_at=5), scorer=scorer)
    fresh = CountingBackend(accuracy=0.6)
    run_eval(make_config(tmp_path, resume=False), problems=problems,
             backend=fresh, scorer=scorer)
    assert fresh.calls == 12, "--no-resume must regenerate every question"


def test_truncated_checkpoint_line_is_tolerated(tmp_path):
    """A hard kill can leave a partial final line; earlier rows must survive."""
    path = tmp_path / "cp.jsonl"
    path.write_text(
        json.dumps({"fingerprint": "abc"}) + "\n"
        + json.dumps({"index": 0, "answer_accuracy": True}) + "\n"
        + '{"index": 1, "answer_ac'
    )
    saved = load_checkpoint(str(path), "abc")
    assert list(saved) == [0]


def test_missing_checkpoint_is_not_an_error(tmp_path):
    assert load_checkpoint(str(tmp_path / "nope.jsonl"), "abc") == {}


# ── Archiving ─────────────────────────────────────────────────────────
def test_archive_name_derives_from_the_real_filename():
    from evalkit.runner import _archive_destination
    assert _archive_destination("logs", "report.html", "S").endswith("logs/report_S.html")
    # Must keep the backend suffix rather than flattening to eval_results_S.json
    assert _archive_destination("logs", "eval_results_qwen.json", "S") \
        .endswith("logs/eval_results_qwen_S.json")


def test_archive_never_clobbers_an_existing_archive(tmp_path):
    """Two runs inside the same second must not overwrite each other's only
    surviving copy."""
    from evalkit.runner import _archive_destination
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "report_S.html").write_text("first")
    second = _archive_destination(str(logs), "report.html", "S")
    assert not os.path.exists(second)
    assert second.endswith("report_S_2.html")


def test_archive_moves_every_artifact_under_one_stamp(tmp_path):
    from evalkit.runner import _archive_previous
    for name in ("eval_debug.log", "eval_results.json", "report.html"):
        (tmp_path / name).write_text(name)
    cfg = make_config(tmp_path)

    archived = _archive_previous(cfg, *(str(tmp_path / n) for n in
                                        ("eval_debug.log", "eval_results.json", "report.html")))
    assert len(archived) == 3
    stamps = {os.path.splitext(os.path.basename(p))[0].rsplit("_", 2)[-2:][0] for p in archived}
    assert len(stamps) == 1, "one run's artifacts must share a timestamp"
    assert not (tmp_path / "report.html").exists()


def test_archive_is_a_noop_when_nothing_exists(tmp_path):
    from evalkit.runner import _archive_previous
    assert _archive_previous(make_config(tmp_path), str(tmp_path / "nope.html")) == []
    assert not (tmp_path / "logs").exists()


# ── Single-question rerun ─────────────────────────────────────────────
def test_single_question_merges_into_existing_results(tmp_path, problems, backend, scorer):
    run_eval(make_config(tmp_path), problems=problems, backend=backend, scorer=scorer)
    before = json.loads((tmp_path / "eval_results.json").read_text())

    cfg = make_config(tmp_path, question=3)
    summary = run_eval(cfg, problems=[problems[2]], backend=backend, scorer=scorer)

    assert len(summary["results"]) == len(before["results"]), "must not truncate the run"
    assert summary["results"][2]["index"] == 2


def test_single_question_does_not_archive_the_run(tmp_path, problems, backend, scorer):
    run_eval(make_config(tmp_path), problems=problems, backend=backend, scorer=scorer)
    cfg = make_config(tmp_path, question=1)
    run_eval(cfg, problems=[problems[0]], backend=backend, scorer=scorer)
    assert os.path.exists(tmp_path / "eval_results.json")


# ── Self-consistency ──────────────────────────────────────────────────
def test_k_greater_than_one_records_every_strategy(tmp_path, problems, scorer):
    cfg = make_config(tmp_path, k=3, temperature=0.8)
    summary = run_eval(cfg, problems=problems[:4], backend=MockBackend(accuracy=0.6),
                       scorer=scorer)
    row = summary["results"][0]
    assert len(row["samples"]) == 3
    assert set(row["selection"]) == {"first", "majority", "prm_best", "prm_weighted"}
    assert set(summary["selection_accuracy"]) == set(row["selection"])


def test_headline_accuracy_is_the_single_shot_baseline(tmp_path, problems, scorer):
    """Reporting a best-of-n number as 'accuracy' would overstate the model."""
    cfg = make_config(tmp_path, k=3, temperature=0.8)
    summary = run_eval(cfg, problems=problems[:6], backend=MockBackend(accuracy=0.6),
                       scorer=scorer)
    assert summary["accuracy"] == summary["selection_accuracy"]["first"]


def test_temperature_defaults_to_sampling_when_k_above_one():
    assert EvalConfig(k=1).resolved_temperature() == 0.0
    assert EvalConfig(k=5).resolved_temperature() > 0
    assert EvalConfig(k=5, temperature=0.0).resolved_temperature() == 0.0


# ── Aggregation ───────────────────────────────────────────────────────
def test_aggregate_counts_parse_failures(tmp_path):
    rows = [
        {"answer_accuracy": True, "parse_matched": True, "predicted": "1", "prm_score": 0.8},
        {"answer_accuracy": False, "parse_matched": False, "predicted": "x", "prm_score": 0.2},
        {"answer_accuracy": False, "parse_matched": True, "predicted": "", "prm_score": 0.3},
    ]
    summary = aggregate(rows, make_config(tmp_path), "m")
    assert summary["parse_failures"] == 2
    assert summary["accuracy"] == pytest.approx(1 / 3)


def test_aggregate_survives_null_token_counts(tmp_path):
    """Regression: a 500-question run crashed here at the final step because
    one truncated response recorded eval_tokens as None. Results files and
    checkpoints written before the backend fix still contain those nulls, so
    aggregation has to tolerate them rather than assume clean input."""
    rows = [
        {"answer_accuracy": True, "parse_matched": True, "predicted": "1",
         "prm_score": 0.8, "eval_tokens": 100},
        {"answer_accuracy": False, "parse_matched": False, "predicted": "",
         "prm_score": None, "eval_tokens": None},
    ]
    summary = aggregate(rows, make_config(tmp_path), "m")
    assert summary["total_output_tokens"] == 100
    assert summary["empty_responses"] == 1


def test_aggregate_counts_empty_responses(tmp_path):
    rows = [
        {"answer_accuracy": True, "parse_matched": True, "predicted": "5", "eval_tokens": 1},
        {"answer_accuracy": False, "parse_matched": False, "predicted": "", "eval_tokens": 0},
        {"answer_accuracy": False, "parse_matched": False, "predicted": "   ", "eval_tokens": 0},
    ]
    assert aggregate(rows, make_config(tmp_path), "m")["empty_responses"] == 2


def test_aggregate_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        aggregate([], make_config(tmp_path), "m")


def test_solve_question_survives_an_empty_response(tmp_path, scorer, problems):
    class EmptyBackend(MockBackend):
        def generate(self, prompt, max_tokens=2048, temperature=0.0, context=None):
            from evalkit.backends import Generation
            return Generation(text="", warning="empty response")

    log = RunLogger(str(tmp_path / "log.txt"), verbose=False, echo=False)
    result = solve_question(problems[0], 1, 1, EmptyBackend(), scorer,
                            make_config(tmp_path), log)
    assert result["answer_accuracy"] is False
    assert result["parse_matched"] is False
