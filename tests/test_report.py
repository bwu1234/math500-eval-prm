"""Report rendering tests.

The report is the artifact people actually look at, so the bar is: it must
never crash on degenerate input, and it must never inject raw model output
into the page unescaped -- MATH-500 problems are full of LaTeX and angle
brackets.
"""

import json

import pytest

from evalkit.backends import MockBackend
from evalkit.report import (
    bar_chart,
    build_report,
    calibration_chart,
    heatmap,
    render_report,
    score_color,
    score_histogram,
    write_report,
)
from evalkit.runner import EvalConfig, run_eval


@pytest.fixture
def rendered(tmp_path, problems, scorer):
    cfg = EvalConfig(num_questions=12, backend="mock", use_prm=True,
                     verbose=False, out_dir=str(tmp_path))
    run_eval(cfg, problems=problems, backend=MockBackend(accuracy=0.6), scorer=scorer)
    return render_report(build_report(out_dir=str(tmp_path))), tmp_path


# ── Structure ─────────────────────────────────────────────────────────
def test_report_contains_every_section(rendered):
    html, _ = rendered
    for heading in ("Where the model fails", "Process reward analysis", "Questions"):
        assert heading in html


def test_report_is_self_contained(rendered):
    """No external fetches: the page must render offline, from a file:// URL."""
    html, _ = rendered
    for forbidden in ("http://", "https://", "<script", "cdn."):
        assert forbidden not in html.lower()


def test_report_renders_charts(rendered):
    html, _ = rendered
    assert html.count("<svg") >= 3


def test_write_report_creates_file(tmp_path, problems, scorer):
    cfg = EvalConfig(num_questions=12, backend="mock", verbose=False,
                     out_dir=str(tmp_path))
    run_eval(cfg, problems=problems, backend=MockBackend(), scorer=scorer)
    path = write_report(build_report(out_dir=str(tmp_path)), str(tmp_path / "r.html"))
    assert (tmp_path / "r.html").read_text().startswith("<title>")
    assert path.endswith("r.html")


def test_missing_results_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run an eval first"):
        build_report(out_dir=str(tmp_path))


# ── Escaping ──────────────────────────────────────────────────────────
def test_model_output_is_escaped(tmp_path, scorer):
    """A problem containing markup must not become markup in the report."""
    nasty = [{
        "index": 0,
        "problem": "<script>alert('xss')</script> & x < 5 \\frac{1}{2}",
        "answer": "<b>7</b>",
        "subject": "Algebra", "level": 1,
    }]
    cfg = EvalConfig(num_questions=1, backend="mock", verbose=False,
                     out_dir=str(tmp_path))
    run_eval(cfg, problems=nasty, backend=MockBackend(accuracy=1.0), scorer=scorer)
    html = render_report(build_report(out_dir=str(tmp_path)))

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_synthetic_runs_are_labelled(rendered):
    """Mock output must be impossible to mistake for a real measurement."""
    html, _ = rendered
    assert "Synthetic data" in html


# ── Degenerate input ──────────────────────────────────────────────────
def test_charts_handle_empty_input():
    assert bar_chart([], "t") == ""
    assert heatmap({}, "t") == ""
    assert heatmap({"rows": [], "cols": [], "cells": {}}, "t") == ""
    assert calibration_chart([]) == ""
    assert score_histogram([], []) == ""


def test_calibration_chart_needs_two_populated_bins():
    one_bin = [{"lo": 0.0, "hi": 0.1, "count": 5, "correct": 3,
                "accuracy": 0.6, "mean_score": 0.05}]
    assert calibration_chart(one_bin) == ""


def test_histogram_with_only_one_class():
    svg = score_histogram([0.2, 0.9], [])
    assert svg.startswith("<svg")


def test_report_with_a_single_question(tmp_path, scorer, problems):
    cfg = EvalConfig(num_questions=1, backend="mock", verbose=False,
                     out_dir=str(tmp_path))
    run_eval(cfg, problems=problems[:1], backend=MockBackend(), scorer=scorer)
    assert render_report(build_report(out_dir=str(tmp_path))).startswith("<title>")


def test_report_without_prm(tmp_path, problems):
    from evalkit.prm import NullScorer
    cfg = EvalConfig(num_questions=12, backend="mock", use_prm=False,
                     verbose=False, out_dir=str(tmp_path))
    run_eval(cfg, problems=problems, backend=MockBackend(), scorer=NullScorer())
    html = render_report(build_report(out_dir=str(tmp_path)))
    assert "--no-prm" in html  # explains the empty panel rather than showing zeros


def test_report_names_the_outcome_model_when_the_orm_scored(
        tmp_path, problems, outcome_scorer):
    """The scores share one column whichever model produced them, so the page
    has to attribute them correctly -- an ORM score captioned as a step reward
    describes a measurement that never happened."""
    cfg = EvalConfig(num_questions=12, backend="mock", use_prm=True,
                     scorer_kind="orm", verbose=False, out_dir=str(tmp_path))
    run_eval(cfg, problems=problems, backend=MockBackend(accuracy=0.6),
             scorer=outcome_scorer)
    html = render_report(build_report(out_dir=str(tmp_path)))

    assert "Outcome reward analysis" in html
    assert "Process reward analysis" not in html
    assert "fake-orm" in html
    assert "PRM800K" not in html
    assert "no per-step shading" in html


def test_report_still_credits_the_prm_by_default(rendered):
    html, _ = rendered
    assert "Process reward analysis" in html
    assert "Outcome reward analysis" not in html
    assert "PRM800K" in html


def test_report_includes_comparison_when_present(tmp_path, problems, scorer):
    cfg = EvalConfig(num_questions=12, backend="mock", verbose=False,
                     out_dir=str(tmp_path))
    run_eval(cfg, problems=problems, backend=MockBackend(), scorer=scorer)
    (tmp_path / "eval_comparison.json").write_text(json.dumps({
        "backends": ["gemini", "qwen"],
        "num_questions": 2,
        "accuracy": {"gemini": 0.5, "qwen": 1.0},
        "agreement": {"gemini|qwen": 0.5},
        "num_divergences": 1,
        "divergences": [{
            "index": 1, "problem": "p", "expected": "2", "subject": "Algebra",
            "difficulty": 3, "outcomes": {"gemini": False, "qwen": True},
            "predicted": {"gemini": "9", "qwen": "2"},
            "prm_score": {"gemini": 0.2, "qwen": 0.8},
        }],
    }))
    html = render_report(build_report(out_dir=str(tmp_path)))
    assert "Backend comparison" in html and "Divergent questions" in html


def test_score_color_ramp():
    assert score_color(None) == "var(--muted)"
    assert "hsl(0" in score_color(0.0)      # red at zero reward
    assert "hsl(120" in score_color(1.0)    # green at full reward
    assert score_color(2.0) == score_color(1.0)  # clamped
