"""CLI-level tests, including automatic report generation.

The report is written after the model work is finished, so these check both
that it appears by default and that a failure to render it cannot destroy a
run that already succeeded.
"""

import json
import os

import pytest

import math500_eval as cli


def run(tmp_path, *args):
    return cli.main(["--out-dir", str(tmp_path), "-m", "mock", "--no-prm",
                     "--quiet", *args])


@pytest.fixture(autouse=True)
def stub_dataset(monkeypatch, problems):
    """Keep the CLI off the network; MATH-500 loading is covered elsewhere."""
    import evalkit.runner as runner
    monkeypatch.setattr(runner, "load_problems",
                        lambda n, q=None, dataset=None: problems[:n] if q is None
                        else [problems[q - 1]])


# ── Automatic generation ──────────────────────────────────────────────
def test_report_is_generated_without_being_asked(tmp_path):
    assert run(tmp_path, "-n", "6") == 0
    assert (tmp_path / "report.html").exists()


def test_report_path_is_configurable(tmp_path):
    assert run(tmp_path, "-n", "4", "--report", "custom.html") == 0
    assert (tmp_path / "custom.html").exists()
    assert not (tmp_path / "report.html").exists()


def test_no_report_opts_out(tmp_path):
    assert run(tmp_path, "-n", "4", "--no-report") == 0
    assert not (tmp_path / "report.html").exists()
    assert (tmp_path / "eval_results.json").exists()


def test_report_reflects_the_run_just_completed(tmp_path):
    run(tmp_path, "-n", "6")
    summary = json.loads((tmp_path / "eval_results.json").read_text())
    html = (tmp_path / "report.html").read_text()
    assert f"{summary['num_questions']} questions" in html


def test_single_question_rerun_refreshes_the_whole_report(tmp_path):
    """The rerun merges into the existing results, so the regenerated report
    must still cover all 6 questions rather than collapsing to the one."""
    run(tmp_path, "-n", "6")
    assert run(tmp_path, "-q", "2") == 0

    summary = json.loads((tmp_path / "eval_results.json").read_text())
    assert summary["num_questions"] == 6
    html = (tmp_path / "report.html").read_text()
    assert "6 questions" in html
    assert html.count("<details") == 6


# ── Report failure must not sink a completed run ──────────────────────
def test_report_failure_is_a_warning_not_a_crash(tmp_path, monkeypatch, capsys):
    """The expensive part already finished; losing it to a rendering bug would
    be the worst possible trade."""
    monkeypatch.setattr(cli, "write_report",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run(tmp_path, "-n", "4") == 0
    assert (tmp_path / "eval_results.json").exists()
    assert "could not generate the report" in capsys.readouterr().err


# ── Comparison mode ───────────────────────────────────────────────────
def test_compare_writes_one_report_per_backend(tmp_path, monkeypatch):
    """A comparison run writes no eval_results.json, so a single report would
    have to fall back to a stale file from an unrelated earlier run."""
    import evalkit.runner as runner
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(
            accuracy=0.8 if name == "gemini" else 0.4, seed=len(name), on_log=on_log)
    )
    assert run(tmp_path, "-n", "6", "--compare", "gemini,qwen") == 0
    assert (tmp_path / "report_gemini.html").exists()
    assert (tmp_path / "report_qwen.html").exists()
    assert not (tmp_path / "report.html").exists()


def test_compare_report_never_uses_a_stale_results_file(tmp_path, monkeypatch):
    import evalkit.runner as runner
    from evalkit.backends import MockBackend

    # Leave a stale results file behind from an unrelated earlier run.
    (tmp_path / "eval_results.json").write_text(json.dumps(
        {"num_questions": 999, "accuracy": 0.0, "correct": 0, "results": [],
         "model": "STALE-MODEL"}))

    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(seed=len(name), on_log=on_log))
    run(tmp_path, "-n", "6", "--compare", "gemini,qwen")

    for name in ("gemini", "qwen"):
        assert "STALE-MODEL" not in (tmp_path / f"report_{name}.html").read_text()


# ── report-only ───────────────────────────────────────────────────────
def test_report_only_rebuilds_without_running_the_model(tmp_path):
    run(tmp_path, "-n", "6", "--no-report")
    assert not (tmp_path / "report.html").exists()
    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "report.html").exists()


def test_report_only_finds_per_backend_results(tmp_path, monkeypatch):
    import evalkit.runner as runner
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(seed=len(name), on_log=on_log))
    run(tmp_path, "-n", "6", "--compare", "gemini,qwen", "--no-report")
    assert not (tmp_path / "report_gemini.html").exists()

    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "report_gemini.html").exists()


def test_report_only_without_results_is_an_error(tmp_path, capsys):
    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 2
    assert "no results found" in capsys.readouterr().err


# ── Argument validation ───────────────────────────────────────────────
def test_rejects_zero_samples(tmp_path, capsys):
    assert run(tmp_path, "-k", "0") == 2
    assert "at least 1" in capsys.readouterr().err


def test_rejects_unknown_compare_backend(tmp_path, capsys):
    assert run(tmp_path, "--compare", "gemini,gpt9") == 2
    assert "unknown backend" in capsys.readouterr().err


def test_rejects_single_backend_comparison(tmp_path, capsys):
    assert run(tmp_path, "--compare", "gemini") == 2
    assert "at least two" in capsys.readouterr().err


def test_warns_when_sampling_is_degenerate(tmp_path, capsys):
    run(tmp_path, "-n", "2", "-k", "3", "--temperature", "0")
    assert "identical samples" in capsys.readouterr().err
