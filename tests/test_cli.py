"""CLI-level tests, including automatic report generation.

The report is written after the model work is finished, so these check both
that it appears by default and that a failure to render it cannot destroy a
run that already succeeded.

Most runs here are shorter than MATH-500, so they land on the partial path
(``report_mock_partial.html``) rather than on ``report.html`` -- see the
"Only a full run owns report.html" section for that rule.
"""

import json
import os

import pytest

import evalkit.runner as runner
import math500_eval as cli

PARTIAL = "report_mock_partial.html"


def run(tmp_path, *args):
    return cli.main(["--out-dir", str(tmp_path), "-m", "mock", "--no-prm",
                     "--quiet", *args])


@pytest.fixture(autouse=True)
def stub_dataset(monkeypatch, problems):
    """Keep the CLI off the network; MATH-500 loading is covered elsewhere."""
    monkeypatch.setattr(runner, "load_problems",
                        lambda n, q=None, dataset=None: problems[:n] if q is None
                        else [problems[q - 1]])


@pytest.fixture
def full_run_is(monkeypatch):
    """Shrink what counts as a full run, so the rule is testable in milliseconds.

    That the threshold is 500 is asserted directly in ``test_runner.py``; what
    these tests need is the difference between a run that reaches it and one
    that does not.
    """
    return lambda n: monkeypatch.setattr(runner, "FULL_RUN", n)


# ── Automatic generation ──────────────────────────────────────────────
def test_report_is_generated_without_being_asked(tmp_path):
    assert run(tmp_path, "-n", "6") == 0
    assert (tmp_path / PARTIAL).exists()


def test_report_path_is_configurable(tmp_path):
    assert run(tmp_path, "-n", "4", "--report", "custom.html") == 0
    assert (tmp_path / "custom_mock_partial.html").exists()
    assert not (tmp_path / PARTIAL).exists()


def test_no_report_opts_out(tmp_path):
    assert run(tmp_path, "-n", "4", "--no-report") == 0
    assert not (tmp_path / PARTIAL).exists()
    assert (tmp_path / "eval_results.json").exists()


def test_report_reflects_the_run_just_completed(tmp_path):
    run(tmp_path, "-n", "6")
    summary = json.loads((tmp_path / "eval_results.json").read_text())
    html = (tmp_path / PARTIAL).read_text()
    assert f"{summary['num_questions']} questions" in html


def test_single_question_rerun_refreshes_the_whole_report(tmp_path):
    """The rerun merges into the existing results, so the regenerated report
    must still cover all 6 questions rather than collapsing to the one."""
    run(tmp_path, "-n", "6")
    assert run(tmp_path, "-q", "2") == 0

    summary = json.loads((tmp_path / "eval_results.json").read_text())
    assert summary["num_questions"] == 6
    html = (tmp_path / PARTIAL).read_text()
    assert "6 questions" in html
    assert html.count("<details") == 6


# ── Only a full run owns report.html ──────────────────────────────────
def test_a_full_run_writes_report_html_and_a_per_model_copy(tmp_path, full_run_is):
    full_run_is(6)
    assert run(tmp_path, "-n", "6") == 0
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report_mock.html").exists()
    assert not (tmp_path / PARTIAL).exists()


def test_a_partial_run_leaves_the_full_run_report_untouched(tmp_path, full_run_is):
    """The point of the rule: a spot check must not overwrite the artifact
    that stands for 500 questions."""
    full_run_is(6)
    run(tmp_path, "-n", "6")
    full = (tmp_path / "report.html").read_text()

    assert run(tmp_path, "-n", "4") == 0
    assert (tmp_path / "report.html").read_text() == full
    assert (tmp_path / "report_mock.html").read_text() == full
    assert (tmp_path / PARTIAL).exists()


def test_a_partial_run_does_not_archive_the_full_run_report(tmp_path, full_run_is):
    """Archiving report.html would retire it just as surely as overwriting it."""
    full_run_is(6)
    run(tmp_path, "-n", "6")
    run(tmp_path, "-n", "4")
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report_mock.html").exists()
    assert [p.name for p in (tmp_path / "logs").glob("report*.html")] == []


def test_each_model_keeps_its_own_full_run_report(tmp_path, monkeypatch, full_run_is):
    """A finished 500-question run must survive the next model's."""
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(
            accuracy=0.8 if name == "gemini" else 0.4, seed=len(name), on_log=on_log))
    full_run_is(6)

    cli.main(["--out-dir", str(tmp_path), "-m", "gemini", "--no-prm", "--quiet",
              "-n", "6"])
    gemini = (tmp_path / "report_gemini.html").read_text()

    cli.main(["--out-dir", str(tmp_path), "-m", "qwen", "--no-prm", "--quiet",
              "-n", "6"])
    assert (tmp_path / "report_gemini.html").read_text() == gemini
    assert (tmp_path / "report_qwen.html").exists()
    # report.html always tracks the most recent full run.
    assert (tmp_path / "report.html").read_text() == \
        (tmp_path / "report_qwen.html").read_text()


def test_a_rerun_into_a_full_run_still_counts_as_full(tmp_path, full_run_is):
    """``-q`` merges into the existing results, so the merged run is still the
    full set and report.html must be refreshed rather than side-lined."""
    full_run_is(6)
    run(tmp_path, "-n", "6")
    before = (tmp_path / "report.html").read_text()

    assert run(tmp_path, "-q", "2") == 0
    assert (tmp_path / "report.html").exists()
    assert not (tmp_path / PARTIAL).exists()
    assert before  # a real report, not an empty file


def test_a_run_that_finishes_short_does_not_claim_report_html(
        tmp_path, monkeypatch, full_run_is, problems):
    """The results file decides, not the flag: a run launched as a full one
    that only covers 4 questions is still a partial measurement."""
    full_run_is(6)
    monkeypatch.setattr(runner, "load_problems",
                        lambda n, q=None, dataset=None: problems[:4])
    assert run(tmp_path, "-n", "6") == 0
    assert not (tmp_path / "report.html").exists()
    assert (tmp_path / PARTIAL).exists()


# ── Archiving ─────────────────────────────────────────────────────────
def test_previous_report_is_archived_not_overwritten(tmp_path):
    """Scoring a new model must not destroy the previous model's report."""
    run(tmp_path, "-n", "4")
    first = (tmp_path / PARTIAL).read_text()

    run(tmp_path, "-n", "6")
    archived = list((tmp_path / "logs").glob("report_*.html"))
    assert len(archived) == 1
    assert archived[0].read_text() == first
    assert (tmp_path / PARTIAL).read_text() != first


def test_each_run_leaves_one_archived_report(tmp_path):
    for n in ("2", "3", "4", "5"):
        run(tmp_path, "-n", n)
    # Four runs -> three archived predecessors plus the current report.
    assert len(list((tmp_path / "logs").glob("report_*.html"))) == 3
    assert (tmp_path / PARTIAL).exists()


def test_a_run_archives_its_predecessor_as_one_correlated_set(tmp_path):
    """Log, results and report of a given run share a timestamp so they can
    be matched up after the fact."""
    run(tmp_path, "-n", "4")
    run(tmp_path, "-n", "4")
    logs = tmp_path / "logs"

    def stamp(pattern, prefix, suffix):
        (path,) = list(logs.glob(pattern))
        return path.name[len(prefix):-len(suffix)]

    assert (stamp("report_*.html", "report_mock_partial_", ".html")
            == stamp("eval_results_*.json", "eval_results_", ".json")
            == stamp("eval_debug_*.log", "eval_debug_", ".log"))


def test_stale_report_is_archived_even_with_no_report(tmp_path):
    """Otherwise --no-report would leave the old report sitting next to the
    results of a newer, different run."""
    run(tmp_path, "-n", "4")
    assert (tmp_path / PARTIAL).exists()

    run(tmp_path, "-n", "6", "--no-report")
    assert not (tmp_path / PARTIAL).exists()
    assert len(list((tmp_path / "logs").glob("report_*.html"))) == 1


def test_custom_report_name_is_archived_under_its_own_name(tmp_path):
    run(tmp_path, "-n", "4", "--report", "scorecard.html")
    run(tmp_path, "-n", "4", "--report", "scorecard.html")
    assert list((tmp_path / "logs").glob("scorecard_*.html"))


def test_single_question_rerun_does_not_archive(tmp_path):
    """A rerun merges into the current run; it is not a new run."""
    run(tmp_path, "-n", "6")
    run(tmp_path, "-q", "2")
    assert not list((tmp_path / "logs").glob("report_*.html"))


def test_compare_archives_each_backend_separately(tmp_path, monkeypatch):
    """The previous archiver flattened every backend onto one name, losing
    which results belonged to which model."""
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(seed=len(name), on_log=on_log))

    run(tmp_path, "-n", "4", "--compare", "gemini,qwen")
    run(tmp_path, "-n", "4", "--compare", "gemini,qwen")

    logs = tmp_path / "logs"
    assert list(logs.glob("report_gemini_*.html"))
    assert list(logs.glob("report_qwen_*.html"))
    assert list(logs.glob("eval_results_gemini_*.json"))
    assert list(logs.glob("eval_results_qwen_*.json"))


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
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(
            accuracy=0.8 if name == "gemini" else 0.4, seed=len(name), on_log=on_log)
    )
    assert run(tmp_path, "-n", "6", "--compare", "gemini,qwen") == 0
    assert (tmp_path / "report_gemini_partial.html").exists()
    assert (tmp_path / "report_qwen_partial.html").exists()
    assert not (tmp_path / "report.html").exists()


def test_compare_at_full_length_writes_a_report_per_model_but_no_report_html(
        tmp_path, monkeypatch, full_run_is):
    """Every backend in the comparison covered all 500 questions, so no single
    one of them can head report.html."""
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(seed=len(name), on_log=on_log))
    full_run_is(6)

    assert run(tmp_path, "-n", "6", "--compare", "gemini,qwen") == 0
    assert (tmp_path / "report_gemini.html").exists()
    assert (tmp_path / "report_qwen.html").exists()
    assert not (tmp_path / "report.html").exists()


def test_compare_report_never_uses_a_stale_results_file(tmp_path, monkeypatch):
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
        assert "STALE-MODEL" not in (tmp_path / f"report_{name}_partial.html").read_text()


# ── report-only ───────────────────────────────────────────────────────
def test_report_only_rebuilds_without_running_the_model(tmp_path):
    run(tmp_path, "-n", "6", "--no-report")
    assert not (tmp_path / PARTIAL).exists()
    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / PARTIAL).exists()


def test_report_only_obeys_the_full_run_rule(tmp_path, full_run_is):
    """The rule belongs to the results, so rebuilding cannot smuggle a short
    run onto report.html -- or strand a full one off it."""
    full_run_is(6)
    run(tmp_path, "-n", "4", "--no-report")
    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert not (tmp_path / "report.html").exists()

    run(tmp_path, "-n", "6", "--no-report")
    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report_mock.html").exists()


def test_report_only_finds_per_backend_results(tmp_path, monkeypatch):
    from evalkit.backends import MockBackend
    monkeypatch.setattr(
        runner, "build_backend",
        lambda name, model=None, on_log=None, **kw: MockBackend(seed=len(name), on_log=on_log))
    run(tmp_path, "-n", "6", "--compare", "gemini,qwen", "--no-report")
    assert not (tmp_path / "report_gemini_partial.html").exists()

    assert cli.main(["--out-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "report_gemini_partial.html").exists()


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
