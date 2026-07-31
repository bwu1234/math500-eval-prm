#!/usr/bin/env python3
"""Evaluate a model on MATH-500, scoring reasoning steps with a Process Reward Model.

Examples
--------
    python math500_eval.py -n 20                       # 20 questions via Gemini
    python math500_eval.py -n 20 -m qwen               # local Ollama model
    python math500_eval.py -n 20 -m qwen --think off   # disable thinking mode
    python math500_eval.py -n 50 -k 5                  # self-consistency, 5 samples
    python math500_eval.py -n 20 --compare gemini,qwen # diff two backends
    python math500_eval.py -n 20 -m mock --no-prm      # smoke test, no GPU or API key
    python math500_eval.py --report-only               # rebuild report from results
"""

import argparse
import glob
import os
import sys

from evalkit.backends import BACKEND_NAMES, THINK_MODES
from evalkit.report import build_report, write_report
from evalkit.runner import (
    EvalConfig,
    is_full_run,
    report_targets,
    run_comparison,
    run_eval,
)


def emit_report(config, out_dir, path, results_file="eval_results.json",
                backend=None, canonical=True) -> list[str] | None:
    """Build the report and write it everywhere this run has earned it.

    How many questions the run actually covered decides the destination, so
    the results file is the authority rather than the flags: a rerun spliced
    into a finished 500-question run is still a full run, and a run that was
    cut short is not one however it was launched.

    Any failure is downgraded to a warning. A run can be hours of GPU time; a
    bug in report rendering must not be the thing that loses it, and the
    results file is already on disk by this point.
    """
    try:
        report = build_report(out_dir=out_dir, results_file=results_file)
        summary = report["summary"]
        num_questions = summary.get("num_questions") or 0
        targets = report_targets(
            path, summary.get("model_backend") or backend or config.backend,
            num_questions, canonical=canonical)
        written = [write_report(report, config.path(t)) for t in targets]
        for target in written:
            print(f"  Report:    {target}")
        if not is_full_run(num_questions):
            print(f"  Partial run ({num_questions} questions): "
                  f"{path} left as it was")
        return written
    except Exception as e:  # noqa: BLE001 -- the run itself already succeeded
        print(f"warning: could not generate the report ({type(e).__name__}: {e})\n"
              f"         results are intact in {os.path.join(out_dir, results_file)}; "
              f"retry with --report-only", file=sys.stderr)
        return None


def emit_comparison_reports(config, out_dir, path, results_files: dict) -> None:
    """One report per backend, each carrying the shared comparison section.

    Building a single report here would mean picking one backend's results to
    head the page while the comparison covers all of them -- which is also why
    no backend in a comparison may claim ``report.html``.
    """
    for name, results_file in results_files.items():
        emit_report(config, out_dir, path, results_file,
                    backend=name, canonical=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a model on MATH-500 with process-level reward scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n")[-1],
    )
    p.add_argument("-n", "--num-questions", type=int, default=3,
                   help="Number of MATH-500 questions to run (default: 3, max: 500)")
    p.add_argument("-m", "--model", dest="backend", choices=BACKEND_NAMES,
                   default="gemini", help="Model backend (default: gemini)")
    p.add_argument("--model-name", default=None,
                   help="Override the backend's default model identifier")
    p.add_argument("-q", "--question", type=int, default=None,
                   help="Rerun only this 1-indexed question, merging the result "
                        "into the existing eval_results.json instead of overwriting it")
    p.add_argument("-k", "--samples", type=int, default=1, dest="k",
                   help="Samples per question. k>1 enables self-consistency and "
                        "best-of-n reranking (default: 1)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (default: 0 for k=1, 0.7 for k>1)")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Max output tokens per solution (default: 16384)")
    p.add_argument("--no-prm", action="store_true",
                   help="Skip Process Reward Model scoring (no GPU required)")
    p.add_argument("--think", choices=THINK_MODES, default="merge",
                   help="For thinking-capable Ollama models: 'merge' folds the "
                        "model's separate thinking trace into the graded text "
                        "(default), 'off' disables thinking so the response is "
                        "used as-is. Ignored by other backends.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any checkpoint and start the run from scratch")
    p.add_argument("--compare", default=None, metavar="A,B",
                   help="Run the same questions through several backends and "
                        "report where they diverge")
    p.add_argument("--report", default="report.html", metavar="PATH",
                   help="Where to write the HTML report, generated automatically "
                        "after every run (default: report.html). Only a full "
                        "500-question run writes this path itself, alongside a "
                        "per-model copy (report_qwen.html); a shorter run "
                        "writes report_<model>_partial.html and leaves it alone")
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report generation")
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild the HTML report from existing results without "
                        "running the model")
    p.add_argument("--out-dir", default=".", help="Directory for outputs (default: .)")
    p.add_argument("--quiet", action="store_true", help="Less verbose logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    config = EvalConfig(
        num_questions=args.num_questions,
        backend=args.backend,
        model=args.model_name,
        question=args.question,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_prm=not args.no_prm,
        resume=not args.no_resume,
        think=args.think,
        verbose=not args.quiet,
        out_dir=args.out_dir,
        # The runner archives the previous run's report alongside its log and
        # results, so it has to know where the report lives.
        report_file=args.report,
    )

    if args.k < 1:
        print("error: -k must be at least 1", file=sys.stderr)
        return 2
    if args.k > 1 and config.resolved_temperature() == 0:
        print("warning: k>1 at temperature 0 produces k identical samples; "
              "self-consistency will be a no-op.", file=sys.stderr)

    report_path = None if args.no_report else args.report

    if args.report_only:
        # A comparison run writes per-backend results and no eval_results.json,
        # so fall back to whatever this directory actually holds.
        default = os.path.join(args.out_dir, "eval_results.json")
        if os.path.exists(default):
            return 0 if emit_report(config, args.out_dir, args.report) else 1
        found = sorted(glob.glob(os.path.join(args.out_dir, "eval_results_*.json")))
        if not found:
            print(f"error: no results found in {args.out_dir}. Run an eval first.",
                  file=sys.stderr)
            return 2
        names = [os.path.basename(p)[len("eval_results_"):-len(".json")] for p in found]
        emit_comparison_reports(
            config, args.out_dir, args.report,
            {n: os.path.basename(p) for n, p in zip(names, found)},
        )
        return 0

    if args.compare:
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        unknown = [n for n in names if n not in BACKEND_NAMES]
        if unknown:
            print(f"error: unknown backend(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        if len(names) < 2:
            print("error: --compare needs at least two backends", file=sys.stderr)
            return 2
        comparison = run_comparison(config, names)
        if report_path:
            emit_comparison_reports(config, args.out_dir, report_path,
                                    comparison["results_files"])
    else:
        run_eval(config)
        if report_path:
            emit_report(config, args.out_dir, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
