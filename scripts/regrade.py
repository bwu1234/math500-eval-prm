"""Re-grade a finished run against the current grader, without re-running it.

Model output is expensive and the grader is not: a results file already stores
every prediction and every step, so a grading fix can be applied to a completed
run for free. Anything that depends only on ``expected`` vs ``predicted`` is
recomputed -- per-sample accuracy, the per-strategy selection accuracy, and all
of the aggregate breakdowns. PRM scores and token counts are model output and
are copied through untouched.

Run from the repo root:

    python scripts/regrade.py                     # report the diff only
    python scripts/regrade.py --write             # rewrite eval_results.json
    python scripts/regrade.py --write --report    # ...and rebuild report.html
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.analysis import SELECTION_STRATEGIES  # noqa: E402
from evalkit.answers import final_answer_correct  # noqa: E402
from evalkit.report import build_report, write_report  # noqa: E402
from evalkit.runner import (  # noqa: E402
    DEFAULT_REPORT,
    EvalConfig,
    aggregate,
    report_targets,
)


def regrade_row(row: dict) -> list[str]:
    """Re-grade one question in place. Returns the fields that changed."""
    expected = row["expected"]
    changed = []

    for sample in row.get("samples") or []:
        was = sample.get("answer_accuracy")
        now = final_answer_correct(expected, sample.get("predicted", ""))
        if now != was:
            sample["answer_accuracy"] = now
            changed.append("sample")

    for strategy, picked in (row.get("selection") or {}).items():
        was = picked.get("correct")
        now = final_answer_correct(expected, picked.get("predicted", ""))
        if now != was:
            picked["correct"] = now
            changed.append(strategy)

    was = row.get("answer_accuracy")
    now = final_answer_correct(expected, row.get("predicted", ""))
    if now != was:
        row["answer_accuracy"] = now
        changed.append("final")

    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="eval_results.json")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the results file (a .bak copy is kept)")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the HTML report after writing")
    args = ap.parse_args()

    with open(args.results) as fh:
        old = json.load(fh)
    rows = old["results"]

    flipped_right, flipped_wrong = [], []
    for row in rows:
        was_correct = row.get("answer_accuracy")
        if "final" in regrade_row(row):
            (flipped_right if row["answer_accuracy"] else flipped_wrong).append(row)
        elif was_correct != row.get("answer_accuracy"):  # defensive
            raise AssertionError("regrade_row did not report a change it made")

    # Rebuild every derived number through the same path a real run uses, so a
    # re-graded file cannot drift from a freshly produced one.
    config = EvalConfig(
        num_questions=old["num_questions"],
        backend=old["model_backend"],
        model=old["model"],
        k=old["k"],
        temperature=old["temperature"],
        use_prm=old["prm_enabled"],
    )
    new = aggregate(rows, config, old["model"],
                    duration=old.get("duration_s"),
                    synthetic=old.get("synthetic", False))
    new["timestamp"] = old["timestamp"]
    new["regraded"] = True

    print(f"{args.results}: {old['num_questions']} questions")
    print(f"  accuracy {old['accuracy']:.1%} -> {new['accuracy']:.1%} "
          f"({old['correct']} -> {new['correct']} correct)")
    print(f"  now correct: {len(flipped_right)}   now wrong: {len(flipped_wrong)}")
    for row in flipped_right:
        print(f"    + [{row['index']:3d}] {row['expected']!r} <- {row['predicted']!r}")
    for row in flipped_wrong:
        print(f"    - [{row['index']:3d}] {row['expected']!r} <- {row['predicted']!r}")

    for strategy in SELECTION_STRATEGIES:
        was = (old.get("selection_accuracy") or {}).get(strategy)
        now = (new.get("selection_accuracy") or {}).get(strategy)
        if was is not None and now is not None and abs(was - now) > 1e-12:
            print(f"  selection[{strategy}] {was:.1%} -> {now:.1%}")

    still_wrong = [r for r in rows if not r["answer_accuracy"]]
    empty = [r for r in still_wrong if not (r.get("predicted") or "").strip()]
    print(f"  remaining failures: {len(still_wrong)} "
          f"({len(empty)} with no model output)")

    if not args.write:
        print("\n(dry run -- pass --write to update the results file)")
        return 0

    shutil.copyfile(args.results, args.results + ".bak")
    with open(args.results, "w") as fh:
        json.dump(new, fh, indent=2)
    print(f"\nwrote {args.results} (previous version at {args.results}.bak)")

    if args.report:
        out_dir = os.path.dirname(os.path.abspath(args.results)) or "."
        report = build_report(out_dir=out_dir,
                             results_file=os.path.basename(args.results))
        # Re-grading does not change how many questions were answered, so the
        # same rule as a live run applies: only a full 500-question result may
        # claim report.html.
        for target in report_targets(DEFAULT_REPORT,
                                     new.get("model_backend") or "model",
                                     new.get("num_questions") or 0):
            print(f"wrote {write_report(report, os.path.join(out_dir, target))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
