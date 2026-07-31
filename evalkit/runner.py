"""The eval pipeline: load -> generate -> score -> grade -> aggregate -> persist.

Each stage is a separate function taking plain data, so the expensive stages
(generation, PRM scoring) can be swapped for test doubles and the aggregation
logic can be exercised on fixture rows without a model anywhere in sight.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .analysis import (
    SELECTION_STRATEGIES,
    compute_group_accuracy,
    cross_tab,
    prm_calibration,
    select_answer,
)
from .answers import extract_final_answer, extract_steps, final_answer_correct
from .backends import build_backend
from .prm import build_scorer

__all__ = ["EvalConfig", "RunLogger", "run_eval", "run_comparison",
           "load_problems", "suffixed_path", "report_targets", "is_full_run",
           "FULL_RUN", "PROMPT_TEMPLATE"]

FULL_RUN = 500  # the whole of MATH-500; anything less is a partial measurement


def suffixed_path(path: str, name: str) -> str:
    """``report.html`` + ``qwen`` -> ``report_qwen.html``."""
    root, ext = os.path.splitext(path)
    return f"{root}_{name}{ext or '.html'}"


def is_full_run(num_questions: int) -> bool:
    return num_questions >= FULL_RUN


def report_targets(report_path: str, backend: str, num_questions: int,
                   canonical: bool = True) -> list[str]:
    """Every report file a run of this size is entitled to write.

    ``report.html`` is the headline artifact -- the one that gets committed and
    linked -- so it stands for a complete 500-question run and nothing else. A
    20-question spot check that overwrote it would leave a 4% sample being read
    as *the* result, so short runs land on their own ``_partial`` path instead.

    Every full run also writes a per-backend copy, so finishing 500 questions
    on one model no longer destroys the report of the model before it.
    """
    if not is_full_run(num_questions):
        return [suffixed_path(report_path, f"{backend}_partial")]
    per_model = suffixed_path(report_path, backend)
    # A comparison run has no single backend that could claim report.html.
    return [report_path, per_model] if canonical else [per_model]


DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_RESULTS = "eval_results.json"
DEFAULT_LOG = "eval_debug.log"
DEFAULT_REPORT = "report.html"
LOG_DIR = "logs"
CHECKPOINT = ".eval_checkpoint.jsonl"

PROMPT_TEMPLATE = (
    "Solve this math problem concisely. Briefly show your reasoning, "
    "then give the answer inside \\boxed{{}}.\n\n{problem}"
)


@dataclass
class EvalConfig:
    num_questions: int = 3
    backend: str = "gemini"
    model: str | None = None
    question: int | None = None
    k: int = 1
    temperature: float | None = None
    max_tokens: int = 16384
    use_prm: bool = True
    resume: bool = True
    think: str = "merge"
    verbose: bool = True
    out_dir: str = "."
    results_file: str = DEFAULT_RESULTS
    log_file: str = DEFAULT_LOG
    report_file: str = DEFAULT_REPORT
    # A comparison sub-run is one model among several, so it never owns the
    # headline report even when it covers all 500 questions.
    canonical_report: bool = True

    def resolved_temperature(self) -> float:
        """Sampling k>1 solutions at temperature 0 yields k identical
        solutions, which makes self-consistency a no-op."""
        if self.temperature is not None:
            return self.temperature
        return 0.7 if self.k > 1 else 0.0

    def path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def fingerprint(self) -> str:
        """Identifies a checkpoint as belonging to this configuration.

        Deliberately excludes ``num_questions`` so that a run can be extended
        (``-n 50`` after ``-n 20``) while still reusing completed work.
        """
        payload = json.dumps({
            "backend": self.backend, "model": self.model, "k": self.k,
            "temperature": self.resolved_temperature(),
            "max_tokens": self.max_tokens, "use_prm": self.use_prm,
            "think": self.think if self.backend == "qwen" else None,
            "prompt": PROMPT_TEMPLATE,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RunLogger:
    """Verbose transcript to a file, terse progress to stdout."""

    def __init__(self, path: str, verbose: bool = True, echo: bool = True):
        self.path = path
        self.verbose = verbose
        self.echo = echo

    def __call__(self, msg: str = "") -> None:
        with open(self.path, "a") as f:
            f.write(msg + "\n")

    def detail(self, msg: str = "") -> None:
        if self.verbose:
            self(msg)

    def block(self, header: str, body: str) -> None:
        if not self.verbose:
            return
        self(header)
        for line in body.strip().split("\n"):
            self(f"    | {line}")

    def status(self, msg: str) -> None:
        if self.echo:
            print(msg, flush=True)


# ── Stage 1: load ─────────────────────────────────────────────────────
def load_problems(num_questions: int, question: int | None = None,
                  dataset: str = DATASET) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415 -- deliberately lazy

    ds = load_dataset(dataset, split="test")
    indices = [question - 1] if question is not None else range(min(num_questions, len(ds)))
    return [dict(ds[int(i)], index=int(i)) for i in indices]


# ── Stage 2+3: generate and score one sample ──────────────────────────
def solve_once(problem: str, backend, scorer, config: EvalConfig,
               log: RunLogger, context: dict, sample_index: int = 0) -> dict:
    """One sampled solution: generate, split into steps, PRM-score, grade."""
    started = time.monotonic()
    gen = backend.generate(
        PROMPT_TEMPLATE.format(problem=problem),
        max_tokens=config.max_tokens,
        temperature=config.resolved_temperature(),
        context={**context, "sample_index": sample_index},
    )
    elapsed = time.monotonic() - started
    if gen.warning:
        log(f"  [WARN] {gen.warning}")

    steps = extract_steps(gen.text)
    step_scores = scorer.step_scores(problem, steps)
    prm = sum(step_scores) / len(step_scores) if step_scores else None
    predicted, matched = extract_final_answer(gen.text)

    return {
        "predicted": predicted,
        "parse_matched": matched,
        "answer_accuracy": final_answer_correct(context.get("answer", ""), predicted),
        "prm_score": round(prm, 3) if prm is not None else None,
        "step_prm_scores": [round(s, 3) for s in step_scores],
        "steps": steps,
        "num_steps": len(steps),
        "prompt_tokens": gen.prompt_tokens,
        "eval_tokens": gen.output_tokens,
        "total_tokens": gen.total_tokens,
        "latency_s": round(elapsed, 2),
        "text": gen.text,
    }


def solve_question(row: dict, position: int, total: int, backend, scorer,
                   config: EvalConfig, log: RunLogger) -> dict:
    """Produce the full result record for one problem."""
    problem, expected = row["problem"], row["answer"]

    log(f"\n{'=' * 60}")
    log.block(f"[{position}/{total}] PROMPT (input):",
              PROMPT_TEMPLATE.format(problem=problem))
    log(f"  Expected answer: {expected}")
    log(f"  Subject: {row.get('subject')} | Difficulty: {row.get('level')}")
    log.status(f"[{position}/{total}] Solving..." + (f" (k={config.k})" if config.k > 1 else ""))

    samples = []
    for j in range(config.k):
        if config.k > 1:
            log(f"\n  --- sample {j + 1}/{config.k} ---")
        sample = solve_once(problem, backend, scorer, config, log, row, sample_index=j)
        log.detail(f"  Tokens: prompt={sample['prompt_tokens']} "
                   f"output={sample['eval_tokens']} total={sample['total_tokens']}")
        log.block(f"  Raw text ({sample['eval_tokens']} tokens):", sample["text"] or "(empty)")
        if sample["step_prm_scores"]:
            log.detail(f"  Intermediate steps ({sample['num_steps']}) with PRM scores:")
            for idx, step in enumerate(sample["steps"]):
                score = sample["step_prm_scores"][idx] if idx < len(sample["step_prm_scores"]) else float("nan")
                log.detail(f"    [{idx}] PRM={score:.3f} | {step}")
            log.detail(f"  Avg PRM score: {sample['prm_score']}")
        log.detail(f"  Final answer extracted: {sample['predicted']!r} "
                   f"(structured match: {sample['parse_matched']})")
        log.detail(f"  Answer correct: {sample['answer_accuracy']}")
        samples.append(sample)

    # The headline metric stays the single-shot baseline; the alternatives are
    # reported alongside it so the comparison is like-for-like.
    selection = {}
    for strategy in SELECTION_STRATEGIES:
        chosen = select_answer(samples, strategy)
        selection[strategy] = {
            "predicted": chosen["predicted"],
            "correct": chosen["answer_accuracy"],
            "prm_score": chosen["prm_score"],
        }
    if config.k > 1:
        log.detail(f"  Selection: " + ", ".join(
            f"{s}={'OK' if v['correct'] else 'X'}({v['predicted']!r})"
            for s, v in selection.items()))

    headline = samples[0]
    result = {
        "index": row.get("index", position - 1),
        "problem": problem,
        "expected": expected,
        "subject": row.get("subject"),
        "difficulty": row.get("level"),
        "k": config.k,
        "selection": selection,
        "samples": [{key: s[key] for key in s if key != "text"} for s in samples],
        **{key: headline[key] for key in (
            "predicted", "parse_matched", "answer_accuracy", "prm_score",
            "step_prm_scores", "steps", "num_steps",
            "prompt_tokens", "eval_tokens", "total_tokens", "latency_s",
        )},
    }
    log.status(f"  Tokens: {headline['eval_tokens']} | "
               f"Correct: {headline['answer_accuracy']} | "
               f"PRM: {headline['prm_score'] if headline['prm_score'] is not None else 'n/a'}")
    return result


# ── Checkpointing ─────────────────────────────────────────────────────
def load_checkpoint(path: str, fingerprint: str) -> dict[int, dict]:
    """Return completed rows keyed by dataset index, or {} on any mismatch.

    A long MATH-500 run is hours of GPU time; losing it to a crash at question
    400 is the failure this exists to prevent.
    """
    if not os.path.exists(path):
        return {}
    rows: dict[int, dict] = {}
    try:
        with open(path) as f:
            header = json.loads(f.readline() or "{}")
            if header.get("fingerprint") != fingerprint:
                return {}  # different config -> previous work is not reusable
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break  # truncated final write from a hard kill; stop here
                rows[row["index"]] = row
    except (OSError, json.JSONDecodeError):
        return {}
    return rows


def start_checkpoint(path: str, fingerprint: str, existing: bool) -> None:
    if existing and os.path.exists(path):
        return
    with open(path, "w") as f:
        f.write(json.dumps({"fingerprint": fingerprint,
                            "created": datetime.now().isoformat()}) + "\n")


def append_checkpoint(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())  # survive a hard kill, not just a clean exit


# ── Stage 4: aggregate ────────────────────────────────────────────────
def aggregate(results: list[dict], config: EvalConfig, model_name: str,
              duration: float | None = None, synthetic: bool = False) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("no results to aggregate")

    correct = sum(1 for r in results if r["answer_accuracy"])
    prm_scores = [r["prm_score"] for r in results if r.get("prm_score") is not None]
    parse_failures = sum(
        1 for r in results if not r.get("parse_matched") or not r.get("predicted")
    )

    selection_accuracy = {}
    if any(r.get("selection") for r in results):
        for strategy in SELECTION_STRATEGIES:
            picked = [r["selection"][strategy]["correct"]
                      for r in results if r.get("selection")]
            if picked:
                selection_accuracy[strategy] = sum(picked) / len(picked)

    return {
        "num_questions": total,
        "accuracy": correct / total,
        "correct": correct,
        "avg_prm_score": round(sum(prm_scores) / len(prm_scores), 3) if prm_scores else None,
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / total, 3),
        "by_subject": compute_group_accuracy(results, "subject"),
        "by_difficulty": compute_group_accuracy(results, "difficulty"),
        "subject_by_difficulty": cross_tab(results, "subject", "difficulty"),
        "prm_calibration": prm_calibration(results),
        "selection_accuracy": selection_accuracy,
        "k": config.k,
        "temperature": config.resolved_temperature(),
        "model_backend": config.backend,
        "model": model_name,
        "prm_enabled": config.use_prm,
        "synthetic": synthetic,
        # `or 0` rather than a .get default: a truncated response records the
        # key with an explicit None, which a default would not catch. Results
        # files and checkpoints written before this was fixed still contain
        # those nulls, so this stays defensive rather than trusting the input.
        "total_output_tokens": sum((r.get("eval_tokens") or 0) for r in results),
        "empty_responses": sum(1 for r in results if not (r.get("predicted") or "").strip()),
        "duration_s": round(duration, 1) if duration is not None else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }


def _archive_destination(log_dir: str, path: str, stamp: str) -> str:
    """``report.html`` -> ``logs/report_<stamp>.html``.

    Derived from the actual filename rather than a hardcoded prefix, so a
    comparison run archives ``eval_results_qwen.json`` as
    ``eval_results_qwen_<stamp>.json`` instead of flattening every backend
    onto the same name.
    """
    stem, ext = os.path.splitext(os.path.basename(path))
    target = os.path.join(log_dir, f"{stem}_{stamp}{ext}")
    if not os.path.exists(target):
        return target
    # Two runs inside the same second would otherwise silently clobber an
    # archive, and these are the only copies.
    for n in range(2, 1000):
        candidate = os.path.join(log_dir, f"{stem}_{stamp}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
    return target


def _archive_previous(config: EvalConfig, *paths: str) -> list[str]:
    """Move the previous run's artifacts aside rather than overwriting them.

    The log, results and report of a single run share one timestamp, so the
    three files belonging to a given run stay correlated in ``logs/``.
    """
    present = [p for p in paths if p and os.path.exists(p)]
    if not present:
        return []
    log_dir = config.path(LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = []
    for path in present:
        target = _archive_destination(log_dir, path, stamp)
        shutil.move(path, target)
        archived.append(target)
    return archived


def _merge_single_question(config: EvalConfig, results_path: str,
                           result: dict, model_name: str) -> dict:
    """Splice a single rerun question back into the existing results file."""
    with open(results_path) as f:
        summary = json.load(f)
    existing = summary["results"]
    pos = config.question - 1
    if pos < len(existing):
        existing[pos] = result
    else:
        existing.extend([None] * (pos - len(existing)))
        existing.append(result)
    merged = [r for r in existing if r is not None]
    summary.update(aggregate(merged, config, model_name))
    summary["results"] = existing
    return summary


def _print_summary(summary: dict, log: RunLogger, results_path: str, log_path: str) -> None:
    total = summary["num_questions"]
    acc = summary["accuracy"]
    lines = [
        f"SUMMARY  (model: {summary['model']})",
        f"  Accuracy:  {acc:.0%} ({summary['correct']}/{total})",
        f"  Avg PRM:   {summary['avg_prm_score'] if summary['avg_prm_score'] is not None else 'n/a (disabled)'}",
        f"  Parse failures: {summary['parse_failures']} ({summary['parse_failure_rate']:.0%})",
    ]
    cal = summary.get("prm_calibration") or {}
    if cal.get("auc") is not None:
        ci = cal.get("auc_ci")
        span = f" 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        lines.append(f"  PRM->correct AUC: {cal['auc']:.3f}{span}")
        if cal.get("auc_beats_chance") is False:
            lines.append(f"    not distinguishable from chance "
                         f"({cal.get('n_incorrect')} incorrect to rank against)")
        elif cal.get("auc_beats_chance") is None:
            lines.append(f"    unvalidated: only {cal.get('n_incorrect')} incorrect "
                         f"solution(s) to rank against")
    if summary.get("selection_accuracy") and summary.get("k", 1) > 1:
        lines.append("  Selection accuracy (k=%d):" % summary["k"])
        for strategy, value in summary["selection_accuracy"].items():
            lines.append(f"    {strategy:<13} {value:.0%}")
    if summary.get("synthetic"):
        lines.append("  NOTE: synthetic mock-backend data, not a real measurement.")

    log(f"\n{'=' * 60}")
    for line in lines:
        log(line)
    log(f"  By subject:    {summary['by_subject']}")
    log(f"  By difficulty: {summary['by_difficulty']}")

    print(f"\n{'=' * 60}")
    for line in lines:
        print(line)
    print(f"  Results:   {results_path}")
    print(f"  Debug log: {log_path}")


# ── Orchestration ─────────────────────────────────────────────────────
def run_eval(config: EvalConfig, problems: list[dict] | None = None,
             backend=None, scorer=None) -> dict:
    """Run the eval end to end and write ``eval_results.json``.

    ``problems``/``backend``/``scorer`` are injectable so the pipeline can be
    tested without the dataset, a network call, or a GPU.
    """
    started = time.monotonic()
    log_path = config.path(config.log_file)
    results_path = config.path(config.results_file)
    checkpoint_path = config.path(CHECKPOINT)
    single = config.question is not None

    os.makedirs(config.out_dir, exist_ok=True)
    if not single:
        # The reports this run is about to replace are archived alongside it,
        # and always -- even with --no-report -- so a stale report can never
        # sit next to the results of a newer run. Only those: a partial run
        # writes no report.html, so it must not archive one either, or a
        # 20-question spot check would quietly retire the 500-question report.
        reports = [config.path(p) for p in report_targets(
            config.report_file, config.backend, config.num_questions,
            canonical=config.canonical_report)]
        _archive_previous(config, log_path, results_path, *reports)
        open(log_path, "w").close()

    log = RunLogger(log_path, verbose=config.verbose)
    backend_kwargs = {"think": config.think} if config.backend == "qwen" else {}
    backend = backend or build_backend(config.backend, config.model, on_log=log,
                                        **backend_kwargs)
    scorer = scorer if scorer is not None else build_scorer(config.use_prm, on_log=log)

    log(f"Model backend: {config.backend} ({backend.model})")
    log(f"k={config.k} temperature={config.resolved_temperature()} "
        f"prm={'on' if config.use_prm else 'off'}")

    if problems is None:
        problems = load_problems(config.num_questions, config.question)

    done: dict[int, dict] = {}
    if config.resume and not single:
        done = load_checkpoint(checkpoint_path, config.fingerprint())
        if done:
            log.status(f"Resuming: {len(done)} question(s) already complete.")
            log(f"Resumed {len(done)} question(s) from checkpoint.")
        start_checkpoint(checkpoint_path, config.fingerprint(), existing=bool(done))

    results = []
    total = len(problems)
    for position, row in enumerate(problems, start=1):
        idx = row.get("index", position - 1)
        if idx in done:
            log.status(f"[{position}/{total}] cached")
            results.append(done[idx])
            continue
        result = solve_question(row, position, total, backend, scorer, config, log)
        results.append(result)
        if config.resume and not single:
            append_checkpoint(checkpoint_path, result)

    synthetic = getattr(backend, "synthetic", False)
    if single and os.path.exists(results_path):
        summary = _merge_single_question(config, results_path, results[0], backend.model)
    else:
        summary = aggregate(results, config, backend.model,
                            duration=time.monotonic() - started, synthetic=synthetic)

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, log, results_path, log_path)

    # The run completed; the checkpoint has served its purpose.
    if config.resume and not single and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return summary


def run_comparison(config: EvalConfig, backends: list[str],
                   problems: list[dict] | None = None) -> dict:
    """Run the same questions through several backends and diff the outcomes.

    The same problems are used for every backend, so divergences are
    attributable to the model rather than to sampling different questions.
    """
    from .analysis import compare_runs  # noqa: PLC0415 -- avoids a cycle at import

    if problems is None:
        problems = load_problems(config.num_questions, config.question)

    runs, summaries, results_files, report_files = {}, {}, {}, {}
    for name in backends:
        print(f"\n{'#' * 60}\n# backend: {name}\n{'#' * 60}", flush=True)
        results_files[name] = f"eval_results_{name}.json"
        report_files[name] = report_targets(config.report_file, name,
                                            config.num_questions, canonical=False)
        sub = EvalConfig(**{**asdict(config), "backend": name, "model": None,
                            "results_file": results_files[name],
                            "log_file": f"eval_debug_{name}.log",
                            "canonical_report": False})
        summary = run_eval(sub, problems=problems)
        runs[name] = summary["results"]
        summaries[name] = {k: v for k, v in summary.items() if k != "results"}

    comparison = compare_runs(runs)
    comparison["summaries"] = summaries
    # Each backend writes its own results file; a comparison run never writes
    # the default eval_results.json. Recording the paths keeps report building
    # from silently falling back to a stale file left by an earlier run.
    comparison["results_files"] = results_files
    # Each backend's reports, for the record: which ones exist depends on
    # whether the comparison covered the full 500 questions.
    comparison["report_files"] = report_files
    comparison["timestamp"] = datetime.now().isoformat(timespec="seconds")

    path = config.path("eval_comparison.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'=' * 60}\nCOMPARISON")
    for name, acc in comparison["accuracy"].items():
        print(f"  {name:<10} {acc:.0%}")
    for pair, agree in comparison["agreement"].items():
        print(f"  agreement {pair}: {agree:.0%}")
    print(f"  Divergent questions: {comparison['num_divergences']}")
    print(f"  Written to: {path}")
    return comparison
