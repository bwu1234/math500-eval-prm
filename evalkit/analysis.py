"""Aggregation and statistics over eval results.

Two questions this module exists to answer:

1.  *Where* does the model fail -- accuracy sliced by subject and difficulty.
2.  *Is the PRM signal worth anything?* The harness spends real GPU time
    scoring reasoning steps; that is only justified if the score actually
    separates correct from incorrect solutions. ``prm_calibration`` measures
    that directly (AUC, correlation, binned reliability curve) instead of
    assuming it.

Pure stdlib on purpose -- no numpy/scipy -- so it runs anywhere and every
statistic here is unit tested against hand-computed values.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .answers import are_equivalent

__all__ = [
    "compute_group_accuracy",
    "roc_auc",
    "pearson",
    "calibration_bins",
    "prm_calibration",
    "equivalence_groups",
    "select_answer",
    "SELECTION_STRATEGIES",
    "compare_runs",
]


def compute_group_accuracy(rows: list[dict], group_key: str,
                           correct_key: str = "answer_accuracy") -> dict:
    """Accuracy broken down by a metadata field (subject, difficulty, ...)."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        value = row.get(group_key)
        if value is None:
            continue
        stats[str(value)]["total"] += 1
        stats[str(value)]["correct"] += int(bool(row.get(correct_key)))
    return {
        k: {
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for k, v in sorted(stats.items(), key=lambda x: x[0])
    }


def cross_tab(rows: list[dict], row_key: str, col_key: str,
              correct_key: str = "answer_accuracy") -> dict:
    """Two-way accuracy table, e.g. subject x difficulty."""
    cells = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        r, c = row.get(row_key), row.get(col_key)
        if r is None or c is None:
            continue
        cell = cells[(str(r), str(c))]
        cell["total"] += 1
        cell["correct"] += int(bool(row.get(correct_key)))
    return {
        "rows": sorted({r for r, _ in cells}),
        "cols": sorted({c for _, c in cells}),
        "cells": {f"{r}|{c}": v for (r, c), v in cells.items()},
    }


# ── Statistics ────────────────────────────────────────────────────────
def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    Interpretation: the probability that a randomly chosen correct solution is
    scored above a randomly chosen incorrect one. 0.5 means the score carries
    no information. Ties get average ranks, so a constant score scores exactly
    0.5 rather than accidentally looking predictive.
    """
    n = len(scores)
    if n != len(labels) or n == 0:
        return None
    pos = sum(1 for l in labels if l)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None  # undefined with only one class present

    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. With a boolean ``ys`` this is the point-biserial
    correlation between PRM score and correctness."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None  # no variance -> correlation undefined, not zero
    return num / (dx * dy)


def calibration_bins(scores: list[float], labels: list[bool],
                     n_bins: int = 10) -> list[dict]:
    """Bin scores into equal-width buckets and report empirical accuracy.

    This is the reliability curve: if the PRM is informative, accuracy should
    rise monotonically across bins.
    """
    bins = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins,
         "count": 0, "correct": 0, "score_sum": 0.0}
        for i in range(n_bins)
    ]
    for score, label in zip(scores, labels):
        clamped = min(max(score, 0.0), 1.0)
        idx = min(int(clamped * n_bins), n_bins - 1)
        bins[idx]["count"] += 1
        bins[idx]["correct"] += int(bool(label))
        bins[idx]["score_sum"] += clamped
    for b in bins:
        b["accuracy"] = b["correct"] / b["count"] if b["count"] else None
        b["mean_score"] = b["score_sum"] / b["count"] if b["count"] else None
        del b["score_sum"]
    return bins


def prm_calibration(rows: list[dict], score_key: str = "prm_score",
                    correct_key: str = "answer_accuracy",
                    n_bins: int = 10) -> dict:
    """Does the PRM score predict final-answer correctness?

    Returns AUC, point-biserial correlation, the reliability curve, and the
    mean score within each outcome class. ``separation`` (mean score on
    correct minus mean on incorrect) is the most directly readable of these:
    positive means the PRM ranks correct solutions higher.
    """
    scored = [r for r in rows if r.get(score_key) is not None]
    scores = [float(r[score_key]) for r in scored]
    labels = [bool(r.get(correct_key)) for r in scored]

    correct_scores = [s for s, l in zip(scores, labels) if l]
    wrong_scores = [s for s, l in zip(scores, labels) if not l]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731

    mean_correct, mean_wrong = mean(correct_scores), mean(wrong_scores)
    bins = calibration_bins(scores, labels, n_bins)
    total = len(scores)
    ece = sum(
        b["count"] / total * abs(b["accuracy"] - b["mean_score"])
        for b in bins if b["count"]
    ) if total else None

    return {
        "n": total,
        "auc": roc_auc(scores, labels),
        "point_biserial_r": pearson(scores, [float(l) for l in labels]),
        "mean_prm_correct": mean_correct,
        "mean_prm_incorrect": mean_wrong,
        "separation": (mean_correct - mean_wrong)
                      if (mean_correct is not None and mean_wrong is not None) else None,
        "expected_calibration_error": ece,
        "bins": bins,
    }


# ── Answer selection / self-consistency ───────────────────────────────
def equivalence_groups(answers: list[str]) -> list[dict]:
    """Cluster answers into mathematical equivalence classes.

    Voting on raw strings splits ``1/2`` from ``0.5`` from ``\\frac{1}{2}`` and
    hands the election to a formatting accident, so cluster with the same
    equivalence relation used for grading.
    """
    groups: list[dict] = []
    for idx, ans in enumerate(answers):
        for g in groups:
            if are_equivalent(ans, g["answer"]):
                g["members"].append(idx)
                break
        else:
            groups.append({"answer": ans, "members": [idx]})
    return groups


def select_answer(samples: list[dict], strategy: str) -> dict:
    """Pick one sample from k sampled solutions.

    Strategies:
      ``first``        -- the first sample (the single-shot baseline)
      ``majority``     -- self-consistency: largest equivalence class
      ``prm_best``     -- highest mean PRM score (best-of-n reranking)
      ``prm_weighted`` -- vote, weighting each sample by its PRM score
    """
    if not samples:
        raise ValueError("no samples to select from")
    if strategy == "first":
        return samples[0]
    if strategy == "prm_best":
        return max(samples, key=lambda s: s.get("prm_score") or 0.0)

    if strategy not in ("majority", "prm_weighted"):
        raise ValueError(f"unknown selection strategy: {strategy!r}")

    groups = equivalence_groups([s.get("predicted", "") for s in samples])
    if strategy == "majority":
        weight = lambda g: (len(g["members"]), -min(g["members"]))  # noqa: E731
    else:
        weight = lambda g: (  # noqa: E731
            sum(samples[i].get("prm_score") or 0.0 for i in g["members"]),
            -min(g["members"]),
        )
    best = max(groups, key=weight)
    # Return the highest-PRM representative of the winning class so the
    # reported reasoning trace matches the reported answer.
    return max(
        (samples[i] for i in best["members"]),
        key=lambda s: s.get("prm_score") or 0.0,
    )


SELECTION_STRATEGIES = ("first", "majority", "prm_best", "prm_weighted")


# ── Backend comparison ────────────────────────────────────────────────
def compare_runs(runs: dict[str, list[dict]]) -> dict:
    """Compare per-question results across backends.

    ``runs`` maps a backend label to its result rows, aligned by position.
    Divergences -- questions where the backends disagree about correctness --
    are where the interesting analysis lives.
    """
    labels = list(runs)
    if not labels:
        return {"backends": [], "accuracy": {}, "agreement": {}, "divergences": []}

    n = min(len(rows) for rows in runs.values())
    accuracy = {
        label: (sum(1 for r in rows[:n] if r.get("answer_accuracy")) / n if n else 0.0)
        for label, rows in runs.items()
    }

    agreement = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            same = sum(
                1 for k in range(n)
                if bool(runs[a][k].get("answer_accuracy")) == bool(runs[b][k].get("answer_accuracy"))
            )
            agreement[f"{a}|{b}"] = same / n if n else 0.0

    divergences = []
    for k in range(n):
        outcomes = {label: bool(runs[label][k].get("answer_accuracy")) for label in labels}
        if len(set(outcomes.values())) > 1:
            first = runs[labels[0]][k]
            divergences.append({
                "index": k,
                "problem": first.get("problem", ""),
                "expected": first.get("expected", ""),
                "subject": first.get("subject"),
                "difficulty": first.get("difficulty"),
                "outcomes": outcomes,
                "predicted": {label: runs[label][k].get("predicted", "") for label in labels},
                "prm_score": {label: runs[label][k].get("prm_score") for label in labels},
            })

    return {
        "backends": labels,
        "num_questions": n,
        "accuracy": accuracy,
        "agreement": agreement,
        "num_divergences": len(divergences),
        "divergences": divergences,
    }
