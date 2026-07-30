"""Tests for the statistics and selection logic.

Every statistic is checked against a value computed by hand, because these are
the numbers the project's central claim -- "the PRM score does/doesn't predict
correctness" -- is built on. A subtly wrong AUC would be invisible otherwise.
"""

import math

import pytest

from evalkit.analysis import (
    calibration_bins,
    compare_runs,
    compute_group_accuracy,
    cross_tab,
    equivalence_groups,
    pearson,
    prm_calibration,
    roc_auc,
    select_answer,
)


# ── AUC ───────────────────────────────────────────────────────────────
def test_auc_perfect_separation():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0


def test_auc_perfect_inversion():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0


def test_auc_all_ties_is_chance():
    """A constant score must score exactly 0.5, not accidentally look good."""
    assert roc_auc([0.5] * 6, [True, False, True, False, True, False]) == 0.5


def test_auc_hand_computed():
    # scores 1,2,3,4 with labels F,T,F,T.
    # Positive ranks are 2 and 4 -> U = (2+4) - 2*3/2 = 3; 3/(2*2) = 0.75
    assert roc_auc([1, 2, 3, 4], [False, True, False, True]) == 0.75


def test_auc_undefined_with_one_class():
    assert roc_auc([0.1, 0.2], [True, True]) is None
    assert roc_auc([0.1, 0.2], [False, False]) is None
    assert roc_auc([], []) is None


def test_auc_partial_ties():
    # scores 1,1,2 labels F,T,T -> tied ranks 1.5,1.5 then 3.
    # U = (1.5+3) - 2*3/2 = 1.5 ; 1.5/(2*1) = 0.75
    assert roc_auc([1, 1, 2], [False, True, True]) == 0.75


# ── Pearson ───────────────────────────────────────────────────────────
def test_pearson_perfect_positive():
    assert math.isclose(pearson([1, 2, 3], [2, 4, 6]), 1.0)


def test_pearson_perfect_negative():
    assert math.isclose(pearson([1, 2, 3], [6, 4, 2]), -1.0)


def test_pearson_zero_variance_is_none():
    """Undefined, not zero -- reporting 0.0 would imply 'no relationship'."""
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_pearson_needs_two_points():
    assert pearson([1], [1]) is None


def test_pearson_hand_computed():
    assert math.isclose(pearson([1, 2, 3, 4], [1, 3, 2, 4]), 0.8)


# ── Calibration ───────────────────────────────────────────────────────
def test_calibration_bins_partition_all_points():
    scores = [0.05, 0.15, 0.95, 1.0, 0.0]
    bins = calibration_bins(scores, [True] * 5, n_bins=10)
    assert sum(b["count"] for b in bins) == len(scores)


def test_calibration_score_of_one_lands_in_last_bin():
    bins = calibration_bins([1.0], [True], n_bins=10)
    assert bins[-1]["count"] == 1


def test_calibration_empty_bins_report_none():
    bins = calibration_bins([0.05], [True], n_bins=10)
    assert bins[0]["accuracy"] == 1.0
    assert bins[5]["accuracy"] is None


def test_calibration_clamps_out_of_range_scores():
    bins = calibration_bins([-0.5, 1.7], [True, True], n_bins=4)
    assert bins[0]["count"] == 1 and bins[-1]["count"] == 1


def test_prm_calibration_separation():
    rows = [
        {"prm_score": 0.9, "answer_accuracy": True},
        {"prm_score": 0.8, "answer_accuracy": True},
        {"prm_score": 0.2, "answer_accuracy": False},
        {"prm_score": 0.1, "answer_accuracy": False},
    ]
    cal = prm_calibration(rows)
    assert cal["auc"] == 1.0
    assert math.isclose(cal["separation"], 0.7)
    assert cal["n"] == 4


def test_prm_calibration_ignores_missing_scores():
    rows = [
        {"prm_score": None, "answer_accuracy": True},
        {"prm_score": 0.5, "answer_accuracy": True},
    ]
    assert prm_calibration(rows)["n"] == 1


def test_prm_calibration_handles_empty():
    cal = prm_calibration([])
    assert cal["n"] == 0 and cal["auc"] is None


# ── Grouping ──────────────────────────────────────────────────────────
def test_group_accuracy():
    rows = [
        {"subject": "Algebra", "answer_accuracy": True},
        {"subject": "Algebra", "answer_accuracy": False},
        {"subject": "Geometry", "answer_accuracy": True},
    ]
    stats = compute_group_accuracy(rows, "subject")
    assert stats["Algebra"] == {"accuracy": 0.5, "correct": 1, "total": 2}
    assert stats["Geometry"]["accuracy"] == 1.0


def test_group_accuracy_skips_missing_key():
    rows = [{"subject": None, "answer_accuracy": True},
            {"subject": "Algebra", "answer_accuracy": True}]
    assert list(compute_group_accuracy(rows, "subject")) == ["Algebra"]


def test_cross_tab():
    rows = [
        {"subject": "Algebra", "difficulty": 1, "answer_accuracy": True},
        {"subject": "Algebra", "difficulty": 1, "answer_accuracy": False},
        {"subject": "Algebra", "difficulty": 5, "answer_accuracy": False},
    ]
    tab = cross_tab(rows, "subject", "difficulty")
    assert tab["rows"] == ["Algebra"]
    assert tab["cells"]["Algebra|1"] == {"correct": 1, "total": 2}


# ── Self-consistency / selection ──────────────────────────────────────
def test_equivalence_groups_merge_formatting_variants():
    """1/2, 0.5 and \\frac{1}{2} are one vote, not three."""
    groups = equivalence_groups(["1/2", "0.5", r"\frac{1}{2}", "3"])
    assert len(groups) == 2
    assert sorted(len(g["members"]) for g in groups) == [1, 3]


def test_majority_vote_beats_first_sample():
    samples = [
        {"predicted": "7", "prm_score": 0.9},
        {"predicted": "4", "prm_score": 0.3},
        {"predicted": r"\frac{8}{2}", "prm_score": 0.2},
    ]
    assert select_answer(samples, "first")["predicted"] == "7"
    # 4 and 8/2 are the same answer, so they outvote the lone 7.
    assert select_answer(samples, "majority")["predicted"] in ("4", r"\frac{8}{2}")


def test_prm_best_picks_highest_scoring_sample():
    samples = [{"predicted": "1", "prm_score": 0.2}, {"predicted": "2", "prm_score": 0.8}]
    assert select_answer(samples, "prm_best")["predicted"] == "2"


def test_prm_weighted_can_outvote_a_bare_majority():
    samples = [
        {"predicted": "1", "prm_score": 0.1},
        {"predicted": "1", "prm_score": 0.1},
        {"predicted": "2", "prm_score": 0.95},
    ]
    assert select_answer(samples, "majority")["predicted"] == "1"
    assert select_answer(samples, "prm_weighted")["predicted"] == "2"


def test_selection_ties_break_deterministically():
    samples = [{"predicted": "1", "prm_score": 0.5}, {"predicted": "2", "prm_score": 0.5}]
    assert select_answer(samples, "majority")["predicted"] == "1"


def test_selection_handles_missing_prm_scores():
    samples = [{"predicted": "1"}, {"predicted": "2"}]
    assert select_answer(samples, "prm_best")["predicted"] == "1"


def test_selection_rejects_bad_input():
    with pytest.raises(ValueError):
        select_answer([], "first")
    with pytest.raises(ValueError):
        select_answer([{"predicted": "1"}], "nonsense")


# ── Backend comparison ────────────────────────────────────────────────
def test_compare_runs_finds_divergences():
    runs = {
        "gemini": [{"answer_accuracy": True, "predicted": "1"},
                   {"answer_accuracy": False, "predicted": "9"}],
        "qwen": [{"answer_accuracy": True, "predicted": "1"},
                 {"answer_accuracy": True, "predicted": "2"}],
    }
    cmp = compare_runs(runs)
    assert cmp["accuracy"] == {"gemini": 0.5, "qwen": 1.0}
    assert cmp["agreement"]["gemini|qwen"] == 0.5
    assert cmp["num_divergences"] == 1
    assert cmp["divergences"][0]["index"] == 1


def test_compare_runs_truncates_to_shortest():
    runs = {"a": [{"answer_accuracy": True}] * 3, "b": [{"answer_accuracy": True}]}
    assert compare_runs(runs)["num_questions"] == 1


def test_compare_runs_empty():
    assert compare_runs({})["backends"] == []
