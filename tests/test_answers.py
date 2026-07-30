"""Tests for the grading logic.

The cases below are the ones that actually bite in practice: LaTeX that looks
different but means the same thing (must match), and LaTeX that looks similar
but means different things (must NOT match). The second group matters more --
a false positive silently inflates reported accuracy.
"""

import pytest

from evalkit.answers import (
    are_equivalent,
    extract_final_answer,
    extract_steps,
    final_answer_correct,
    find_boxed,
    latex_to_ascii,
    normalize_answer,
)


# ── Normalisation must preserve structure ─────────────────────────────
@pytest.mark.parametrize("a,b", [
    # Regression: a strip-everything normaliser mapped all of these to digits,
    # making them compare equal. 3\sqrt{13} -> "313" and \frac{1}{3} -> "13".
    (r"3\sqrt{13}", "313"),
    (r"\frac{1}{3}", "13"),
    (r"\sqrt{2}", "2"),
    (r"\frac{1}{2}", r"\frac{2}{1}"),
    (r"2^{10}", "210"),
    (r"\frac{1}{3x}", r"\frac{1}{3}x"),
])
def test_distinct_answers_do_not_collide(a, b):
    assert normalize_answer(a) != normalize_answer(b)
    assert not final_answer_correct(b, a)


# Real pairs of *distinct* gold answers from the MATH-500 test split that the
# original strip-everything normaliser mapped to the same string. Sweeping the
# full split found 20 such groups; grading could not tell these apart.
MATH500_COLLISION_PAIRS = [
    (r"\frac{14}{3}", "143"),
    (r"\sqrt{5}", "5"),
    (r"\frac{11}{2}", r"11\sqrt2"),
    (r"\frac{7}{2}", "72"),
    (r"\frac{2}{3}", "23"),
    (r"\frac{1}{2}", "12"),
    (r"\frac{3}{2}", "32"),
    (r"\frac{\sqrt{3}}{3}", "33"),
    (r"1\frac{4}{5}", r"145^\circ"),
    (r"\frac{1}{3}", "13"),
    (r"-\sqrt{3}", "-3"),
    (r"\frac{4}{9}", "49"),
]


@pytest.mark.parametrize("a,b", MATH500_COLLISION_PAIRS)
def test_real_math500_gold_answers_stay_distinct(a, b):
    """Regression: these are answers to different questions in the actual
    benchmark. Grading one as the other silently inflates accuracy."""
    assert normalize_answer(a) != normalize_answer(b)
    assert not final_answer_correct(b, a)
    assert not final_answer_correct(a, b)


@pytest.mark.parametrize("raw,expected", [
    (r"\frac{1}{3}", "1/3"),
    (r"\dfrac{1}{3}", "1/3"),
    (r"\tfrac{1}{3}", "1/3"),
    (r"3\sqrt{13}", "3*sqrt(13)"),
    (r"\sqrt[3]{8}", "8**(1/3)"),
    (r"\boxed{42}", "42"),
    (r"\text{cm}", "cm"),
    (r"90^\circ", "90"),
    (r"1{,}000", "1000"),
    (r"1,000", "1000"),
    (r"\pi", "pi"),
    (r"2\cdot 3", "2*3"),
    (r"\left(1,2\right)", "(1,2)"),
    (r"-\frac{1}{2}", "-1/2"),
])
def test_normalisation_targets(raw, expected):
    assert normalize_answer(raw) == expected


def test_nested_fractions_and_roots():
    # A non-recursive \frac regex fails outright on nested braces.
    assert normalize_answer(r"\frac{\sqrt{3}}{2}") == "(sqrt(3))/2"
    assert normalize_answer(r"\frac{\frac{1}{2}}{3}") == "(1/2)/3"


def test_implicit_multiplication_is_made_explicit():
    """LaTeX juxtaposition has to become `*` or sympy cannot parse it."""
    assert normalize_answer(r"3\sqrt{13}") == "3*sqrt(13)"
    assert normalize_answer("2(x+1)") == "2*(x+1)"
    assert normalize_answer("2e5") == "2e5"  # scientific notation left alone


def test_latex_to_ascii_is_sympy_parsable():
    import sympy as sp
    for raw in [r"\frac{\sqrt{3}}{2}", r"3\sqrt{13}", r"2^{10}", r"\frac{1}{2}+\frac{1}{3}"]:
        assert sp.sympify(latex_to_ascii(raw)) is not None


# ── Equivalence must catch genuinely equal answers ────────────────────
@pytest.mark.parametrize("pred,gold", [
    ("0.5", "1/2"),
    (r"\frac{1}{2}", "0.5"),
    (r"\frac{1}{2}", "1/2"),
    (r"\dfrac{2}{4}", "1/2"),
    ("2", "2.0"),
    (r"\frac{6}{3}", "2"),
    ("1000", "1,000"),
    (r"90^\circ", "90"),
    (r"\boxed{7}", "7"),
    ("x+1", "1+x"),
    (r"\sqrt{4}", "2"),
    (r"3\sqrt{13}", r"\sqrt{117}"),
    ("-0.25", r"-\frac{1}{4}"),
    (r"\left(\frac{1}{2}, 3\right)", "(0.5, 3)"),
    (r"\text{even}", "even"),
])
def test_equivalent_answers(pred, gold):
    assert final_answer_correct(gold, pred), f"{pred!r} should grade equal to {gold!r}"


@pytest.mark.parametrize("pred,gold", [
    ("3", "4"),
    ("1/2", "1/3"),
    ("x+1", "x+2"),
    ("(1,2)", "(2,1)"),
    ("", "5"),
    ("even", "odd"),
    (r"\sqrt{2}", r"\sqrt{3}"),
])
def test_inequivalent_answers(pred, gold):
    assert not final_answer_correct(gold, pred), f"{pred!r} must not grade equal to {gold!r}"


# ── Presentation tolerance ────────────────────────────────────────────
# Every pair below is a real MATH-500 grading disagreement: the model's answer
# was right and was scored wrong purely over how it was written. Sweeping the
# 500-question run, 20 of 25 recorded failures were of this kind.
@pytest.mark.parametrize("pred,gold", [
    # a unit or label on one side only, in either direction
    (r"12\text{ cm}", "12"),
    (r"10\%", "10"),
    (r"12^{\mathrm{th}} \text{ grade}", "12"),
    ("270/7", r"\frac{270}7\text{ degrees}"),
    ("864", r"864 \mbox{ inches}^2"),
    ("5.4", r"5.4 \text{ cents}"),
    # currency and digit grouping in the gold
    ("36", r"\$36"),
    ("18.90", r"\$18.90"),
    ("32,348", r"\$32,\!348"),
    ("10080", r"10,\!080"),
    ("11111111100", r"11,\! 111,\! 111,\! 100"),
    # the gold restates the variable
    ("5", "x=5"),
    ("[-2, 7]", r"x \in [-2,7]"),
    # numeral base spelled out on one side only
    ("2516", "2516_8"),
    ("4210_5", "4210_{5}"),
    # +- against the roots written out, in any order
    (r"1+\sqrt{19}, 1-\sqrt{19}", r"1 \pm \sqrt{19}"),
    (r"\{-2, 1+\sqrt{5}, 1-\sqrt{5}\}", r"\{1\pm\sqrt{5},-2\}"),
    ("1, -2", "-2,1"),
    # an interval asked for, an inequality given
    (r"3 < \lambda \le 4", "(3,4]"),
    (r"x \ge 5", r"[5,\infty)"),
    # decimals inside a vector
    (r"\begin{pmatrix} 0.2 \\ -3.6 \end{pmatrix}",
     r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}"),
    # spacing inside a union of intervals whose endpoints look like grouped digits
    (r"(2, 12) \cup (12, 102)", r"(2,12) \cup (12,102)"),
])
def test_presentation_differences_are_forgiven(pred, gold):
    assert final_answer_correct(gold, pred), f"{pred!r} should grade equal to {gold!r}"


@pytest.mark.parametrize("pred,gold", [
    # Tolerance is one-sided: when both answers carry the decoration and it
    # disagrees, they are different answers.
    (r"12\text{ m}", r"12\text{ cm}"),
    (r"5.4\text{ dollars}", r"5.4\text{ cents}"),
    ("2516_9", "2516_8"),
    # An interval's brackets carry meaning, and a set is not a tuple.
    ("(3,4)", "(3,4]"),
    ("[3,4]", "(3,4]"),
    (r"\{1,2\}", "(1,2)"),
    (r"x < 4", "(-oo,4]"),
    # Extra roots are a wrong answer, not a formatting difference.
    (r"3+2\sqrt{2}, 3-2\sqrt{2}, -3+2\sqrt{2}, -3-2\sqrt{2}", r"3 \pm 2 \sqrt{2}"),
    (r"1 \pm \sqrt{19}", r"1+\sqrt{19}"),
    # A percentage is not the same number as its decimal.
    (r"10\%", "0.1"),
])
def test_presentation_tolerance_is_not_a_free_pass(pred, gold):
    assert not final_answer_correct(gold, pred), f"{pred!r} must not grade equal to {gold!r}"


@pytest.mark.parametrize("raw,expected", [
    (r"\$36", "36"),
    (r"10,\!080", "10080"),
    (r"4210_{5}", "4210_5"),
    (r"\{1,2\}", "{1,2}"),                      # set braces survive as braces
    (r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}", "(1/5,-18/5)"),
    (r"1\pm\sqrt{5}", "1+-sqrt(5)"),            # regression: became "1*5"
    # Digit grouping resolves against the LaTeX brackets: braces group, so this
    # comma is a thousands separator...
    (r"\frac{20,000}{\pi}", "20000/(pi)"),
    # ...while parentheses delimit, so these commas are separators.
    ("(1, 234)", "(1,234)"),
    (r"(2,12) \cup (12,102)", "(2,12)*u(12,102)"),
])
def test_normalisation_of_previously_mangled_forms(raw, expected):
    assert normalize_answer(raw) == expected


def test_equivalence_is_symmetric():
    for a, b in [("0.5", "1/2"), (r"\frac{3}{6}", "0.5"), ("2", "2.00")]:
        assert are_equivalent(a, b) == are_equivalent(b, a)


# ── Boxed extraction with balanced braces ─────────────────────────────
def test_boxed_does_not_over_capture():
    # A greedy `\\boxed\{(.*)\}` with DOTALL swallows everything up to the
    # final closing brace anywhere in the response.
    text = r"First \boxed{5} and then some \text{trailing} prose."
    assert find_boxed(text) == ["5"]
    assert extract_final_answer(text)[0] == "5"


def test_boxed_with_nested_braces():
    assert find_boxed(r"\boxed{\frac{1}{2}}") == [r"\frac{1}{2}"]
    assert find_boxed(r"\boxed{\text{a} \frac{x}{y}}") == [r"\text{a} \frac{x}{y}"]


def test_boxed_with_escaped_set_braces():
    assert find_boxed(r"\boxed{\{1,2\}}") == [r"\{1,2\}"]


def test_last_box_wins():
    text = r"Intermediate \boxed{3}. Correcting: \boxed{4}."
    assert extract_final_answer(text) == ("4", True)


def test_unbalanced_box_degrades_gracefully():
    ans, matched = extract_final_answer(r"the answer is \boxed{5")
    assert isinstance(ans, str)  # must not raise


# ── Final-answer extraction fallbacks ─────────────────────────────────
@pytest.mark.parametrize("text,expected,matched", [
    (r"Therefore \boxed{42}", "42", True),
    ("The final answer is 42", "42", True),
    ("The answer is: 42", "42", True),
    (r"So we get \[ x = 42 \]", "x = 42", True),
    (r"So we get \(42\)", "42", True),
    ("no structure here\njust 42", "just 42", False),
])
def test_extraction(text, expected, matched):
    assert extract_final_answer(text) == (expected, matched)


def test_extraction_prefers_last_answer_phrase():
    text = "The answer is 3\nActually, the final answer is 4"
    assert extract_final_answer(text)[0] == "4"


def test_empty_input():
    assert extract_final_answer("") == ("", False)
    assert extract_final_answer("   \n ") == ("", False)


def test_parse_failure_is_reported():
    """Unstructured output must be flagged, not silently graded."""
    _, matched = extract_final_answer("I am not sure about this one")
    assert matched is False


# ── Step splitting ────────────────────────────────────────────────────
def test_extract_steps_attaches_display_math_to_its_prose():
    text = "First we substitute:\n\\[\nx = 1\n\\]\nThen simplify."
    assert extract_steps(text) == ["First we substitute: x = 1", "Then simplify."]


def test_extract_steps_handles_single_line_display_math():
    text = "Use the formula:\n$$d = \\sqrt{a^2+b^2}$$\nSo d = 5."
    steps = extract_steps(text)
    assert steps == ["Use the formula: d = \\sqrt{a^2+b^2}", "So d = 5."]


def test_extract_steps_never_emits_bare_delimiters():
    """A step of "x = 1 \\]" is out of distribution for the PRM."""
    text = "Lead in:\n\\[\na = 1\n\\]\nMore:\n\\[\nb = 2\n\\]"
    for step in extract_steps(text):
        assert "\\[" not in step and "\\]" not in step


def test_extract_steps_drops_boxed_and_short_lines():
    text = "ok\nA real reasoning step here.\n\\boxed{5}"
    assert extract_steps(text) == ["A real reasoning step here."]


def test_extract_steps_empty():
    assert extract_steps("") == []


# ── Robustness: the grader must never raise on model output ───────────
@pytest.mark.parametrize("junk", [
    "", "   ", "\\", "{{{{", "}}}}", r"\frac{", r"\sqrt[", "$" * 50,
    r"\boxed{" * 20, "a" * 5000, "\x00\x01",
])
def test_grader_never_raises(junk):
    extract_final_answer(junk)
    normalize_answer(junk)
    final_answer_correct("5", junk)
    extract_steps(junk)
