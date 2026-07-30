"""Answer extraction, normalisation and equivalence checking for MATH-500.

This module is deliberately dependency-light (sympy only) so that the grading
logic -- the part of an eval harness most likely to silently break -- can be
unit tested without a GPU, an API key, or the transformers stack.

Grading runs in two tiers:

1.  ``normalize_answer`` produces a canonical string for a cheap exact match.
    It is *structure preserving*: unlike a naive "delete every backslash
    command and every brace" approach, ``\\frac{1}{3}`` and ``3\\sqrt{13}`` do
    not both collapse into a bare digit string. Structural collapse creates
    false positives, which inflate reported accuracy -- the worst failure mode
    an eval can have.
2.  ``are_equivalent`` falls back to numeric/fraction/symbolic comparison for
    genuinely equal answers written differently (``0.5`` vs ``1/2`` vs
    ``\\frac{1}{2}``).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy as sp

__all__ = [
    "normalize_answer",
    "are_equivalent",
    "final_answer_correct",
    "extract_final_answer",
    "extract_steps",
    "find_boxed",
    "latex_to_ascii",
]

# ── LaTeX command tables ──────────────────────────────────────────────
# Commands are rewritten in a single pass over ``\\[a-zA-Z]+``. Anything not
# listed here is dropped, which is safe only because the structural commands
# (\frac, \sqrt, \text) are expanded *before* that pass runs.

LATEX_SYMBOLS = {
    r"\pi": "pi", r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\theta": "theta", r"\phi": "phi", r"\varphi": "phi",
    r"\omega": "omega", r"\sigma": "sigma", r"\mu": "mu", r"\tau": "tau",
    r"\epsilon": "epsilon", r"\varepsilon": "epsilon", r"\lambda": "lambda",
    r"\rho": "rho", r"\eta": "eta", r"\zeta": "zeta", r"\kappa": "kappa",
    r"\nu": "nu", r"\xi": "xi", r"\psi": "psi", r"\chi": "chi", r"\iota": "iota",
    r"\Gamma": "Gamma", r"\Delta": "Delta", r"\Theta": "Theta",
    r"\Lambda": "Lambda", r"\Pi": "Pi", r"\Sigma": "Sigma", r"\Phi": "Phi",
    r"\Psi": "Psi", r"\Omega": "Omega",
    r"\infty": "oo", r"\partial": "d",
}

LATEX_OPERATORS = {
    r"\cdot": "*", r"\times": "*", r"\ast": "*",
    r"\div": "/", r"\pm": "+-", r"\mp": "-+",
    r"\rightarrow": "->", r"\Rightarrow": "->", r"\to": "->",
    r"\approx": "~=", r"\neq": "!=", r"\ne": "!=",
    r"\leq": "<=", r"\le": "<=", r"\geq": ">=", r"\ge": ">=",
    r"\%": "%", r"\$": "", r"\&": "&",
    r"\cup": "U", r"\cap": "n", r"\in": " in ",
}

# Purely cosmetic commands -- dropped, but listed explicitly so the intent is
# visible rather than relying on the catch-all.
LATEX_NOOPS = frozenset({
    r"\left", r"\right", r"\displaystyle", r"\textstyle", r"\limits",
    r"\quad", r"\qquad", r"\;", r"\:", r"\,", r"\!", r"\ ",
    r"\big", r"\Big", r"\bigg", r"\Bigg", r"\mathrm", r"\mathbf", r"\bf",
})

# Functions that survive the rewrite pass under their own name so that sympy
# can still parse them (\sin{x} -> sin(x)). \ln is spelled log in sympy.
LATEX_FUNCTIONS = {
    f"\\{name}": name for name in (
        "sin cos tan sec csc cot arcsin arccos arctan sinh cosh tanh "
        "log exp min max gcd lcm det deg"
    ).split()
}
LATEX_FUNCTIONS[r"\ln"] = "log"

LATEX_COMMANDS = {**LATEX_SYMBOLS, **LATEX_OPERATORS, **LATEX_FUNCTIONS}

# Implicit multiplication: LaTeX writes "3\sqrt{13}", sympy needs "3*sqrt(13)".
# The scientific-notation lookahead keeps "2e5" from becoming "2*e5".
_IMPLICIT_MULT_RES = (
    re.compile(r"(?<=\d)(?![eE][-+]?\d)(?=[A-Za-z])"),
    re.compile(r"(?<=\d)(?=\()"),
    re.compile(r"(?<=\))(?=[A-Za-z0-9(])"),
)

_DEGREE_RE = re.compile(r"\^?\s*(?:\\circ|\\degree|\u00b0)")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
# Unwrap parentheses only around a bare number or single letter, and only when
# not adjacent to a word character. Anything looser risks re-introducing the
# collisions this module exists to prevent: "1/(3)x" must not become "1/3x",
# which is also what "1/(3x)" would reduce to.
_REDUNDANT_PARENS_RE = re.compile(r"(?<!\w)\((\d+(?:\.\d+)?|[A-Za-z])\)(?!\w)")


# ── Brace scanning ────────────────────────────────────────────────────
def _extract_braced(s: str, start: int) -> tuple[str, int] | None:
    """Read the balanced brace group whose opening ``{`` is at ``start``.

    Returns ``(content, index_after_closing_brace)``, or None if unbalanced.
    """
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "\\":  # skip escaped char, e.g. the \{ \} of set notation
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
        i += 1
    return None


def _read_arg(s: str, i: int) -> tuple[str, int] | None:
    """Read one LaTeX argument at/after ``i``: a brace group or a single token.

    Supports the abbreviated forms LaTeX allows, e.g. ``\\frac12``.
    """
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s):
        return None
    if s[i] == "{":
        return _extract_braced(s, i)
    return s[i], i + 1


def find_boxed(text: str) -> list[str]:
    """Return the contents of every ``\\boxed{...}`` / ``\\fbox{...}``.

    Uses balanced-brace scanning; a greedy regex would swallow everything from
    the first box to the last closing brace in the document.
    """
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", text):
        arg = _read_arg(text, m.end())
        if arg is not None:
            out.append(arg[0].strip())
    return out


# ── LaTeX -> ASCII core ───────────────────────────────────────────────
def _rewrite_fracs(s: str) -> str:
    """``\\frac{a}{b}`` -> ``(a)/(b)``, innermost args left intact.

    Each pass rewrites the outermost remaining fraction; nested fractions
    survive verbatim inside the arguments and are picked up by later passes,
    so the loop converges without needing a recursive-descent parser.
    """
    for _ in range(64):  # bounded: pathological input must not hang the grader
        m = re.search(r"\\[dt]?frac", s)
        if not m:
            return s
        num = _read_arg(s, m.end())
        if num is None:
            s = s[:m.start()] + s[m.end():]
            continue
        den = _read_arg(s, num[1])
        if den is None:
            s = s[:m.start()] + num[0] + s[num[1]:]
            continue
        s = f"{s[:m.start()]}({num[0]})/({den[0]}){s[den[1]:]}"
    return s


def _rewrite_sqrts(s: str) -> str:
    """``\\sqrt{x}`` -> ``sqrt(x)``; ``\\sqrt[n]{x}`` -> ``(x)**(1/(n))``."""
    for _ in range(64):
        m = re.search(r"\\sqrt\s*(\[([^\]]*)\])?", s)
        if not m:
            return s
        arg = _read_arg(s, m.end())
        if arg is None:
            s = s[:m.start()] + s[m.end():]
            continue
        root = m.group(2)
        body = f"({arg[0]})**(1/({root}))" if root else f"sqrt({arg[0]})"
        s = s[:m.start()] + body + s[arg[1]:]
    return s


def _rewrite_text(s: str) -> str:
    """Unwrap ``\\text{...}`` / ``\\mbox{...}``, keeping the content."""
    for _ in range(64):
        m = re.search(r"\\(?:text|mbox|textrm|mathrm|operatorname)\s*", s)
        if not m:
            return s
        arg = _read_arg(s, m.end())
        if arg is None:
            s = s[:m.start()] + s[m.end():]
            continue
        s = s[:m.start()] + arg[0] + s[arg[1]:]
    return s


def strip_latex_wrappers(s: str) -> str:
    """Peel outer ``\\boxed{}``/``\\text{}``/``$...$`` wrappers, repeatedly."""
    s = s.strip()
    for _ in range(16):
        before = s
        for cmd in (r"\boxed", r"\fbox", r"\text", r"\mbox"):
            if s.startswith(cmd):
                arg = _read_arg(s, len(cmd))
                if arg is not None and arg[1] >= len(s.rstrip()):
                    s = arg[0].strip()
        if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
            s = s[1:-1].strip()
        if s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
        if s == before:
            return s
    return s


def latex_to_ascii(s: str) -> str:
    """Convert a LaTeX answer fragment into plain, sympy-parsable ASCII.

    Shared by the exact-match path and the symbolic path so the two can never
    disagree about what a piece of LaTeX means.
    """
    s = strip_latex_wrappers(s)
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("{,}", ",")
    s = _DEGREE_RE.sub("", s)
    s = _rewrite_text(s)
    s = _rewrite_fracs(s)
    s = _rewrite_sqrts(s)
    s = _COMMAND_RE.sub(lambda m: LATEX_COMMANDS.get("\\" + m.group(1), ""), s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    s = _THOUSANDS_RE.sub("", s)
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    for pattern in _IMPLICIT_MULT_RES:
        s = pattern.sub("*", s)
    return s.rstrip(".")


def normalize_answer(ans: str) -> str:
    """Canonical form used for the cheap exact-match comparison."""
    s = latex_to_ascii(ans)
    for _ in range(8):
        reduced = _REDUNDANT_PARENS_RE.sub(r"\1", s)
        if reduced == s:
            break
        s = reduced
    s = re.sub(r"\s+", "", s)
    return s.lower()


def normalize_text(s: str) -> str:
    """Looser normalisation preserving case/spacing, for the numeric path."""
    return latex_to_ascii(s)


def sanitize_for_sympy(s: str) -> str:
    return latex_to_ascii(s)


# ── Parsing helpers ───────────────────────────────────────────────────
def try_parse_fraction(s: str) -> Fraction | None:
    s = normalize_text(s)
    if re.fullmatch(r"\(?-?\d+\)?\s*/\s*\(?-?\d+\)?", s):
        a, b = s.split("/")
        try:
            return Fraction(int(a.strip(" ()")), int(b.strip(" ()")))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def try_parse_number(s: str) -> float | None:
    s = normalize_text(s)
    frac = try_parse_fraction(s)
    if frac is not None:
        return float(frac)
    try:
        return float(s)
    except ValueError:
        pass
    expr = try_sympy_parse(s)
    if expr is not None:
        try:
            if expr.is_real and expr.is_number:
                return float(expr.evalf())
        except (TypeError, AttributeError, ValueError):
            pass
    return None


def try_sympy_parse(s: str):
    try:
        return sp.sympify(sanitize_for_sympy(s))
    except Exception:  # sympify raises a wide and undocumented set
        return None


def _split_collection(s: str) -> list[str] | None:
    """Split ``(1,2)``/``{1,2}``/``[1,2]``/``1,2`` into its comma-separated parts.

    MATH-500 answers include coordinate pairs, intervals and solution sets, so
    comparing them elementwise catches equalities that a whole-string compare
    misses (e.g. ``(1/2, 3)`` vs ``(0.5, 3)``).
    """
    s = normalize_text(s).strip()
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        inner = s[1:-1]
    else:
        inner = s
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    parts = [p.strip() for p in parts]
    if len(parts) < 2 or any(not p for p in parts):
        return None
    return parts


def are_equivalent(pred: str, gold: str, tol: float = 1e-9) -> bool:
    """Numeric/symbolic equivalence fallback for answers the exact-match
    comparison rejects (``1/2`` vs ``0.5`` vs ``\\frac{1}{2}``, algebraically
    equal expressions, elementwise-equal tuples)."""
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if pred_n == gold_n:
        return True

    pred_num = try_parse_number(pred_n)
    gold_num = try_parse_number(gold_n)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=tol, abs_tol=tol)

    pred_frac = try_parse_fraction(pred_n)
    gold_frac = try_parse_fraction(gold_n)
    if pred_frac is not None and gold_frac is not None:
        return pred_frac == gold_frac

    pred_parts = _split_collection(pred_n)
    gold_parts = _split_collection(gold_n)
    if pred_parts and gold_parts and len(pred_parts) == len(gold_parts):
        return all(are_equivalent(p, g, tol) for p, g in zip(pred_parts, gold_parts))

    pred_expr = try_sympy_parse(pred_n)
    gold_expr = try_sympy_parse(gold_n)
    if pred_expr is not None and gold_expr is not None:
        try:
            if sp.simplify(pred_expr - gold_expr) == 0:
                return True
        except Exception:
            pass
        try:
            return bool(pred_expr.equals(gold_expr))
        except Exception:
            pass

    return False


def final_answer_correct(expected: str, actual: str) -> bool:
    if not actual:
        return False
    if normalize_answer(expected) == normalize_answer(actual):
        return True
    return are_equivalent(actual, expected)


# ── Extraction from model output ──────────────────────────────────────
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|answer|result|equals)\s*(?:is|:|=)?\s*(.+?)$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_final_answer(text: str) -> tuple[str, bool]:
    """Pull the final answer out of a model response.

    Returns ``(answer, matched)`` where ``matched=False`` means no structured
    pattern was found and the last line was used -- a weak signal worth
    tracking separately as a parse failure rather than scoring blindly.
    """
    if not text or not text.strip():
        return "", False

    boxed = find_boxed(text)
    if boxed:
        return boxed[-1], True  # last box wins: models often box intermediates

    matches = _ANSWER_PHRASE_RE.findall(text)
    if matches:
        ans = matches[-1].strip().rstrip(".")
        ans = re.sub(r"\\[\(\[]$", "", ans).strip()
        ans = re.sub(r"^[=:]\s*", "", ans)
        if ans:
            return ans, True

    for pattern in (r"\\\[(.*?)\\\]", r"\$\$(.*?)\$\$", r"\\\((.*?)\\\)"):
        found = re.findall(pattern, text, re.DOTALL)
        if found:
            return found[-1].strip(), True

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[-1] if lines else text.strip()), False


_DISPLAY_DELIMS = ((r"\[", r"\]"), ("$$", "$$"))
_MIN_STEP_LEN = 5


def _read_display_block(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a display-math block starting at ``lines[i]``.

    Returns ``(body_without_delimiters, index_after_block)``. Handles both the
    single-line form (``$$x = 1$$``) and the spread-over-several-lines form.
    """
    first = lines[i]
    opener, closer = next(
        (o, c) for o, c in _DISPLAY_DELIMS if first.startswith(o)
    )
    body = first[len(opener):]
    if closer in body:
        return body.split(closer)[0].strip(), i + 1

    parts, i = [body], i + 1
    while i < len(lines):
        if closer in lines[i]:
            parts.append(lines[i].split(closer)[0])
            i += 1
            break
        parts.append(lines[i])
        i += 1
    return " ".join(p.strip() for p in parts if p.strip()), i


def extract_steps(text: str) -> list[str]:
    """Split reasoning into steps, attaching display math to the prose it
    belongs to.

    The PRM is trained on ``\\n\\n``-separated reasoning steps where a step is
    a claim *and* the equation supporting it, so a bare equation scored on its
    own is out of distribution. Keeping each block whole and attached to its
    lead-in sentence is what makes the resulting scores meaningful.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    steps: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("\\boxed"):
            i += 1
            continue
        if any(line.startswith(o) for o, _ in _DISPLAY_DELIMS):
            body, i = _read_display_block(lines, i)
            if not body or body.startswith("\\boxed"):
                continue
            if steps:
                steps[-1] += " " + body
            else:
                steps.append(body)
            continue
        if line.startswith("\\]"):  # unmatched closer
            i += 1
            continue
        i += 1
        if len(line) > _MIN_STEP_LEN:
            steps.append(line)
    return steps
