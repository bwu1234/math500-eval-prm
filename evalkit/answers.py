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
    r"\cup": "U", r"\cap": "n", r"\in": " in ",
    r"\pmod": "mod", r"\bmod": "mod", r"\mod": "mod",
}

# Escapes whose name is *not* alphabetic. ``_COMMAND_RE`` matches
# ``\\[a-zA-Z]+`` only, so these can never be reached by the command pass and
# have to be rewritten in their own pass -- otherwise ``\$36`` keeps its
# backslash and ``10,\!080`` keeps the ``\!`` that hides the digit grouping.
# ``\{``/``\}`` become sentinels so that set braces survive the later
# ``{`` -> ``(`` rewrite: a set is unordered, a tuple is not, and grading has
# to keep telling them apart.
_SET_OPEN, _SET_CLOSE = "\x01", "\x02"

LATEX_ESCAPES = {
    r"\%": "%", r"\$": "", r"\&": "&", r"\#": "", r"\_": "_",
    r"\{": _SET_OPEN, r"\}": _SET_CLOSE,
    r"\!": "", r"\,": "", r"\;": " ", r"\:": " ", "\\ ": " ", "\\\n": " ",
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
# Digit grouping. A comma between digits is ambiguous -- it separates the
# thousands of "20,000" but also the elements of the pair "(1,234)" -- so the
# grouping must be well formed (leading group of 1-3 digits, then groups of
# exactly 3) and bounded by non-digits. Spaces after the comma are tolerated
# only when the whole answer is one number: "11,\! 111,\! 111,\! 100" is
# unambiguous, "(1, 234)" is not.
_GROUPED_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?")
_TIGHT_GROUPED_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d.,])")
_GROUP_COMMA_RE = re.compile(r",\s*(?=\d)")
_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
_ESCAPE_RE = re.compile(r"\\[%$&#_{}!,;: \n]")
# ``x_{10}`` and ``x_10`` must normalise alike; base subscripts on numeral
# answers ("2516_8") are otherwise compared as different strings.
_SUBSCRIPT_BRACE_RE = re.compile(r"_\s*\{([^{}]*)\}")
# "12^{th}" -> "12". Runs before ^ becomes ** so the ordinal never turns into
# an exponent.
_ORDINAL_RE = re.compile(r"\^\s*\{?\s*(st|nd|rd|th)\s*\}?")
# Matrix/vector environments: rows are separated by \\, columns by &. Without
# this the whole environment collapses to the literal text "pmatrix" and the
# elementwise comparison never runs.
_MATRIX_ENV_RE = re.compile(
    r"\\begin\s*\{(p|b|B|v|V|small)?matrix\*?\}(.*?)\\end\s*\{\1?matrix\*?\}",
    re.DOTALL,
)
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
_TRAILING_COMMAND_RE = re.compile(r"\\[a-zA-Z]+$")


def _splice(prefix: str, body: str, suffix: str) -> str:
    """Join a rewritten fragment back on, without fusing two command names.

    ``1\\pm\\sqrt{5}`` rewrites the root first, giving ``1\\pmsqrt(5)``. The
    command pass then reads one command named ``pmsqrt``, finds it in no table
    and drops it -- silently turning ``1 \\pm \\sqrt 5`` into ``1*5``. Padding
    the boundary keeps ``\\pm`` a separate token.
    """
    if body[:1].isalpha() and _TRAILING_COMMAND_RE.search(prefix):
        prefix += " "
    return prefix + body + suffix


def _rewrite_environments(s: str) -> str:
    """``\\begin{pmatrix} a \\\\ b \\end{pmatrix}`` -> ``(a,b)``.

    Rows and columns both become comma-separated elements, so a vector is
    compared elementwise -- which is what lets ``1/5`` match ``0.2`` inside a
    matrix.
    """
    for _ in range(16):
        m = _MATRIX_ENV_RE.search(s)
        if m is None:
            return s
        body = m.group(2).replace("\\\\", ",").replace("&", ",")
        body = ",".join(p.strip() for p in body.split(",") if p.strip())
        s = s[:m.start()] + "(" + body + ")" + s[m.end():]
    return s


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
            s = _splice(s[:m.start()], num[0], s[num[1]:])
            continue
        s = _splice(s[:m.start()], f"({num[0]})/({den[0]})", s[den[1]:])
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
        s = _splice(s[:m.start()], body, s[arg[1]:])
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
        s = _splice(s[:m.start()], arg[0], s[arg[1]:])
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
    # Environments first: they own the `\\` row separator, which the escape
    # pass would otherwise eat. Escapes next, so that digit grouping sees
    # "10,080" rather than "10,\!080". Grouping has to be resolved while the
    # LaTeX brackets still mean what the author wrote -- once `{` becomes `(`,
    # the grouping of \frac{20,000}{\pi} is indistinguishable from the interval
    # (12,102), and one of the two gets mangled.
    s = _rewrite_environments(s)
    s = _ESCAPE_RE.sub(lambda m: LATEX_ESCAPES.get(m.group(0), ""), s)
    s = _strip_digit_grouping(s)
    s = _rewrite_text(s)
    s = _rewrite_fracs(s)
    s = _rewrite_sqrts(s)
    s = _COMMAND_RE.sub(lambda m: LATEX_COMMANDS.get("\\" + m.group(1), ""), s)
    s = _SUBSCRIPT_BRACE_RE.sub(r"_\1", s)
    s = _ORDINAL_RE.sub("", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(_SET_OPEN, "{").replace(_SET_CLOSE, "}")
    s = s.replace("^", "**")
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    for pattern in _IMPLICIT_MULT_RES:
        s = pattern.sub("*", s)
    return s.rstrip(".")


def _strip_digit_grouping(s: str) -> str:
    """Drop thousands separators from well-formed grouped numbers.

    A comma between digits separates the thousands of ``20,000`` but also the
    elements of ``(12,102)``. Inside round or square brackets it is read as a
    separator and left alone; elsewhere -- including inside LaTeX grouping
    braces, which never delimit a list -- it is read as digit grouping.
    """
    body = s.strip().lstrip("$")
    if _GROUPED_NUMBER_RE.fullmatch(body):
        return _GROUP_COMMA_RE.sub("", body)
    out, depth, i = [], 0, 0
    while i < len(s):
        ch = s[i]
        if ch in "([" or ch == _SET_OPEN:
            depth += 1
        elif ch in ")]" or ch == _SET_CLOSE:
            depth -= 1
        if depth == 0:
            m = _TIGHT_GROUPED_RE.match(s, i)
            if m is not None:
                out.append(m.group(1).replace(",", ""))
                i = m.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out)


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


def _split_top_level(inner: str) -> list[str]:
    """Split on commas that are not nested inside brackets or braces."""
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _split_collection(s: str) -> tuple[list[str], str] | None:
    """Split ``(1,2)``/``{1,2}``/``[1,2]``/``1,2`` into parts plus delimiters.

    MATH-500 answers include coordinate pairs, intervals and solution sets, so
    comparing them elementwise catches equalities that a whole-string compare
    misses (e.g. ``(1/2, 3)`` vs ``(0.5, 3)``). The delimiter pair is returned
    with the parts because it carries meaning: ``(3,4)`` and ``(3,4]`` are
    different intervals, so two collections may only be compared elementwise
    when their brackets agree.

    Takes an *already normalised* string. Re-normalising here would rewrite the
    braces of a set into parentheses -- ``latex_to_ascii`` is not idempotent on
    ``{``, by design, since grouping braces have to become parentheses -- and
    a set would then be indistinguishable from a tuple.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        inner, delims = s[1:-1], s[0] + s[-1]
    else:
        inner, delims = s, ""
    parts = _split_top_level(inner)
    if len(parts) < 2 or any(not p for p in parts):
        return None
    return parts, delims


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

    pred_split = _split_collection(pred_n)
    gold_split = _split_collection(gold_n)
    if pred_split and gold_split:
        pred_parts, pred_delims = pred_split
        gold_parts, gold_delims = gold_split
        compatible = not pred_delims or not gold_delims or pred_delims == gold_delims
        if len(pred_parts) == len(gold_parts) and compatible:
            return all(are_equivalent(p, g, tol) for p, g in zip(pred_parts, gold_parts))
        return False

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


# ── Presentation-tolerant tier ────────────────────────────────────────
# The two tiers above compare *values*. This one absorbs differences in how an
# answer is presented -- the unit label, the "x =" prefix, the order of a root
# list -- none of which a human grader would mark wrong. Every rule here is
# one-sided on purpose: a difference is forgiven only when one side omits the
# decoration entirely. Two answers that both carry it and disagree (cents vs
# dollars, base 8 vs base 9) still grade wrong.

_UNIT_WORDS = (
    "cm|mm|km|m|inches|inch|in|feet|foot|ft|yards|yard|yd|"
    "degrees|degree|radians|radian|"
    "cents|cent|dollars|dollar|"
    "squares|square|sq|units|unit|"
    "seconds|second|sec|minutes|minute|hours|hour|hr|"
    "days|day|weeks|week|months|month|years|year|"
    "students|people|ways|times|grades|grade|points|point|"
    "gallons|gallon|liters|liter|litres|litre|"
    "pounds|pound|ounces|ounce|kg|lb|oz|mph|percent"
)
_UNIT_RE = re.compile(r"(?:\*|\s)*(" + _UNIT_WORDS + r"|%)(?:\*\*\d+)?$")
_BASE_RE = re.compile(r"(.+?)_(\d+)$")
_VAR_PREFIX_RE = re.compile(r"^[a-z][a-z0-9]*(?:=|in)(?=[^=<>])")
_INEQ_TWO_SIDED_RE = re.compile(
    r"^(.+?)(<=|<|>=|>)([a-z][a-z0-9_]*)(<=|<|>=|>)(.+)$")
_INEQ_ONE_SIDED_RE = re.compile(r"^([a-z][a-z0-9_]*)(<=|<|>=|>)(.+)$")
_PM_RE = re.compile(r"\+-|-\+")


def _strip_units(n: str) -> tuple[str, frozenset[str]]:
    """Peel trailing unit words off a normalised answer.

    ``864*inches**2`` -> ``("864", {"inches"})``. The unit *set* comes back so
    the caller can tell "one side omitted the unit" from "the two sides
    disagree about the unit".
    """
    units = set()
    for _ in range(3):
        m = _UNIT_RE.search(n)
        if m is None or m.start() == 0:
            break
        units.add(m.group(1))
        n = n[:m.start()].rstrip("*").strip()
    return n, frozenset(units)


def _strip_base(n: str) -> tuple[str, str | None]:
    """``2516_8`` -> ``("2516", "8")``: split off a numeral-base subscript."""
    m = _BASE_RE.fullmatch(n)
    return (m.group(1), m.group(2)) if m else (n, None)


def _strip_var_prefix(n: str) -> str:
    """``x=5`` -> ``5``; ``xin[-2,7]`` -> ``[-2,7]`` (``\\in`` normalises to ``in``)."""
    return _VAR_PREFIX_RE.sub("", n, count=1)


def _interval_from_inequality(n: str) -> str | None:
    """Rewrite an inequality as the interval it denotes, or None.

    ``3<lambda<=4`` -> ``(3,4]``. MATH-500 asks for interval notation and
    accepts the inequality a model tends to write instead.
    """
    m = _INEQ_TWO_SIDED_RE.fullmatch(n)
    if m:
        lo, op1, _, op2, hi = m.groups()
        if op1 in ("<", "<=") and op2 in ("<", "<="):
            left = "[" if op1 == "<=" else "("
            right = "]" if op2 == "<=" else ")"
            return f"{left}{lo},{hi}{right}"
        if op1 in (">", ">=") and op2 in (">", ">="):
            left = "[" if op2 == ">=" else "("
            right = "]" if op1 == ">=" else ")"
            return f"{left}{hi},{lo}{right}"
        return None
    m = _INEQ_ONE_SIDED_RE.fullmatch(n)
    if m:
        _, op, bound = m.groups()
        if op in ("<", "<="):
            return f"(-oo,{bound}{']' if op == '<=' else ')'}"
        return f"{'[' if op == '>=' else '('}{bound},oo)"
    return None


def _expand_pm(part: str) -> list[str]:
    """``1+-sqrt(19)`` -> ``["1+sqrt(19)", "1-sqrt(19)"]``."""
    if not _PM_RE.search(part):
        return [part]
    plus = _PM_RE.sub(lambda m: "+" if m.group(0) == "+-" else "-", part)
    minus = _PM_RE.sub(lambda m: "-" if m.group(0) == "+-" else "+", part)
    return [plus, minus]


def _answer_elements(n: str) -> tuple[list[str], bool, str]:
    """Split an answer into elements, its ordering, and its delimiters.

    Sets (``\\{...\\}``) and bare comma lists ("enter all the roots, separated
    by commas") are unordered; tuples, intervals and vectors are not. The
    delimiters come back because they carry meaning of their own -- ``(3,4]``
    is not ``(3,4)`` and a set is not a coordinate pair.
    """
    if len(n) >= 2 and n[0] == "{" and n[-1] == "}":
        parts, ordered, delims = _split_top_level(n[1:-1]), False, "{}"
    elif len(n) >= 2 and n[0] in "([" and n[-1] in ")]":
        parts = [p for p in _split_top_level(n[1:-1]) if p]
        return parts, True, n[0] + n[-1]
    else:
        parts, ordered, delims = _split_top_level(n), False, ""
    parts = [p for p in parts if p]
    expanded = [x for p in parts for x in _expand_pm(p)]
    return expanded, ordered and len(expanded) == len(parts), delims


def _scalar_equal(a: str, b: str) -> bool:
    return a == b or are_equivalent(a, b)


def _elements_match(a: str, b: str) -> bool:
    ea, ordered_a, delims_a = _answer_elements(a)
    eb, ordered_b, delims_b = _answer_elements(b)
    if len(ea) != len(eb):
        return False
    if delims_a and delims_b and delims_a != delims_b:
        return False
    if ordered_a or ordered_b:
        return all(_scalar_equal(x, y) for x, y in zip(ea, eb))
    remaining = list(eb)
    for x in ea:
        for i, y in enumerate(remaining):
            if _scalar_equal(x, y):
                remaining.pop(i)
                break
        else:
            return False
    return True


def _decorated_match(a: str, b: str) -> bool:
    """Compare two normalised answers modulo units and numeral base."""
    a_core, a_units = _strip_units(a)
    b_core, b_units = _strip_units(b)
    if a_units and b_units and a_units != b_units:
        return False
    a_core, a_base = _strip_base(a_core)
    b_core, b_base = _strip_base(b_core)
    if a_base and b_base and a_base != b_base:
        return False
    return bool(a_core) and bool(b_core) and _elements_match(a_core, b_core)


def _presentation_forms(raw: str) -> list[str]:
    """Normalised spellings of one answer that differ only in presentation."""
    forms = [normalize_answer(raw)]
    for form in list(forms):
        without_prefix = _strip_var_prefix(form)
        if without_prefix != form and without_prefix:
            forms.append(without_prefix)
    for form in list(forms):
        interval = _interval_from_inequality(form)
        if interval:
            forms.append(interval)
    return forms


def final_answer_correct(expected: str, actual: str) -> bool:
    if not actual:
        return False
    if normalize_answer(expected) == normalize_answer(actual):
        return True
    if are_equivalent(actual, expected):
        return True
    return any(
        _decorated_match(a, b)
        for a in _presentation_forms(actual)
        for b in _presentation_forms(expected)
    )


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
