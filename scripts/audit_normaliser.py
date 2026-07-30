"""Sweep every MATH-500 gold answer for answer-grading collisions.

A collision is two *different* gold answers that normalise to the same string.
When the two are genuinely equal (`90` and `90^\\circ`) that is correct and
desirable. When they are not (`\\frac{1}{2}` and `12`) the grader cannot tell
them apart, and any model prediction of one form scores as the other -- a
false positive that silently inflates reported accuracy.

Run from the repo root:  python scripts/audit_normaliser.py
"""
import itertools
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset  # noqa: E402

from evalkit.answers import (  # noqa: E402
    _interval_from_inequality,
    _strip_base,
    _strip_units,
    _strip_var_prefix,
    are_equivalent,
    final_answer_correct,
    normalize_answer,
)

# ── the original implementation, kept verbatim for comparison ─────────
LATEX_SYMBOLS = {
    r"\pi": "pi", r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\theta": "theta", r"\phi": "phi", r"\varphi": "phi",
    r"\omega": "omega", r"\sigma": "sigma", r"\mu": "mu", r"\tau": "tau",
    r"\epsilon": "epsilon", r"\lambda": "lambda", r"\infty": "inf",
    r"\partial": "d", r"\rightarrow": "to", r"\Rightarrow": "to",
    r"\approx": "~=", r"\neq": "!=", r"\leq": "<=", r"\geq": ">=",
}


def old_normalize_answer(ans: str) -> str:
    ans = re.sub(r"\\boxed\{(.*?)\}", r"\1", ans, flags=re.DOTALL)
    ans = re.sub(r"\^?\\circ|\^?\\degree|°", "", ans)
    for cmd, text in LATEX_SYMBOLS.items():
        ans = ans.replace(cmd, text)
    ans = re.sub(r"\\[a-zA-Z]+", "", ans)
    ans = re.sub(r"[{}]", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("/", "")
    return ans.strip().lower()


def collisions(golds, fn):
    buckets = defaultdict(set)
    for g in golds:
        buckets[fn(g)].add(g)
    return {k: v for k, v in buckets.items() if len(v) > 1}


def classify(pairs):
    """Split collisions into genuine equivalences vs false positives."""
    genuine, false_pos = [], []
    for key, variants in pairs.items():
        vs = sorted(variants)
        # A collision is benign only if every variant really is the same value.
        if all(are_equivalent(vs[0], v) for v in vs[1:]):
            genuine.append((key, vs))
        else:
            false_pos.append((key, vs))
    return genuine, false_pos


ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
golds = [r["answer"] for r in ds]

for label, fn in (("ORIGINAL", old_normalize_answer), ("NEW", normalize_answer)):
    cols = collisions(golds, fn)
    genuine, false_pos = classify(cols)
    print(f"\n=== {label} normalize_answer ===")
    print(f"  distinct-answer collisions: {len(cols)}")
    print(f"    genuine equivalences   : {len(genuine)}")
    print(f"    FALSE POSITIVES        : {len(false_pos)}")
    for key, vs in false_pos[:12]:
        print(f"      {key!r}  <-  {vs}")


# ── tier 3: presentation tolerance ────────────────────────────────────
# The lenient tier forgives units, "x =" prefixes, numeral bases and root
# order. That leniency is where new false positives would come from, so sweep
# for it directly: bucket the golds by the core the tier compares, then grade
# every within-bucket pair against each other. Distinct golds that grade equal
# are answers to different questions the grader cannot tell apart.
def presentation_core(s: str) -> str:
    n = _strip_var_prefix(normalize_answer(s))
    n = _interval_from_inequality(n) or n
    core, _ = _strip_units(n)
    core, _ = _strip_base(core)
    return core


buckets = defaultdict(set)
for g in golds:
    buckets[presentation_core(g)].add(g)

pairs = [
    (a, b)
    for variants in buckets.values() if len(variants) > 1
    for a, b in itertools.combinations(sorted(variants), 2)
]
equated = [(a, b) for a, b in pairs if final_answer_correct(a, b)]
spurious = [(a, b) for a, b in equated if not are_equivalent(a, b)]

print("\n=== NEW final_answer_correct (presentation tier) ===")
print(f"  distinct golds sharing a comparison core: {len(pairs)} pairs")
print(f"    graded equal               : {len(equated)}")
print(f"    of those, not truly equal  : {len(spurious)}")
for a, b in equated[:12]:
    verdict = "value-equal" if are_equivalent(a, b) else "DECORATION ONLY"
    print(f"      {a!r} <-> {b!r}  [{verdict}]")
