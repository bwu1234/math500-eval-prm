# MATH-500 evaluation harness

An evaluation harness for mathematical reasoning that measures **how a model
reasons**, not just whether its final answer matches. Reasoning steps are
scored by a Process Reward Model trained on human step-level correctness
labels, and the harness then checks whether that score is actually worth
anything.

```
python math500_eval.py -n 100                        # 100 questions
python math500_eval.py -n 100 -k 5                   # self-consistency, 5 samples
python math500_eval.py -n 100 --compare gemini,qwen  # diff two backends
python math500_eval.py -n 20 -m mock --no-prm        # runs anywhere, no GPU or API key
```

Every run writes a report automatically; the committed `report.html` is the
latest **full 500-question** run, which is the only kind of run allowed to
replace it:

**[View the current report →](https://htmlpreview.github.io/?https://github.com/bwu1234/math500-eval-prm/blob/main/report.html)**
&nbsp;·&nbsp; [source](report.html)

<sub>GitHub serves `.html` as source text, so the link above renders it through
htmlpreview. The report is fully self-contained — downloading it and opening it
locally works identically.</sub>

---

## Results: `gemma-4-31b-it` on the full MATH-500

**99.0%** (495/500), 191,822 output tokens, 4 responses truncated at
`MAX_TOKENS` and counted as parse failures rather than graded on a guess.

| Level | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Accuracy | 100% | 100% | 99% | 99% | 98% |

Accuracy falls with difficulty, which is the sanity check you want; the weakest
subject is Prealgebra (96%), not the ones that sound hardest.

<sub>The first pass over this run reported 95.0% (475/500). Filtering the 25
failures showed 20 were the *grader's* fault, not the model's — an answer
marked wrong for carrying a unit, a `\$`, an `x =`, or its roots in a different
order. The grading tiers below were extended to absorb presentation
differences, and the stored predictions re-graded with
`python scripts/regrade.py`. Of the 5 remaining failures, 4 are empty responses
and one is a genuine disagreement with the reference answer.</sub>

### Does the PRM signal actually hold up? Not at this sample size.

| | |
|---|---|
| AUC (PRM score → correctness) | **0.826** |
| 95% CI (Hanley-McNeil) | **[0.539, 1.000]** |
| Mean PRM, correct / incorrect | 0.521 / 0.395 |
| Solutions to rank | 495 correct vs **1** incorrect |

The point estimate looks strong, the interval excludes 0.5, and it would be
easy to write this up as "the PRM predicts correctness". It doesn't support
that at all: the entire negative class is **one solution**. Hanley-McNeil is a
normal approximation, and at that sample size it produces a confident-looking
interval resting on nothing, so the harness refuses the positive verdict and
reports the reward as **unvalidated** rather than confirmed.

This is the more interesting failure mode of the two. An interval straddling
0.5 is self-evidently weak evidence; an interval that *excludes* 0.5 on one
negative example looks like a result. Establishing whether the reward is
informative needs failures to rank against — a weaker model, or a harder
question set. That threshold lives in
`analysis.MIN_CLASS_SIZE_FOR_AUC_VERDICT`.

---

## Why a process reward model

The usual way to evaluate reasoning quality is to ask a large language model
to grade it. That gives you one model's opinion, with the biases that come
with it — verbosity preference, self-preference, and sensitivity to the
grading prompt.

This harness instead uses
[`Qwen2.5-Math-1.5B-Instruct-PRM-0.2`](https://huggingface.co/HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2),
a token-classification model trained (TRL stepwise-reward) on **PRM800K**, a
dataset of human step-level correctness annotations. Each reasoning step is
scored on the cumulative prefix that precedes it, and the reward is
`P(LABEL_1)` read off the step's trailing separator token — no prompting, no
answer parsing, no judge model. It loads 4-bit quantised to fit a ~4 GB GPU
budget.

Because the PRM never sees the gold answer, it cannot be reading correctness
off the final result — it is scoring the reasoning itself.

**And then the harness checks whether that signal is real.** Spending GPU time
on step scoring is only justified if the score predicts something, so every
run reports the AUC of PRM score against final-answer correctness, the
point-biserial correlation, a reliability curve, and the score distributions
for correct vs incorrect solutions.

Crucially it reports a **confidence interval on that AUC**, and the verdict is
driven by the interval and the size of the smaller outcome class rather than the
point estimate. When a strong model leaves only a handful of failures there is
not enough evidence to rank anything, however far the point estimate happens to
sit from 0.5 — see the result above, where an AUC of 0.826 with an interval
excluding chance is still reported as unvalidated, because it is measured
against one incorrect solution. An eval that quietly reported the bare number
would have overstated its own central claim.

---

## Answer grading is the part that's easy to get wrong

An eval's most dangerous failure mode is a **false positive**: grading a wrong
answer as correct silently inflates every number downstream, and nothing about
the output looks broken.

Normalising LaTeX by deleting backslash commands, braces, and slashes — a
common shortcut — does exactly this. `\frac{1}{2}` and `12` both collapse to
`"12"`. Sweeping the full MATH-500 test split, that approach maps **20 groups
of genuinely different gold answers onto shared normalised forms**:

| These answers to different questions | both normalise to |
|---|---|
| `\frac{14}{3}` and `143` | `143` |
| `\sqrt{5}` and `5` | `5` |
| `\frac{1}{2}` and `12` | `12` |
| `\frac{\sqrt{3}}{3}` and `33` | `33` |
| `-\sqrt{3}` and `-3` | `-3` |

Grading here is structure-preserving instead, and runs in three tiers:

1. **Canonical form** (`normalize_answer`) — a real LaTeX-to-ASCII conversion:
   balanced-brace `\frac{a}{b}` → `(a)/(b)` (handles nesting), `\sqrt{x}` →
   `sqrt(x)`, implicit multiplication made explicit (`3\sqrt{13}` →
   `3*sqrt(13)`), degrees and thousands separators stripped.
2. **Symbolic fallback** (`are_equivalent`) — numeric, rational, elementwise
   (for tuples and intervals), then sympy simplification, so `0.5`, `1/2` and
   `\frac{1}{2}` still grade equal.
3. **Presentation tolerance** — the first two tiers compare *values*; this one
   absorbs how an answer is written. A unit or label (`12\text{ cm}` vs `12`), a
   `\$`, a restated variable (`x=5` vs `5`), a numeral base (`2516` vs
   `2516_8`), `\pm` against the roots written out, root order, an inequality
   where an interval was asked for. This tier is what the 20 misgraded answers
   above needed.

Under-reporting accuracy is recoverable and over-reporting is not, so every
rule in tier 3 is **one-sided**: a difference is forgiven only when one side
omits the decoration entirely. Two answers that both carry it and disagree
(`cm` vs `m`, base 8 vs base 9, `(3,4]` vs `(3,4)`, a set vs a coordinate pair)
still grade wrong — as do extra roots, which are a wrong answer rather than a
formatting difference.

On the same sweep this produces **zero false positives** in tier 1, and across
tier 3 exactly three pairs of distinct gold answers become indistinguishable —
`5`/`x=5`, `15`/`15\mbox{ cm}^2`, and `40`/`40_9` — each of which requires the
model to have produced the right value to begin with. Reproduce it:

```bash
python scripts/audit_normaliser.py
```

Because grading is cheap and model output is not, a grader fix can be applied to
a finished run instead of paying for it again:

```bash
python scripts/regrade.py            # show the diff
python scripts/regrade.py --write --report
```

Extraction is hardened too — `\boxed{...}` is read with a balanced-brace
scanner rather than a greedy `\\boxed\{(.*)\}` regex, which otherwise captures
everything from the first box to the last closing brace anywhere in the
response. Responses with no structured answer are counted as **parse
failures** and reported separately rather than being graded on a guess.

---

## What a run measures

| | |
|---|---|
| **Accuracy** | overall, and sliced by subject and difficulty level |
| **Parse failure rate** | responses with no extractable structured answer |
| **PRM score** | mean step reward, plus every individual step's score |
| **PRM validity** | AUC **with a Hanley-McNeil confidence interval**, point-biserial *r*, reliability curve, ECE, class separation |
| **Selection strategies** | single-shot vs majority vote vs PRM best-of-*n* vs PRM-weighted vote |
| **Cost** | output tokens and wall-clock latency per question |

### Self-consistency and reranking

With `-k N`, the harness samples *N* solutions per question and resolves them
into a final answer four ways:

- `first` — the single-shot baseline
- `majority` — self-consistency: the largest **equivalence class**, clustered
  with the same relation used for grading, so `1/2` and `0.5` are one vote
  rather than two
- `prm_best` — best-of-*n*, taking the highest mean step reward
- `prm_weighted` — vote weighted by PRM score

The headline `accuracy` always stays the single-shot number; reporting a
best-of-*n* result as "accuracy" would overstate the model.

---

## The report

Every run writes a self-contained HTML file — inline SVG charts, no
dependencies, no external requests, no build step. Open it from disk, commit
it, or attach it to a PR. It renders accuracy by difficulty and subject, a
subject × difficulty heatmap, the PRM reliability curve and score
distributions, selection-strategy comparison, backend divergences, and a
per-question drilldown where **each reasoning step is shaded by its PRM
reward** — which localises where a solution went wrong.

```bash
python math500_eval.py -n 500                    # writes report.html + report_gemini.html
python math500_eval.py -n 100                    # writes report_gemini_partial.html
python math500_eval.py -n 100 --report out.html  # somewhere else
python math500_eval.py -n 100 --no-report        # skip it
python math500_eval.py --report-only             # rebuild, no model run
```

### Only a full run owns `report.html`

`report.html` is the artifact that gets committed and linked, so it stands for
a complete 500-question run and nothing else. A 20-question spot check that
overwrote it would leave a 4% sample being read as *the* result, so where a
report lands is decided by how many questions the run actually covered:

| Run | Writes |
|---|---|
| 500 questions | `report.html` **and** `report_<model>.html` |
| fewer than 500 | `report_<model>_partial.html` only |
| `--compare`, 500 questions | `report_<model>.html` per backend |
| `--compare`, fewer | `report_<model>_partial.html` per backend |

Every full run also writes a per-model copy, so finishing 500 questions on
`qwen` no longer destroys the `gemini` report that came before it — the full
runs accumulate side by side instead of overwriting one another. A comparison
run never writes `report.html`: no single backend in it can head the page.

The results file is the authority, not the flags. A `-q` rerun spliced back
into a finished 500-question run is still a full run and refreshes
`report.html`; a run launched with `-n 500` that stops early is not, and does
not. A custom `--report scorecard.html` follows the same rule under its own
name (`scorecard_qwen.html`, `scorecard_qwen_partial.html`).

`--compare` writes one report per backend, each carrying the shared comparison
section — a comparison run produces no combined `eval_results.json`, so a
single report would have to pick one backend's results to head the page.

Report generation runs after the model work is finished and downgrades any
failure to a warning, so a rendering bug can never cost you a completed run.
`--report-only` rebuilds from the results already on disk, under the same rule.

The committed `report.html` is the full 500-question `gemma-4-31b-it` run
described above.

### Nothing is overwritten

Scoring a new model archives the previous run rather than replacing it. Its
log, results and report move into `logs/` under a **shared timestamp**, so the
three files belonging to one run stay correlated:

```
report.html                         <- the last full 500-question run
eval_results.json                   <- the run you just did
logs/report_20260730_001752.html    <- the full run before it
logs/eval_results_20260730_001752.json
logs/eval_debug_20260730_001752.log
```

A comparison run archives each backend separately
(`logs/report_qwen_<stamp>.html`). A run only archives the reports it is about
to replace, so a partial run leaves `report.html` where it is rather than
quietly retiring it — the log and results still rotate. Reports are archived
even under `--no-report`, so a stale report can never sit next to the results
of a newer run. Rebuild any archived report with
`python math500_eval.py --report-only --out-dir <dir>`.

---

## Usage

| Flag | Meaning |
|---|---|
| `-n, --num-questions` | how many MATH-500 questions (default 3, max 500) |
| `-m, --model` | backend: `gemini`, `qwen` (local Ollama), or `mock` |
| `--model-name` | override the backend's default model id |
| `-k, --samples` | samples per question; `>1` enables self-consistency |
| `--temperature` | default 0 for `k=1`, 0.7 for `k>1` |
| `-q, --question` | rerun one question, merging it into existing results |
| `--compare A,B` | run the same questions through several backends |
| `--report PATH` | base path for the report (default `report.html`); only a full 500-question run writes it |
| `--no-report` | skip report generation |
| `--report-only` | rebuild the report from existing results |
| `--no-prm` | skip PRM scoring (no GPU needed) |
| `--no-resume` | ignore any checkpoint and start fresh |

### Long runs are resumable

Every completed question is appended to a fsync'd JSONL checkpoint, so a crash
at question 400 of 500 does not cost the whole run — rerun the same command
and it picks up where it stopped. The checkpoint is keyed by a fingerprint of
the sampling configuration, so changing the model, `k`, or temperature
correctly invalidates it instead of silently mixing incomparable generations.

### Running without a GPU or API key

`-m mock` swaps in a deterministic test double that exercises the entire
pipeline — checkpointing, selection, aggregation, reporting — with no model
anywhere. Results produced this way are tagged `synthetic` and the report
carries a banner, so they can never be mistaken for a measurement.

---

## Setup

```bash
pip install -r requirements.txt        # add -r requirements-dev.txt for tests
export GOOGLE_API_KEY=...              # for the gemini backend
export OLLAMA_HOST=http://localhost:11434   # optional, for the qwen backend
```

---

## Architecture

```
math500_eval.py        CLI entry point
evalkit/
  answers.py           LaTeX normalisation, equivalence, extraction   (sympy only)
  analysis.py          accuracy slices, AUC, calibration, voting      (stdlib only)
  report.py            self-contained HTML + inline SVG               (stdlib only)
  backends.py          Gemini / Ollama / mock, shared retry logic
  prm.py               process reward model scoring
  runner.py            pipeline, checkpointing, aggregation
scripts/
  audit_normaliser.py  reproduces the grader collision sweep
  regrade.py           re-grades a finished run against the current grader
tests/                 244 tests, ~1s, no GPU required
```

The split is load-bearing: `answers`, `analysis` and `report` import no
torch, transformers, or datasets. That keeps the grading logic — the part most
likely to silently break — testable in under a second on any machine, and a
test asserts the property so it cannot regress.

```bash
make test        # or: python -m pytest tests/ -q
```

---

## Notes and limitations

- **Step segmentation drives PRM scores.** The PRM was trained on `\n\n`
  separated steps where a step is a claim together with its supporting
  equation, so `extract_steps` keeps display-math blocks whole and attached to
  the prose that introduces them. An earlier version split `\[ ... \]` across
  step boundaries, producing fragments like `x = 1 \]` that are out of
  distribution for the model.
- **PRM utility is model- and sample-dependent.** A mean step reward is a
  coarse summary of a variable-length score sequence; min-step or last-step
  aggregation may separate outcomes better. The reported AUC is the honest
  check on whether the signal is usable at all.
- **Symbolic equivalence is undecidable in general.** `are_equivalent` falls
  back to sympy `simplify`/`equals`, which can return false negatives on hard
  expressions. It is deliberately biased toward false negatives over false
  positives — under-reporting accuracy is recoverable, over-reporting is not.
- Accuracy figures depend on the prompt, which is fixed in
  `runner.PROMPT_TEMPLATE`; changing it invalidates comparisons to earlier
  runs (and correctly invalidates checkpoints).
