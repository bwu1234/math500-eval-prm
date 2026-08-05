"""Self-contained HTML report.

Charts are hand-rolled SVG rather than matplotlib output: the report stays a
single file with no image assets, no dependencies, and no build step, so it
can be opened straight from disk, committed, attached to a PR, or served
as-is. Everything here is stdlib.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime

from .analysis import compute_group_accuracy, cross_tab, prm_calibration

__all__ = ["build_report", "render_report", "write_report"]

PALETTE = {
    "correct": "#16a34a",
    "incorrect": "#dc2626",
    "accent": "#2563eb",
}


# ── small helpers ─────────────────────────────────────────────────────
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def score_color(score: float | None) -> str:
    """Red -> amber -> green ramp for a 0..1 reward."""
    if score is None:
        return "var(--muted)"
    s = min(max(score, 0.0), 1.0)
    return f"hsl({s * 120:.0f} 65% 45%)"


# ── SVG charts ────────────────────────────────────────────────────────
def _axes(left: int, top: int, right: int, bottom: int) -> str:
    """L-shaped axis lines for a plot area bounded by the given edges."""
    return (
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>'
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>'
    )


def bar_chart(items: list[tuple[str, float, int, int]], title: str,
              width: int = 520, height: int = 260) -> str:
    """Vertical bars of accuracy. Items are (label, accuracy, correct, total)."""
    if not items:
        return ""
    pad_l, pad_b, pad_t, pad_r = 44, 46, 16, 12
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    slot = plot_w / len(items)
    bar_w = min(slot * 0.62, 64)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end">{frac:.0%}</text>')

    for i, (label, acc, correct, total) in enumerate(items):
        acc = acc or 0.0
        x = pad_l + slot * i + (slot - bar_w) / 2
        h = plot_h * acc
        y = pad_t + plot_h - h
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(h, 1):.1f}" rx="3"><title>{esc(label)}: '
            f'{correct}/{total} ({acc:.1%})</title></rect>'
        )
        parts.append(f'<text class="bar-value" x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" '
                     f'text-anchor="middle">{acc:.0%}</text>')
        parts.append(f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{height - pad_b + 18}" '
                     f'text-anchor="middle">{esc(label)}</text>')
        parts.append(f'<text class="tick dim" x="{x + bar_w / 2:.1f}" y="{height - pad_b + 32}" '
                     f'text-anchor="middle">n={total}</text>')

    parts.append(_axes(pad_l, pad_t, width - pad_r, pad_t + plot_h))
    parts.append("</svg>")
    return "".join(parts)


def heatmap(tab: dict, title: str) -> str:
    """Subject x difficulty accuracy grid, shaded by accuracy."""
    rows, cols, cells = tab.get("rows", []), tab.get("cols", []), tab.get("cells", {})
    if not rows or not cols:
        return ""
    cell_w, cell_h, label_w, header_h = 62, 34, 150, 26
    width = label_w + cell_w * len(cols) + 8
    height = header_h + cell_h * len(rows) + 8

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for j, col in enumerate(cols):
        x = label_w + cell_w * j + cell_w / 2
        parts.append(f'<text class="tick" x="{x}" y="{header_h - 8}" text-anchor="middle">L{esc(col)}</text>')

    for i, row in enumerate(rows):
        y = header_h + cell_h * i
        parts.append(f'<text class="tick" x="{label_w - 10}" y="{y + cell_h / 2 + 4}" '
                     f'text-anchor="end">{esc(row)}</text>')
        for j, col in enumerate(cols):
            x = label_w + cell_w * j
            cell = cells.get(f"{row}|{col}")
            if not cell or not cell["total"]:
                parts.append(f'<rect class="cell-empty" x="{x}" y="{y}" width="{cell_w - 3}" '
                             f'height="{cell_h - 3}" rx="3"/>')
                continue
            acc = cell["correct"] / cell["total"]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" rx="3" '
                f'fill="{score_color(acc)}" opacity="0.85"><title>{esc(row)} / level {esc(col)}: '
                f'{cell["correct"]}/{cell["total"]}</title></rect>'
            )
            parts.append(f'<text class="cell-text" x="{x + (cell_w - 3) / 2}" '
                         f'y="{y + cell_h / 2 + 4}" text-anchor="middle">'
                         f'{cell["correct"]}/{cell["total"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def calibration_chart(bins: list[dict], width: int = 520, height: int = 300) -> str:
    """Reliability curve: mean PRM score vs empirical accuracy per bin.

    The diagonal is perfect calibration. What matters for reranking is not
    that the points sit on the diagonal but that they trend upward -- that is
    what says the score carries ordering information.
    """
    populated = [b for b in bins if b["count"]]
    if len(populated) < 2:
        return ""
    pad_l, pad_b, pad_t, pad_r = 46, 42, 16, 12
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    fx = lambda v: pad_l + plot_w * v          # noqa: E731
    fy = lambda v: pad_t + plot_h * (1 - v)    # noqa: E731

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="PRM calibration">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{fy(frac):.1f}" '
                     f'x2="{width - pad_r}" y2="{fy(frac):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{fy(frac) + 4:.1f}" '
                     f'text-anchor="end">{frac:.0%}</text>')
        parts.append(f'<text class="tick" x="{fx(frac):.1f}" y="{height - pad_b + 18}" '
                     f'text-anchor="middle">{frac:.1f}</text>')

    parts.append(f'<line class="diagonal" x1="{fx(0):.1f}" y1="{fy(0):.1f}" '
                 f'x2="{fx(1):.1f}" y2="{fy(1):.1f}"/>')

    max_count = max(b["count"] for b in populated)
    points = [(fx(b["mean_score"]), fy(b["accuracy"]), b) for b in populated]
    parts.append('<polyline class="series" points="' +
                 " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points) + '"/>')
    for x, y, b in points:
        r = 3 + 5 * (b["count"] / max_count)
        parts.append(
            f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}">'
            f'<title>score {b["lo"]:.1f}-{b["hi"]:.1f}: {b["correct"]}/{b["count"]} '
            f'correct ({b["accuracy"]:.0%})</title></circle>'
        )
    parts.append(f'<text class="axis-label" x="{pad_l + plot_w / 2:.0f}" y="{height - 6}" '
                 f'text-anchor="middle">mean PRM score</text>')
    parts.append(f'<text class="axis-label" transform="translate(12,{pad_t + plot_h / 2:.0f}) '
                 f'rotate(-90)" text-anchor="middle">accuracy</text>')
    parts.append("</svg>")
    return "".join(parts)


def score_histogram(correct: list[float], wrong: list[float],
                    n_bins: int = 10, width: int = 520, height: int = 280) -> str:
    """Overlaid PRM distributions for correct vs incorrect solutions.

    Two well-separated humps mean the PRM is discriminative; one overlapping
    blob means it is not, whatever the headline average says.
    """
    if not correct and not wrong:
        return ""
    pad_l, pad_b, pad_t, pad_r = 40, 46, 16, 12
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def histogram(values):
        counts = [0] * n_bins
        for v in values:
            counts[min(int(min(max(v, 0.0), 1.0) * n_bins), n_bins - 1)] += 1
        return counts

    hc, hw = histogram(correct), histogram(wrong)
    peak = max([*hc, *hw, 1])
    slot = plot_w / n_bins
    bar_w = slot * 0.38

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="PRM score distribution">']
    parts.append(_axes(pad_l, pad_t, width - pad_r, pad_t + plot_h))
    for i in range(n_bins):
        base_x = pad_l + slot * i + slot / 2
        for offset, counts, color, label in (
            (-bar_w, hc, PALETTE["correct"], "correct"),
            (0, hw, PALETTE["incorrect"], "incorrect"),
        ):
            h = plot_h * counts[i] / peak
            if h <= 0:
                continue
            x = base_x + offset
            parts.append(
                f'<rect x="{x:.1f}" y="{pad_t + plot_h - h:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{color}" opacity="0.8" rx="2">'
                f'<title>{label}, score {i / n_bins:.1f}-{(i + 1) / n_bins:.1f}: '
                f'{counts[i]}</title></rect>'
            )
        if i % 2 == 0:
            parts.append(f'<text class="tick" x="{base_x:.1f}" y="{height - pad_b + 18}" '
                         f'text-anchor="middle">{i / n_bins:.1f}</text>')
    parts.append(f'<text class="axis-label" x="{pad_l + plot_w / 2:.0f}" y="{height - 8}" '
                 f'text-anchor="middle">PRM score</text>')
    parts.append("</svg>")
    return "".join(parts)


# ── report assembly ───────────────────────────────────────────────────
def build_report(out_dir: str = ".", results_file: str = "eval_results.json",
                 comparison_file: str = "eval_comparison.json") -> dict:
    """Load results (and a comparison, if one exists) into a report payload."""
    results_path = os.path.join(out_dir, results_file)
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"No results at {results_path}. Run an eval first, e.g. "
            f"`python math500_eval.py -n 10 -m mock --no-prm`."
        )
    with open(results_path) as f:
        summary = json.load(f)

    comparison = None
    comparison_path = os.path.join(out_dir, comparison_file)
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)

    rows = [r for r in summary.get("results", []) if r]
    for i, row in enumerate(rows):
        row.setdefault("index", i)

    # Backfill anything a results file written by an older version lacks, so
    # archived runs stay reportable instead of rendering as empty panels.
    # Recompute when the stored block predates a field, not just when it is
    # missing entirely: a calibration without auc_ci would otherwise render the
    # significance verdict from a point estimate alone.
    stored_cal = summary.get("prm_calibration")
    if stored_cal is None or "auc_ci" not in stored_cal:
        summary["prm_calibration"] = prm_calibration(rows)
    if summary.get("correct") is None:
        summary["correct"] = sum(1 for r in rows if r.get("answer_accuracy"))
    if not summary.get("by_subject"):
        summary["by_subject"] = compute_group_accuracy(rows, "subject")
    if not summary.get("by_difficulty"):
        summary["by_difficulty"] = compute_group_accuracy(rows, "difficulty")
    if not summary.get("subject_by_difficulty"):
        summary["subject_by_difficulty"] = cross_tab(rows, "subject", "difficulty")

    return {"summary": summary, "rows": rows, "comparison": comparison}


def _stat_cards(summary: dict, rows: list[dict]) -> str:
    cal = summary.get("prm_calibration") or {}
    labels = _scorer_labels(summary)
    abbr = labels["abbr"]
    cards = [
        ("Accuracy", pct(summary.get("accuracy"), 1),
         f"{summary.get('correct', 0)}/{summary.get('num_questions', len(rows))} correct"),
        (f"Avg {abbr} score", num(summary.get("avg_prm_score")),
         labels["note"] if summary.get("avg_prm_score") is not None
         else "reward scoring disabled"),
        (f"{abbr} &rarr; correct AUC", num(cal.get("auc")),
         "0.5 = no signal"),
        ("Parse failures", str(summary.get("parse_failures", 0)),
         f"{pct(summary.get('parse_failure_rate'))} of responses"),
        ("Output tokens", f"{summary.get('total_output_tokens', 0):,}",
         f"{summary.get('duration_s') or 0:.0f}s wall clock"),
    ]
    return "".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'<div class="card-note">{note}</div></div>'
        for label, value, note in cards
    )


def _scorer_labels(summary: dict) -> dict:
    """What to call the reward model in prose, for this run.

    The scores share one column (``prm_score``) whichever model produced them,
    so every label the reader sees has to come from the recorded scorer kind
    rather than being hardcoded -- an ORM score captioned "step reward" would
    describe a measurement that never took place.
    """
    if (summary.get("scorer_kind") or "prm") == "orm":
        return {
            "abbr": "ORM",
            "heading": "Outcome reward analysis",
            "model": summary.get("scorer_model") or "Llama3.1-8B-ORM-Deepseek-Data",
            "note": "mean outcome reward",
            "lede": ("Does the outcome reward predict whether the final answer is "
                     "right? The ORM is shown the response in full, boxed answer "
                     "included, so this measures its usefulness for ranking "
                     "candidate solutions &mdash; not its judgement of the reasoning."),
            "footnote": ("a generative outcome reward model: the score is the "
                         "probability it assigns to a &ldquo;+&rdquo; verdict on "
                         "the solution as a whole."),
        }
    return {
        "abbr": "PRM",
        "heading": "Process reward analysis",
        "model": summary.get("scorer_model") or "Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
        "note": "mean step reward",
        "lede": ("Does the step-level reward predict whether the final answer is "
                 "right? This is the question that justifies spending GPU time on "
                 "the PRM at all."),
        "footnote": ("a process reward model trained on PRM800K step-level "
                     "human labels."),
    }


def _prm_section(summary: dict, rows: list[dict]) -> str:
    cal = summary.get("prm_calibration") or {}
    labels = _scorer_labels(summary)
    abbr = labels["abbr"]
    if not cal.get("n"):
        return (f'<section><h2>{labels["heading"]}</h2>'
                '<p class="empty">No reward scores in this run (scoring was disabled '
                'with <code>--no-prm</code>).</p></section>')

    correct = [r["prm_score"] for r in rows
               if r.get("prm_score") is not None and r.get("answer_accuracy")]
    wrong = [r["prm_score"] for r in rows
             if r.get("prm_score") is not None and not r.get("answer_accuracy")]

    auc = cal.get("auc")
    ci = cal.get("auc_ci")
    beats_chance = cal.get("auc_beats_chance")
    n_wrong = cal.get("n_incorrect")

    if auc is None:
        verdict = "Only one outcome class present, so discrimination is undefined."
    elif beats_chance is None:
        # The interval excludes 0.5, but it rests on too few solutions in the
        # smaller class for the approximation behind it to mean anything.
        verdict = (
            f"<strong>Too few incorrect solutions to judge.</strong> The AUC point "
            f"estimate is {num(auc)} and the 95% interval "
            f"[{num(ci[0], 2)}, {num(ci[1], 2)}] happens to exclude 0.5, but with only "
            f"{n_wrong} scored incorrect solution(s) that interval is a normal "
            f"approximation resting on almost no evidence. Treat the reward as "
            f"unvalidated until a harder question set or a weaker model produces "
            f"enough failures to rank against."
        )
    elif beats_chance is False:
        # The interval, not the point estimate, decides. Reading 0.59 as a real
        # effect when the interval spans 0.5 is exactly the over-claim this
        # analysis exists to prevent.
        verdict = (
            f"<strong>Not distinguishable from chance at this sample size.</strong> "
            f"The AUC point estimate is {num(auc)}, but with only {n_wrong} incorrect "
            f"solution(s) to rank against, the 95% interval "
            f"[{num(ci[0], 2)}, {num(ci[1], 2)}] straddles 0.5. More failures &mdash; "
            f"a harder question set or a weaker model &mdash; are needed before this "
            f"reward can be called informative."
        )
    elif auc >= 0.7:
        verdict = (f"The {abbr} separates correct from incorrect solutions well "
                   "enough to be useful for reranking, and the interval excludes "
                   "chance.")
    elif auc > 0.5:
        verdict = (f"The {abbr} carries a weak but statistically real signal; "
                   "best-of-n reranking may help marginally.")
    else:
        verdict = (f"The {abbr} ranks incorrect solutions <em>above</em> correct "
                   "ones more often than not &mdash; worse than chance.")

    stats = [
        ("AUC", num(auc), "P(correct scored above incorrect)"),
        ("AUC 95% CI", f"[{num(ci[0], 2)}, {num(ci[1], 2)}]" if ci else "n/a",
         f"{cal.get('n_correct', 0)} correct vs {n_wrong} incorrect"),
        ("Point-biserial r", num(cal.get("point_biserial_r")), "score vs correctness"),
        (f"Mean {abbr} | correct", num(cal.get("mean_prm_correct")), ""),
        (f"Mean {abbr} | incorrect", num(cal.get("mean_prm_incorrect")), ""),
        ("Separation", num(cal.get("separation")), "correct minus incorrect"),
        ("Calibration error", num(cal.get("expected_calibration_error")), "ECE"),
    ]
    table = "".join(
        f"<tr><th>{label}</th><td class='mono'>{value}</td>"
        f"<td class='dim'>{note}</td></tr>"
        for label, value, note in stats
    )

    return f"""<section>
  <h2>{labels["heading"]}</h2>
  <p class="lede">{labels["lede"]}</p>
  <p class="verdict">{verdict}</p>
  <div class="grid-2">
    <figure><figcaption>Reliability curve &mdash; dot size is bin population</figcaption>
      {calibration_chart(cal.get("bins", []))}</figure>
    <figure><figcaption>Score distribution &mdash;
      <span class="key correct">correct</span>
      <span class="key incorrect">incorrect</span></figcaption>
      {score_histogram(correct, wrong)}</figure>
  </div>
  <table class="stats">{table}</table>
</section>"""


def _selection_section(summary: dict) -> str:
    selection = summary.get("selection_accuracy") or {}
    if not selection or summary.get("k", 1) <= 1:
        return ""
    abbr = _scorer_labels(summary)["abbr"]
    labels = {
        "first": "Single shot",
        "majority": "Majority vote",
        "prm_best": f"{abbr} best-of-n",
        "prm_weighted": f"{abbr}-weighted vote",
    }
    total = summary.get("num_questions", 0)
    items = [(labels.get(k, k), v, round(v * total), total) for k, v in selection.items()]
    baseline = selection.get("first")
    rows = "".join(
        f"<tr><th>{labels.get(k, k)}</th><td class='mono'>{pct(v, 1)}</td>"
        f"<td class='mono {'pos' if baseline is not None and v > baseline else ('neg' if baseline is not None and v < baseline else 'dim')}'>"
        f"{'' if baseline is None else f'{(v - baseline) * 100:+.1f} pts'}</td></tr>"
        for k, v in selection.items()
    )
    return f"""<section>
  <h2>Answer selection strategies (k={summary['k']})</h2>
  <p class="lede">With {summary['k']} samples per question at temperature
  {summary.get('temperature')}, the same generations can be resolved into a final
  answer four different ways. Everything is measured against the single-shot baseline.</p>
  <div class="grid-2">
    <figure>{bar_chart(items, "Selection strategy accuracy")}</figure>
    <table class="stats"><tr><th>Strategy</th><th>Accuracy</th><th>vs baseline</th></tr>{rows}</table>
  </div>
</section>"""


def _comparison_section(comparison: dict | None) -> str:
    if not comparison or not comparison.get("backends"):
        return ""
    backends = comparison["backends"]
    acc_rows = "".join(
        f"<tr><th>{esc(b)}</th><td class='mono'>{pct(comparison['accuracy'].get(b), 1)}</td></tr>"
        for b in backends
    )
    agree_rows = "".join(
        f"<tr><th>{esc(pair.replace('|', ' vs '))}</th><td class='mono'>{pct(v, 1)}</td></tr>"
        for pair, v in comparison.get("agreement", {}).items()
    )

    divergences = comparison.get("divergences", [])[:40]
    div_rows = []
    for d in divergences:
        cells = "".join(
            f"<td class='{'pos' if d['outcomes'].get(b) else 'neg'}'>"
            f"{'correct' if d['outcomes'].get(b) else 'wrong'}"
            f"<span class='mono dim'> {esc(d['predicted'].get(b, ''))[:40]}</span></td>"
            for b in backends
        )
        div_rows.append(
            f"<tr><td class='mono'>#{d['index'] + 1}</td>"
            f"<td>{esc(d.get('subject') or '')} L{esc(d.get('difficulty') or '')}</td>"
            f"<td class='mono'>{esc(d.get('expected', ''))[:24]}</td>{cells}</tr>"
        )

    head = "".join(f"<th>{esc(b)}</th>" for b in backends)
    return f"""<section>
  <h2>Backend comparison</h2>
  <p class="lede">Same {comparison.get('num_questions', 0)} questions through every
  backend, so differences are attributable to the model rather than to the sample.
  {comparison.get('num_divergences', 0)} question(s) diverged.</p>
  <div class="grid-2">
    <table class="stats"><tr><th>Backend</th><th>Accuracy</th></tr>{acc_rows}</table>
    <table class="stats"><tr><th>Pair</th><th>Outcome agreement</th></tr>{agree_rows}</table>
  </div>
  <h3>Divergent questions</h3>
  <div class="scroll"><table class="rows">
    <tr><th>#</th><th>Topic</th><th>Expected</th>{head}</tr>{"".join(div_rows)}
  </table></div>
</section>"""


def _steps_html(row: dict) -> str:
    steps = row.get("steps") or []
    scores = row.get("step_prm_scores") or []
    if not steps:
        return "<p class='empty'>No reasoning steps were extracted.</p>"
    out = []
    for i, step in enumerate(steps):
        score = scores[i] if i < len(scores) else None
        badge = (
            f"<span class='score' style=\"background:{score_color(score)}\">"
            f"{num(score, 2)}</span>"
        ) if score is not None else ""
        out.append(f"<li>{badge}<span class='step'>{esc(step)}</span></li>")
    return f"<ol class='steps'>{''.join(out)}</ol>"


def _questions_section(rows: list[dict], abbr: str = "PRM") -> str:
    items = []
    for r in rows:
        ok = bool(r.get("answer_accuracy"))
        flags = []
        if not r.get("parse_matched"):
            flags.append("<span class='flag'>parse fallback</span>")
        selection = r.get("selection") or {}
        if selection and r.get("k", 1) > 1:
            flags.append("<span class='flag'>" + " ".join(
                f"{s}:{'Y' if v['correct'] else 'N'}" for s, v in selection.items()
            ) + "</span>")
        items.append(f"""<details class="q {'ok' if ok else 'bad'}">
  <summary>
    <span class="pill">{'correct' if ok else 'wrong'}</span>
    <span class="qmeta">#{r.get('index', 0) + 1} &middot; {esc(r.get('subject') or '?')}
      &middot; level {esc(r.get('difficulty') or '?')}</span>
    <span class="qans mono">got {esc(r.get('predicted', ''))[:60]}
      &middot; want {esc(r.get('expected', ''))[:60]}</span>
    <span class="qprm mono" style="color:{score_color(r.get('prm_score'))}">
      {abbr} {num(r.get('prm_score'), 2)}</span>
    {''.join(flags)}
  </summary>
  <div class="qbody">
    <p class="problem">{esc(r.get('problem', ''))}</p>
    {_steps_html(r)}
  </div>
</details>""")
    return f"""<section>
  <h2>Questions <span class="dim">({len(rows)})</span></h2>
  <p class="lede">{"Each step is shaded by its PRM reward. A run of green steps "
  "ending in a wrong answer is the interesting case: it localises where the "
  "reasoning actually broke down." if abbr == "PRM" else
  "The ORM scores each solution as a whole, so there is no per-step shading "
  "here &mdash; the steps are shown as extracted, unscored."}</p>
  {''.join(items)}
</section>"""


STYLE = """
:root {
  --bg: #ffffff; --fg: #14171f; --muted: #6b7280; --line: #e4e7ec;
  --panel: #f8fafc; --accent: #2563eb; --pos: #16a34a; --neg: #dc2626;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1218; --fg:#e6e8ec; --muted:#9aa3b2; --line:#252b36;
          --panel:#161b23; --accent:#60a5fa; --pos:#4ade80; --neg:#f87171; }
}
:root[data-theme="dark"] { --bg:#0f1218; --fg:#e6e8ec; --muted:#9aa3b2; --line:#252b36;
  --panel:#161b23; --accent:#60a5fa; --pos:#4ade80; --neg:#f87171; }
:root[data-theme="light"] { --bg:#ffffff; --fg:#14171f; --muted:#6b7280; --line:#e4e7ec;
  --panel:#f8fafc; --accent:#2563eb; --pos:#16a34a; --neg:#dc2626; }

body { background: var(--bg); color: var(--fg); margin: 0;
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 25px; margin: 0 0 4px; letter-spacing: -0.02em; }
h2 { font-size: 18px; margin: 0 0 6px; letter-spacing: -0.01em; }
h3 { font-size: 15px; margin: 22px 0 8px; color: var(--muted); }
section { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 22px; }
.sub { color: var(--muted); font-size: 13.5px; }
.lede { color: var(--muted); margin: 0 0 16px; max-width: 68ch; }
.verdict { background: var(--panel); border-left: 3px solid var(--accent);
  padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 0 0 18px; max-width: 78ch; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
.dim { color: var(--muted); }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.empty { color: var(--muted); font-style: italic; }
.banner { background: #fef3c7; color: #78350f; border-radius: 8px;
  padding: 10px 14px; margin: 16px 0 0; font-size: 14px; }

.cards { display: grid; gap: 12px; margin-top: 22px;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 13px 15px; }
.card-label { font-size: 12px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.05em; }
.card-value { font-size: 25px; font-weight: 600; margin: 3px 0 1px;
  letter-spacing: -0.02em; }
.card-note { font-size: 12px; color: var(--muted); }

.grid-2 { display: grid; gap: 22px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
  align-items: start; }
figure { margin: 0; }
figcaption { font-size: 12.5px; color: var(--muted); margin-bottom: 6px; }
svg { width: 100%; height: auto; overflow: visible; }
.grid { stroke: var(--line); stroke-width: 1; }
.axis { stroke: var(--line); stroke-width: 1.5; }
.tick { fill: var(--muted); font-size: 11px;
  font-family: ui-monospace, Menlo, monospace; }
.axis-label { fill: var(--muted); font-size: 11.5px; }
.bar { fill: var(--accent); }
.bar-value { fill: var(--fg); font-size: 11px; font-weight: 600; }
.cell-text { fill: #fff; font-size: 11px; font-weight: 600;
  font-family: ui-monospace, Menlo, monospace; }
.cell-empty { fill: var(--panel); stroke: var(--line); }
.diagonal { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; opacity: .6; }
.series { fill: none; stroke: var(--accent); stroke-width: 2; }
.point { fill: var(--accent); }
.key { padding: 1px 7px; border-radius: 99px; color: #fff; font-size: 11px; }
.key.correct { background: #16a34a; } .key.incorrect { background: #dc2626; }

table { border-collapse: collapse; width: 100%; }
table.stats th { text-align: left; font-weight: 500; color: var(--muted);
  padding: 6px 10px 6px 0; font-size: 13.5px; white-space: nowrap; }
table.stats td { padding: 6px 10px 6px 0; }
table.rows { font-size: 13px; }
table.rows th, table.rows td { text-align: left; padding: 6px 10px 6px 0;
  border-bottom: 1px solid var(--line); vertical-align: top; }
.scroll { overflow-x: auto; }

details.q { border: 1px solid var(--line); border-radius: 8px; margin-bottom: 7px;
  background: var(--panel); }
details.q summary { cursor: pointer; padding: 9px 13px; display: flex;
  flex-wrap: wrap; gap: 10px; align-items: baseline; font-size: 13.5px; }
details.q[open] summary { border-bottom: 1px solid var(--line); }
.pill { font-size: 11px; padding: 1px 8px; border-radius: 99px; color: #fff; }
.q.ok .pill { background: var(--pos); } .q.bad .pill { background: var(--neg); }
.qmeta { color: var(--muted); }
.qans { color: var(--fg); }
.flag { font-size: 11px; color: var(--muted); border: 1px solid var(--line);
  padding: 0 6px; border-radius: 4px; }
.qbody { padding: 12px 15px 16px; }
.problem { background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px 12px; font-family: ui-monospace, Menlo, monospace; font-size: 12.5px;
  white-space: pre-wrap; overflow-x: auto; }
ol.steps { padding-left: 0; list-style: none; margin: 12px 0 0; }
ol.steps li { display: flex; gap: 9px; padding: 5px 0; align-items: flex-start;
  border-top: 1px solid var(--line); }
.score { color: #fff; font-size: 11px; font-family: ui-monospace, Menlo, monospace;
  padding: 1px 6px; border-radius: 4px; min-width: 34px; text-align: center;
  flex-shrink: 0; margin-top: 2px; }
.step { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px;
  word-break: break-word; }
"""


def render_report(report: dict) -> str:
    summary, rows = report["summary"], report["rows"]
    labels = _scorer_labels(summary)

    difficulty_items = [
        (f"L{k}", v["accuracy"], v["correct"], v["total"])
        for k, v in (summary.get("by_difficulty") or {}).items()
    ]
    subject_items = [
        (k[:12], v["accuracy"], v["correct"], v["total"])
        for k, v in (summary.get("by_subject") or {}).items()
    ]

    banner = ""
    if summary.get("synthetic"):
        banner = ('<p class="banner"><strong>Synthetic data.</strong> These results '
                  'come from the mock backend, a deterministic test double used to '
                  'exercise the pipeline. They are not a measurement of any model.</p>')

    meta = (f"{esc(summary.get('model', '?'))} &middot; "
            f"{summary.get('num_questions', len(rows))} questions &middot; "
            f"k={summary.get('k', 1)} &middot; "
            f"temperature {summary.get('temperature', 0)} &middot; "
            f"{labels['abbr']} {'on' if summary.get('prm_enabled') else 'off'} &middot; "
            f"{esc(summary.get('timestamp') or datetime.now().isoformat(timespec='seconds'))}")

    return f"""<title>MATH-500 evaluation report</title>
<style>{STYLE}</style>
<div class="wrap">
  <header>
    <h1>MATH-500 evaluation report</h1>
    <p class="sub">{meta}</p>
    {banner}
    <div class="cards">{_stat_cards(summary, rows)}</div>
  </header>

  <section>
    <h2>Where the model fails</h2>
    <p class="lede">A single accuracy number hides the shape of the failures.
    MATH-500 ships subject and difficulty labels, so the same run answers
    "how good" and "at what" simultaneously.</p>
    <div class="grid-2">
      <figure><figcaption>Accuracy by difficulty level</figcaption>
        {bar_chart(difficulty_items, "Accuracy by difficulty")}</figure>
      <figure><figcaption>Accuracy by subject</figcaption>
        {bar_chart(subject_items, "Accuracy by subject")}</figure>
    </div>
    <h3>Subject &times; difficulty</h3>
    <div class="scroll">{heatmap(summary.get('subject_by_difficulty', {}), 'Subject by difficulty')}</div>
  </section>

  {_prm_section(summary, rows)}
  {_selection_section(summary)}
  {_comparison_section(report.get("comparison"))}
  {_questions_section(rows, labels["abbr"])}

  <section>
    <p class="sub">Generated by <span class="mono">math500_eval.py</span>.
    Solutions are scored by
    <span class="mono">{esc(labels['model'])}</span>,
    {labels['footnote']}</p>
  </section>
</div>"""


def write_report(report: dict, path: str = "report.html") -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(render_report(report))
    return path
