import os
import re
import json
import math
import time
import random
import ctypes
import glob
import sysconfig
import shutil
import urllib.request
import urllib.error
from datetime import datetime
from collections import defaultdict
from fractions import Fraction

import sympy as sp


def _preload_cuda_libs():
    """Preload CUDA libs that bitsandbytes (built for CUDA 13) needs.

    libnvJitLink.so.13 ships inside PyTorch's pip CUDA wheels but isn't on the
    default loader path, so bitsandbytes' native extension fails to dlopen it.
    Loading it RTLD_GLOBAL here makes its symbols available process-wide without
    requiring LD_LIBRARY_PATH. No-op (best effort) if the lib isn't present.
    """
    site = sysconfig.get_paths()["purelib"]
    for path in glob.glob(os.path.join(site, "nvidia/cu13/lib/libnvJitLink.so*")):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_cuda_libs()

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset
from google import genai
from google.genai import types, errors

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY environment variable")

GEMINI_MODEL = "gemma-4-31b-it"
QWEN_MODEL = "qwen3.6:35b-mlx"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PRM_MODEL = "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate(prompt: str, max_tokens: int = 2048, retries: int = 8, model: str = GEMINI_MODEL) -> dict:
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0,
                ),
            )
            text = resp.text or ""
            usage = resp.usage_metadata
            tokens = usage.candidates_token_count if usage else 0
            prompt_tokens = usage.prompt_token_count if usage else 0
            total_tokens = usage.total_token_count if usage else 0
            if not text and resp.candidates:
                c = resp.candidates[0]
                log(f"  [WARN] empty response | finish_reason={c.finish_reason} blocked={c.finish_message}")
            return {"full_text": text, "eval_count": tokens, "prompt_tokens": prompt_tokens, "total_tokens": total_tokens}
        except errors.APIError as e:
            wait = (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"  [{getattr(e,'code','?')}] {type(e).__name__}; retry {attempt+1}/{retries} in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Still hitting quota after {retries} retries")

def generate_ollama(prompt: str, max_tokens: int = 2048, retries: int = 8, model: str = QWEN_MODEL) -> dict:
    """Generate via a local Ollama server's /api/generate endpoint."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body.get("response", "")
            tokens = body.get("eval_count", 0)
            prompt_tokens = body.get("prompt_eval_count", 0)
            if not text:
                log(f"  [WARN] empty response from Ollama model={model} | done_reason={body.get('done_reason')}")
            return {"full_text": text, "eval_count": tokens, "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens + tokens}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            wait = (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"  [Ollama] {type(e).__name__}: {e}; retry {attempt+1}/{retries} in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Still failing against Ollama ({url}) after {retries} retries")

# ── Model backend registry ──────────────────────────────────────────────
MODEL_BACKENDS = {
    "gemini": {"generate": generate, "model": GEMINI_MODEL},
    "qwen": {"generate": generate_ollama, "model": QWEN_MODEL},
}

VERBOSE = True
LOG_FILE = "eval_debug.log"
LOG_DIR = "logs"

def log(msg: str = ""):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ── PRM evaluator ─────────────────────────────────────────────────────
# A real Process Reward Model. Qwen2.5-Math-1.5B-Instruct-PRM-0.2 is a
# token-classification model trained (TRL stepwise-reward) on PRM800K:
# reasoning steps are concatenated with a "\n\n" separator, and the reward
# for a step is P(LABEL_1) at that step's trailing separator token.
# We load it 4-bit so it fits a 4 GB GPU. No prompting, no answer parsing.
PRM_SEP = "\n\n"  # must match the separator used during training

_prm_pipe = None

def _load_prm():
    """Lazily load the PRM pipeline once (4-bit quantized)."""
    global _prm_pipe
    if _prm_pipe is None:
        if VERBOSE: log(f"    [PRM] loading {PRM_MODEL} (4-bit) ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tok = AutoTokenizer.from_pretrained(PRM_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(
            PRM_MODEL,
            quantization_config=bnb,
            device_map="auto",
        ).eval()
        _prm_pipe = pipeline("token-classification", model=model, tokenizer=tok)
    return _prm_pipe

def prm_step_scores(problem: str, steps: list[str]) -> list[float]:
    """Score each step on the cumulative prefix (problem + steps so far), reading
    the reward at the trailing separator token — exactly the model-card recipe."""
    if not steps:
        return []
    pipe = _load_prm()
    step_scores = []
    for idx in range(1, len(steps) + 1):
        text = PRM_SEP.join((problem, *steps[:idx])) + PRM_SEP
        last = pipe(text)[-1]
        # pipeline gives the predicted label + its prob; convert to P(correct)
        p_correct = last["score"] if last["entity"] == "LABEL_1" else 1.0 - last["score"]
        step_scores.append(float(p_correct))
    return step_scores

def prm_score(problem: str, steps: list[str]) -> float:
    step_scores = prm_step_scores(problem, steps)
    if not step_scores:
        if VERBOSE: log(f"    [PRM] no steps extracted")
        return 0.0
    avg = sum(step_scores) / len(step_scores)
    if VERBOSE:
        log(f"    [PRM] {len(step_scores)} step scores: "
            f"{[round(s, 3) for s in step_scores]}, avg: {avg:.3f}")
    return avg

# ── Answer normalisation ──────────────────────────────────────────────
LATEX_SYMBOLS = {
    r"\pi": "pi", r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\theta": "theta", r"\phi": "phi", r"\varphi": "phi",
    r"\omega": "omega", r"\sigma": "sigma", r"\mu": "mu", r"\tau": "tau",
    r"\epsilon": "epsilon", r"\lambda": "lambda", r"\infty": "inf",
    r"\partial": "d", r"\rightarrow": "to", r"\Rightarrow": "to",
    r"\approx": "~=", r"\neq": "!=", r"\leq": "<=", r"\geq": ">=",
}

def normalize_answer(ans: str) -> str:
    ans = re.sub(r"\\boxed\{(.*?)\}", r"\1", ans, flags=re.DOTALL)
    ans = re.sub(r"\^?\\circ|\^?\\degree|°", "", ans)
    for cmd, text in LATEX_SYMBOLS.items():
        ans = ans.replace(cmd, text)
    # Remove remaining formatting commands (e.g. \left, \right, \frac, \text, \quad)
    ans = re.sub(r"\\[a-zA-Z]+", "", ans)
    ans = re.sub(r"[{}]", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("/", "")
    return ans.strip().lower()

def strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    wrappers = [
        (r"^\\boxed\{(.+)\}$", 1),
        (r"^\\text\{(.+)\}$", 1),
        (r"^\$(.+)\$$", 1),
    ]
    changed = True
    while changed:
        changed = False
        for pattern, group_idx in wrappers:
            m = re.match(pattern, s)
            if m:
                s = m.group(group_idx).strip()
                changed = True
    return s

def normalize_latex_frac(s: str) -> str:
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\tfrac", r"\\frac", s)

    def repl(match):
        return f"({match.group(1)})/({match.group(2)})"

    return re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", repl, s)

def normalize_text(s: str) -> str:
    s = s.strip()
    s = strip_latex_wrappers(s)
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("\u2212", "-")
    s = s.replace("\u2013", "-")
    s = re.sub(r"\^?\\circ|\^?\\degree|\u00b0", "", s)
    s = s.replace("^", "**")
    s = s.replace("\\left", "").replace("\\right", "")
    s = normalize_latex_frac(s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\\!", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip().rstrip(".")
    return s

def try_parse_fraction(s: str):
    s = normalize_text(s)
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", s):
        a, b = s.split("/")
        try:
            return Fraction(int(a.strip()), int(b.strip()))
        except Exception:
            return None
    return None

def try_parse_number(s: str):
    s = normalize_text(s)
    frac = try_parse_fraction(s)
    if frac is not None:
        return float(frac)
    try:
        return float(s)
    except Exception:
        pass
    try:
        expr = sp.sympify(s)
        if expr.is_real and expr.is_number:
            return float(expr.evalf())
    except Exception:
        pass
    return None

def sanitize_for_sympy(s: str) -> str:
    s = normalize_text(s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\sqrt\((.+)\)", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    return s

def try_sympy_parse(s: str):
    s = sanitize_for_sympy(s)
    try:
        return sp.sympify(s)
    except Exception:
        return None

def are_equivalent(pred: str, gold: str, tol: float = 1e-9) -> bool:
    """Numeric/symbolic fallback equivalence check (handles 1/2 vs 0.5 vs
    \\frac{1}{2}, algebraically-equal expressions, etc.) for cases the plain
    string normalization in normalize_answer() doesn't catch."""
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

    pred_expr = try_sympy_parse(pred_n)
    gold_expr = try_sympy_parse(gold_n)
    if pred_expr is not None and gold_expr is not None:
        try:
            if sp.simplify(pred_expr - gold_expr) == 0:
                return True
        except Exception:
            pass
        try:
            return sp.simplify(sp.Eq(pred_expr, gold_expr)) is True
        except Exception:
            pass

    return False

def final_answer_correct(expected: str, actual: str) -> bool:
    if normalize_answer(expected) == normalize_answer(actual):
        return True
    return are_equivalent(actual, expected)

# ── Extract answer from agent output ──────────────────────────────────
def extract_final_answer(text: str) -> tuple[str, bool]:
    """Returns (answer, matched) where matched=False means no structured
    pattern was found and we fell back to the last line — a weak signal
    worth tracking separately as a parse failure."""
    # 1) \boxed{...} (may have nested braces — match balanced pairs)
    m = re.search(r"\\boxed\{(.*)\}", text, re.DOTALL)
    if m:
        return m.group(1).strip(), True
    # 2) Last \( ... \) inline math expression
    m = re.findall(r"\\\((.*?)\\\)", text, re.DOTALL)
    if m:
        return m[-1].strip(), True
    # 3) Last \[ ... \] display math expression
    m = re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)
    if m:
        return m[-1].strip(), True
    # 4) "answer is ..." pattern
    m = re.search(r"(?:answer|result|equals)\s+(?:is\s+)?(.+?)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        ans = m.group(1).strip().rstrip(".")
        # Remove trailing \( or \[
        ans = re.sub(r"\\[\(\[]$", "", ans).strip()
        return ans, True
    # 5) Last non-empty line as fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[-1] if lines else text.strip()), False

# ── Split reasoning into steps ────────────────────────────────────────
def extract_steps(text: str) -> list[str]:
    """Split reasoning into steps, merging display math with adjacent text."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    steps = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("\\boxed"):
            i += 1
            continue
        if line.startswith("\\[") or line.startswith("\\]"):
            # standalone display math — skip it (equations without context aren't useful)
            i += 1
            continue
        # text line: merge any following display math
        merged = line
        i += 1
        while i < len(lines) and (lines[i].startswith("\\[") or lines[i].startswith("\\]")):
            merged += " " + lines[i]
            i += 1
        if len(merged) > 5:
            steps.append(merged)
    return steps

# ── Grouped accuracy ──────────────────────────────────────────────────
def compute_group_accuracy(rows: list[dict], group_key: str, correct_key: str = "answer_accuracy") -> dict:
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        value = row.get(group_key)
        if value is None:
            continue
        stats[str(value)]["total"] += 1
        stats[str(value)]["correct"] += int(row[correct_key])
    return {
        k: {
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for k, v in sorted(stats.items(), key=lambda x: x[0])
    }

# ── Main eval ─────────────────────────────────────────────────────────
def run_eval(num_questions: int = 5, model_backend: str = "gemini", question: int = None):
    if model_backend not in MODEL_BACKENDS:
        raise ValueError(f"Unknown model_backend: {model_backend!r} (expected one of {list(MODEL_BACKENDS)})")
    backend = MODEL_BACKENDS[model_backend]
    gen_fn, model_name = backend["generate"], backend["model"]

    # Archive the previous run's log + results instead of overwriting them,
    # unless we're only rerunning a single question (then append/merge into
    # the current run's files)
    if question is None:
        if os.path.exists(LOG_FILE) or os.path.exists("eval_results.json"):
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(LOG_FILE):
                shutil.move(LOG_FILE, os.path.join(LOG_DIR, f"eval_debug_{stamp}.log"))
            if os.path.exists("eval_results.json"):
                shutil.move("eval_results.json", os.path.join(LOG_DIR, f"eval_results_{stamp}.json"))
        open(LOG_FILE, "w").close()
    log(f"Model backend: {model_backend} ({model_name})")

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = [question - 1] if question is not None else range(num_questions)
    questions = ds.select(indices)
    results = []
    parse_failures = 0

    for i, row in enumerate(questions):
        qnum = question if question is not None else i + 1
        problem = row["problem"]
        expected = row["answer"]
        subject = row.get("subject")
        difficulty = row.get("level")

        prompt = (
            "Solve this math problem concisely. Briefly show your reasoning, "
            "then give the answer inside \\boxed{}.\n\n"
            f"{problem}"
        )

        log(f"\n{'='*60}")
        log(f"[{qnum}/{num_questions if question is None else question}] PROMPT (input):")
        for line in prompt.strip().split("\n"):
            log(f"    | {line}")
        log(f"  Expected answer: {expected}")
        log(f"  Subject: {subject} | Difficulty: {difficulty}")
        print(f"[{i+1}/{num_questions}] Solving...", flush=True)

        out = gen_fn(prompt, max_tokens=16384)
        full_text = out["full_text"]
        steps = extract_steps(full_text)
        if VERBOSE:
            log(f"  Tokens: prompt={out['prompt_tokens']} output={out['eval_count']} total={out['total_tokens']}")
            log(f"  Raw text ({out['eval_count']} tokens):")
            for line in full_text.strip().split("\n"):
                log(f"    | {line}")

        step_scores = prm_step_scores(problem, steps)
        p_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        if VERBOSE:
            log(f"  Intermediate steps ({len(steps)}) with PRM scores:")
            for j, s in enumerate(steps):
                score = step_scores[j] if j < len(step_scores) else float("nan")
                log(f"    [{j}] PRM={score:.3f} | {s}")
            log(f"  Avg PRM score: {p_score:.3f}")

        predicted, matched = extract_final_answer(full_text)
        answer_ok = final_answer_correct(expected, predicted)
        if not matched or not predicted:
            parse_failures += 1
        if VERBOSE:
            log(f"  Final answer extracted: {predicted!r} (structured match: {matched})")
            log(f"    normalized predicted: {normalize_answer(predicted)}")
            log(f"    normalized expected:  {normalize_answer(expected)}")
            log(f"  Answer correct: {answer_ok}")

        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "answer_accuracy": answer_ok,
            "parse_matched": matched,
            "subject": subject,
            "difficulty": difficulty,
            "prm_score": round(p_score, 3),
            "steps": steps,
            "step_prm_scores": [round(s, 3) for s in step_scores],
            "num_steps": len(steps),
            "prompt_tokens": out["prompt_tokens"],
            "eval_tokens": out["eval_count"],
            "total_tokens": out["total_tokens"],
        })

        print(f"  Tokens: {out['eval_count']} | Correct: {answer_ok} | PRM: {p_score:.2f}")

    if question is not None and os.path.exists("eval_results.json"):
        with open("eval_results.json") as f:
            summary = json.load(f)
        existing_results = summary["results"]
        pos = question - 1
        if pos < len(existing_results):
            existing_results[pos] = results[0]
        else:
            existing_results.extend([None] * (pos - len(existing_results)))
            existing_results.append(results[0])
        merged = [r for r in existing_results if r is not None]
        acc = sum(1 for r in merged if r["answer_accuracy"]) / len(merged)
        avg_prm = sum(r["prm_score"] for r in merged) / len(merged)
        parse_failures = sum(1 for r in merged if not r["parse_matched"] or not r["predicted"])
        summary.update({
            "accuracy": acc,
            "avg_prm_score": round(avg_prm, 3),
            "parse_failures": parse_failures,
            "parse_failure_rate": round(parse_failures / len(merged), 3),
            "by_subject": compute_group_accuracy(merged, "subject"),
            "by_difficulty": compute_group_accuracy(merged, "difficulty"),
            "results": existing_results,
        })
    else:
        acc = sum(1 for r in results if r["answer_accuracy"]) / len(results)
        avg_prm = sum(r["prm_score"] for r in results) / len(results)
        by_subject = compute_group_accuracy(results, "subject")
        by_difficulty = compute_group_accuracy(results, "difficulty")

        summary = {
            "num_questions": num_questions,
            "accuracy": acc,
            "avg_prm_score": round(avg_prm, 3),
            "parse_failures": parse_failures,
            "parse_failure_rate": round(parse_failures / len(results), 3),
            "by_subject": by_subject,
            "by_difficulty": by_difficulty,
            "model_backend": model_backend,
            "model": model_name,
            "results": results,
        }

    total = len(summary["results"])
    acc, avg_prm, parse_failures = summary["accuracy"], summary["avg_prm_score"], summary["parse_failures"]

    log(f"\n{'='*60}")
    log(f"SUMMARY  (model: {model_name})")
    log(f"  Accuracy:  {acc:.0%} ({int(round(acc * total))}/{total})")
    log(f"  Avg PRM:   {avg_prm:.3f}")
    log(f"  Parse failures: {parse_failures} ({parse_failures / total:.0%})")
    log(f"  By subject:    {summary['by_subject']}")
    log(f"  By difficulty: {summary['by_difficulty']}")

    print(f"\n{'='*60}")
    print(f"SUMMARY  (model: {model_name})")
    print(f"  Accuracy:  {acc:.0%} ({int(round(acc * total))}/{total})")
    print(f"  Avg PRM:   {avg_prm:.3f}")
    print(f"  Parse failures: {parse_failures} ({parse_failures / total:.0%})")
    print(f"  Results:   eval_results.json")
    print(f"  Debug log: {LOG_FILE}")

    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model on MATH-500.")
    parser.add_argument("-n", "--num-questions", type=int, default=3,
                         help="Number of MATH-500 questions to run (default: 3, max: 500)")
    parser.add_argument("-m", "--model", dest="model_backend", choices=list(MODEL_BACKENDS),
                         default="gemini", help="Which model backend to use (default: gemini)")
    parser.add_argument("-q", "--question", type=int, default=None,
                         help="Rerun only this 1-indexed MATH-500 question (e.g. 10 for the "
                              "10th problem shown in the log). Appends to the log and merges "
                              "the result into eval_results.json instead of overwriting them.")
    args = parser.parse_args()

    run_eval(args.num_questions, model_backend=args.model_backend, question=args.question)
