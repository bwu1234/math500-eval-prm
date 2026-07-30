"""Backend contract tests, including the retry loop and lazy-import guarantee."""

import sys
import urllib.error

import pytest

from evalkit.backends import (
    Backend,
    GeminiBackend,
    Generation,
    MockBackend,
    OllamaBackend,
    build_backend,
)


def test_mock_is_deterministic():
    """Same prompt, same result -- otherwise checkpoint resume is unverifiable."""
    a = MockBackend(accuracy=0.6)
    b = MockBackend(accuracy=0.6)
    ctx = {"answer": "12"}
    assert a.generate("solve x", context=ctx).text == b.generate("solve x", context=ctx).text


def test_mock_varies_with_temperature():
    m = MockBackend()
    ctx = {"answer": "12"}
    assert m.generate("p", temperature=0.0, context=ctx).text != \
           m.generate("p", temperature=0.7, context=ctx).text


def test_mock_samples_differ_above_temperature_zero():
    """Otherwise k>1 would be tested with k identical samples and self-
    consistency would look like a no-op for reasons unrelated to the method."""
    m = MockBackend()
    texts = {m.generate("p", temperature=0.8,
                        context={"answer": "12", "sample_index": i}).text
             for i in range(5)}
    assert len(texts) > 1


def test_mock_samples_agree_at_temperature_zero():
    m = MockBackend()
    texts = {m.generate("p", temperature=0.0,
                        context={"answer": "12", "sample_index": i}).text
             for i in range(5)}
    assert len(texts) == 1


def test_mock_emits_a_parsable_boxed_answer():
    gen = MockBackend(accuracy=1.0).generate("p", context={"answer": "12"})
    from evalkit.answers import extract_final_answer
    assert extract_final_answer(gen.text) == ("12", True)


def test_mock_hits_its_target_accuracy():
    from evalkit.answers import extract_final_answer
    m = MockBackend(accuracy=0.6)
    hits = 0
    n = 300
    for i in range(n):
        gen = m.generate(f"problem {i}", context={"answer": str(i)})
        hits += extract_final_answer(gen.text)[0] == str(i)
    assert 0.5 < hits / n < 0.7


def test_null_token_counts_become_zero():
    """Regression: Gemini reports candidates_token_count=None when a response
    is truncated at MAX_TOKENS having emitted no text. Left as None it
    propagates into the results file and blows up aggregation at the very end
    of a run -- after every question has been paid for."""
    gen = Generation(text="", prompt_tokens=None, output_tokens=None, total_tokens=None)
    assert (gen.prompt_tokens, gen.output_tokens, gen.total_tokens) == (0, 0, 0)
    assert gen.prompt_tokens + gen.output_tokens == 0  # must be summable


def test_real_token_counts_are_preserved():
    gen = Generation(text="hi", prompt_tokens=7, output_tokens=3, total_tokens=10)
    assert (gen.prompt_tokens, gen.output_tokens, gen.total_tokens) == (7, 3, 10)


def test_gemini_truncated_response_yields_summable_tokens(monkeypatch):
    """End-to-end shape of the MAX_TOKENS case, through the real backend."""
    class Usage:
        prompt_token_count = 391
        candidates_token_count = None   # what Gemini actually sends
        total_token_count = 16772

    class Candidate:
        finish_reason = "FinishReason.MAX_TOKENS"
        finish_message = None

    class Resp:
        text = ""
        usage_metadata = Usage()
        candidates = [Candidate()]

    b = GeminiBackend.__new__(GeminiBackend)   # bypass SDK/client construction
    Backend.__init__(b, model="gemma-4-31b-it")
    b._types = type("T", (), {"GenerateContentConfig": lambda **kw: None})
    b._transient = ()
    b._client = type("C", (), {"models": type("M", (), {
        "generate_content": staticmethod(lambda **kw: Resp())})()})()

    gen = b.generate("p")
    assert gen.output_tokens == 0
    assert "MAX_TOKENS" in gen.warning
    assert sum([gen.output_tokens, gen.prompt_tokens]) == 391


def test_build_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend("gpt-9")


def test_build_mock_backend():
    assert isinstance(build_backend("mock"), MockBackend)


# ── Retry loop ────────────────────────────────────────────────────────
class Boom(Exception):
    pass


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    b = Backend(model="m", retries=5)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom("transient")
        return Generation(text="ok")

    assert b._with_retries(flaky, (Boom,)).text == "ok"
    assert calls["n"] == 3


def test_retries_give_up_with_a_clear_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    b = Backend(model="m", retries=3)

    def always_fails():
        raise Boom("nope")

    with pytest.raises(RuntimeError, match="failed after 3 retries"):
        b._with_retries(always_fails, (Boom,))


def test_unexpected_errors_are_not_retried(monkeypatch):
    """A bug in our own code must surface immediately, not after 8 backoffs."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    b = Backend(model="m", retries=8)
    calls = {"n": 0}

    def type_error():
        calls["n"] += 1
        raise TypeError("programmer error")

    with pytest.raises(TypeError):
        b._with_retries(type_error, (Boom,))
    assert calls["n"] == 1


def test_ollama_reports_token_counts(monkeypatch):
    import json as _json

    class FakeResponse:
        def read(self):
            return _json.dumps({"response": "hi", "prompt_eval_count": 7,
                                "eval_count": 3}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())
    gen = OllamaBackend(model="m").generate("p")
    assert (gen.prompt_tokens, gen.output_tokens, gen.total_tokens) == (7, 3, 10)


def test_ollama_flags_an_empty_response(monkeypatch):
    import json as _json

    class FakeResponse:
        def read(self):
            return _json.dumps({"response": "", "done_reason": "length"}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())
    assert "length" in OllamaBackend(model="m").generate("p").warning


# ── Lazy imports ──────────────────────────────────────────────────────
def test_grading_and_reporting_do_not_require_the_ml_stack():
    """The point of the module split: the grader must be testable without torch.

    If this regresses, the fast unit suite starts requiring a GPU-class
    dependency install and stops being run.
    """
    import subprocess
    code = (
        "import sys;"
        "sys.modules['torch'] = None;"
        "sys.modules['transformers'] = None;"
        "sys.modules['datasets'] = None;"
        "import evalkit.answers, evalkit.analysis, evalkit.report, evalkit.backends;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
