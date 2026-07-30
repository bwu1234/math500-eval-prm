import pytest

from evalkit.backends import MockBackend


@pytest.fixture
def problems():
    """A small synthetic problem set with the same shape as MATH-500 rows."""
    subjects = ["Algebra", "Geometry", "Prealgebra"]
    return [
        {
            "index": i,
            "problem": f"Problem {i}: compute the value of x.",
            "answer": str(i * 3),
            "subject": subjects[i % len(subjects)],
            "level": (i % 5) + 1,
        }
        for i in range(12)
    ]


class FakeScorer:
    """Stand-in for the PRM.

    Emits a deterministic score per step so the pipeline, aggregation and
    report code can be tested without loading a 1.5B model. It is a fixture,
    never a source of reported numbers.
    """

    enabled = True

    def __init__(self, base: float = 0.5):
        self.base = base
        self.calls = 0

    def step_scores(self, problem, steps):
        self.calls += 1
        return [min(0.99, self.base + 0.05 * i) for i in range(len(steps))]

    def score(self, problem, steps):
        scores = self.step_scores(problem, steps)
        return sum(scores) / len(scores) if scores else None


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def backend():
    return MockBackend(accuracy=0.6)
