"""Synthetic objectives and a counting evaluator for GOMEA engine tests.

Everything here maximizes a float score: ``worse(a, b)`` is ``a < b``. The
evaluator implements ``pymoses.gomea.gom.Evaluator`` without any MOSES
machinery, so the engine can be exercised on classic GA benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

from pymoses.gomea.gom import Individual
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.util.rng import MT19937RandGen


def onemax(target_code: int) -> Callable[[tuple[int, ...]], float]:
    """Categorical OneMax: the number of fields equal to ``target_code``."""

    def f(g: tuple[int, ...]) -> float:
        return float(sum(1 for v in g if v == target_code))

    return f


def trap(k: int) -> Callable[[tuple[int, ...]], float]:
    """Concatenated deceptive trap over bits, block size ``k``: a block of
    all ones scores ``k``, otherwise ``k - 1 - ones``."""

    def f(g: tuple[int, ...]) -> float:
        if len(g) % k:
            raise ValueError(f"genotype length {len(g)} is not a multiple of {k}")
        total = 0
        for b in range(0, len(g), k):
            ones = sum(g[b : b + k])
            total += k if ones == k else k - 1 - ones
        return float(total)

    return f


class CountingEvaluator:
    """A ``pymoses.gomea.gom.Evaluator`` over a plain function.

    ``calls`` counts every scored genotype; with ``budget`` set, ``check``
    reports ``'global_budget'`` once ``calls >= budget`` and ``evaluate``
    raises ``BudgetExhausted`` without calling ``fn``. ``pkey`` defaults to
    ``None`` (no proven phenotype skips).
    """

    def __init__(
        self,
        fn: Callable[[tuple[int, ...]], float],
        budget: int | None = None,
        pkey: Callable[[tuple[int, ...]], Hashable | None] | None = None,
    ) -> None:
        self.fn = fn
        self.budget = budget
        self.pkey = pkey
        self.calls = 0
        self.noop_skips = 0
        self.phenotype_skips = 0
        self.evaluated: list[tuple[int, ...]] = []

    def check(self) -> str | None:
        if self.budget is not None and self.calls >= self.budget:
            return "global_budget"
        return None

    def evaluate(self, g: tuple[int, ...]) -> float:
        reason = self.check()
        if reason is not None:
            raise BudgetExhausted(reason)
        self.calls += 1
        self.evaluated.append(tuple(g))
        return self.fn(tuple(g))

    def phenotype_key(self, g: tuple[int, ...]) -> Hashable | None:
        return None if self.pkey is None else self.pkey(g)

    def worse(self, a: float, b: float) -> bool:
        return a < b

    def record_noop_skip(self) -> None:
        self.noop_skips += 1

    def record_phenotype_skip(self) -> None:
        self.phenotype_skips += 1

    def individual(self, g: tuple[int, ...]) -> Individual:
        """Evaluate ``g`` (counted) and wrap it as an ``Individual``."""
        g = tuple(g)
        return Individual(g, self.evaluate(g), self.phenotype_key(g))


def random_population(rng: MT19937RandGen, n: int, domains: list[int]) -> list[tuple[int, ...]]:
    """``n`` uniform random genotypes drawn with ``rng.randint_n``."""
    return [tuple(rng.randint_n(k) for k in domains) for _ in range(n)]
