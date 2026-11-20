"""NOT A PORT: the IT linkage study (BACKLOG 13.9). ``RegressionEvaluator``, the
``pymoses.gomea.gom.Evaluator`` of the stand-alone harness, and the scorers of the two genomes
(plan section 5.5).

**Score.** The score of a genotype is -MSE of its scaled model on the training rows (higher is
better), with the target scaled as the runner scales it (y' = 10 (y - mean) / sd). An invalid
model scores -inf; a NaN score never leaves a scorer (it becomes -inf).

* ``TemplateScorer``: linear scaling a + b f of the program output f (GP-GOMEA
  ``--linearscaling``), the least-squares (a, b) of ``sr_eval.linear_scaling``. See
  ``TemplateScorer``.
* ``ITScorer``: per-term least squares w0 + sum wi ti over the distinct terms (as
  ``--linear-scaling terms``), with the drop of constant and aliased columns. See ``ITScorer``.

Both scorers get the SSE from the least-squares identity (SSE = Syy - the explained sum of
squares), not from the fitted values: that is fast, and equal to the MSE of the fit up to
rounding. ``run.py`` scores the reported model again with the exact formulas of ``sr_eval``
(the copies ``linear_scaling``, ``nmse``, ``clip_range`` and ``nmse_clipped`` below).

**Budget and score cache.** ``evaluate`` raises ``BudgetExhausted("global_budget")`` when E
evaluations are done, before it looks at anything else. A score cache keyed by the phenotype
key answers a repeated phenotype: a cache hit is not an evaluation (it does not count against
E) and is counted in ``cache_hits``. Only a cache miss runs the scorer and counts.
"""

from __future__ import annotations

import math
import operator
from array import array
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pymoses.gomea.oracle import BudgetExhausted
from pymoses.gomea.standalone._tools import sr_linear
from pymoses.moses.scoring.least_squares import PIVOT_TOL, solve

# ---- the NMSE formulas of tools/gomea/sr_eval.py, restated for pymoses (same expressions,
# so the same floats). pymoses does not import tools/; run.py checks the two against each
# other on every reported model (the cross-check).


def _mean(v: Sequence[float]) -> float:
    return math.fsum(v) / len(v)


def _var(v: Sequence[float]) -> float:
    m = _mean(v)
    return math.fsum((x - m) ** 2 for x in v) / len(v)


def variance(v: Sequence[float]) -> float:
    """The population variance of ``v``, as ``sr_eval`` computes it."""
    return _var(v)


def _finite(v) -> bool:
    return all(math.isfinite(x) for x in v)


def linear_scaling(y: Sequence[float], p: Sequence[float]) -> tuple[float, float]:
    """``sr_eval.linear_scaling``: the least-squares (a, b) of y ~ a + b p."""
    if not _finite(p):
        return math.nan, math.nan
    my, mp_ = _mean(y), _mean(p)
    vp = math.fsum((x - mp_) ** 2 for x in p)
    if vp == 0.0 or not math.isfinite(vp):
        return my, 0.0
    b = math.fsum((x - mp_) * (t - my) for x, t in zip(p, y, strict=True)) / vp
    return my - b * mp_, b


def nmse(y: Sequence[float], p: Sequence[float], a: float = 0.0, b: float = 1.0) -> float:
    """``sr_eval.nmse``: 100 MSE(y, a + b p) / var(y); inf if a scaled prediction is not
    finite."""
    q = [a + b * x for x in p]
    if not _finite(q):
        return math.inf
    mse = math.fsum((t - x) ** 2 for t, x in zip(y, q, strict=True)) / len(y)
    vy = _var(y)
    return 100.0 * mse / vy if vy > 0 else (0.0 if mse == 0 else math.inf)


def clip_range(y: Sequence[float]) -> tuple[float, float]:
    """``sr_eval.clip_range``."""
    lo, hi = min(y), max(y)
    r = hi - lo
    return lo - 0.1 * r, hi + 0.1 * r


def nmse_clipped(y: Sequence[float], q: Sequence[float], lo: float, hi: float) -> float:
    """``sr_eval.nmse_clipped``."""
    return nmse(y, [x if math.isnan(x) else min(max(x, lo), hi) for x in q])


def _mse_scaled(y: Sequence[float], p: Sequence[float], a: float, b: float) -> float:
    """MSE(y, a + b p) as ``nmse`` computes it (``inf`` if not finite)."""
    q = [a + b * x for x in p]
    if not _finite(q):
        return math.inf
    return math.fsum((t - x) ** 2 for t, x in zip(y, q, strict=True)) / len(y)


def _safe(score: float) -> float:
    return -math.inf if score != score else score


@dataclass
class TrainData:
    """The training rows of a run: ``cols[j]`` is input j, ``y`` the scaled target."""

    cols: list[list[float]]
    y: list[float]

    @property
    def n(self) -> int:
        return len(self.y)


# ---- the template scorer


class TemplateScorer:
    """-MSE of the linearly scaled program output f on the training rows.

    An output with a value that is not finite scores -inf. With pc = f - mean f and yc =
    y - mean y: vp = pc . pc and cov = pc . yc (exactly rounded sums). If vp is 0 or not
    finite, the fit is the mean (``sr_eval.linear_scaling`` gives b = 0), so SSE = yc . yc;
    else b = cov / vp and SSE = yc . yc - b cov (at least 0)."""

    def __init__(self, genome, data: TrainData) -> None:
        self.genome = genome
        self.data = data
        n = data.n
        my = math.fsum(data.y) / n
        self.yc = [t - my for t in data.y]
        self.syy = math.fsum(map(operator.mul, self.yc, self.yc, strict=True))

    def score(self, program: Sequence[int]) -> float:
        p = self.genome.predict(program, self.data.cols)
        if not all(map(math.isfinite, p)):
            return -math.inf
        mp = math.fsum(p) / len(p)
        pc = [x - mp for x in p]
        vp = math.fsum(map(operator.mul, pc, pc, strict=True))
        if vp == 0.0 or not math.isfinite(vp):
            return _safe(-self.syy / self.data.n)
        cov = math.fsum(map(operator.mul, pc, self.yc, strict=True))
        sse = self.syy - (cov / vp) * cov
        if not sse >= 0.0:
            sse = 0.0 if math.isfinite(sse) else math.inf
        return _safe(-sse / self.data.n)


# ---- the IT scorer


@dataclass(frozen=True)
class ITFit:
    """The fit of an IT model. ``kept``: the terms the model keeps (the printed model);
    ``mode``: ``"terms"`` (per-term least squares, 2 or more terms), ``"single"`` (one term),
    ``"plain"`` (a + b times the sum of the kept terms), ``"mean"`` (no term: w0 only) or
    ``"invalid"``; ``sse``: the sum of squared errors on the training rows (``inf`` when
    invalid); ``dropped_constant`` and ``dropped_aliased``: the terms left out."""

    kept: tuple
    mode: str
    sse: float
    dropped_constant: int = 0
    dropped_aliased: int = 0


class _Col:
    """The statistics of one term column on the training rows, as ``fit_terms`` computes
    them: ``mean``, ``ss`` (centred sum of squares), ``raw`` (uncentred), ``constant``,
    ``norm``, ``z`` (the centred unit-length column), ``diag`` (z . z) and ``rhs``
    (z . (y - mean y)). ``finite`` is False when a value of the column is not finite;
    ``stats_finite`` is False when a statistic is not finite."""

    __slots__ = ("id", "finite", "stats_finite", "mean", "ss", "raw", "constant", "norm", "z",
                 "diag", "rhs")


class ITScorer:
    """-MSE of the per-term least-squares model on the training rows.

    **The fit** of a model with the distinct terms M (in key order):

    1. A term column with a value that is not finite makes the model invalid.
    2. A statistic of a column that is not finite (an overflow): a + b times the sum of all
       terms of M (``"plain"``), as ``fit_terms`` falls back for ``nonfinite``.
    3. Constant columns (the rule of ``fit_terms``) are left out. No column left: ``"mean"``.
    4. The normal equations of the centred unit-length columns (the Gram matrix and the right
       side of ``fit_terms``, the same expressions) are solved by ``least_squares.solve``. A
       matrix or right side that is not finite: ``"plain"`` over the columns of step 3.
    5. If ``solve`` refuses the matrix (singular), the aliased columns are left out
       (``sr_linear.independent_columns``, modified Gram-Schmidt) and the rest are solved
       again. If ``solve`` refuses again, or the weights are not finite: ``"plain"``.
    6. SSE = (y - mean y) . (y - mean y) - v . rhs (the least-squares identity in the centred
       unit-length coordinates; at least 0).

    ``kept`` is the set of terms of the last step. The weights are those of
    ``least_squares.fit_terms`` on the kept columns (the same floats), so
    ``sr_eval.evaluate(model_to_moses(kept), terms=True)`` fits the same model. The search
    score uses the identity of step 6 and not the fitted values; ``run.py`` scores the
    reported model again with the formulas of ``sr_eval``.

    **The column cache** (``column_cache=True``): the statistics of each distinct term column
    and the Gram entry of each pair of columns are kept, so a new model computes only its new
    parts. The kept values are the values that the uncached path computes again for each
    model, so both paths give the same floats. The cache is cleared when it holds
    ``max_columns`` columns or ``max_pairs`` pairs; this changes only the speed.
    """

    def __init__(
        self,
        genome,
        data: TrainData,
        column_cache: bool = True,
        max_columns: int | None = None,
        max_pairs: int = 1_000_000,
    ) -> None:
        self.genome = genome
        self.data = data
        self.n = data.n
        self.y = data.y
        self.my = math.fsum(self.y) / self.n
        self.yc = [t - self.my for t in self.y]
        self.syy = math.fsum(v * v for v in self.yc)
        self.column_cache = column_cache
        self.max_columns = (max_columns if max_columns is not None
                            else max(512, (128 * 2**20) // (8 * self.n)))
        self.max_pairs = max_pairs
        self._cols: dict = {}
        self._pairs: dict[int, float] = {}
        self._next_id = 0
        self.stats = Counter()

    # -- columns and pairs

    def column(self, m) -> list[float]:
        return self.genome.monomial_column(m, self.data.cols)

    def _new_col(self, m) -> _Col:
        c = _Col()
        col = self.column(m)
        c.finite = _finite(col)
        c.stats_finite = False
        c.constant = False
        c.z = None
        if not c.finite:
            return c
        n = self.n
        # The expressions of fit_terms. fsum is exactly rounded, so map(operator.mul) over the
        # same products gives the same floats as fit_terms' generators.
        c.mean = math.fsum(col) / n
        centred = [x - c.mean for x in col]
        c.ss = math.fsum(map(operator.mul, centred, centred, strict=True))
        c.raw = math.fsum(map(operator.mul, col, col, strict=True))
        c.stats_finite = math.isfinite(c.mean) and math.isfinite(c.ss) and math.isfinite(c.raw)
        if not c.stats_finite:
            return c
        c.constant = c.ss <= PIVOT_TOL * c.raw
        if c.constant:
            return c
        c.norm = math.sqrt(c.ss)
        z = [x / c.norm for x in centred]
        c.diag = math.fsum(map(operator.mul, z, z, strict=True))
        c.rhs = math.fsum(map(operator.mul, z, self.yc, strict=True))
        c.z = array("d", z) if self.column_cache else z
        return c

    def _col(self, m) -> _Col:
        if not self.column_cache:
            self.stats["column_computed"] += 1
            return self._new_col(m)
        c = self._cols.get(m)
        if c is not None:
            self.stats["column_hits"] += 1
            return c
        if len(self._cols) >= self.max_columns:
            # The ids go on from where they are: a fit can still hold a column from before
            # the clear, so an id must never be given twice (the pair keys use the ids).
            self._cols.clear()
            self._pairs.clear()
            self.stats["cache_clears"] += 1
        self.stats["column_computed"] += 1
        c = self._new_col(m)
        c.id = self._next_id
        self._next_id += 1
        self._cols[m] = c
        return c

    def _gram(self, a: _Col, b: _Col) -> float:
        if a is b:
            return a.diag
        if not self.column_cache:
            self.stats["pair_computed"] += 1
            return math.fsum(map(operator.mul, a.z, b.z, strict=True))
        ia, ib = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        key = (ia << 32) | ib
        v = self._pairs.get(key)
        if v is not None:
            self.stats["pair_hits"] += 1
            return v
        if len(self._pairs) >= self.max_pairs:
            self._pairs.clear()
            self.stats["pair_clears"] += 1
        self.stats["pair_computed"] += 1
        # fsum is exactly rounded, so the order of the two columns does not matter
        v = math.fsum(map(operator.mul, a.z, b.z, strict=True))
        self._pairs[key] = v
        return v

    # -- the fit

    def _plain(self, terms: Sequence, why: str, dc: int = 0, da: int = 0) -> ITFit:
        """a + b times the sum of ``terms`` (the interpreter's sum, left to right from 0)."""
        self.stats[f"plain_{why}"] += 1
        cols = [self.column(m) for m in terms]
        p = []
        for r in range(self.n):
            s = 0.0
            for c in cols:
                s += c[r]
            p.append(s)
        a, b = linear_scaling(self.y, p)
        if a != a:
            return ITFit(tuple(terms), "invalid", math.inf, dc, da)
        sse = _mse_scaled(self.y, p, a, b) * self.n
        if not math.isfinite(sse):
            return ITFit(tuple(terms), "invalid", math.inf, dc, da)
        return ITFit(tuple(terms), "plain", sse, dc, da)

    def _solve(self, cs: Sequence[_Col]) -> list[float] | None | str:
        k = len(cs)
        gram = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i, k):
                gram[i][j] = gram[j][i] = self._gram(cs[i], cs[j])
        rhs = [c.rhs for c in cs]
        if not (all(_finite(row) for row in gram) and _finite(rhs) and math.isfinite(self.my)):
            return "nonfinite"
        return solve(gram, rhs)

    def fit(self, monomials: Sequence) -> ITFit:
        terms = list(monomials)
        cs = [self._col(m) for m in terms]
        if not all(c.finite for c in cs):
            return ITFit(tuple(terms), "invalid", math.inf)
        if not all(c.stats_finite for c in cs):
            return self._plain(terms, "nonfinite")
        keep = [i for i, c in enumerate(cs) if not c.constant]
        dc = len(terms) - len(keep)
        if not keep:
            return ITFit((), "mean", self.syy, dc)
        terms = [terms[i] for i in keep]
        cs = [cs[i] for i in keep]
        v = self._solve(cs)
        if v == "nonfinite":
            return self._plain(terms, "nonfinite", dc)
        da = 0
        if v is None:
            idx = sr_linear().independent_columns([self.column(m) for m in terms])
            da = len(terms) - len(idx)
            self.stats["aliased_models"] += 1
            if da == 0:
                return self._plain(terms, "singular", dc)
            terms = [terms[i] for i in idx]
            cs = [cs[i] for i in idx]
            if not terms:
                return ITFit((), "mean", self.syy, dc, da)
            v = self._solve(cs)
            if not isinstance(v, list):
                return self._plain(terms, "singular", dc, da)
        w = [vi / c.norm for vi, c in zip(v, cs, strict=True)]
        w0 = self.my - math.fsum(wi * c.mean for wi, c in zip(w, cs, strict=True))
        if not (_finite(w) and math.isfinite(w0)):
            return self._plain(terms, "nonfinite", dc, da)
        sse = self.syy - math.fsum(vi * c.rhs for vi, c in zip(v, cs, strict=True))
        if not sse >= 0.0:
            sse = 0.0 if math.isfinite(sse) else math.inf
        mode = "terms" if len(terms) >= 2 else "single"
        return ITFit(tuple(terms), mode, sse, dc, da)

    def score(self, monomials: Sequence) -> float:
        f = self.fit(monomials)
        self.stats[f"mode_{f.mode}"] += 1
        return _safe(-f.sse / self.n)


# ---- the evaluator


class RegressionEvaluator:
    """The ``gom.Evaluator`` of the stand-alone harness (see the module docstring).

    ``checkpoints``: evaluation counts at which ``anytime`` records the best score so far.
    ``trace`` lists ``(evaluations, score)`` at each strict improvement of the best score.

    ``stall_generations`` (opt-in; NOT A PORT: the IT linkage study, BACKLOG 13.9, server round
    15): K, or ``None`` (the default: no stall rule, and ``check`` is as before). The engine calls
    ``check`` once at the start of each generation (``GomeaEngine._loop``). With K, ``check``
    counts the generations in a row that made no new evaluation (the evaluation count did not
    grow from one call to the next; cache hits and skips do not count), and returns
    ``"stalled"`` when that count reaches K. The budget check comes first. ``stall_count`` is the
    count at the last call.
    """

    def __init__(self, genome, scorer, max_evals: int, checkpoints: Sequence[int] = (),
                 stall_generations: int | None = None) -> None:
        if max_evals < 1:
            raise ValueError("max_evals must be at least 1")
        if stall_generations is not None and stall_generations < 1:
            raise ValueError("stall_generations must be at least 1")
        self.genome = genome
        self.scorer = scorer
        self.max_evals = int(max_evals)
        self.n_evals = 0
        self.cache_hits = 0
        self.noop_skips = 0
        self.phenotype_skips = 0
        self.forced_negative = 0  # IT: negative powers made positive, over counted evaluations
        self.scores: dict = {}
        self.best = -math.inf
        self.trace: list[tuple[int, float]] = []
        self.checkpoints = sorted({int(c) for c in checkpoints if 1 <= int(c) <= self.max_evals})
        self.anytime: list[tuple[int, float]] = []
        self._next_cp = 0
        self._last_g: tuple | None = None
        self._last: tuple = (None, None)
        self._it = genome.kind == "it"
        self.stall_generations = stall_generations
        self.stall_count = 0
        self._evals_at_check: int | None = None

    def _decode(self, g: tuple):
        if g is self._last_g or g == self._last_g:
            return self._last
        if self._it:
            m = self.genome.decode(g)
            out = (m.monomials, m)
        else:
            out = self.genome.decode(g)
        self._last_g, self._last = g, out
        return out

    def phenotype_key(self, g: tuple):
        return self._decode(g)[0]

    def check(self) -> str | None:
        if self.n_evals >= self.max_evals:
            return "global_budget"
        if self.stall_generations is None:
            return None
        # the opt-in stall rule: one call at the start of each generation
        if self._evals_at_check is not None:
            self.stall_count = (self.stall_count + 1 if self.n_evals == self._evals_at_check
                                else 0)
        self._evals_at_check = self.n_evals
        return "stalled" if self.stall_count >= self.stall_generations else None

    def evaluate(self, g: tuple) -> float:
        if self.n_evals >= self.max_evals:
            raise BudgetExhausted("global_budget")
        key, model = self._decode(g)
        hit = self.scores.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        self.n_evals += 1
        if self._it:
            self.forced_negative += model.forced
            s = self.scorer.score(model.monomials)
        else:
            s = self.scorer.score(model)
        s = _safe(s)
        self.scores[key] = s
        if s > self.best:
            self.best = s
            self.trace.append((self.n_evals, s))
        cps = self.checkpoints
        while self._next_cp < len(cps) and cps[self._next_cp] <= self.n_evals:
            self.anytime.append((cps[self._next_cp], self.best))
            self._next_cp += 1
        return s

    def worse(self, a: float, b: float) -> bool:
        return _safe(a) < _safe(b)

    def record_noop_skip(self) -> None:
        self.noop_skips += 1

    def record_phenotype_skip(self) -> None:
        self.phenotype_skips += 1
