"""NOT A PORT (pyMoses extension, no C++ counterpart): epsilon-lexicase exemplar
selection (`--exemplar-selection lexicase`, server round 13, arm BASE-LEX, backlog A4).

Classic MOSES selects the exemplar of the next deme by a Boltzmann roulette on the
penalized scores (`metapopulation.select_exemplar`). With `--exemplar-selection lexicase`
the metapopulation selects it from the per-row errors of the behavioural scores instead:

* **Pool:** exactly the candidates that the Boltzmann rule may select (finite penalized
  score, the revisit rule; on a reserved turn of `--reserve-empty-every`, also the lineage
  rule). The `size() == 1` shortcut of `select_exemplar` stays as it is.
* **Errors:** `err = -(bscore entry)`, one entry for each training row (a case). The
  regression scorer (`contin_bscore`) gives exactly one entry for each row, after the
  linear scaling that it applies, and no extra entry (the complexity penalty is not in
  the behavioural score). The rule ignores the complexity penalty.
* **Selection:** the cases in a random order from the run RNG (`randGen()`). For each
  case, epsilon is the median absolute deviation (MAD), median(|e_i - median(e)|), of the
  errors of the WHOLE pool on that case. Keep the candidates whose error on the case is
  at most (the lowest error among the remaining candidates) + epsilon. Stop when one
  candidate remains or after the last case, then pick one of the remaining candidates at
  random (run RNG).

Two details that do not change the rule:

* The random order is drawn lazily, by a front-to-back Fisher-Yates shuffle that draws
  the next case only when the selection needs it. The cases that the selection reaches
  are a uniform random order, as with a full shuffle; the cases after the stop are not
  drawn.
* Epsilon of a case depends only on the whole pool, so it is computed when the selection
  reaches the case (the same value as a computation for all cases at the start).

Random draws: one `randint_n` call for each case used, and one for the final pick only
when more than one candidate remains. With `boltzmann` (the default) nothing in this
module runs and the run RNG draws exactly the numbers of classic MOSES.

Source: La Cava, Spector and Danai, GECCO 2016 (DOI 10.1145/2908812.2908898); the
"semi-dynamic" epsilon (MAD over the whole pool, survivors filtered against the lowest
error among them). See `docs/gomea/SERVER-ROUND13-PREREGISTRATION.md`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# The values of `--exemplar-selection`.
EXEMPLAR_SELECTION_BOLTZMANN = "boltzmann"
EXEMPLAR_SELECTION_LEXICASE = "lexicase"
EXEMPLAR_SELECTION_CHOICES = (EXEMPLAR_SELECTION_BOLTZMANN, EXEMPLAR_SELECTION_LEXICASE)


def median(values: Sequence[float]) -> float:
    """The median of a non-empty sequence (the mean of the two middle values for an even
    count)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("median of an empty sequence")
    h = n // 2
    return s[h] if n % 2 else (s[h - 1] + s[h]) / 2.0


def mad(values: Sequence[float]) -> float:
    """The median absolute deviation, median(|v - median(v)|), of a non-empty sequence."""
    m = median(values)
    return median([abs(v - m) for v in values])


def lexicase_select(bscores: Sequence[Sequence[float]], rng) -> tuple[int, int]:
    """Epsilon-lexicase selection over the pool `bscores` (one behavioural score for each
    pool candidate, all of the same length; the error on case c is `-bs[c]`).

    Returns `(index into bscores, number of cases used)`. `rng` is the run RNG
    (`randGen()`); only its `randint_n` is called (see the module docstring)."""
    p = len(bscores)
    if p == 0:
        raise ValueError("lexicase_select: empty pool")
    n = len(bscores[0])
    if n == 0 or any(len(bs) != n for bs in bscores):
        raise AssertionError(
            "lexicase_select: every pool candidate needs a behavioural score with one "
            f"entry for each training row; lengths {sorted({len(bs) for bs in bscores})}"
        )
    survivors = list(range(p))
    order = list(range(n))
    used = 0
    while len(survivors) > 1 and used < n:
        # front-to-back Fisher-Yates: the next case of the random order
        j = used + rng.randint_n(n - used)
        order[used], order[j] = order[j], order[used]
        c = order[used]
        used += 1
        errors = [-bs[c] for bs in bscores]
        if not all(math.isfinite(e) for e in errors):
            # A pool candidate has a finite penalized score, so (all entries <= 0) every
            # entry of its behavioural score is finite. Anything else is a bug.
            raise AssertionError(f"lexicase_select: a pool error on case {c} is not finite")
        eps = mad(errors)
        best = min(errors[i] for i in survivors)
        survivors = [i for i in survivors if errors[i] <= best + eps]
    if len(survivors) == 1:
        return survivors[0], used
    return survivors[rng.randint_n(len(survivors))], used


class lexicase_stats:
    """NOT A PORT: the run's telemetry of `--exemplar-selection lexicase` (the "lexicase"
    event of the GOMEA log, `report`). It exists only while the option is on."""

    def __init__(self) -> None:
        self.n_cases = None
        self.selections = 0
        self.cases_used_total = 0
        self.cases_used_max = 0
        self.pool_size_total = 0
        self.bscore_recomputes = 0
        self.bscore_recompute_cpu_s = 0.0

    def note_selection(self, cases_used: int, pool_size: int, n_cases: int) -> None:
        self.n_cases = n_cases
        self.selections += 1
        self.cases_used_total += cases_used
        self.cases_used_max = max(self.cases_used_max, cases_used)
        self.pool_size_total += pool_size

    def note_recompute(self, n: int, cpu_s: float) -> None:
        """`n` behavioural scores recomputed by the merge (not counted as evaluations)."""
        self.bscore_recomputes += n
        self.bscore_recompute_cpu_s += cpu_s

    def report(self) -> dict:
        s = self.selections
        return {
            "n_cases": self.n_cases,
            "selections": s,
            "cases_used_mean": self.cases_used_total / s if s else None,
            "cases_used_max": self.cases_used_max,
            "pool_size_mean": self.pool_size_total / s if s else None,
            "bscore_recomputes": self.bscore_recomputes,
            "bscore_recompute_cpu_s": self.bscore_recompute_cpu_s,
        }
