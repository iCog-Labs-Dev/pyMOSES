"""Gene-pool optimal mixing (GOM) and forced improvements for GOMEA.

The operator works on plain tuples of categorical codes through an injected
``Evaluator``; it knows nothing about MOSES. A later adapter wraps
``EvaluationOracle`` (``pymoses/gomea/oracle.py``) into this protocol.

GOM is ported from GP-GOMEA (https://github.com/marcovirgolin/GP-GOMEA,
commit 750220c, ``src/GOMEA/GOMVariator.cpp``, Apache-2.0, Copyright Marco
Virgolin); forced improvements are a labelled reconstruction. See
``docs/gomea/PROVENANCE.md`` (GOM and FI rows).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pymoses.gomea.oracle import BudgetExhausted
from pymoses.util.rng import MT19937RandGen

Score = Any


class Evaluator(Protocol):
    """What GOM and the engine need from a scorer.

    ``worse(a, b)`` is "``a`` strictly worse than ``b``" (MOSES: ``a < b``
    under the composite comparator). ``phenotype_key`` returns ``None``
    when no proof of phenotype equality is available.
    """

    def check(self) -> str | None: ...

    def evaluate(self, g: tuple[int, ...]) -> Score: ...

    def phenotype_key(self, g: tuple[int, ...]) -> Hashable | None: ...

    def worse(self, a: Score, b: Score) -> bool: ...

    def record_noop_skip(self) -> None: ...

    def record_phenotype_skip(self) -> None: ...


@dataclass(frozen=True)
class Individual:
    """An evaluated genotype. Frozen: GOM always builds new individuals, so a
    donor snapshot can never be changed behind the engine's back."""

    genotype: tuple[int, ...]
    score: Score
    pkey: Hashable | None


@dataclass
class GomStats:
    """Counters for one generation (or one call) of GOM/FI.

    ``trials`` counts every FOS subset GOM visits; ``evaluated`` GOM's
    scorer calls; a proven phenotype skip in GOM counts as both
    ``accepted_neutral`` and ``phenotype_skips``. Forced improvements have
    their own counters: ``fi_trials`` (subsets visited), ``fi_evaluated``
    (scorer calls) and ``fi_improved`` (strict improvements accepted); the
    GOM counters above never include FI work. ``noop_skips`` and
    ``phenotype_skips`` are shared by GOM and FI, since they mirror
    ``ev.record_noop_skip`` / ``ev.record_phenotype_skip``.
    """

    trials: int = 0
    evaluated: int = 0
    accepted_improving: int = 0
    accepted_neutral: int = 0
    rejected: int = 0
    noop_skips: int = 0
    phenotype_skips: int = 0
    fi_events: int = 0
    fi_elite_copies: int = 0
    fi_trials: int = 0
    fi_evaluated: int = 0
    fi_improved: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


class GomInterrupted(Exception):
    """The budget ran out inside ``gom``/``forced_improvement``.

    ``offspring`` is the last accepted state (a valid, evaluated
    individual); ``reason`` is the ``BudgetExhausted`` reason.
    """

    def __init__(self, offspring: Individual, reason: str) -> None:
        super().__init__(reason)
        self.offspring = offspring
        self.reason = reason


ACCEPTANCE_MODES = ("nonworse", "strict")


def _shuffled_order(n: int, rng: MT19937RandGen) -> list[int]:
    """Fisher-Yates permutation of ``range(n)`` drawn with ``randint_n``
    (stands in for upstream's ``std::random_shuffle``)."""
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint_n(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def _mix(current: tuple[int, ...], source: tuple[int, ...], subset: Sequence[int]):
    trial = list(current)
    for p in subset:
        trial[p] = source[p]
    return tuple(trial)


def gom(
    recipient: Individual,
    donors: Sequence[Individual],
    fos: Sequence[Sequence[int]],
    ev: Evaluator,
    rng: MT19937RandGen,
    stats: GomStats,
    skip_inactive: bool,
    acceptance: str = "nonworse",
) -> tuple[Individual, bool, bool]:
    """GP-GOMEA: src/GOMEA/GOMVariator.cpp:22 GOMVariator::GOM

    The path with ``coeff_mut_prob == 0``. Clone the recipient, shuffle the
    FOS order, and for each subset copy the values of a donor drawn
    uniformly from ``donors`` (self allowed); keep the change iff the trial
    is not worse. Adaptations: ``new_fit <= back_fit`` becomes ``not
    ev.worse(trial, current)`` (MOSES maximizes a composite score);
    upstream's "no active node changed" test becomes a proof via
    ``ev.phenotype_key`` (when ``skip_inactive``): a trial with the current
    key is kept with the current score, unevaluated, like upstream keeps a
    non-meaningful change. A subset that changes nothing is skipped without
    evaluation. ``donors`` must be the immutable generation-start snapshot
    (upstream passes the live ``population``, which is equivalent there
    because offspring are separate clones). ``std::random_shuffle`` and
    ``randu() * donors.size()`` are replaced by draws from ``rng``.

    ``acceptance`` is ``"nonworse"`` (the above, upstream's rule) or
    ``"strict"`` (the Phase 1d ablation, hypothesis E): keep a trial only if
    it is strictly better (``ev.worse(current, trial)``); an equal-score
    trial is restored and counted as ``rejected``, and so is a proven
    phenotype skip, which can never be strictly better (it still counts as
    a ``phenotype_skips`` and is never evaluated).

    Returns ``(offspring, changed, strictly_improved)``. A
    ``BudgetExhausted`` from ``ev.evaluate`` is re-raised as
    ``GomInterrupted`` carrying the last accepted state.
    """
    if acceptance not in ACCEPTANCE_MODES:
        raise ValueError(f"acceptance must be one of {ACCEPTANCE_MODES}, got {acceptance!r}")
    strict = acceptance == "strict"
    cur_g, cur_s, cur_k = recipient.genotype, recipient.score, recipient.pkey
    improved = False
    for idx in _shuffled_order(len(fos), rng):
        subset = fos[idx]
        donor = donors[rng.randint_n(len(donors))]
        stats.trials += 1
        trial = _mix(cur_g, donor.genotype, subset)
        if trial == cur_g:
            stats.noop_skips += 1
            ev.record_noop_skip()
            continue
        trial_k = ev.phenotype_key(trial)
        if skip_inactive and trial_k is not None and trial_k == cur_k:
            stats.phenotype_skips += 1
            ev.record_phenotype_skip()
            if strict:
                stats.rejected += 1
                continue
            stats.accepted_neutral += 1
            cur_g = trial
            continue
        try:
            trial_s = ev.evaluate(trial)
        except BudgetExhausted as exc:
            raise GomInterrupted(Individual(cur_g, cur_s, cur_k), exc.reason) from exc
        stats.evaluated += 1
        if ev.worse(trial_s, cur_s) or (strict and not ev.worse(cur_s, trial_s)):
            stats.rejected += 1
            continue
        if ev.worse(cur_s, trial_s):
            stats.accepted_improving += 1
            improved = True
        else:
            stats.accepted_neutral += 1
        cur_g, cur_s, cur_k = trial, trial_s, trial_k
    offspring = Individual(cur_g, cur_s, cur_k)
    return offspring, cur_g != recipient.genotype, improved


def forced_improvement(
    offspring: Individual,
    elite: Individual,
    fos: Sequence[Sequence[int]],
    ev: Evaluator,
    rng: MT19937RandGen,
    stats: GomStats,
) -> Individual:
    """Forced improvement: a labelled reconstruction.

    Neither GP-GOMEA (commit 750220c) nor gpg implements it; this follows
    the LT-GOMEA convention described in the GOMEA papers, including its
    live best-ever elitist: ``elite`` is the best individual found so far,
    updated by the engine as soon as any offspring strictly beats it (the
    engine also decides when to trigger FI, and skips it when the offspring
    already is the elite). For each subset of a shuffled FOS, copy the
    elite's values. A subset that changes nothing is skipped without
    evaluation. A trial proven (via ``ev.phenotype_key``) to share the
    current phenotype is skipped too, whatever ``skip_inactive`` says: FI
    accepts only strict improvements, and a phenotype-equal trial scores
    the same, so it can never strictly improve. Accept only a strict
    improvement, and stop at the first one. If none is found, the offspring
    becomes a copy of the elite (genotype, score and key) only when the
    elite is strictly better; otherwise the offspring is returned unchanged,
    so FI never returns something worse than its input.

    Counts one ``fi_events`` per call, and ``fi_trials``, ``fi_evaluated``,
    ``fi_improved`` and ``fi_elite_copies`` for its work; the GOM counters
    ``trials``, ``evaluated``, ``accepted_*`` and ``rejected`` are left
    alone. A ``BudgetExhausted`` is re-raised as ``GomInterrupted``
    carrying the unchanged ``offspring``.
    """
    stats.fi_events += 1
    cur_g, cur_s, cur_k = offspring.genotype, offspring.score, offspring.pkey
    for idx in _shuffled_order(len(fos), rng):
        stats.fi_trials += 1
        trial = _mix(cur_g, elite.genotype, fos[idx])
        if trial == cur_g:
            stats.noop_skips += 1
            ev.record_noop_skip()
            continue
        trial_k = ev.phenotype_key(trial)
        if trial_k is not None and trial_k == cur_k:
            stats.phenotype_skips += 1
            ev.record_phenotype_skip()
            continue
        try:
            trial_s = ev.evaluate(trial)
        except BudgetExhausted as exc:
            raise GomInterrupted(offspring, exc.reason) from exc
        stats.fi_evaluated += 1
        if ev.worse(cur_s, trial_s):
            stats.fi_improved += 1
            return Individual(trial, trial_s, trial_k)
    if not ev.worse(cur_s, elite.score):
        return offspring
    stats.fi_elite_copies += 1
    return Individual(elite.genotype, elite.score, elite.pkey)
