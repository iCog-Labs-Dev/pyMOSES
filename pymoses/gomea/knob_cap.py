"""GOMEA extension (``--gomea-knob-cap``, server round 2): a cap on the knobs in the FOS.

``docs/gomea/SERVER-ROUND2-PREREGISTRATION.md``, "The knob cap". With a cap K > 0 the FOS
of each generation is built (by the arm's own ``--gomea-fos``) over K selected knobs only;
GOM and forced improvements never copy a knob outside the selection, so every individual
keeps its own value there.

* **Candidates:** the deme's search positions (``LinkageModel.search_positions``: the
  knobs that vary in the initial population).
* **Score in generation 0** (``rank0``):

  * ``contrast``: ``|mean fitness of the individuals whose value at the knob differs from
    the exemplar's - mean fitness of the others|``; 0 when all or none differ (and when
    the difference is NaN, e.g. two infinite means). The exemplar's value is code 0 of
    every knob (``FrozenSchema``: "value 0 of every knob is the exemplar's own setting").
  * ``entropy``: the knob's entropy in the initial population (the diagonal of the
    linkage model's ``H0``; 0 for a knob left out of the linkage tree).
  * ``random``: a random order (every score equal; ties are broken at random).

* **Score from generation 1** (``explore`` only): linkage strength, the knob's largest
  LT-B similarity (``stats.biased_similarity`` against ``H0``) to any other search
  position, in the current population; computed with LT-B whatever the arm's FOS mode.
  A knob outside the linkage tree (no ``H0`` row) has strength ``-inf``.
* **Selection** (``select``):

  * ``static``: the K best by the generation-0 score, fixed for the deme;
  * ``random``: K random search positions in generation 0, fixed for the deme;
  * ``explore``: each generation a core of the K - E best (generation 0: the
    generation-0 score; later: linkage strength) plus E random knobs from the rest.

  Ties are broken at random. With K at least the number of candidates every candidate is
  selected.

All draws come from the selector's own ``rng`` (a separate stream, see
``gomea_optimization``), never from the linkage or variation streams.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pymoses.util.rng import MT19937RandGen

KNOB_SELECT_MODES = ("explore", "static", "random")
KNOB_RANK0_MODES = ("contrast", "entropy", "random")


def default_explore(cap: int) -> int:
    """E's default, ``ceil(K / 4)``."""
    return -(-cap // 4)


def fitness_contrast(
    genotypes: Sequence[tuple[int, ...]],
    fitnesses: Sequence[float],
    positions: Sequence[int],
    exemplar: Sequence[int] | None = None,
) -> list[float]:
    """Per position, ``|mean fitness where the value differs from the exemplar's - mean
    fitness of the others|``; 0.0 when all or none differ, or when the difference is NaN.
    ``exemplar`` defaults to code 0 at every position."""
    out = []
    for p in positions:
        ex = 0 if exemplar is None else exemplar[p]
        diff = [f for g, f in zip(genotypes, fitnesses, strict=True) if g[p] != ex]
        same = [f for g, f in zip(genotypes, fitnesses, strict=True) if g[p] == ex]
        if not diff or not same:
            out.append(0.0)
            continue
        c = abs(sum(diff) / len(diff) - sum(same) / len(same))
        out.append(0.0 if math.isnan(c) else c)
    return out


def shuffled(items: Sequence[int], rng: MT19937RandGen) -> list[int]:
    """A Fisher-Yates permutation of ``items`` drawn with ``randint_n``."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randint_n(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def rank(items: Sequence[int], scores: Sequence[float], rng: MT19937RandGen) -> list[int]:
    """``items`` by score, highest first; ties in a random order (a random permutation,
    then a stable sort). A NaN score ranks last."""
    by_item = dict(zip(items, scores, strict=True))
    order = shuffled(items, rng)
    return sorted(order, key=lambda p: -by_item[p] if not math.isnan(by_item[p]) else math.inf)


def draw(items: Sequence[int], k: int, rng: MT19937RandGen) -> list[int]:
    """``min(k, len(items))`` distinct items, uniformly without replacement (a partial
    Fisher-Yates shuffle), in draw order."""
    pool = list(items)
    k = min(k, len(pool))
    for i in range(k):
        j = i + rng.randint_n(len(pool) - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


@dataclass
class KnobSelection:
    """One generation's knobs: ``selected`` (sorted), ``core`` (rank order), ``explore``
    (draw order), ``entered`` (knobs in the core that were not in the previous
    generation's core; 0 in generation 0 and for the fixed modes) and ``H`` (the entropy
    matrix over the model's linked positions, when linkage strength was computed, for
    ``LinkageModel.learn_subset`` to reuse; else ``None``)."""

    selected: tuple[int, ...]
    core: tuple[int, ...]
    explore: tuple[int, ...]
    entered: int
    n_candidates: int
    H: list[list[float]] | None = None


class KnobSelector:
    """The knob cap of one deme: call ``select`` once per generation, in order."""

    def __init__(
        self,
        cap: int,
        select: str,
        explore: int,
        rank0: str,
        rng: MT19937RandGen,
        exemplar: Sequence[int] | None = None,
    ) -> None:
        if cap < 1:
            raise ValueError(f"knob cap must be >= 1, got {cap}")
        if select not in KNOB_SELECT_MODES:
            raise ValueError(f"knob select must be one of {KNOB_SELECT_MODES}, got {select!r}")
        if rank0 not in KNOB_RANK0_MODES:
            raise ValueError(f"knob rank0 must be one of {KNOB_RANK0_MODES}, got {rank0!r}")
        if not 0 <= explore <= cap:
            raise ValueError(f"knob explore must be in [0, {cap}], got {explore}")
        self.cap = cap
        self.select_mode = select
        self.explore = explore
        self.rank0 = rank0
        self.rng = rng
        self.exemplar = exemplar
        self._fixed: KnobSelection | None = None
        self._prev_core: tuple[int, ...] | None = None
        self.entered_total = 0

    def scores0(
        self,
        genotypes: Sequence[tuple[int, ...]],
        fitness: Callable[[], Sequence[float]],
        model,
    ) -> list[float]:
        """The generation-0 score of every search position (``fitness()`` is called only
        for ``contrast``)."""
        cands = model.search_positions
        if self.rank0 == "contrast":
            return fitness_contrast(genotypes, fitness(), cands, self.exemplar)
        if self.rank0 == "entropy":
            h = model.initial_entropies()
            return [h.get(p, 0.0) for p in cands]
        return [0.0] * len(cands)

    def select(
        self,
        generation: int,
        genotypes: Sequence[tuple[int, ...]],
        fitness: Callable[[], Sequence[float]],
        model,
    ) -> KnobSelection:
        """The knobs of generation ``generation`` for the population ``genotypes``.
        ``fitness()`` returns the population's fitnesses (higher is better)."""
        if self._fixed is not None:
            f = self._fixed
            return KnobSelection(f.selected, f.core, f.explore, 0, f.n_candidates)
        cands = list(model.search_positions)
        n = len(cands)
        H = None
        if self.select_mode == "random":
            chosen = draw(cands, self.cap, self.rng)
            sel = KnobSelection(tuple(sorted(chosen)), tuple(chosen), (), 0, n)
            self._fixed = sel
            return sel
        if generation == 0 or self.select_mode == "static":
            scores = self.scores0(genotypes, fitness, model)
        else:
            strengths, H = model.knob_strengths(genotypes)
            scores = [strengths[p] for p in cands]
        ranked = rank(cands, scores, self.rng)
        if self.select_mode == "static":
            core = tuple(ranked[: self.cap])
            sel = KnobSelection(tuple(sorted(core)), core, (), 0, n)
            self._fixed = sel
            return sel
        n_core = min(self.cap - self.explore, n)
        core = tuple(ranked[:n_core])
        ex = tuple(draw(ranked[n_core:], self.explore, self.rng))
        entered = 0 if self._prev_core is None else len(set(core) - set(self._prev_core))
        self._prev_core = core
        self.entered_total += entered
        return KnobSelection(tuple(sorted(core + ex)), core, ex, entered, n, H)

    def record(self, sel: KnobSelection) -> dict:
        """The generation record's knob-cap fields."""
        return {
            "knob_cap": self.cap,
            "knob_select": self.select_mode,
            "knob_rank0": self.rank0,
            "knob_explore_n": self.explore,
            "knob_n_candidates": sel.n_candidates,
            "knob_selected": list(sel.selected),
            "knob_core": list(sel.core),
            "knob_explore": list(sel.explore),
            "knob_core_entered": sel.entered,
        }
