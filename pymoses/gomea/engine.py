"""The GOMEA generation loop over tuples of categorical codes.

MOSES-independent: scores come from an injected ``Evaluator``
(``pymoses/gomea/gom.py``). Adapted from GP-GOMEA
(https://github.com/marcovirgolin/GP-GOMEA, commit 750220c,
``src/GOMEA/GOMEAGenerationHandler.cpp``, Apache-2.0, Copyright Marco
Virgolin); see ``docs/gomea/PROVENANCE.md`` (generation-loop row).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.gom import (
    Evaluator,
    GomInterrupted,
    GomStats,
    Individual,
    Score,
    forced_improvement,
    gom,
)
from pymoses.util.rng import MT19937RandGen


@dataclass
class EngineParams:
    fos_mode: str = "lt-b"
    forced_improvement: bool = False
    skip_inactive: bool = True
    max_generations: int = 100
    stall_generations: int = 5
    # Phase 1d ablation (hypothesis E): "nonworse" (default) or "strict";
    # see ``gom``.
    acceptance: str = "nonworse"


@dataclass
class EngineResult:
    population: list[Individual]
    best: Individual
    generations: int
    termination_reason: str
    history: list[dict] = field(default_factory=list)


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GomeaEngine:
    """Runs GOM generations until a termination condition holds.

    Termination reasons: ``target_reached``, ``no_search_fields``,
    ``no_valid_moves``, ``converged``, ``stagnation``, ``max_generations``,
    the budget reasons of ``BudgetExhausted`` (``global_budget``,
    ``inner_budget``, ``time_limit``) and ``interrupted`` (recorded on
    ``last_result`` before a ``KeyboardInterrupt`` is re-raised).

    Precedence: a budget interrupt wins over ``target_reached``. When a
    ``BudgetExhausted`` cuts a generation short, the run ends with the
    budget reason even if the partial population (for instance the
    interrupted offspring's last accepted state) meets the target. An
    offspring that meets the target ends its generation at once, so no
    later evaluation in that generation can exhaust the budget.

    ``elite`` is the best-ever individual (LT-GOMEA's live elitist). It
    starts as the best of the initial population and is replaced, within a
    generation, by any offspring strictly better than it, after GOM or
    after FI. Forced improvements use the elite current at that moment.
    """

    def __init__(
        self,
        domains: Sequence[int],
        params: EngineParams,
        ev: Evaluator,
        rng_var: MT19937RandGen,
        rng_link: MT19937RandGen,
        target: Callable[[Score], bool] | None = None,
        on_generation: Callable[[dict, list[Individual]], None] | None = None,
        symbolize: Callable[[Sequence[tuple[int, ...]]], list[tuple[int, ...]]] | None = None,
    ) -> None:
        self.domains = list(domains)
        self.params = params
        self.ev = ev
        self.rng_var = rng_var
        self.rng_link = rng_link
        self.target = target
        self.on_generation = on_generation
        # GOMEA extension: the constant recoding for linkage learning
        # (``LinkageModel``'s ``symbolize``; ``None`` = none).
        self.symbolize = symbolize
        self.last_result: EngineResult | None = None
        self.elite: Individual | None = None
        # The run's calibrated linkage model (``None`` before ``run``), so
        # ``on_generation`` can report over its ``search_positions``.
        self.model: LinkageModel | None = None

    def best_of(self, pop: Sequence[Individual]) -> Individual:
        """The best individual; ties go to the lowest index."""
        best = pop[0]
        for ind in pop[1:]:
            if self.ev.worse(best.score, ind.score):
                best = ind
        return best

    def target_met(self, ind: Individual) -> bool:
        return self.target is not None and bool(self.target(ind.score))

    def offer_elite(self, ind: Individual) -> None:
        """Make ``ind`` the elite if it is strictly better (or none is set)."""
        if self.elite is None or self.ev.worse(self.elite.score, ind.score):
            self.elite = ind

    def run(self, population: list[Individual]) -> EngineResult:
        """Evolve an already-evaluated ``population``."""
        if not population:
            raise ValueError("population must not be empty")
        self._pop = list(population)
        self._gens = 0
        self._history: list[dict] = []
        self.elite = self.best_of(self._pop)
        try:
            reason = self._loop()
        except KeyboardInterrupt:
            self._finish("interrupted")
            raise
        return self._finish(reason)

    def _finish(self, reason: str) -> EngineResult:
        self.last_result = EngineResult(
            population=list(self._pop),
            best=self.best_of(self._pop),
            generations=self._gens,
            termination_reason=reason,
            history=self._history,
        )
        return self.last_result

    def _loop(self) -> str:
        P = self._pop
        if any(self.target_met(ind) for ind in P):
            return "target_reached"
        if self.symbolize is None:
            model = LinkageModel(self.params.fos_mode, self.rng_link)
        else:
            model = LinkageModel(self.params.fos_mode, self.rng_link, self.symbolize)
        model.calibrate([ind.genotype for ind in P], self.domains)
        self.model = model
        # N < 2 needs no check of its own: a single individual makes every
        # position constant, so calibration yields no search fields.
        if not model.search_positions:
            return "no_search_fields"
        if self.params.max_generations <= 0:
            return "max_generations"
        best = self.best_of(P)
        nis = 0
        while True:
            reason = self.ev.check()
            if reason is not None:
                return reason
            try:
                new_pop, record, stop = perform_generation(self, P, model, nis, self._gens)
            except _Stop as s:
                return s.reason
            P = self._pop = new_pop
            new_best = self.best_of(P)
            improved = self.ev.worse(best.score, new_best.score)
            best = new_best
            nis = 0 if improved else nis + 1
            record["nis"] = nis
            record["best_improved"] = improved
            self._history.append(record)
            self._gens += 1
            if self.on_generation is not None:
                self.on_generation(record, P)
            if stop is not None:
                return stop
            if any(self.target_met(ind) for ind in P):
                return "target_reached"
            first = P[0].genotype
            if all(ind.genotype == first for ind in P):
                return "converged"
            if nis >= self.params.stall_generations:
                return "stagnation"
            if self._gens >= self.params.max_generations:
                return "max_generations"


def perform_generation(
    eng: GomeaEngine, P: list[Individual], model: LinkageModel, nis: int, generation: int
) -> tuple[list[Individual], dict, str | None]:
    """GP-GOMEA: src/GOMEA/GOMEAGenerationHandler.cpp:20 GOMEAGenerationHandler::PerformGeneration

    Learn the FOS, then GOM every individual against the immutable
    generation-start snapshot ``donors``, in index order. Adaptations:
    no re-initialization on convergence (the caller stops instead and
    control returns to MOSES), no worst-replacement, batching, coefficient
    mutation or explicit elitism (GOM never accepts a worse individual,
    so elitism is implicit), and a sequential loop instead of OpenMP.
    Optional forced improvements (a reconstruction, see
    ``forced_improvement``) follow each GOM when the offspring did not
    change or ``nis > 1 + floor(log10(N))``, unless the offspring already
    is the elite. The elite is ``eng.elite``, LT-GOMEA's live best-ever
    individual: any offspring strictly better than it, after GOM or FI,
    replaces it at once, so later recipients' FI use it. A budget
    interruption or a reached target ends the generation at once: the
    finished offspring plus the untouched parents form the new population.
    An empty FOS raises ``_Stop('no_valid_moves')``; the real
    ``LinkageModel`` never returns one after calibration found search
    fields, so only a substituted model can reach it.
    """
    t0 = time.perf_counter()
    donors = list(P)
    F = model.learn([ind.genotype for ind in P])
    linkage_s = time.perf_counter() - t0
    if not F:
        raise _Stop("no_valid_moves")
    if eng.elite is None:
        eng.elite = eng.best_of(donors)
    fi_threshold = 1 + math.floor(math.log10(len(P)))
    stats = GomStats()
    offspring: list[Individual] = []
    stop: str | None = None
    t1 = time.perf_counter()
    for parent in P:
        try:
            off, changed, _ = gom(
                parent,
                donors,
                F,
                eng.ev,
                eng.rng_var,
                stats,
                eng.params.skip_inactive,
                eng.params.acceptance,
            )
            eng.offer_elite(off)
            if (
                eng.params.forced_improvement
                and (not changed or nis > fi_threshold)
                and off.genotype != eng.elite.genotype
            ):
                off = forced_improvement(off, eng.elite, F, eng.ev, eng.rng_var, stats)
                eng.offer_elite(off)
        except GomInterrupted as exc:
            eng.offer_elite(exc.offspring)
            offspring.append(exc.offspring)
            stop = exc.reason
            break
        offspring.append(off)
        if eng.target_met(off):
            stop = "target_reached"
            break
    offspring.extend(P[len(offspring) :])
    record = {
        "generation": generation,
        "fos_size": len(F),
        "fos_mean_subset_size": sum(len(s) for s in F) / len(F),
        "stats": stats.as_dict(),
        "nis": nis,
        "best_improved": False,
        "linkage_s": linkage_s,
        "variation_s": time.perf_counter() - t1,
    }
    return offspring, record, stop
