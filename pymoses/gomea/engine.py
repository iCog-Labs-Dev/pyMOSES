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

from pymoses.gomea.fos import LinkageModel, select_fos_topk
from pymoses.gomea.gom import (
    Evaluator,
    GomInterrupted,
    GomStats,
    Individual,
    Score,
    forced_improvement,
    gom,
)
from pymoses.gomea.knob_cap import KnobSelector, default_explore
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
    # GOMEA extension (``--gomea-fos-topk``): visit only ``fos_topk`` FOS
    # subsets per generation (0 = all, the default); ``fos_topk_mode`` is
    # ``"linkage"`` or ``"random"``. See ``perform_generation``.
    fos_topk: int = 0
    fos_topk_mode: str = "linkage"
    # GOMEA extension (``--gomea-knob-cap``, server round 2): build each
    # generation's FOS over ``knob_cap`` selected knobs only (0 = off, the
    # default); see ``pymoses.gomea.knob_cap``. ``knob_explore`` ``None`` is
    # ``ceil(knob_cap / 4)``.
    knob_cap: int = 0
    knob_select: str = "explore"
    knob_explore: int | None = None
    knob_rank0: str = "contrast"


@dataclass
class EngineResult:
    population: list[Individual]
    best: Individual
    generations: int
    termination_reason: str
    history: list[dict] = field(default_factory=list)


def _default_fitness(score: Score) -> float:
    """A score's scalar fitness, higher is better: MOSES's penalized score, or the
    score itself when it is a number."""
    get = getattr(score, "get_penalized_score", None)
    return float(get() if get is not None else score)


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
        on_accept: Callable[[tuple[int, ...]], None] | None = None,
        rng_knob: MT19937RandGen | None = None,
        fitness: Callable[[Score], float] | None = None,
        mutate: Callable[[Individual], Individual] | None = None,
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
        # GOMEA extension (``--gomea-deme-store accepted``): called with
        # every evaluated trial GOM or FI accepts (``gom``'s ``on_accept``;
        # ``None`` = none).
        self.on_accept = on_accept
        self.last_result: EngineResult | None = None
        self.elite: Individual | None = None
        # The run's calibrated linkage model (``None`` before ``run``), so
        # ``on_generation`` can report over its ``search_positions``.
        self.model: LinkageModel | None = None
        # GOMEA extension (``--gomea-knob-cap``): the knob-cap stream (all of
        # the selector's draws; required when ``knob_cap > 0``), the scalar
        # fitness of a score for the generation-0 contrast (higher is better;
        # default ``_default_fitness``) and the deme's selector (``None``
        # when off, and before ``run``).
        self.rng_knob = rng_knob
        self.fitness = fitness if fitness is not None else _default_fitness
        self.knobs: KnobSelector | None = None
        # GOMEA extension (``--const-opt mutate``): called with each GOM
        # offspring, returns it with its constants mutated (``None`` = off).
        # It raises ``GomInterrupted`` when the budget runs out.
        self.mutate = mutate

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
        p = self.params
        if p.knob_cap > 0:
            if self.rng_knob is None:
                raise ValueError("knob_cap > 0 needs rng_knob, the knob-cap stream")
            explore = default_explore(p.knob_cap) if p.knob_explore is None else p.knob_explore
            self.knobs = KnobSelector(
                p.knob_cap, p.knob_select, explore, p.knob_rank0, self.rng_knob
            )
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

    GOMEA extension (``--gomea-fos-topk``): with ``fos_topk`` K > 0 the
    learned FOS is narrowed, once per generation and before any GOM, to
    the K subsets ``select_fos_topk`` picks (``model.last_strengths`` for
    ``"linkage"``; K draws from ``eng.rng_link`` for ``"random"`` when
    ``K < |FOS|``). Every individual's GOM, and forced improvements, then
    use only those subsets (in a fresh shuffled order, as before). The
    record gains ``fos_topk``, ``fos_topk_mode``, ``fos_effective_size``,
    ``fos_effective_mean_subset_size``, ``fos_selected`` (the subsets, as
    genotype positions) and ``fos_selected_strength`` (their strengths,
    ``None`` where a subset has none); ``fos_size`` stays the full FOS.
    With K = 0 nothing is drawn and the record is unchanged.

    GOMEA extension (``--gomea-knob-cap``): with ``eng.knobs`` set (``knob_cap``
    K > 0), the generation's knobs are chosen first (``KnobSelector.select``,
    drawing only from ``eng.rng_knob``) and the FOS is learned over them only
    (``LinkageModel.learn_subset``), so GOM and forced improvements never copy
    another knob. ``fos_size`` is then the size of that FOS, and the record
    gains the ``knob_*`` fields of ``KnobSelector.record``. With K = 0 the FOS
    is ``model.learn``'s and the record is unchanged.
    """
    t0 = time.perf_counter()
    donors = list(P)
    knobs = getattr(eng, "knobs", None)
    knob_rec: dict = {}
    if knobs is None:
        F = model.learn([ind.genotype for ind in P])
    else:
        genotypes = [ind.genotype for ind in P]
        sel = knobs.select(generation, genotypes, lambda: [eng.fitness(i.score) for i in P], model)
        F = model.learn_subset(genotypes, sel.selected, sel.H)
        knob_rec = knobs.record(sel)
    linkage_s = time.perf_counter() - t0
    if not F:
        raise _Stop("no_valid_moves")
    full_size = len(F)
    full_mean = sum(len(s) for s in F) / len(F)
    topk: dict = {}
    if eng.params.fos_topk > 0:
        strengths = getattr(model, "last_strengths", None)
        if strengths is not None and len(strengths) != len(F):
            strengths = None
        sel = select_fos_topk(
            len(F), strengths, eng.params.fos_topk, eng.params.fos_topk_mode, eng.rng_link
        )
        F = [F[i] for i in sel]
        topk = {
            "fos_topk": eng.params.fos_topk,
            "fos_topk_mode": eng.params.fos_topk_mode,
            "fos_effective_size": len(F),
            "fos_effective_mean_subset_size": sum(len(s) for s in F) / len(F),
            "fos_selected": [list(s) for s in F],
            "fos_selected_strength": [None if strengths is None else strengths[i] for i in sel],
        }
    if eng.elite is None:
        eng.elite = eng.best_of(donors)
    fi_threshold = 1 + math.floor(math.log10(len(P)))
    stats = GomStats()
    # Passed only when set, so the call is unchanged without the hook.
    hook = {} if eng.on_accept is None else {"on_accept": eng.on_accept}
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
                **hook,
            )
            mutate = getattr(eng, "mutate", None)
            if mutate is not None:
                off = mutate(off)
            eng.offer_elite(off)
            if (
                eng.params.forced_improvement
                and (not changed or nis > fi_threshold)
                and off.genotype != eng.elite.genotype
            ):
                off = forced_improvement(off, eng.elite, F, eng.ev, eng.rng_var, stats, **hook)
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
        "fos_size": full_size,
        "fos_mean_subset_size": full_mean,
        **topk,
        **knob_rec,
        "stats": stats.as_dict(),
        "nis": nis,
        "best_improved": False,
        "linkage_s": linkage_s,
        "variation_s": time.perf_counter() - t1,
    }
    return offspring, record, stop
