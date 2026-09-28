"""The MOSES optimizer seam for the GOMEA population optimizer.

``deme_expander.optimize_demes`` calls ``optimizer(deme, iscorer, max_evals,
max_time)`` on an empty deme and expects the optimizer to fill it with every
instance it evaluated and to set ``deme.n_evals`` (``docs/gomea/P1.0-AUDIT.md``
Sec 3). ``gomea_optimization`` meets that contract by routing every scorer
call through one ``EvaluationOracle`` (which archives each trial into the
deme) and then calling ``oracle.finalize()``.

Each call is self-contained: the frozen schema, initial population, linkage
model and engine are all built from the call's own representation and
discarded afterwards. The only state kept across calls is the
``optim_stats`` counters (``total_steps``, ``total_evals``), exactly like
hill climbing's.
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from collections.abc import Hashable, Sequence

from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.gom import ACCEPTANCE_MODES, Individual
from pymoses.gomea.oracle import BudgetExhausted, EvaluationOracle
from pymoses.gomea.population import (
    INIT_MODES,
    InitResult,
    build_population,
    build_uniform_population,
    parse_radii,
    top_up_population,
)
from pymoses.gomea.schema import CONTIN_MODES, FrozenSchema
from pymoses.gomea.stats import CONTIN_STATS_MODES, contin_symbols
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymoses.moses.moses.types import composite_score
from pymoses.util.rng import MT19937RandGen, randGen

__all__ = [
    "TERMINATION_REASONS",
    "gomea_optimization",
    "gomea_parameters",
    "population_diversity",
]

FOS_MODES = ("lt-b", "lt-mi", "rt", "uni")

BUDGET_REASONS = frozenset({"global_budget", "inner_budget", "time_limit"})

# Every value ``last_result["termination_reason"]`` can take: the engine's own
# reasons (``GomeaEngine``'s docstring) plus the oracle's budget reasons,
# which can also end a call during the initial evaluation. ``stall_evals``
# (Phase 1e) is the oracle's stall rule: the deme is finalized normally, as
# for ``inner_budget``, but it is not a budget reason (not ``over_budget``).
TERMINATION_REASONS = (
    "target_reached",
    "no_search_fields",
    "no_valid_moves",
    "converged",
    "stagnation",
    "max_generations",
    "global_budget",
    "inner_budget",
    "time_limit",
    "interrupted",
    "stall_evals",
)


class gomea_parameters:
    """Settings for ``gomea_optimization`` (the counterpart of
    ``hc_parameters``). Validated on construction; raises ``ValueError``.

    ``skip_inactive`` (skip a trial proven, via ``FrozenSchema.
    phenotype_key``, to decode to the current phenotype) defaults to off:
    the proof costs about 10% of wall time and was measured at about 0
    skips on real MOSES representations (single-field neutral changes
    0/210, 0/570, 9/720 and 0/3330). It stays available as an option.

    ``pop_size`` defaults to 16: the Phase 1 pilot found N=16 best of
    {16, 32, 64} under a 20k-evaluation budget
    (``docs/gomea/pilot-a/aggregate_results.csv``).

    Phase 1d ablation options, each off by default so a default run
    reproduces Phase 1 bit for bit: ``init_mode`` (``"sampler"``, or
    ``"uniform"`` for ``build_uniform_population``; hypothesis A),
    ``acceptance`` (``"nonworse"``, or ``"strict"``, see ``gom``;
    hypothesis E) and ``genotype_memo`` (``EvaluationOracle``'s
    GOMEA-only genotype memo; hypothesis C).

    Phase 1e: ``stall_evals`` (0 = off) ends a deme once that many charged
    evaluations have passed since its best last strictly improved
    (``EvaluationOracle``'s stall rule; termination reason
    ``"stall_evals"``).

    GOMEA extension, paper-style constants: ``contin`` (``"reject"`` by
    default, or ``"copy"``: each contin knob is one copy-only gene, see
    ``FrozenSchema``) and ``contin_stats`` (how constants enter the linkage
    tree: ``"bin"`` by default, the paper's bin-const with gamma = 100, or
    ``"all"``/``"none"``; see ``stats.contin_symbols``). ``contin_stats``
    has no effect unless ``contin`` is ``"copy"`` and the representation has
    contin knobs."""

    def __init__(
        self,
        pop_size: int = 16,
        fos: str = "lt-b",
        forced_improvement: bool = False,
        skip_inactive: bool = False,
        radii: str = "1:0.4,2:0.3,4:0.2,max:0.1",
        dedupe: bool = True,
        max_generations: int = 100,
        stall_generations: int = 5,
        max_evals_per_deme: int | None = None,
        telemetry_path: str | None = None,
        init_mode: str = "sampler",
        acceptance: str = "nonworse",
        genotype_memo: bool = False,
        stall_evals: int = 0,
        contin: str = "reject",
        contin_stats: str = "bin",
    ) -> None:
        if pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {pop_size}")
        if fos not in FOS_MODES:
            raise ValueError(f"fos must be one of {FOS_MODES}, got {fos!r}")
        if max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {max_generations}")
        if stall_generations < 1:
            raise ValueError(f"stall_generations must be >= 1, got {stall_generations}")
        if max_evals_per_deme is not None and max_evals_per_deme < 1:
            raise ValueError(f"max_evals_per_deme must be None or >= 1, got {max_evals_per_deme}")
        if init_mode not in INIT_MODES:
            raise ValueError(f"init_mode must be one of {INIT_MODES}, got {init_mode!r}")
        if acceptance not in ACCEPTANCE_MODES:
            raise ValueError(f"acceptance must be one of {ACCEPTANCE_MODES}, got {acceptance!r}")
        if stall_evals < 0:
            raise ValueError(f"stall_evals must be >= 0 (0 = off), got {stall_evals}")
        if contin not in CONTIN_MODES:
            raise ValueError(f"contin must be one of {CONTIN_MODES}, got {contin!r}")
        if contin_stats not in CONTIN_STATS_MODES:
            raise ValueError(
                f"contin_stats must be one of {CONTIN_STATS_MODES}, got {contin_stats!r}"
            )
        parse_radii(radii)

        self.pop_size = pop_size
        self.fos = fos
        self.forced_improvement = forced_improvement
        self.skip_inactive = skip_inactive
        self.radii = radii
        self.dedupe = dedupe
        self.max_generations = max_generations
        self.stall_generations = stall_generations
        self.max_evals_per_deme = max_evals_per_deme
        self.telemetry_path = telemetry_path
        self.init_mode = init_mode
        self.acceptance = acceptance
        self.genotype_memo = bool(genotype_memo)
        self.stall_evals = stall_evals
        self.contin = contin
        self.contin_stats = contin_stats

    def as_dict(self) -> dict:
        return dict(vars(self))


class _OracleEvaluator:
    """The engine's ``Evaluator`` over one ``EvaluationOracle``.

    ``worse(a, b)`` is MOSES's own ``composite_score.__lt__``. Phenotype keys
    are only computed when ``skip_inactive`` is on. Time spent scoring and
    computing phenotype keys is accumulated for the call's timings.
    """

    def __init__(self, oracle: EvaluationOracle, schema: FrozenSchema, skip_inactive: bool):
        self.oracle = oracle
        self.schema = schema
        self.skip_inactive = skip_inactive
        self.eval_s = 0.0
        self.pkey_s = 0.0

    def check(self) -> str | None:
        return self.oracle.check()

    def evaluate(self, g: tuple[int, ...]) -> composite_score:
        t0 = time.perf_counter()
        try:
            return self.oracle.evaluate(g)
        finally:
            self.eval_s += time.perf_counter() - t0

    def phenotype_key(self, g: tuple[int, ...]) -> Hashable | None:
        if not self.skip_inactive:
            return None
        t0 = time.perf_counter()
        try:
            return self.schema.phenotype_key(g)
        finally:
            self.pkey_s += time.perf_counter() - t0

    def worse(self, a: composite_score, b: composite_score) -> bool:
        return a < b

    def record_noop_skip(self) -> None:
        self.oracle.record_noop_skip()

    def record_phenotype_skip(self) -> None:
        self.oracle.record_phenotype_skip()


def population_diversity(
    genotypes: Sequence[tuple[int, ...]], raws: Sequence[float], positions: Sequence[int]
) -> dict:
    """Diversity/coverage of one population over ``positions`` (the linkage
    model's search positions), for the per-generation telemetry:

    * ``unique_genotypes`` -- distinct genotypes;
    * ``mean_entropy_bits`` -- mean per-field Shannon entropy (base 2) over
      ``positions`` (0.0 when there are none);
    * ``n_varying_fields`` -- positions with more than one observed value;
    * ``best_raw_count`` -- individuals whose raw score equals the
      population's best (highest) raw score.

    One pass with a ``Counter`` per field.
    """
    n = len(genotypes)
    counters = [Counter() for _ in positions]
    for g in genotypes:
        for c, p in zip(counters, positions, strict=True):
            c[g[p]] += 1
    entropies = []
    for c in counters:
        h = 0.0
        for k in c.values():
            q = k / n
            h -= q * math.log2(q)
        entropies.append(h)
    best_raw = max(raws) if raws else None
    return {
        "unique_genotypes": len(set(genotypes)),
        "mean_entropy_bits": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "n_varying_fields": sum(1 for c in counters if len(c) > 1),
        "best_raw_count": sum(1 for r in raws if r == best_raw),
    }


def _draw_streams() -> tuple[MT19937RandGen, MT19937RandGen, MT19937RandGen]:
    """The init, linkage and variation streams, seeded from ``randGen()`` in
    that order."""
    rng_init = MT19937RandGen(randGen().randint())
    rng_link = MT19937RandGen(randGen().randint())
    rng_var = MT19937RandGen(randGen().randint())
    return rng_init, rng_link, rng_var


def _score_fields(score: composite_score) -> dict:
    return {
        "raw": score.get_score(),
        "penalized": score.get_penalized_score(),
        "complexity": score.get_complexity(),
    }


class gomea_optimization(optimizer_base):
    """GP-GOMEA-style population optimizer behind MOSES's optimizer seam.

    ``last_result`` describes the most recent call (termination reason,
    generations, ledger, init statistics, timings, best score); it is
    ``None`` before the first call.
    """

    def __init__(
        self, op: optim_parameters | None = None, gp: gomea_parameters | None = None
    ) -> None:
        super().__init__(op)
        self.gomea_params = gp if gp is not None else gomea_parameters()
        self.last_result: dict | None = None

    def __call__(self, deme: deme_t, iscorer, max_evals: int, max_time: int) -> None:
        """Fill the empty ``deme`` with every evaluated trial and set
        ``deme.n_evals``, then update ``optim_stats`` like hill climbing.

        Raises ``UnsupportedRepresentation`` (before drawing from the RNG or
        evaluating anything) for a representation with contin fields, unless
        ``contin`` is ``"copy"``.
        """
        t_start = time.perf_counter()
        gp = self.gomea_params
        schema = FrozenSchema(iscorer.get_rep(), gp.contin)

        rng_init, rng_link, rng_var = _draw_streams()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive)
        self._emit_start(sink, schema, deme, max_evals, max_time)

        t0 = time.perf_counter()
        if gp.init_mode == "uniform":
            init = build_uniform_population(schema, gp.pop_size, rng_init, gp.dedupe)
        else:
            init = build_population(schema, gp.pop_size, parse_radii(gp.radii), rng_init, gp.dedupe)
        init_s = time.perf_counter() - t0
        self._emit_init(sink, schema, gp.init_mode, init)
        self._evolve(deme, oracle, ev, schema, init, [], rng_var, rng_link, init_s, t_start, sink)

    # GOMEA extension, Phase 1e: the GOMEA phase of ``hc_gomea_optimization``.
    def run_seeded(
        self,
        deme: deme_t,
        iscorer,
        schema: FrozenSchema,
        max_evals: int,
        max_time: float,
    ) -> None:
        """Continue a deme another optimizer (hill climbing) already filled.

        Every instance already in ``deme`` is preloaded into the oracle
        (never charged, scored or appended again). The initial population is
        the ``pop_size`` best distinct genotypes among them, ranked by the
        MOSES composite comparator (earliest first on ties), with their
        existing scores; if there are fewer, it is topped up with MOSES
        sampler draws around the best one, which are evaluated normally.
        ``max_evals``/``max_time`` are what is left of the deme's budget;
        ``max_evals_per_deme`` and ``stall_evals`` apply on top. On return
        ``deme.n_evals`` is the earlier count plus this phase's charged
        evaluations. Draws its three RNG streams from ``randGen()`` first,
        exactly like ``__call__``.
        """
        t_start = time.perf_counter()
        gp = self.gomea_params
        rng_init, rng_link, rng_var = _draw_streams()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        prior = [(schema.to_genotype(si.first), si.second) for si in deme]
        oracle.preload(prior, deme.n_evals)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive)
        self._emit_start(sink, schema, deme, max_evals, max_time, prior_evals=deme.n_evals)

        t0 = time.perf_counter()
        ranked = sorted(range(len(prior)), key=lambda i: prior[i][1], reverse=True)
        seeds: list[tuple[int, ...]] = []
        seed_scores: list[composite_score] = []
        seen: set[tuple[int, ...]] = set()
        for i in ranked:
            g, score = prior[i]
            if g in seen:
                continue
            seen.add(g)
            seeds.append(g)
            seed_scores.append(score)
            if len(seeds) == gp.pop_size:
                break
        center = seeds[0] if seeds else schema.default_genotype()
        init = top_up_population(
            schema, seeds, center, gp.pop_size, parse_radii(gp.radii), rng_init, gp.dedupe
        )
        seeded = [
            Individual(g, copy.copy(score), ev.phenotype_key(g))
            for g, score in zip(seeds, seed_scores, strict=True)
        ]
        init_s = time.perf_counter() - t0
        self._emit_init(sink, schema, "deme", init, from_deme=len(seeded))
        self._evolve(
            deme, oracle, ev, schema, init, seeded, rng_var, rng_link, init_s, t_start, sink
        )

    def _make_oracle(
        self, schema: FrozenSchema, iscorer, deme: deme_t, max_evals: int, max_time: float
    ) -> EvaluationOracle:
        gp = self.gomea_params
        return EvaluationOracle(
            schema,
            iscorer,
            deme,
            max_evals,
            max_time,
            inner_cap=gp.max_evals_per_deme,
            genotype_memo=gp.genotype_memo,
            stall_evals=gp.stall_evals,
        )

    def _emit_start(
        self,
        sink: TelemetrySink,
        schema: FrozenSchema,
        deme: deme_t,
        max_evals: int,
        max_time: float,
        **extra,
    ) -> None:
        if schema.contin_positions:
            # GOMEA extension: paper-style constant genes.
            extra["n_contin_genes"] = len(schema.contin_positions)
        sink.emit(
            "invocation_start",
            schema_id=schema.schema_id,
            deme_id=deme.getID(),
            dim=schema.dim,
            domains=list(schema.domains),
            max_evals=max_evals,
            max_time=max_time,
            params={
                **self.gomea_params.as_dict(),
                "terminate_if_gte": self.opt_params.terminate_if_gte,
            },
            **extra,
        )

    def _emit_init(
        self, sink: TelemetrySink, schema: FrozenSchema, init_mode: str, init: InitResult, **extra
    ) -> None:
        sink.emit(
            "init",
            schema_id=schema.schema_id,
            init_mode=init_mode,
            n=len(init.genotypes),
            radii=init.radii,
            attempts=init.attempts,
            duplicates_rejected=init.duplicates_rejected,
            coverage_replacements=init.coverage_replacements,
            feasible_short=init.feasible_short,
            diagnostics=init.diagnostics,
            **extra,
        )

    def _evolve(
        self,
        deme: deme_t,
        oracle: EvaluationOracle,
        ev: _OracleEvaluator,
        schema: FrozenSchema,
        init: InitResult,
        seeded: list[Individual],
        rng_var: MT19937RandGen,
        rng_link: MT19937RandGen,
        init_s: float,
        t_start: float,
        sink: TelemetrySink,
    ) -> None:
        """Evaluate ``init.genotypes`` after the already-scored ``seeded``
        prefix, run the engine, and finalize (also on an interrupt)."""
        gp = self.gomea_params
        reason: str | None = None
        engine: GomeaEngine | None = None
        try:
            population: list[Individual] = list(seeded)
            for g in init.genotypes[len(seeded) :]:
                try:
                    score = ev.evaluate(g)
                except BudgetExhausted as exc:
                    reason = exc.reason
                    break
                population.append(Individual(g, score, ev.phenotype_key(g)))

            if reason is None:
                terminate_if_gte = self.opt_params.terminate_if_gte

                def on_generation(record: dict, pop: list[Individual]) -> None:
                    if sink.path is None:
                        return  # no telemetry file: skip the diversity pass
                    best = engine.best_of(pop)
                    diversity = population_diversity(
                        [ind.genotype for ind in pop],
                        [ind.score.get_score() for ind in pop],
                        engine.model.search_positions,
                    )
                    sink.emit(
                        "generation",
                        **record,
                        best_raw=best.score.get_score(),
                        best_penalized=best.score.get_penalized_score(),
                        best_complexity=best.score.get_complexity(),
                        ledger=oracle.ledger.as_dict(),
                        **diversity,
                    )

                engine = GomeaEngine(
                    schema.domains,
                    EngineParams(
                        fos_mode=gp.fos,
                        forced_improvement=gp.forced_improvement,
                        skip_inactive=gp.skip_inactive,
                        max_generations=gp.max_generations,
                        stall_generations=gp.stall_generations,
                        acceptance=gp.acceptance,
                    ),
                    ev,
                    rng_var,
                    rng_link,
                    target=lambda s: s.get_score() >= terminate_if_gte,
                    on_generation=on_generation,
                    symbolize=self._symbolizer(schema),
                )
                reason = engine.run(population).termination_reason
        except KeyboardInterrupt:
            self._finish(
                deme, oracle, ev, schema, init, engine, "interrupted", init_s, t_start, sink
            )
            raise
        self._finish(deme, oracle, ev, schema, init, engine, reason, init_s, t_start, sink)

    # GOMEA extension: paper-style constant genes.
    def _symbolizer(self, schema: FrozenSchema):
        """The engine's constant recoding for linkage learning, or ``None``
        when the schema has no contin genes (the plain model)."""
        positions = schema.contin_positions
        if not positions:
            return None
        mode = self.gomea_params.contin_stats
        return lambda pop: contin_symbols(pop, positions, schema.contin_value, mode)

    def _finish(
        self,
        deme: deme_t,
        oracle: EvaluationOracle,
        ev: _OracleEvaluator,
        schema: FrozenSchema,
        init: InitResult,
        engine: GomeaEngine | None,
        reason: str,
        init_s: float,
        t_start: float,
        sink: TelemetrySink,
    ) -> None:
        """Finalize the deme, set ``optim_stats`` and ``last_result``, and
        emit the ``end`` event."""
        oracle.finalize()
        result = engine.last_result if engine is not None else None
        history = result.history if result is not None else []
        generations = result.generations if result is not None else 0
        requests = oracle.ledger.requests

        self.nsteps = generations
        self.total_steps += generations
        self.total_evals += requests
        self.over_budget = reason in BUDGET_REASONS
        self.demeID = deme.getID()

        constant = init.diagnostics["constant_fields_after_coverage"]
        self.last_result = {
            "termination_reason": reason,
            "generations": generations,
            "ledger": oracle.ledger.as_dict(),
            "schema_id": schema.schema_id,
            "dim": schema.dim,
            "n_search_fields": schema.dim - len(constant),
            "n_constant_fields": len(constant),
            "pop_size_effective": len(init.genotypes),
            "init": {
                "attempts": init.attempts,
                "duplicates_rejected": init.duplicates_rejected,
                "coverage_replacements": init.coverage_replacements,
                "feasible_short": init.feasible_short,
                "radius_histogram": init.diagnostics["radius_histogram"],
            },
            "timings": {
                "init_s": init_s,
                "eval_s": ev.eval_s,
                "pkey_s": ev.pkey_s,
                "linkage_s": sum(r["linkage_s"] for r in history),
                "variation_s": sum(r["variation_s"] for r in history),
                "total_s": time.perf_counter() - t_start,
            },
            "best": _score_fields(oracle.best_so_far()) if requests > 0 else None,
        }
        sink.emit("end", **self.last_result)
