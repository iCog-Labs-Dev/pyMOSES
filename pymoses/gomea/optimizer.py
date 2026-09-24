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

import time
from collections.abc import Hashable

from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.gom import Individual
from pymoses.gomea.oracle import BudgetExhausted, EvaluationOracle
from pymoses.gomea.population import InitResult, build_population, parse_radii
from pymoses.gomea.schema import FrozenSchema
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymoses.moses.moses.types import composite_score
from pymoses.util.rng import MT19937RandGen, randGen

__all__ = ["TERMINATION_REASONS", "gomea_optimization", "gomea_parameters"]

FOS_MODES = ("lt-b", "lt-mi", "rt", "uni")

BUDGET_REASONS = frozenset({"global_budget", "inner_budget", "time_limit"})

# Every value ``last_result["termination_reason"]`` can take: the engine's own
# reasons (``GomeaEngine``'s docstring) plus the oracle's budget reasons,
# which can also end a call during the initial evaluation.
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
)


class gomea_parameters:
    """Settings for ``gomea_optimization`` (the counterpart of
    ``hc_parameters``). Validated on construction; raises ``ValueError``."""

    def __init__(
        self,
        pop_size: int = 64,
        fos: str = "lt-b",
        forced_improvement: bool = False,
        skip_inactive: bool = True,
        radii: str = "1:0.4,2:0.3,4:0.2,max:0.1",
        dedupe: bool = True,
        max_generations: int = 100,
        stall_generations: int = 5,
        max_evals_per_deme: int | None = None,
        telemetry_path: str | None = None,
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
        evaluating anything) for a representation with contin fields.
        """
        t_start = time.perf_counter()
        gp = self.gomea_params
        schema = FrozenSchema(iscorer.get_rep())

        rng_init = MT19937RandGen(randGen().randint())
        rng_link = MT19937RandGen(randGen().randint())
        rng_var = MT19937RandGen(randGen().randint())

        sink = TelemetrySink(gp.telemetry_path)
        oracle = EvaluationOracle(
            schema, iscorer, deme, max_evals, max_time, inner_cap=gp.max_evals_per_deme
        )
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive)
        sink.emit(
            "invocation_start",
            schema_id=schema.schema_id,
            deme_id=deme.getID(),
            dim=schema.dim,
            domains=list(schema.domains),
            max_evals=max_evals,
            max_time=max_time,
            params={**gp.as_dict(), "terminate_if_gte": self.opt_params.terminate_if_gte},
        )

        t0 = time.perf_counter()
        init = build_population(schema, gp.pop_size, parse_radii(gp.radii), rng_init, gp.dedupe)
        init_s = time.perf_counter() - t0
        sink.emit(
            "init",
            schema_id=schema.schema_id,
            n=len(init.genotypes),
            radii=init.radii,
            attempts=init.attempts,
            duplicates_rejected=init.duplicates_rejected,
            coverage_replacements=init.coverage_replacements,
            feasible_short=init.feasible_short,
            diagnostics=init.diagnostics,
        )

        reason: str | None = None
        engine: GomeaEngine | None = None
        try:
            population: list[Individual] = []
            for g in init.genotypes:
                try:
                    score = ev.evaluate(g)
                except BudgetExhausted as exc:
                    reason = exc.reason
                    break
                population.append(Individual(g, score, ev.phenotype_key(g)))

            if reason is None:
                terminate_if_gte = self.opt_params.terminate_if_gte

                def on_generation(record: dict, pop: list[Individual]) -> None:
                    best = engine.best_of(pop)
                    sink.emit(
                        "generation",
                        **record,
                        best_raw=best.score.get_score(),
                        best_penalized=best.score.get_penalized_score(),
                        best_complexity=best.score.get_complexity(),
                        ledger=oracle.ledger.as_dict(),
                    )

                engine = GomeaEngine(
                    schema.domains,
                    EngineParams(
                        fos_mode=gp.fos,
                        forced_improvement=gp.forced_improvement,
                        skip_inactive=gp.skip_inactive,
                        max_generations=gp.max_generations,
                        stall_generations=gp.stall_generations,
                    ),
                    ev,
                    rng_var,
                    rng_link,
                    target=lambda s: s.get_score() >= terminate_if_gte,
                    on_generation=on_generation,
                )
                reason = engine.run(population).termination_reason
        except KeyboardInterrupt:
            self._finish(
                deme, oracle, ev, schema, init, engine, "interrupted", init_s, t_start, sink
            )
            raise
        self._finish(deme, oracle, ev, schema, init, engine, reason, init_s, t_start, sink)

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
