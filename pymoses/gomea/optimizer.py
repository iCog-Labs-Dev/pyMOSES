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
from fractions import Fraction

from pymoses.gomea.const_opt import (
    CONST_OPT_MODES,
    RegressionRows,
    find_contin_bscore,
    gaussian,
    levenberg_marquardt,
    mutation_sigma,
    scored_tree_instance,
)
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import FOS_TOPK_MODES
from pymoses.gomea.gom import ACCEPTANCE_MODES, GomInterrupted, Individual
from pymoses.gomea.knob_cap import KNOB_RANK0_MODES, KNOB_SELECT_MODES, default_explore
from pymoses.gomea.long_deme import LONG_DEME_RULES, LongDemePolicy
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
    "DEME_STORE_MODES",
    "TERMINATION_REASONS",
    "accept_worse",
    "gomea_optimization",
    "gomea_parameters",
    "population_diversity",
]

FOS_MODES = ("lt-b", "lt-mi", "rt", "uni")

# GOMEA extension, ``--gomea-deme-store``: which evaluated trials the deme
# hands back to MOSES (see ``gomea_parameters``).
DEME_STORE_MODES = ("all", "accepted")

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
    contin knobs.

    GOMEA extension, budget-derived sizes (all 0 = off by default, and then
    ``as_dict`` leaves them out, so a default run's telemetry is unchanged):
    ``pop_fraction`` (F > 0) makes each deme's population
    ``N = max(2, floor(F * deme_budget))``, overriding ``pop_size``, where
    ``deme_budget = min(max_evals_per_deme, max_evals)`` is what the deme
    may spend (``max_evals`` being what the caller hands this call);
    ``pop_max`` (M >= 2, 0 = no limit) then clamps ``N <= M``.
    ``deme_share`` (D) only records that ``max_evals_per_deme`` was derived
    as ``max(1, total_budget // D)`` (``_gomea_params_from_args`` does the
    arithmetic, since only it knows the run's ``-m``).

    GOMEA extension, deme-local complexity penalty (both off by default,
    and then left out of ``as_dict``, so a default run is unchanged):
    ``accept_complexity_ratio`` (R > 0; 0 = off) makes every comparison
    GOMEA makes inside a deme use ``accept_worse`` (``raw - complexity /
    R``, MOSES's ``-z`` semantics) instead of MOSES's composite comparator:
    GOM and forced-improvement acceptance, the elite, the engine's best
    and its stagnation test, the ``stall_evals`` rule and ``run_seeded``'s
    choice of seeds. What MOSES sees is unchanged: the scores stored in
    the deme, ``deme.n_best_evals``, ``last_result["best"]``, and MOSES's
    own merge and exemplar selection. ``deme_store`` (``"all"``, the
    default, or ``"accepted"``) chooses which evaluated trials stay in the
    deme handed back to MOSES: all of them, or only the initial population
    and the trials GOM/FI accepted (``EvaluationOracle``'s
    ``store_accepted``). Every evaluation is still charged and counted in
    ``deme.n_evals``.

    GOMEA extension, ``--gomea-fos-topk`` (off by default, and then left
    out of ``as_dict``): ``fos_topk`` K > 0 makes every generation's GOM
    (and forced improvement) visit only K FOS subsets, chosen by
    ``fos_topk_mode``: ``"linkage"`` (the K linkage-tree merges with the
    highest UPGMA similarity; needs ``fos`` ``lt-b`` or ``lt-mi``) or
    ``"random"`` (K drawn uniformly each generation). See
    ``fos.select_fos_topk`` and ``engine.perform_generation``.

    GOMEA extension, long demes for promising exemplars (Phase 1af, off by
    default, and then left out of ``as_dict``): ``long_deme_cap`` C > 0
    gives the deme of an exemplar chosen by ``long_deme_rule`` (``"top"``
    or ``"random"`` with ``long_deme_prob``) cap C instead of
    ``max_evals_per_deme``; see ``pymoses.gomea.long_deme``. N and
    ``deme_pop_size`` are unchanged.

    GOMEA extension, knob cap (server round 2, off by default, and then left
    out of ``as_dict``): ``knob_cap`` K > 0 builds each generation's FOS over K
    selected knobs only, chosen by ``knob_select`` (``"explore"``, ``"static"``
    or ``"random"``) with ``knob_explore`` E rotating knobs (``None``: ``ceil(K /
    4)``) and the generation-0 score ``knob_rank0`` (``"contrast"``,
    ``"entropy"`` or ``"random"``); see ``pymoses.gomea.knob_cap``. Its draws
    come from a fourth stream, seeded from ``randGen()`` after the other three
    and only when K > 0, so a K = 0 run draws exactly as before."""

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
        pop_fraction: float = 0.0,
        pop_max: int = 0,
        deme_share: int = 0,
        accept_complexity_ratio: float = 0.0,
        deme_store: str = "all",
        fos_topk: int = 0,
        fos_topk_mode: str = "linkage",
        long_deme_cap: int = 0,
        long_deme_rule: str = "top",
        long_deme_prob: float = 0.0,
        knob_cap: int = 0,
        knob_select: str = "explore",
        knob_explore: int | None = None,
        knob_rank0: str = "contrast",
        const_opt: str = "none",
        const_opt_val: str = "",
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
        if not pop_fraction >= 0:
            raise ValueError(f"pop_fraction must be >= 0 (0 = off), got {pop_fraction}")
        if pop_max != 0 and pop_max < 2:
            raise ValueError(f"pop_max must be 0 (no limit) or >= 2, got {pop_max}")
        if deme_share < 0:
            raise ValueError(f"deme_share must be >= 0 (0 = off), got {deme_share}")
        if not (accept_complexity_ratio >= 0 and math.isfinite(accept_complexity_ratio)):
            raise ValueError(
                "accept_complexity_ratio must be a finite number >= 0 (0 = off), "
                f"got {accept_complexity_ratio}"
            )
        if deme_store not in DEME_STORE_MODES:
            raise ValueError(f"deme_store must be one of {DEME_STORE_MODES}, got {deme_store!r}")
        if fos_topk < 0:
            raise ValueError(f"fos_topk must be >= 0 (0 = off), got {fos_topk}")
        if fos_topk_mode not in FOS_TOPK_MODES:
            raise ValueError(
                f"fos_topk_mode must be one of {FOS_TOPK_MODES}, got {fos_topk_mode!r}"
            )
        if fos_topk > 0 and fos_topk_mode == "linkage" and fos in ("uni", "rt"):
            raise ValueError(
                f"fos_topk_mode 'linkage' needs a linkage-tree fos (lt-b, lt-mi), got {fos!r}"
            )
        if long_deme_cap < 0:
            raise ValueError(f"long_deme_cap must be >= 0 (0 = off), got {long_deme_cap}")
        if long_deme_rule not in LONG_DEME_RULES:
            raise ValueError(
                f"long_deme_rule must be one of {LONG_DEME_RULES}, got {long_deme_rule!r}"
            )
        if not (0.0 <= long_deme_prob <= 1.0):
            raise ValueError(f"long_deme_prob must be in [0, 1], got {long_deme_prob}")
        if knob_cap < 0:
            raise ValueError(f"knob_cap must be >= 0 (0 = off), got {knob_cap}")
        if knob_select not in KNOB_SELECT_MODES:
            raise ValueError(f"knob_select must be one of {KNOB_SELECT_MODES}, got {knob_select!r}")
        if knob_rank0 not in KNOB_RANK0_MODES:
            raise ValueError(f"knob_rank0 must be one of {KNOB_RANK0_MODES}, got {knob_rank0!r}")
        if knob_cap > 0:
            if knob_explore is None:
                knob_explore = default_explore(knob_cap)
            if not 0 <= knob_explore <= knob_cap:
                raise ValueError(f"knob_explore must be in [0, {knob_cap}], got {knob_explore}")
        elif knob_explore is not None or knob_select != "explore" or knob_rank0 != "contrast":
            raise ValueError("knob_select, knob_explore and knob_rank0 need knob_cap > 0")
        if const_opt not in CONST_OPT_MODES:
            raise ValueError(f"const_opt must be one of {CONST_OPT_MODES}, got {const_opt!r}")
        if const_opt == "mutate" and contin != "copy":
            raise ValueError("const_opt 'mutate' needs contin 'copy' (constants as genes)")
        if const_opt == "polish" and not const_opt_val:
            raise ValueError("const_opt 'polish' needs const_opt_val (the validation table)")
        if const_opt_val and const_opt != "polish":
            raise ValueError("const_opt_val is used by const_opt 'polish' only")
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
        self.pop_fraction = pop_fraction
        self.pop_max = pop_max
        self.deme_share = deme_share
        self.accept_complexity_ratio = accept_complexity_ratio
        self.deme_store = deme_store
        self.fos_topk = fos_topk
        self.fos_topk_mode = fos_topk_mode
        self.long_deme_cap = long_deme_cap
        self.long_deme_rule = long_deme_rule
        self.long_deme_prob = long_deme_prob
        self.knob_cap = knob_cap
        self.knob_select = knob_select
        self.knob_explore = knob_explore
        self.knob_rank0 = knob_rank0
        self.const_opt = const_opt
        self.const_opt_val = const_opt_val

    # The opt-in options, left out of ``as_dict`` while at their default
    # (off), so a default run's telemetry is unchanged.
    _OPTIONAL_DEFAULTS = {
        "pop_fraction": 0.0,
        "pop_max": 0,
        "deme_share": 0,
        "accept_complexity_ratio": 0.0,
        "deme_store": "all",
        "fos_topk": 0,
        "fos_topk_mode": "linkage",
        "long_deme_cap": 0,
        "long_deme_rule": "top",
        "long_deme_prob": 0.0,
        "const_opt": "none",
        "const_opt_val": "",
    }

    # The knob-cap options: all left out of ``as_dict`` while ``knob_cap`` is 0
    # (off), all kept otherwise.
    _KNOB_KEYS = ("knob_cap", "knob_select", "knob_explore", "knob_rank0")

    def as_dict(self) -> dict:
        d = dict(vars(self))
        for k, default in self._OPTIONAL_DEFAULTS.items():
            if d[k] == default:
                del d[k]
        if self.knob_cap == 0:
            for k in self._KNOB_KEYS:
                del d[k]
        return d

    def budget_derived(self) -> bool:
        """Whether any budget-derived option (``pop_fraction``,
        ``deme_share``) is on."""
        return self.pop_fraction > 0 or self.deme_share > 0

    def deme_budget(self, max_evals: int) -> int:
        """The evaluations one deme may spend: ``max_evals`` (what is left
        of the run's budget for this call), capped at ``max_evals_per_deme``
        when one is set."""
        if self.max_evals_per_deme is None:
            return max_evals
        return min(max_evals, self.max_evals_per_deme)

    def deme_pop_size(self, max_evals: int) -> int:
        """The population size N of a deme handed ``max_evals``: ``pop_size``
        when ``pop_fraction`` is off, else ``max(2, floor(pop_fraction *
        deme_budget(max_evals)))`` clamped to ``pop_max`` when set. The
        product is taken exactly (on the decimal value of ``pop_fraction``),
        so 0.29 x 100 is 29, not 28."""
        if self.pop_fraction <= 0:
            return self.pop_size
        budget = max(0, self.deme_budget(max_evals))
        n = max(2, math.floor(Fraction(repr(float(self.pop_fraction))) * budget))
        if self.pop_max:
            n = min(n, self.pop_max)
        return n


def accept_worse(a: composite_score, b: composite_score, ratio: float) -> bool:
    """GOMEA extension (``--gomea-accept-complexity-ratio``): "``a`` strictly
    worse than ``b``" under a deme-local complexity penalty.

    Each score is ranked by ``raw - complexity / ratio``, the same form as
    MOSES's ``-z`` (a penalty of ``1 / ratio`` per complexity unit), with
    MOSES's own complexity and uniformity penalties left out. Otherwise it
    mirrors ``composite_score.__lt__`` exactly: on equal penalized values
    the higher complexity is worse, and NaN is worse than anything except
    NaN. Computed in double precision. ``ratio`` must be > 0.
    """
    lef = a.get_score() - a.get_complexity() / ratio
    rig = b.get_score() - b.get_complexity() / ratio
    if math.isnan(lef):
        return not math.isnan(rig)
    return (lef < rig) or (lef == rig and a.get_complexity() > b.get_complexity())


def _new_const_opt_counts() -> dict:
    """GOMEA extension (``--const-opt``): one call's tuning counts."""
    return {
        "lm_calls": 0,
        "lm_improved": 0,
        "lm_not_improved": 0,
        "lm_singular": 0,
        "lm_failed": 0,
        "lm_no_constants": 0,
        "lm_budget": 0,
        "mutate_attempts": 0,
        "mutate_accepted": 0,
        "tuning_evals": 0,
        "lm": None,
    }


def _const_opt_mode(opt) -> str:
    return opt.gomea_params.const_opt


class _WorseKey:
    """A sort key ordering scores by an evaluator's ``worse``."""

    __slots__ = ("score", "worse")

    def __init__(self, score: composite_score, worse) -> None:
        self.score = score
        self.worse = worse

    def __lt__(self, other: _WorseKey) -> bool:
        return self.worse(self.score, other.score)


class _OracleEvaluator:
    """The engine's ``Evaluator`` over one ``EvaluationOracle``.

    ``worse(a, b)`` is MOSES's own ``composite_score.__lt__``, or
    ``accept_worse`` with ``accept_complexity_ratio > 0``. Phenotype keys
    are only computed when ``skip_inactive`` is on. Time spent scoring and
    computing phenotype keys is accumulated for the call's timings.
    """

    def __init__(
        self,
        oracle: EvaluationOracle,
        schema: FrozenSchema,
        skip_inactive: bool,
        accept_complexity_ratio: float = 0.0,
    ):
        self.oracle = oracle
        self.schema = schema
        self.skip_inactive = skip_inactive
        self.ratio = accept_complexity_ratio
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
        if self.ratio > 0:
            return accept_worse(a, b, self.ratio)
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


def _knob_stream(gp: gomea_parameters) -> MT19937RandGen | None:
    """GOMEA extension (``--gomea-knob-cap``): the knob-cap stream, seeded from
    ``randGen()`` after the other three; ``None`` (and nothing drawn) when the
    cap is off."""
    if gp.knob_cap <= 0:
        return None
    return MT19937RandGen(randGen().randint())


def _mutate_stream(gp: gomea_parameters) -> MT19937RandGen | None:
    """GOMEA extension (``--const-opt mutate``): the coefficient-mutation
    stream, seeded from ``randGen()`` after the others; ``None`` (and nothing
    drawn) for every other mode."""
    if gp.const_opt != "mutate":
        return None
    return MT19937RandGen(randGen().randint())


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
        # GOMEA extension, Phase 1af: long demes (None when off). ``_long``
        # is the decision ``prepare_deme`` made for the next deme(s).
        g = self.gomea_params
        self.long_deme: LongDemePolicy | None = (
            LongDemePolicy(g.long_deme_cap, g.long_deme_rule, g.long_deme_prob)
            if g.long_deme_cap > 0
            else None
        )
        self._long = False
        # GOMEA extension (``--gomea-knob-cap``): the current call's knob-cap
        # stream (``None`` when off).
        self._rng_knob: MT19937RandGen | None = None
        # GOMEA extension (``--const-opt``): the current call's mutation
        # stream (``None`` unless ``mutate``) and tuning counts.
        self._rng_mut: MT19937RandGen | None = None
        self._co: dict = _new_const_opt_counts()

    def prepare_deme(self, mp, exemplar) -> None:
        """GOMEA extension, Phase 1af: called by ``expand_deme`` once the
        deme of ``exemplar`` (just chosen from the metapopulation ``mp``) is
        built and before it is optimized. Decides whether it is a long deme.
        Does nothing (and draws nothing) when the option is off."""
        if self.long_deme is not None:
            self._long = self.long_deme.decide(mp, exemplar)

    def deme_cap(self) -> int | None:
        """The cap of the deme being optimized: ``long_deme_cap`` for a long
        deme, else ``max_evals_per_deme``."""
        if self._long:
            return self.gomea_params.long_deme_cap
        return self.gomea_params.max_evals_per_deme

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
        self._rng_knob = _knob_stream(gp)
        self._rng_mut = _mutate_stream(gp)
        self._co = _new_const_opt_counts()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive, gp.accept_complexity_ratio)
        n = gp.deme_pop_size(max_evals)
        self._emit_start(sink, schema, deme, max_evals, max_time)

        t0 = time.perf_counter()
        if gp.init_mode == "uniform":
            init = build_uniform_population(schema, n, rng_init, gp.dedupe)
        else:
            init = build_population(schema, n, parse_radii(gp.radii), rng_init, gp.dedupe)
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
        the N best distinct genotypes among them (N = ``deme_pop_size``),
        ranked by the engine's comparator (MOSES's composite comparator, or
        ``accept_worse`` with ``accept_complexity_ratio``; earliest first on
        ties),
        with their existing scores; if there are fewer, it is topped up with MOSES
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
        self._rng_knob = _knob_stream(gp)
        self._rng_mut = _mutate_stream(gp)
        self._co = _new_const_opt_counts()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        prior = [(schema.to_genotype(si.first), si.second) for si in deme]
        oracle.preload(prior, deme.n_evals)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive, gp.accept_complexity_ratio)
        n = gp.deme_pop_size(max_evals)
        self._emit_start(sink, schema, deme, max_evals, max_time, prior_evals=deme.n_evals)

        t0 = time.perf_counter()
        ranked = sorted(
            range(len(prior)), key=lambda i: _WorseKey(prior[i][1], ev.worse), reverse=True
        )
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
            if len(seeds) == n:
                break
        center = seeds[0] if seeds else schema.default_genotype()
        init = top_up_population(
            schema, seeds, center, n, parse_radii(gp.radii), rng_init, gp.dedupe
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
            inner_cap=self.deme_cap(),
            genotype_memo=gp.genotype_memo,
            stall_evals=gp.stall_evals,
            stall_worse=self._accept_worse(),
            store_accepted=gp.deme_store == "accepted",
        )

    def _accept_worse(self):
        """The deme-local comparator for ``accept_complexity_ratio > 0``,
        ``None`` when it is off (MOSES's comparator)."""
        ratio = self.gomea_params.accept_complexity_ratio
        if ratio <= 0:
            return None
        return lambda a, b: accept_worse(a, b, ratio)

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
        gp = self.gomea_params
        if gp.budget_derived():
            # GOMEA extension, budget-derived sizes: this deme's effective
            # budget, cap and population (absent when the options are off).
            extra["deme_budget"] = gp.deme_budget(max_evals)
            extra["deme_cap"] = gp.max_evals_per_deme
            extra["deme_pop_size"] = gp.deme_pop_size(max_evals)
        if self.long_deme is not None:
            # GOMEA extension, Phase 1af (absent when the option is off).
            extra.update(self._long_deme_record())
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
                oracle.keep(g)
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
                        fos_topk=gp.fos_topk,
                        fos_topk_mode=gp.fos_topk_mode,
                        knob_cap=gp.knob_cap,
                        knob_select=gp.knob_select,
                        knob_explore=gp.knob_explore,
                        knob_rank0=gp.knob_rank0,
                    ),
                    ev,
                    rng_var,
                    rng_link,
                    target=lambda s: s.get_score() >= terminate_if_gte,
                    on_generation=on_generation,
                    symbolize=self._symbolizer(schema),
                    on_accept=oracle.keep if gp.deme_store == "accepted" else None,
                    rng_knob=self._rng_knob,
                    fitness=self._fitness(),
                    mutate=self._mutator(schema, oracle, ev),
                )
                reason = engine.run(population).termination_reason
        except KeyboardInterrupt:
            self._finish(
                deme, oracle, ev, schema, init, engine, "interrupted", init_s, t_start, sink
            )
            raise
        self._finish(deme, oracle, ev, schema, init, engine, reason, init_s, t_start, sink)

    def _fitness(self):
        """GOMEA extension (``--gomea-knob-cap``): a score's scalar fitness for the
        generation-0 contrast, as the deme ranks it: the penalized score, or ``raw -
        complexity / R`` with ``accept_complexity_ratio`` R > 0."""
        ratio = self.gomea_params.accept_complexity_ratio
        if ratio > 0:
            return lambda s: s.get_score() - s.get_complexity() / ratio
        return lambda s: s.get_penalized_score()

    def _long_deme_record(self) -> dict:
        """Phase 1af telemetry, flat keys on ``invocation_start`` and ``end``:
        whether this deme is long (``long``), the cap it ran with
        (``deme_cap_used``), the rule, and the run's counts so far, this deme
        included (``n_demes``, ``n_long_demes``; the last deme's values are the
        run's totals)."""
        pol = self.long_deme
        return {
            "long": self._long,
            "deme_cap_used": self.deme_cap(),
            "long_deme_rule": pol.rule,
            "n_demes": pol.n_demes,
            "n_long_demes": pol.n_long,
        }

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
        if _const_opt_mode(self) == "lm" and reason != "interrupted":
            self._tune_best(deme, oracle, schema)
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
        if self.long_deme is not None:
            # GOMEA extension, Phase 1af (absent when the option is off).
            self.last_result.update(self._long_deme_record())
        if self.gomea_params.knob_cap > 0:
            # GOMEA extension, knob cap (absent when the option is off).
            knobs = engine.knobs if engine is not None else None
            self.last_result["knob_cap"] = self.gomea_params.knob_cap
            self.last_result["knob_core_entered_total"] = (
                knobs.entered_total if knobs is not None else 0
            )
        if _const_opt_mode(self) != "none":
            # GOMEA extension (``--const-opt``; absent when it is none).
            self.last_result["const_opt"] = {"mode": _const_opt_mode(self), **self._co}
        sink.emit("end", **self.last_result)

    # GOMEA extension (``--const-opt mutate``): coefficient mutation.
    def _mutator(self, schema: FrozenSchema, oracle: EvaluationOracle, ev: _OracleEvaluator):
        """The engine's ``mutate`` hook, or ``None`` unless ``mutate``.

        After each GOM step, one Gaussian change ``c + N(0, sigma)``, sigma =
        ``max(0.1 |c|, 0.1)``, to each constant of the offspring in turn (each
        active contin gene, in gene order), kept if the score is not worse
        under the deme's comparator. Each change is one evaluation through the
        oracle (the deme cap applies). The new value is a free contin value
        (``FrozenSchema.register_contin_value``). A ``BudgetExhausted`` is
        raised as ``GomInterrupted`` with the last kept state."""
        if self.gomea_params.const_opt != "mutate":
            return None
        rng = self._rng_mut
        co = self._co
        keep = oracle.keep if self.gomea_params.deme_store == "accepted" else None

        def mutate(ind: Individual) -> Individual:
            cur_g, cur_s, cur_k = ind.genotype, ind.score, ind.pkey
            before = oracle.ledger.requests
            try:
                for p in schema.active_contin(cur_g):
                    c = schema.contin_value(p, cur_g[p])
                    code = schema.register_contin_value(p, c + mutation_sigma(c) * gaussian(rng))
                    if code == cur_g[p]:
                        continue
                    trial = cur_g[:p] + (code,) + cur_g[p + 1 :]
                    try:
                        score = ev.evaluate(trial)
                    except BudgetExhausted as exc:
                        raise GomInterrupted(Individual(cur_g, cur_s, cur_k), exc.reason) from exc
                    co["mutate_attempts"] += 1
                    if ev.worse(score, cur_s):
                        continue
                    co["mutate_accepted"] += 1
                    cur_g, cur_s, cur_k = trial, score, ev.phenotype_key(trial)
                    if keep is not None:
                        keep(trial)
            finally:
                co["tuning_evals"] += oracle.ledger.requests - before
            return Individual(cur_g, cur_s, cur_k)

        return mutate

    # GOMEA extension (``--const-opt lm``): elite-only LM at the end of a deme.
    def _tune_best(self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema) -> None:
        """LM on the deme's best program (``const_opt.levenberg_marquardt``),
        after ``oracle.finalize``. Every pass over the training rows is charged
        through ``oracle.charge_tuning`` (the run's budget; not the deme cap)
        and added to ``deme.n_evals``. If LM lowered the training MSE, the
        tuned program, scored (one more pass), replaces the best program's
        entries in the deme (Lamarckian). A singular or failed call keeps the
        old program; the call's record is logged in ``const_opt.lm``."""
        co = self._co
        g = oracle.best_genotype()
        if g is None:
            return
        iscorer = oracle.iscorer
        cscorer = iscorer._cscorer
        rows = RegressionRows.of_bscore(find_contin_bscore(cscorer))
        tree = schema.candidate(g, getattr(iscorer, "_reduce", True))
        before = oracle.ledger.requests
        lm = levenberg_marquardt(tree, rows, oracle.charge_tuning)
        rec = lm.record()
        if lm.tree is not None:
            try:
                oracle.charge_tuning()
                score = cscorer.get_cscore(lm.tree)
            except BudgetExhausted:
                rec["budget"] = True
                rec["written_back"] = False
            else:
                inst = schema.to_instance(g)
                deme[:] = [
                    si
                    for si in deme
                    if not (si.first == inst and getattr(si, "tree", None) is None)
                ]
                deme.append(scored_tree_instance(inst, score, lm.tree))
                rec["written_back"] = True
        used = oracle.ledger.requests - before
        deme.n_evals += used
        rec["evals"] = used
        co["tuning_evals"] += used
        co["lm_calls"] += 1
        co["lm_" + lm.status] += 1
        co["lm"] = rec
