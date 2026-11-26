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
import hashlib
import json
import math
import operator
import time
from collections import Counter
from collections.abc import Hashable, Sequence
from fractions import Fraction

from pymoses.gomea.best_genes import best_genes_record, scorer_fit
from pymoses.gomea.const_opt import (
    CONST_OPT_DURING_MODES,
    CONST_OPT_GUARDS,
    CONST_OPT_MODES,
    POLE_CHECK_MODES,
    RegressionRows,
    constant_nodes,
    find_contin_bscore,
    gaussian,
    guard_accepts,
    levenberg_marquardt,
    load_rows,
    mutation_sigma,
    pole_check,
    pole_probes,
    scored_tree_instance,
    structure_key,
)
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import FOS_TOPK_MODES
from pymoses.gomea.gom import ACCEPTANCE_MODES, GomInterrupted, Individual
from pymoses.gomea.knob_cap import KNOB_RANK0_MODES, KNOB_SELECT_MODES, default_explore
from pymoses.gomea.long_deme import LONG_DEME_RULES, LongDemePolicy
from pymoses.gomea.node_proximity import distance_fn
from pymoses.gomea.oracle import BudgetExhausted, EvaluationOracle
from pymoses.gomea.population import (
    INIT_MODES,
    InitResult,
    build_population,
    build_uniform_population,
    parse_radii,
    top_up_population,
)
from pymoses.gomea.rv_core import RV_BUDGETS, RV_MODES, RVCore, rv_stream
from pymoses.gomea.schema import CONTIN_MODES, FrozenSchema
from pymoses.gomea.stats import CONTIN_STATS_MODES, contin_symbols
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymoses.moses.moses.types import composite_score
from pymoses.moses.scoring.behave_cscore import _cache_key
from pymoses.util.rng import MT19937RandGen, randGen

__all__ = [
    "DEME_STORE_MODES",
    "DYNAMIC_POP_SIZE",
    "TERMINATION_REASONS",
    "accept_worse",
    "gomea_optimization",
    "gomea_parameters",
    "population_diversity",
]

# GOMEA extension, server round 6 (``--gomea-pop-size dynamic``): the ``pop_size``
# value that sizes each deme's population with classic hill climbing's count.
DYNAMIC_POP_SIZE = "dynamic"


def dynamic_pop_size(fields, evals_left: int, fraction_of_nn: float, max_nn_evals: int) -> int:
    """GOMEA extension, server round 6: the population size of a deme under
    ``--gomea-pop-size dynamic``.

    The count is the number of new instances classic hill climbing takes at
    distance 1 from the exemplar: ``hill_climbing.estimate_neighborhood(1, fields)``
    and then ``hill_climbing.n_new_instances(1, ...)``, with ``fraction_of_nn``
    (``--hc-fraction-of-nn``), ``max_nn_evals`` (``--hc-max-nn-evals``), the classic
    minimum (``_MINIMUM_DEME_SIZE``) and the evaluations left. The classic functions
    are called, not copied, on an ``hc_parameters`` whose two fields are assigned as
    ``problem_params`` assigns them (so a value the CLI accepts is accepted here).

    DIVERGENCE (from classic hill climbing's own call, hill-climbing.cc:110): classic
    passes ``current_number_of_evals`` = the evaluations its deme already spent (1,
    the exemplar, at its first distance-1 step). A GOMEA deme has spent none when its
    population is sized, so this passes 0 and ``evals_left`` is all of the deme's
    budget. The result is at least 2, because GOM needs a donor (as
    ``--gomea-pop-fraction`` does); classic's count is below 2 only when fewer than
    2 evaluations are left.
    """
    hc = hc_parameters()
    hc.fraction_of_nn = fraction_of_nn
    hc.max_nn_evals = max_nn_evals
    climber = hill_climbing(None, hc)
    nn = climber.estimate_neighborhood(1, fields)
    return max(2, climber.n_new_instances(1, max(0, evals_left), 0, nn))


# ``np`` (server round 12): node-proximity linkage, see ``fos.LinkageModel``.
FOS_MODES = ("lt-b", "lt-mi", "rt", "uni", "np")

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
    highest UPGMA similarity; needs ``fos`` ``lt-b``, ``lt-mi`` or ``np``)
    or ``"random"`` (K drawn uniformly each generation). See
    ``fos.select_fos_topk`` and ``engine.perform_generation``.

    NOT A PORT, node-proximity linkage (server round 12, T2.1): ``fos``
    ``"np"`` builds each deme's FOS once, as a linkage tree on the tree
    distance between the exemplar nodes of the knobs (``fos.LinkageModel``,
    ``pymoses.gomea.node_proximity``). It refuses ``knob_cap`` > 0 (the
    knob cap picks the knobs of each generation from population statistics,
    so the FOS would not be static). With top-k ``"linkage"`` it keeps the
    K merges with the smallest mean distance.

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
    and only when K > 0, so a K = 0 run draws exactly as before.

    GOMEA extension, dynamic population (server round 6, off by default): ``pop_size``
    ``"dynamic"`` sizes each deme's population with classic hill climbing's count at
    distance 1 (``dynamic_pop_size``), from the deme's field set, its budget
    (``deme_budget``), ``hc_fraction_of_nn`` and ``hc_max_nn_evals``. The two ``hc_``
    options are left out of ``as_dict`` unless ``pop_size`` is ``"dynamic"``, so a
    fixed-N run's telemetry is unchanged. ``"dynamic"`` refuses ``pop_fraction``
    (both size the population).

    NOT A PORT, LM reserve (server round 11, off by default, and then left out of
    ``as_dict``): with ``const_opt`` ``"lm"`` and ``const_opt_reserve`` R > 0, the search
    of each deme stops at the deme budget minus ``lm_reserve(deme budget)`` = min(R,
    deme budget // 2) evaluations, so that LM on the deme's best program can use the
    evaluations that are left (the rule of ``hc_lm.lm_reserve``). A deme that ends
    before that point is not changed. N still comes from the full deme budget.

    NOT A PORT, server round 12 (T1.6, off by default, and then left out of ``as_dict``):
    with ``const_opt`` ``"lm"`` and ``const_opt_top_k`` K > 0, LM at the end of each deme
    tunes up to K programs in place of the best one (``_tune_top_k``): the deme's best
    program (the one that K = 0 tunes), then the deme entries in MOSES score order (ties:
    entry order) whose structure (``const_opt.structure_key`` of the reduced tree) is new,
    until K structures are taken. Each one gets the LM, guard, charges and write-back of
    ``_tune_best``. ``-a gomea`` only.

    NOT A PORT, server round 12 (T2.3, off by default, and then left out of ``as_dict``):
    with ``const_opt`` ``"lm"``, ``contin`` ``"copy"`` and ``const_opt_during`` ``"elite"``,
    each GOM offspring that is a new deme best (a strict improvement under the MOSES score)
    gets LM with the guard during the search (``_elite_tuner``). LM tunes the constants of its
    unreduced program that are not 0, which are its contin genes; the tuned values go back
    into its genotype as free contin codes, and the tuned offspring replaces it if it is not
    worse under the deme's comparator. These passes obey the deme cap and the stall rule
    (``EvaluationOracle.charge_search_tuning``). LM at the end of the deme is that of
    ``_tune_best``, with a write-back that also handles a best program with free contin codes
    (``_tune_deme_best``). ``-a gomea`` only; not with ``const_opt_top_k``.

    NOT A PORT, server round 13 (BASE-PS, off by default, and then left out of ``as_dict``):
    with ``const_opt`` ``"lm"`` and ``const_opt_pole_check`` ``"box"``, after LM at the end of
    a deme improves a program, ``const_opt.pole_check`` predicts the old and the tuned program
    on 256 probe points in the box of the training inputs (each range widened by 10 % on each
    side), each with its linear scaling fitted on the training rows. It rejects the tuned
    program (the old one is kept, as for a guard rejection) if a probe prediction of it is not
    finite while all of the old one are finite, or if its largest distance from the mean
    training target is above 10 times the range of the training target and above that of the
    old program. The check runs before the guard; each of its two probe passes is charged as
    one evaluation, as the LM passes are. The probes are drawn once per optimizer (one run),
    at its first LM call, from a new generator seeded by ``const_opt_pole_seed`` (the run seed,
    ``-r``), so no other random stream changes. ``const_opt_pole_seed`` is left out of
    ``as_dict`` while the check is off. ``-a gomea`` only; not with ``const_opt_top_k`` or
    ``const_opt_during``.

    NOT A PORT, server round 14 (NP-LMR-PSR): ``const_opt_pole_check`` ``"box-rel"`` is
    ``"box"`` with a relative size rule. The probes, the place, the charges, the non-finite
    rule, the write-back and the telemetry do not change. The tuned program is rejected if its
    largest distance from the mean training target is above 10 times the larger of that of
    the old program and the range of the training target (``const_opt.pole_rejects``). The
    same refusals as ``"box"`` apply.

    NOT A PORT, T3.2 (server round 16, off by default, and then left out of ``as_dict``):
    with ``rv_step`` ``"rvg"``, after each GOM generation the RV-GOMEA core
    (``pymoses.gomea.rv_core.RVCore``) can run a real-valued generation on the non-zero
    constant genes. ``rv_budget`` is its budget rule: ``"balance"`` or ``"late"``. With the
    hook, the engine's convergence test is the structure-only test of the RV core. Each deme
    gets its own RV stream, ``rv_core.rv_stream(rv_seed, k)`` for the k-th GOMEA deme of the
    optimizer (k = 0, 1, ...). ``rv_seed`` is the run seed (``-r``). The stream does not draw
    from the global ``randGen()``, so everything up to the first RV generation is the same as
    with ``"off"``. ``rv_step`` ``"rvg"`` needs ``contin`` ``"copy"`` and ``const_opt``
    ``"none"``. The CLI also refuses it without ``-a gomea`` and ``--linear-scaling terms``,
    and ``hc_gomea_optimization`` refuses it (the RV16 review). ``rv_budget`` and ``rv_seed``
    are left out of ``as_dict`` while ``rv_step`` is ``"off"``.

    NOT A PORT, server round 16 (the RV16 review, finding 1; off by default, and then left
    out of ``as_dict``): with ``trace`` on, the oracle of each deme
    (``EvaluationOracle.trace``) records an ``[evaluation, raw score]`` pair for each charged
    evaluation that gives a new best raw score of the deme. The "end" event gets the pairs as
    ``trace``, and the "init" event gets ``genotypes_sha256`` (``init_digest``). The search
    does not change. Not with ``const_opt`` ``"lm"``: LM charges its passes without a score.
    The CLI also refuses it without ``-a gomea``, and ``hc_gomea_optimization`` refuses it.

    NOT A PORT, backlog 15.4 (diagnostic D1; off by default, and always left out of
    ``as_dict``): with ``log_best_genes`` on and a telemetry file, the "end" event of each deme
    gets ``best_genes``: the programs and the contin genes of the deme's best individual (the
    one that ``best`` describes; ``pymoses.gomea.best_genes``). It only reads, so the search,
    ``last_result`` and every other event do not change. The CLI refuses it without ``-a
    gomea`` or without ``--gomea-log``, and ``hc_gomea_optimization`` refuses it."""

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
        const_opt_guard: str = "none",
        hc_fraction_of_nn: float = 2.0,
        hc_max_nn_evals: int = 20000,
        const_opt_reserve: int = 0,
        const_opt_top_k: int = 0,
        const_opt_during: str = "none",
        const_opt_pole_check: str = "none",
        const_opt_pole_seed: int = 0,
        rv_step: str = "off",
        rv_budget: str = "balance",
        rv_seed: int = 0,
        trace: bool = False,
        log_best_genes: bool = False,
    ) -> None:
        if pop_size == DYNAMIC_POP_SIZE:
            if pop_fraction > 0:
                raise ValueError(
                    "pop_size 'dynamic' and pop_fraction both size the population: choose one"
                )
        elif not isinstance(pop_size, int) or isinstance(pop_size, bool):
            raise ValueError(f"pop_size must be an integer >= 2 or 'dynamic', got {pop_size!r}")
        elif pop_size < 2:
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
                f"fos_topk_mode 'linkage' needs a linkage-tree fos (lt-b, lt-mi, np), got {fos!r}"
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
        if knob_cap > 0 and fos == "np":
            raise ValueError("fos 'np' (a static FOS) does not support knob_cap > 0")
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
        if const_opt_guard not in CONST_OPT_GUARDS:
            raise ValueError(
                f"const_opt_guard must be one of {CONST_OPT_GUARDS}, got {const_opt_guard!r}"
            )
        if const_opt_guard == "val" and const_opt != "lm":
            raise ValueError("const_opt_guard 'val' needs const_opt 'lm'")
        if const_opt_guard == "val" and not const_opt_val:
            raise ValueError("const_opt_guard 'val' needs const_opt_val (the validation table)")
        if const_opt_val and const_opt != "polish" and const_opt_guard != "val":
            raise ValueError(
                "const_opt_val is used by const_opt 'polish' and const_opt_guard 'val' only"
            )
        if (not isinstance(const_opt_reserve, int) or isinstance(const_opt_reserve, bool)
                or const_opt_reserve < 0):
            raise ValueError(
                f"const_opt_reserve must be an integer >= 0 (0 = off), got {const_opt_reserve!r}"
            )
        if const_opt_reserve > 0 and const_opt != "lm":
            raise ValueError("const_opt_reserve needs const_opt 'lm'")
        if (not isinstance(const_opt_top_k, int) or isinstance(const_opt_top_k, bool)
                or const_opt_top_k < 0):
            raise ValueError(
                f"const_opt_top_k must be an integer >= 0 (0 = off), got {const_opt_top_k!r}"
            )
        if const_opt_top_k > 0 and const_opt != "lm":
            raise ValueError("const_opt_top_k needs const_opt 'lm'")
        if const_opt_during not in CONST_OPT_DURING_MODES:
            raise ValueError(
                f"const_opt_during must be one of {CONST_OPT_DURING_MODES}, "
                f"got {const_opt_during!r}"
            )
        if const_opt_during != "none":
            if const_opt != "lm":
                raise ValueError("const_opt_during 'elite' needs const_opt 'lm'")
            if contin != "copy":
                raise ValueError(
                    "const_opt_during 'elite' needs contin 'copy' (constants as genes)"
                )
            if const_opt_top_k > 0:
                raise ValueError(
                    "const_opt_during 'elite' and const_opt_top_k > 0 exclude each other"
                )
        if const_opt_pole_check not in POLE_CHECK_MODES:
            raise ValueError(
                f"const_opt_pole_check must be one of {POLE_CHECK_MODES}, "
                f"got {const_opt_pole_check!r}"
            )
        if const_opt_pole_check != "none":
            if const_opt != "lm":
                raise ValueError(
                    f"const_opt_pole_check {const_opt_pole_check!r} needs const_opt 'lm'"
                )
            if const_opt_top_k > 0:
                raise ValueError("const_opt_pole_check and const_opt_top_k > 0 exclude each other")
            if const_opt_during != "none":
                raise ValueError("const_opt_pole_check and const_opt_during exclude each other")
        if not isinstance(const_opt_pole_seed, int) or isinstance(const_opt_pole_seed, bool):
            raise ValueError(f"const_opt_pole_seed must be an integer, got {const_opt_pole_seed!r}")
        # NOT A PORT, T3.2 (server round 16): the RV core (``--gomea-rv-step``).
        if rv_step not in RV_MODES:
            raise ValueError(f"rv_step must be one of {RV_MODES}, got {rv_step!r}")
        if rv_budget not in RV_BUDGETS:
            raise ValueError(f"rv_budget must be one of {RV_BUDGETS}, got {rv_budget!r}")
        if rv_step != "off" and contin != "copy":
            raise ValueError("rv_step needs contin 'copy' (constants as genes)")
        if rv_step != "off" and const_opt != "none":
            raise ValueError("rv_step needs const_opt 'none'")
        if not isinstance(rv_seed, int) or isinstance(rv_seed, bool):
            raise ValueError(f"rv_seed must be an integer, got {rv_seed!r}")
        # NOT A PORT, server round 16 (the RV16 review): the best-so-far trace.
        if not isinstance(trace, bool):
            raise ValueError(f"trace must be True or False, got {trace!r}")
        if trace and const_opt == "lm":
            raise ValueError("trace does not support const_opt 'lm' (LM passes have no score)")
        # NOT A PORT, backlog 15.4 (D1): the genes of each deme's best program in the log.
        if not isinstance(log_best_genes, bool):
            raise ValueError(f"log_best_genes must be True or False, got {log_best_genes!r}")
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
        self.const_opt_guard = const_opt_guard
        self.hc_fraction_of_nn = hc_fraction_of_nn
        self.hc_max_nn_evals = hc_max_nn_evals
        self.const_opt_reserve = const_opt_reserve
        self.const_opt_top_k = const_opt_top_k
        self.const_opt_during = const_opt_during
        self.const_opt_pole_check = const_opt_pole_check
        self.const_opt_pole_seed = const_opt_pole_seed
        self.rv_step = rv_step
        self.rv_budget = rv_budget
        self.rv_seed = rv_seed
        self.trace = trace
        self.log_best_genes = log_best_genes

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
        "const_opt_guard": "none",
        "const_opt_reserve": 0,
        "const_opt_top_k": 0,
        "const_opt_during": "none",
        "const_opt_pole_check": "none",
        "rv_step": "off",
        "trace": False,
    }

    # The knob-cap options: all left out of ``as_dict`` while ``knob_cap`` is 0
    # (off), all kept otherwise.
    _KNOB_KEYS = ("knob_cap", "knob_select", "knob_explore", "knob_rank0")

    # The dynamic-population options: left out of ``as_dict`` unless ``pop_size`` is
    # ``"dynamic"`` (server round 6).
    _DYNAMIC_KEYS = ("hc_fraction_of_nn", "hc_max_nn_evals")

    # The seed of the pole check: left out of ``as_dict`` while ``const_opt_pole_check`` is
    # ``"none"`` (server round 13).
    _POLE_KEYS = ("const_opt_pole_seed",)

    # The RV core options: left out of ``as_dict`` while ``rv_step`` is ``"off"`` (server
    # round 16).
    _RV_KEYS = ("rv_budget", "rv_seed")

    # NOT A PORT, backlog 15.4 (D1): options that only add a field to the log. They are always
    # left out of ``as_dict``, so that the log of a run with the option differs from the log
    # of the run without it only by the added field (``best_genes`` of each "end" event).
    _LOG_ONLY_KEYS = ("log_best_genes",)

    def as_dict(self) -> dict:
        d = dict(vars(self))
        for k in self._LOG_ONLY_KEYS:
            del d[k]
        for k, default in self._OPTIONAL_DEFAULTS.items():
            if d[k] == default:
                del d[k]
        if self.knob_cap == 0:
            for k in self._KNOB_KEYS:
                del d[k]
        if not self.dynamic():
            for k in self._DYNAMIC_KEYS:
                del d[k]
        if self.const_opt_pole_check == "none":
            for k in self._POLE_KEYS:
                del d[k]
        if self.rv_step == "off":
            for k in self._RV_KEYS:
                del d[k]
        return d

    def dynamic(self) -> bool:
        """Whether ``pop_size`` is ``"dynamic"`` (server round 6)."""
        return self.pop_size == DYNAMIC_POP_SIZE

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

    def lm_reserve(self, budget: int) -> int:
        """NOT A PORT, server round 11: the evaluations of a deme with budget ``budget``
        that the search leaves for LM, min(``const_opt_reserve``, budget // 2); 0 when the
        reserve is off."""
        return min(self.const_opt_reserve, max(0, budget) // 2)

    def deme_pop_size(self, max_evals: int, fields=None) -> int:
        """The population size N of a deme handed ``max_evals``: ``pop_size``
        when ``pop_fraction`` is off, else ``max(2, floor(pop_fraction *
        deme_budget(max_evals)))`` clamped to ``pop_max`` when set. The
        product is taken exactly (on the decimal value of ``pop_fraction``),
        so 0.29 x 100 is 29, not 28.

        With ``pop_size`` ``"dynamic"`` (server round 6), N is
        ``dynamic_pop_size(fields, deme_budget(max_evals), ...)``; ``fields`` (the
        deme's ``field_set``) is then required."""
        if self.dynamic():
            if fields is None:
                raise ValueError("pop_size 'dynamic' needs the deme's field set")
            return dynamic_pop_size(
                fields, self.deme_budget(max_evals), self.hc_fraction_of_nn, self.hc_max_nn_evals
            )
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


# NOT A PORT, server round 12 (``--const-opt-during elite``): the counts of LM during the
# search, added to ``const_opt`` (the "end" event) only when the option is on.
ELITE_COUNT_KEYS = (
    "lm_elite_calls",
    "lm_elite_evals",
    "lm_elite_kept",
    "lm_elite_worse",
    "lm_elite_guard_rejected",
    "lm_elite_not_knob",
)


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


def init_digest(genotypes: Sequence[tuple[int, ...]]) -> str:
    """NOT A PORT, server round 16 (``--gomea-trace``, the RV16 review): the SHA-256 (hex) of
    the initial genotypes in their order, as the compact JSON list of lists of their codes
    (``json.dumps([list(g) for g in genotypes], separators=(",", ":"))``). Two runs with the
    same initial population have the same digest."""
    text = json.dumps([[int(c) for c in g] for g in genotypes], separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


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
        self._val_rows: RegressionRows | None = None
        # GOMEA extension, server round 6: the N of the current deme.
        self._deme_n: int | None = None
        # NOT A PORT, server round 11 (``const_opt_reserve``): the current deme's LM
        # reserve and the run budget the oracle gets back before LM.
        self._reserve = 0
        self._call_max_evals = 0
        # NOT A PORT, server round 13 (``const_opt_pole_check``): the probe points of the run,
        # drawn at the first LM call (``None`` before, and always when the check is off).
        self._pole_probes: list[list[float]] | None = None
        # NOT A PORT, T3.2 (server round 16, ``rv_step``): the current call's RV stream and
        # RV core (``None`` when off), and the number of GOMEA demes that got an RV stream
        # (the deme index of the seed rule, ``_rv_stream``).
        self._rng_rv: MT19937RandGen | None = None
        self._rv: RVCore | None = None
        self._rv_demes = 0

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
        self._rng_rv = self._rv_stream()  # NOT A PORT, T3.2 (None = off)
        self._co = _new_const_opt_counts()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive, gp.accept_complexity_ratio)
        n = gp.deme_pop_size(max_evals, schema.fields)
        self._deme_n = n
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
        self._rng_rv = self._rv_stream()  # NOT A PORT, T3.2 (None = off)
        self._co = _new_const_opt_counts()
        sink = TelemetrySink(gp.telemetry_path)
        oracle = self._make_oracle(schema, iscorer, deme, max_evals, max_time)
        prior = [(schema.to_genotype(si.first), si.second) for si in deme]
        oracle.preload(prior, deme.n_evals)
        ev = _OracleEvaluator(oracle, schema, gp.skip_inactive, gp.accept_complexity_ratio)
        n = gp.deme_pop_size(max_evals, schema.fields)
        self._deme_n = n
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
        cap = self.deme_cap()
        search_evals = max_evals
        self._reserve = 0
        self._call_max_evals = max_evals
        if gp.const_opt_reserve > 0:
            # NOT A PORT, server round 11: the search stops at the deme budget minus the
            # reserve. The limit that sets the deme budget is lowered, so the termination
            # reason stays "global_budget" or "inner_budget"; ``_finish`` gives the oracle
            # back the full ``max_evals`` before LM (``charge_tuning`` ignores the cap).
            budget = max_evals if cap is None else min(max_evals, cap)
            self._reserve = gp.lm_reserve(budget)
            if cap is not None and cap < max_evals:
                cap -= self._reserve
            else:
                search_evals = max_evals - self._reserve
        return EvaluationOracle(
            schema,
            iscorer,
            deme,
            search_evals,
            max_time,
            inner_cap=cap,
            genotype_memo=gp.genotype_memo,
            stall_evals=gp.stall_evals,
            stall_worse=self._accept_worse(),
            store_accepted=gp.deme_store == "accepted",
            trace=gp.trace,
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
            # The field set is read only by --gomea-pop-size dynamic (server round 11:
            # dynamic N with --gomea-deme-share, the arms BASE-D5, -D10 and -D20).
            extra["deme_pop_size"] = gp.deme_pop_size(max_evals, schema.fields)
        if gp.dynamic():
            # GOMEA extension, server round 6: this deme's budget and dynamic N
            # (absent with a fixed N).
            extra["deme_budget"] = gp.deme_budget(max_evals)
            extra["deme_pop_size"] = gp.deme_pop_size(max_evals, schema.fields)
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
        if self.gomea_params.trace and sink.path is not None:
            # NOT A PORT, server round 16 (``--gomea-trace``; absent when off).
            extra["genotypes_sha256"] = init_digest(init.genotypes)
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
                    distance=distance_fn(schema) if gp.fos == "np" else None,
                    tune_elite=self._elite_tuner(schema, oracle, ev),
                    rv_generation=self._rv_hook(schema, oracle, ev),
                )
                reason = engine.run(population).termination_reason
        except KeyboardInterrupt:
            self._finish(
                deme, oracle, ev, schema, init, engine, "interrupted", init_s, t_start, sink
            )
            raise
        self._finish(deme, oracle, ev, schema, init, engine, reason, init_s, t_start, sink)

    def _rv_stream(self) -> MT19937RandGen | None:
        """NOT A PORT, T3.2 (server round 16): the RV stream of this call (deme), or ``None``
        (and no count) when ``rv_step`` is ``"off"``. It also clears the RV core of the
        previous call.

        The seed rule: the k-th call (k = 0, 1, ...) of the optimizer gets
        ``rv_core.rv_stream(rv_seed, k)``, with ``rv_seed`` the run seed (``-r``). Only
        ``-a gomea`` can use the RV option (the CLI and ``hc_gomea_optimization`` refuse it
        otherwise, the RV16 review), each ``__call__`` is one deme, and one optimizer serves
        one run, so k is the deme index of the run. The rule does not draw from, or change,
        the global ``randGen()``."""
        self._rv = None
        gp = self.gomea_params
        if gp.rv_step == "off":
            return None
        k = self._rv_demes
        self._rv_demes += 1
        return rv_stream(gp.rv_seed, k)

    def _rv_hook(self, schema: FrozenSchema, oracle: EvaluationOracle, ev: _OracleEvaluator):
        """NOT A PORT, T3.2 (server round 16): the engine's ``rv_generation`` hook (a new
        ``RVCore`` on this call's RV stream), or ``None`` (nothing built) when ``rv_step`` is
        ``"off"``."""
        gp = self.gomea_params
        self._rv = None
        if gp.rv_step == "off":
            return None
        keep = oracle.keep if gp.deme_store == "accepted" else None
        self._rv = RVCore(schema, oracle, ev, self._rng_rv, keep, gp.rv_budget)
        return self._rv

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
            if self._reserve:
                oracle.max_evals = self._call_max_evals  # server round 11: LM gets the rest
            if self.gomea_params.const_opt_top_k > 0:
                self._tune_top_k(deme, oracle, schema)  # server round 12
            elif self.gomea_params.const_opt_during != "none":
                self._tune_deme_best(deme, oracle, schema)  # server round 12
            else:
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
        if self.gomea_params.dynamic():
            # GOMEA extension, server round 6: the dynamic N of this deme (absent
            # with a fixed N; ``pop_size_effective`` is the size actually drawn).
            self.last_result["pop_size_dynamic"] = self._deme_n
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
            if self.gomea_params.const_opt_guard != "none":
                # server round 5 (absent without the guard).
                self.last_result["const_opt"].update(
                    guard=self.gomea_params.const_opt_guard,
                    lm_guard_checked=self._co.get("lm_guard_checked", 0),
                    lm_guard_rejected=self._co.get("lm_guard_rejected", 0),
                )
            if self.gomea_params.const_opt_pole_check != "none":
                # NOT A PORT, server round 13 (absent without the pole check).
                self.last_result["const_opt"].update(
                    lm_pole_checked=self._co.get("lm_pole_checked", 0),
                    lm_pole_rejected=self._co.get("lm_pole_rejected", 0),
                )
        if self.gomea_params.rv_step != "off":
            # NOT A PORT, T3.2 (server round 16): the RV totals of this deme (absent when off;
            # None when the deme ended before the engine ran).
            self.last_result["rv"] = self._rv.record() if self._rv is not None else None
        if self.gomea_params.trace:
            # NOT A PORT, server round 16 (``--gomea-trace``; absent when off): the best-so-far
            # trace of the deme's raw score, [deme evaluation, raw score] pairs.
            self.last_result["trace"] = list(oracle.trace)
        if self.gomea_params.const_opt_reserve > 0:
            # NOT A PORT, server round 11: this deme's LM reserve (absent when off).
            self.last_result["const_opt_reserve"] = self._reserve
        if self.gomea_params.const_opt_top_k > 0:
            # NOT A PORT, server round 12: the records of the top-K LM calls (absent when off).
            self.last_result["const_opt"].setdefault("lm_topk", [])
        if self.gomea_params.const_opt_during != "none":
            # NOT A PORT, server round 12: the counts and records of LM during the search
            # (absent when off).
            for k in ELITE_COUNT_KEYS:
                self.last_result["const_opt"].setdefault(k, 0)
            self.last_result["const_opt"].setdefault("lm_elite", [])
        extra = {}
        if self.gomea_params.log_best_genes and sink.path is not None:
            # NOT A PORT, backlog 15.4 (D1; absent when off): the genes of the deme's best
            # program, the one that ``best`` describes. Only in the event: ``last_result`` and
            # the search do not change.
            extra["best_genes"] = self._best_genes(oracle, schema, requests)
        sink.emit("end", **self.last_result, **extra)

    def _best_genes(self, oracle: EvaluationOracle, schema: FrozenSchema, requests: int):
        """NOT A PORT, backlog 15.4 (D1): the field ``best_genes`` of the "end" event
        (``best_genes.best_genes_record`` of ``oracle.best_genotype()``), ``None`` when the
        deme has no evaluation (``best`` is then ``None`` too). The record only reads; an
        error gives ``{"error": text}`` and the run goes on."""
        g = oracle.best_genotype() if requests > 0 else None
        if g is None:
            return None
        try:
            return best_genes_record(
                schema,
                g,
                scorer_fit(oracle.iscorer),
                getattr(oracle.iscorer, "_reduce", True),
            )
        except Exception as exc:  # a log field must never stop the run
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}

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

    # NOT A PORT, server round 12 (``--const-opt-during elite``, T2.3).
    def _elite_tuner(self, schema: FrozenSchema, oracle: EvaluationOracle, ev: _OracleEvaluator):
        """The engine's ``tune_elite`` hook, or ``None`` (nothing built, nothing drawn) unless
        ``const_opt_during`` is ``"elite"``.

        The hook gets each GOM offspring (after ``mutate``). The offspring is a new deme best
        when the deme's best under the MOSES score (``oracle.best_genotype``, the program LM
        tunes at the end of the deme) strictly improved since the previous call of the hook
        (or since the initial population, for the first call) and the offspring is that best
        program. Only then ``_tune_elite`` runs LM on it; else the offspring is returned as it
        is. The hook draws no random numbers."""
        if self.gomea_params.const_opt_during != "elite":
            return None
        co = self._co
        for k in ELITE_COUNT_KEYS:
            co.setdefault(k, 0)
        co.setdefault("lm_elite", [])
        keep = oracle.keep if self.gomea_params.deme_store == "accepted" else None
        cscorer = oracle.iscorer._cscorer
        rows = RegressionRows.of_bscore(find_contin_bscore(cscorer))
        val = None
        if self.gomea_params.const_opt_guard == "val":
            val = self._guard_rows(cscorer, rows)
        state = {"at": oracle.evals_at_best()}

        def tune(ind: Individual) -> Individual:
            at = oracle.evals_at_best()
            new_best = at != state["at"] and oracle.best_genotype() == ind.genotype
            state["at"] = at
            if not new_best:
                return ind
            try:
                return self._tune_elite(ind, schema, oracle, ev, rows, val, keep)
            finally:
                state["at"] = oracle.evals_at_best()

        return tune

    def _tune_elite(
        self,
        ind: Individual,
        schema: FrozenSchema,
        oracle: EvaluationOracle,
        ev: _OracleEvaluator,
        rows: RegressionRows,
        val: RegressionRows | None,
        keep,
    ) -> Individual:
        """LM with the guard on the offspring ``ind`` (a new deme best) during the search.

        1. LM tunes the constants of the UNREDUCED program of ``ind``
           (``FrozenSchema.candidate_genes``) whose value is not 0: each one is the value of
           one contin gene. A constant with value 0 (an inactive coefficient that knob
           building inserts, for example the 0 of an absent linear term) is not an LM
           parameter: it keeps the value 0 and its gene keeps its code (user decision,
           2026-11-16). If a constant is not the value of a contin gene (or a gene gives two
           constants), the offspring is returned as it is (``lm_elite_not_knob``).
        2. With ``const_opt_guard`` ``"val"``, ``guard_accepts`` compares the old and the
           tuned unreduced program on the validation rows; a worse validation NMSE keeps the
           offspring.
        3. Each tuned value that differs from the gene's value becomes a free contin code of
           that gene (``FrozenSchema.register_contin_value``, as ``--const-opt mutate``
           does). The new genotype is scored through the oracle (one evaluation). It
           replaces the offspring only if it is not worse under the deme's comparator
           (``ev.worse``), and it is then marked with ``keep`` (``--gomea-deme-store
           accepted``).

        Every LM and guard pass is charged by ``oracle.charge_search_tuning`` (the run's
        budget, the deme cap, the time limit, the stall rule). When one of them stops a pass,
        or the scoring, ``GomInterrupted`` is raised with the offspring ``ind`` (the last
        kept state), as ``_mutator`` does. The call counts in ``lm_calls``, ``lm_<status>``,
        ``tuning_evals`` (its LM and guard passes and the scoring) and the guard counts, and
        in the ``ELITE_COUNT_KEYS``; its record goes to ``const_opt.lm_elite``."""
        co = self._co
        g, score0 = ind.genotype, ind.score
        tree, genes = schema.candidate_genes(g)
        if None in genes or len(set(genes)) != len(genes):
            co["lm_elite_not_knob"] += 1
            return ind
        stop: list[str] = []

        def charge() -> None:
            try:
                oracle.charge_search_tuning()
            except BudgetExhausted as exc:
                stop.append(exc.reason)
                raise

        tune = [float(n.data) != 0.0 for n in constant_nodes(tree)]
        before = oracle.ledger.requests
        lm = levenberg_marquardt(tree, rows, charge, tune=tune)
        rec = lm.record()
        rec["n_constants_unreduced"] = len(genes)
        reduced = schema.candidate(g, getattr(oracle.iscorer, "_reduce", True))
        rec["n_constants_reduced"] = len(constant_nodes(reduced))
        rec["written_back"] = False
        try:
            if stop:
                rec["outcome"] = "budget"
                raise GomInterrupted(ind, stop[0])
            if lm.tree is None:
                rec["outcome"] = "not_improved"
                return ind
            if val is not None:
                try:
                    verdict = guard_accepts(tree, lm.tree, rows, val, charge)
                except BudgetExhausted as exc:
                    rec["budget"] = True
                    rec["outcome"] = "budget"
                    raise GomInterrupted(ind, exc.reason) from exc
                rec.update(
                    guard="accepted" if verdict["accepted"] else "rejected",
                    val_nmse_old=verdict["val_nmse_old"],
                    val_nmse_new=verdict["val_nmse_new"],
                )
                co["lm_guard_checked"] = co.get("lm_guard_checked", 0) + 1
                if not verdict["accepted"]:
                    co["lm_guard_rejected"] = co.get("lm_guard_rejected", 0) + 1
                    co["lm_elite_guard_rejected"] += 1
                    rec["outcome"] = "guard_rejected"
                    return ind
            new = list(g)
            tuned = [p for p, t in zip(genes, tune, strict=True) if t]
            for p, v in zip(tuned, lm.values, strict=True):
                if v != schema.contin_value(p, g[p]):
                    new[p] = schema.register_contin_value(p, v)
            trial = tuple(new)
            if trial == g:
                rec["outcome"] = "unchanged"
                return ind
            try:
                score = ev.evaluate(trial)
            except BudgetExhausted as exc:
                rec["budget"] = True
                rec["outcome"] = "budget"
                raise GomInterrupted(ind, exc.reason) from exc
            if ev.worse(score, score0):
                co["lm_elite_worse"] += 1
                rec["outcome"] = "worse"
                return ind
            co["lm_elite_kept"] += 1
            rec["outcome"] = "kept"
            rec["written_back"] = True
            if keep is not None:
                keep(trial)
            return Individual(trial, score, ev.phenotype_key(trial))
        finally:
            used = oracle.ledger.requests - before
            rec["evals"] = used
            co["lm_calls"] += 1
            co["lm_" + lm.status] += 1
            co["tuning_evals"] += used
            co["lm_elite_calls"] += 1
            co["lm_elite_evals"] += used
            co["lm_elite"].append(rec)

    # NOT A PORT (``--const-opt-guard val``, server round 5).
    def _guard_rows(self, cscorer, rows: RegressionRows) -> RegressionRows:
        """The validation rows, read once per optimizer, with the input types of the
        training table (backlog 12.1, ``load_rows``)."""
        if self._val_rows is None:
            cti = find_contin_bscore(cscorer)._cti
            self._val_rows = load_rows(
                self.gomea_params.const_opt_val, cti.get_labels(), rows.scaled, cti.get_types()
            )
        return self._val_rows

    # GOMEA extension (``--const-opt lm``): elite-only LM at the end of a deme.
    def _tune_best(self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema) -> None:
        """LM on the deme's best program (``const_opt.levenberg_marquardt``),
        after ``oracle.finalize``. Every pass over the training rows is charged
        through ``oracle.charge_tuning`` (the run's budget; not the deme cap)
        and added to ``deme.n_evals``. If LM lowered the training MSE, the
        tuned program, scored (one more pass), replaces the best program's
        entries in the deme (Lamarckian). A singular or failed call keeps the
        old program; the call's record is logged in ``const_opt.lm``.

        NOT A PORT (``--const-opt-guard val``, server round 5): between LM and
        that score, ``guard_accepts`` compares the old and the tuned program on
        the validation rows (two charged passes). A worse validation NMSE keeps
        the old program.

        NOT A PORT (``--const-opt-pole-check box`` or ``box-rel``, server rounds 13 and 14):
        before the guard, ``_pole_accepts`` checks the tuned program on the probe points (two
        charged passes). A rejection keeps the old program, as a guard rejection does."""
        co = self._co
        g = oracle.best_genotype()
        if g is None:
            return
        iscorer = oracle.iscorer
        cscorer = iscorer._cscorer
        rows = RegressionRows.of_bscore(find_contin_bscore(cscorer))
        if self.gomea_params.const_opt_pole_check != "none" and self._pole_probes is None:
            # server round 13: the probes of the run, drawn once (their own generator)
            self._pole_probes = pole_probes(rows, self.gomea_params.const_opt_pole_seed)
        tree = schema.candidate(g, getattr(iscorer, "_reduce", True))
        before = oracle.ledger.requests
        lm = levenberg_marquardt(tree, rows, oracle.charge_tuning)
        rec = lm.record()
        if lm.tree is not None:
            try:
                if self.gomea_params.const_opt_pole_check != "none":
                    # server round 13: the pole check, before the guard
                    if not self._pole_accepts(tree, lm.tree, rows, oracle.charge_tuning, rec):
                        rec["written_back"] = False
                        return self._close_tuning(deme, oracle, lm, rec, before)
                if self.gomea_params.const_opt_guard == "val":
                    verdict = guard_accepts(
                        tree, lm.tree, rows, self._guard_rows(cscorer, rows), oracle.charge_tuning
                    )
                    rec.update(
                        guard="accepted" if verdict["accepted"] else "rejected",
                        val_nmse_old=verdict["val_nmse_old"],
                        val_nmse_new=verdict["val_nmse_new"],
                    )
                    co["lm_guard_checked"] = co.get("lm_guard_checked", 0) + 1
                    if not verdict["accepted"]:
                        co["lm_guard_rejected"] = co.get("lm_guard_rejected", 0) + 1
                        rec["written_back"] = False
                        return self._close_tuning(deme, oracle, lm, rec, before)
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
        self._close_tuning(deme, oracle, lm, rec, before)

    # NOT A PORT, server round 13 (``--const-opt-pole-check box``, BASE-PS); server round 14
    # (``box-rel``, NP-LMR-PSR).
    def _pole_accepts(self, old, new, rows: RegressionRows, charge, rec: dict) -> bool:
        """``const_opt.pole_check`` of the tuned program ``new`` against ``old`` on the run's
        probe points (two passes, each charged by ``charge``; ``BudgetExhausted`` goes to the
        caller), with the rule of ``const_opt_pole_check`` (``box`` or ``box-rel``). Adds
        ``pole`` (``accepted`` or ``rejected``), ``pole_s_old`` and ``pole_s_new`` to the
        call's record ``rec`` and counts ``lm_pole_checked`` and ``lm_pole_rejected``. Returns
        whether the tuned program passes."""
        verdict = pole_check(
            old, new, rows, self._pole_probes, charge, mode=self.gomea_params.const_opt_pole_check
        )
        rec.update(
            pole="accepted" if verdict["accepted"] else "rejected",
            pole_s_old=verdict["pole_s_old"],
            pole_s_new=verdict["pole_s_new"],
        )
        co = self._co
        co["lm_pole_checked"] = co.get("lm_pole_checked", 0) + 1
        if not verdict["accepted"]:
            co["lm_pole_rejected"] = co.get("lm_pole_rejected", 0) + 1
        return verdict["accepted"]

    # NOT A PORT, server round 12 (``--const-opt-top-k K``, T1.6).
    def _tune_top_k(self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema) -> None:
        """LM on up to K programs of the deme (``_top_k_genotypes``), after
        ``oracle.finalize``, in place of ``_tune_best``. Each program gets exactly what
        ``_tune_best`` does for the best one (``_tune_program``): the same LM, guard, charges
        (``oracle.charge_tuning``, the run's budget) and Lamarckian write-back. The first
        program is the one ``_tune_best`` tunes, so K = 1 is the same as K = 0. After a call
        that ran out of the budget or the time limit, no more calls are made.

        Telemetry (only with K > 0): ``const_opt.lm_topk`` (the record of each call, in
        order), ``lm_topk_picked`` (the programs picked, at most K) and
        ``lm_topk_best_in_deme`` (whether the deme's best program still has an entry in the
        deme; see ``_top_k_genotypes``). ``const_opt.lm`` is the record of the first call,
        the one on the deme's best program; ``lm_calls`` and the other counts include every
        call."""
        co = self._co
        picks, in_deme = self._top_k_genotypes(deme, oracle, schema)
        records = []
        for g in picks:
            self._tune_program(deme, oracle, schema, g)
            records.append(co["lm"])
            if co["lm"]["budget"]:
                break
        if records:
            co["lm"] = records[0]
        co["lm_topk"] = records
        co["lm_topk_picked"] = len(picks)
        co["lm_topk_best_in_deme"] = in_deme

    def _top_k_genotypes(
        self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema
    ) -> tuple[list[tuple[int, ...]], bool]:
        """``(picks, best_in_deme)``: the genotypes of the K programs that ``_tune_top_k``
        tunes, and whether the deme's best program has an entry in the deme.

        The first pick is ``oracle.best_genotype()``, the program ``_tune_best`` tunes. Then
        the deme entries, sorted by the MOSES score (``composite_score``'s ``<``; ties keep
        the entry order), are walked from the best down, and each entry whose structure
        (``structure_key`` of its reduced tree, the tree ``_tune_best`` gives LM) is not
        taken yet is picked, until there are K picks.

        With ``--gomea-accept-complexity-ratio`` R > 0 and ``--gomea-deme-store accepted``,
        the best program under the MOSES score can be missing from the kept deme (GOM can
        reject it under the penalized comparison). It is still the first pick, as for
        ``_tune_best``; when it is in the deme, it is also the first entry of the sorted
        deme, so the rule is the same in both cases. No pick when nothing was evaluated."""
        best = oracle.best_genotype()
        if best is None:
            return [], False
        k = self.gomea_params.const_opt_top_k
        reduce = getattr(oracle.iscorer, "_reduce", True)
        order = sorted(
            range(len(deme)), key=lambda i: _WorseKey(deme[i].second, operator.lt), reverse=True
        )
        best_inst = schema.to_instance(best)
        in_deme = any(si.first == best_inst for si in deme)
        picks = [best]
        keys = {structure_key(schema.candidate(best, reduce))}
        seen = {best}
        for i in order:
            if len(picks) >= k:
                break
            g = schema.to_genotype(deme[i].first)
            if g in seen:
                continue
            seen.add(g)
            key = structure_key(schema.candidate(g, reduce))
            if key in keys:
                continue
            keys.add(key)
            picks.append(g)
        return picks, in_deme

    # NOT A PORT, server round 12 (``--const-opt-during elite``, T2.3).
    def _tune_deme_best(
        self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema
    ) -> None:
        """LM at the end of a deme with ``const_opt_during`` on: ``_tune_best`` with the
        write-back of ``_tune_program``, which also handles a best program with free contin
        codes (``_tune_best`` stays byte-identical for every other arm)."""
        g = oracle.best_genotype()
        if g is None:
            return
        self._tune_program(deme, oracle, schema, g)

    def _tune_program(
        self, deme: deme_t, oracle: EvaluationOracle, schema: FrozenSchema, g: tuple[int, ...]
    ) -> None:
        """``_tune_best`` on the program of genotype ``g`` in place of the deme's best one:
        the same LM, guard, charges, Lamarckian write-back and record. A copy of its body,
        because HC-LM borrows ``_tune_best`` as it is (``hc_lm.py``) and ``_tune_best`` stays
        byte-identical.

        Server round 12 (T2.3): if ``g`` has free contin codes (``--const-opt-during
        elite``), its deme entries carry their program (``scored_tree_instance``, with the
        placeholder instance ``to_instance(g)``). The write-back then removes the entries
        with that instance whose program is the program of ``g`` (the tree LM started from),
        and keeps the plain entries of the placeholder instance, which are another program.
        For ``g`` without free codes (every top-K pick), the write-back is that of
        ``_tune_best``."""
        co = self._co
        iscorer = oracle.iscorer
        cscorer = iscorer._cscorer
        rows = RegressionRows.of_bscore(find_contin_bscore(cscorer))
        tree = schema.candidate(g, getattr(iscorer, "_reduce", True))
        before = oracle.ledger.requests
        lm = levenberg_marquardt(tree, rows, oracle.charge_tuning)
        rec = lm.record()
        if lm.tree is not None:
            try:
                if self.gomea_params.const_opt_guard == "val":
                    verdict = guard_accepts(
                        tree, lm.tree, rows, self._guard_rows(cscorer, rows), oracle.charge_tuning
                    )
                    rec.update(
                        guard="accepted" if verdict["accepted"] else "rejected",
                        val_nmse_old=verdict["val_nmse_old"],
                        val_nmse_new=verdict["val_nmse_new"],
                    )
                    co["lm_guard_checked"] = co.get("lm_guard_checked", 0) + 1
                    if not verdict["accepted"]:
                        co["lm_guard_rejected"] = co.get("lm_guard_rejected", 0) + 1
                        rec["written_back"] = False
                        return self._close_tuning(deme, oracle, lm, rec, before)
                oracle.charge_tuning()
                score = cscorer.get_cscore(lm.tree)
            except BudgetExhausted:
                rec["budget"] = True
                rec["written_back"] = False
            else:
                inst = schema.to_instance(g)
                if schema.free_values(g):
                    key = _cache_key(tree)
                    deme[:] = [
                        si
                        for si in deme
                        if not (
                            si.first == inst
                            and getattr(si, "tree", None) is not None
                            and _cache_key(si.tree) == key
                        )
                    ]
                else:
                    deme[:] = [
                        si
                        for si in deme
                        if not (si.first == inst and getattr(si, "tree", None) is None)
                    ]
                deme.append(scored_tree_instance(inst, score, lm.tree))
                rec["written_back"] = True
        self._close_tuning(deme, oracle, lm, rec, before)

    def _close_tuning(self, deme, oracle, lm, rec: dict, before: int) -> None:
        """The counts and the record of one LM call."""
        co = self._co
        used = oracle.ledger.requests - before
        deme.n_evals += used
        rec["evals"] = used
        co["tuning_evals"] += used
        co["lm_calls"] += 1
        co["lm_" + lm.status] += 1
        co["lm"] = rec
