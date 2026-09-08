"""C++: moses/moses/optimization/optimization.{h,cc}

`information_theoretic_bits`, `optim_parameters`, `optim_stats`, and
`optimizer_base` -- the algorithm-agnostic scaffolding every optimizer
(only `hill_climbing`, in this port) derives from.

NOT ported:

* `print_stats_header` (optimization.h:168, optimization.cc:140-186) --
  pure logging: builds and emits a legend string through `logger()`, with
  no other observable effect. This port has no logging surface (see
  `instance_scorer.py`'s `distance_based_scorer.__call__` docstring, and
  `scoring_base.py`'s module docstring, for the same omission elsewhere).
* `optim_stats`'s `#ifdef GATHER_STATS` fields (`scores`, `counts`,
  `hiscore`, `hicount`, `num_improved`, `count_improved`) -- dead code by
  construction: the header's own `#define GATHER_STATS 1` that would
  enable them is commented out ("Turned off by default because it's a CPU-
  waster", optimization.h:135-137), matching CLAUDE.md's treatment of
  other never-defined `#ifdef` branches (`EXEMPLAR_INST_IS_UNDEAD`,
  `LATER`, `HAVE_MPI`). `hill-climbing.cc`'s own matching `#ifdef
  GATHER_STATS` block (lines 291-322) is dropped for the same reason.
"""

from __future__ import annotations

import math

from pymetta.moses.moses.types import _to_score_t as _narrow
from pymetta.moses.moses.types import demeID_t, score_t
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance_set import instance_set as deme_t
from pymetta.moses.representation.scored_instance import scored_instance as deme_inst_t

# C++: moses/moses/optimization/optimization.h:170-171 deme_t / deme_inst_t
#
# `typedef instance_set<composite_score> deme_t;` and
# `typedef scored_instance<composite_score> deme_inst_t;` -- both classes
# already fix `ScoreT = composite_score` directly (see instance_set.py's and
# scored_instance.py's own DIVERGENCE notes), so these are plain aliases
# re-exported under the C++ names hill_climbing.py's own breadcrumbs use,
# not new types. A plain comment, not a docstring: `check_alignment.py`
# cannot attach a breadcrumb to a bare assignment any more than
# `behavioral_score = list[score_t]` (types.py) can carry one.

__all__ = [
    "deme_inst_t",
    "deme_t",
    "information_theoretic_bits",
    "optim_parameters",
    "optim_stats",
    "optimizer_base",
]


def information_theoretic_bits(fs: field_set) -> float:
    """C++: moses/moses/optimization/optimization.cc:37 information_theoretic_bits

    "The information content of a single variable is defined as log base
    two of the number of possible values that the variable can take ...
    This routine sums the total information content in a field set" --
    optimization.h:40-62's own doc comment, reproduced in spirit. Plain
    `double` arithmetic throughout (the C++ return type is `double`, not
    `score_t`), so nothing here is narrowed via `_to_score_t`.

    NOT ported: the term-field contribution (`fs.term()`'s loop,
    optimization.cc:54-55) -- term fields do not exist anywhere in this
    port (field_set.py excludes them entirely, the same scope decision
    `neighborhood_sampling.py`'s module docstring already documents for
    every other term-touching function here), so that loop's iteration
    range is always empty and it adds nothing.

    Load-bearing detail, found by hand-tracing rather than by the probe
    corpus (every hillclimbing.json fixture case happens to have zero bit
    fields, so a regression here would NOT have shown up in the oracle
    trajectory tests): the C++ loop bounds its `disc_and_bit()` iterator to
    exactly `n_disc_fields()` steps (optimization.cc:44-49), not the whole
    vector -- `disc_and_bit()` lists every multi-valued disc spec FIRST
    (higher multiplicity sorts first, disc_spec.__lt__/field_set.py's
    `_compare_specs`) and every single-bit spec LAST, so `n_disc_fields()`
    -- itself a RAW-FIELD count, and each disc spec occupies exactly one
    raw field -- is precisely the count of genuine multi-valued specs at
    the front. Iterating the FULL sequence instead would also visit the
    trailing bit specs, adding `log2(2) == 1` for each of them a SECOND
    time on top of the `res += fs.n_bits()` line below -- double-counting
    every bit field's contribution. Slicing to `[:n_disc_fields]` is what
    keeps this a single count, per field, exactly once.

    A second, same-shaped catch from the same review pass: C++ multiplies
    the contin term by `fs.contin().size()` (optimization.cc:52) -- the
    NUMBER OF CONTIN SPECS -- not `n_contin_fields()`'s raw/depth count. A
    single `contin_spec` of depth 3 occupies THREE raw positions
    (field_set.py's `_build_contin_spec`) but is still exactly one contin
    variable, contributing `log2(5)` once, not three times.
    """
    log_five = math.log2(5.0)
    res = 0.0
    for spec in fs.disc_and_bit()[: fs.n_disc_fields()]:
        res += math.log2(spec.multy)
    res += fs.n_bits()  # log_2(2) == 1
    # DIVERGENCE-shaped catch, fixed before shipping: C++ multiplies by
    # `fs.contin().size()` -- the NUMBER OF CONTIN SPECS -- not by the
    # raw/depth count `n_contin_fields()` would give (one contin_spec with
    # depth=3 occupies 3 raw positions but is still exactly ONE contin
    # variable, contributing log2(5) once, not three times).
    res += len(fs.contin()) * log_five
    return res


class optim_parameters:
    """C++: moses/moses/optimization/optimization.h:73 optim_parameters

    "Parameters used mostly for EDA algorithms but also possibly by other
    algo" (the header's own comment).
    """

    def __init__(
        self,
        opt_algo: str = "hc",
        pop_size_ratio: float = 20,
        terminate_if_gte: score_t = 0,
        max_dist: int = 4,
        min_score_improv: score_t = 0.5,
    ) -> None:
        """C++: moses/moses/optimization/optimization.cc:60 optim_parameters

        `_terminate_if_gte`/`_min_score_improv` are `score_t`-typed C++
        parameters (optimization.h:77,79); narrowed here at the binding,
        matching `composite_score.__init__`'s own treatment of its
        `score_t` parameters (types.py).
        """
        self.opt_algo = opt_algo
        self.term_improv = 1.0
        self.window_size_pop = 0.05
        self.window_size_len = 1.0
        self.pop_size_ratio = pop_size_ratio
        self.terminate_if_gte: score_t = _narrow(terminate_if_gte)
        self.max_dist = max_dist
        self._min_score_improvement: score_t = 0.0
        self.set_min_score_improv(min_score_improv)

    def pop_size(self, fs: field_set) -> int:
        """C++: moses/moses/optimization/optimization.cc:81 pop_size

        "N = p.popsize_ratio * n^1.05" (the header's own comment).
        """
        return math.ceil(self.pop_size_ratio * math.pow(information_theoretic_bits(fs), 1.05))

    def max_gens_improv(self, fs: field_set) -> int:
        """C++: moses/moses/optimization/optimization.cc:88 max_gens_improv

        "term_improv*sqrt(n/w)" (the header's own comment).
        """
        return math.ceil(
            self.term_improv * math.sqrt(information_theoretic_bits(fs) / self.rtr_window_size(fs))
        )

    def rtr_window_size(self, fs: field_set) -> int:
        """C++: moses/moses/optimization/optimization.cc:96 rtr_window_size

        "min(windowsize_pop*N,windowsize_len*n)" (the header's own comment).
        """
        return math.ceil(
            min(
                self.window_size_pop * self.pop_size(fs),
                self.window_size_len * information_theoretic_bits(fs),
            )
        )

    def max_distance(self, fs: field_set) -> int:
        """C++: moses/moses/optimization/optimization.cc:102 max_distance"""
        return min(self.max_dist, fs.dim_size())

    def set_min_score_improv(self, s: score_t) -> None:
        """C++: moses/moses/optimization/optimization.cc:111 set_min_score_improv

        `s` is `score_t`-typed in C++; narrowed at the binding.
        """
        self._min_score_improvement = _narrow(s)

    def min_score_improv(self) -> score_t:
        """C++: moses/moses/optimization/optimization.cc:116 min_score_improv"""
        return self._min_score_improvement

    def score_improved(self, best_score: score_t, prev_hi: score_t) -> bool:
        """C++: moses/moses/optimization/optimization.cc:121 score_improved

        "The score must improve by at least 's' to be considered ... If
        's' is negative, then it is interpreted as a fraction" (types.h's
        `optim_parameters::score_improved` doc comment, reproduced at
        optimization.h:94-98). Every intermediate C++ computes here is
        `score_t` (float32) arithmetic -- `prev_hi + imp` and
        `imp * fabs(prev_hi)` are each narrowed separately, matching
        `composite_score.update_penalized_score`'s own two-separate-
        narrowings convention (types.py) rather than one combined
        double-precision expression narrowed once at the end.
        """
        imp = self.min_score_improv()
        if 0.0 <= imp:
            threshold = _narrow(prev_hi + imp)
            return best_score > threshold
        threshold = _narrow(prev_hi - _narrow(imp * abs(prev_hi)))
        return best_score > threshold


class optim_stats:
    """C++: moses/moses/optimization/optimization.h:140 optim_stats

    "Statistics obtained during optimization run, useful for tuning."
    (the header's own comment.) See the module docstring for why the
    `#ifdef GATHER_STATS` fields are not ported.
    """

    def __init__(self) -> None:
        """C++: moses/moses/optimization/optimization.h:142 optim_stats"""
        self.nsteps = 0
        self.demeID: demeID_t = demeID_t(0)
        self.total_steps = 0
        self.total_evals = 0
        self.field_set_size = 0
        self.over_budget = False


class optimizer_base(optim_stats):
    """C++: moses/moses/optimization/optimization.h:174 optimizer_base

    "Base class for all optimizers" (the header's own comment). Python has
    no pure-virtual mechanism as terse as C++'s `= 0`, so `__call__` raises
    `NotImplementedError` instead of using `abc.ABC` -- `hill_climbing`
    (the only subclass this port ever constructs) overrides it directly,
    and nothing in this port needs `isinstance`-based enforcement that a
    subclass exists before instantiation.
    """

    def __init__(self, op: optim_parameters | None = None) -> None:
        """C++: moses/moses/optimization/optimization.h:176 optimizer_base"""
        super().__init__()
        self.opt_params = optim_parameters() if op is None else op

    def __call__(self, deme: deme_t, iscorer, max_evals: int, max_time: int):
        """C++: moses/moses/optimization/optimization.h:179 optimizer_base

        Non-identifier member per the project's convention (see
        `scored_instance.py`'s `__eq__` docstring for the same pattern):
        the real member is the pure-virtual `operator()`
        (optimization.h:179-182).
        """
        raise NotImplementedError
