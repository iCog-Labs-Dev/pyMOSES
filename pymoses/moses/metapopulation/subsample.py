"""C++: moses/moses/metapopulation/subsample.cc

Ports `metapopulation::recompute_scores_over_whole_dataset`, `ss_filter`,
`ss_score_dev_filter`, `ss_average_agreement`, both `ss_tanimoto_stats`
overloads, and `ss_tanimoto_filter` -- the whole of subsample.cc (488
lines), previously ported into `merging.py` as two gates
(`recompute_scores_over_whole_dataset`/`ss_filter`) that raised
`NotImplementedError` whenever `_filter_params.n_subsample_demes > 1`
(P2 metapopulation task, prior report:
`.superpowers/sdd/2026-09-08-pymoses-s13-cli/p2-metapop-report.md`).
Those two gates are REMOVED from `merging.py` and now live here, alongside
the five private helpers subsample.cc defines that `merging.py` never
declared at all (`ss_score_dev_filter`, `ss_average_agreement`, both
`ss_tanimoto_stats` overloads, `ss_tanimoto_filter`) -- matching exactly
how `domination.py`/`diversity.py` (same prior task) each moved a whole
out-of-slice `.cc` file's surface into its own module rather than leaving
it split across `merging.py`'s gate stubs.

Same free-function + `setattr` composition as `merging.py`/`domination.py`/
`diversity.py` (see `merging.py`'s own module docstring for the three
reasons this was chosen over a mixin) -- every function below binds onto
`metapopulation` at the bottom of this file. `merging.py` imports this
module (after its own `_sort` is defined -- see that import's own inline
comment for why the ORDER matters) for the same "import the package, get
the complete method set" transitivity `test_package_binding.py` pins.

## The ten-uninitialized-fields trap is genuinely live here

`subsample_deme_filter_parameters` (`metapop_params.py`) leaves TEN fields
unset by construction -- reading one before an explicit assignment raises
`AttributeError` rather than returning C++'s indeterminate stack garbage
(see that class's own DIVERGENCE note). Of those ten, THIS file reads
EIGHT: `std_dev_threshold`, `tanimoto_mean_threshold`,
`tanimoto_geo_mean_threshold`, `tanimoto_max_threshold`, `n_best_bfdemes`,
`tanimoto_mean_weight`, `tanimoto_geo_mean_weight`, `tanimoto_max_weight`
-- every one of them ONLY inside the `n_subsample_demes > 1` arm, so the
default-off path never touches an unset field. The other two
(`n_subsample_fitnesses`, `low_dev_pressure`) are read elsewhere
(`moses_main.h:364-372`, outside this whole port's ported surface) and
`by_time`/`contiguous_time` are read by `deme_expander.cc` (also outside
this file) -- neither pair is this file's own concern. Verified live, not
merely reasoned: `pymoses/moses/main/problem_params.py`'s own
`_wire_subsample` (a concurrent/prior task, `pymoses/moses/main/` --
out of this task's scope, not edited here) already populates ALL FIFTEEN
fields unconditionally from the CLI's `ss-*` options, matching
`problem-params.cc:2339-2355`'s own inline body exactly -- so a real CLI
run reaching this file's `n_subsample_demes > 1` arm always has every
field this file reads already set. Every test below that exercises the
enabled arm builds its own `subsample_deme_filter_parameters` with all
eight fields this file reads populated explicitly (mirroring the CLI's
values when the test wants classic's own defaults), never relying on the
bare constructor for the enabled path.

## RNG fidelity: `lazy_random_selector` draw ORDER, verified against a
## real `opencog::lazy_random_selector`, not just its resulting set

`ss_score_dev_filter`/`ss_average_agreement`/`ss_tanimoto_stats` (the
deme-based overload) all share one shape: construct
`lazy_random_selector(n_possible_tuples)`, draw `selector()` exactly
`sample_tuples` times (always `<= n_possible_tuples`, `subsample.cc`'s own
`std::min`), and decompose each draw into per-deme indices via repeated
`% base` / `/= base`. `pymoses/util/lazy_random_selector.py` (reused
verbatim here -- not reimplemented, per this task's own brief) already
has Floyd's-algorithm bookkeeping covered by hand-traced fixed-candidate
tests, and `MT19937RandGen.randint_n` already has 12000-draws/6-seeds
coverage (`rng_probe.cc`/`rng.json`) -- but NEITHER exercises a real RNG
feeding a real `lazy_random_selector` through THIS specific draw/decompose
loop, forced into real Floyd's-algorithm collisions (small
`n_possible_tuples`, exhaustive draws). `tools/cpp_probe/
subsample_selector_probe.cc` (new, this task) builds and links the real
`opencog::MT19937RandGen` + `opencog::lazy_random_selector` and drives
that exact loop for seven configurations (including two exhaustive-draw,
heavy-collision cases and two zero/one-arity edge cases);
`tests/fixtures/subsample_selector.json` is its captured output;
`tests/moses/metapopulation/test_subsample.py`'s
`test_lazy_random_selector_draw_order_matches_cpp_probe` replays the
identical construction/draw/decompose loop in Python against every
config and asserts EQUALITY of both the raw draw sequence and the
decomposed tuple indices -- draw ORDER, not merely the resulting set,
per this task's own acceptance criterion.

## `tagged_item::operator const Item&()` has no Python equivalent

`recompute_scores_over_whole_dataset`/`ss_average_agreement`/
`ss_tanimoto_stats` (deme overload) all pass a `scored_instance` where
C++'s `representation::get_candidate(const instance&, bool)` expects an
`instance` (`reps[i].get_candidate(sc_inst, true)`,
`rep.get_candidate(inst, true)`, subsample.cc:52,290,406) -- this compiles
in C++ only because `scored_instance` inherits `tagged_item<instance,
ScoreT>`, which defines an implicit `operator const Item&() const { return
this->first; }` (`opencog/util/functional.h`'s own `tagged_item`, NOT
ported anywhere in this project's `scored_instance.py` -- Python has no
implicit-conversion-operator mechanism to port it INTO). Every call site
below spells this out explicitly as `.first` instead
(`rep.get_candidate(sc_inst.first, True)`), matching the same explicit
`.first` `keep_top_unique_candidates` (`merging.py`) already uses at its
own `get_candidate` call site -- a mechanical adaptation of an implicit
C++ conversion, not a behavioural divergence.

## `behave_cscore` HAS `get_ctable()`/`get_ctable_usize()` now -- the
## `._bscorer` reach-around below is a leftover, not a convention

`ss_average_agreement`/`ss_tanimoto_stats` (trees overload) call `_cscorer.
get_ctable()`/`get_ctable_usize()` in C++ (`behave_cscore.h:98,104`, both
thin one-line delegations to `_bscorer.get_ctable()`/`get_ctable_usize()`).
When this module was written `behave_cscore.py` had neither, so the code
below reaches `self._cscorer._bscorer.get_ctable()`/`.get_ctable_usize()`
directly instead, following `ensemble.py`'s "a thin one-line delegation
was not ported; call the underlying `_bscorer` attribute directly"
convention for `cs.get_bscorer()` -> `cs._bscorer`.

**Both methods are now ported** (`behave_cscore.py`, with their own
docstrings and tests). The reach-around still WORKS -- it resolves to the
identical object, since the new methods are exactly
`return self._bscorer.get_ctable()`/`get_ctable_usize()` -- so nothing
here is wrong, but the two call sites below should be re-spelled
`self._cscorer.get_ctable()`/`.get_ctable_usize()` to match C++ literally.
Not done in the same change that added them, because
`pymoses/moses/metapopulation/` was outside that task's owned files.

REQUIRED FOLLOW-UP (not fixed here -- `pymoses/moses/scoring/` is outside
this task's owned scope): this port's `ctruth_table_bscore` (`bscores.py`)
extends `bscore_base` DIRECTLY rather than `bscore_ctable_base` (a
disclosed divergence from classic's real `struct ctruth_table_bscore :
public bscore_ctable_base`, `bscores.h:360` -- `bscore_ctable_base` is
where `get_ctable`/`get_ctable_usize`'s real bodies live,
`scoring_base.cc:318,323`), so a `ctruth_table_bscore`-backed
`behave_cscore` has NO `get_ctable`/`get_ctable_usize` at all in this
port -- `self._cscorer._bscorer.get_ctable()` raises `AttributeError`
against it. Every test in `test_subsample.py` that reaches this code path
therefore uses a minimal test-local `_CTableBscore` double instead of a
real `ctruth_table_bscore` -- see that test module's own docstring for
the full explanation and exactly what the real fix would be.

## The dead "weighted activations grouped by timestamps" block is NOT ported

`ss_tanimoto_stats` (trees overload, subsample.cc:323-376) fills
`weighted_activations_grouped_by_timestamps` (subsample.cc:343-349) but
its ONLY reader is a `logger().fine()` debug block wrapped in `#ifdef
DOESNT_COMPILE` (subsample.cc:353-368) -- a macro nothing defines, so
that whole block is dead code in EVERY real build, in C++ itself, not
merely in this port. This mirrors `log_best_candidates`'s own precedent
(`merging.py`: "compute only for an unported log call" is safe to drop
once every consumer is gone) -- except here the computation has no
consumer in real C++ EITHER, so there is nothing to reproduce even in
principle. It would not be reproducible regardless: this port's own
`CTable` "carries no timestamps at all" (`table.py`'s own class
docstring, S7/S14: `TimedCounter` -> plain `Counter[vertex]`), so there is
no per-timestamp structure left to group by. Both facts are independently
sufficient; recorded together since either one alone settles it.

## `ss_average_agreement` is dead code in classic itself

`git grep -n ss_average_agreement` over `$CLASSIC/moses/` finds exactly
two hits: this method's own declaration (`metapopulation.h:470`) and
definition (`subsample.cc:241`) -- no call site anywhere, in `subsample.cc`
itself, `merging.cc`, or any other file. Ported anyway (named explicitly
in this task's own "what to port" list: "the tuple sampling and Tanimoto
filters"), tested directly as a standalone unit rather than reached
through any integration path -- there is none, in either language.

## Types: `score_t` (float) vs `count_t`/`contin_t` (double), same rule
## as everywhere else in this port

Every intermediate a real C++ `float`-typed local would hold (`float
weight`, `float total_agreement`, `float mean_std`, `stat_acc_t`'s own
`Sample=float`) is narrowed via `_to_score_t` at each step. `tanimoto_acc_t`
(`metapopulation.h:475-483`) is `Sample=double` -- `_TanimotoAcc` below
accumulates in plain Python `float` (already double precision) with no
narrowing, matching a real `double` accumulator fed already-`score_t`-
narrowed samples (each promotes to double at the `acc(...)` call site,
same "narrow the input once, accumulate at full precision" shape
`diversity.py`'s own `_gather_stats_over` docstring establishes for its
own double accumulator).

Divisions that classic's own operand types make IEEE-754 float divisions
(never raising, `x/0.0 == +-inf` or `nan`) use `_ieee_div`
(`scoring_base.py`, imported like `time_dispersion.py`/`precision_bscore.py`/
`select_bscore.py` already do -- not reimplemented) rather than Python's
raising `float.__truediv__`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pymoses.comboreduct.combo.vertex import builtin, combo_tree, vertex_equal
from pymoses.comboreduct.interpreter.interpreter import interpreter_visitor
from pymoses.moses.metapopulation.merging import _sort
from pymoses.moses.metapopulation.metapop_params import tanimoto_distance
from pymoses.moses.metapopulation.metapopulation import metapopulation
from pymoses.moses.moses.types import _to_score_t, score_t
from pymoses.moses.scoring.scoring_base import _ieee_div
from pymoses.util.lazy_random_selector import lazy_random_selector

if TYPE_CHECKING:
    from pymoses.moses.moses.optimization.optimization import deme_t
    from pymoses.moses.representation.representation import representation

__all__: list[str] = []
# Nothing new is exported -- every symbol below binds onto `metapopulation`
# (see module docstring's Composition section), matching merging.py/
# domination.py/diversity.py's own convention.

_FLT_MAX = 3.4028234663852886e38  # C++: std::numeric_limits<float>::max()
# Same literal already established in pymoses/moses/main/problem_params.py
# for `ss-std-dev-threshold`'s own CLI default -- reused, not reinvented.


def _pow_to_unsigned(base: int, exponent: int) -> int:
    """Not a ported C++ symbol -- reproduces `(unsigned)pow(base, size)`
    (subsample.cc:177,253,389: `n_possible_tuples = (unsigned)pow(base,
    ss_demes.size())` / `pow(base, exponent)`), the exact double-precision
    `::pow()` call followed by a truncating cast to `unsigned` that C++
    performs, rather than Python's exact-integer `**`. For every base/
    exponent this port's own tests use, `math.pow` and `**` agree exactly
    (IEEE-754 double has 52 mantissa bits; every value exercised here is
    far smaller) -- ported as the literal OPERATION anyway, matching this
    project's "reproduce what C++ computes, not merely an equivalent
    result" rule for any narrowing/rounding-observable call.
    """
    return int(math.pow(float(base), float(exponent)))


def _decompose_tuple(rnd_tuple: int, n_demes: int, base: int) -> list[int]:
    """Not a ported C++ symbol -- the tuple_idxs decomposition loop
    duplicated verbatim at subsample.cc:169-173 (`ss_score_dev_filter`),
    270-274 (`ss_average_agreement`), 401-405 (`ss_tanimoto_stats`, deme
    overload): `tuple_idxs.push_back(rnd_tuple % base); rnd_tuple /=
    base;`, `n_demes` times. Both operands are non-negative throughout
    this port (`rnd_tuple` comes from `lazy_random_selector.__call__`,
    always `>= 0`; `base` is `n_top_candidates`, an unsigned CLI field),
    so Python's `%`/`//` agree with C++'s unsigned `%`/`/=` exactly --
    no truncate-toward-zero sign hazard (contrast `MT19937RandGen.
    randint_n`'s own docstring, which needs `_ieee_div`-style care
    precisely because ITS dividend can be produced from signed
    arithmetic; this one never is).
    """
    idxs: list[int] = []
    for _ in range(n_demes):
        idxs.append(rnd_tuple % base)
        rnd_tuple //= base
    return idxs


class _TanimotoAcc:
    """Not itself a ported C++ symbol -- reproduces `metapopulation::
    tanimoto_acc_t` (`metapopulation.h:475-483`,
    `accumulator_set<double, stats<tag::count, tag::mean,
    tag::geometric_mean_mirror, tag::max>>`), the accumulator
    `ss_filter`/`ss_tanimoto_filter` gather Tanimoto-distance statistics
    into. `count`/`mean`/`max` are the ordinary running statistics;
    `geometric_mean_mirror` is a cogutil-local boost accumulator extension
    (`opencog/util/boost_ext/accumulators/statistics/geometric_mean_mirror.h`,
    read directly from the CogUtil checkout, not the classic MOSES one):
    "1 - sqrt[n]{prod_{i=1}^n (1 - x_i)}" (that header's own comment,
    reproduced verbatim) -- `prod` starts at `1.0` and multiplies by
    `(1.0 - x)` on every push, `result()` returns `1 - prod**(1/count)`.

    Sample type is `double` throughout (class docstring's own "Types"
    section) -- every value `add()` receives is an already-`score_t`-
    narrowed `tanimoto_distance(...)` result, promoted to `double` at the
    call boundary and never re-narrowed, matching a real `double`
    accumulator fed `float` samples.
    """

    def __init__(self) -> None:
        self._values: list[float] = []
        self._prod = 1.0  # C++: geometric_mean_mirror_impl's own prod(1.0)

    def add(self, x: float) -> None:
        x = float(x)
        self._values.append(x)
        self._prod *= 1.0 - x

    def count(self) -> int:
        """C++: boost::accumulators::tag::count"""
        return len(self._values)

    def mean(self) -> float:
        """C++: boost::accumulators::tag::mean

        `_ieee_div`: boost's own `mean` on zero pushed samples is `0/0`,
        i.e. `nan` under IEEE-754 -- not a crash, unlike Python's bare
        `/`. No test here exercises this edge (every configuration this
        port's own tests use pushes at least one value), but the
        operation is reproduced faithfully regardless of whether it is
        reached, matching `_score_mean`'s identical treatment below.
        """
        return _ieee_div(sum(self._values), len(self._values))

    def geometric_mean_mirror(self) -> float:
        """C++: opencog/util/boost_ext/accumulators/statistics/
        geometric_mean_mirror.h geometric_mean_mirror_impl::result
        """
        n = len(self._values)
        return 1.0 - self._prod ** _ieee_div(1.0, float(n))

    def max(self) -> float:
        """C++: boost::accumulators::tag::max

        boost's own zero-samples default is not reproduced precisely
        here (this port has no access to boost's own uninitialized-
        accumulator internals) -- `-inf` is used instead, a reasonable
        stand-in that can only be OBSERVED via `ss_filter`'s
        `n_best_bfdemes` branch, itself only reachable when
        `ss_demes.size() >= 2` (see `ss_tanimoto_stats`'s own docstring:
        `trs.size() == ss_demes.size()`, and the nested-pair loop this
        class's samples come from pushes zero values only when
        `trs.size() < 2`) -- i.e. only when `n_subsample_demes <= 1`,
        which is ALSO the condition under which `ss_filter`'s entire
        `n_subsample_demes > 1` arm (this method's only caller) never
        runs. Unreachable in every configuration this file's own gate
        allows, recorded rather than silently assumed.
        """
        return max(self._values) if self._values else -math.inf


def _score_variance_stat(scores: list[score_t]) -> tuple[score_t, score_t]:
    """Not a ported C++ symbol -- reproduces `stat_acc_t`
    (`accumulator_set<float, stats<tag::count, tag::mean, tag::variance>>`,
    subsample.cc:166-169,178) count/mean/(population)variance, Sample=
    `float` -- unlike `diversity.py`'s own `_gather_stats_over` (Sample=
    `double`), every intermediate here is narrowed via `_to_score_t`,
    matching a real float32 accumulator. Same two-pass population-variance
    formula as `_gather_stats_over` (divide by `n`, not `n - 1`) --
    `diversity.py`'s own docstring records that this two-pass shape was
    measured directly against boost's output for known samples, not
    merely assumed equivalent to boost's real incremental algorithm;
    the same caveat applies here.
    """
    n = len(scores)
    total: score_t = 0.0
    for x in scores:
        total = _to_score_t(total + x)
    mean = _to_score_t(_ieee_div(total, n))
    sq_total: score_t = 0.0
    for x in scores:
        diff = _to_score_t(x - mean)
        sq_total = _to_score_t(sq_total + _to_score_t(diff * diff))
    variance = _to_score_t(_ieee_div(sq_total, n))
    return mean, variance


def _score_mean(scores: list[score_t]) -> score_t:
    """Not a ported C++ symbol -- the OUTER `mean_std_acc` in
    `ss_score_dev_filter` (Sample=`float` too), collecting one std-
    deviation value per sampled tuple.
    """
    total: score_t = 0.0
    for x in scores:
        total = _to_score_t(total + x)
    return _to_score_t(_ieee_div(total, len(scores)))


def recompute_scores_over_whole_dataset(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> None:
    """C++: moses/moses/metapopulation/subsample.cc:42
    metapopulation::recompute_scores_over_whole_dataset

    "In case of ss filter, the candidates during optimization will not
    have their scores computed over the whole dataset, that method
    corrects that" (metapopulation.h:483-485's own comment). The disabled
    arm (`n_subsample_demes <= 1`, the default) is a pure no-op, exactly
    as in C++ -- this function used to BE that no-op gate, in `merging.py`
    (see this module's own docstring); now the enabled arm is real too.

    `select_tag()(sc_inst) = sc` (subsample.cc:70) assigns THROUGH the
    `tagged_item`/`std::pair` reference -- `scored_instance.second = sc`
    mutates the SAME object still sitting in `ss_deme` (a plain Python
    list of `scored_instance`s, `scored_instance.__slots__` includes
    `second`), matching that in-place update exactly; no re-insertion
    into `ss_deme` is needed or performed, in either language.
    """
    if self._filter_params.n_subsample_demes > 1:
        for i in range(len(all_demes)):
            for ss_deme in all_demes[i]:
                for sc_inst in ss_deme:
                    tr = reps[i].get_candidate(sc_inst.first, True)
                    sc_inst.second = self._cscorer.get_cscore(tr)


def ss_score_dev_filter(
    self: metapopulation,
    rep: representation,
    ss_demes: list[deme_t],
) -> bool:
    """C++: moses/moses/metapopulation/subsample.cc:160
    metapopulation::ss_score_dev_filter

    "Subsample filter. Return true if it passes the subsample filter.
    I.e. if the variance of the score across demes pass the filter"
    (metapopulation.h:461-463's own comment). Samples `sample_tuples`
    tuples (each tuple picking one candidate index per SS-deme, `[0,
    n_top_candidates)`), computes the per-tuple score std-deviation
    across demes, and passes iff the MEAN of those std-deviations is
    below `std_dev_threshold`.

    `logger().fine()`/`is_fine_enabled()`/`is_debug_enabled()` calls
    dropped throughout (module-wide port convention, no logging surface
    anywhere in this project).
    """
    n_top_candidates = self._filter_params.n_top_candidates
    n_tuples = self._filter_params.n_tuples
    base = n_top_candidates
    n_possible_tuples = _pow_to_unsigned(base, len(ss_demes))
    sample_tuples = min(n_possible_tuples, n_tuples)

    selector = lazy_random_selector(n_possible_tuples)

    std_values: list[score_t] = []
    for _i_tuple in range(sample_tuples):
        rnd_tuple = selector()
        tuple_idxs = _decompose_tuple(rnd_tuple, len(ss_demes), base)

        var_scores: list[score_t] = []
        for i_deme in range(len(ss_demes)):
            ss_deme = ss_demes[i_deme]
            i_inst = tuple_idxs[i_deme]
            inst = ss_deme[i_inst]
            var_scores.append(inst.second.get_penalized_score())

        _mean, variance = _score_variance_stat(var_scores)
        std_values.append(_to_score_t(math.sqrt(variance)))

    mean_std = _score_mean(std_values)
    return mean_std < self._filter_params.std_dev_threshold


def ss_average_agreement(
    self: metapopulation,
    rep: representation,
    ss_demes: list[deme_t],
) -> score_t:
    """C++: moses/moses/metapopulation/subsample.cc:241
    metapopulation::ss_average_agreement

    "Given the top candidates of each deme, what is the average agreement
    between those candidates, for each data point, across all ss-demes"
    (metapopulation.h:467-469's own comment). DEAD CODE in classic itself
    -- see this module's own docstring; ported and tested directly, never
    reached through any integration path in either language.
    """
    n_top_candidates = self._filter_params.n_top_candidates
    n_tuples = self._filter_params.n_tuples
    exponent = len(ss_demes)
    base = n_top_candidates
    n_possible_tuples = _pow_to_unsigned(base, exponent)
    sample_tuples = min(n_possible_tuples, n_tuples)

    selector = lazy_random_selector(n_possible_tuples)

    total_agreement: score_t = 0.0
    for _i_tuple in range(sample_tuples):
        rnd_tuple = selector()
        tuple_idxs = _decompose_tuple(rnd_tuple, exponent, base)

        trs: list[combo_tree] = []
        for i_deme in range(exponent):
            deme = ss_demes[i_deme]
            i_inst = tuple_idxs[i_deme]
            inst = deme[i_inst]
            trs.append(rep.get_candidate(inst.first, True))

        for irow, orow in self._cscorer._bscorer.get_ctable().items():
            agreement = 0
            for tr in trs:
                iv = interpreter_visitor(tr)
                result = iv(list(irow))
                if vertex_equal(result, builtin.logical_true):
                    agreement += 1
                else:
                    agreement -= 1
            normalized_agreement = _to_score_t(_ieee_div(agreement, exponent))
            product = abs(normalized_agreement) * sum(orow.values())
            total_agreement = _to_score_t(total_agreement + product)

    mean_agreement = _to_score_t(
        _ieee_div(total_agreement, sample_tuples * self._cscorer._bscorer.get_ctable_usize())
    )
    return mean_agreement


def _ss_tanimoto_stats_trees(
    self: metapopulation,
    trs: list[combo_tree],
    acc: _TanimotoAcc,
) -> None:
    """C++: moses/moses/metapopulation/subsample.cc:323
    metapopulation::ss_tanimoto_stats (vector<combo_tree>& overload)

    "Return stats about the Tanimoto distances of a vector of combo
    trees" (metapopulation.h:495-496's own comment). Renamed with a
    leading underscore and a `_trees` suffix: Python has no overloading,
    and this is the PRIVATE, tree-vector-taking overload the public
    (deme-taking) `ss_tanimoto_stats` below calls internally -- matching
    the header's own comment ("Return stats about the Tanimoto distances
    of the TOP CANDIDATES of a certain breadth first deme", :500-502)
    that distinguishes the public overload as the one real callers use.

    NOT ported: the "Fill weighted activations grouped by timestamps"
    inner loop and its sole (dead, `#ifdef DOESNT_COMPILE`) reader -- see
    this module's own docstring for why neither is reproducible or
    observable.

    `float weight = result == id::logical_true ? orow.total_count() :
    0.0;` (subsample.cc:340) -- `vertex_equal`, not a bare `==`, per this
    project's type-before-value rule; `orow.total_count()` -> `sum(orow.
    values())`, the established `Counter.total_count()` idiom
    (`discriminating_bscore.py`'s own docstring).
    """
    weighted_activations: list[list[score_t]] = [[] for _ in trs]
    for irow, orow in self._cscorer._bscorer.get_ctable().items():
        for i, tr in enumerate(trs):
            iv = interpreter_visitor(tr)
            result = iv(list(irow))
            is_true = vertex_equal(result, builtin.logical_true)
            weight: score_t = _to_score_t(sum(orow.values())) if is_true else 0.0
            weighted_activations[i].append(weight)

    for i in range(len(trs)):
        for j in range(i):
            acc.add(tanimoto_distance(weighted_activations[i], weighted_activations[j]))


def ss_tanimoto_stats(
    self: metapopulation,
    rep: representation,
    ss_demes: list[deme_t],
    acc: _TanimotoAcc,
) -> None:
    """C++: moses/moses/metapopulation/subsample.cc:378
    metapopulation::ss_tanimoto_stats (deme overload)

    "Return stats about the Tanimoto distances of the top candidates of a
    certain breadth first deme" (metapopulation.h:500-502's own comment).
    `trs` built here always has `len(trs) == len(ss_demes)` (`exponent`)
    -- see `_TanimotoAcc.max`'s own docstring, which depends on this
    invariant.
    """
    n_top_candidates = self._filter_params.n_top_candidates
    n_tuples = self._filter_params.n_tuples
    exponent = len(ss_demes)
    base = n_top_candidates
    n_possible_tuples = _pow_to_unsigned(base, exponent)
    sample_tuples = min(n_possible_tuples, n_tuples)

    selector = lazy_random_selector(n_possible_tuples)

    for _i_tuple in range(sample_tuples):
        rnd_tuple = selector()
        tuple_idxs = _decompose_tuple(rnd_tuple, exponent, base)

        trs: list[combo_tree] = []
        for i_deme in range(exponent):
            deme = ss_demes[i_deme]
            i_inst = tuple_idxs[i_deme]
            inst = deme[i_inst]
            trs.append(rep.get_candidate(inst.first, True))

        self._ss_tanimoto_stats_trees(trs, acc)


def ss_tanimoto_filter(
    self: metapopulation,
    rep: representation,
    ss_demes: list[deme_t],
) -> bool:
    """C++: moses/moses/metapopulation/subsample.cc:430
    metapopulation::ss_tanimoto_filter

    "Return true if the top candidates of the subsampled demes have
    sufficiently low Tanimoto distance (mean or max)" (metapopulation.h:
    506-511's own comment, condensed) -- passes iff mean, geometric mean
    AND max all fall below their respective thresholds.
    """
    acc = _TanimotoAcc()
    self.ss_tanimoto_stats(rep, ss_demes, acc)

    tanimoto_mean = acc.mean()
    tanimoto_geo_mean = acc.geometric_mean_mirror()
    tanimoto_max = acc.max()

    mean_pass = tanimoto_mean < self._filter_params.tanimoto_mean_threshold
    geo_mean_pass = tanimoto_geo_mean < self._filter_params.tanimoto_geo_mean_threshold
    max_pass = tanimoto_max < self._filter_params.tanimoto_max_threshold

    return mean_pass and geo_mean_pass and max_pass


def ss_filter(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> list[bool]:
    """C++: moses/moses/metapopulation/subsample.cc:77 metapopulation::ss_filter

    "Breadth first demes pass vector. By default they all pass the
    filter" (subsample.cc's own comment) -- `bfpass` starts all-`True`,
    unconditionally, exactly as in C++; the disabled arm (`n_subsample_
    demes <= 1`) returns it untouched, matching this function's PRIOR
    (gate-only) body in `merging.py` exactly (see this module's own
    docstring).

    Two mutually exclusive branches once enabled: `n_best_bfdemes > 0`
    picks the `n_best_bfdemes` demes with the LOWEST aggregated Tanimoto
    distance (discarding the rest); otherwise each deme is checked
    independently against the score-deviation and/or Tanimoto thresholds.
    `_filter_params.std_dev_threshold < _FLT_MAX` is subsample.cc's own
    "(enabled)" check (subsample.cc:132-133) -- classic's CLI default
    (`std::numeric_limits<float>::max()`, `problem-params.cc:1951`) is
    what keeps this branch off by default even when subsampling itself is
    on.
    """
    bfpass = [True] * len(all_demes)

    if self._filter_params.n_subsample_demes > 1:
        if self._filter_params.n_best_bfdemes > 0:
            n_best_bfdemes = min(self._filter_params.n_best_bfdemes, len(all_demes))

            ag_dsts: list[float] = [0.0] * len(all_demes)
            for i in range(len(all_demes)):
                acc = _TanimotoAcc()
                self.ss_tanimoto_stats(reps[i], all_demes[i], acc)

                tanimoto_mean = acc.mean()
                tanimoto_geo_mean = acc.geometric_mean_mirror()
                tanimoto_max = acc.max()

                ag_dsts[i] = (
                    self._filter_params.tanimoto_mean_weight * tanimoto_mean
                    + self._filter_params.tanimoto_geo_mean_weight * tanimoto_geo_mean
                    + self._filter_params.tanimoto_max_weight * tanimoto_max
                )

            # `boost::sort(worst_idxs, [&](i1, i2){ return ag_dsts[i1] <
            # ag_dsts[i2]; })` (subsample.cc:137-139) -- ascending by
            # aggregated distance. Reused `_sort` (merging.py), not a new
            # sort -- see this module's own docstring's "Sorting is not
            # free" rule.
            worst_idxs = list(range(len(ag_dsts)))
            _sort(worst_idxs, 0, len(worst_idxs), lambda i1, i2: ag_dsts[i1] < ag_dsts[i2])
            del worst_idxs[:n_best_bfdemes]

            for idx in worst_idxs:
                bfpass[idx] = False
        else:
            for i in range(len(all_demes)):
                if self._filter_params.std_dev_threshold < _FLT_MAX:
                    bfpass[i] = self.ss_score_dev_filter(reps[i], all_demes[i])
                    if not bfpass[i]:
                        continue
                if (
                    self._filter_params.tanimoto_mean_threshold < 1.0
                    or self._filter_params.tanimoto_geo_mean_threshold < 1.0
                    or self._filter_params.tanimoto_max_threshold < 1.0
                ):
                    bfpass[i] = self.ss_tanimoto_filter(reps[i], all_demes[i])

    return bfpass


metapopulation.recompute_scores_over_whole_dataset = recompute_scores_over_whole_dataset
metapopulation.ss_score_dev_filter = ss_score_dev_filter
metapopulation.ss_average_agreement = ss_average_agreement
metapopulation._ss_tanimoto_stats_trees = _ss_tanimoto_stats_trees
metapopulation.ss_tanimoto_stats = ss_tanimoto_stats
metapopulation.ss_tanimoto_filter = ss_tanimoto_filter
metapopulation.ss_filter = ss_filter
