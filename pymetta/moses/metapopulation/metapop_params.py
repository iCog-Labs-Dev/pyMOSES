"""C++: moses/moses/metapopulation/metapop_params.{h,cc}

`diversity_parameters`, `subsample_deme_filter_parameters` and
`metapop_parameters` -- ported verbatim, field for field and default for
default. These defaults decide which paths through the rest of the S10
slice (and every later task that reads `metapop_parameters`) are live:

* `metapop_parameters.do_boosting` defaults `False` (metapop_params.h:170) --
  the boosting/ensemble-merge path in `merging.cc` (S10 Task 3) is dead
  unless a caller flips this.
* `metapop_parameters.discard_dominated` defaults `False`
  (metapop_params.h:211) -- `merging.cc:163-175`'s `else` arm
  (`merge_nondominated`) is likewise dead by default; the plain insert-all
  path runs instead.
* `diversity_parameters.enabled()` returns `False` by default (`pressure
  == 0.0`) -- the diversity penalty in `merging.cc`/`diversity.cc` (P2) is
  dead unless a caller raises `pressure` above `0.0`.
* `subsample_deme_filter_parameters`'s filter stays "disactivated" by
  default (`n_subsample_demes == 0`, per the header's own :124-126
  comment).

`p_norm_distance`/`tanimoto_distance`/`angular_distance` (cogutil
`opencog/util/numeric.h:294,361,401`) are ported alongside `diversity_
parameters` -- the same "port the cogutil helper where its first real
caller needs it" convention `field_set.py`'s `nbits_to_pack` already
follows for the same header -- because `diversity_parameters.set_dst`'s
own body (metapop_params.cc:36-61) directly wires one of them up as the
default `dst` callable; leaving them unported would make even
`diversity_parameters()`'s OWN constructor (which calls `set_dst(p_norm,
2.0)` unconditionally, regardless of whether diversity is enabled) call
into a stub. None of these three are on any default-live path either
(`dst`/`dst2dp` are read only from `diversity.cc`/`merging.cc`'s diversity
branch, itself gated on `enabled()` -- P2, not ported), but they are cheap,
self-contained formulas (not half-implemented algorithms), so they are
ported for real rather than stubbed.

`score_t` (`dp_t` in `diversity_parameters`) is `float` (types.py's own
module docstring); every intermediate this module computes over `dp_t`
values is narrowed via `_to_score_t` at each step a C++ `float` variable
would itself be re-rounded, the same convention `types.py`'s
`composite_score` and `_is_approx_eq` use. `metapop_parameters.cap_coef`
and `ensemble_parameters`'s `expalpha`/`bias_scale` are plain C++ `double`
(NOT `score_t`) and are therefore never narrowed here, matching that same
established double-vs-score_t distinction.
"""

from __future__ import annotations

import copy
import math
from enum import Enum

from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters
from pymetta.moses.moses.types import _to_score_t, score_t

__all__ = [
    "angular_distance",
    "diversity_parameters",
    "metapop_parameters",
    "p_norm_distance",
    "subsample_deme_filter_parameters",
    "tanimoto_distance",
]

_UINT_MAX = 0xFFFFFFFF  # C++: <climits> UINT_MAX (32-bit unsigned int)

_DISTANCE_EPSILON: score_t = _to_score_t(1e-32)  # C++: opencog/util/numeric.h:51 DISTANCE_EPSILON


def _dot(a: list[score_t], b: list[score_t]) -> score_t:
    """Not itself a ported C++ symbol -- `boost::inner_product(a, b,
    Float(0))`, the dot product `tanimoto_distance`/`angular_distance`
    both compute three times each (`ab`, `aa`, `bb`). `Float` is `dp_t` =
    `score_t` at every call site in this module, so the accumulator itself
    is narrowed after each term, matching a real float32 running sum
    rather than a double one narrowed once at the end (contrast
    `scoring_base.py`'s `sum_bscore`, which deliberately does NOT do this
    because its own C++ accumulator is `double` -- see that function's
    docstring for why the two cases differ).
    """
    total: score_t = 0.0
    for x, y in zip(a, b, strict=True):
        total = _to_score_t(total + _to_score_t(x * y))
    return total


def p_norm_distance(a: list[score_t], b: list[score_t], p: score_t = 1.0) -> score_t:
    """C++: opencog/util/numeric.h:294 p_norm_distance

    "Compute the distance between two vectors, using the p-norm. For p=2,
    this is the usual Euclidean distance, and for p=1, this is the
    Manhattan distance, and for p=0 or negative, this is the maximum
    difference for one element." (numeric.h:289-292's own doc comment.)
    Instantiated in this port only at `Float = dp_t = score_t`
    (`diversity_parameters.dst`'s p_norm branch), so `p` and every
    intermediate below are narrowed via `_to_score_t` the same way a real
    `float p_norm_distance<vector<score_t>, score_t>` instantiation would
    compute in float32 throughout, not in double.

    `OC_ASSERT(a.size() == b.size())` (numeric.h:296-298) becomes
    `ValueError`, matching this port's established OC_ASSERT-to-exception
    convention (see e.g. `lazy_random_selector.py`'s own DIVERGENCE notes).
    """
    if len(a) != len(b):
        raise ValueError(f"Cannot compare unequal-sized vectors! {len(a)} {len(b)}")
    p = _to_score_t(p)

    if p == 1.0:
        total: score_t = 0.0
        for x, y in zip(a, b, strict=True):
            total = _to_score_t(total + _to_score_t(abs(_to_score_t(x - y))))
        return total

    if p == 2.0:
        total = 0.0
        for x, y in zip(a, b, strict=True):
            diff = _to_score_t(x - y)
            total = _to_score_t(total + _to_score_t(diff * diff))
        return _to_score_t(math.sqrt(total))

    if p <= 0.0:
        total = 0.0
        for x, y in zip(a, b, strict=True):
            diff = _to_score_t(abs(_to_score_t(x - y)))
            if total < diff:
                total = diff
        return total

    total = 0.0
    for x, y in zip(a, b, strict=True):
        diff = _to_score_t(abs(_to_score_t(x - y)))
        if diff > 0.0:
            total = _to_score_t(total + _to_score_t(diff**p))
    return _to_score_t(total ** _to_score_t(1.0 / p))


def tanimoto_distance(a: list[score_t], b: list[score_t]) -> score_t:
    """C++: opencog/util/numeric.h:361 tanimoto_distance

    "Return the Tanimoto distance, a continuous extension of the Jaccard
    distance, between 2 vector[s] ... f(a,b) = (sum_i a_i * b_i) / (sum_i
    a_i^2 + sum_i b_i^2 - sum_i a_i * b_i)" (numeric.h:333-359's own doc
    comment, condensed). The `#ifdef HAVE_BOOST`/`#else throw
    RuntimeException` split (numeric.h:367-379) is not ported: cogutil is
    always built with boost in this project (`$COGUTIL`'s own build), so
    the `#else` arm is dead code here exactly the way `CLAUDE.md`
    documents for other never-taken `#ifdef` branches elsewhere in this
    port.
    """
    if len(a) != len(b):
        raise ValueError(f"Cannot compare unequal-sized vectors! {len(a)} {len(b)}")
    ab = _dot(a, b)
    aa = _dot(a, a)
    bb = _dot(b, b)
    numerator = _to_score_t(_to_score_t(aa + bb) - ab)
    if numerator >= _DISTANCE_EPSILON:
        return _to_score_t(1 - _to_score_t(ab / numerator))
    return 0.0


def angular_distance(a: list[score_t], b: list[score_t], pos_n_neg: bool = True) -> score_t:
    """C++: opencog/util/numeric.h:401 angular_distance

    "f(a,b) = alpha*cos^{-1}((sum_i a_i * b_i) / (sqrt(sum_i a_i^2) *
    sqrt(sum_i b_i^2))) / pi ... If pos_n_neg == true then alpha = 1,
    otherwise alpha = 2." (numeric.h:390-398's own doc comment.) Same
    `#ifdef HAVE_BOOST` non-porting note as `tanimoto_distance` above.
    """
    if len(a) != len(b):
        raise ValueError(f"Cannot compare unequal-sized vectors! {len(a)} {len(b)}")
    ab = _dot(a, b)
    aa = _dot(a, a)
    bb = _dot(b, b)
    numerator = _to_score_t(math.sqrt(_to_score_t(aa * bb)))
    if numerator >= _DISTANCE_EPSILON:
        r = _to_score_t(ab / numerator)
        r = max(_to_score_t(-1.0), min(_to_score_t(1.0), r))  # C++: numeric.h:199 clamp
        alpha = 1 if pos_n_neg else 2
        return _to_score_t(alpha * math.acos(r) / math.pi)
    return 0.0


class diversity_parameters:
    """C++: moses/moses/metapopulation/metapop_params.h:32 diversity_parameters

    `dst`/`dst2dp` are C++ `std::function` DATA MEMBERS wired up by
    `set_dst`/`set_dst2dp` -- here they are METHODS that dispatch on a
    stored kind instead, so `dp.dst(lhs, rhs)`/`dp.dst2dp(x)` are called
    exactly the same way from either language (`git grep '\\.dst('`/
    `'\\.dst2dp('` over classic's `moses/` finds every real call site going
    through this same `obj.dst(...)`/`obj.dst2dp(...)` syntax, never a
    direct `obj.dst = ...` reassignment outside `set_dst` itself). The
    method form is not merely equivalent, it is MORE faithful to one part
    of the C++ semantics: `dst2dp`'s three lambdas all capture `pressure`
    BY REFERENCE (`[&]`, metapop_params.cc:68,71,74), so C++ itself reads
    whatever `pressure` holds AT CALL TIME, not at `set_dst2dp` time -- a
    plain Python method reading `self.pressure` reproduces that exactly. A
    stored closure capturing `self` would too, but would silently keep
    pointing at the ORIGINAL object through a shallow `copy.copy` (needed
    because `metapop_parameters.__init__` copies its own `_diversity`
    parameter into the `diversity` member by value, metapop_params.h:167,
    238) -- a method looked up via the copy's own class does not have that
    problem, since Python binds `self` to whichever instance the method is
    looked up on.
    """

    dp_t = score_t  # C++: moses/moses/metapopulation/metapop_params.h:34 dp_t

    class dst_enum_t(Enum):
        """C++: moses/moses/metapopulation/metapop_params.h:69 dst_enum_t"""

        p_norm = "p_norm"
        tanimoto = "tanimoto"
        angular = "angular"

    class dst2dp_enum_t(Enum):
        """C++: moses/moses/metapopulation/metapop_params.h:95 dst2dp_enum_t"""

        inverse = "inverse"
        complement = "complement"
        pthpower = "pthpower"

    def __init__(self) -> None:
        """C++: moses/moses/metapopulation/metapop_params.cc:27 diversity_parameters

        `pressure(0.0)`, `exponent(-1.0)` ("max"), `normalize(true)`
        ("sum or mean (default mean)") per the initializer list, then the
        body calls `set_dst(p_norm, 2.0)` ("Euclidean") and
        `set_dst2dp(inverse)` unconditionally -- both run regardless of
        whether diversity is ever enabled (`enabled()` checks `pressure`
        alone), matching this port's ordering exactly.

        # DIVERGENCE(C++ moses/moses/metapopulation/metapop_params.h:46):
        # `autoscale` ("Whether pressure should be rescaled to match the
        # magnitude of the score", the header's own comment) is declared
        # with NO default-member-initializer, and this constructor's own
        # initializer list (metapop_params.cc:27-30) never touches it
        # either -- the same genuinely-uninitialized-in-C++ shape
        # `subsample_deme_filter_parameters`'s own DIVERGENCE note
        # documents for its ten fields, and the same fix: this port never
        # sets `autoscale`, so reading it before an explicit assignment
        # raises `AttributeError` instead of returning C++'s indeterminate
        # garbage. Verified live (not assumed): `autoscale` IS both set --
        # by the S13-scope `--diversity-autoscale` CLI option
        # (`problem-params.cc:2295`, `meta_params.diversity.autoscale =
        # diversity_autoscale;`) -- and read (`moses_main.cc:106`,
        # `autoscale_diversity`, gated on `diversity.enabled()`) in a real
        # run, so whoever ports that S13 CLI wiring must populate this
        # field before `diversity.enabled()` can ever be true, exactly
        # the same obligation the subsample fields' own note names for
        # their `ss-*` options.
        """
        self.pressure: score_t = 0.0
        self.exponent: score_t = -1.0
        self.normalize = True
        self._dst_kind: diversity_parameters.dst_enum_t | None = None
        self._dst_p: score_t = 0.0
        self.dst2dp_type: diversity_parameters.dst2dp_enum_t | None = None
        self.set_dst(self.dst_enum_t.p_norm, 2.0)
        self.set_dst2dp(self.dst2dp_enum_t.inverse)

    def set_dst(self, de: dst_enum_t, p: score_t = 0.0) -> None:
        """C++: moses/moses/metapopulation/metapop_params.cc:36 diversity_parameters::set_dst

        `p`'s C++ default is `0.0` (metapop_params.h:70); the constructor
        above overrides it to `2.0` explicitly, exactly as C++ does. The
        `default: OC_ASSERT(false);` branch (metapop_params.cc:58-59) is
        unreachable through this port's closed `Enum` the way it is
        (barring UB) through C++'s own `enum` -- kept anyway as `ValueError`
        for the same "document the assert" reason `set_dst2dp` below does.
        """
        if de not in (self.dst_enum_t.p_norm, self.dst_enum_t.tanimoto, self.dst_enum_t.angular):
            raise ValueError(de)  # C++: metapop_params.cc:59 OC_ASSERT(false)
        self._dst_kind = de
        self._dst_p = _to_score_t(p)

    def dst(self, lhs: list[score_t], rhs: list[score_t]) -> score_t:
        """C++: moses/moses/metapopulation/metapop_params.h:72 dst

        The `std::function<dp_t(const vector<score_t>&, const
        vector<score_t>&)>` member itself -- see the class docstring for
        why this is a method rather than a stored closure.
        """
        if self._dst_kind is self.dst_enum_t.p_norm:
            return p_norm_distance(lhs, rhs, self._dst_p)
        if self._dst_kind is self.dst_enum_t.tanimoto:
            return tanimoto_distance(lhs, rhs)
        if self._dst_kind is self.dst_enum_t.angular:
            return angular_distance(lhs, rhs)
        raise ValueError(self._dst_kind)  # set_dst was never called; unreachable via __init__

    def set_dst2dp(self, d2de: dst2dp_enum_t) -> None:
        """C++: moses/moses/metapopulation/metapop_params.cc:63 diversity_parameters::set_dst2dp

        The `default:` branch (metapop_params.cc:76-80) builds a
        `std::stringstream` message before asserting; reproduced as the
        `ValueError`'s own message rather than a separately formatted
        stream, since Python's exception message plays the same role.
        """
        if d2de not in (
            self.dst2dp_enum_t.inverse,
            self.dst2dp_enum_t.complement,
            self.dst2dp_enum_t.pthpower,
        ):
            raise ValueError(f"diversity_parameters::set_dst2dp error: no case for {d2de}")
        self.dst2dp_type = d2de

    def dst2dp(self, dst: score_t) -> score_t:
        """C++: moses/moses/metapopulation/metapop_params.h:99 dst2dp

        The `std::function<dp_t(dp_t)>` member itself -- see the class
        docstring for the by-reference-capture reasoning this method form
        reproduces. Each branch narrows its own result, matching the
        single float32 expression each C++ lambda body computes
        (metapop_params.cc:68,71,74).
        """
        dst = _to_score_t(dst)
        if self.dst2dp_type is self.dst2dp_enum_t.inverse:
            return _to_score_t(self.pressure / _to_score_t(1 + dst))
        if self.dst2dp_type is self.dst2dp_enum_t.complement:
            return _to_score_t(self.pressure * _to_score_t(1 - dst))
        if self.dst2dp_type is self.dst2dp_enum_t.pthpower:
            return _to_score_t(math.pow(dst, self.pressure))
        raise ValueError(self.dst2dp_type)  # set_dst2dp was never called; unreachable via __init__

    def enabled(self) -> bool:
        """C++: moses/moses/metapopulation/metapop_params.cc:84 diversity_parameters::enabled

        "Return true if diversity is enabled" (metapop_params.h:101's own
        comment) -- `pressure > 0.0`. `0.0` (the default) is disabled.
        """
        return self.pressure > 0.0


class subsample_deme_filter_parameters:
    """C++: moses/moses/metapopulation/metapop_params.h:108 subsample_deme_filter_parameters

    "Parameters about subsample deme filter" (the header's own comment).
    """

    def __init__(self, n_subsample_demes: int = 0, n_top_candidates: int = 1) -> None:
        """C++: moses/moses/metapopulation/metapop_params.h:110 subsample_deme_filter_parameters

        Only the FIVE members the constructor's own initializer list
        actually sets are given values here -- `by_time`, `contiguous_time`,
        `n_subsample_demes`, `n_top_candidates`, `n_tuples`. See the
        DIVERGENCE note below for the other ten. Ported verbatim: "If set 0
        or 1 subsample_deme_filter remains disactivated" (metapop_params.h:
        124-126's own comment) -- the default `n_subsample_demes=0`
        satisfies that.
        """
        self.by_time = True
        self.contiguous_time = True
        self.n_subsample_demes = n_subsample_demes
        self.n_top_candidates = n_top_candidates
        self.n_tuples = _UINT_MAX

        # DIVERGENCE(C++ moses/moses/metapopulation/metapop_params.h:138-155):
        # `std_dev_threshold`, `tanimoto_mean_threshold`,
        # `tanimoto_geo_mean_threshold`, `tanimoto_max_threshold`,
        # `n_best_bfdemes`, `tanimoto_mean_weight`, `tanimoto_geo_mean_weight`,
        # `tanimoto_max_weight`, `n_subsample_fitnesses`, `low_dev_pressure`
        # are declared as plain `float`/`unsigned` members with NO
        # default-member-initializer, and the constructor's own initializer
        # list (metapop_params.h:111-116) never touches them either -- so a
        # freshly-constructed C++ `subsample_deme_filter_parameters`
        # genuinely leaves these ten fields holding INDETERMINATE
        # (uninitialized stack) values until something else assigns them.
        # Python has no equivalent of an uninitialized primitive, so rather
        # than invent a plausible-looking default (which would silently
        # mask "this was never set" the way a stray `0.0` could) this port
        # simply never sets these ten attributes: reading one before an
        # explicit assignment raises `AttributeError` instead of returning
        # C++'s indeterminate garbage. Louder than C++, not equivalent to
        # it -- the same "poison rather than silently substitute" choice
        # `util/tree.py`'s `_DEAD` sentinel documents for a different class
        # of hazard. This filter is off by default (`n_subsample_demes=0`)
        # and nothing in this task's scope reads these fields; whoever
        # ports the S13 CLI wiring that sets them (`problem-params.cc`'s
        # `ss-*` options) must populate every one of them before
        # `n_subsample_demes` is ever raised above 1.


class metapop_parameters:
    """C++: moses/moses/metapopulation/metapop_params.h:161 metapop_parameters

    "parameters about metapopulation management" (the header's own
    comment).
    """

    def __init__(
        self,
        max_candidates_per_deme: int = -1,
        revisit: int = 0,
        complexity_temperature: score_t = 6.0,
        jobs: int = 1,
        diversity: diversity_parameters | None = None,
    ) -> None:
        """C++: moses/moses/metapopulation/metapop_params.h:163 metapop_parameters

        `diversity` defaults to a FRESH `diversity_parameters()` per call
        (matching C++'s own `_diversity = diversity_parameters()` default
        argument, which likewise constructs a new temporary at every call
        site that does not pass one explicitly) rather than one shared
        mutable default object -- the standard Python trap a literal
        `diversity_parameters()` default value would otherwise create.
        When a caller DOES pass one explicitly, it is `copy.copy`'d before
        being stored: C++ copies `_diversity` BY VALUE into the `diversity`
        member (metapop_params.h:167,238, `diversity_parameters diversity;`
        with no `&`), so two `metapop_parameters` built from the same
        caller-owned `diversity_parameters` must not alias -- the same
        "copy a subtree's data, don't alias it" rule CARRY-FORWARD.md 2.4
        documents for `tree.py`. A shallow copy is sufficient here (see
        `diversity_parameters`'s own class docstring: every field is an
        immutable scalar or Enum member, and `dst`/`dst2dp` are methods
        looked up via the copy's own class, not stored closures over the
        original object).

        `complexity_temperature` is `score_t`-typed in C++
        (metapop_params.h:165, `6.0f`) and is narrowed at the binding, the
        same convention every other `score_t` parameter in this port
        follows. `cap_coef` is plain C++ `double` (`50.0`, not `50.0f`)
        and is therefore NOT narrowed, matching `ensemble_parameters`'s own
        `expalpha`/`bias_scale`.

        `ensemble_params` (metapop_params.h:241) is a class-type member
        with no entry in the constructor's own initializer list, so C++
        default-constructs it in place -- `ensemble_parameters()`, exactly
        what this port assigns here.

        `merge_callback`/`callback_user_data` (metapop_params.h:243-244)
        are a raw C function pointer and a raw `void*` hook -- both
        default `NULL`/`None`. Nothing in this port's scope ever sets or
        calls either.
        """
        self.max_candidates_per_deme = max_candidates_per_deme
        self.revisit = revisit
        self.do_boosting = False
        self.discard_dominated = False
        self.keep_bscore = False
        self.complexity_temperature: score_t = _to_score_t(complexity_temperature)
        self.cap_coef = 50.0
        self.jobs = jobs
        self.diversity = copy.copy(diversity) if diversity is not None else diversity_parameters()
        self.ensemble_params = ensemble_parameters()
        self.merge_callback = None
        self.callback_user_data = None
