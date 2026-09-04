"""C++: moses/moses/moses/types.h and types.cc

`composite_score` (types.h:77), `behavioral_score` (types.h:212),
`scored_combo_tree` (types.h:259) -- narrowed to exactly the surface
task-6-brief.md's Interfaces section asks for. Everything else in types.h
is out of scope for this task and this whole S0-S6 cycle:

* `demeID_t` -- a deme-bookkeeping string used by the outer search loop
  (moses/moses/moses/), which this cycle does not port.
* `scored_combo_tree_set`/`_tset`/`_ptr_set` and their comparator/hash/
  equality functors (`sct_score_greater`, `sct_tree_greater`,
  `scored_combo_tree_hash`, `scored_combo_tree_equal`) -- metapopulation
  container plumbing, likewise out of scope. `bscore_base::operator()
  (const scored_combo_tree_set&)` (scoring_base.h) is consequently not
  ported either -- see scoring_base.py's module docstring.
* The `ostream_*`/`istream_*` free functions at the bottom of types.h --
  no stream-printing surface is in this cycle's scope (same narrowing
  table.py's module docstring makes for `Table`).

`score_t` (a C++ `typedef float`, i.e. IEEE-754 binary32) and `count_t`
(CTable's `typedef double`, table.h:494) are DIFFERENT C++ types that this
port cannot distinguish at the type-system level -- Python has one
built-in floating-point type, and every score_t-typed value here is a
plain Python `float` (64-bit, i.e. what C++ calls `double`).

# DIVERGENCE(C++ moses/moses/moses/types.h:60): `score_t` is `float`
# (32-bit); this port keeps every score_t-typed value as a 64-bit Python
# `float` instead of narrowing it to binary32 precision at each of the
# several points classic's own arithmetic does (sum_bscore's return,
# composite_score's constructor/update_penalized_score, the
# complexity_penalty = cpxy * cpxy_coef multiplication in
# behave_cscore.get_cscore_nocache). Reproducing that exactly would need
# `struct.pack('f', ...)` round-trips at every one of those sites (stdlib-
# only, so it *is* available -- see util/rng.py's randfloat() note on the
# mechanism) purely to shave off double's extra ~9 significant decimal
# digits.
#
# This was verified empirically, not just reasoned about: running
# tools/cpp_probe/score_probe.cc over tests/fixtures/table_data/
# weighted.csv (fractional row weights, so the float32-vs-double gap is
# actually visible) prints a behavioral score element of
# -0.40000000596046448 -- the exact IEEE-754 binary32 value nearest 0.4,
# not 0.4 itself. The gap between that and this port's double-precision
# -0.4 is about 6e-9. classic's OWN composite_score::operator==
# (types.cc:152) tolerates exactly this class of gap already -- it uses
# is_approx_eq with EPSILON=1e-6 ("used in test cases -- compare equality
# to 7 decimal places", types.h:142's own comment) specifically because
# score_t arithmetic is float32 and cross-platform/cross-compiler bit-
# exactness was never assumed even in C++. Porting that same tolerance
# (composite_score.__eq__ below) absorbs the double-vs-float32 gap
# completely; the oracle tests in tests/moses/scoring/ compare against
# tools/cpp_probe/score_probe.cc's output the same tolerant way
# (pytest.approx), rather than by bit-exact equality.
"""

import functools
import math
import struct

from pymetta.moses.moses.complexity import complexity_t, least_complexity

score_t = float  # C++: moses/moses/moses/types.h:60 score_t
# `complexity_t`/`least_complexity` (complexity.h:35,37) are imported
# rather than redefined here -- complexity.py is where they actually
# live; re-exported from this module too since C++ callers of
# composite_score reach them via types.h in the same breath (complexity.h
# is one of types.h's own includes).

# `std::numeric_limits<score_t>::lowest()`/`::max()` for IEEE-754 binary32,
# computed bit-exactly from the well-known 0x7F7FFFFF FLT_MAX pattern via
# `struct` (stdlib-only, same mechanism util/rng.py's randfloat() note
# describes) rather than spelled as an approximate decimal literal --
# deliberately NOT `float("-inf")`/`float("inf")`: `very_worst_score` is
# compared and subtracted from in composite_score arithmetic
# (update_penalized_score, is_approx_eq), and `inf - inf` is `nan` in
# IEEE-754, which would make two independently-constructed "worst"
# composite_scores compare UNEQUAL under is_approx_eq (nan is never
# `< epsilon`, and `nan <= anything` is also false) -- caught by this
# port's own tests: `worst_composite_score == composite_score()` must
# hold, and only does with a finite sentinel.
_FLT_MAX: score_t = struct.unpack(">f", bytes.fromhex("7f7fffff"))[0]
very_best_score: score_t = _FLT_MAX  # C++: moses/moses/moses/types.h:69 very_best_score
very_worst_score: score_t = -_FLT_MAX  # C++: moses/moses/moses/types.h:70 very_worst_score
epsilon_score: score_t = 2**-23  # C++: moses/moses/moses/types.h:73 epsilon_score
# `FLT_EPSILON` = 2^-23 for IEEE-754 binary32 (the gap between 1.0 and the
# next representable float32) -- a literal constant, not a computed one,
# so no float32 narrowing is needed to reproduce it exactly.

_EQ_EPSILON: score_t = 1e-6  # C++: moses/moses/moses/types.cc:161 EPSILON
# (the `#define EPSILON static_cast<score_t>(1e-6)` local to
# composite_score::operator==, types.cc:161)


def _is_approx_eq(x: float, y: float, epsilon: float) -> bool:
    """C++: opencog/util/numeric.h:185 opencog::is_approx_eq

    "Compare 2 FloatT with precision epsilon ... note that, unlike
    isWithin, the precision adapts with the scale of x and y" (the
    header's own doc comment) -- an absolute check (`|x-y| < epsilon`)
    OR, failing that, a scale-adaptive relative one
    (`|x-y| <= epsilon * |x+y|`). Not itself one of this task's ported
    symbols (composite_score::operator== is), but cogutil's own helper
    that operator== calls -- ported alongside it since types.cc has no
    Python analogue to import it from (opencog/util/numeric.h is a
    CogUtil header, out of this cycle's ported surface).
    """
    diff = abs(x - y)
    if diff < epsilon:
        return True
    return diff <= epsilon * abs(x + y)


class composite_score:
    """C++: moses/moses/moses/types.h:77 composite_score

    `boost::less_than_comparable<composite_score>` +
    `boost::equality_comparable<composite_score>` (types.h:78-79) are CRTP
    mixins that generate >, <=, >=, != from operator< and operator== --
    ported the same way argument.py's `__lt__`/`__eq__` are, via
    `functools.total_ordering` (see its own docstring there for why that
    is the Python equivalent).

    "By convention, we expect score to be negative (so that higher
    scores==better scores) while cpxy and penalty are both positive ...
    The penalty is *SUBTRACTED* from the score during evaluation!"
    (types.h:82-86's own comment, reproduced verbatim -- this sign
    convention is exactly what ctruth_table_bscore.py's negated-count
    scoring and get_penalized_score()'s subtraction below both rely on).
    """

    def __init__(
        self,
        score: score_t = very_worst_score,
        complexity: complexity_t = least_complexity,
        complexity_penalty: score_t = 0.0,
        uniformity_penalty: score_t = 0.0,
    ) -> None:
        """C++: moses/moses/moses/types.h:89 composite_score

        Covers both the 4-argument constructor (types.h:89-99, which
        calls `update_penalized_score()` at the end of its body) and the
        empty/default constructor (types.h:101, "build the worst score",
        defined at types.cc:114-118: `score(very_worst_score),
        complexity(least_complexity), complexity_penalty(0.0),
        uniformity_penalty(0.0), penalized_score(very_worst_score)`, which
        sets `penalized_score` directly rather than computing it) --
        merged into one Python method the same way `ITable.__init__`
        (table.py) merges its own two C++ overloads, since the default
        ctor's field values are exactly this method's own defaults, and
        `very_worst_score - 0.0 - 0.0 == very_worst_score` makes calling
        `update_penalized_score()` here equivalent to types.cc's direct
        assignment.
        """
        self.multiply_diversity = False  # C++: types.h:145 multiply_diversity
        self.score = score
        self.complexity = complexity
        self.complexity_penalty = complexity_penalty
        self.uniformity_penalty = uniformity_penalty
        self.penalized_score: score_t = 0.0
        self.update_penalized_score()

    def get_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:104 composite_score::get_score"""
        return self.score

    def get_complexity(self) -> complexity_t:
        """C++: moses/moses/moses/types.h:105 composite_score::get_complexity"""
        return self.complexity

    def get_penalized_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:106 composite_score::get_penalized_score"""
        return self.penalized_score

    def set_score(self, sc: score_t) -> None:
        """C++: moses/moses/moses/types.h:109 composite_score::set_score

        "Use this only to over-ride the score, when re-scoring."
        (types.h:108's own comment.)
        """
        self.score = sc
        self.update_penalized_score()

    def get_complexity_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:118 composite_score::get_complexity_penalty"""
        return self.complexity_penalty

    def set_complexity_penalty(self, penalty: score_t) -> None:
        """C++: moses/moses/moses/types.h:119 composite_score::set_complexity_penalty"""
        self.complexity_penalty = penalty
        self.update_penalized_score()

    def get_uniformity_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:124 composite_score::get_uniformity_penalty"""
        return self.uniformity_penalty

    def set_uniformity_penalty(self, penalty: score_t) -> None:
        """C++: moses/moses/moses/types.h:125 composite_score::set_uniformity_penalty"""
        self.uniformity_penalty = penalty
        self.update_penalized_score()

    def get_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:129 composite_score::get_penalty"""
        return self.complexity_penalty + self.uniformity_penalty

    def update_penalized_score(self) -> None:
        """C++: moses/moses/moses/types.h:135 composite_score::update_penalized_score

        Protected in C++ (called only from within the class itself, on
        every field mutation above); kept as a plain, un-prefixed method
        here since the identifier itself -- not its C++ access
        specifier, which Python has no equivalent of -- is what the
        project's naming convention preserves verbatim.
        """
        self.penalized_score = self.score - self.complexity_penalty
        if self.multiply_diversity:
            self.penalized_score *= self.uniformity_penalty
        else:
            self.penalized_score -= self.uniformity_penalty

    def __lt__(self, other: composite_score) -> bool:
        """C++: moses/moses/moses/types.h:140 composite_score

        Non-identifier member per the project's convention (see
        table.py's `ITable.__eq__`/`Table.__eq__` docstrings for the same
        pattern): cites the enclosing class at the member's own line
        (types.h:140 is the `bool operator<(const composite_score &r)
        const;` declaration itself). The real member is
        `composite_score::operator<`, defined at types.cc:131-149:
        "Compare penalized scores ... If the 2 score-penalty are equal
        then we compare by complexity (decreasing order, as complexity is
        positive) ... nan is always smaller than anything (including
        -inf) except nan" (types.h:132-139's own doc comment).
        """
        lef = self.penalized_score
        rig = other.penalized_score
        if math.isnan(lef):
            return not math.isnan(rig)
        return (lef < rig) or (lef == rig and self.complexity > other.complexity)

    def __eq__(self, other: object) -> bool:
        """C++: moses/moses/moses/types.h:143 composite_score

        Non-identifier member, same convention as `__lt__` above (types.h:
        143 is the `bool operator==(const composite_score& r) const;`
        declaration). The real member is `composite_score::operator==`,
        defined at types.cc:152-166 -- "Check for equality, to within
        floating-point error ... used in test cases -- compare equality
        to 7 decimal places" (types.h:142, types.cc:151's own comments).
        See the module-level DIVERGENCE note above for why that tolerance
        is exactly what makes this port's double-precision score_t safe
        to compare against classic's float32 one.
        """
        if not isinstance(other, composite_score):
            return NotImplemented
        return (
            _is_approx_eq(self.score, other.score, _EQ_EPSILON)
            and self.complexity == other.complexity
            and _is_approx_eq(self.complexity_penalty, other.complexity_penalty, _EQ_EPSILON)
            and _is_approx_eq(self.uniformity_penalty, other.uniformity_penalty, _EQ_EPSILON)
            and _is_approx_eq(self.penalized_score, other.penalized_score, _EQ_EPSILON)
        )

    __hash__ = None  # type: ignore[assignment]


composite_score = functools.total_ordering(composite_score)

# C++: moses/moses/moses/types.h:212 behavioral_score
#
# `public std::vector<score_t>`, narrowed to a bare `list[float]` per
# task-6-brief.md's Interfaces section. The two ported operators
# (`operator-=`, types.h:220-231, and the free `operator-`, types.h:234-
# 246 -- both OC_ASSERT-checked elementwise subtraction, used by
# ctruth_table_bscore's `bs -= _best_possible_score` when boosting) have
# no separate Python identifier to attach a breadcrumb to, since a bare
# `list` cannot be subclassed-and-then-narrowed the way `ITable`/`OTable`
# (table.py) narrow `std::vector`: `bscores.py`'s ctruth_table_bscore does
# the equivalent elementwise subtraction inline with a list comprehension
# instead of via `-=`. NOT ported: `behavioral_score(size_t)`/
# `behavioral_score(size_t, score_t)` sized constructors (types.h:215-216)
# -- no caller in this task's scope constructs one that way; every
# behavioral_score here is built by appending.
behavioral_score = list[score_t]


class scored_combo_tree:
    """C++: moses/moses/moses/types.h:259 scored_combo_tree

    "A single combo tree, together with various score metrics for it."
    (types.h:247's doc comment, reproduced verbatim.) Only the fields
    and accessors task-6-brief.md's interface actually needs are ported:
    the tree, its composite score, its behavioral score. NOT ported:
    `demeID_t`/`get_demeID()` (deme-bookkeeping, out of scope -- see
    module docstring) and `get_weight()`/`set_weight()` (the boosting
    weight, `_weight` -- boosting is out of scope for this whole task,
    same narrowing bscores.py's ctruth_table_bscore makes for the
    ensemble `operator()` overload).
    """

    __slots__ = ("bscore", "cscore", "tree")

    def __init__(
        self,
        tr: object,
        cs: composite_score | None = None,
        bs: behavioral_score | None = None,
    ) -> None:
        """C++: moses/moses/moses/types.h:262 scored_combo_tree

        `tr` is a `combo_tree` (comboreduct/combo/vertex.py); typed
        `object` here rather than imported, so this module -- itself
        imported by combo_tree's own consumers (bscores.py,
        behave_cscore.py) -- carries no import-cycle risk. `demeID_t`
        (types.h:263's `id` parameter) is dropped -- see class docstring.
        """
        self.tree = tr
        self.cscore = cs if cs is not None else composite_score()
        self.bscore: behavioral_score = bs if bs is not None else []

    def get_tree(self) -> object:
        """C++: moses/moses/moses/types.h:277 scored_combo_tree::get_tree"""
        return self.tree

    def get_bscore(self) -> behavioral_score:
        """C++: moses/moses/moses/types.h:283 scored_combo_tree::get_bscore"""
        return self.bscore

    def set_bscore(self, bs: behavioral_score) -> None:
        """C++: moses/moses/moses/types.h:287 scored_combo_tree::set_bscore"""
        self.bscore = bs

    def get_composite_score(self) -> composite_score:
        """C++: moses/moses/moses/types.h:299 scored_combo_tree::get_composite_score"""
        return self.cscore

    def get_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:309 scored_combo_tree::get_score"""
        return self.cscore.get_score()

    def get_complexity(self) -> complexity_t:
        """C++: moses/moses/moses/types.h:310 scored_combo_tree::get_complexity"""
        return self.cscore.get_complexity()

    def get_penalized_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:311 scored_combo_tree::get_penalized_score"""
        return self.cscore.get_penalized_score()

    def __eq__(self, other: object) -> bool:
        """C++: moses/moses/moses/types.cc:52 scored_combo_tree

        Non-identifier member per the project's convention (see
        `composite_score.__lt__` above): the real member is
        `scored_combo_tree::operator==` (types.cc:52-58). NARROWED versus
        C++: the real operator also compares `get_demeID()` and
        `get_weight()` (types.cc:54, 56) -- both dropped fields, see class
        docstring -- so this compares only tree, bscore and cscore.
        """
        if not isinstance(other, scored_combo_tree):
            return NotImplemented
        return (
            self.tree == other.tree
            and self.bscore == other.bscore
            and self.cscore == other.cscore
        )

    __hash__ = None  # type: ignore[assignment]


worst_composite_score = composite_score()
# C++: moses/moses/moses/types.cc:112 worst_composite_score
#
# "the empty composite_score ctor returns the worst composite score"
# (types.cc:111's own comment) -- a module-level constant, not a ported
# function/class, so (like `combo_tree` in vertex.py) it gets a plain
# comment rather than a docstring breadcrumb.
