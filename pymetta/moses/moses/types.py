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
built-in floating-point type, so every score_t-typed value here is held
as a plain Python `float` (64-bit, i.e. what C++ calls `double`) -- BUT,
unlike an ordinary double, it is explicitly narrowed to the nearest
IEEE-754 binary32 value, via `_to_score_t()` below, at every point
classic's own C++ performs an implicit or explicit narrowing conversion
to `score_t`. This is not optional: MOSES's exemplar selection depends on
`composite_score::operator<`'s tie-break-by-complexity branch
(`lef == rig and complexity > ...`), which is reached far more often
under float32 (many double-precision gaps collapse to the same float32
value) than under plain double arithmetic. A fix-round review demonstrated
this concretely: brute-forcing integer scores 0..-400 against complexity
1..120 at MOSES's default `complexity_ratio` of 3.5 produced 20,656 pairs
where naive double arithmetic and true float32 arithmetic disagreed on
which of two candidates ties -- each one a potential exemplar-selection
inversion. Every one of the nine sites below was traced directly against
the C++ source, not inferred:

* `bscores.cc:285` -- each `ctruth_table_bscore` bscore element (`bscores.py`).
* `bscores.cc:369` -- `-score_t(fmin(...))` in `set_best_possible_bscore`.
* `bscores.cc:399,419` -- `score_t w = fabs(...)` in `worst_possible_bscore`
  and `min_improv` (and `bscores.cc:423`, `min_improv`'s own
  `return 0.5 * min_weight` -- a necessary consequence of `min_improv`
  itself returning `score_t`, not separately traced by the review but
  required for the same reason).
* `scoring_base.cc:139` -- `sum_bscore`: accumulated in DOUBLE (see
  scoring_base.py's own note on why `sum()` itself is a separate,
  compensated-summation problem), narrowed ONCE at the return, not per
  term.
* `scoring_base.cc:41,56,67` -- `_complexity_coef`, in both
  `set_complexity_coef` overloads (and the `complexity_ratio`/`p`
  parameters themselves, since they are `score_t`/`float`-typed C++
  parameters and so are narrowed at the call boundary, BEFORE the
  reciprocal/`discrete_complexity_coef` call that uses them).
* `behave_cscore.cc:97` -- `cpxy * cpxy_coef`, before it reaches
  `composite_score`'s constructor.
* `types.h:166` -- `update_penalized_score`: TWO separate narrowings, one
  after `score - complexity_penalty`, another after the following
  `*=`/`-=` against `uniformity_penalty`.
* `types.h:89-91,107,116,122` -- `composite_score`'s own constructor and
  `set_score`/`set_complexity_penalty`/`set_uniformity_penalty`, an
  eleventh (twelfth, counting the `get_penalty` sum below) site found in
  fix round 2: every one of these binds a `score_t`-typed C++ parameter
  without narrowing it Python-side. Latent until now (every call site in
  this port already passed pre-narrowed values), but S7+ (metapopulation/
  merging/deme expansion) will construct `composite_score` directly.
* `types.h:129` -- `get_penalty`'s `complexity_penalty + uniformity_
  penalty`: both operands are already `score_t`, but their SUM is its own
  float32 operation in C++, not a double-precision one narrowed on the
  way in.
* `types.cc:158` -- `EPSILON`, the local `#define` `composite_score::
  operator==` compares against (`_EQ_EPSILON` below), and every
  arithmetic step inside `is_approx_eq<FloatT=score_t>` itself
  ($COGUTIL/opencog/util/numeric.h:185) that it feeds -- the thirteenth
  and fourteenth sites, found in the final fix wave. `_EQ_EPSILON` was a
  bare Python double (`1e-6`, not `static_cast<score_t>(1e-6)` =
  9.999999974752427e-07), and `_is_approx_eq` computed `x - y`/`x + y`/
  `epsilon * amp` in double where C++'s `is_approx_eq<score_t>` computes
  every one of those in float32. Impact is confined to
  `composite_score.__eq__`, but that is exactly the operator the
  metapopulation uses.

`count_t` (`CTable`'s counters, table.py) and `complexity_t` (an
`unsigned`/Python `int`, complexity.py) are unaffected -- neither is
`score_t`-typed in C++, so neither is ever narrowed.

# DIVERGENCE(C++ moses/moses/moses/types.h:63): `score_t` is C++ `float`
# (IEEE-754 binary32). Python has no native 32-bit float type, so every
# score_t-typed value in this port is held as a plain Python `float`
# (double) and is explicitly re-narrowed to the nearest float32 value via
# `_to_score_t()` at every one of the fourteen sites enumerated above --
# this branch's single largest behavioural deviation from C++. Skipping
# even one narrowing reintroduces double-precision comparisons where C++
# has float32 ones, which changes how often `composite_score.__lt__`'s
# tie-break-by-complexity branch fires and, with it, which candidate
# exemplar selection prefers -- see the 20,656-pair discrepancy
# demonstrated above.
"""

import math
import struct

from pymetta.moses.moses.complexity import complexity_t, least_complexity

score_t = float  # C++: moses/moses/moses/types.h:63 score_t


def _to_score_t(x: float) -> score_t:
    """Not itself a ported C++ symbol -- the narrowing conversion to
    `score_t` (IEEE-754 binary32) that C++ performs implicitly at every
    assignment/parameter-binding site the module docstring's numbered
    list names, and that this port must perform explicitly since Python
    has no native 32-bit float type. Round-trips `x` through `struct`'s
    4-byte `f` format (stdlib-only, same mechanism util/rng.py's
    `randfloat()` note describes), which rounds to the nearest
    representable float32 value exactly the way a C++ `double`-to-`float`
    narrowing conversion does, then widens back to a plain Python `float`
    (double) so ordinary arithmetic continues to work on the result.

    A finite `x` outside float32's representable range raises
    `OverflowError` from `struct.pack` -- narrowed here to signed
    infinity, matching what an actual C++ narrowing conversion of an
    out-of-range double to `float` does (freely permitted, unlike the
    reverse direction, by both the IEEE-754 standard and the C++
    standard's own narrowing-conversion rules for floating point).
    `nan`/`+-inf` pack and unpack without error and are returned as-is
    (mod the smaller float32 mantissa, immaterial for a nan/inf).
    """
    try:
        return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf

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
very_best_score: score_t = _FLT_MAX  # C++: moses/moses/moses/types.h:71 very_best_score
very_worst_score: score_t = -_FLT_MAX  # C++: moses/moses/moses/types.h:72 very_worst_score
epsilon_score: score_t = 2**-23  # C++: moses/moses/moses/types.h:74 epsilon_score
# `FLT_EPSILON` = 2^-23 for IEEE-754 binary32 (the gap between 1.0 and the
# next representable float32) -- a literal constant, not a computed one,
# so no float32 narrowing is needed to reproduce it exactly.

# FIX WAVE (final review, Important): `types.cc:158`'s
# `#define EPSILON static_cast<score_t>(1e-6)` is a C++ `float` literal,
# not a `double` one -- `static_cast<score_t>(1e-6)` rounds to
# 9.999999974752427e-07 in float32, a different bit pattern than the
# double `1e-6` this constant held before. Narrowed via `_to_score_t()`,
# the same narrowing convention every other site in this module uses.
_EQ_EPSILON: score_t = _to_score_t(1e-6)  # C++: moses/moses/moses/types.cc:158 EPSILON
# (the `#define EPSILON static_cast<score_t>(1e-6)` local to
# composite_score::operator==, types.cc:158)


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

    FIX WAVE (final review, Important): cogutil's template is instantiated
    here as `is_approx_eq<FloatT=score_t>` (`composite_score.__eq__` only
    ever calls this with `score_t` operands) -- every arithmetic step
    (`x - y`, `x + y`, `epsilon * amp`) is therefore a float32 operation in
    C++, computed and rounded in float32 at each step, not accumulated in
    double the way plain Python arithmetic on these `float` operands would.
    Each intermediate is narrowed via `_to_score_t()` immediately after
    the operation that produces it (`abs()` itself loses no precision --
    it only clears the sign bit -- so it needs no narrowing of its own).
    """
    diff = abs(_to_score_t(x - y))
    if diff < epsilon:
        return True
    amp = abs(_to_score_t(x + y))
    return diff <= _to_score_t(epsilon * amp)


class composite_score:
    """C++: moses/moses/moses/types.h:77 composite_score

    `boost::less_than_comparable<composite_score>` (types.h:78) is a CRTP
    mixin that generates `>`, `<=`, `>=` PURELY from `operator<` --
    `a>b := b<a`, `a<=b := !(b<a)`, `a>=b := !(a<b)` -- deliberately NOT
    `functools.total_ordering` (contrast argument.py's `__lt__`/`__eq__`,
    which legitimately uses it): `total_ordering` derives its comparisons
    from whichever of `__lt__`/`__eq__` it is missing, and would here
    derive `>`/`<=` through THIS class's `__eq__` -- which is the
    intentionally TOLERANT `is_approx_eq` comparison below, not a bit-exact
    one. That substitution is a real, demonstrated divergence from boost's
    semantics: for `composite_score(-10.0, 5)` vs
    `composite_score(-10.0000005, 5)` (a difference `is_approx_eq` treats
    as equal but `<` does not), `total_ordering` gives `a>b == False` and
    `a<=b == True`, while boost's CRTP -- and this class's own explicit
    `__gt__`/`__le__`/`__ge__` below -- give `True`/`False`. `!=` is left
    to Python's own default (`object.__ne__`, which inverts `__eq__`
    automatically) -- `boost::equality_comparable<composite_score>`
    (types.h:79) derives `!=` from `==` the exact same way.

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

        Covers both the 4-argument constructor (types.h:89-97, which
        calls `update_penalized_score()` at the end of its body) and the
        empty/default constructor (types.h:99, "build the worst score",
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

        `score`/`complexity_penalty`/`uniformity_penalty` are narrowed to
        `score_t` here too (fix round 2's "eleventh site"): C++'s 4-arg
        constructor takes them as `score_t scor, ..., score_t
        complexity_penalty_ = 0.0, score_t uniformity_penalty_ = 0.0`
        (types.h:89-91), so a caller passing a plain double narrows AT
        this binding, before `update_penalized_score()` ever runs -- this
        was latent (every call site in this port already passed
        pre-narrowed values) but S7+ (metapopulation/merging/deme
        expansion) will construct `composite_score` directly, and an
        unnarrowed constructor is exactly the shape of bug a full review
        cycle was needed to find once already (see the module docstring's
        numbered list).
        """
        self.multiply_diversity = False  # C++: types.h:155 multiply_diversity
        self.score = _to_score_t(score)
        self.complexity = complexity
        self.complexity_penalty = _to_score_t(complexity_penalty)
        self.uniformity_penalty = _to_score_t(uniformity_penalty)
        self.penalized_score: score_t = 0.0
        self.update_penalized_score()

    def get_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:102 composite_score::get_score"""
        return self.score

    def get_complexity(self) -> complexity_t:
        """C++: moses/moses/moses/types.h:103 composite_score::get_complexity"""
        return self.complexity

    def get_penalized_score(self) -> score_t:
        """C++: moses/moses/moses/types.h:104 composite_score::get_penalized_score"""
        return self.penalized_score

    def set_score(self, sc: score_t) -> None:
        """C++: moses/moses/moses/types.h:107 composite_score::set_score

        "Use this only to over-ride the score, when re-scoring."
        (types.h:106's own comment.) `sc` is `score_t`-typed in C++
        (`void set_score(score_t sc)`) -- narrowed here for the same
        "eleventh site" reason `__init__` above is.
        """
        self.score = _to_score_t(sc)
        self.update_penalized_score()

    def get_complexity_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:115 composite_score::get_complexity_penalty"""
        return self.complexity_penalty

    def set_complexity_penalty(self, penalty: score_t) -> None:
        """C++: moses/moses/moses/types.h:116 composite_score::set_complexity_penalty

        `penalty` is `score_t`-typed in C++ -- narrowed here, same reason
        as `set_score` above.
        """
        self.complexity_penalty = _to_score_t(penalty)
        self.update_penalized_score()

    def get_uniformity_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:121 composite_score::get_uniformity_penalty"""
        return self.uniformity_penalty

    def set_uniformity_penalty(self, penalty: score_t) -> None:
        """C++: moses/moses/moses/types.h:122 composite_score::set_uniformity_penalty

        `penalty` is `score_t`-typed in C++ -- narrowed here, same reason
        as `set_score` above.
        """
        self.uniformity_penalty = _to_score_t(penalty)
        self.update_penalized_score()

    def get_penalty(self) -> score_t:
        """C++: moses/moses/moses/types.h:127 composite_score::get_penalty

        `complexity_penalty + uniformity_penalty` (types.h:129) is itself
        `score_t + score_t` in C++ -- both operands are already `float`,
        so C++ computes this SUM natively in float32, not promoted to
        double. Even though both operands here are already narrowed
        Python-side, their double-precision SUM is not guaranteed to
        equal the float32 sum of the same two float32 values (this is the
        same "apply your own rule to every score_t lvalue/parameter/
        return" principle the fix-round-2 review asked to sweep for), so
        the result is narrowed too.
        """
        return _to_score_t(self.complexity_penalty + self.uniformity_penalty)

    def update_penalized_score(self) -> None:
        """C++: moses/moses/moses/types.h:166 composite_score::update_penalized_score

        Protected in C++ (called only from within the class itself, on
        every field mutation above); kept as a plain, un-prefixed method
        here since the identifier itself -- not its C++ access
        specifier, which Python has no equivalent of -- is what the
        project's naming convention preserves verbatim.

        TWO separate `score_t` narrowings, matching C++'s two separate
        float32 operations exactly (module docstring's numbered list,
        `types.h:166` entry): the subtraction is one `score_t` value in
        C++ (computed and stored before the `if`), and the following
        `*=`/`-=` against `uniformity_penalty` is a SECOND, independent
        float32 operation on top of that already-narrowed intermediate --
        not a single combined double-precision expression narrowed once
        at the end.
        """
        self.penalized_score = _to_score_t(self.score - self.complexity_penalty)
        if self.multiply_diversity:
            self.penalized_score = _to_score_t(self.penalized_score * self.uniformity_penalty)
        else:
            self.penalized_score = _to_score_t(self.penalized_score - self.uniformity_penalty)

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
        defined at types.cc:152-164 -- "Check for equality, to within
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

    def __gt__(self, other: composite_score) -> bool:
        """C++: moses/moses/moses/types.h:78 composite_score

        `boost::less_than_comparable<composite_score>`'s CRTP-generated
        `operator>` -- `a>b := b<a` -- cited at the mixin's own line (see
        class docstring for why this is spelled out explicitly rather
        than left to `functools.total_ordering`).
        """
        return other < self

    def __le__(self, other: composite_score) -> bool:
        """C++: moses/moses/moses/types.h:78 composite_score

        `boost::less_than_comparable<composite_score>`'s CRTP-generated
        `operator<=` -- `a<=b := !(b<a)`.
        """
        return not (other < self)

    def __ge__(self, other: composite_score) -> bool:
        """C++: moses/moses/moses/types.h:78 composite_score

        `boost::less_than_comparable<composite_score>`'s CRTP-generated
        `operator>=` -- `a>=b := !(a<b)`.
        """
        return not (self < other)

    __hash__ = None  # type: ignore[assignment]

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
