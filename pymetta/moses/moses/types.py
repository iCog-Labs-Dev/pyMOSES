"""C++: moses/moses/moses/types.h and types.cc

`composite_score` (types.h:77), `demeID_t` (types.h:190), `behavioral_score`
(types.h:212) and `scored_combo_tree` (types.h:259) were narrowed, by the
S0-S6 foundation cycle, to exactly the surface task-6-brief.md's Interfaces
section asked for -- deliberately leaving out the metapopulation container
plumbing below. S10 Task 0 ports exactly that deferred plumbing:

* `scored_combo_tree_set`/`_tset`/`_ptr_set` and their comparator/hash/
  equality functors (`sct_score_greater`, `sct_tree_greater`,
  `scored_combo_tree_hash`, `scored_combo_tree_equal`). `bscore_base::
  operator()(const scored_combo_tree_set&)` (scoring_base.h), whose port
  was blocked on this set type not existing yet, is ported too -- see
  scoring_base.py's own module docstring.
* `demeID_t` MOVES here. It was already ported, by S9 Task 0, but had to
  live in `pymetta/moses/representation/instance_set.py` instead of its
  real C++ home (types.h:190) because a concurrent S9 task owned this file
  at the time. S10 Task 0 moves the class itself here verbatim and updates
  every importer, leaving exactly one definition.

Still out of scope for this whole S0-S6+S10 lineage:

* The `ostream_*`/`istream_*` free functions at the bottom of types.h --
  no stream-printing surface is in this cycle's scope (same narrowing
  table.py's module docstring makes for `Table`).
* `scored_combo_tree`'s `demeID_t`-typed `_deme_id` field and its
  `get_demeID()` accessor -- `demeID_t` the TYPE now lives in this module,
  but `scored_combo_tree` itself still does not carry one: deme
  bookkeeping is out of scope for this whole cycle (see
  `scored_combo_tree`'s own class docstring for specifics).

  CORRECTED: this bullet used to list `get_weight()`/`set_weight()` (the
  boosting weight) alongside `demeID_t`. They are ported now -- see
  `scored_combo_tree.get_weight` below -- because `ensemble.py`'s
  AdaBoost/experts engine has since landed and had to keep every member's
  weight in a private side dict keyed by object identity for want of the
  field. Porting the field does NOT by itself rewire `ensemble.py`; that
  is recorded as a follow-up in `get_weight`'s own docstring.

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
from collections.abc import Sequence

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, output_format
from pymetta.comboreduct.combo.vertex import builtin, lexicographic_subtree_less, vertex_equal
from pymetta.moses.moses.complexity import complexity_t, least_complexity

score_t = float  # C++: moses/moses/moses/types.h:63 score_t

# C++: moses/moses/moses/types.h:69 io_score_precision
#
# "A number of external tools read the printed results, and parse them.
# The floating-point scores must be printed with reasonable accuracy, else
# these tools fail." (moses_main.h:107-109's own comment, its one call
# site: `ss << std::setprecision(opencog::io_score_precision);` -- applies
# to the WHOLE `metapop_printer` output stream, not merely composite_score
# printing, which is why every `_format_score_t` call below shares this
# one constant rather than each hardcoding "18".
io_score_precision = 18


def _format_score_t(x: float) -> str:
    """Not itself a ported C++ symbol -- factors out the string rendering
    every `operator<<`/`ostream_*` function below performs on a raw
    `score_t`/`double` once `std::setprecision(io_score_precision)`
    (defaultfloat: no `std::fixed`/`std::scientific`) is in effect on the
    enclosing stream.

    Verified equivalent to Python's `"{:.18g}"` format spec against a real
    compiled C++ probe (`std::cout << std::setprecision(18) << v`, this
    project's own toolchain) over: integral values (-3.0 -> "-3"),
    negative zero ("-0"), a value needing all 18 significant digits after
    a float32 round-trip (-3.5714285373687744 -> "-3.57142853736877441",
    matching `MOSESUTest`'s own captured stdout for `test_it`'s penalized
    score), scientific notation at both magnitude extremes (1e20, 1e-20),
    and nan/+-inf (all three print identically -- "nan"/"inf"/"-inf" -- in
    both languages on this toolchain). See task-3-report.md for the full
    probe transcript.

    `x` is expected to already be narrowed to its exact float32 value via
    `_to_score_t` (this module) at the point it was last computed/stored;
    this function does not re-narrow -- matching C++, where `operator<<`
    never narrows the `float` it is handed (narrowing, if any, already
    happened at the last `score_t` assignment).
    """
    return format(x, f".{io_score_precision}g")


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

    def __str__(self) -> str:
        """C++: moses/moses/moses/types.h:427 composite_score

        Non-identifier member per the project's convention (see
        `__lt__`/`__eq__` above, this same class): cites the enclosing
        class at the free `operator<<`'s own declaration line. The real
        member is the free function `operator<<(ostream&, const
        composite_score&)` (types.h:427-436):

            return out << "[score="
                       << std::setprecision(moses::io_score_precision)
                       << ts.get_score()
                       << ", penalized score=" << ts.get_penalized_score()
                       << ", complexity=" << ts.get_complexity()
                       << ", complexity penalty=" << ts.get_complexity_penalty()
                       << ", uniformity penalty=" << ts.get_uniformity_penalty()
                       << "]";

        Ported as `__str__` (Python's `operator<<`-for-printing
        equivalent, per this project's convention -- see `demeID_t`'s own
        class docstring for the analogous `std::string`-subclass choice)
        rather than a free function, since -- unlike `ostream_scored_
        combo_tree` below -- C++'s version here is ALREADY a one-tree
        recursive-free leaf formatter with no extra parameters (labels,
        output_format, ...) that would make a bound method awkward. Note
        `std::setprecision` is set only ONCE, before `get_score()`, but a
        `std::ios`' precision is sticky for the whole stream -- every
        later `<<` on this same `out` (penalized score, complexity
        penalty, uniformity penalty) is affected too, not just the first
        value; `_format_score_t` is applied to all four `score_t` fields
        for that reason. `complexity` is an `int` (`complexity_t`), never
        `score_t`-formatted, matching `<<`'s own overload resolution on a
        plain integer.
        """
        return (
            f"[score={_format_score_t(self.score)}"
            f", penalized score={_format_score_t(self.penalized_score)}"
            f", complexity={self.complexity}"
            f", complexity penalty={_format_score_t(self.complexity_penalty)}"
            f", uniformity penalty={_format_score_t(self.uniformity_penalty)}"
            f"]"
        )


class demeID_t(str):
    """C++: moses/moses/moses/types.h:190 demeID_t

    MOVED here (S10 Task 0) from `pymetta/moses/representation/
    instance_set.py`, where it was ported, verbatim, by S9 Task 0 -- that
    task owned `pymetta/moses/representation/` while a concurrent S9 task
    owned this file, and `instance_set` could not be constructed at all
    without *some* `demeID_t` to default to, so it was ported inline there
    rather than left undone. Nothing about the class changed in the move;
    every importer was updated to reach it here instead (see
    `instance_set.py`'s own module docstring for what it now imports).

    "Assign a unique ID for each deme. This is used to keep track of which
    deme each candidate comes from. The demeID is stored as a string,
    formatted as either a single a single integer EXPANSION or as a pair
    EXPANSION.BREADTH_FIRST ... Here, EXPANSION is the number of times that
    the deme expander has been called thus far. BREADTH_FIRST is the index
    of the deme created by a single call of the deme expander." (types.h:
    182-189's own comment, reproduced verbatim.) `struct demeID_t : public
    std::string` with three numbered constructors (types.h:192-194) --
    Python subclasses `str` the same way `instance.py` subclasses `list`
    for `std::vector`.
    """

    def __new__(
        cls,
        expansion: int = 0,
        breadth_first: int | None = None,
        ss_deme: int | None = None,
    ) -> demeID_t:
        """C++: moses/moses/moses/types.cc:43 demeID_t::demeID_t

        Three overloads by trailing-argument presence, matching
        types.cc:43-50 exactly: `str(expansion)`,
        `f"{expansion}.{breadth_first}"`, or
        `f"{expansion}.{breadth_first}.SS-{ss_deme}"`. `demeID_t()` (the
        default used throughout the codebase) is `demeID_t(0)` == `"0"`,
        not the empty string.
        """
        if breadth_first is None:
            if ss_deme is not None:
                raise TypeError("demeID_t requires breadth_first before ss_deme")
            text = str(expansion)
        elif ss_deme is None:
            text = f"{expansion}.{breadth_first}"
        else:
            text = f"{expansion}.{breadth_first}.SS-{ss_deme}"
        return super().__new__(cls, text)


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
    (types.h:247's doc comment, reproduced verbatim.) The class's own
    doc comment lists exactly what it holds: "a composite score ..., a
    behavioral score ..., a boosting weight (used to implement the
    boosting algorithm)" (types.h:253-258). All three are ported, plus
    the tree. NOT ported: `demeID_t`/`get_demeID()` (deme-bookkeeping,
    out of scope -- see module docstring).

    `weight` corresponds to C++'s `double _weight` (types.h:274). The
    port's other three fields already drop C++'s leading underscore
    (`_tree`/`_cscore`/`_bscore` -> `tree`/`cscore`/`bscore`, since
    Python has no private-with-public-accessor idiom to reproduce), and
    this one follows that established local convention.
    """

    __slots__ = ("bscore", "cscore", "tree", "weight")

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

        `weight` is NOT a constructor parameter in either language: the
        C++ member-initializer list ends `_weight(1.0)` (types.h:266), a
        fixed initial value with no way to set it at construction. Only
        `set_weight` changes it.

        `_weight` is declared `double` (types.h:274), NOT `score_t` --
        so it is deliberately not routed through `_to_score_t`. Every
        C++ reader of it either keeps it as `double` (`scoring_base.cc:
        103`) or narrows at the point of use, in its own expression
        (`score_t trweight = sct.get_weight();`, select_bscore.cc:198,
        precision_bscore.cc:460; `score_t weight = sct.get_weight();`,
        bscores.cc:92,314). Narrowing it here would narrow it twice for
        those readers and once too often for `scoring_base.cc:103`.
        """
        self.tree = tr
        self.cscore = cs if cs is not None else composite_score()
        self.bscore: behavioral_score = bs if bs is not None else []
        self.weight: float = 1.0

    def get_tree(self) -> object:
        """C++: moses/moses/moses/types.h:277 scored_combo_tree::get_tree"""
        return self.tree

    def get_bscore(self) -> behavioral_score:
        """C++: moses/moses/moses/types.h:283 scored_combo_tree::get_bscore"""
        return self.bscore

    def set_bscore(self, bs: behavioral_score) -> None:
        """C++: moses/moses/moses/types.h:287 scored_combo_tree::set_bscore"""
        self.bscore = bs

    def get_weight(self) -> float:
        """C++: moses/moses/moses/types.h:291 scored_combo_tree::get_weight

        The boosting weight -- "a boosting weight (used to implement the
        boosting algorithm)" (types.h:258). `double`, not `score_t`; see
        `__init__` for why it is not narrowed.

        Read in C++ by every `call_ensemble` implementation
        (bscores.cc:92,314; select_bscore.cc:198; precision_bscore.cc:460),
        by `bscore_base::get_complexity(const scored_combo_tree_set&)`
        (scoring_base.cc:103), by `metapop_printer`'s python-format
        ensemble output (moses_main.h:124) and by `merging.cc:671`'s debug
        log. Written only by `ensemble.cc`'s AdaBoost/experts weight
        updates.

        FOLLOW-UP, deliberately not done in the change that added this
        field: `ensemble.py` still keeps its members' weights in a private
        `_sct_weights` dict keyed by `id(scored_combo_tree)`, because this
        field did not exist when that file was written (its own module
        docstring, "Two REAL prerequisite gaps found", item 1, names this
        exact fix). Adding the field does not rewire it, so an
        `ensemble`'s members still all read back `1.0` here. Everything
        that would print or score from that -- classic's whole
        `output_ensemble` path -- is gated; see `metapop_printer.__call__`
        (moses_main.py) for the gate that exists precisely so this field's
        arrival could not silently turn a raise into wrong output.
        """
        return self.weight

    def set_weight(self, w: float) -> None:
        """C++: moses/moses/moses/types.h:295 scored_combo_tree::set_weight"""
        self.weight = w

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
        C++ by exactly one term: the real operator also compares
        `get_demeID()` (types.cc:54), a field this port does not carry
        (see class docstring).

        `get_weight()` (types.cc:56) IS compared, as of the change that
        added the field. It was listed alongside `demeID_t` here while
        the field did not exist; leaving it out once it does would have
        been a NEW divergence introduced by adding the field, not an
        inherited one. Nothing in this port sets a weight yet, so every
        `scored_combo_tree` compares at the constructor's own 1.0 default
        and this term is inert today -- which is exactly why it needs its
        own test rather than being assumed covered.

        FIX (S10 Task 0): `get_tree() == r.get_tree()` (types.cc:53) is
        real `combo_tree::operator==` -- a genuine STRUCTURAL comparison
        (`opencog/util/tree.h:3011`, see `_combo_tree_structural_eq`
        below). This port's own `tree`/`tree_node` classes
        (`pymetta/util/tree.py`) define no `__eq__` of their own, so a
        bare `self.tree == other.tree` silently fell back to Python's
        default identity comparison (`object.__eq__`) -- `True` only when
        both `scored_combo_tree`s shared the exact same tree OBJECT, never
        for two independently-built-but-structurally-identical trees,
        which real C++ correctly treats as equal. Latent until now: no
        prior task compared two distinct `scored_combo_tree` instances for
        equality, and this task's own `scored_combo_tree_equal` functor
        needed the real structural comparison anyway (see its docstring),
        so this uses that same helper instead of bare `==`.
        """
        if not isinstance(other, scored_combo_tree):
            return NotImplemented
        return (
            _combo_tree_structural_eq(self.tree, other.tree)
            and self.bscore == other.bscore
            and self.weight == other.weight
            and self.cscore == other.cscore
        )

    __hash__ = None  # type: ignore[assignment]


# =======================================================================
# ostream functions (C++: moses/moses/moses/types.h:399-424, types.cc:
# 164-199) -- S13 Task 3 addition (`metapop_printer`, moses_main.py). Out
# of scope for every prior task that touched this module (module
# docstring's own "Still out of scope" section): S13 is the first caller
# that needs these to print search results to stdout.
# =======================================================================

_BEHAVIORAL_SCORE_PREFIX = "behavioral score:"  # C++: types.cc:164 behavioral_score_prefix_str


def ostream_behavioral_score(bs: behavioral_score) -> str:
    """C++: moses/moses/moses/types.cc:166 ostream_behavioral_score

        out << behavioral_score_prefix_str << " ";
        return ostream_container(out, bs, " ", "[", "]");

    `ostream_container` ($COGUTIL/opencog/util/iostreamContainer.h:61-77)
    with its own default `empty_lr=true`: left/right brackets print even
    for an empty `bs` (`"[]"`), space-separated elements, no trailing
    separator after the last one -- reproduced directly rather than
    imported, since this port has no generic `ostream_container` helper
    (nothing else in this port prints an arbitrary container this way).
    Each element is a `score_t`, formatted the same way every other
    `score_t` reaching this stream is (`_format_score_t`) -- matches the
    measured oracle exactly: `-3 and($1 $2)  behavioral score: [-1 -1 -1]`
    (`$CLASSIC/build/moses/moses/main/moses -i dataset.csv -m1000 -t1`).
    """
    elems = " ".join(_format_score_t(x) for x in bs)
    return f"{_BEHAVIORAL_SCORE_PREFIX} [{elems}]"


def ostream_scored_combo_tree(
    sct: scored_combo_tree,
    output_score: bool = True,
    output_cscore: bool = True,
    output_demeID: bool = True,
    output_bscore: bool = True,
    labels: Sequence[str] = (),
    fmt: output_format = output_format.combo,
) -> str:
    """C++: moses/moses/moses/types.cc:175 ostream_scored_combo_tree

        if (output_score) out << sct.get_score() << " ";
        ostream_combo_tree(out, sct.get_tree(), labels, fmt);
        static const bool output_weight = false;
        if (output_weight) out << " weight:" << sct.get_weight();
        if (output_cscore) out << " " << sct.get_composite_score();
        if (output_demeID) out << " demeID: " << sct.get_demeID();
        if (output_bscore and sct.get_bscore().size() > 0)
            ostream_behavioral_score(out << " ", sct.get_bscore());
        return out << std::endl;

    Ported line-for-line, string-concatenation in place of `<<` chaining.
    `output_weight` is a LOCAL `static const bool` hardcoded to `false`
    with no way to set it from outside this function (not a CLI flag, not
    a parameter) -- its whole `if` body is dead code in every real build;
    not ported at all, matching that it can never fire.

    `output_demeID` raises `NotImplementedError` when `True`: this port's
    `scored_combo_tree` carries no `demeID_t`/`get_demeID()` at all (see
    that class's own module docstring -- deme bookkeeping is out of scope
    for this whole port cycle, not merely deferred), so there is no value
    to print. Classic's CLI default is `--output-deme-id=false`
    (`problem-params.cc:1121`, `output_deme_id`), so a real `moses_exec.py`
    run never engages this gate; only a caller passing `output_demeID=True`
    explicitly (including this function's OWN default, matching
    `types.h:408`'s `bool output_demeID = true` exactly, which is what
    classic's bare `operator<<(ostream&, const scored_combo_tree&)`
    (types.h:421-424) relies on) reaches it -- gate-then-raise for an
    engaged-but-unported field, not a silent wrong answer, per this
    slice's design rule.

    `labels`/`fmt` are new parameters with no direct C++ counterpart
    ordering issue: `ostream_combo_tree`'s own `labels`/`fmt` parameters
    (iostream_combo.h) are threaded straight through to
    `combo_tree_to_str`.
    """
    parts: list[str] = []
    if output_score:
        parts.append(f"{_format_score_t(sct.get_score())} ")
    # `io_score_precision`, not `combo_tree_to_str`'s default 6. C++ writes
    # the tree into the SAME stream `metapop_printer` already set to
    # `setprecision(io_score_precision)` (moses_main.h:110), and that
    # setting is sticky for everything written afterwards -- including
    # contin constants nested inside the tree. Measured before the fix:
    # `moses -Hsr -k3 -m600 -W1 -r9` printed `*(0.543044125185779492 $1)`
    # in classic and `*(0.543044 $1)` here.
    parts.append(combo_tree_to_str(sct.get_tree(), labels, fmt, io_score_precision))
    if output_cscore:
        parts.append(f" {sct.get_composite_score()}")
    if output_demeID:
        raise NotImplementedError(
            "ostream_scored_combo_tree: output_demeID=True needs "
            "scored_combo_tree.get_demeID(), which this port's "
            "scored_combo_tree does not carry -- deme bookkeeping is out "
            "of scope for this whole port cycle (scored_combo_tree's own "
            "module docstring). Classic's own CLI default "
            "(--output-deme-id) is False; pass output_demeID=False."
        )
    bscore = sct.get_bscore()
    if output_bscore and len(bscore) > 0:
        parts.append(" " + ostream_behavioral_score(bscore))
    return "".join(parts) + "\n"


# =======================================================================
# collections of trees (C++: moses/moses/moses/types.h:320-395)
#
# `std::set`/`std::unordered_set`/`boost::ptr_set` with a custom Compare
# (or hash+equal pair) have no direct Python equivalent -- a bare Python
# `set()` uses only `__hash__`/`__eq__`, and `scored_combo_tree.__hash__`
# is explicitly `None` above (its own `__eq__` compares tree, bscore AND
# cscore together, which is deliberately NOT what any of these three
# containers dedups by). Every container below is therefore its own small
# class, reproducing the real container's ORDERING and DEDUPLICATION
# semantics exactly -- the actual, measured exit criterion for this task --
# rather than borrowing a stdlib container that would silently get either
# wrong. See each container's own `# DIVERGENCE` marker for precisely what
# C++ container it stands in for and why a 1:1 Python equivalent does not
# exist.


def _combo_tree_structural_eq(t1: object, t2: object) -> bool:
    """C++: opencog/util/tree.h:3011 tree

    Non-identifier member per the project's convention (see
    `composite_score.__lt__`'s docstring, types.py, for the same pattern):
    cites the class the operator is templated over (`tree<T>`) at the
    free function's own line, since a free template `operator==` has no
    enclosing class of its own to cite. The real symbol is the free
    template `bool operator==(const tree<T>& t1, const tree<T>& t2)`
    (tree.h:3011-3014).

    Not itself one of this task's four named functors -- `scored_combo_tree_
    equal` below (types.cc:70-74) calls it via `tr1.get_tree() ==
    tr2.get_tree()`, and `scored_combo_tree.__eq__` above now calls it too
    (see that method's own FIX note) -- but the real *combo_tree* equality
    check both depend on, ported here since this port's own `tree`/
    `tree_node` classes (`pymetta/util/tree.py`) define no `__eq__`: a bare
    `t1 == t2` on two `tree` objects falls back to Python's identity
    comparison, never to structural comparison, so it cannot be used to
    implement this.

    `t1.empty() ? t2.empty() : (t2.empty() ? false :
    t1.equal_subtree(t1.begin(), t2.begin()))` (tree.h:3011-3014), ported
    directly against `pymetta.util.tree.tree`'s own already-ported
    `equal_subtree` (tree.py:841). Passes `vertex_equal`, NOT
    `equal_subtree`'s bare-`==` default, as the element predicate: C++'s
    real default there is `std::equal_to<T>`, which for `vertex` (a
    `boost::variant`) discriminates by held alternative before comparing a
    value, exactly what `vertex_equal` reproduces and bare `==` does not
    (`tree.equal_subtree`'s own docstring names this trap explicitly; see
    also `combo/assumption.py`'s `equal_assumptions`, the first call site
    to hit it).
    """
    if t1.empty():
        return t2.empty()
    if t2.empty():
        return False
    return t1.equal_subtree(t1.begin(), t2.begin(), vertex_equal)


def _size_tree_less(t1: object, t2: object) -> bool:
    """C++: opencog/util/tree.h:3299 size_tree_order<vertex>::operator()(tree, tree)

    Not itself a ported C++ *identifier* (it has no name of its own in
    C++ beyond the class template `size_tree_order<vertex>` -- see
    `sct_score_greater`/`sct_tree_greater` below, whose `types.cc` bodies
    both call `size_tree_order<vertex>()(bs_tr1.get_tree(),
    bs_tr2.get_tree())` verbatim), but load-bearing for both: "if
    size(tr1)!=size(tr2) then tr1 < tr2; otherwise
    lexicographic_subtree_order" (tree.h:3297-3298's own comment) -- a
    strict "less than" over whole combo trees, smaller tree first, then
    (for equal sizes) `lexicographic_subtree_order<vertex>::cmp(tr1.begin(),
    tr2.begin()) > 0`, already ported and promoted for reuse as
    `lexicographic_subtree_less` (comboreduct/combo/vertex.py).
    """
    s1, s2 = t1.size(), t2.size()
    if s1 == s2:
        return lexicographic_subtree_less(t1.begin(), t2.begin())
    return s1 < s2


class sct_score_greater:
    """C++: moses/moses/moses/types.h:337 sct_score_greater

    Non-identifier member per the project's convention (see
    `instance_scorer.py`'s `iscorer_base.__call__` docstring for the same
    pattern): the real member is `sct_score_greater::operator()`
    (types.cc:77-100). "greater_than operator for scored_combo_tree. The
    order is determined by the composite score ... If the scores are
    equal ... the tree sizes are compared; if these are equal, the trees
    are compared lexicographically ... this ordering makes the
    non-standard assumption that anything is greater than nan" (types.h:
    323-334's own doc comment, condensed). `sct_score_greater`'s ordering
    is what `scored_combo_tree_ptr_set` (below) sorts by, and what
    metapopulation exemplar selection (S10 Task 2, out of THIS task's
    scope) walks in order -- this is the one comparator this whole task's
    exit criterion singles out as load-bearing.

    DO NOT FIX (verbatim classic quirk, `types.cc:99`): once score AND raw
    score are both tied, the final tie-break returns `size_tree_order<
    vertex>()(bs_tr1.get_tree(), bs_tr2.get_tree())` DIRECTLY -- i.e. the
    ASCENDING "smaller tree sorts first" relation (`_size_tree_less`
    above) -- with NO negation, even though every earlier stage of this
    same comparator returns `True` for the *higher*-scoring (i.e.
    "greater") operand. `types.cc:108`'s comment even undersells it: "size_
    tree_order first uses tree size, then lexicographic order" -- it does
    not say the relation gets flipped, because it does not. Confirmed
    directly against `types.cc:99`'s literal `return` statement, not
    inferred from the class's own "greater_than operator" doc comment.
    """

    def __call__(self, bs_tr1: scored_combo_tree, bs_tr2: scored_combo_tree) -> bool:
        csc1 = bs_tr1.get_composite_score()
        csc2 = bs_tr2.get_composite_score()

        if csc1 > csc2:
            return True
        if csc1 < csc2:
            return False

        # Tied composite scores. Break the tie on the raw score (types.cc:
        # 89-93) before falling through to tree shape below.
        sc1 = csc1.get_score()
        sc2 = csc2.get_score()

        if sc1 > sc2:
            return True
        if sc1 < sc2:
            return False

        # DO NOT FIX: see class docstring -- this is the ASCENDING
        # size/lex relation, not its negation, ported verbatim from
        # types.cc:99.
        return _size_tree_less(bs_tr1.get_tree(), bs_tr2.get_tree())


class sct_tree_greater:
    """C++: moses/moses/moses/types.h:346 sct_tree_greater

    Non-identifier member, same convention as `sct_score_greater` above:
    the real member is `sct_tree_greater::operator()` (types.cc:103-109).
    "greater_than operator for scored_combo_tree. The order is determined
    first by tree size, then by tree lexicographic order" (types.h:343-345's
    own doc comment). This is the comparator `scored_combo_tree_tset`
    (below) sorts by.

    DO NOT FIX (verbatim classic quirk, `types.cc:108`): the ENTIRE body is
    `return size_tree_order<vertex>()(bs_tr1.get_tree(), bs_tr2.get_tree());`
    -- the ASCENDING "smaller/lexicographically-first tree sorts first"
    relation (`_size_tree_less` above), despite this functor's own
    "greater_than" name. Same quirk `sct_score_greater` documents at
    greater length -- see that class's own docstring for why this is
    ported as-is rather than negated to match the name.
    """

    def __call__(self, bs_tr1: scored_combo_tree, bs_tr2: scored_combo_tree) -> bool:
        return _size_tree_less(bs_tr1.get_tree(), bs_tr2.get_tree())


_BUILTIN_HASH_HEADROOM = len(builtin)
# Analogous to C++'s `static const size_t c1 = size_t(id::builtin_count);`
# (vertex.h:436) -- an offset added to a non-`builtin` vertex's hash so it
# cannot collide with a `builtin`'s own raw-ordinal hash range (`0` ..
# `_BUILTIN_HASH_HEADROOM - 1`, see `_vertex_hash` below). NOT the same
# NUMBER as C++'s `id::builtin_count`: this port's `builtin` enum
# (comboreduct/combo/vertex.py) does not carry the full real `id::builtin`
# enumerator list, only this port's own narrower subset -- see
# `_vertex_hash`'s own DIVERGENCE note for why the exact number does not
# matter here the way it would for a `score_t` value.


def _hash_combine(seed: int, value: int) -> int:
    """No direct C++ symbol -- stands in for `boost::hash_combine`, which
    both `hash_value(vertex)` (vertex.h:432) and `scored_combo_tree_hash::
    operator()` (types.cc:60-68) call in real C++. See `_vertex_hash`'s own
    DIVERGENCE note for why boost's actual mixing algorithm is not
    reproduced bit-for-bit. This keeps the same *shape* (fold each new
    value into a running seed, order-sensitively) using a well-known
    golden-ratio-constant mix, masked to 64 bits so folding over a large
    tree cannot grow an unbounded Python int.
    """
    seed &= 0xFFFFFFFFFFFFFFFF
    value &= 0xFFFFFFFFFFFFFFFF
    seed ^= (value + 0x9E3779B97F4A7C15 + (seed << 6) + (seed >> 2)) & 0xFFFFFFFFFFFFFFFF
    return seed & 0xFFFFFFFFFFFFFFFF


def _vertex_hash(v: object) -> int:
    """C++: moses/comboreduct/combo/vertex.h:432 hash_value(const vertex&)

    Ported for exactly the four vertex alternatives this port's `vertex`
    union carries (`builtin | argument | contin_t | str`, comboreduct/
    combo/vertex.py) -- the other nine `hash_value` branches (wild_card,
    action, builtin_action, perception, definite_object,
    indefinite_object, message, procedure_call, action_symbol, ann_type)
    have no case to dispatch to here, the same narrowing that module's own
    docstring already makes for the union itself.

    # DIVERGENCE(C++ moses/comboreduct/combo/vertex.h:432-518): the literal
    # arithmetic is NOT reproduced bit-for-bit for the `contin_t`/`str`
    # cases. Real C++ combines `boost::hash_value(double)`/`boost::
    # hash_value(std::string)` with `boost::hash_combine` -- both
    # library-internal, boost-version-dependent algorithms (they have
    # changed across boost releases even within C++ itself) -- plus the
    # offset constant `c_last` (`vertex.h:441`, `c3`'s own alias), where
    # `c2` (`vertex.h:439`) is sized off `MAX_TREE_ARGS` (a fixed literal,
    # 256000) and only `c3` (`vertex.h:440`) is sized off
    # `id::action_count` -- CORRECTED here; this docstring previously
    # attributed BOTH offsets to `action_count`. Either way, this port's
    # `vertex` union has no equivalent alternative for either constant.
    # None of that
    # is reproducible from Python, and -- UNLIKE `score_t` narrowing (this
    # module's own top-of-file DIVERGENCE marker) -- it does not need to
    # be: this hash value is never compared against a literal oracle
    # value anywhere, only used internally by `scored_combo_tree_set`'s
    # hash-bucketing (below), whose only real behavioural requirement is
    # hash/equality CONSISTENCY -- `a == b` under `scored_combo_tree_equal`
    # must imply `hash(a) == hash(b)` -- which this satisfies by
    # construction. `std::unordered_set` itself makes no iteration-order
    # promise in C++ either, so no observable behaviour depends on the
    # literal value. The `builtin`/`argument` cases below need no such
    # divergence: neither one calls `boost::hash_value` in C++ at all, so
    # both are ported as exact integer arithmetic.
    """
    if isinstance(v, contin_t):
        return _hash_combine(_BUILTIN_HASH_HEADROOM, hash(v))
    if isinstance(v, str):
        return _hash_combine(_BUILTIN_HASH_HEADROOM, hash(v))
    if isinstance(v, builtin):
        # C++: vertex.h:450-451 `if (const builtin* h = boost::get<builtin>
        # (&v)) return size_t(*h);` -- the raw enum ordinal, verbatim.
        return int(v)
    if isinstance(v, argument):
        # C++: vertex.h:454-455 `size_t(a->idx * (a->is_negated() + 2)) +
        # c1` -- pure integer arithmetic, no `boost::hash_value` call, so
        # ported exactly (`_BUILTIN_HASH_HEADROOM` stands in for `c1`; see
        # that constant's own note for why the NUMBER differs from C++'s
        # while the FORMULA does not).
        return v.idx * (int(v.is_negated()) + 2) + _BUILTIN_HASH_HEADROOM
    raise TypeError(f"vertex {v!r} is not one of this port's four alternatives")


class scored_combo_tree_hash:
    """C++: moses/moses/moses/types.h:352 scored_combo_tree_hash

    Non-identifier member, same convention as `sct_score_greater` above:
    the real member is `scored_combo_tree_hash::operator()` (types.cc:
    60-68), which folds `_vertex_hash` (standing in for C++'s
    `combo::hash_value(vertex)`) over EVERY node of the tree -- `for
    (combo::vertex vtx : tr)` walks the WHOLE tree pre-order, not just its
    root or top-level children -- via `_hash_combine` (standing in for
    `boost::hash_combine`), seeded at 0 (types.cc:62's `size_t hash = 0;`).
    See `_vertex_hash`'s own DIVERGENCE note for what is and is not
    bit-exact here.
    """

    def __call__(self, sct: scored_combo_tree) -> int:
        h = 0
        for node in sct.get_tree():
            h = _hash_combine(h, _vertex_hash(node.data))
        return h


class scored_combo_tree_equal:
    """C++: moses/moses/moses/types.h:356 scored_combo_tree_equal

    Non-identifier member, same convention as `sct_score_greater` above:
    the real member is `scored_combo_tree_equal::operator()` (types.cc:
    70-74): `tr1.get_tree() == tr2.get_tree()` -- genuine STRUCTURAL
    `combo_tree::operator==` (`_combo_tree_structural_eq` above), NOT
    `scored_combo_tree::operator==` (which also compares bscore/cscore).
    This functor cares about the TREE alone -- exactly what makes
    `scored_combo_tree_set`'s dedup "same combo tree regardless of score",
    not "identical in every field".
    """

    def __call__(self, tr1: scored_combo_tree, tr2: scored_combo_tree) -> bool:
        return _combo_tree_structural_eq(tr1.get_tree(), tr2.get_tree())


class _sct_comparator_set:
    """Not itself a ported C++ symbol -- shared sorted-list-plus-binary-
    search implementation backing `scored_combo_tree_tset` and
    `scored_combo_tree_ptr_set` below, the SAME approach `comboreduct/
    reduct/logical_rules.py`'s `subtree_set` and `comboreduct/reduct/
    flat_normal_form.py`'s `clause` already use for this port's other
    `std::set`/`boost::ptr_set`-with-custom-`Compare` instances (this port
    has no balanced-tree/skip-list primitive to reuse instead).

    `comp(a, b)` must be a strict weak ordering ("goes before" -- the same
    role a `Compare` template parameter plays for `boost::ptr_set`/
    `std::set`: elements sort ascending by `comp`, i.e. `comp(a, b) is
    True` places `a` before `b`). Two elements are DEDUPLICATED -- a real
    `std::set`'s own equivalence-class semantics -- whenever NEITHER
    compares less than the other under `comp` (`not comp(a, b) and not
    comp(b, a)`), which is emphatically NOT the same test as `a == b`:
    `subtree_set`'s own docstring makes exactly this point, and it matters
    even more here, since neither `sct_tree_greater` nor `sct_score_
    greater` has anything to do with `scored_combo_tree.__eq__` (which
    also isn't even reachable this way -- `scored_combo_tree.__hash__` is
    `None`, so a `scored_combo_tree` could never go into a bare Python
    `set()` regardless; this class exists precisely to fill that gap).
    """

    __slots__ = ("_items", "_comp")

    def __init__(self, comp) -> None:
        self._comp = comp
        self._items: list[scored_combo_tree] = []

    def _bisect(self, x: scored_combo_tree) -> tuple[int, bool]:
        lo, hi = 0, len(self._items)
        comp = self._comp
        while lo < hi:
            mid = (lo + hi) // 2
            v = self._items[mid]
            if comp(v, x):
                lo = mid + 1
            elif comp(x, v):
                hi = mid
            else:
                return mid, True
        return lo, False

    def insert(self, sct: scored_combo_tree) -> bool:
        """C++: std::set<T, Compare>::insert / boost::ptr_set<T, Compare>::insert

        Returns whether `sct` was newly inserted (`False` if an element
        already comparing EQUIVALENT to it -- see class docstring -- was
        already present).
        """
        idx, found = self._bisect(sct)
        if found:
            return False
        self._items.insert(idx, sct)
        return True

    def erase(self, sct: scored_combo_tree) -> None:
        """C++: std::set<T, Compare>::erase(const key_type&) / boost::
        ptr_set<T, Compare>::erase(const key_type&)

        A no-op if no equivalent element is present, matching `std::set`'s
        own permissive contract (same convention `subtree_set.erase`
        documents).
        """
        idx, found = self._bisect(sct)
        if found:
            del self._items[idx]

    def __contains__(self, sct: scored_combo_tree) -> bool:
        return self._bisect(sct)[1]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        """C++: std::set::empty / boost::ptr_set::empty"""
        return not self._items


class scored_combo_tree_tset:
    """C++: moses/moses/moses/types.h:378 scored_combo_tree_tset

    `typedef boost::ptr_set<scored_combo_tree, sct_tree_greater>
    scored_combo_tree_tset;` -- "fairly fast, mutable storage for combo
    trees, based on the combo tree itself, and not how its scored ... more
    stable [than `scored_combo_tree_ptr_set`], precisely because it does
    not use the scores (which thus are allowed to change, depending on the
    situation)." (types.h:373-378's own comment, condensed.)

    # DIVERGENCE(C++ moses/moses/moses/types.h:378): `boost::ptr_set` owns
    # heap-allocated elements by pointer, using `Compare` purely to order
    # them -- Python has no pointer-owning container at all (ownership is
    # moot in a garbage-collected language), so this stores plain
    # `scored_combo_tree` object references directly in a sorted list
    # (`_sct_comparator_set`, binary-search insert/erase -- see its own
    # docstring). What this DOES reproduce exactly is the thing this
    # task's exit criterion actually measures: `sct_tree_greater`'s
    # ordering, and its "neither compares less" deduplication semantics.
    """

    def __init__(self, items=()) -> None:
        self._set = _sct_comparator_set(sct_tree_greater())
        for x in items:
            self._set.insert(x)

    def insert(self, sct: scored_combo_tree) -> bool:
        """C++: boost::ptr_set<scored_combo_tree, sct_tree_greater>::insert"""
        return self._set.insert(sct)

    def erase(self, sct: scored_combo_tree) -> None:
        """C++: boost::ptr_set<scored_combo_tree, sct_tree_greater>::erase"""
        self._set.erase(sct)

    def __contains__(self, sct: scored_combo_tree) -> bool:
        return sct in self._set

    def __iter__(self):
        return iter(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def empty(self) -> bool:
        """C++: boost::ptr_set::empty"""
        return self._set.empty()


class scored_combo_tree_ptr_set:
    """C++: moses/moses/moses/types.h:388 scored_combo_tree_ptr_set

    `typedef boost::ptr_set<scored_combo_tree, sct_score_greater>
    scored_combo_tree_ptr_set;` -- "holds scored combo trees, using the
    composite score to order the elements ... if the scorer is changing
    the way in which it is scoring over time ... there is no way to
    guarantee that a given tree appears only once in the set. You have
    been warned!" (types.h:381-389's own comment, condensed.) This is the
    container the metapopulation's exemplar pool is ordered by (S10 Task 2,
    out of THIS task's scope) -- iterating it in order visits the
    BEST-scoring candidate first, per `sct_score_greater`'s own ordering.

    # DIVERGENCE(C++ moses/moses/moses/types.h:388): same reason as
    # `scored_combo_tree_tset` above -- `boost::ptr_set`'s pointer
    # ownership has no Python equivalent and is not what this task's exit
    # criterion measures. Backed by the same `_sct_comparator_set` sorted-
    # list-plus-binary-search implementation, parameterized on
    # `sct_score_greater` instead of `sct_tree_greater`.
    """

    def __init__(self, items=()) -> None:
        self._set = _sct_comparator_set(sct_score_greater())
        for x in items:
            self._set.insert(x)

    def insert(self, sct: scored_combo_tree) -> bool:
        """C++: boost::ptr_set<scored_combo_tree, sct_score_greater>::insert"""
        return self._set.insert(sct)

    def erase(self, sct: scored_combo_tree) -> None:
        """C++: boost::ptr_set<scored_combo_tree, sct_score_greater>::erase"""
        self._set.erase(sct)

    def __contains__(self, sct: scored_combo_tree) -> bool:
        return sct in self._set

    def __iter__(self):
        return iter(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def empty(self) -> bool:
        """C++: boost::ptr_set::empty"""
        return self._set.empty()


class scored_combo_tree_set:
    """C++: moses/moses/moses/types.h:369 scored_combo_tree_set

    `typedef std::unordered_set<scored_combo_tree, scored_combo_tree_hash,
    scored_combo_tree_equal> scored_combo_tree_set;` -- "provides an O(1)
    way of determining if a combo tree is in the set, or not (and getting
    its score, if it is) ... invokes the copy constructor for insertion."
    (types.h:362-367's own comment, condensed.)

    # DIVERGENCE(C++ moses/moses/moses/types.h:369): a real
    # `std::unordered_set` is a hash table with implementation-defined
    # bucket layout and, consequently, an UNSPECIFIED iteration order --
    # C++ itself makes no promise about what order `for (auto& sct :
    # candidates)` visits elements in. Python has no built-in hash-table
    # type that accepts a custom hash+equality PAIR the way
    # `unordered_set`'s two extra template parameters do (a bare `set()`
    # only ever uses `__hash__`/`__eq__`, and `scored_combo_tree.__hash__`
    # is explicitly `None` above -- unhashable -- precisely because its
    # own `__eq__` also compares bscore/cscore, which `scored_combo_tree_
    # hash`/`scored_combo_tree_equal` deliberately do NOT: types.cc:70-74
    # defines `scored_combo_tree_equal` as `tr1.get_tree() ==
    # tr2.get_tree()` alone). This reproduces the actual observable
    # contract instead: a small hand-rolled hash table (`dict[int,
    # list[scored_combo_tree]]`, bucketed by `scored_combo_tree_hash`)
    # that dedups two elements whenever `scored_combo_tree_equal` says
    # they match -- i.e. whenever their TREES alone are equal, regardless
    # of score -- which is the one behaviour every real caller of this
    # type actually depends on. Like real C++, this makes no promise about
    # iteration order.
    """

    def __init__(self, items=()) -> None:
        self._buckets: dict[int, list[scored_combo_tree]] = {}
        self._size = 0
        for x in items:
            self.insert(x)

    def insert(self, sct: scored_combo_tree) -> bool:
        """C++: std::unordered_set<scored_combo_tree, scored_combo_tree_hash,
        scored_combo_tree_equal>::insert

        Returns whether `sct` was newly inserted.
        """
        h = scored_combo_tree_hash()(sct)
        eq = scored_combo_tree_equal()
        bucket = self._buckets.setdefault(h, [])
        for existing in bucket:
            if eq(existing, sct):
                return False
        bucket.append(sct)
        self._size += 1
        return True

    def erase(self, sct: scored_combo_tree) -> None:
        """C++: std::unordered_set<...>::erase(const key_type&)"""
        h = scored_combo_tree_hash()(sct)
        bucket = self._buckets.get(h)
        if not bucket:
            return
        eq = scored_combo_tree_equal()
        for i, existing in enumerate(bucket):
            if eq(existing, sct):
                del bucket[i]
                self._size -= 1
                return

    def __contains__(self, sct: scored_combo_tree) -> bool:
        h = scored_combo_tree_hash()(sct)
        eq = scored_combo_tree_equal()
        return any(eq(existing, sct) for existing in self._buckets.get(h, ()))

    def __iter__(self):
        for bucket in self._buckets.values():
            yield from bucket

    def __len__(self) -> int:
        return self._size

    def empty(self) -> bool:
        """C++: std::unordered_set::empty"""
        return self._size == 0

    def clear(self) -> None:
        """C++: std::unordered_set<...>::clear

        Empties the container IN PLACE -- added for `metapopulation.
        update_best_candidates` (S10 branch review, Minor): C++'s
        `_best_candidates.clear()` (merging.cc) mutates the SAME object
        every existing `best_candidates()` reference/alias already points
        at, so a caller holding one sees the clear too. Before this
        method existed, that call site rebound `self._best_candidates =
        scored_combo_tree_set()` instead -- a NEW object -- which left any
        prior `best_candidates()` caller looking at a now-stale set that
        never sees subsequent inserts. `clear()` is the fix: mutate, don't
        rebind, matching what a real reference target does in C++.
        """
        self._buckets.clear()
        self._size = 0


worst_composite_score = composite_score()
# C++: moses/moses/moses/types.cc:112 worst_composite_score
#
# "the empty composite_score ctor returns the worst composite score"
# (types.cc:111's own comment) -- a module-level constant, not a ported
# function/class, so (like `combo_tree` in vertex.py) it gets a plain
# comment rather than a docstring breadcrumb.
