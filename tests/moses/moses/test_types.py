"""C++: moses/moses/moses/types.h and types.cc

Regression guard for fix round 2's "eleventh site": `composite_score`'s own
constructor and setters (`set_score`, `set_complexity_penalty`,
`set_uniformity_penalty`) must narrow their `score_t`-typed arguments to
IEEE-754 binary32, not hold a caller's raw double verbatim. This was found
latent -- every call site inside this port (`behave_cscore.get_cscore_
nocache`) already passed pre-narrowed values, so no existing test caught
it -- but S7+ (metapopulation, merging, deme expansion) will construct
`composite_score` DIRECTLY, the exact shape of bug a full review cycle was
needed to find once already (fix round 1's `score_t` narrowing). These
tests construct `composite_score` directly with plain, unnarrowed Python
doubles and assert every field comes back float32-narrowed.
"""

from pymetta.moses.moses.types import _EQ_EPSILON, _is_approx_eq, _to_score_t, composite_score

# 1/3.5 is not exactly representable in binary32 (the fix round 1 review's
# own witness value: |1/3.5 - to_score_t(1/3.5)| ~= 1.28e-8) -- a passthrough
# bug is directly observable via `!=` against the raw double.
_UNNARROWED = 1.0 / 3.5

# Sanity check on the fixture itself: if this ever becomes exactly
# representable in binary32 (it won't, but a future refactor of `_to_score_t`
# could silently break this), every test below would pass vacuously.
assert _UNNARROWED != _to_score_t(_UNNARROWED), (
    "fixture value must actually require float32 rounding, or these tests "
    "prove nothing"
)


def test_constructor_narrows_score():
    cs = composite_score(_UNNARROWED, 5, 0.0, 0.0)
    assert cs.get_score() == _to_score_t(_UNNARROWED)
    assert cs.get_score() != _UNNARROWED


def test_constructor_narrows_complexity_penalty():
    cs = composite_score(0.0, 5, _UNNARROWED, 0.0)
    assert cs.get_complexity_penalty() == _to_score_t(_UNNARROWED)
    assert cs.get_complexity_penalty() != _UNNARROWED


def test_constructor_narrows_uniformity_penalty():
    cs = composite_score(0.0, 5, 0.0, _UNNARROWED)
    assert cs.get_uniformity_penalty() == _to_score_t(_UNNARROWED)
    assert cs.get_uniformity_penalty() != _UNNARROWED


def test_constructor_narrows_penalized_score_end_to_end():
    """`update_penalized_score` (types.h:166) computes `score -
    complexity_penalty`, then `-= uniformity_penalty`, each its own
    separate float32 operation -- reproduced here independently (not by
    calling the port's own helper) so this test does not just restate the
    implementation.
    """
    cs = composite_score(_UNNARROWED, 5, _UNNARROWED, _UNNARROWED)
    narrowed = _to_score_t(_UNNARROWED)
    expected = _to_score_t(_to_score_t(narrowed - narrowed) - narrowed)
    assert cs.get_penalized_score() == expected
    assert cs.get_penalized_score() != _UNNARROWED - _UNNARROWED - _UNNARROWED


def test_set_score_narrows():
    cs = composite_score()
    cs.set_score(_UNNARROWED)
    assert cs.get_score() == _to_score_t(_UNNARROWED)
    assert cs.get_score() != _UNNARROWED


def test_set_complexity_penalty_narrows():
    cs = composite_score()
    cs.set_complexity_penalty(_UNNARROWED)
    assert cs.get_complexity_penalty() == _to_score_t(_UNNARROWED)
    assert cs.get_complexity_penalty() != _UNNARROWED


def test_set_uniformity_penalty_narrows():
    cs = composite_score()
    cs.set_uniformity_penalty(_UNNARROWED)
    assert cs.get_uniformity_penalty() == _to_score_t(_UNNARROWED)
    assert cs.get_uniformity_penalty() != _UNNARROWED


def test_get_penalty_narrows_the_sum_not_just_the_operands():
    """`get_penalty` (types.h:127) returns `complexity_penalty +
    uniformity_penalty` (types.h:129) -- itself a `score_t + score_t`
    float32 operation in C++, not merely two already-narrow operands
    summed in double. Chosen so the SUM itself needs rounding even though
    both operands are already narrow (1/3.5 + 1/7.0, verified by direct
    computation to differ between the double sum and its float32
    narrowing) -- a weaker test (e.g. adding a narrow value to itself)
    would pass even if `get_penalty` forgot to narrow its own result.
    """
    a, b = 1.0 / 3.5, 1.0 / 7.0
    cs = composite_score(0.0, 0, a, b)
    narrowed_a, narrowed_b = _to_score_t(a), _to_score_t(b)
    double_sum = narrowed_a + narrowed_b
    narrowed_sum = _to_score_t(double_sum)
    assert double_sum != narrowed_sum, "fixture must need rounding on the SUM itself"
    assert cs.get_penalty() == narrowed_sum
    assert cs.get_penalty() != double_sum


# ---------------------------------------------------------------------------
# Regression tests for the final review's Important finding: `_EQ_EPSILON`
# and `_is_approx_eq` (both feed `composite_score.__eq__`) are the
# thirteenth and fourteenth narrowing sites -- found in the very module
# whose round-2 re-review concluded "no thirteenth found".
# ---------------------------------------------------------------------------

def test_eq_epsilon_is_narrowed_to_float32():
    """`types.cc:158`'s `#define EPSILON static_cast<score_t>(1e-6)` is a
    C++ `float` literal, not a `double` one: `static_cast<score_t>(1e-6)`
    rounds to 9.999999974752427e-07 in float32 -- a different value than
    the bare Python double `1e-6`.
    """
    assert _EQ_EPSILON == _to_score_t(1e-6)
    assert _EQ_EPSILON != 1e-6


def test_is_approx_eq_computes_each_step_in_float32_not_double():
    """`_is_approx_eq` instantiates cogutil's `is_approx_eq<FloatT=
    score_t>` (opencog/util/numeric.h:185) -- every arithmetic step
    (`x - y`, `x + y`, `epsilon * amp`) must be its own float32 operation,
    not accumulated in double.

    This witness pair is chosen so that `x - y`'s TRUE (infinite-precision)
    value needs float32 rounding, and the raw double result and its
    float32-narrowed counterpart straddle a deliberately-placed `epsilon`
    -- so the pre-fix implementation (`abs(x - y)` computed and compared
    in double, with no narrowing at all) and the fixed one give OPPOSITE
    answers for the exact same inputs, not just different intermediate
    values. `amp = abs(x + y)` is tiny here (~0.03) specifically so the
    relative-tolerance fallback branch cannot paper over the absolute-
    check disagreement the way it would for a large `amp`.
    """
    x, y = 299768.875, -299768.84375
    assert x == _to_score_t(x) and y == _to_score_t(y), "fixture operands must be float32-exact"

    raw_diff = x - y
    narrowed_diff = _to_score_t(x - y)
    assert raw_diff < narrowed_diff, "fixture must need rounding, and round UP, on x - y"

    epsilon = (raw_diff + narrowed_diff) / 2
    assert raw_diff < epsilon < narrowed_diff

    def old_is_approx_eq(a, b, eps):
        # The pre-fix implementation: plain double arithmetic throughout,
        # no `_to_score_t` narrowing anywhere.
        diff = abs(a - b)
        if diff < eps:
            return True
        return diff <= eps * abs(a + b)

    assert old_is_approx_eq(x, y, epsilon) is True
    assert _is_approx_eq(x, y, epsilon) is False
