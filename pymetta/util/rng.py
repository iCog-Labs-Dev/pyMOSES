"""MT19937 random generator, bit-compatible with cogutil's RandGen.

C++: opencog/util/mt19937ar.cc:44 MT19937RandGen
"""

import struct

_N, _M = 624, 397
_MATRIX_A, _UPPER_MASK, _LOWER_MASK = 0x9908B0DF, 0x80000000, 0x7FFFFFFF
_UINT32_MAX = 0xFFFFFFFF

# `(float) std::mt19937::max()`, precomputed. `max()` is 4294967295, which is
# NOT representable in binary32 -- the two neighbours are 4294967296 (2**32)
# and 4294966784 (2**32 - 512, one ulp below), and 4294967295 is one away from
# the first and 511 from the second, so it rounds UP to exactly 2**32.
# Measured on the real cogutil class: `(double)(float)MT19937RandGen::max()`
# prints 4294967296.
_RANDFLOAT_DIVISOR = 4294967296.0


def _to_float32(x: float) -> float:
    """No direct C++ symbol: the implicit conversion to a C ``float``
    (IEEE-754 binary32, round-to-nearest-even) that ``randfloat``'s
    ``operator()() / (float)max()`` performs on its unsigned operand.

    Same mechanism as ``pymetta.moses.moses.types._to_score_t``, restated
    here rather than imported: ``pymetta.util`` sits below
    ``pymetta.moses`` and this value is a plain C ``float``, not a
    ``score_t``.
    """
    return struct.unpack("f", struct.pack("f", x))[0]


class MT19937RandGen:
    """C++: opencog/util/RandGen.h:49 RandGen

    cogutil's RandGen *is* std::mt19937 with convenience methods layered on.
    Python's `random` module cannot be substituted: random.seed(int) routes
    through init_by_array, while std::mt19937::seed(unsigned) uses
    init_genrand, and the two streams diverge from the first draw.
    """

    def __init__(self, seed: int = 0) -> None:
        self._mt = [0] * _N
        self._mti = _N + 1
        self.seed(seed)

    def seed(self, s: int) -> None:
        """C++: opencog/util/RandGen.h:49 RandGen

        RandGen.h:49 is `class RandGen : public std::mt19937` -- the most
        precise citation available, since seed() is std::mt19937::seed
        (init_genrand), inherited from the standard-library base and never
        itself declared anywhere in RandGen.h. init_genrand is not
        init_by_array: Python's random.seed(int) uses the latter, and the two
        streams diverge from the first draw (see the class docstring).
        """
        mt = self._mt
        mt[0] = s & _UINT32_MAX
        for i in range(1, _N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & _UINT32_MAX
        self._mti = _N

    def genrand_int32(self) -> int:
        """C++: opencog/util/RandGen.h:49 RandGen

        RandGen.h:49 is `class RandGen : public std::mt19937` -- the most
        precise citation available, since this is std::mt19937::operator(),
        inherited from the standard-library base and never itself declared
        anywhere in RandGen.h.
        """
        mt = self._mt
        if self._mti >= _N:
            for i in range(_N):
                y = (mt[i] & _UPPER_MASK) | (mt[(i + 1) % _N] & _LOWER_MASK)
                mt[i] = mt[(i + _M) % _N] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & _UINT32_MAX

    def randint(self) -> int:
        """C++: opencog/util/mt19937ar.cc:44 MT19937RandGen::randint

        std::uniform_int_distribution<int> over [0, INT_MAX] against a 2^32
        generator: scaling is exactly 2 and the rejection branch is
        unreachable, so this is `raw >> 1` on any libstdc++, consuming one
        draw. Verified against a C++ probe over 12000 draws / 6 seeds.
        """
        return self.genrand_int32() >> 1

    def randint_n(self, n: int) -> int:
        """C++: opencog/util/mt19937ar.cc:66 MT19937RandGen::randint(int n)

        `(int)randint() % n`: C++'s `%` truncates toward zero, so the
        result's sign follows the DIVIDEND, not the divisor. `randint()`
        (above) is always non-negative (`std::uniform_int_distribution<int>`
        over `[0, INT_MAX]`), so for a non-negative dividend the truncated
        result is non-negative regardless of `n`'s sign, with magnitude
        `randint() % abs(n)` -- traced by hand and confirmed against a
        compiled probe on the real cogutil class: fresh `MT19937RandGen(1)`,
        three sequential calls, `randint(-25) == 22`, `randint(-1) == 0`,
        `randint(-29) == 6`. Python's own `%` instead takes the sign of the
        DIVISOR, so the naive `self.randint() % n` used here previously
        returned -3/0/-23 for that same sequence -- silently wrong for any
        negative `n`, just never observed, because every OTHER call site in
        this project (`lazy_random_selector.py`, `deme_expander.py`,
        `neighborhood_sampling.py`) only ever passes a non-negative `n`
        (`abs(n) == n` there, so this fix changes nothing for them).
        `merging.py`'s `resize_metapop` is the one caller that can reach a
        negative `n` (see that function's own docstring for why), and only
        once its own C++-int32 narrowing is applied correctly.
        """
        return 0 if n == 0 else self.randint() % abs(n)

    def randdouble(self) -> float:
        """C++: opencog/util/mt19937ar.cc:55 MT19937RandGen::randdouble"""
        return self.genrand_int32() / float(_UINT32_MAX)

    def randbool(self) -> bool:
        """C++: opencog/util/mt19937ar.cc:78 MT19937RandGen::randbool"""
        return self.randint() % 2 == 0

    def randfloat(self) -> float:
        """C++: opencog/util/mt19937ar.cc:50 MT19937RandGen::randfloat

        ``return operator()() / (float)max();`` -- ONE raw draw, divided by
        ``(float)max()``. Every type in that one line matters:

        * ``operator()()`` is ``std::mt19937::operator()``, whose
          ``result_type`` is ``uint_fast32_t`` (64-bit on x86-64) holding a
          value in ``[0, 2**32)``. The usual arithmetic conversions against
          the ``float`` divisor narrow it to binary32 FIRST -- so a raw draw
          needing more than 24 significand bits is ROUNDED before the
          division, and 2**8 consecutive raw values can share one result.
          ``_to_float32`` is that narrowing.
        * ``(float)max()`` is ``(float)4294967295`` == ``2**32`` exactly
          (see ``_RANDFLOAT_DIVISOR``), a power of two, so the division
          itself is exact -- a pure exponent decrement, with no second
          rounding and no subnormal case (the smallest non-zero numerator
          is 1.0f, giving 2**-32).

        The C++ return type is ``float``; the Python ``float`` returned here
        holds that binary32 value exactly, so ``b > rng.randfloat()``
        (``biased_randbool``, $COGUTIL/opencog/util/random.h:127) compares
        the same two numbers in both languages provided ``b`` is itself
        float32-narrowed -- which it is, at its only caller
        (``feature_selector.sample_enforced_features``, whose weights
        problem-params.cc:2248 stores in a ``std::map<std::string,float>``).

        Verified against the real cogutil ``MT19937RandGen``, linked and
        called directly: ``tests/fixtures/rng.json``'s ``randfloat`` series,
        2000 draws for each of seeds 0 and 42, agrees bit for bit.
        """
        return _to_float32(self.genrand_int32()) / _RANDFLOAT_DIVISOR

    def randdouble_one_excluded(self) -> float:
        """C++: opencog/util/mt19937ar.cc:60 MT19937RandGen::randdouble_one_excluded"""
        raise NotImplementedError(
            "Zero call sites in the MOSES core, and std::uniform_real_distribution "
            "is implementation-defined. See spec section 6.2."
        )

    def rand_discrete(self, weights) -> int:
        """C++: opencog/util/mt19937ar.cc:84 MT19937RandGen::rand_discrete"""
        raise NotImplementedError(
            "Zero call sites in the MOSES core, and std::discrete_distribution "
            "is implementation-defined. See spec section 6.2."
        )


_instance: MT19937RandGen | None = None


# DIVERGENCE(C++ opencog/util/mt19937ar.cc:90): classic's singleton is
# thread_local; ours is process-global. The MOSES core paths in scope are
# single-threaded, and a process-global singleton is what makes a seeded run
# reproducible at all in Python.
def randGen() -> MT19937RandGen:
    """C++: opencog/util/mt19937ar.cc:90 opencog::randGen"""
    global _instance
    if _instance is None:
        _instance = MT19937RandGen(0)
    return _instance
