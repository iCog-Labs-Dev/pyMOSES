"""C++: opencog/util/digraph.h:41 digraph

A directed graph over the integers ``[0, n)``, plus a randomized topological
sort. Ported for `local_structure.py`'s `local_structure_model` constructor,
which builds a `digraph` of raw-field dependencies (`_initial_deps`) purely to
feed it to `randomized_topological_sort` and derive a sampling order.

# DIVERGENCE(C++ opencog/util/digraph.h:96): `randomized_topological_sort`
# shuffles its ready-node queue with a BARE `std::random_shuffle(nodes.begin(),
# nodes.end())` -- no RNG argument, so it does NOT read `randGen()`. That
# stream is reproduced here rather than substituted; the residual divergence
# is only that a C library's global state is reproduced by a Python object,
# and the reproduction is HOST-SPECIFIC (glibc + libstdc++, see below).
#
# The chain of measured facts, each re-derived on 2026-09-10 and pinned by
# `tests/fixtures/univariate_rng.json` (captured by
# `tools/capture_univariate_rng.py` from `tools/cpp_probe/univariate_rng_probe.cc`):
#
#   1. libstdc++'s two-argument `std::random_shuffle` is
#      `for (i = first+1; i != last; ++i) { j = first + rand() % ((i-first)+1);
#      if (i != j) iter_swap(i, j); }` (`/usr/include/c++/16/bits/stl_algo.h`,
#      the `random_shuffle(_RandomAccessIterator, _RandomAccessIterator)`
#      overload). It has a second, xorshift-seeded arm guarded by
#      `#if RAND_MAX < __INT_MAX__`; glibc's `RAND_MAX` is `2147483647`
#      (`/usr/include/stdlib.h:91`) which IS `__INT_MAX__`, so that arm is not
#      even compiled here. `_random_shuffle` below is the surviving loop,
#      transliterated.
#   2. `git grep 'srand(' $CLASSIC $COGUTIL` finds ZERO call sites, so classic
#      consumes `rand()` from the state glibc statically initialises it to.
#      `opencog::randGen().seed(rand_seed)` (problem-params.cc:2116, the `-r`
#      flag) reseeds ONLY the mt19937 singleton; it never touches `rand()`.
#      digraph.h:96 is also the ONLY `rand()`/`random_shuffle` call site in
#      either checkout, so nothing else perturbs the stream.
#   3. glibc's static default state IS the state `srand(1)` produces:
#      `stdlib/random_r.c` initialises `unsafe_state` to point into the
#      TYPE_3 table `randtbl`, and `__srandom_r` maps `seed == 0` to 1.
#      MEASURED by the probe's `rand_default` / `rand_srand1` / `rand_srand0`
#      sections: 512 values, identical, and seed 0 == seed 1.
#   4. glibc's `rand()` is `__random()`, i.e. the TYPE_3 additive-feedback
#      generator: state of `DEG_3 == 31` `int32_t`s, `SEP_3 == 3`, seeded by
#      the Lehmer recurrence `state[i] = (16807 * state[i-1]) % 2147483647`
#      evaluated via Schrage's trick, then `31 * 10 == 310` warm-up draws
#      discarded; each draw is `*fptr += (uint32_t)*rptr` with the result
#      `>> 1` ("chucking least random bit"). Source: glibc `stdlib/random_r.c`
#      (`__srandom_r`, `__random_r`) and `stdlib/random.c`. `_CStdlibRand`
#      below transliterates exactly that, and the fixture pins it element for
#      element against the real `std::rand()`.
#
# What remains divergent, and cannot be otherwise: `-r <seed>` does not and
# cannot control this stream in classic either -- it is a property of the
# glibc/libstdc++ build classic happens to run on, not of MOSES's seeding
# contract. `_CStdlibRand` reproduces THIS host's; a classic binary built
# against a different libc would shuffle differently and this port would then
# be wrong in the same way its own oracle would be. The fixture's manifest
# records the toolchain (schema 2) precisely so that shows up as drift.
#
# Second, C's `rand()` state is per-PROCESS and classic runs one process per
# `moses` invocation, whereas `_cstdlib_rand()` below is a per-INTERPRETER
# singleton. Any in-process harness that drives the CLI more than once must
# call `_cstdlib_rand().srand(1)` between runs, exactly as CARRY-FORWARD.md
# 5.11 requires for the reduction singletons. Every differential sweep in
# this project uses a fresh subprocess per run and is unaffected.
"""

from __future__ import annotations

from collections import deque

__all__ = ["digraph", "randomized_topological_sort"]

_UINT32_MAX = 0xFFFFFFFF

# glibc `stdlib/random_r.c`'s TYPE_3 parameters: `DEG_3` state words with the
# feedback tap `SEP_3` ahead of the read pointer.
_DEG_3 = 31
_SEP_3 = 3


def _int32(value: int) -> int:
    """Wrap `value` into the C `int32_t` range, as GCC's implementation-defined
    conversion does. Needed because glibc's state array is `int32_t[]` while
    its update step (`*fptr += (uint32_t) *rptr`) is unsigned: the sum wraps
    modulo 2**32 and is then stored back as a signed word.
    """
    value &= _UINT32_MAX
    return value - 0x100000000 if value >= 0x80000000 else value


def _c_divmod(numerator: int, denominator: int) -> tuple[int, int]:
    """C's `/` and `%` on `long int`, which truncate the quotient TOWARD ZERO.

    Python's `//` and `%` floor instead, so they disagree with C for a
    negative numerator -- reachable here for any seed >= 2**31, since
    `__srandom_r` stores the seed into an `int32_t`. Reproduced explicitly
    rather than clamped (`score_t` convention, CARRY-FORWARD.md 4.2).
    """
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        quotient = -quotient
    return quotient, numerator - quotient * denominator


class _CStdlibRand:
    """glibc's `rand()` -- `stdlib/random_r.c`'s TYPE_3 additive-feedback
    generator -- reproduced draw for draw.

    Private (and so exempt from the breadcrumb requirement) because there is
    no C++ symbol to cite: this is libc, not MOSES or cogutil. The nearest
    citable declarations are `stdlib.h:577 rand` and `stdlib.h:579 srand`;
    the algorithm itself lives in glibc's `stdlib/random_r.c`
    (`__srandom_r`, `__random_r`) and is transliterated below.

    Reachable from this port only through `_random_shuffle`, which is
    reachable only from `randomized_topological_sort` -- see the module
    docstring for the whole chain and for what is still divergent.
    """

    def __init__(self, seed: int = 1) -> None:
        self.srand(seed)

    def srand(self, seed: int = 1) -> None:
        """glibc `stdlib/random_r.c` `__srandom_r`, TYPE_3 arm.

        `if (seed == 0) seed = 1;` then `state[0] = seed;` and

            for (i = 1; i < kc; ++i) {
                long int hi = word / 127773;
                long int lo = word % 127773;
                word = 16807 * lo - 2836 * hi;
                if (word < 0) word += 2147483647;
                *++dst = word;
            }

        -- Schrage's trick for `state[i] = (16807 * state[i-1]) % 2147483647`
        without overflowing 31 bits. Note the ORDER: `word` is an `int32_t`,
        so the product-difference is truncated to 32 bits BEFORE the `< 0`
        correction, which is why `_int32` wraps first here too.

        Then `fptr = &state[SEP_3]; rptr = &state[0]; kc *= 10;` and
        `while (--kc >= 0) __random_r(...)` -- 310 discarded warm-up draws.
        """
        seed &= _UINT32_MAX
        if seed == 0:
            seed = 1
        word = _int32(seed)
        state = [0] * _DEG_3
        state[0] = word
        for i in range(1, _DEG_3):
            hi, lo = _c_divmod(word, 127773)
            word = _int32(16807 * lo - 2836 * hi)
            if word < 0:
                word += 2147483647
            state[i] = word
        self._state = state
        self._fptr = _SEP_3
        self._rptr = 0
        for _ in range(_DEG_3 * 10):
            self.rand()

    def rand(self) -> int:
        """glibc `stdlib/random_r.c` `__random_r`, TYPE_3 arm; `rand()` is
        `__random()` is `__random_r(&unsafe_state, &retval)`.

            val = *fptr += (uint32_t) *rptr;
            *result = val >> 1;              // "Chucking least random bit."
            ++fptr;
            if (fptr >= end_ptr) { fptr = state; ++rptr; }
            else { ++rptr; if (rptr >= end_ptr) rptr = state; }

        The pointer bump is transliterated literally, INCLUDING the asymmetry
        that the `fptr` wrap advances `rptr` without range-checking it. That
        is safe in C only because `SEP_3 < DEG_3` keeps the two pointers a
        fixed distance apart; it is kept as written rather than "tidied" into
        one modulus, per this project's transliteration convention.
        """
        state = self._state
        value = (state[self._fptr] + state[self._rptr]) & _UINT32_MAX
        state[self._fptr] = _int32(value)
        result = value >> 1
        self._fptr += 1
        if self._fptr >= _DEG_3:
            self._fptr = 0
            self._rptr += 1
        else:
            self._rptr += 1
            if self._rptr >= _DEG_3:
                self._rptr = 0
        return result


_cstdlib_rand_instance: _CStdlibRand | None = None


def _cstdlib_rand() -> _CStdlibRand:
    """The process-wide `rand()` state, standing in for glibc's `unsafe_state`.

    Created in the state glibc statically initialises its own to -- which is
    measured (module docstring, fact 3) to be `srand(1)`'s state -- so the
    first draw of a fresh interpreter matches the first draw of a fresh
    `moses` process. See the module docstring's last paragraph for the
    in-process-reuse caveat.
    """
    global _cstdlib_rand_instance
    if _cstdlib_rand_instance is None:
        _cstdlib_rand_instance = _CStdlibRand(1)
    return _cstdlib_rand_instance


def _random_shuffle(values: list[int], rand: _CStdlibRand) -> None:
    """libstdc++'s two-argument `std::random_shuffle`, from
    `/usr/include/c++/16/bits/stl_algo.h`:

        if (first == last) return;
        for (_RandomAccessIterator i = first + _Dist(1); i != last; ++i) {
            // XXX rand() % N is not uniformly distributed
            _RandomAccessIterator j = first + _Dist(std::rand() % ((i - first) + 1));
            if (i != j) std::iter_swap(i, j);
        }

    This is a FORWARD Fisher-Yates with a biased `% n` index, not the
    textbook backward one, and the two produce different permutations from
    the same draw stream -- which is why it is transliterated rather than
    replaced with `random.shuffle` or a reversed loop.

    The `#if RAND_MAX < __INT_MAX__` xorshift arm above it is not reproduced:
    on glibc `RAND_MAX == __INT_MAX__ == 2147483647` (`stdlib.h:91`), so the
    preprocessor deletes that arm entirely from the build classic uses.
    """
    n = len(values)
    if n == 0:
        return
    for i in range(1, n):
        j = rand.rand() % (i + 1)
        if i != j:
            values[i], values[j] = values[j], values[i]


class digraph:
    """C++: opencog/util/digraph.h:41 digraph

    Each node is one of the integers ``[0, n)``. `_incoming[x]`/`_outgoing[x]`
    are ported as Python `set`s, matching C++'s `std::set<value_type>`.
    """

    def __init__(self, n: int) -> None:
        """C++: opencog/util/digraph.h:47 digraph"""
        self._incoming: list[set[int]] = [set() for _ in range(n)]
        self._outgoing: list[set[int]] = [set() for _ in range(n)]

    def insert(self, src: int, dst: int) -> None:
        """C++: opencog/util/digraph.h:50 insert"""
        self._incoming[dst].add(src)
        self._outgoing[src].add(dst)

    def erase(self, src: int, dst: int) -> None:
        """C++: opencog/util/digraph.h:55 erase"""
        self._incoming[dst].discard(src)
        self._outgoing[src].discard(dst)

    def n_nodes(self) -> int:
        """C++: opencog/util/digraph.h:60 n_nodes"""
        return len(self._incoming)

    def n_edges(self) -> int:
        """C++: opencog/util/digraph.h:64 n_edges"""
        return sum(len(s) for s in self._incoming)

    def empty(self) -> bool:
        """C++: opencog/util/digraph.h:67 empty"""
        return self.n_edges() == 0

    def incoming(self, x: int) -> set[int]:
        """C++: opencog/util/digraph.h:71 incoming

        Returns the live internal set (C++ returns `const value_set&`) --
        read-only by convention, matching `field_set.py`'s own aliasing
        precedent for `const&`-returning accessors (CARRY-FORWARD.md 4.1).
        """
        return self._incoming[x]

    def outgoing(self, x: int) -> set[int]:
        """C++: opencog/util/digraph.h:75 outgoing"""
        return self._outgoing[x]

    def _copy(self) -> digraph:
        """Not itself a ported C++ symbol -- `randomized_topological_sort`
        takes its `digraph g` argument BY VALUE (digraph.h:89), so it mutates
        its own copy, leaving the caller's graph untouched. Python passes
        objects by reference, so the function below makes this copy itself
        before mutating; kept as a private helper here since nothing else
        needs a `digraph` copy constructor.
        """
        clone = digraph(self.n_nodes())
        clone._incoming = [set(s) for s in self._incoming]
        clone._outgoing = [set(s) for s in self._outgoing]
        return clone


def randomized_topological_sort(g: digraph, out: list[int]) -> list[int]:
    """C++: opencog/util/digraph.h:89 randomized_topological_sort

    Appends the sorted node order to `out` (standing in for the C++ output
    iterator) and also returns it, for convenience. `g` is copied first --
    see `digraph._copy`'s docstring -- so the caller's graph is left intact,
    matching the C++ by-value parameter.

    See the module docstring's DIVERGENCE note for what the ready-queue
    shuffle below actually reads: classic's unseeded `std::rand()` stream,
    reproduced by `_CStdlibRand`, NOT `randGen()`.
    """
    g = g._copy()
    n = g.n_nodes()
    nodes = list(range(n))

    # `std::random_shuffle(nodes.begin(), nodes.end())` -- see the module
    # docstring for why this reads a separate, unseeded stream.
    _random_shuffle(nodes, _cstdlib_rand())

    q: deque[int] = deque()
    for node in nodes:
        if not g.incoming(node):
            q.append(node)

    while q:
        src = q.popleft()
        out.append(src)
        for dst in list(g.outgoing(src)):
            g.erase(src, dst)
            if not g.incoming(dst):
                q.append(dst)

    # DIVERGENCE(C++ opencog/util/digraph.h:117): OC_ASSERT becomes AssertionError.
    if not g.empty():
        raise AssertionError("digraph - g must be a DAG.")
    return out
