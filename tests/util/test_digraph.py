"""Oracle and guard tests for pymoses.util.digraph.

The oracle fixture (tests/fixtures/univariate_rng.json) is produced by
tools/cpp_probe/univariate_rng_probe.cc against the REAL
`opencog::randomized_topological_sort` and the REAL `std::rand()` (see
tools/capture_univariate_rng.py) -- not reimplemented by hand. The sections
this file consumes:

  * "rand_default"  -- 512 values of the process-start `std::rand()` stream,
    i.e. exactly the stream `digraph.h:96`'s bare `std::random_shuffle`
    consumes in a classic `moses` process, which never calls `srand`.
  * "rand_srand1" / "rand_srand0" / "rand_seeded" -- the same generator from
    explicit seeds, including glibc's `seed == 0 -> 1` special case.
  * "topo_sort_edgeless" -- 12 consecutive `randomized_topological_sort`
    calls at each of 15 graph sizes, from a freshly `srand(1)`'d state: the
    shape `local_structure_model`'s constructor builds for a Boolean-only
    field_set, where the emitted order IS the shuffle's permutation.
  * "topo_sort_chains" -- the same, for graphs made of dependency chains,
    the shape a contin field's `_initial_deps.insert(base+i-1, base+i)`
    builds.

Every comparison below is ELEMENT FOR ELEMENT against those sections; a
set-wise or "is a permutation" check would pass for any shuffle at all,
which is precisely the gap this module used to have.
"""

import json
from pathlib import Path

import pytest

from pymoses.util.digraph import (
    _cstdlib_rand,
    _CStdlibRand,
    _random_shuffle,
    digraph,
    randomized_topological_sort,
)
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "univariate_rng.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(autouse=True)
def _reset_cstdlib_rand():
    """C's `rand()` state is per-PROCESS and classic runs one process per
    `moses` invocation; this port's is a per-interpreter singleton (see
    digraph.py's module docstring, last paragraph). Reset it around every
    test here so the file is order-independent -- the same discipline
    CARRY-FORWARD.md 5.11 demands of any in-process CLI harness.
    """
    _cstdlib_rand().srand(1)
    yield
    _cstdlib_rand().srand(1)


# --------------------------------------------------------------- digraph itself


def test_insert_and_query():
    g = digraph(3)
    g.insert(0, 1)
    g.insert(0, 2)
    assert g.outgoing(0) == {1, 2}
    assert g.incoming(1) == {0}
    assert g.incoming(2) == {0}
    assert g.n_edges() == 2
    assert not g.empty()


def test_erase_removes_both_directions():
    g = digraph(2)
    g.insert(0, 1)
    g.erase(0, 1)
    assert g.outgoing(0) == set()
    assert g.incoming(1) == set()
    assert g.empty()


# --------------------------------------------------------------- glibc rand()


@pytest.mark.oracle
def test_default_state_reproduces_the_real_std_rand_stream():
    """The whole point: a fresh `_CStdlibRand()` must sit exactly where a
    fresh `moses` process's `rand()` sits, because classic never calls
    `srand`. Compared element for element over all 512 recorded draws.
    """
    rand = _CStdlibRand()
    got = [rand.rand() for _ in range(len(_GOLDEN["rand_default"]))]
    assert got == _GOLDEN["rand_default"]


@pytest.mark.oracle
def test_process_default_state_equals_srand_1():
    """glibc initialises its `unsafe_state` to the TYPE_3 table `randtbl`,
    which IS the post-warm-up state for seed 1. Measured on the C++ side
    (the two fixture sections are equal there) and reproduced here, so the
    singleton's `srand(1)` default is grounded rather than assumed.
    """
    assert _GOLDEN["rand_default"] == _GOLDEN["rand_srand1"]
    rand = _CStdlibRand(1)
    got = [rand.rand() for _ in range(len(_GOLDEN["rand_srand1"]))]
    assert got == _GOLDEN["rand_srand1"]


@pytest.mark.oracle
def test_seed_zero_is_remapped_to_one():
    """`__srandom_r`: `if (seed == 0) seed = 1;`. Dropping that line is a
    silent one-seed bug, so it is pinned against the real C++ output.
    """
    rand = _CStdlibRand(0)
    got = [rand.rand() for _ in range(len(_GOLDEN["rand_srand0"]))]
    assert got == _GOLDEN["rand_srand0"]
    assert got == _GOLDEN["rand_srand1"][: len(got)]


@pytest.mark.oracle
@pytest.mark.parametrize("seed", sorted(_GOLDEN["rand_seeded"], key=int))
def test_explicit_seeds_reproduce_the_real_stream(seed):
    """Includes 2147483647 and 4294967295, which store a NEGATIVE `int32_t`
    into `state[0]` and so exercise `_c_divmod`'s truncate-toward-zero
    behaviour and `_int32`'s wrap -- the two places Python's own semantics
    differ from C's.
    """
    expected = _GOLDEN["rand_seeded"][seed]
    rand = _CStdlibRand(int(seed))
    got = [rand.rand() for _ in range(len(expected))]
    assert got == expected


def test_rand_stays_in_the_documented_range():
    """`rand()` returns `[0, RAND_MAX]` with `RAND_MAX == 2147483647`
    (`/usr/include/stdlib.h:91`); the generator's own `val >> 1` on a
    uint32 is what guarantees it.
    """
    rand = _CStdlibRand()
    for _ in range(2000):
        value = rand.rand()
        assert 0 <= value <= 2147483647


# ------------------------------------------------- random_shuffle + topo sort


@pytest.mark.oracle
@pytest.mark.parametrize("size", sorted(_GOLDEN["topo_sort_edgeless"], key=int))
def test_edgeless_topological_sort_matches_classic(size):
    """An edgeless `digraph` is what a Boolean-only field_set produces, so
    the emitted order is exactly `std::random_shuffle`'s permutation. All 12
    consecutive calls are compared, in order, off ONE `srand(1)` state --
    which also pins that each call consumes exactly `n - 1` draws.
    """
    expected = _GOLDEN["topo_sort_edgeless"][size]
    _cstdlib_rand().srand(1)
    got = [randomized_topological_sort(digraph(int(size)), []) for _ in expected]
    assert got == expected


@pytest.mark.oracle
@pytest.mark.parametrize("key", sorted(_GOLDEN["topo_sort_chains"]))
def test_chained_topological_sort_matches_classic(key):
    """The shape a contin field builds: `_initial_deps.insert(idx_base + i -
    1, idx_base + i)` (local_structure.h:240) makes one chain per contin
    field. The key encodes `<n_nodes>_<start>.<len>...`.
    """
    expected = _GOLDEN["topo_sort_chains"][key]
    parts = key.split("_")
    n_nodes = int(parts[0])
    chains = [tuple(int(x) for x in p.split(".")) for p in parts[1:]]
    _cstdlib_rand().srand(1)
    got = []
    for _ in expected:
        g = digraph(n_nodes)
        for start, length in chains:
            for i in range(1, length):
                g.insert(start + i - 1, start + i)
        got.append(randomized_topological_sort(g, []))
    assert got == expected


def test_random_shuffle_is_the_forward_biased_loop_not_a_backward_one():
    """Mutation guard. libstdc++'s `random_shuffle` walks i = 1..n-1 and
    swaps with `rand() % (i + 1)`. The textbook BACKWARD Fisher-Yates
    (i = n-1..1) consumes the same number of draws from the same stream and
    still produces a permutation -- so nothing but an element-for-element
    comparison can tell them apart. This shows they genuinely differ on
    this stream, so the oracle tests above are not vacuous.
    """
    forward = list(range(12))
    _random_shuffle(forward, _CStdlibRand(1))

    backward = list(range(12))
    rand = _CStdlibRand(1)
    for i in range(len(backward) - 1, 0, -1):
        j = rand.rand() % (i + 1)
        if i != j:
            backward[i], backward[j] = backward[j], backward[i]

    assert forward != backward
    assert forward == _GOLDEN["topo_sort_edgeless"]["12"][0]


def test_random_shuffle_on_a_short_range_draws_nothing():
    """`if (__first == __last) return;` -- and the loop starts at
    `first + 1`, so a one-element range also draws nothing.
    """
    rand = _CStdlibRand(1)
    _random_shuffle([], rand)
    _random_shuffle([7], rand)
    assert rand.rand() == _GOLDEN["rand_default"][0]


# --------------------------------------------------------------- invariants


def test_topological_sort_respects_a_chain():
    g = digraph(4)
    g.insert(0, 1)
    g.insert(1, 2)
    g.insert(2, 3)
    order = randomized_topological_sort(g, [])
    assert sorted(order) == [0, 1, 2, 3]
    assert order.index(0) < order.index(1) < order.index(2) < order.index(3)


def test_topological_sort_does_not_mutate_the_caller_graph():
    """C++ takes `digraph g` BY VALUE (digraph.h:89) -- the caller's graph
    must be untouched after the call.
    """
    g = digraph(3)
    g.insert(0, 1)
    g.insert(1, 2)
    randomized_topological_sort(g, [])
    assert g.outgoing(0) == {1}
    assert g.outgoing(1) == {2}
    assert not g.empty()


def test_topological_sort_with_no_edges_is_a_permutation():
    g = digraph(5)
    order = randomized_topological_sort(g, [])
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_the_shuffle_stream_is_NOT_randgen_and_ignores_the_r_seed():
    """The corrected DIVERGENCE: this shuffle reads C's `rand()`, which
    `-r <seed>` cannot reach -- in classic OR here. Re-seeding `randGen()`
    must therefore leave the order alone, while letting the C stream carry
    on must change it. (Before this was fixed the port drew from
    `randGen()` here, and this test asserted the opposite.)
    """
    g = digraph(8)

    _cstdlib_rand().srand(1)
    randGen().seed(10)
    order_a = randomized_topological_sort(g, [])

    _cstdlib_rand().srand(1)
    randGen().seed(11)
    order_b = randomized_topological_sort(g, [])
    assert order_a == order_b

    order_c = randomized_topological_sort(g, [])  # stream carried on
    assert order_a != order_c


def test_cycle_raises_assertion_error():
    """C++: OC_ASSERT(g.empty(), "digraph - g must be a DAG.") -- a graph
    with a cycle can never fully drain via the incoming-degree-zero queue.
    """
    g = digraph(2)
    g.insert(0, 1)
    g.insert(1, 0)
    with pytest.raises(AssertionError):
        randomized_topological_sort(g, [])


def test_mutation_removing_dag_assertion_silently_drops_cyclic_nodes():
    """Mutation proof: without the trailing `if not g.empty(): raise`
    guard, a cyclic graph would silently return an INCOMPLETE order (the
    cyclic nodes are never enqueued, since their in-degree never reaches
    zero) instead of failing loudly. Demonstrated directly by replicating
    the function's own BFS body without the final check.
    """
    from collections import deque

    g = digraph(2)
    g.insert(0, 1)
    g.insert(1, 0)
    g = g._copy()
    nodes = list(range(g.n_nodes()))
    q: deque[int] = deque(n for n in nodes if not g.incoming(n))
    out: list[int] = []
    while q:
        src = q.popleft()
        out.append(src)
        for dst in list(g.outgoing(src)):
            g.erase(src, dst)
            if not g.incoming(dst):
                q.append(dst)
    # Without the DAG assertion, the mutant would return an incomplete
    # (silently wrong) order rather than raising.
    assert out == []
    assert not g.empty()
