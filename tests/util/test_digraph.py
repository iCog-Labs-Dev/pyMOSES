"""Unit tests for pymetta.util.digraph.

No C++ probe: `digraph`/`randomized_topological_sort` are exercised
end-to-end (through `local_structure_model`'s constructor) by
`test_local_structure.py`. This file pins the class's own small API and
the topological-order INVARIANT (every node appears after all its
dependencies) directly, plus the module's own documented RNG-source
DIVERGENCE.
"""

import pytest

from pymetta.util.digraph import digraph, randomized_topological_sort
from pymetta.util.rng import randGen


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


def test_topological_sort_respects_a_chain():
    randGen().seed(1)
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
    randGen().seed(2)
    g = digraph(3)
    g.insert(0, 1)
    g.insert(1, 2)
    randomized_topological_sort(g, [])
    assert g.outgoing(0) == {1}
    assert g.outgoing(1) == {2}
    assert not g.empty()


def test_topological_sort_with_no_edges_is_a_permutation():
    randGen().seed(3)
    g = digraph(5)
    order = randomized_topological_sort(g, [])
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_topological_sort_draws_from_randgen_not_a_disconnected_stream():
    """Module DIVERGENCE marker: this port's shuffle draws from
    `randGen()` (unlike classic's own unseeded `std::rand()`). Proven by
    observing that re-seeding `randGen()` changes the resulting order for
    a graph with no edges (a pure permutation of the shuffle itself).
    """
    g = digraph(8)
    randGen().seed(10)
    order_a = randomized_topological_sort(g, [])
    randGen().seed(11)
    order_b = randomized_topological_sort(g, [])
    assert order_a != order_b


def test_cycle_raises_assertion_error():
    """C++: OC_ASSERT(g.empty(), "digraph - g must be a DAG.") -- a graph
    with a cycle can never fully drain via the incoming-degree-zero queue.
    """
    randGen().seed(4)
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
