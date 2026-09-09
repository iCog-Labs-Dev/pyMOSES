"""C++: opencog/util/digraph.h:41 digraph

A directed graph over the integers ``[0, n)``, plus a randomized topological
sort. Ported for `local_structure.py`'s `local_structure_model` constructor,
which builds a `digraph` of raw-field dependencies (`_initial_deps`) purely to
feed it to `randomized_topological_sort` and derive a sampling order.

# DIVERGENCE(C++ opencog/util/digraph.h:96): `randomized_topological_sort`
# shuffles its ready-node queue with a BARE `std::random_shuffle(nodes.begin(),
# nodes.end())` -- no RNG argument. Per the C++03/11 standard that overload
# draws from an IMPLEMENTATION-DEFINED internal generator, and on every
# libstdc++ this fork's toolchain targets that is `std::rand()`, seeded (if at
# all) by `::srand()`. `git grep -rn 'srand(' $CLASSIC` (moses/, tests/) finds
# ZERO call sites anywhere in classic MOSES: `opencog::randGen().seed(rand_seed)`
# (problem-params.cc:2116, the `-r` flag) reseeds ONLY the `RandGen`/`mt19937ar`
# singleton this port's own `pymetta.util.rng.randGen()` already reproduces
# bit-for-bit -- it never touches `::rand()`'s own default-seeded global state.
# So classic's `_ordering` here is driven by a SEPARATE, UNSEEDED random stream
# that `-r <seed>` does not and cannot control: the same binary reproduces the
# same shuffle every run (glibc's `rand()` always starts from the same default
# state absent a `srand()` call), but that fixed sequence is a property of the
# glibc/libstdc++ build classic happens to run on, not of MOSES's own seeding
# contract. Reproducing it bit-for-bit would mean porting glibc's non-standard
# additive-feedback `random()` generator (TYPE_3, the algorithm behind `rand()`
# on glibc) for a value classic's own `-r` flag was never able to pin down in
# the first place. This port instead draws the shuffle from
# `pymetta.util.rng.randGen()` -- the ONE RNG stream this whole port promises
# to keep bit-identical -- via an explicit-RNG Fisher-Yates, so that univariate
# search *within this port* is at least reproducible under its own `-r` seed
# (which classic's own build, on this exact point, is not). This is the one
# place in the optimizers this task ports where an extra `randGen()` draw is
# consumed relative to classic's real draw sequence at the corresponding
# point -- see local_structure.py's own module docstring for the consequence
# this has for trajectory-level comparison against classic.
"""

from __future__ import annotations

from collections import deque

from pymetta.util.rng import randGen

__all__ = ["digraph", "randomized_topological_sort"]


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

    See the module docstring's DIVERGENCE note for the one deliberate
    deviation: the ready-queue shuffle below draws from `randGen()` via an
    explicit-RNG Fisher-Yates rather than classic's unseeded `std::rand()`.
    """
    g = g._copy()
    n = g.n_nodes()
    nodes = list(range(n))

    # DIVERGENCE: see module docstring. Fisher-Yates via randGen(), not
    # std::random_shuffle(nodes.begin(), nodes.end()) (unseeded std::rand()).
    rng = randGen()
    for i in range(n - 1, 0, -1):
        j = rng.randint_n(i + 1)
        nodes[i], nodes[j] = nodes[j], nodes[i]

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
