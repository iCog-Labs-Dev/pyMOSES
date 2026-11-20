"""NOT A PORT: the IT linkage study (BACKLOG 13.9). The position tree of each genome, the
node-proximity distance of FOS mode ``np`` on it, and the structural relation of two positions
(for the telemetry).

Plan section 5.4. Each genome has an implicit tree, given as a parent array over its nodes
(``-1`` for the root), and a map from each genome position to its node. The distance d(i, j)
is the number of edges between the nodes of positions i and j, through their lowest common
ancestor. This is the definition of ``pymoses.gomea.node_proximity.tree_distances``.

* Template: the full binary tree itself. Position p is node p; the parent of p > 0 is
  (p - 1) // 2.
* IT: root, then the term slots, then the factors of each term, then the two boxes of each
  factor. The genome positions are the boxes (the leaves). So d = 2 for the two boxes of one
  factor, 4 for two boxes of one term in different factors, and 6 for boxes of different terms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def template_tree(height: int) -> tuple[list[int], list[int]]:
    """``(parent, node_of)`` of the template of ``height``: a full binary tree in
    breadth-first order; position p is node p."""
    n = 2 ** (height + 1) - 1
    parent = [-1] + [(p - 1) // 2 for p in range(1, n)]
    return parent, list(range(n))


def it_tree(terms: int, factors: int) -> tuple[list[int], list[int]]:
    """``(parent, node_of)`` of the IT genome: node 0 is the root, nodes 1 .. terms are the
    term slots, then one node for each factor, then one node for each box. Position
    ``(t * factors + f) * 2 + b`` is box ``b`` of factor ``f`` of term ``t``."""
    parent = [-1]
    term_node = []
    for _ in range(terms):
        term_node.append(len(parent))
        parent.append(0)
    factor_node = []
    for t in range(terms):
        for _ in range(factors):
            factor_node.append(len(parent))
            parent.append(term_node[t])
    node_of = []
    for fn in factor_node:
        for _ in range(2):
            node_of.append(len(parent))
            parent.append(fn)
    return parent, node_of


def _depths(parent: Sequence[int]) -> list[int]:
    depth = [0] * len(parent)
    for v in range(len(parent)):
        d, u = 0, v
        while parent[u] >= 0:
            u = parent[u]
            d += 1
        depth[v] = d
    return depth


def node_distance(parent: Sequence[int], depth: Sequence[int], a: int, b: int) -> int:
    """The number of edges between nodes ``a`` and ``b``: depth(a) + depth(b) - 2 depth(lca)."""
    da, db = depth[a], depth[b]
    u, v = a, b
    while da > db:
        u, da = parent[u], da - 1
    while db > da:
        v, db = parent[v], db - 1
    while u != v:
        u, v = parent[u], parent[v]
        da -= 1
    return depth[a] + depth[b] - 2 * da


def distance_matrix(parent: Sequence[int], node_of: Sequence[int]) -> list[list[int]]:
    """d(i, j) between every pair of genome positions (symmetric, 0 on the diagonal)."""
    depth = _depths(parent)
    n = len(node_of)
    return [[node_distance(parent, depth, node_of[i], node_of[j]) for j in range(n)]
            for i in range(n)]


def distance_fn(D: Sequence[Sequence[int]]) -> Callable[[Sequence[int]], list[list[int]]]:
    """The ``distance`` input of ``LinkageModel`` mode ``"np"``: a function of genome
    positions that gives the rows and columns of ``D`` for them, in the same order."""
    return lambda positions: [[int(D[i][j]) for j in positions] for i in positions]


def template_related(p: int, q: int) -> bool:
    """Template: positions ``p`` and ``q`` are parent and child, or siblings."""
    if p == q:
        return False
    pp = (p - 1) // 2 if p > 0 else -1
    pq = (q - 1) // 2 if q > 0 else -1
    return pp == q or pq == p or (p > 0 and q > 0 and pp == pq)


def it_related(p: int, q: int, factors: int) -> bool:
    """IT: positions ``p`` and ``q`` are in the same term slot (this includes the same
    factor)."""
    return p != q and p // (2 * factors) == q // (2 * factors)
