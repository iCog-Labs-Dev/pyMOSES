"""NOT A PORT, server round 12 (T2.1, ``--gomea-fos np``): the tree distance between the
knobs of a deme, for the node-proximity linkage of Koch, Alderliesten and Bosman (GECCO
2026, arXiv 2602.02311; ``docs/gomea/BACKLOG.md`` section 8, A1).

The distance d(i, j) between genes i and j is the number of edges between the exemplar
nodes of their knobs in the decorated exemplar (the exemplar with the knob nodes in it):
the path through the lowest common ancestor of the two nodes. It uses no population
data. ``LinkageModel`` (mode ``"np"``) builds a UPGMA linkage tree on S = -d.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pymoses.gomea.schema import FrozenSchema
from pymoses.util.tree import tree_node


def gene_nodes(schema: FrozenSchema) -> list[tree_node]:
    """The exemplar node of the knob of each gene of ``schema``, in gene order.

    The representation maps the node of each knob to its field
    (``representation._it_disc_idx``: raw field index; ``_it_contin_idx``: contin index),
    as ``representation.get_candidate`` and ``FrozenSchema.candidate`` read them. A contin
    gene ``p`` (``--gomea-contin copy``) is contin index ``p``. A disc or bit gene comes
    after the contin genes, one for each raw field: gene ``n_contin + raw - r0``, with
    ``r0`` the number of contin raw fields (as ``FrozenSchema.to_genotype`` orders them).

    Raises ``ValueError`` if a gene has no node, or two genes have the same node.
    """
    rep = schema.rep
    n_contin = len(schema.contin_positions)
    r0 = schema.fields.n_contin_fields()
    nodes: list[tree_node | None] = [None] * schema.dim
    entries = [(idx, node) for node, idx in rep._it_contin_idx.items()]
    entries += [(n_contin + raw - r0, node) for node, raw in rep._it_disc_idx.items()]
    for gene, node in entries:
        if not 0 <= gene < schema.dim or nodes[gene] is not None:
            raise ValueError(f"knob node map gives gene {gene} twice or out of range")
        nodes[gene] = node
    missing = [g for g, node in enumerate(nodes) if node is None]
    if missing:
        raise ValueError(f"genes {missing} have no knob node in the exemplar")
    return nodes  # type: ignore[return-value]


def tree_distances(nodes: Sequence[tree_node]) -> list[list[int]]:
    """The number of edges between each pair of ``nodes`` (nodes of one tree).

    d(a, b) = depth(a) + depth(b) - 2 depth(lca(a, b)). Each node walks up to the root
    once. At each ancestor ``v``, the nodes below ``v`` are put in groups by the child of
    ``v`` that leads to them (``v`` itself is its own group). Two nodes in different
    groups of ``v`` have ``v`` as their lowest common ancestor, so each pair gets its
    value once. The matrix is symmetric, with 0 on the diagonal.
    """
    n = len(nodes)
    depth: list[int] = [0] * n
    groups: dict[tree_node, dict[tree_node | None, list[int]]] = {}
    node_depth: dict[tree_node, int] = {}
    for k, node in enumerate(nodes):
        path = [node]
        while path[-1].parent is not None:
            path.append(path[-1].parent)
        top = len(path) - 1
        depth[k] = top
        groups.setdefault(node, {}).setdefault(None, []).append(k)
        for i in range(1, len(path)):
            v = path[i]
            node_depth[v] = top - i
            groups.setdefault(v, {}).setdefault(path[i - 1], []).append(k)
        node_depth[node] = top
    D = [[0] * n for _ in range(n)]
    for v, by_child in groups.items():
        dv2 = 2 * node_depth[v]
        parts = list(by_child.values())
        # Nodes in the same group of ``v`` meet below ``v`` (or are the same node, which
        # stays at 0); only pairs across two groups get their value here.
        for i, part in enumerate(parts):
            for other in parts[i + 1 :]:
                for a in part:
                    row = D[a]
                    da = depth[a] - dv2
                    for b in other:
                        x = da + depth[b]
                        row[b] = x
                        D[b][a] = x
    return D


def distance_fn(schema: FrozenSchema) -> Callable[[Sequence[int]], list[list[int]]]:
    """The distance input of ``LinkageModel`` mode ``"np"`` for ``schema``: a function of
    genotype positions that gives their ``tree_distances`` matrix, in the same order."""
    nodes = gene_nodes(schema)
    return lambda positions: tree_distances([nodes[p] for p in positions])
