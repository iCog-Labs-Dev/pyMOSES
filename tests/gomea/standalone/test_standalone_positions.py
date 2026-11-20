"""The IT linkage study (BACKLOG 13.9, plan sections 5.4 and 8.1): the position trees and the
node-proximity distance of FOS mode ``np`` on both genomes, checked against
``pymoses.gomea.node_proximity.tree_distances``."""

from __future__ import annotations

import random

from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.node_proximity import tree_distances
from pymoses.gomea.standalone import positions
from pymoses.util.rng import MT19937RandGen
from pymoses.util.tree import tree_node


def _nodes(parent):
    nodes = [tree_node(k) for k in range(len(parent))]
    for k, p in enumerate(parent):
        if p >= 0:
            nodes[k].parent = nodes[p]
    return nodes


def _check_against_node_proximity(parent, node_of):
    D = positions.distance_matrix(parent, node_of)
    nodes = _nodes(parent)
    assert D == tree_distances([nodes[v] for v in node_of])
    return D


def test_template_distances():
    parent, node_of = positions.template_tree(4)
    assert len(node_of) == 31
    D = _check_against_node_proximity(parent, node_of)
    assert D[0][1] == 1 and D[1][2] == 2 and D[3][4] == 2 and D[3][5] == 4
    assert D[15][30] == 8 and D[15][16] == 2 and D[0][30] == 4
    assert all(D[i][i] == 0 for i in range(31))


def test_it_distances_are_2_4_and_6():
    parent, node_of = positions.it_tree(8, 2)
    assert len(node_of) == 32
    D = _check_against_node_proximity(parent, node_of)
    for i in range(32):
        for j in range(32):
            if i == j:
                want = 0
            elif i // 2 == j // 2:
                want = 2  # the two boxes of one factor
            elif i // 4 == j // 4:
                want = 4  # one term, different factors
            else:
                want = 6
            assert D[i][j] == want


def test_a_random_small_tree_agrees_with_node_proximity():
    r = random.Random(11)
    parent = [-1] + [r.randrange(k) for k in range(1, 25)]
    node_of = r.sample(range(25), 12)
    _check_against_node_proximity(parent, node_of)


def test_the_distance_fn_gives_the_rows_of_the_positions():
    parent, node_of = positions.it_tree(2, 2)
    D = positions.distance_matrix(parent, node_of)
    fn = positions.distance_fn(D)
    assert fn([0, 5, 2]) == [[0, 6, 4], [6, 0, 6], [4, 6, 0]]


def test_the_np_fos_on_it_groups_each_factor_then_each_term():
    parent, node_of = positions.it_tree(8, 2)
    fn = positions.distance_fn(positions.distance_matrix(parent, node_of))
    model = LinkageModel("np", MT19937RandGen(3), distance=fn)
    pop = [tuple((i + p) % 3 for p in range(32)) for i in range(3)]
    model.calibrate(pop, [6, 4] * 16)
    fos = {frozenset(s) for s in model.learn(pop)}
    for t in range(8):
        assert frozenset(range(4 * t, 4 * t + 4)) in fos
        for f in range(2):
            assert frozenset({4 * t + 2 * f, 4 * t + 2 * f + 1}) in fos


def test_structural_relations():
    assert positions.template_related(0, 1) and positions.template_related(2, 0)
    assert positions.template_related(3, 4) and not positions.template_related(4, 5)
    assert not positions.template_related(0, 3) and not positions.template_related(1, 1)
    assert positions.it_related(0, 1, 2) and positions.it_related(0, 3, 2)
    assert not positions.it_related(3, 4, 2) and not positions.it_related(5, 5, 2)
