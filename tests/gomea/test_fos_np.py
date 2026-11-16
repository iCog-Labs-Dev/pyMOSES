"""Server round 12 (T2.1): node-proximity linkage (``--gomea-fos np``).

The FOS is a UPGMA linkage tree, built as ``lt-b`` builds it, on S(i, j) = -d(i, j), where
d is the number of edges between the exemplar nodes of the knobs of genes i and j. It is
built once for each deme. See ``pymoses.gomea.node_proximity`` and ``fos.LinkageModel``.
"""

from __future__ import annotations

import argparse
import random

import pytest
from diagnostic_objectives import CountingEvaluator, onemax, random_population

import pymoses.gomea.engine as engine_mod
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea import gomea_parameters
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel, linkage_tree_fos, select_fos_topk
from pymoses.gomea.node_proximity import distance_fn, gene_nodes, tree_distances
from pymoses.gomea.optimizer import FOS_MODES
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.main.problem_params import _GOMEA_FOS_CHOICES, _gomea_params_from_args
from pymoses.moses.representation.field_set import contin_spec
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import MT19937RandGen
from pymoses.util.tree import tree_node
from tests.gomea.test_fos_np_identity import _base, _reg
from tests.gomea.test_integration import _cli_parser, _events, _run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_tree(parents: list[int | None]) -> list[tree_node]:
    """Nodes linked by ``parents`` (``parents[k]`` is the index of the parent of node k,
    ``None`` for the root). Only ``parent`` is set: ``tree_distances`` reads only it."""
    nodes = [tree_node(k) for k in range(len(parents))]
    for k, p in enumerate(parents):
        nodes[k].parent = None if p is None else nodes[p]
    return nodes


def _brute_distance(a: tree_node, b: tree_node) -> int:
    def path(v):
        out = [v]
        while out[-1].parent is not None:
            out.append(out[-1].parent)
        return out[::-1]

    pa, pb = path(a), path(b)
    common = 0
    while common < min(len(pa), len(pb)) and pa[common] is pb[common]:
        common += 1
    return (len(pa) - common) + (len(pb) - common)


def _random_parents(rng: random.Random, n: int) -> list[int | None]:
    return [None] + [rng.randrange(k) for k in range(1, n)]


def _contin_rep(arity: int = 2) -> representation:
    return representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, arity),
        linear_contin=True,
    )


def _mixed_rep() -> representation:
    """Contin inputs, a Boolean output: contin genes and disc genes in one schema."""
    return representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("0<($1)"),
        gen_signature(type_node.contin_type, type_node.boolean_type, 2),
        linear_contin=True,
        perm_ratio=-1.0,
    )


def _all_varying_population(d: int, n: int = 8) -> list[tuple[int, ...]]:
    """A binary population in which every position varies."""
    rows = [tuple((r >> (k % 3) ^ k) & 1 for k in range(d)) for r in range(n)]
    rows[0] = (0,) * d
    rows[1] = (1,) * d
    return rows


def _np_model(nodes: list[tree_node], seed: int = 5) -> LinkageModel:
    return LinkageModel(
        "np", MT19937RandGen(seed), distance=lambda ps: tree_distances([nodes[p] for p in ps])
    )


def _check_linkage_tree(fos: list[list[int]], positions: list[int]) -> None:
    """A valid linkage tree without its root: the singletons, then merges, each the union
    of two disjoint current clusters; 2d - 2 subsets; the full set is not in it."""
    d = len(positions)
    assert len(fos) == 2 * d - 2
    assert sorted(s[0] for s in fos[:d]) == sorted(positions)
    assert all(len(s) == 1 for s in fos[:d])
    clusters = [frozenset(s) for s in fos[:d]]
    for merged in fos[d:]:
        m = frozenset(merged)
        assert len(m) == len(merged)
        parts = [c for c in clusters if c <= m]
        assert len(parts) == 2 and parts[0] | parts[1] == m and not parts[0] & parts[1]
        clusters = [c for c in clusters if c not in parts] + [m]
    assert frozenset(positions) not in {frozenset(s) for s in fos}
    assert len(clusters) == 2


# ---------------------------------------------------------------------------
# Tree distances
# ---------------------------------------------------------------------------


def test_tree_distances_on_a_known_tree():
    # +(*($1 $2) sin($3) $4): the root has three children; * has two, sin has one.
    tr = str_to_combo_tree("+(*($1 $2) sin($3) $4)")
    root = tr.begin()
    times, sin, x4 = list(tr.children(root))
    x1, x2 = list(tr.children(times))
    (x3,) = list(tr.children(sin))
    nodes = [x1, x2, x3, x4, times, root]
    D = tree_distances(nodes)
    expected = [
        # x1 x2 x3 x4 *  +
        [0, 2, 4, 3, 1, 2],  # x1
        [2, 0, 4, 3, 1, 2],  # x2
        [4, 4, 0, 3, 3, 2],  # x3
        [3, 3, 3, 0, 2, 1],  # x4
        [1, 1, 3, 2, 0, 1],  # *
        [2, 2, 2, 1, 1, 0],  # +
    ]
    assert D == expected


@pytest.mark.parametrize("seed", range(20))
def test_tree_distances_match_a_brute_force_on_random_trees(seed):
    rng = random.Random(seed)
    nodes = _node_tree(_random_parents(rng, rng.randint(1, 40)))
    chosen = rng.sample(nodes, rng.randint(1, len(nodes)))
    D = tree_distances(chosen)
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            assert D[i][j] == _brute_distance(a, b)


def test_tree_distances_of_the_same_node_twice_is_zero():
    nodes = _node_tree([None, 0, 1])
    assert tree_distances([nodes[2], nodes[2], nodes[0]]) == [[0, 0, 2], [0, 0, 2], [2, 2, 0]]


# ---------------------------------------------------------------------------
# Gene to exemplar node
# ---------------------------------------------------------------------------


def _check_disc_gene(schema: FrozenSchema, gene: int, node: tree_node) -> None:
    """Gene ``gene`` set to 1 sets the raw field of the knob at ``node``, and no other."""
    rep = schema.rep
    raw = rep._it_disc_idx[node]
    g = list(schema.default_genotype())
    g[gene] = 1
    inst = schema.to_instance(g)
    fields = schema.fields
    for k in range(fields.n_contin_fields(), fields.raw_size()):
        assert fields.get_raw(inst, k) == (1 if k == raw else 0)


def _check_contin_gene(schema: FrozenSchema, gene: int, node: tree_node) -> None:
    """Gene ``gene`` with a Right step changes the contin field of the knob at ``node``,
    and no other."""
    rep = schema.rep
    idx = rep._it_contin_idx[node]
    fields = schema.fields
    base = schema.to_instance(schema.default_genotype())
    g = list(schema.default_genotype())
    g[gene] = contin_spec.Right
    inst = schema.to_instance(g)
    for c in range(len(fields.contin())):
        changed = fields.get_contin(inst, c) != fields.get_contin(base, c)
        assert changed == (c == idx)


def test_gene_nodes_boolean(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 3)
    schema = FrozenSchema(rep)
    nodes = gene_nodes(schema)
    assert len(nodes) == schema.dim == len(rep._it_disc_idx)
    assert len({id(n) for n in nodes}) == schema.dim
    for gene, node in enumerate(nodes):
        assert rep._it_disc_idx[node] == gene
        _check_disc_gene(schema, gene, node)


def test_gene_nodes_contin_copy():
    rep = _contin_rep()
    schema = FrozenSchema(rep, contin="copy")
    nodes = gene_nodes(schema)
    assert len(nodes) == schema.dim == len(rep._it_contin_idx)
    for gene, node in enumerate(nodes):
        assert rep._it_contin_idx[node] == gene
        _check_contin_gene(schema, gene, node)


def test_gene_nodes_mixed_contin_and_disc():
    rep = _mixed_rep()
    schema = FrozenSchema(rep, contin="copy")
    n_contin = len(schema.contin_positions)
    assert n_contin > 0 and len(rep._it_disc_idx) > 0
    nodes = gene_nodes(schema)
    assert len(nodes) == schema.dim == n_contin + len(rep._it_disc_idx)
    for gene, node in enumerate(nodes):
        if gene < n_contin:
            _check_contin_gene(schema, gene, node)
        else:
            _check_disc_gene(schema, gene, node)


def test_gene_nodes_refuses_a_knob_map_with_a_missing_gene(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 3)
    schema = FrozenSchema(rep)
    node = next(iter(rep._it_disc_idx))
    del rep._it_disc_idx[node]
    with pytest.raises(ValueError, match="no knob node"):
        gene_nodes(schema)


def test_sibling_knobs_are_two_edges_apart(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 3)
    schema = FrozenSchema(rep)
    nodes = gene_nodes(schema)
    D = distance_fn(schema)(range(schema.dim))
    for i in range(schema.dim):
        for j in range(schema.dim):
            assert D[i][j] == _brute_distance(nodes[i], nodes[j])
            if i != j and nodes[i].parent is nodes[j].parent:
                assert D[i][j] == 2


# ---------------------------------------------------------------------------
# The linkage model
# ---------------------------------------------------------------------------


def test_np_is_a_mode_everywhere():
    assert "np" in LinkageModel.MODES
    assert "np" in FOS_MODES
    assert "np" in _GOMEA_FOS_CHOICES


def test_np_needs_a_distance():
    with pytest.raises(ValueError, match="needs distance"):
        LinkageModel("np", MT19937RandGen(1))


# Root R; A and B under R; knobs 0, 1, 2 under A, knobs 3, 4 under B; knob 5 under R.
_SHAPE = [None, 0, 0, 1, 1, 1, 2, 2, 0]  # R, A, B, k0, k1, k2, k3, k4, k5
_KNOBS = [3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("seed", range(12))
def test_knobs_on_the_same_parent_merge_first(seed):
    """The knobs under A (and under B) form a cluster before any of them joins a knob
    elsewhere, whatever the order: every subset that holds a knob of a sibling group and
    a knob outside it holds the whole group. The merges inside the groups have the
    largest similarity, -2 (two edges)."""
    tree = _node_tree(_SHAPE)
    nodes = [tree[k] for k in _KNOBS]
    d = len(nodes)
    model = _np_model(nodes, seed=seed)
    pop = _all_varying_population(d)
    model.calibrate(pop, [2] * d)
    fos = model.learn(pop)
    _check_linkage_tree(fos, list(range(d)))
    sets = [frozenset(s) for s in fos]
    for group in (frozenset({0, 1, 2}), frozenset({3, 4})):
        assert group in sets
        for s in sets:
            if s & group and s - group:
                assert group <= s
    strengths = model.last_strengths
    assert strengths[:d] == [None] * d
    inside = [v for s, v in zip(sets, strengths, strict=True) if v is not None and len(s) <= 3
              and (s <= {0, 1, 2} or s <= {3, 4})]  # fmt: skip
    assert inside == [-2.0, -2.0, -2.0]
    assert all(v < -2.0 for v in strengths if v is not None and v not in inside)


def test_np_fos_equals_the_lt_b_builder_on_minus_d_with_the_same_order():
    rng = random.Random(3)
    tree = _node_tree(_random_parents(rng, 30))
    nodes = rng.sample(tree, 12)
    d = len(nodes)
    model = _np_model(nodes, seed=11)
    pop = _all_varying_population(d)
    model.calibrate(pop, [2] * d)
    fos = model.learn(pop)
    # The order of lt-b: one Fisher-Yates shuffle on the seeded stream.
    order_rng = MT19937RandGen(11)
    order = list(range(d))
    for i in range(d - 1, 0, -1):
        j = order_rng.randint_n(i + 1)
        order[i], order[j] = order[j], order[i]
    S = [[-float(x) for x in row] for row in tree_distances(nodes)]
    sims: list[float] = []
    expected = linkage_tree_fos(S, order, sims)
    assert fos == expected[:-1]
    assert model.last_strengths == ([None] * d + sims)[: len(expected) - 1]
    _check_linkage_tree(fos, list(range(d)))


@pytest.mark.parametrize("seed", range(10))
def test_np_fos_is_a_valid_linkage_tree_over_the_search_positions(seed):
    rng = random.Random(100 + seed)
    tree = _node_tree(_random_parents(rng, 40))
    d = rng.randint(2, 20)
    nodes = rng.sample(tree, d)
    model = _np_model(nodes, seed=seed)
    pop = _all_varying_population(d)
    # Make two positions constant: they are not search positions.
    const = sorted(rng.sample(range(d), min(2, d - 2)))
    pop = [tuple(0 if k in const else v for k, v in enumerate(row)) for row in pop]
    model.calibrate(pop, [2] * d)
    search = [k for k in range(d) if k not in const]
    assert list(model.search_positions) == search
    _check_linkage_tree(model.learn(pop), search)


def test_np_fos_is_static_and_draws_once():
    tree = _node_tree(_SHAPE)
    nodes = [tree[k] for k in _KNOBS]
    d = len(nodes)
    model = _np_model(nodes, seed=21)
    pop = _all_varying_population(d)
    model.calibrate(pop, [2] * d)
    first = model.learn(pop)
    strengths = list(model.last_strengths)
    other = random_population(MT19937RandGen(0), 8, [2] * d)
    for _ in range(4):
        fos = model.learn(other)
        assert fos == first
        assert model.last_strengths == strengths
        fos[0].append(99)  # the caller gets a copy: the next FOS does not change
        model.last_strengths.append(None)
    assert model.learn(pop) == first and model.last_strengths == strengths
    # The stream made only the d - 1 draws of one shuffle (at the first learn).
    ref = MT19937RandGen(21)
    for i in range(d - 1, 0, -1):
        ref.randint_n(i + 1)
    assert model.rng.randint_n(1 << 30) == ref.randint_n(1 << 30)


def test_np_ignores_symbolize():
    tree = _node_tree(_SHAPE)
    nodes = [tree[k] for k in _KNOBS]
    d = len(nodes)
    pop = _all_varying_population(d)
    plain = _np_model(nodes, seed=4)
    plain.calibrate(pop, [2] * d)
    sym = LinkageModel(
        "np",
        MT19937RandGen(4),
        lambda p: [tuple(0 for _ in row) for row in p],  # every position looks constant
        distance=lambda ps: tree_distances([nodes[p] for p in ps]),
    )
    sym.calibrate(pop, [2] * d)
    assert sym.learn(pop) == plain.learn(pop)


def test_np_learn_subset_is_refused():
    tree = _node_tree(_SHAPE)
    nodes = [tree[k] for k in _KNOBS]
    model = _np_model(nodes)
    pop = _all_varying_population(len(nodes))
    model.calibrate(pop, [2] * len(nodes))
    with pytest.raises(ValueError, match="knob cap"):
        model.learn_subset(pop, [0, 1, 2])


def test_np_with_topk_linkage_keeps_the_closest_merges():
    tree = _node_tree(_SHAPE)
    nodes = [tree[k] for k in _KNOBS]
    d = len(nodes)
    model = _np_model(nodes)
    pop = _all_varying_population(d)
    model.calibrate(pop, [2] * d)
    fos = model.learn(pop)
    sel = select_fos_topk(len(fos), model.last_strengths, 3, "linkage", MT19937RandGen(1))
    a, b = frozenset({0, 1, 2}), frozenset({3, 4})
    assert all(frozenset(fos[i]) <= a or frozenset(fos[i]) <= b for i in sel)
    assert all(len(fos[i]) > 1 for i in sel)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class _RecordingNp(LinkageModel):
    log: list[list[list[int]]] = []

    def learn(self, pop):
        fos = super().learn(pop)
        type(self).log.append(fos)
        return fos


def test_engine_learns_the_same_fos_in_every_generation(monkeypatch):
    rng = random.Random(9)
    tree = _node_tree(_random_parents(rng, 40))
    d = 16
    nodes = rng.sample(tree, d)
    monkeypatch.setattr(engine_mod, "LinkageModel", _RecordingNp)
    _RecordingNp.log = []
    ev = CountingEvaluator(onemax(2))
    pop = [ev.individual(g) for g in random_population(MT19937RandGen(31), 24, [3] * d)]
    eng = GomeaEngine(
        [3] * d,
        EngineParams(fos_mode="np", max_generations=6, stall_generations=100),
        ev,
        MT19937RandGen(32),
        MT19937RandGen(33),
        distance=lambda ps: tree_distances([nodes[p] for p in ps]),
    )
    res = eng.run(pop)
    log = _RecordingNp.log
    assert res.generations >= 2 and len(log) == res.generations
    assert all(f == log[0] for f in log)
    _check_linkage_tree(log[0], list(eng.model.search_positions))


def test_engine_without_distance_refuses_np():
    ev = CountingEvaluator(onemax(2))
    pop = [ev.individual(g) for g in random_population(MT19937RandGen(1), 8, [3] * 6)]
    eng = GomeaEngine(
        [3] * 6, EngineParams(fos_mode="np"), ev, MT19937RandGen(2), MT19937RandGen(3)
    )
    with pytest.raises(ValueError, match="needs distance"):
        eng.run(pop)


# ---------------------------------------------------------------------------
# Parameters and the command line
# ---------------------------------------------------------------------------


def test_parameters_accept_np_and_topk_linkage():
    assert gomea_parameters(fos="np").fos == "np"
    gp = gomea_parameters(fos="np", fos_topk=3, fos_topk_mode="linkage")
    assert (gp.fos, gp.fos_topk) == ("np", 3)
    assert gomea_parameters(fos="np").as_dict()["fos"] == "np"


def test_parameters_refuse_np_with_the_knob_cap():
    with pytest.raises(ValueError, match="knob_cap"):
        gomea_parameters(fos="np", knob_cap=8)


def test_cli_accepts_np():
    gp = _gomea_params_from_args(_cli_parser(True).parse_args(["--gomea-fos", "np"]))
    assert gp.fos == "np"
    gp = _gomea_params_from_args(
        _cli_parser(True).parse_args(["--gomea-fos", "np", "--gomea-fos-topk", "4"])
    )
    assert (gp.fos, gp.fos_topk, gp.fos_topk_mode) == ("np", 4, "linkage")


@pytest.mark.parametrize(
    "argv",
    [
        ["--gomea-fos", "np", "--gomea-knob-cap", "8"],
        ["--gomea-fos", "npx"],
    ],
)
def test_cli_refuses(argv):
    with pytest.raises(ValueError):
        _gomea_params_from_args(_cli_parser(True).parse_args(argv))


def test_cli_default_is_still_lt_b():
    assert _gomea_params_from_args(argparse.Namespace()).fos == "lt-b"
    assert _gomea_params_from_args(_cli_parser(True).parse_args([])).fos == "lt-b"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _deme_fos(events: list[dict]) -> list[list[tuple]]:
    """For each deme, the (fos_size, fos_mean_subset_size) of each generation."""
    demes: list[list[tuple]] = []
    for e in events:
        if e["type"] == "invocation_start":
            demes.append([])
        elif e["type"] == "generation":
            demes[-1].append((e["fos_size"], e["fos_mean_subset_size"]))
    return demes


def test_end_to_end_boolean_np(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "g.jsonl"
    argv = ["-H", "mux", "-k", "2", "-r", "5", "-m", "6000", "-a", "gomea",
            "--gomea-fos", "np", "--gomea-pop-size", "dynamic", "--hc-fraction-of-nn",
            "0.05", "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "0",
            "--gomea-genotype-memo", "1", "--knob-probe-baseline", "reduced",
            "--gomea-log", str(log)]  # fmt: skip
    rc, stdout = _run(argv)
    assert rc == 0 and stdout
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert starts and all(e["params"]["fos"] == "np" for e in starts)
    demes = _deme_fos(events)
    assert any(len(g) >= 2 for g in demes)
    for gens in demes:
        assert len(set(gens)) <= 1


def test_end_to_end_regression_np_with_contin_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "g.jsonl"
    argv = _reg("1500") + _base("np") + ["--gomea-log", str(log)]
    rc, stdout = _run(argv)
    assert rc == 0 and stdout
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert starts and all(e["params"]["fos"] == "np" for e in starts)
    assert all(e["params"]["contin"] == "copy" for e in starts)
    for gens in _deme_fos(events):
        assert len(set(gens)) <= 1


def test_end_to_end_np_with_topk_linkage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "g.jsonl"
    argv = ["-H", "mux", "-k", "2", "-r", "5", "-m", "3000", "-a", "gomea",
            "--gomea-fos", "np", "--gomea-fos-topk", "3", "--gomea-log", str(log)]  # fmt: skip
    rc, stdout = _run(argv)
    assert rc == 0 and stdout
    gens = [e for e in _events(log) if e["type"] == "generation"]
    assert gens and any(e["fos_size"] > 3 for e in gens)
    for e in gens:
        assert e["fos_effective_size"] == min(3, e["fos_size"])
        # A tree over d positions has 2d - 2 subsets (no root) and d - 2 merges.
        merges = (e["fos_size"] + 2) // 2 - 2
        picked = [s for s in e["fos_selected_strength"] if s is not None]
        assert len(picked) == min(3, merges)
        assert all(s < 0 for s in picked)  # minus the mean distance
