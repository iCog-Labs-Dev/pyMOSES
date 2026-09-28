"""Tests for the opt-in paper-style constant genes (``--gomea-contin copy``).

Each MOSES contin knob becomes ONE categorical gene whose code packs all of
its raw ``{Stop, Left, Right}`` fields. GOM copies whole codes between
individuals, so constants are only ever copied, never mutated (Virgolin et
al. 2021, Sec 6). Linkage learning sees the constants through one of the
paper's three strategies: ``all`` (all-const), ``none`` (no-const) or
``bin`` (bin-const, gamma = 100).
"""

from __future__ import annotations

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.gom import GomStats, Individual, gom
from pymoses.gomea.optimizer import gomea_optimization, gomea_parameters
from pymoses.gomea.population import build_population, build_uniform_population, parse_radii
from pymoses.gomea.schema import CONTIN_MODES, FrozenSchema, UnsupportedRepresentation
from pymoses.gomea.stats import CONTIN_STATS_MODES, contin_symbols
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.field_set import contin_spec
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import MT19937RandGen, randGen


def _contin_rep(arity: int = 1) -> representation:
    return representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, arity),
        linear_contin=True,
    )


def _canonical(schema: FrozenSchema, pos: int, code: int) -> bool:
    """No Left/Right step after the first Stop."""
    depth = schema.contin_depths[pos]
    digits = [(code >> (2 * k)) & 3 for k in range(depth)]
    if 0 in digits:
        return all(d == 0 for d in digits[digits.index(0) :])
    return True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_modes():
    assert CONTIN_MODES == ("reject", "copy")
    assert CONTIN_STATS_MODES == ("all", "none", "bin")


def test_default_mode_still_rejects_contin():
    rep = _contin_rep()
    with pytest.raises(UnsupportedRepresentation):
        FrozenSchema(rep)
    with pytest.raises(UnsupportedRepresentation):
        FrozenSchema(rep, contin="reject")
    with pytest.raises(ValueError):
        FrozenSchema(rep, contin="mutate")


def test_copy_maps_each_contin_knob_to_one_gene():
    rep = _contin_rep()
    fields = rep.fields()
    schema = FrozenSchema(rep, contin="copy")
    n_contin = len(fields.contin())
    assert n_contin > 0
    assert schema.dim == n_contin + len(fields.disc_and_bit()) == fields.dim_size()
    assert schema.contin_positions == tuple(range(n_contin))
    assert schema.contin_depths == tuple(c.depth for c in fields.contin())
    assert schema.domains[:n_contin] == tuple(4**c.depth for c in fields.contin())
    assert schema.default_genotype() == (0,) * schema.dim
    decoded = schema.decode(schema.default_genotype())
    assert combo_tree_to_str(decoded) == combo_tree_to_str(rep.get_clean_exemplar(reduce=True))


def test_discrete_schema_is_unchanged_by_copy_mode(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    a, b = FrozenSchema(rep), FrozenSchema(rep, contin="copy")
    assert (a.dim, a.domains, a.schema_id) == (b.dim, b.domains, b.schema_id)
    assert a.contin_positions == b.contin_positions == ()


def test_contin_genotype_round_trips_sampler_instances_and_values():
    from pymoses.moses.moses.neighborhood_sampling import sample_from_neighborhood

    rep = _contin_rep()
    fields = rep.fields()
    schema = FrozenSchema(rep, contin="copy")
    randGen().seed(3)
    center = schema.to_instance(schema.default_genotype())
    for dist in (1, 2, 3, 5):
        out: list = []
        sample_from_neighborhood(fields, dist, 5, out, 5, center)
        for inst in out:
            g = schema.to_genotype(inst)
            assert schema.to_instance(g) == inst
            for i in schema.contin_positions:
                assert schema.contin_value(i, g[i]) == fields.get_contin(inst, i)


def test_to_instance_rejects_out_of_domain_contin_codes():
    schema = FrozenSchema(_contin_rep(), contin="copy")
    g = list(schema.default_genotype())
    g[0] = schema.domains[0]
    with pytest.raises(ValueError):
        schema.to_instance(g)


# ---------------------------------------------------------------------------
# GOM copies whole contin values
# ---------------------------------------------------------------------------


class _AcceptAll:
    def __init__(self):
        self.calls = 0

    def check(self):
        return None

    def evaluate(self, g):
        self.calls += 1
        return 0

    def phenotype_key(self, g):
        return None

    def worse(self, a, b):
        return a < b

    def record_noop_skip(self):
        pass

    def record_phenotype_skip(self):
        pass


def test_gom_copies_the_donors_whole_contin_value():
    rep = _contin_rep()
    fields = rep.fields()
    schema = FrozenSchema(rep, contin="copy")
    donor_inst = schema.to_instance(schema.default_genotype())
    fields.set_contin(donor_inst, 0, 3.7)
    fields.set_contin(donor_inst, 1, -1.25)
    donor_g = schema.to_genotype(donor_inst)
    begin = fields.contin_to_raw_idx(0)
    depth = fields.contin()[0].depth
    assert sum(fields.get_raw(donor_inst, begin + k) != 0 for k in range(depth)) >= 2

    recipient = Individual(schema.default_genotype(), 0, None)
    donor = Individual(donor_g, 0, None)
    off, changed, _ = gom(
        recipient, [donor], [[0]], _AcceptAll(), MT19937RandGen(1), GomStats(), False
    )
    assert changed
    inst = schema.to_instance(off.genotype)
    assert [fields.get_raw(inst, begin + k) for k in range(depth)] == [
        fields.get_raw(donor_inst, begin + k) for k in range(depth)
    ]
    assert fields.get_contin(inst, 0) == fields.get_contin(donor_inst, 0)
    # Only the mixed gene moved: the donor's second constant stays behind.
    assert fields.get_contin(inst, 1) == fields.contin()[1].mean


# ---------------------------------------------------------------------------
# Linkage statistics over constants
# ---------------------------------------------------------------------------

_VALUES = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.4, 5: 2.5, 6: 10.0}


def _value(pos, code):
    return _VALUES[code]


def test_contin_symbols_all_gives_each_distinct_value_its_own_symbol():
    pop = [(0, 7), (1, 7), (2, 8), (1, 8)]
    out = contin_symbols(pop, (0,), _value, "all")
    assert [row[1] for row in out] == [7, 7, 8, 8]  # discrete genes untouched
    col = [row[0] for row in out]
    assert col[1] == col[3] and len(set(col)) == 3


def test_contin_symbols_none_collapses_constants_to_one_symbol():
    pop = [(0, 7), (1, 7), (2, 8)]
    out = contin_symbols(pop, (0,), _value, "none")
    assert [row[0] for row in out] == [0, 0, 0]
    assert [row[1] for row in out] == [7, 7, 8]


def test_contin_symbols_bin_maps_later_constants_to_the_nearest_retained_one():
    # Row-major traversal (row by row, contin positions in order) retains
    # the first three distinct values 0.0, 1.0, 2.0; later ones are binned
    # to the nearest retained value: 1.4 -> 1.0, and 2.5, 10.0 and 3.0 -> 2.0
    # (3.0 was seen too late to be retained).
    pop = [(0, 1), (2, 1), (4, 5), (6, 3)]
    out = contin_symbols(pop, (0, 1), _value, "bin", max_constants=3)
    sym = {}
    for row, src in zip(out, pop, strict=True):
        for p in (0, 1):
            sym.setdefault(_VALUES[src[p]], set()).add(row[p])
    assert all(len(s) == 1 for s in sym.values())
    s = {v: next(iter(k)) for v, k in sym.items()}
    assert len({s[0.0], s[1.0], s[2.0]}) == 3
    assert s[1.4] == s[1.0]
    assert s[2.5] == s[2.0]
    assert s[10.0] == s[2.0]
    assert s[3.0] == s[2.0]


def test_contin_symbols_bin_ties_go_to_the_lower_constant():
    pop = [(0,), (2,), (1,)]
    out = contin_symbols(pop, (0,), _value, "bin", max_constants=2)
    assert out[2][0] == out[0][0]


def test_contin_symbols_bin_equals_all_below_gamma():
    pop = [(0, 1), (2, 1), (4, 5), (6, 3)]
    assert contin_symbols(pop, (0, 1), _value, "bin") == contin_symbols(
        pop, (0, 1), _value, "all"
    )


def test_contin_symbols_rejects_unknown_mode():
    with pytest.raises(ValueError):
        contin_symbols([(0,)], (0,), _value, "some")


def _linkage_pop():
    # positions 0 and 1 are "constants" (codes 0..6), 2 and 3 discrete.
    return [
        (0, 1, 0, 0),
        (1, 2, 1, 1),
        (2, 3, 0, 1),
        (4, 5, 1, 0),
        (6, 1, 0, 0),
        (1, 4, 1, 1),
    ]


def _symbolizer(mode):
    return lambda pop: contin_symbols(pop, (0, 1), _value, mode)


@pytest.mark.parametrize("fos", ["lt-b", "lt-mi"])
def test_no_const_puts_constants_in_singletons_outside_the_tree(fos):
    model = LinkageModel(fos, MT19937RandGen(5), symbolize=_symbolizer("none"))
    pop = _linkage_pop()
    model.calibrate(pop, (7, 7, 2, 2))
    assert model.search_positions == (0, 1, 2, 3)
    F = model.learn(pop)
    assert [0] in F and [1] in F
    assert all(0 not in s and 1 not in s for s in F if len(s) > 1)
    assert any(set(s) == {2} for s in F) and any(set(s) == {3} for s in F)


@pytest.mark.parametrize("mode", ["all", "bin"])
def test_all_and_bin_keep_constants_in_the_linkage_tree(mode):
    model = LinkageModel("lt-b", MT19937RandGen(5), symbolize=_symbolizer(mode))
    pop = _linkage_pop()
    model.calibrate(pop, (7, 7, 2, 2))
    F = model.learn(pop)
    # 4 linked positions: 4 singletons + 2 merges (root dropped).
    assert len(F) == 6
    assert any(len(s) > 1 and (0 in s or 1 in s) for s in F)


def test_no_symbolizer_is_the_plain_model():
    pop = _linkage_pop()
    a = LinkageModel("lt-b", MT19937RandGen(9))
    b = LinkageModel("lt-b", MT19937RandGen(9), symbolize=None)
    a.calibrate(pop, (7, 7, 2, 2))
    b.calibrate(pop, (7, 7, 2, 2))
    assert a.learn(pop) == b.learn(pop)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_uniform_init_draws_canonical_contin_codes_that_vary():
    schema = FrozenSchema(_contin_rep(), contin="copy")
    init = build_uniform_population(schema, 16, MT19937RandGen(4))
    assert len(init.genotypes) == 16
    for g in init.genotypes:
        for p in schema.contin_positions:
            assert 0 <= g[p] < schema.domains[p]
            assert _canonical(schema, p, g[p])
    varying = [p for p in schema.contin_positions if len({g[p] for g in init.genotypes}) > 1]
    assert len(varying) == len(schema.contin_positions)


def test_sampler_init_works_on_contin_schema():
    schema = FrozenSchema(_contin_rep(), contin="copy")
    randGen().seed(1)
    init = build_population(schema, 16, parse_radii("1:0.5,2:0.5"), MT19937RandGen(2))
    assert len(init.genotypes) == 16
    for g in init.genotypes:
        schema.to_instance(g)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_gomea_parameters_contin_options():
    gp = gomea_parameters()
    assert (gp.contin, gp.contin_stats) == ("reject", "bin")
    gp = gomea_parameters(contin="copy", contin_stats="none")
    assert (gp.contin, gp.contin_stats) == ("copy", "none")
    with pytest.raises(ValueError):
        gomea_parameters(contin="x")
    with pytest.raises(ValueError):
        gomea_parameters(contin_stats="x")


class _SumScorer:
    """Scores an instance by -|sum of its constants - 4| - 1 (never the 0 target)."""

    def __init__(self, rep):
        self.rep = rep
        self.calls = 0

    def get_rep(self):
        return self.rep

    def __call__(self, inst):
        self.calls += 1
        fields = self.rep.fields()
        total = sum(fields.get_contin(inst, i) for i in range(len(fields.contin())))
        return composite_score(-abs(total - 4.0) - 1.0, 0, 0.0, 0.0)


@pytest.mark.parametrize("stats", ["all", "none", "bin"])
@pytest.mark.parametrize("init", ["sampler", "uniform"])
def test_optimizer_runs_on_contin_and_keeps_the_deme_invariants(stats, init):
    rep = _contin_rep()
    randGen().seed(7)
    isc = _SumScorer(rep)
    deme = instance_set(rep.fields())
    opt = gomea_optimization(
        gp=gomea_parameters(pop_size=8, contin="copy", contin_stats=stats, init_mode=init)
    )
    opt(deme, isc, 400, 10**6)
    assert deme.n_evals == len(deme) == isc.calls
    assert 0 < deme.n_evals <= 400
    assert opt.last_result["termination_reason"] in (
        "inner_budget",
        "global_budget",
        "stagnation",
        "converged",
        "max_generations",
    )


def test_optimizer_default_still_rejects_contin():
    rep = _contin_rep()
    deme = instance_set(rep.fields())
    with pytest.raises(UnsupportedRepresentation):
        gomea_optimization()(deme, _SumScorer(rep), 100, 100)
    assert len(deme) == 0


def test_engine_accepts_a_symbolizer():
    ev = _AcceptAll()
    pop = [Individual(g, 0, None) for g in _linkage_pop()]
    eng = GomeaEngine(
        (7, 7, 2, 2),
        EngineParams(max_generations=1),
        ev,
        MT19937RandGen(1),
        MT19937RandGen(2),
        symbolize=_symbolizer("none"),
    )
    eng.run(pop)
    assert eng.model.symbolize is not None


def test_contin_spec_codes_are_two_bits_per_raw_field():
    # The code packs raw field k of the knob at bits 2k..2k+1, so every raw
    # value MOSES can store (Stop=0, Left=1, Right=2) round-trips exactly.
    assert (contin_spec.Stop, contin_spec.Left, contin_spec.Right) == (0, 1, 2)
