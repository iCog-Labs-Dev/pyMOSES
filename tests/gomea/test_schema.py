"""Tests for pymoses.gomea.schema.FrozenSchema."""

import itertools
import random

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.interpreter.interpreter import boolean_interpreter
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea.schema import FrozenSchema, UnsupportedRepresentation
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import randGen

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_genotype_decodes_to_the_exemplar_program_and_score(make_boolean_rep):
    source = "and($1 $2)"
    rep = make_boolean_rep(source, 2)
    schema = FrozenSchema(rep)

    g0 = schema.default_genotype()
    assert g0 == (0,) * schema.dim

    decoded = schema.decode(g0)
    expected = rep.get_clean_exemplar(reduce=True)
    assert combo_tree_to_str(decoded) == combo_tree_to_str(expected)

    # "Same score": functionally equivalent to the original exemplar text
    # over every input row -- a real bscorer would give both the identical
    # score for exactly this reason.
    original = str_to_combo_tree(source)
    bi = boolean_interpreter()
    for bits in itertools.product((builtin.logical_false, builtin.logical_true), repeat=2):
        bi.boolean_inputs = list(bits)
        assert bi(decoded) == bi(original)


def test_unsupported_representation_rejects_contin_fields():
    """A representation with contin fields has no fixed integer domain per
    field (see FrozenSchema's own docstring) -- reuses the same bare
    construction test_contin_representation.py's own linear_contin default
    test builds, which is known to produce contin fields (depth 5, per
    representation.py's module-level ``_depth`` default).
    """
    rep = representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 1),
        linear_contin=True,
    )
    assert rep.fields().n_contin_fields() > 0

    with pytest.raises(UnsupportedRepresentation):
        FrozenSchema(rep)


def test_constructor_rejects_a_non_representation():
    with pytest.raises(TypeError):
        FrozenSchema("not-a-representation")


# ---------------------------------------------------------------------------
# Encode / decode round trip
# ---------------------------------------------------------------------------


def test_to_genotype_round_trips_every_field(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)

    for k in range(schema.dim):
        for value in range(schema.domains[k]):
            g = list(schema.default_genotype())
            g[k] = value
            inst = schema.to_instance(g)
            assert schema.to_genotype(inst) == tuple(g)


def test_to_genotype_of_to_instance_round_trips_200_random_legal_genotypes(make_boolean_rep):
    rep = make_boolean_rep("or(and($1 $2) $3)", 3)
    schema = FrozenSchema(rep)

    rng = random.Random(20260924)
    for _ in range(200):
        g = tuple(rng.randrange(domain) for domain in schema.domains)
        inst = schema.to_instance(g)
        assert schema.to_genotype(inst) == g


def test_to_instance_rejects_code_3_on_a_multy_3_field(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)

    idx = next(k for k, domain in enumerate(schema.domains) if domain == 3)
    g = list(schema.default_genotype())
    g[idx] = 3
    with pytest.raises(ValueError):
        schema.to_instance(g)


def test_to_instance_rejects_the_wrong_length(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)

    with pytest.raises(ValueError):
        schema.to_instance(list(schema.default_genotype()) + [0])
    with pytest.raises(ValueError):
        schema.to_instance(list(schema.default_genotype())[:-1])


# ---------------------------------------------------------------------------
# schema_id
# ---------------------------------------------------------------------------


def test_schema_id_differs_for_different_exemplars_with_the_same_dim(make_boolean_rep):
    rep_and = make_boolean_rep("and($1 $2)", 2)
    rep_or = make_boolean_rep("or($1 $2)", 2)

    schema_and = FrozenSchema(rep_and)
    schema_or = FrozenSchema(rep_or)

    assert schema_and.dim == schema_or.dim
    assert schema_and.schema_id != schema_or.schema_id


def test_schema_id_is_stable_across_two_constructions_of_the_same_representation(
    make_boolean_rep,
):
    rep1 = make_boolean_rep("and($1 $2)", 2)
    rep2 = make_boolean_rep("and($1 $2)", 2)

    schema1 = FrozenSchema(rep1)
    schema2 = FrozenSchema(rep2)

    assert schema1.rep is not schema2.rep
    assert schema1.schema_id == schema2.schema_id


# ---------------------------------------------------------------------------
# A field nested under a disabled knob
# ---------------------------------------------------------------------------


def test_phenotype_key_equal_for_a_field_nested_under_a_disabled_knob():
    """Not every raw field always affects the decoded tree: build_knobs
    (build_knobs.py's own ``build_logical``) adds insertion-possibility
    knobs at every and/or node it recurses into, including nested ones
    reached only through another junctor -- and for a specific combination
    of the OTHER fields' settings, one such field can be a complete no-op.
    Found empirically (brute-force search over single-field changes) per
    the task brief's own guidance, rather than hand-derived: this exemplar
    (a genuinely nested `and(or(..) or(..))`, with real-pair knobs enabled
    via `perm_ratio=0.0` -- classic's own default, see
    build_knobs.py's module docstring) is the first one this search found
    to work; simpler two/three-argument exemplars were tried first and
    none of them had such a field (see task-2-report.md).

    The process-wide RNG is seeded so representation construction (pair
    sampling draws from `randGen()`) is reproducible; the SEARCH itself
    uses a separate, local `random.Random` so it never disturbs that
    global state.
    """
    randGen().seed(7)
    rule = logical_reduction(frozenset())(2)
    rep = representation(
        rule,
        rule,
        str_to_combo_tree("and(or($1 $2) or($3 $4))"),
        gen_signature(type_node.boolean_type, 4),
        perm_ratio=0.0,
    )
    schema = FrozenSchema(rep)

    search_rng = random.Random(999)
    found = None
    for _ in range(50_000):
        g1 = [search_rng.randrange(domain) for domain in schema.domains]
        k = search_rng.randrange(schema.dim)
        v = search_rng.randrange(schema.domains[k])
        if v == g1[k]:
            continue
        g2 = list(g1)
        g2[k] = v
        if schema.phenotype_key(g1) == schema.phenotype_key(g2):
            found = (tuple(g1), tuple(g2))
            break

    assert found is not None, "no nested-disabled-knob field found within the trial budget"
    g1, g2 = found
    assert sum(a != b for a, b in zip(g1, g2, strict=True)) == 1

    assert schema.phenotype_key(g1) == schema.phenotype_key(g2)
    assert combo_tree_to_str(schema.decode(g1)) == combo_tree_to_str(schema.decode(g2))
