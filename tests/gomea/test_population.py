"""Tests for pymoses.gomea.population: the sampler-based GOMEA initial
population.

See ``pymoses/gomea/population.py`` for the design rationale -- in short,
``build_population`` seeds a population of distinct legal genotypes around
one ``FrozenSchema``'s default (exemplar) genotype using MOSES's own
``sample_from_neighborhood`` at a mixture of radii, with no fitness
involved anywhere.
"""

import math

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea.population import (
    InitResult,
    build_population,
    build_uniform_population,
    choose_radius,
    field_diagnostics,
    parse_radii,
    resolve_radii,
    top_up_population,
)
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import MT19937RandGen, randGen

DEFAULT_MIX = ((1, 0.4), (2, 0.3), (4, 0.2), ("max", 0.1))


def _make_rep(exemplar: str, arity: int) -> representation:
    """Same construction ``tests/gomea/conftest.py``'s ``make_boolean_rep``
    fixture uses -- restated locally per this task's brief (this test
    module may not edit ``conftest.py``).
    """
    rule = logical_reduction(frozenset())(2)
    return representation(
        rule,
        rule,
        str_to_combo_tree(exemplar),
        gen_signature(type_node.boolean_type, arity),
        perm_ratio=-1.0,
    )


def _hamming(a: tuple, b: tuple) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


class _EmptySchema:
    """Minimal stand-in for a ``dim == 0`` ``FrozenSchema``.

    A real MOSES representation cannot easily be driven down to zero
    logical fields through the fixtures available to this test module, so
    ``build_population``'s ``dim == 0`` short-circuit (which never touches
    ``schema.fields``, ``schema.to_instance``, or ``resolve_radii``) is
    exercised through this tiny duck-typed object instead.
    """

    dim = 0
    domains: tuple = ()

    def default_genotype(self) -> tuple:
        return ()


# ---------------------------------------------------------------------------
# parse_radii
# ---------------------------------------------------------------------------


def test_parse_radii_parses_the_canonical_example():
    assert parse_radii("1:0.4,2:0.3,4:0.2,max:0.1") == (
        (1, 0.4),
        (2, 0.3),
        (4, 0.2),
        ("max", 0.1),
    )


def test_parse_radii_strips_whitespace():
    assert parse_radii(" 1 : 0.5 , max : 0.5 ") == ((1, 0.5), ("max", 0.5))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "1:0.4,,2:0.3",
        "1-0.4",
        "abc:0.5",
        "1:xyz",
        "1:0.4,",
    ],
)
def test_parse_radii_rejects_malformed_text(text):
    with pytest.raises(ValueError):
        parse_radii(text)


# ---------------------------------------------------------------------------
# resolve_radii
# ---------------------------------------------------------------------------


def test_resolve_radii_turns_max_into_dim_and_sorts_by_radius():
    resolved = resolve_radii(((2, 0.5), ("max", 0.5)), dim=5)
    assert resolved == ((2, 0.5), (5, 0.5))


def test_resolve_radii_clamps_radii_above_dim():
    resolved = resolve_radii(((1, 0.5), (10, 0.5)), dim=3)
    assert resolved == ((1, 0.5), (3, 0.5))


def test_resolve_radii_merges_equal_resolved_radii_and_renormalises():
    # Both components clamp to 3 on a dim=3 schema -- they must merge into
    # one, and the single merged weight must renormalise to 1.0.
    resolved = resolve_radii(((3, 0.2), (5, 0.2)), dim=3)
    assert resolved == ((3, 1.0),)


def test_resolve_radii_renormalises_weights_that_do_not_sum_to_one():
    resolved = resolve_radii(((1, 2.0), (2, 2.0)), dim=5)
    assert resolved == ((1, 0.5), (2, 0.5))


def test_parse_radii_accepts_uniform_shorthand_and_component():
    assert parse_radii("uniform") == (("uniform", 1.0),)
    assert parse_radii(" uniform ") == (("uniform", 1.0),)
    assert parse_radii("1:0.5,uniform:0.5") == ((1, 0.5), ("uniform", 0.5))


def test_resolve_radii_uniform_gives_every_radius_equal_weight():
    resolved = resolve_radii((("uniform", 1.0),), dim=4)
    assert [r for r, _ in resolved] == [1, 2, 3, 4]
    assert all(w == pytest.approx(0.25) for _, w in resolved)


def test_resolve_radii_uniform_mixes_with_explicit_radii():
    # Half the mass on radius 1, the other half spread over 1..4: radius 1
    # gets 0.5 + 0.125, every other radius 0.125.
    resolved = dict(resolve_radii(((1, 0.5), ("uniform", 0.5)), dim=4))
    assert resolved[1] == pytest.approx(0.625)
    assert [resolved[r] for r in (2, 3, 4)] == pytest.approx([0.125] * 3)


def test_resolve_radii_uniform_rejects_non_positive_weight():
    with pytest.raises(ValueError):
        resolve_radii((("uniform", 0.0),), dim=4)


def test_resolve_radii_rejects_empty_mixture():
    with pytest.raises(ValueError):
        resolve_radii((), dim=5)


def test_resolve_radii_rejects_a_literal_radius_below_one():
    with pytest.raises(ValueError):
        resolve_radii(((0, 1.0),), dim=5)


def test_resolve_radii_rejects_a_non_positive_weight():
    with pytest.raises(ValueError):
        resolve_radii(((1, 0.0),), dim=5)
    with pytest.raises(ValueError):
        resolve_radii(((1, -0.5),), dim=5)


# ---------------------------------------------------------------------------
# choose_radius -- pure RNG distribution check (no sampler involved)
# ---------------------------------------------------------------------------


def test_choose_radius_distribution_matches_weights_within_five_sigma():
    # A real 5-input Boolean schema (radii 1, 2, 4 are all distinct and
    # below its dim of 5) -- but this test never touches the (slow)
    # sampler, only the RNG-driven radius CHOICE, which is what
    # choose_radius/resolve_radii alone are responsible for.
    randGen().seed(42)
    schema = FrozenSchema(_make_rep("and($1 $2 $3 $4 $5)", 5))
    assert schema.dim == 5

    resolved = resolve_radii(DEFAULT_MIX, dim=schema.dim)
    assert [r for r, _ in resolved] == [1, 2, 4, 5]

    rng = MT19937RandGen(20260924)
    draws = 2000
    counts = {r: 0 for r, _ in resolved}
    for _ in range(draws):
        counts[choose_radius(resolved, rng)] += 1

    for radius, weight in resolved:
        mean = draws * weight
        std = math.sqrt(draws * weight * (1 - weight))
        low, high = mean - 5 * std, mean + 5 * std
        assert low <= counts[radius] <= high, (
            f"radius {radius}: observed {counts[radius]}, expected {mean} +/- 5*{std}"
        )


# ---------------------------------------------------------------------------
# build_population -- structural properties
# ---------------------------------------------------------------------------


@pytest.fixture
def big_schema():
    randGen().seed(0)
    rep = _make_rep("and($1 $2 $3 $4 $5)", 5)
    return FrozenSchema(rep)


def test_every_genotype_is_legal(big_schema):
    randGen().seed(1)
    rng = MT19937RandGen(1)
    result = build_population(big_schema, 40, DEFAULT_MIX, rng)

    for g in result.genotypes:
        # to_instance raises ValueError on any out-of-domain value or wrong
        # length -- if this doesn't raise, g is legal.
        big_schema.to_instance(g)


def test_index_zero_is_the_default_genotype(big_schema):
    randGen().seed(2)
    rng = MT19937RandGen(2)
    result = build_population(big_schema, 10, DEFAULT_MIX, rng)

    assert result.genotypes[0] == big_schema.default_genotype()
    assert result.radii[0] == 0


def test_sampler_distance_equals_the_chosen_radius(big_schema):
    randGen().seed(3)
    rng = MT19937RandGen(3)
    result = build_population(big_schema, 40, DEFAULT_MIX, rng)

    default = big_schema.default_genotype()
    for g, r in zip(result.genotypes, result.radii, strict=True):
        assert _hamming(g, default) == r


def test_no_duplicates_when_dedupe(big_schema):
    randGen().seed(4)
    rng = MT19937RandGen(4)
    result = build_population(big_schema, 40, DEFAULT_MIX, rng, dedupe=True)

    assert len(result.genotypes) == len(set(result.genotypes))
    assert not result.feasible_short


def test_tiny_schema_returns_at_most_all_legal_genotypes_and_terminates():
    randGen().seed(5)
    rep = _make_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    assert schema.dim == 2
    assert schema.domains == (3, 3)  # 9 legal genotypes total

    rng = MT19937RandGen(5)
    result = build_population(schema, 64, ((1, 0.5), (2, 0.5)), rng)

    assert len(result.genotypes) <= 9
    assert result.feasible_short is True
    assert len(result.genotypes) == len(set(result.genotypes))


def test_build_population_never_imports_or_touches_a_scorer(big_schema):
    """No fitness anywhere: build_population's own module imports none of
    the scorer-shaped symbols, and a plain structural check confirms the
    InitResult carries no score-shaped field. (The module's own prose
    docstrings freely use words like "scorer" to EXPLAIN this absence, so
    the check below looks only at import/from lines, not the whole file.)
    """
    import pymoses.gomea.population as population_module

    source = population_module.__file__
    with open(source, encoding="utf-8") as fh:
        import_lines = [
            line for line in fh if line.lstrip().startswith(("import ", "from "))
        ]
    assert not any("scor" in line.lower() for line in import_lines), import_lines

    randGen().seed(6)
    rng = MT19937RandGen(6)
    result = build_population(big_schema, 10, DEFAULT_MIX, rng)
    assert not hasattr(result, "score")
    assert not hasattr(result, "scores")


def test_dim_zero_schema_returns_only_the_default_genotype():
    schema = _EmptySchema()
    rng = MT19937RandGen(7)

    result = build_population(schema, 1, DEFAULT_MIX, rng)
    assert result.genotypes == [()]
    assert result.radii == [0]
    assert result.feasible_short is False

    result_short = build_population(schema, 5, DEFAULT_MIX, rng)
    assert result_short.genotypes == [()]
    assert result_short.feasible_short is True


# ---------------------------------------------------------------------------
# Coverage pass
# ---------------------------------------------------------------------------


def test_coverage_makes_a_previously_constant_field_vary_when_feasible():
    randGen().seed(8)
    rep = _make_rep("and($1 $2 $3 $4 $5)", 5)
    schema = FrozenSchema(rep)

    rng = MT19937RandGen(8)
    # A small population drawn only at radius 1 leaves most of the 5 fields
    # constant (each draw touches exactly one field), and a generous
    # coverage budget makes fixing at least one of them near-certain.
    result = build_population(
        schema,
        3,
        ((1, 1.0),),
        rng,
        coverage_cap_multiple=200,
    )

    before = set(result.diagnostics["constant_fields_before_coverage"])
    after = set(result.diagnostics["constant_fields_after_coverage"])
    assert before, "test setup expected at least one constant field before coverage"
    # At least one field that was constant before coverage must have been
    # fixed. (A DIFFERENT field may have become newly constant as a
    # documented side effect of a replacement -- that is not a failure,
    # see build_population's own docstring on this.)
    assert before - after, f"no previously-constant field was fixed: {before=} {after=}"
    assert result.coverage_replacements > 0
    # The default genotype must never be the one replaced.
    assert result.genotypes[0] == schema.default_genotype()


def test_coverage_never_replaces_the_default_member():
    randGen().seed(9)
    rep = _make_rep("and($1 $2 $3)", 3)
    schema = FrozenSchema(rep)
    rng = MT19937RandGen(9)

    result = build_population(schema, 4, ((1, 1.0),), rng, coverage_cap_multiple=50)
    assert result.genotypes[0] == schema.default_genotype()
    assert result.radii[0] == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_population_is_deterministic_given_both_seeds():
    rep = _make_rep("and($1 $2 $3 $4 $5)", 5)
    randGen().seed(0)  # representation build itself never draws (perm_ratio=-1.0)
    schema = FrozenSchema(rep)

    randGen().seed(123)
    rng1 = MT19937RandGen(456)
    result1 = build_population(schema, 20, DEFAULT_MIX, rng1, coverage_cap_multiple=5)

    randGen().seed(123)
    rng2 = MT19937RandGen(456)
    result2 = build_population(schema, 20, DEFAULT_MIX, rng2, coverage_cap_multiple=5)

    assert result1 == result2


# ---------------------------------------------------------------------------
# field_diagnostics
# ---------------------------------------------------------------------------


def test_field_diagnostics_shape_and_values():
    genotypes = [(0, 0), (1, 0), (2, 0), (1, 1)]
    domains = (3, 2)

    diag = field_diagnostics(genotypes, domains)

    assert diag["n"] == 4
    assert diag["unique_genotypes"] == 4
    assert len(diag["per_field"]) == 2

    field0 = diag["per_field"][0]
    assert field0["domain"] == 3
    assert field0["observed"] == [0, 1, 2]
    assert field0["default_freq"] == pytest.approx(0.25)
    # Values 0,1,2,1 -> counts {0:1, 1:2, 2:1} -> H = -(.25 log2 .25 *2 + .5 log2 .5)
    expected_entropy = -(0.25 * math.log2(0.25) * 2 + 0.5 * math.log2(0.5))
    assert field0["entropy_bits"] == pytest.approx(expected_entropy)

    field1 = diag["per_field"][1]
    assert field1["domain"] == 2
    assert field1["observed"] == [0, 1]
    assert field1["default_freq"] == pytest.approx(0.75)


def test_field_diagnostics_zero_entropy_for_a_constant_field():
    genotypes = [(0, 0), (0, 1), (0, 2)]
    diag = field_diagnostics(genotypes, (1, 3))
    assert diag["per_field"][0]["entropy_bits"] == 0.0
    assert diag["per_field"][0]["observed"] == [0]
    assert diag["per_field"][0]["default_freq"] == 1.0


def test_init_result_is_a_dataclass_with_the_documented_fields():
    fields = InitResult.__dataclass_fields__
    for name in (
        "genotypes",
        "radii",
        "attempts",
        "duplicates_rejected",
        "coverage_replacements",
        "feasible_short",
        "diagnostics",
    ):
        assert name in fields


# ---------------------------------------------------------------------------
# build_uniform_population -- the Phase 1d `--gomea-init uniform` ablation
# ---------------------------------------------------------------------------


def test_uniform_every_genotype_is_legal_and_index_zero_is_the_default(big_schema):
    result = build_uniform_population(big_schema, 40, MT19937RandGen(1))

    assert len(result.genotypes) == 40
    assert result.genotypes[0] == big_schema.default_genotype()
    assert result.radii[0] == 0
    for g in result.genotypes:
        big_schema.to_instance(g)  # raises on an out-of-domain value
        assert all(0 <= v < d for v, d in zip(g, big_schema.domains, strict=True))
    assert len(result.genotypes) == len(set(result.genotypes))
    assert not result.feasible_short
    assert result.coverage_replacements == 0


def test_uniform_radii_are_hamming_distances_and_diagnostics_match_sampler_keys(big_schema):
    result = build_uniform_population(big_schema, 20, MT19937RandGen(2))
    default = big_schema.default_genotype()
    for g, r in zip(result.genotypes, result.radii, strict=True):
        assert _hamming(g, default) == r

    sampler = build_population(big_schema, 20, DEFAULT_MIX, MT19937RandGen(2))
    assert set(result.diagnostics) == set(sampler.diagnostics)
    assert (
        result.diagnostics["constant_fields_before_coverage"]
        == result.diagnostics["constant_fields_after_coverage"]
    )


def test_uniform_draws_are_deterministic_under_a_seed_and_never_touch_the_global_rng(
    big_schema,
):
    randGen().seed(7)
    a = build_uniform_population(big_schema, 20, MT19937RandGen(5))
    after_a = randGen().randint()
    randGen().seed(7)
    b = build_uniform_population(big_schema, 20, MT19937RandGen(5))
    after_b = randGen().randint()
    randGen().seed(7)
    untouched = randGen().randint()

    assert a.genotypes == b.genotypes and a.attempts == b.attempts
    # MOSES's sampler (which draws from the global RNG) is never used.
    assert after_a == after_b == untouched
    c = build_uniform_population(big_schema, 20, MT19937RandGen(6))
    assert c.genotypes != a.genotypes


def test_uniform_differs_from_sampler_mode(big_schema):
    randGen().seed(3)
    sampler = build_population(big_schema, 20, DEFAULT_MIX, MT19937RandGen(3))
    uniform = build_uniform_population(big_schema, 20, MT19937RandGen(3))
    assert sampler.genotypes != uniform.genotypes
    # Uniform draws land far from the exemplar: most knobs are non-default.
    mean_uniform = sum(uniform.radii) / len(uniform.radii)
    mean_sampler = sum(sampler.radii) / len(sampler.radii)
    assert mean_uniform > mean_sampler


def test_uniform_tiny_schema_dedupes_with_the_same_attempt_cap_and_goes_short():
    rep = _make_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    assert schema.domains == (3, 3)  # 9 legal genotypes total

    result = build_uniform_population(schema, 64, MT19937RandGen(5))

    assert len(result.genotypes) == 9
    assert len(set(result.genotypes)) == 9
    assert result.feasible_short is True
    assert result.attempts == 20 * 64  # the sampler mode's fill budget
    assert result.duplicates_rejected == result.attempts - 8


def test_uniform_dim_zero_schema_returns_only_the_default_genotype():
    result = build_uniform_population(_EmptySchema(), 5, MT19937RandGen(1))
    assert result.genotypes == [()]
    assert result.radii == [0]
    assert result.feasible_short is True


# ---------------------------------------------------------------------------
# Phase 1e: top_up_population (the HC->GOMEA hybrid's initial population)
# ---------------------------------------------------------------------------


def test_top_up_keeps_the_seeds_first_and_draws_around_the_center():
    schema = FrozenSchema(_make_rep("and", 3))
    g0 = schema.default_genotype()
    center = tuple([1] + list(g0[1:]))
    seeds = [center, g0]
    randGen().seed(3)
    res = top_up_population(schema, seeds, center, 10, DEFAULT_MIX, MT19937RandGen(5))
    assert res.genotypes[:2] == seeds
    assert len(res.genotypes) == len(set(res.genotypes)) == 10
    assert res.radii[:2] == [0, 1]
    for g, r in zip(res.genotypes[2:], res.radii[2:], strict=True):
        assert _hamming(g, center) == r
    assert res.coverage_replacements == 0 and not res.feasible_short
    assert res.attempts == 8 + res.duplicates_rejected


def test_top_up_with_enough_seeds_draws_nothing():
    schema = FrozenSchema(_make_rep("and", 3))
    randGen().seed(3)
    seeds = build_population(schema, 6, DEFAULT_MIX, MT19937RandGen(1)).genotypes
    before = randGen().randint()
    randGen().seed(3)
    build_population(schema, 6, DEFAULT_MIX, MT19937RandGen(1))
    res = top_up_population(schema, seeds, seeds[0], 4, DEFAULT_MIX, MT19937RandGen(5))
    assert res.genotypes == list(seeds[:4]) and res.attempts == 0
    assert randGen().randint() == before


def test_top_up_without_seeds_starts_from_the_center():
    schema = FrozenSchema(_make_rep("and", 3))
    g0 = schema.default_genotype()
    randGen().seed(3)
    res = top_up_population(schema, [], g0, 5, DEFAULT_MIX, MT19937RandGen(5))
    assert res.genotypes[0] == g0 and len(res.genotypes) == 5
