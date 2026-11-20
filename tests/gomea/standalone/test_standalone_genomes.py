"""The IT linkage study (BACKLOG 13.9, plan section 8.1): the two genomes of the stand-alone
harness. Domains and lengths; decode on hand-written genomes; ``active_mask`` agrees with
``phenotype_key``; the IT merge of equal monomials, the negative-power rule, 0/1 inputs and the
empty model; ``to_moses``."""

from __future__ import annotations

import random

import pytest

from pymoses.gomea.standalone.genome_it import ITGenome, nodes_of
from pymoses.gomea.standalone.genome_template import TemplateGenome


def _it_genome(n=5, **kw):
    kw.setdefault("zero_inputs", frozenset({2, 4}))
    kw.setdefault("zero_one_inputs", frozenset({4}))
    kw.setdefault("bool_inputs", frozenset({4}))
    return ITGenome(n, **kw)


def _it(slots, terms=8, factors=2, powers=(-2, -1, 1, 2)):
    """An IT genotype from ``slots``: for each term a list of (input 1-based or 0, power)."""
    g = []
    for t in range(terms):
        facs = slots[t] if t < len(slots) else []
        for k in range(factors):
            inp, p = facs[k] if k < len(facs) else (0, powers[0])
            g += [inp, powers.index(p)]
    return tuple(g)


# ---- domains and lengths


def test_template_domains_and_length():
    g = TemplateGenome(5)
    assert g.length == 31 and g.n_internal == 15
    assert g.domains == tuple([9] * 15 + [5] * 16)
    assert TemplateGenome(3, height=2).domains == tuple([7] * 3 + [3] * 4)


def test_it_domains_and_length():
    g = ITGenome(5)
    assert g.length == 32
    assert g.domains == tuple([6, 4] * 16)
    g3 = ITGenome(7, terms=4, factors=3, powers=(-1, 1, 2))
    assert g3.length == 24 and g3.domains == tuple([8, 3] * 12)


@pytest.mark.parametrize("powers", [(), (0, 1), (1, 1)])
def test_it_refuses_bad_powers(powers):
    with pytest.raises(ValueError):
        ITGenome(3, powers=powers)


# ---- the template


def test_template_input_at_the_root_makes_every_other_position_inactive():
    t = TemplateGenome(5)
    g = tuple([4 + 2] + [0] * 14 + [1] * 16)  # root holds input x2 (code 4 + 2)
    assert t.phenotype_key(g) == "x2"
    assert t.active_mask(g) == tuple([True] + [False] * 30)
    assert t.size(g) == 1
    assert t.to_moses(g) == "$3"


def test_template_decode_of_a_hand_written_tree():
    t = TemplateGenome(4, height=2)  # 7 positions: 3 internal, 4 leaves
    # root aq; left: x0 (input at an internal position); right: '-' over leaves x2, x3
    g = (3, 4 + 0, 1, 1, 2, 2, 3)
    key, post = t.decode(g)
    assert key == "aq(x0,-(x2,x3))"
    assert t.active_mask(g) == (True, True, True, False, False, True, True)
    assert t.size(g) == 5
    assert t.to_moses(g) == "/($1 exp(*(0.5 log(+(1 *(+($3 *(-1 $4)) +($3 *(-1 $4))))))))"
    assert t.program_to_infix(post) == "aq(x1,(x3-x4))"


def test_template_boolean_input_prints_as_impulse():
    t = TemplateGenome(3, height=1, bool_inputs={1})
    assert t.to_moses((2, 1, 0)) == "*(impulse($2) $1)"


# ---- active_mask agrees with phenotype_key


@pytest.mark.parametrize("make", [lambda: TemplateGenome(4), lambda: _it_genome()],
                         ids=["template", "it"])
def test_changing_an_inactive_position_never_changes_the_key(make):
    genome = make()
    r = random.Random(7)
    checked = 0
    for _ in range(300):
        g = tuple(r.randrange(k) for k in genome.domains)
        key = genome.phenotype_key(g)
        mask = genome.active_mask(g)
        for p in range(genome.length):
            if mask[p]:
                continue
            for c in range(genome.domains[p]):
                h = list(g)
                h[p] = c
                assert genome.phenotype_key(tuple(h)) == key
                checked += 1
    assert checked > 1000


@pytest.mark.parametrize("make", [lambda: TemplateGenome(4), lambda: _it_genome()],
                         ids=["template", "it"])
def test_equal_keys_mean_equal_models(make):
    """Two genotypes with the same key print the same model."""
    genome = make()
    r = random.Random(3)
    by_key = {}
    for _ in range(3000):
        g = tuple(r.randrange(k) for k in genome.domains)
        by_key.setdefault(genome.phenotype_key(g), set()).add(genome.to_moses(g))
    assert all(len(v) == 1 for v in by_key.values())


# ---- the IT genome


def test_it_form_a_of_the_lessons():
    """x1 x2 + x3 (inputs 1-based in MOSES syntax)."""
    genome = ITGenome(5)
    g = _it([[(1, 1), (2, 1)], [(3, 1)]])
    m = genome.decode(g)
    assert m.monomials == (((0, 1), (1, 1)), ((2, 1),))
    assert m.slots[:2] == (((0, 1), (1, 1)), ((2, 1),)) and m.slots[2:] == (None,) * 6
    assert genome.to_moses(g) == "+(*($1 $2) $3)"
    assert nodes_of("+(*($1 $2) $3)") == 7  # phase_report's rule: 2 "(" plus 5 atoms
    assert genome.size(g) == 7 + 2


def test_it_merges_equal_monomials_and_adds_powers_of_one_input():
    genome = ITGenome(5)
    a = _it([[(1, 1), (2, 1)], [(2, 1), (1, 1)], [(1, 2)], [(1, 1), (1, 1)]])
    m = genome.decode(a)
    assert m.monomials == (((0, 1), (1, 1)), ((0, 2),))
    assert genome.to_moses(a) == "+(*($1 $2) *($1 $1))"
    # x1^1 x1^-1 cancels: the term is off
    b = _it([[(1, 1), (1, -1)], [(3, 2)]])
    assert genome.decode(b).monomials == (((2, 2),),)
    assert genome.decode(b).slots[0] is None


def test_it_negative_power_rule_for_inputs_with_zeros():
    genome = _it_genome()  # input index 2 (x3) has zeros
    g = _it([[(3, -2)], [(3, -1), (1, -1)]])
    m = genome.decode(g)
    assert m.monomials == (((0, -1), (2, 1)), ((2, 2),))
    assert m.forced == 2
    assert genome.to_moses(g) == "+(*(/(1 $1) $3) *($3 $3))"
    # the same model written with the positive powers has the same key
    assert genome.phenotype_key(_it([[(3, 2)], [(1, -1), (3, 1)]])) == m.monomials


def test_it_zero_one_input_has_power_one():
    genome = _it_genome()  # input index 4 (x5) is 0/1 and boolean
    for p in (-2, -1, 1, 2):
        m = genome.decode(_it([[(5, p)]]))
        assert m.monomials == (((4, 1),),)
    assert genome.decode(_it([[(5, 2), (5, -1)]])).monomials == (((4, 1),),)
    assert genome.to_moses(_it([[(5, 2), (1, 1)]])) == "*($1 impulse($5))"
    mask = genome.active_mask(_it([[(5, 2), (1, 1)]]))
    assert mask[:4] == (True, False, True, True)


def test_it_empty_model():
    genome = ITGenome(5)
    g = _it([])
    assert genome.decode(g).monomials == ()
    assert genome.to_moses(g) == "0"
    assert genome.size(g) == 1
    assert genome.active_mask(g) == tuple([True, False] * 16)


def test_it_one_term_prints_without_the_sum():
    genome = ITGenome(5)
    assert genome.to_moses(_it([[(2, -2)]])) == "/(1 *($2 $2))"
    assert genome.size(_it([[(2, -2)]])) == nodes_of("/(1 *($2 $2))")


def test_it_powers_above_two_print_as_products():
    genome = ITGenome(3)
    g = _it([[(1, 2), (1, 2)], [(2, -2), (2, -1)]])
    assert genome.decode(g).monomials == (((0, 4),), ((1, -3),))
    assert genome.to_moses(g) == "+(*($1 $1 $1 $1) /(1 *($2 $2 $2)))"


def test_it_monomial_column_follows_the_interpreter():
    genome = ITGenome(2)
    cols = [[2.0, 0.0, -1.5], [4.0, 3.0, 0.0]]
    assert genome.monomial_column(((0, 2),), cols) == [4.0, 0.0, 2.25]
    c = genome.monomial_column(((1, -1),), cols)
    assert c[:2] == [0.25, 1 / 3.0] and c[2] != c[2]  # 1/0: an overflow (nan)
    # a zero first factor stops the product, so the nan of the second factor is not reached
    c = genome.monomial_column(((0, 1), (1, -1)), [[2.0, 0.0], [4.0, 0.0]])
    assert c == [0.5, 0.0]
