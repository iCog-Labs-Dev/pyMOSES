"""The IT linkage study (BACKLOG 13.9, plan section 5.6): the start populations. The template
generators follow GP-GOMEA's ``GenerateTreeFull`` and ``GenerateTreeGrow`` draw for draw; the
shape rule is FULL for individuals 0 .. N / 2; every position varies; duplicates are drawn
again up to the cap."""

from __future__ import annotations

from pymoses.gomea.standalone.genome_it import ITGenome
from pymoses.gomea.standalone.genome_template import TemplateGenome
from pymoses.gomea.standalone.init import (
    diversity,
    generate_tree_full,
    generate_tree_grow,
    it_individual,
    it_population,
    template_population,
)
from pymoses.util.rng import MT19937RandGen


class _Seq:
    """A stand-in RNG that returns the given doubles in order and counts the draws."""

    def __init__(self, values):
        self.values = list(values)
        self.draws = 0

    def randdouble(self):
        self.draws += 1
        return self.values.pop(0)


class _Counting(MT19937RandGen):
    def __init__(self, seed):
        super().__init__(seed)
        self.draws = 0

    def randdouble(self):
        self.draws += 1
        return super().randdouble()


def test_full_draws_once_per_node_in_pre_order():
    t = TemplateGenome(5, height=1)
    rng = _Seq([0.3, 0.5, 0.9])
    assert generate_tree_full(t, rng) == (1, 2, 4)  # '-' over x2 and x4
    assert rng.draws == 3


def test_grow_draws_twice_per_node_in_pre_order():
    t = TemplateGenome(5, height=1)
    # root: 0.3 < 0.5 -> a function, picked by 0.8 (aq); leaves: one unused draw, then the input
    rng = _Seq([0.3, 0.8, 0.0, 0.1, 0.0, 0.99])
    assert generate_tree_grow(t, rng) == (3, 0, 4)
    assert rng.draws == 6
    # root: 0.7 >= 0.5 -> an input (0.45 * 5 -> x2, code 4 + 2); the children still exist
    rng = _Seq([0.7, 0.45, 0.0, 0.6, 0.0, 0.2])
    assert generate_tree_grow(t, rng) == (6, 3, 1)
    # randdouble() can return 1.0: the index is capped
    rng = _Seq([0.2, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert generate_tree_grow(t, rng) == (3, 4, 4)


def test_height_4_draw_counts():
    t = TemplateGenome(5)
    rng = _Counting(1)
    g = generate_tree_full(t, rng)
    assert rng.draws == 31 and all(c < 4 for c in g[:15])
    rng = _Counting(1)
    generate_tree_grow(t, rng)
    assert rng.draws == 62


def test_the_template_population_shape_rule_and_diversity():
    t = TemplateGenome(5)
    pop, rec = template_population(t, 1000, MT19937RandGen(7))
    assert len(pop) == 1000 and rec["duplicates_kept"] == 0
    full = pop[:501]
    assert all(all(c < 4 for c in g[:15]) for g in full)
    grow = pop[501:]
    assert sum(any(c >= 4 for c in g[:15]) for g in grow) > 400
    d = diversity(pop, t.domains)
    assert d["constant_positions"] == [] and d["unique_genotypes"] == 1000


def test_duplicates_are_drawn_again_up_to_the_cap():
    # one input: 4 distinct FULL trees (the root function), plus the input at the root (GROW)
    t = TemplateGenome(1, height=1)
    pop, rec = template_population(t, 5, MT19937RandGen(3), try_multiple=50)
    assert len(set(pop)) == 5 and rec["duplicates_kept"] == 0
    # 12 slots but 5 distinct trees: the cap (12 redraws) is spent, and duplicates are kept
    pop, rec = template_population(t, 12, MT19937RandGen(3), try_multiple=1)
    assert len(pop) == 12 and rec["duplicates_kept"] > 0 and rec["redraws"] == 12


def test_the_it_start_individual():
    genome = ITGenome(5)
    rng = MT19937RandGen(5)
    active = total = 0
    for _ in range(500):
        g = it_individual(genome, rng)
        for t in range(8):
            inputs = [g[4 * t], g[4 * t + 2]]
            total += 1
            if any(inputs):
                active += 1
            assert all(0 <= c <= 5 for c in inputs)
    assert 0.45 < active / total < 0.55


def test_the_it_population_varies_every_position():
    genome = ITGenome(5)
    pop, rec = it_population(genome, 1000, MT19937RandGen(9))
    d = diversity(pop, genome.domains)
    assert d["constant_positions"] == [] and len(pop) == 1000
    assert rec["duplicates_kept"] == 0
