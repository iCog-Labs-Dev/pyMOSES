"""The start population of each genome of the IT linkage study (plan section 5.6).

**Template: ported.** ``generate_tree_full`` and ``generate_tree_grow`` are ported from
GP-GOMEA (https://github.com/marcovirgolin/GP-GOMEA, commit 750220c,
``src/GOMEA/GOMEATreeInitializer.cpp``, ``GOMEATreeInitializer::GenerateTreeFull`` and
``GOMEATreeInitializer::GenerateTreeGrow``, Apache-2.0, Copyright Marco Virgolin). The shape
rule of ``template_population`` is ported from ``src/Evolution/PopulationInitializer.cpp``
(``InitializeTreePopulation``): FULL for the individuals 0 .. N / 2, GROW for the others, at
the full height (GP-GOMEA's GOMEA default ``--inittype`` HH, ``EvolutionState.cpp``:
``TreeInitHH`` when the option is not given, and ``InitializeRandomTree`` then uses
``height = max_height``). Adaptations:

* ``arma::randu()`` becomes ``rng.randdouble()`` and ``index = rnd * size`` becomes
  ``min(int(rnd * size), size - 1)`` (``randdouble`` can return 1.0; ``randu`` cannot), as in
  ``pymoses.gomea.fos.random_tree_fos``.
* The tree is written into a breadth-first genotype (position p has the children 2p + 1 and
  2p + 2), not into ``Node`` objects. The draws happen in GP-GOMEA's order (pre-order: the
  node, then child 0's subtree, then child 1's subtree), with the same number of draws for
  each node (FULL: one; GROW: two).
* Every node gets all its children, also under an input (``max_arity`` children always), so
  the full template exists and the introns vary, as in GP-GOMEA.
* NOT A PORT: duplicate genotypes are drawn again up to a try cap (the plan's rule, as
  ``pymoses.gomea.population.build_population`` does). GP-GOMEA's default draws no again
  (``syntactic_uniqueness_tries`` = -1). The redraw keeps the shape of its slot.

**IT: NOT A PORT** (the IT linkage study, BACKLOG 13.9; agent proposal D7, accepted by the
user). See ``it_individual``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pymoses.gomea.standalone.genome_template import FUNCTIONS
from pymoses.util.rng import MT19937RandGen

FULL, GROW = "full", "grow"


def generate_tree_full(genome, rng: MT19937RandGen) -> tuple[int, ...]:
    """GP-GOMEA: src/GOMEA/GOMEATreeInitializer.cpp GOMEATreeInitializer::GenerateTreeFull

    With ``actual_height_left == max_height_left`` (HH): a function at every internal
    position, an input at every leaf, one draw for each node."""
    g = [0] * genome.length
    n_fun = len(FUNCTIONS)
    n_in = genome.n_inputs

    def rec(p: int, max_left: int, actual_left: int) -> None:
        rnd = rng.randdouble()
        if max_left > 0:
            if actual_left > 0:
                g[p] = min(int(rnd * n_fun), n_fun - 1)
            else:
                g[p] = n_fun + min(int(rnd * n_in), n_in - 1)
            for i in range(2):
                rec(2 * p + 1 + i, max_left - 1, actual_left - 1)
        else:
            g[p] = min(int(rnd * n_in), n_in - 1)

    rec(0, genome.height, genome.height)
    return tuple(g)


def generate_tree_grow(genome, rng: MT19937RandGen) -> tuple[int, ...]:
    """GP-GOMEA: src/GOMEA/GOMEATreeInitializer.cpp GOMEATreeInitializer::GenerateTreeGrow

    At an internal position: one draw; a function with probability 0.5 (``rnd < 0.5``) when
    height is left, else an input; a second draw picks it. At a leaf: two draws (the first is
    not used), the second picks the input. Every node gets both children."""
    g = [0] * genome.length
    n_fun = len(FUNCTIONS)
    n_in = genome.n_inputs

    def rec(p: int, max_left: int, actual_left: int) -> None:
        rnd = rng.randdouble()
        if max_left > 0:
            if actual_left > 0 and rnd < 0.5:
                rnd = rng.randdouble()
                g[p] = min(int(rnd * n_fun), n_fun - 1)
            else:
                rnd = rng.randdouble()
                g[p] = n_fun + min(int(rnd * n_in), n_in - 1)
            for i in range(2):
                rec(2 * p + 1 + i, max_left - 1, actual_left - 1)
        else:
            rnd = rng.randdouble()
            g[p] = min(int(rnd * n_in), n_in - 1)

    rec(0, genome.height, genome.height)
    return tuple(g)


def it_individual(genome, rng: MT19937RandGen) -> tuple[int, ...]:
    """NOT A PORT (agent proposal D7). For each term slot t in order: first a power for each
    factor (uniform over the powers, also in the boxes that stay inactive, so every position
    varies); then the slot is active with probability 0.5 (``randdouble() < 0.5``); in an
    active slot each factor gets an input uniform over the n inputs with probability 0.5 and
    "none" otherwise, and the factor draws are repeated until at least one factor has an
    input. An inactive slot has "none" in every input box."""
    g = [0] * genome.length
    f, n_pow, n_in = genome.factors, len(genome.powers), genome.n_inputs
    for t in range(genome.terms):
        base = t * f * 2
        for k in range(f):
            g[base + 2 * k + 1] = rng.randint_n(n_pow)
        if rng.randdouble() < 0.5:
            while True:
                inputs = [1 + rng.randint_n(n_in) if rng.randdouble() < 0.5 else 0
                          for _ in range(f)]
                if any(inputs):
                    break
            for k in range(f):
                g[base + 2 * k] = inputs[k]
    return tuple(g)


def _fill(n: int, draw: Callable[[int], tuple[int, ...]], try_cap: int) -> tuple[list, dict]:
    """``n`` genotypes from ``draw(i)``; a duplicate is drawn again (the same slot ``i``)
    while the total number of extra draws is below ``try_cap``; after that a duplicate is
    kept."""
    pop: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    redraws = kept = 0
    for i in range(n):
        g = draw(i)
        while g in seen and redraws < try_cap:
            redraws += 1
            g = draw(i)
        if g in seen:
            kept += 1
        seen.add(g)
        pop.append(g)
    return pop, {"redraws": redraws, "duplicates_kept": kept}


def template_population(genome, n: int, rng: MT19937RandGen,
                        try_multiple: int = 20) -> tuple[list[tuple[int, ...]], dict]:
    """GP-GOMEA: src/Evolution/PopulationInitializer.cpp
    PopulationInitializer::InitializeTreePopulation

    Individual i is FULL for i <= N / 2 (integer division) and GROW after, both at the full
    height. Plus the duplicate redraw (see the module docstring)."""
    def draw(i: int) -> tuple[int, ...]:
        shape = GROW if i > n // 2 else FULL
        return (generate_tree_grow if shape == GROW else generate_tree_full)(genome, rng)

    return _fill(n, draw, try_multiple * n)


def it_population(genome, n: int, rng: MT19937RandGen,
                  try_multiple: int = 20) -> tuple[list[tuple[int, ...]], dict]:
    """NOT A PORT: ``n`` IT genotypes from ``it_individual``, with the duplicate redraw."""
    return _fill(n, lambda i: it_individual(genome, rng), try_multiple * n)


def diversity(pop: Sequence[tuple[int, ...]], domains: Sequence[int]) -> dict:
    """The start diversity: the positions that are constant in ``pop`` (``LinkageModel.
    calibrate`` leaves them out for the whole run), those of them with more than one code
    (a real loss), and the number of distinct genotypes."""
    constant = [p for p in range(len(domains)) if len({g[p] for g in pop}) == 1]
    return {
        "constant_positions": constant,
        "constant_positions_with_choice": [p for p in constant if domains[p] > 1],
        "unique_genotypes": len(set(pop)),
    }
