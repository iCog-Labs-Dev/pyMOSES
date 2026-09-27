"""Build a GOMEA initial population with MOSES's own neighbourhood sampler.

GP-GOMEA-style population optimizers need N distinct legal genotypes to seed
their model-building generation. Classic MOSES already has exactly the
generator a "seed around the exemplar" policy wants:
``sample_from_neighborhood`` (``pymoses/moses/moses/neighborhood_sampling.
py``) draws one instance that differs from a center instance in exactly
``dist`` distinct logical fields, each changed to a different legal value --
see ``docs/gomea/P1.0-AUDIT.md`` Sec 4. This module never invents its own
notion of a "neighbour": every non-default genotype in the population it
returns is literally one draw from that function, called through the
``FrozenSchema`` adapter (``pymoses.gomea.schema``) so distances are counted
in logical fields, not raw packed bits.

Two independent random sources are involved, and the split is deliberate:
``sample_from_neighborhood`` always draws from the process-wide
``pymoses.util.rng.randGen()`` singleton (it has no ``rng`` parameter -- see
that module's own docstring), while every OTHER random decision this module
makes -- which radius to draw at, the coverage-pass field order, which
population member a coverage fix replaces -- goes through the ``rng``
argument ``build_population`` is explicitly given. A caller that wants a
fully reproducible run seeds both.

``build_uniform_population`` is the one exception to "around the
exemplar": the Phase 1d ``--gomea-init uniform`` ablation, which draws the
non-default genotypes uniformly over the whole legal space from ``rng``
alone and never calls the sampler.

No fitness. ``build_population`` never scores a genotype, never imports a
scorer, and never looks at one -- it is a population of DISTINCT LEGAL
GENOTYPES drawn around the exemplar at a mixture of radii, nothing more.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.moses.neighborhood_sampling import sample_from_neighborhood
from pymoses.util.rng import MT19937RandGen

# A radius mixture: each component is either an explicit integer radius or
# the literal string ``"max"`` (meaning "the schema's own dimensionality"),
# paired with a (not necessarily normalised) weight. ``parse_radii`` builds
# one from text; ``resolve_radii`` turns it into a normalised, dim-bound
# ``tuple[tuple[int, float], ...]`` ``build_population`` can actually draw
# from.
RadiusMixture = tuple[tuple[int | str, float], ...]


def parse_radii(text: str) -> RadiusMixture:
    """Parse a ``"1:0.4,2:0.3,4:0.2,max:0.1"``-style radius mixture.

    Each comma-separated component is ``<radius>:<weight>``, where
    ``<radius>`` is either a base-10 integer or the literal ``max`` and
    ``<weight>`` is anything ``float()`` accepts. Raises ``ValueError`` on
    any malformed component (missing ``:``, empty component, an
    unparseable radius or weight) or on empty/whitespace-only text.
    """
    components: list[tuple[int | str, float]] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"malformed radius mixture {text!r}: empty component")
        if ":" not in part:
            raise ValueError(f"malformed radius mixture component {part!r}: missing ':'")
        radius_text, _, weight_text = part.partition(":")
        radius_text = radius_text.strip()
        weight_text = weight_text.strip()

        radius: int | str
        if radius_text == "max":
            radius = "max"
        else:
            try:
                radius = int(radius_text)
            except ValueError as exc:
                raise ValueError(f"malformed radius {radius_text!r} in {part!r}") from exc

        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"malformed weight {weight_text!r} in {part!r}") from exc

        components.append((radius, weight))

    if not components:
        raise ValueError(f"empty radius mixture: {text!r}")
    return tuple(components)


def resolve_radii(mix: RadiusMixture, dim: int) -> tuple[tuple[int, float], ...]:
    """Resolve a parsed radius mixture against one schema's dimensionality.

    ``'max'`` becomes ``dim``; any literal radius greater than ``dim`` is
    clamped down to ``dim`` (a mixture written for a bigger schema still
    makes sense on a smaller one); components that land on the same
    resolved radius are merged by summing their weights; the result is
    sorted by radius and renormalised to sum to 1.

    Raises ``ValueError`` if ``mix`` is empty, if any literal (non-``'max'``)
    radius is less than 1, or if any weight is not strictly positive --
    these are malformed INPUT, not a resolution outcome, so they are
    reported rather than silently dropped. The one case a component *is*
    silently dropped after resolution: a schema with ``dim < 1`` clamps
    even an originally-valid radius down to 0, which is not a usable
    sampler distance -- if every component clamps away like that, nothing
    remains and ``ValueError`` is raised for that instead (indistinguishable
    from an empty ``mix`` from the caller's point of view: either way there
    is no radius left to draw from).
    """
    if not mix:
        raise ValueError("radius mixture must have at least one component")

    resolved: dict[int, float] = {}
    for literal_radius, weight in mix:
        if weight <= 0:
            raise ValueError(f"radius weight must be positive: {weight!r}")

        if literal_radius == "max":
            r = dim
        else:
            if not isinstance(literal_radius, int) or isinstance(literal_radius, bool):
                raise ValueError(f"invalid radius: {literal_radius!r}")
            if literal_radius < 1:
                raise ValueError(f"radius must be >= 1 or 'max': {literal_radius!r}")
            r = literal_radius

        if r > dim:
            r = dim
        if r < 1:
            # Only reachable when dim < 1: a schema with no room for any
            # neighbourhood move at all clamps every component away.
            continue

        resolved[r] = resolved.get(r, 0.0) + weight

    if not resolved:
        raise ValueError("no radius component remains after resolving against dim")

    total = sum(resolved.values())
    return tuple((r, w / total) for r, w in sorted(resolved.items()))


def choose_radius(resolved: tuple[tuple[int, float], ...], rng: MT19937RandGen) -> int:
    """Draw one radius from an already-resolved mixture.

    Inverse-CDF sampling: draw ``u = rng.randdouble()`` (uniform on
    ``[0, 1)``) and walk the cumulative resolved weights, returning the
    first radius whose cumulative weight exceeds ``u``. The final
    component is returned as a floating-point safety net if rounding ever
    leaves the cumulative sum a hair under 1.0.
    """
    u = rng.randdouble()
    cumulative = 0.0
    for radius, weight in resolved:
        cumulative += weight
        if u < cumulative:
            return radius
    return resolved[-1][0]


@dataclass
class InitResult:
    """The outcome of one ``build_population`` call.

    ``genotypes[0]`` is always the default (exemplar) genotype, and
    ``radii[0]`` is always 0 to match it -- every other ``radii[i]`` is the
    sampler distance the current ``genotypes[i]`` was actually drawn at
    (updated in place if a coverage-pass replacement later overwrote that
    slot). ``feasible_short`` is set when the population came back with
    fewer than ``n`` members because the fill phase ran out of its draw
    budget before finding that many distinct legal genotypes -- it is
    never set by the coverage pass, which only replaces existing members,
    never adds or removes any.
    """

    genotypes: list[tuple[int, ...]]
    radii: list[int]
    attempts: int
    duplicates_rejected: int
    coverage_replacements: int
    feasible_short: bool
    diagnostics: dict


def _is_constant_field(population: Sequence[tuple[int, ...]], field_idx: int) -> bool:
    first = population[0][field_idx]
    return all(g[field_idx] == first for g in population)


def _constant_fields(population: Sequence[tuple[int, ...]], dim: int) -> list[int]:
    return [k for k in range(dim) if _is_constant_field(population, k)]


def _fisher_yates_shuffle(order: list[int], rng: MT19937RandGen) -> None:
    """In-place Fisher-Yates shuffle, drawing indices from ``rng``."""
    for i in range(len(order) - 1, 0, -1):
        j = rng.randint_n(i + 1)
        order[i], order[j] = order[j], order[i]


def field_diagnostics(genotypes: Sequence[tuple[int, ...]], domains: Sequence[int]) -> dict:
    """Per-field observed-value/entropy/default-frequency diagnostics.

    ``entropy_bits`` is the base-2 Shannon entropy of the field's observed
    value distribution across ``genotypes`` (0 for a constant field).
    ``default_freq`` is the fraction of ``genotypes`` whose value at that
    field is 0 (the exemplar's own setting -- see ``FrozenSchema.
    default_genotype``).
    """
    n = len(genotypes)
    per_field = []
    for field_idx, domain in enumerate(domains):
        counts = Counter(g[field_idx] for g in genotypes)
        entropy = 0.0
        for c in counts.values():
            p = c / n
            entropy -= p * math.log2(p)
        per_field.append(
            {
                "domain": domain,
                "observed": sorted(counts),
                "entropy_bits": entropy,
                "default_freq": counts.get(0, 0) / n,
            }
        )
    return {
        "per_field": per_field,
        "unique_genotypes": len(set(genotypes)),
        "n": n,
    }


def build_population(
    schema: FrozenSchema,
    n: int,
    mix: RadiusMixture,
    rng: MT19937RandGen,
    dedupe: bool = True,
    attempt_cap_multiple: int = 20,
    coverage_cap_multiple: int = 2,
) -> InitResult:
    """Draw an ``n``-member GOMEA initial population around ``schema``'s
    default (exemplar) genotype, using MOSES's own neighbourhood sampler.

    Slot 0 is always the default genotype itself, at radius 0. Every other
    slot is filled by picking a radius from ``mix`` (resolved against
    ``schema.dim`` via ``resolve_radii``) with ``rng``, then drawing one
    sample from ``sample_from_neighborhood`` at that radius around the
    default instance -- so radius really is Hamming distance to the
    default in logical-field space, and every genotype is legal by
    construction (the sampler never produces an out-of-domain value).

    A duplicate draw is rejected (and counted in ``duplicates_rejected``)
    when ``dedupe`` holds; either way it counts toward ``attempts``. The
    fill phase spends up to ``attempt_cap_multiple * n`` sampler draws
    total; if it still has not reached ``n`` distinct genotypes when that
    budget runs out, the returned population is shorter and
    ``feasible_short`` is set.

    A coverage pass then runs (skipped entirely for a population of size
    1, and for ``schema.dim == 0``): every logical field is visited once,
    in an order fixed by one Fisher-Yates shuffle of ``range(schema.dim)``
    drawn from ``rng``. A field is checked for being constant across the
    CURRENT population at the moment it is visited (not precomputed) --
    this is what lets a field whose constancy was only broken, or only
    created, as a side effect of an earlier field's fix be handled
    correctly without extra bookkeeping: a field fixed early is simply
    found already-varying when its own turn comes, and a field that
    becomes newly constant because a replacement happened to match the
    rest of the population on it is still ahead in the shuffled order and
    gets its own turn. For a field found constant, draws are made (radius
    again chosen from the resolved mixture) until one changes that field's
    value and is not a duplicate, and it replaces a uniformly random
    population member at index 1 or later (never the default at index 0).
    All coverage draws share one budget of ``coverage_cap_multiple * n``
    sampler draws; once it is spent the pass stops wherever it is,
    successful fixes so far are counted in ``coverage_replacements``, and
    the diagnostics record which fields were constant both before and
    after.

    Never scores anything: no scorer is imported, called, or referenced.
    """
    g0 = schema.default_genotype()

    if schema.dim == 0:
        diagnostics = field_diagnostics([g0], schema.domains)
        diagnostics["constant_fields_before_coverage"] = []
        diagnostics["constant_fields_after_coverage"] = []
        diagnostics["radius_histogram"] = {0: 1}
        return InitResult(
            genotypes=[g0],
            radii=[0],
            attempts=0,
            duplicates_rejected=0,
            coverage_replacements=0,
            feasible_short=n > 1,
            diagnostics=diagnostics,
        )

    resolved = resolve_radii(mix, schema.dim)
    center_inst = schema.to_instance(g0)

    population: list[tuple[int, ...]] = [g0]
    radii: list[int] = [0]
    seen: set[tuple[int, ...]] = {g0}

    attempts = 0
    duplicates_rejected = 0
    fill_budget = attempt_cap_multiple * n

    while len(population) < n and attempts < fill_budget:
        r = choose_radius(resolved, rng)
        draw: list = []
        sample_from_neighborhood(schema.fields, r, 1, draw, 1, center_inst)
        attempts += 1
        g = schema.to_genotype(draw[0])
        if dedupe and g in seen:
            duplicates_rejected += 1
            continue
        population.append(g)
        radii.append(r)
        seen.add(g)

    feasible_short = len(population) < n
    constant_before = _constant_fields(population, schema.dim)
    coverage_replacements = 0

    if len(population) > 1:
        order = list(range(schema.dim))
        _fisher_yates_shuffle(order, rng)
        coverage_budget = coverage_cap_multiple * n

        for field_idx in order:
            if coverage_budget <= 0:
                break
            if not _is_constant_field(population, field_idx):
                continue
            target_value = population[0][field_idx]

            while coverage_budget > 0:
                r = choose_radius(resolved, rng)
                draw = []
                sample_from_neighborhood(schema.fields, r, 1, draw, 1, center_inst)
                attempts += 1
                coverage_budget -= 1
                g = schema.to_genotype(draw[0])
                if g[field_idx] == target_value:
                    continue
                if dedupe and g in seen:
                    duplicates_rejected += 1
                    continue

                replace_idx = 1 + rng.randint_n(len(population) - 1)
                old = population[replace_idx]
                if population.count(old) <= 1:
                    seen.discard(old)
                population[replace_idx] = g
                radii[replace_idx] = r
                seen.add(g)
                coverage_replacements += 1
                break

        constant_after = _constant_fields(population, schema.dim)
    else:
        constant_after = constant_before

    diagnostics = field_diagnostics(population, schema.domains)
    diagnostics["constant_fields_before_coverage"] = constant_before
    diagnostics["constant_fields_after_coverage"] = constant_after
    diagnostics["radius_histogram"] = dict(sorted(Counter(radii).items()))

    return InitResult(
        genotypes=population,
        radii=radii,
        attempts=attempts,
        duplicates_rejected=duplicates_rejected,
        coverage_replacements=coverage_replacements,
        feasible_short=feasible_short,
        diagnostics=diagnostics,
    )


INIT_MODES = ("sampler", "uniform")


def build_uniform_population(
    schema: FrozenSchema,
    n: int,
    rng: MT19937RandGen,
    dedupe: bool = True,
    attempt_cap_multiple: int = 20,
) -> InitResult:
    """Phase 1d ablation (``--gomea-init uniform``, hypothesis A): an
    ``n``-member population drawn uniformly over the whole legal genotype
    space instead of around the exemplar.

    Slot 0 is still the default (exemplar) genotype. Every other slot has
    each knob ``k`` drawn independently and uniformly from ``0 ..
    schema.domains[k] - 1`` with ``rng.randint_n`` -- MOSES's neighbourhood
    sampler (and so the process-wide ``randGen()``) is never used. Dedupe,
    the ``attempt_cap_multiple * n`` fill budget and ``feasible_short``
    behave exactly as in ``build_population``. There is no coverage pass
    (``coverage_replacements`` is always 0), but the same diagnostics are
    reported; ``radii[i]`` is the Hamming distance of ``genotypes[i]`` from
    the default genotype, so ``radius_histogram`` stays comparable with
    sampler mode.

    Never scores anything.
    """
    g0 = schema.default_genotype()
    population: list[tuple[int, ...]] = [g0]
    attempts = 0
    duplicates_rejected = 0

    if schema.dim > 0:
        seen: set[tuple[int, ...]] = {g0}
        fill_budget = attempt_cap_multiple * n
        while len(population) < n and attempts < fill_budget:
            g = tuple(rng.randint_n(d) for d in schema.domains)
            attempts += 1
            if dedupe and g in seen:
                duplicates_rejected += 1
                continue
            population.append(g)
            seen.add(g)

    radii = [sum(v != d for v, d in zip(g, g0, strict=True)) for g in population]
    constant = _constant_fields(population, schema.dim)
    diagnostics = field_diagnostics(population, schema.domains)
    diagnostics["constant_fields_before_coverage"] = constant
    diagnostics["constant_fields_after_coverage"] = list(constant)
    diagnostics["radius_histogram"] = dict(sorted(Counter(radii).items()))

    return InitResult(
        genotypes=population,
        radii=radii,
        attempts=attempts,
        duplicates_rejected=duplicates_rejected,
        coverage_replacements=0,
        feasible_short=len(population) < n,
        diagnostics=diagnostics,
    )


def top_up_population(
    schema: FrozenSchema,
    seeds: Sequence[tuple[int, ...]],
    center: tuple[int, ...],
    n: int,
    mix: RadiusMixture,
    rng: MT19937RandGen,
    dedupe: bool = True,
    attempt_cap_multiple: int = 20,
) -> InitResult:
    """Phase 1e (the HC->GOMEA hybrid): an ``n``-member population whose
    first members are ``seeds`` (distinct genotypes the caller already has
    scores for, at most ``n`` of them are kept), topped up with MOSES
    neighbourhood-sampler draws around ``center``.

    With no seeds the population starts with ``center`` itself. Each draw
    picks its radius from ``mix`` with ``rng`` and calls
    ``sample_from_neighborhood`` (so the process-wide ``randGen()``), like
    ``build_population``'s fill phase; dedupe (against the seeds too), the
    ``attempt_cap_multiple * n`` draw budget and ``feasible_short`` behave
    the same. There is no coverage pass: the seeds are the population the
    caller wants the linkage model calibrated on. ``radii[i]`` is the
    Hamming distance of a seed from ``center``, or a draw's sampler radius.

    Never scores anything; the caller decides which members to evaluate.
    """
    population: list[tuple[int, ...]] = list(seeds[:n]) or [tuple(center)]
    radii = [sum(v != c for v, c in zip(g, center, strict=True)) for g in population]
    seen: set[tuple[int, ...]] = set(population)
    attempts = 0
    duplicates_rejected = 0

    if schema.dim > 0 and len(population) < n:
        resolved = resolve_radii(mix, schema.dim)
        center_inst = schema.to_instance(center)
        fill_budget = attempt_cap_multiple * n
        while len(population) < n and attempts < fill_budget:
            r = choose_radius(resolved, rng)
            draw: list = []
            sample_from_neighborhood(schema.fields, r, 1, draw, 1, center_inst)
            attempts += 1
            g = schema.to_genotype(draw[0])
            if dedupe and g in seen:
                duplicates_rejected += 1
                continue
            population.append(g)
            radii.append(r)
            seen.add(g)

    constant = _constant_fields(population, schema.dim)
    diagnostics = field_diagnostics(population, schema.domains)
    diagnostics["constant_fields_before_coverage"] = constant
    diagnostics["constant_fields_after_coverage"] = list(constant)
    diagnostics["radius_histogram"] = dict(sorted(Counter(radii).items()))

    return InitResult(
        genotypes=population,
        radii=radii,
        attempts=attempts,
        duplicates_rejected=duplicates_rejected,
        coverage_replacements=0,
        feasible_short=len(population) < n,
        diagnostics=diagnostics,
    )
