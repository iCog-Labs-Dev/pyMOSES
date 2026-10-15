"""Entropy and similarity statistics for GOMEA linkage learning.

Ported from GP-GOMEA (https://github.com/marcovirgolin/GP-GOMEA, commit
750220c, ``src/GOMEA/GOMEAFOS.cpp``, Apache-2.0, Copyright Marco Virgolin).
See ``docs/gomea/PROVENANCE.md`` for the register entry and
``tests/gomea/test_golden_parity.py`` for the parity proof against a harness
that compiles the upstream code verbatim.

A genotype is a tuple of small non-negative integer codes, one per logical
field. Codes are only meaningful within their own field (code ``k`` in one
field is unrelated to code ``k`` in another), so frequencies are counted per
field and per field pair only. Entropies use the natural log, like upstream.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Sequence
from math import log
from operator import add


def _entropy_of_counts(counts: Counter, pop_size: int) -> float:
    # Upstream sums over the (k, l) frequency table in row-major code order;
    # iterating the sorted keys reproduces that order, so the result is
    # bit-identical to the golden harness (which matters for exact ties).
    h = 0.0
    for key in sorted(counts):
        freq = counts[key] / pop_size
        h += -freq * log(freq)
    return h


def entropy_matrix(
    pop: Sequence[Sequence[int]], positions: Sequence[int]
) -> list[list[float]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:303 GOMEAFOS::GenerateLinkageTreeFOS

    The frequency / entropy loops of that function (lines 386-426):
    ``H[i][i]`` is the marginal entropy of ``positions[i]`` and ``H[i][j]``
    (i != j) the joint entropy of ``positions[i]`` and ``positions[j]``,
    natural log. Adaptation: the population is already integer coded, so
    upstream's symbol encoding and constant binning (lines 312-380) have no
    counterpart. A pair ``(a, b)`` is counted as the single key
    ``a * K + b`` (``K`` = 1 + the largest code), whose numeric order is
    upstream's ``(k, l)`` row-major order.
    """
    positions = list(positions)
    d = len(positions)
    pop_size = len(pop)
    H = [[0.0] * d for _ in range(d)]
    if d == 0 or pop_size == 0:
        return H
    columns = [[row[p] for row in pop] for p in positions]
    K = 1 + max(max(col) for col in columns)
    for i in range(d):
        col_i = columns[i]
        scaled = [v * K for v in col_i]
        row = H[i]
        for j in range(i + 1, d):
            h = _entropy_of_counts(Counter(map(add, scaled, columns[j], strict=True)), pop_size)
            row[j] = h
            H[j][i] = h
        row[i] = _entropy_of_counts(Counter(col_i), pop_size)
    return H


def marginal_entropies(
    pop: Sequence[Sequence[int]], positions: Sequence[int]
) -> list[float]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:303 GOMEAFOS::GenerateLinkageTreeFOS

    The diagonal of ``entropy_matrix(pop, positions)`` only: entry ``i`` is
    bit-identical to ``entropy_matrix(pop, [positions[i]])[0][0]`` (the same
    counts, summed by ``_entropy_of_counts`` in the same key order).
    Port-only optimisation: one pass over the population instead of one
    ``entropy_matrix`` call per position, which ``LinkageModel.calibrate``
    used to make.
    """
    positions = list(positions)
    pop_size = len(pop)
    if not positions or pop_size == 0:
        return [0.0] * len(positions)
    columns = list(zip(*pop, strict=True))
    return [_entropy_of_counts(Counter(columns[p]), pop_size) for p in positions]


def mi_similarity(H: Sequence[Sequence[float]]) -> list[list[float]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:303 GOMEAFOS::GenerateLinkageTreeFOS

    The ordinary mutual-information branch (lines 430-437), which is dead
    upstream (``if (false)``) at this commit and is enabled here as the LT-MI
    control arm: ``S[i][j] = H[i][i] + H[j][j] - H[i][j]`` for i != j; the
    diagonal keeps ``H[i][i]`` (upstream leaves it untouched, and the tree
    builder ignores it).
    """
    d = len(H)
    S = [[0.0] * d for _ in range(d)]
    for i in range(d):
        S[i][i] = H[i][i]
        for j in range(i + 1, d):
            s = H[i][i] + H[j][j] - H[i][j]
            S[i][j] = s
            S[j][i] = s
    return S


def biased_similarity(
    H: Sequence[Sequence[float]], H0: Sequence[Sequence[float]]
) -> list[list[float]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:303 GOMEAFOS::GenerateLinkageTreeFOS

    The live, initialization-biased branch (lines 454-484): the adjustments
    ``1/H0[i][i]`` (diagonal) and ``2/H0[i][j]`` (off-diagonal) are taken from
    the first population's entropies ``H0`` and multiplied in, then
    ``S[i][j] = H[i][i]/H0[i][i] + H[j][j]/H0[j][j] - 2 H[i][j]/H0[i][j]`` and
    ``S[i][i] = H[i][i]/H0[i][i]``. The arithmetic follows upstream's
    reciprocal-then-multiply form exactly, so results are bit-identical.
    Negative values are kept (no clamping). Adaptation: upstream divides by
    ``H0 = 0`` unguarded (producing inf/NaN); here any used ``H0`` entry that
    is not > 0 raises ``ValueError`` (``LinkageModel`` excludes such fields).
    """
    d = len(H)
    if len(H0) != d:
        raise ValueError("H and H0 must have the same size")
    adj = [[0.0] * d for _ in range(d)]
    for i in range(d):
        if not H0[i][i] > 0.0:
            raise ValueError(f"H0[{i}][{i}] must be > 0, got {H0[i][i]!r}")
        adj[i][i] = 1.0 / H0[i][i]
        for j in range(i + 1, d):
            if not H0[i][j] > 0.0:
                raise ValueError(f"H0[{i}][{j}] must be > 0, got {H0[i][j]!r}")
            adj[i][j] = 2.0 / H0[i][j]
    diag = [H[i][i] * adj[i][i] for i in range(d)]
    S = [[0.0] * d for _ in range(d)]
    for i in range(d):
        S[i][i] = diag[i]
        for j in range(i + 1, d):
            s = diag[i] + diag[j] - H[i][j] * adj[i][j]
            S[i][j] = s
            S[j][i] = s
    return S


# GOMEA extension: how constants (contin genes, ``--gomea-contin copy``) enter
# linkage learning -- Virgolin et al. 2021, Sec 6: ``all`` = all-const,
# ``none`` = no-const, ``bin`` = bin-const (the paper's choice, gamma = 100).
CONTIN_STATS_MODES = ("all", "none", "bin")
MAX_CONSTANTS = 100


def contin_symbols(
    pop: Sequence[Sequence[int]],
    positions: Sequence[int],
    value: Callable[[int, int], float],
    mode: str,
    max_constants: int = MAX_CONSTANTS,
) -> list[tuple[int, ...]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:303 GOMEAFOS::GenerateLinkageTreeFOS

    The symbol encoding of constants (lines 312-380), as a recoding of the
    contin genes at ``positions`` (``value(p, code)`` is the constant gene
    ``p`` with ``code`` decodes to) ahead of ``entropy_matrix``. Other
    positions are returned unchanged. ``mode``:

    * ``"all"`` (all-const): every distinct constant is its own symbol;
    * ``"none"`` (no-const): every constant is the same symbol, so a
      constant carries no linkage information (``LinkageModel`` then keeps
      the gene out of the tree, as a singleton subset);
    * ``"bin"`` (bin-const, upstream's live path): the first
      ``max_constants`` distinct constants met in a row-major traversal
      (individual by individual, ``positions`` in order, like upstream's
      node loop) keep their own symbol; every other constant takes the
      symbol of the nearest of them (ties to the lower constant), found by
      binary search as the paper describes.

    A symbol is the rank of its (binned) constant among all the distinct
    (binned) constants of the population, shared by every contin gene like
    upstream's single symbol table, so below ``max_constants`` distinct
    constants ``"bin"`` equals ``"all"`` exactly.

    Adaptations from the upstream code: upstream starts ``least_dist`` at
    the first retained constant's value rather than at its distance to the
    query (so its sorted-order early stop can end on the wrong constant),
    and stores the result
    with ``v = constants_v[least_idx]``, which assigns a ``double`` to a
    ``std::string`` through ``operator=(char)`` -- so binned constants
    collapse into one-character symbols unrelated to the retained ones.
    None of that is ported: this is the paper's rule (the true nearest
    retained constant, whose symbol a binned constant shares).
    """
    if mode not in CONTIN_STATS_MODES:
        raise ValueError(f"contin stats mode must be one of {CONTIN_STATS_MODES}, got {mode!r}")
    positions = list(positions)
    if not positions:
        return [tuple(row) for row in pop]
    if mode == "none":
        symbol_of = None
    else:
        values = [[value(p, row[p]) for p in positions] for row in pop]
        if mode == "bin":
            retained: dict[float, None] = {}
            for vals in values:
                for v in vals:
                    if len(retained) >= max_constants:
                        break
                    retained.setdefault(v)
            kept = sorted(retained)

            def nearest(v: float) -> float:
                if v in retained:
                    return v
                i = bisect_left(kept, v)
                if i == 0:
                    return kept[0]
                if i == len(kept):
                    return kept[-1]
                lo, hi = kept[i - 1], kept[i]
                return lo if v - lo <= hi - v else hi

            values = [[nearest(v) for v in vals] for vals in values]
        rank = {v: r for r, v in enumerate(sorted({v for vals in values for v in vals}))}
        symbol_of = [[rank[v] for v in vals] for vals in values]
    out = []
    for i, row in enumerate(pop):
        new = list(row)
        for j, p in enumerate(positions):
            new[p] = 0 if symbol_of is None else symbol_of[i][j]
        out.append(tuple(new))
    return out
