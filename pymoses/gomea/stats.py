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

from collections import Counter
from collections.abc import Sequence
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
