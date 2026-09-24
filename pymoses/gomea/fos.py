"""Family-of-subsets (FOS) builders and the linkage model for GOMEA.

Ported from GP-GOMEA (https://github.com/marcovirgolin/GP-GOMEA, commit
750220c, ``src/GOMEA/GOMEAFOS.cpp``, Apache-2.0, Copyright Marco Virgolin).
See ``docs/gomea/PROVENANCE.md`` for the register entry and
``tests/gomea/test_golden_parity.py`` for the parity proof against a harness
that compiles the upstream code verbatim.

Indices inside ``linkage_tree_fos`` / ``random_tree_fos`` /
``univariate_fos`` are local (``0 .. d-1``); ``LinkageModel`` maps them back
to genotype indices.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymoses.gomea.stats import biased_similarity, entropy_matrix, mi_similarity
from pymoses.util.rng import MT19937RandGen


def univariate_fos(d: int) -> list[list[int]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:532 GOMEAFOS::GenerateUnivariateFOS

    All singletons in index order. ``fos_no_root_swap`` is not supported
    (it is off in every pyMoses configuration).
    """
    return [[i] for i in range(d)]


def random_tree_fos(d: int, rng: MT19937RandGen) -> list[list[int]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:545 GOMEAFOS::GenerateRandomTreeFOS

    Singletons, then random pairwise merges while more than two clusters
    remain (so no root is ever built). Adaptation: armadillo's ``randu()``
    is replaced by ``rng.randdouble()``, and ``size_t idx = randu() * size``
    by ``int(rng.randdouble() * size)``. ``randdouble()`` can return exactly
    1.0 (``randu()`` cannot), so the index is capped at ``size - 1``.
    ``fos_no_root_swap`` is not supported.
    """
    fos = [[i] for i in range(d)]
    clone = [list(s) for s in fos]
    while len(clone) > 2:
        size = len(clone)
        first = min(int(rng.randdouble() * size), size - 1)
        while True:
            second = min(int(rng.randdouble() * size), size - 1)
            if second != first:
                break
        if first > second:
            first, second = second, first
        merged = clone[first] + clone[second]
        del clone[first]
        del clone[second - 1]
        clone.append(merged)
        fos.append(list(merged))
    return fos


def _determine_nearest_neighbour(
    index: int, S: list[list[float]], mpm_number_of_indices: list[int], mpm_length: int
) -> int:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:289 GOMEAFOS::DetermineNearestNeighbour

    Unchanged. ``result`` starts at an index (not a similarity of 0), so
    negative similarities rank correctly; ties prefer the smaller cluster,
    then the lower index.
    """
    result = 0
    if result == index:
        result += 1
    row = S[index]
    for i in range(1, mpm_length):
        if (
            row[i] > row[result]
            or (row[i] == row[result] and mpm_number_of_indices[i] < mpm_number_of_indices[result])
        ) and i != index:
            result = i
    return result


def linkage_tree_fos(S: Sequence[Sequence[float]], order: Sequence[int]) -> list[list[int]]:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:41 GOMEAFOS::BuildLinkageTreeFromSimilarityMatrix

    UPGMA via reciprocal nearest-neighbour chains, returning all ``2d - 1``
    sets (singletons in ``order``, then merges in merge order; the last one
    is the root). Adaptations: ``order`` replaces the ``random_shuffle``d
    ``random_order``; the chain start ``(int) randu() * mpm_length`` is
    always 0 upstream (the cast binds first) and is 0 here; ``d == 1``
    returns ``[[order[0]]]`` and ``d == 0`` returns ``[]`` (upstream would
    read out of bounds). The chain tie rule, the ``NN_chain_length >
    number_of_nodes`` break, the ``r1 < mpm_length`` guard, the
    cardinality-weighted merge and the mpm compaction are unchanged.
    """
    number_of_nodes = len(order)
    if number_of_nodes == 0:
        return []
    if number_of_nodes == 1:
        return [[order[0]]]

    mpm = [[order[i]] for i in range(number_of_nodes)]
    mpm_number_of_indices = [1] * number_of_nodes
    mpm_length = number_of_nodes

    fos: list[list[int]] = [list(m) for m in mpm]

    S_matrix = [[S[mpm[i][0]][mpm[j][0]] for j in range(mpm_length)] for i in range(mpm_length)]
    for i in range(mpm_length):
        S_matrix[i][i] = 0.0

    NN_chain = [0] * (number_of_nodes + 2)
    NN_chain_length = 0
    nn = _determine_nearest_neighbour

    while True:
        if NN_chain_length == 0:
            NN_chain[0] = 0  # upstream: (int) randu() * mpm_length == 0
            NN_chain_length = 1

        while NN_chain_length < 3:
            NN_chain[NN_chain_length] = nn(
                NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length
            )
            NN_chain_length += 1

        while NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]:
            NN_chain[NN_chain_length] = nn(
                NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length
            )
            last = NN_chain[NN_chain_length - 1]
            if (
                S_matrix[last][NN_chain[NN_chain_length]]
                == S_matrix[last][NN_chain[NN_chain_length - 2]]
                and NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]
            ):
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2]
            NN_chain_length += 1
            if NN_chain_length > number_of_nodes:
                break

        r0 = NN_chain[NN_chain_length - 2]
        r1 = NN_chain[NN_chain_length - 1]
        if r0 > r1:
            r0, r1 = r1, r0
        NN_chain_length -= 3

        # Required for exceptional cases in which the nearest-neighbour
        # ordering has changed within the chain while merging within it.
        if r1 < mpm_length:
            n0 = mpm_number_of_indices[r0]
            n1 = mpm_number_of_indices[r1]
            indices = mpm[r0][:n0] + mpm[r1][:n1]
            fos.append(list(indices))

            mul0 = float(n0) / (float(n0) + n1)
            mul1 = float(n1) / (float(n0) + n1)
            for i in range(mpm_length):
                if i != r0 and i != r1:
                    v = mul0 * S_matrix[i][r0] + mul1 * S_matrix[i][r1]
                    S_matrix[i][r0] = v
                    S_matrix[r0][i] = v

            mpm_new_length = mpm_length - 1
            mpm_new = mpm[:mpm_new_length]
            mpm_new_number_of_indices = mpm_number_of_indices[:mpm_new_length]
            mpm_new[r0] = indices
            mpm_new_number_of_indices[r0] = n0 + n1
            if r1 < mpm_length - 1:
                mpm_new[r1] = mpm[mpm_length - 1]
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length - 1]
                for i in range(r1):
                    S_matrix[i][r1] = S_matrix[i][mpm_length - 1]
                    S_matrix[r1][i] = S_matrix[i][r1]
                for j in range(r1 + 1, mpm_new_length):
                    S_matrix[r1][j] = S_matrix[j][mpm_length - 1]
                    S_matrix[j][r1] = S_matrix[r1][j]

            for i in range(NN_chain_length):
                if NN_chain[i] == mpm_length - 1:
                    NN_chain[i] = r1
                    break

            mpm = mpm_new
            mpm_number_of_indices = mpm_new_number_of_indices
            mpm_length = mpm_new_length

            if mpm_length == 1:
                break

    return fos


class LinkageModel:
    """GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:22 GOMEAFOS::GenerateFOS

    Dispatches to the univariate (``'uni'``), random-tree (``'rt'``) or
    linkage-tree FOS, the latter with ordinary MI (``'lt-mi'``, upstream's
    dead branch) or the initialization-biased similarity (``'lt-b'``, the
    live upstream path). Adaptations: ``calibrate`` plays the role of
    upstream's first call (``accu(MI_distribution_adjustments) == 0``): it
    fixes ``H0`` once, and fields with ``H0 == 0`` (constant in the first
    population) are excluded from linkage altogether instead of dividing by
    zero. ``random_shuffle`` is replaced by a Fisher-Yates permutation drawn
    from the seeded MT19937 ``rng``. The linkage-tree root is dropped
    (``FOS.pop_back()``), except that ``d == 1`` keeps its single singleton.
    """

    MODES = ("uni", "rt", "lt-mi", "lt-b")

    def __init__(self, mode: str, rng: MT19937RandGen) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown linkage mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.rng = rng
        self.search_positions: tuple[int, ...] = ()
        self.constant_positions: tuple[int, ...] = ()
        self._H0: list[list[float]] | None = None
        self._length: int | None = None

    def calibrate(self, pop: Sequence[tuple[int, ...]], domains: Sequence[int]) -> None:
        """Fix the search positions and ``H0`` from the initial population.

        Must be called exactly once, before ``learn``. ``domains[p]`` is the
        number of legal codes at genotype index ``p``; every value is checked
        against it.
        """
        if self._length is not None:
            raise RuntimeError("LinkageModel.calibrate may only be called once")
        length = len(domains)
        self._check_pop(pop, length)
        for row in pop:
            for p, (v, k) in enumerate(zip(row, domains, strict=True)):
                if not 0 <= v < k:
                    raise ValueError(f"value {v} at position {p} is outside domain {k}")
        marginals = [entropy_matrix(pop, [p])[0][0] for p in range(length)]
        search = tuple(p for p in range(length) if marginals[p] > 0.0)
        self.search_positions = search
        self.constant_positions = tuple(p for p in range(length) if marginals[p] <= 0.0)
        self._H0 = entropy_matrix(pop, search)
        self._length = length

    def learn(self, pop: Sequence[tuple[int, ...]]) -> list[list[int]]:
        """The FOS for ``pop`` over genotype indices (root excluded)."""
        if self._length is None or self._H0 is None:
            raise RuntimeError("LinkageModel.learn called before calibrate")
        self._check_pop(pop, self._length)
        search = self.search_positions
        d = len(search)
        if d == 0:
            return []
        if d == 1:
            return [[search[0]]]
        if self.mode == "uni":
            local = univariate_fos(d)
        elif self.mode == "rt":
            local = random_tree_fos(d, self.rng)
        else:
            H = entropy_matrix(pop, search)
            S = mi_similarity(H) if self.mode == "lt-mi" else biased_similarity(H, self._H0)
            local = linkage_tree_fos(S, self._random_order(d))[:-1]
        return [[search[i] for i in s] for s in local]

    def _random_order(self, d: int) -> list[int]:
        order = list(range(d))
        for i in range(d - 1, 0, -1):
            j = self.rng.randint_n(i + 1)
            order[i], order[j] = order[j], order[i]
        return order

    @staticmethod
    def _check_pop(pop: Sequence[tuple[int, ...]], length: int) -> None:
        if not pop:
            raise ValueError("population must not be empty")
        for row in pop:
            if len(row) != length:
                raise ValueError(f"genotype length {len(row)} != expected {length}")
