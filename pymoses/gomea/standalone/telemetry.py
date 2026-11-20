"""NOT A PORT: the IT linkage study (BACKLOG 13.9). The linkage telemetry of each generation
(plan section 6.3), one JSON record per generation in ``events.jsonl``.

Each "generation" record has:

* ``best_train_nmse``, ``median_train_nmse``: of the population, from the search scores
  (100 MSE / var(y') on the training rows);
* ``n_evals``, ``cache_hits``, ``noop_skips``, ``phenotype_skips`` (run totals so far);
* ``gom``: the generation's GOM counters (``GomStats``: ``accepted_improving``,
  ``accepted_neutral``, ``fi_events``, ...);
* ``fos_size``, ``fos_mean_subset_size``, ``fos_hash`` and ``fos_jaccard_prev`` (the Jaccard
  similarity of the FOS, as a set of subsets, to the FOS of the previous generation; ``None``
  in the first generation);
* ``mi_top_pairs`` (modes ``lt-mi`` and ``lt-b``): the 20 position pairs with the highest
  similarity in the matrix that the linkage tree used (``LinkageModel.last_similarity``), as
  ``[i, j, s]``, and ``mi_top_aligned_share``, the share of them that are structurally related
  (IT: the same term slot, which includes the same factor; template: parent and child, or
  siblings);
* ``trials_by_subset_size`` (outcome counts of the GOM trials for each FOS subset size, from
  ``gom``'s ``on_trial``) and ``accepted_improving_by_subset_size``;
* ``active_share``: the mean share of active positions in the population;
* IT only: ``slot_dispersion_median`` and ``slot_dispersion_max`` (for each distinct monomial
  in the population, the number of term slots that hold it somewhere in the population),
  ``distinct_monomials`` and ``forced_negative_population`` (negative powers made positive by
  the zero rule, summed over the population).

The telemetry only reads. It draws nothing and changes no score.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence


def _nmse(score: float, var_y: float) -> float:
    if not math.isfinite(score):
        return math.inf
    return 100.0 * (-score) / var_y


def fos_hash(fos: Sequence[Sequence[int]]) -> str:
    """A short hash of the FOS as a set of subsets (order free)."""
    canon = sorted(sorted(int(p) for p in s) for s in fos)
    return hashlib.sha1(json.dumps(canon).encode()).hexdigest()[:16]


def jaccard(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """|A and B| / |A or B| of two FOS as sets of subsets."""
    sa = {frozenset(s) for s in a}
    sb = {frozenset(s) for s in b}
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def top_pairs(positions: Sequence[int], S: Sequence[Sequence[float]], k: int = 20) -> list:
    """The ``k`` pairs with the highest similarity, as ``[i, j, s]`` (genome positions,
    i < j); ties by the lower positions."""
    pairs = []
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            i, j = positions[a], positions[b]
            if i > j:
                i, j = j, i
            pairs.append((-S[a][b], i, j))
    pairs.sort()
    return [[i, j, -s] for s, i, j in pairs[:k]]


class LinkageTelemetry:
    """Writes the per-generation records. ``related(p, q)`` is the structural relation of
    the genome (``positions.template_related`` or ``positions.it_related``)."""

    def __init__(self, genome, ev, var_y: float, related: Callable[[int, int], bool],
                 out=None) -> None:
        self.genome = genome
        self.ev = ev
        self.var_y = var_y
        self.related = related
        self.out = out
        self.engine = None
        self._prev_fos: list | None = None
        self._trials: dict[int, Counter] = defaultdict(Counter)
        self._t0 = time.monotonic()
        self.records: list[dict] = []

    def on_trial(self, size: int, outcome: str) -> None:
        self._trials[size][outcome] += 1

    def write(self, rec: dict) -> None:
        self.records.append(rec)
        if self.out is not None:
            self.out.write(json.dumps(rec, sort_keys=True) + "\n")
            self.out.flush()

    def on_generation(self, record: dict, pop: Sequence) -> None:
        ev = self.ev
        nm = sorted(_nmse(ind.score, self.var_y) for ind in pop)
        rec = {
            "type": "generation",
            "generation": record["generation"],
            "n_evals": ev.n_evals,
            "cache_hits": ev.cache_hits,
            "noop_skips": ev.noop_skips,
            "phenotype_skips": ev.phenotype_skips,
            "best_train_nmse": nm[0],
            "median_train_nmse": statistics.median(nm),
            "gom": dict(record["stats"]),
            "fos_size": record["fos_size"],
            "fos_mean_subset_size": record["fos_mean_subset_size"],
            "linkage_s": record["linkage_s"],
            "variation_s": record["variation_s"],
            "wall_s": time.monotonic() - self._t0,
        }
        model = getattr(self.engine, "model", None)
        fos = getattr(model, "last_fos", None)
        if fos is not None:
            rec["fos_hash"] = fos_hash(fos)
            rec["fos_jaccard_prev"] = None if self._prev_fos is None else jaccard(fos,
                                                                                  self._prev_fos)
            self._prev_fos = [list(s) for s in fos]
        sim = getattr(model, "last_similarity", None)
        if sim is not None:
            positions, S = sim
            top = top_pairs(positions, S)
            rec["mi_top_pairs"] = top
            rec["mi_top_aligned_share"] = (
                sum(self.related(i, j) for i, j, _ in top) / len(top) if top else None)
        rec["trials_by_subset_size"] = {str(k): dict(v) for k, v in sorted(self._trials.items())}
        rec["accepted_improving_by_subset_size"] = {
            str(k): v.get("improving", 0) for k, v in sorted(self._trials.items())}
        self._trials = defaultdict(Counter)
        g = self.genome
        rec["active_share"] = (sum(sum(g.active_mask(ind.genotype)) for ind in pop)
                               / (len(pop) * g.length))
        if g.kind == "it":
            where: dict = defaultdict(set)
            forced = 0
            for ind in pop:
                m = g.decode(ind.genotype)
                forced += m.forced
                for t, mono in enumerate(m.slots):
                    if mono is not None:
                        where[mono].add(t)
            disp = [len(s) for s in where.values()]
            rec["slot_dispersion_median"] = statistics.median(disp) if disp else None
            rec["slot_dispersion_max"] = max(disp) if disp else None
            rec["distinct_monomials"] = len(disp)
            rec["forced_negative_population"] = forced
        self.write(rec)
