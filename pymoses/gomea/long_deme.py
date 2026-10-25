"""GOMEA extension (not in classic MOSES): long demes for promising exemplars.

Backlog item 19, Phase 1af. With a per-deme cap of 250 and N = 16 almost
every GOMEA deme learns its linkage model once, from the unselected initial
population. This option gives a few demes a much larger cap, so that GOM
runs several generations and relearns linkage from a selected population.

``--gomea-long-deme-cap C`` (0 = off) turns it on; the deme of an exemplar
chosen by the rule gets cap C instead of ``--gomea-max-evals-per-deme``.
The cap is still bounded by what is left of the run's budget, exactly as the
normal cap is (``EvaluationOracle`` stops at ``min(max_evals, cap)``), and
the population size N is unchanged. The rules:

* ``top``: the deme is long when its exemplar's penalized score (the score
  ``metapopulation.select_exemplar`` ranks and samples by) is the highest
  finite penalized score in the metapopulation at the moment the exemplar
  is chosen (ties count as best), and that exemplar (keyed by its printed
  combo tree) has not had a long deme before in this run.
* ``random``: the deme is long with probability ``--gomea-long-deme-prob p``,
  drawn from the run's RNG (``randGen()``), so a seed fixes the choices.
  With p = 0 nothing is drawn, and with p = 1 neither, so p = 0 is the same
  run as the option off.

``LongDemePolicy`` keeps the run's state: the exemplars that have had a
long deme, and the counts of demes and long demes (reported in each deme's
``invocation_start`` telemetry event).
"""

from __future__ import annotations

import math

from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str
from pymoses.util.rng import randGen

LONG_DEME_RULES = ("top", "random")


class LongDemePolicy:
    """Decides, deme by deme, whether a deme is long, and counts them."""

    def __init__(self, cap: int, rule: str = "top", prob: float = 0.0) -> None:
        if cap < 1:
            raise ValueError(f"long deme cap must be >= 1, got {cap}")
        if rule not in LONG_DEME_RULES:
            raise ValueError(f"long deme rule must be one of {LONG_DEME_RULES}, got {rule!r}")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"long deme probability must be in [0, 1], got {prob}")
        self.cap = cap
        self.rule = rule
        self.prob = prob
        self.had_long: set[str] = set()
        self.n_demes = 0
        self.n_long = 0

    @staticmethod
    def is_best(mp, exemplar) -> bool:
        """Whether ``exemplar``'s penalized score is finite and no lower than
        any finite penalized score in ``mp``."""
        sc = exemplar.get_penalized_score()
        if not math.isfinite(sc):
            return False
        return all(
            not math.isfinite(t.get_penalized_score()) or t.get_penalized_score() <= sc
            for t in mp.get_trees()
        )

    def decide(self, mp, exemplar) -> bool:
        """Whether the deme about to be built from ``exemplar`` (just chosen
        from ``mp``) is long. Counts the deme, and the long deme if it is."""
        if self.rule == "top":
            key = tree_to_stream_str(exemplar.get_tree())
            long = key not in self.had_long and self.is_best(mp, exemplar)
            if long:
                self.had_long.add(key)
        elif self.prob <= 0.0:
            long = False  # no draw: p = 0 is the same run as the option off
        elif self.prob >= 1.0:
            long = True
        else:
            long = randGen().randdouble() < self.prob
        self.n_demes += 1
        self.n_long += long
        return long
