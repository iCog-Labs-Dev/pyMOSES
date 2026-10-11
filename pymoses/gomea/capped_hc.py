"""``-a hc-capped``: hill climbing with a per-deme evaluation cap (Phase 1g).

GOMEA extension (not in classic MOSES). The control arm for asking whether
GOMEA's gain inside MOSES comes from short demes rather than from linkage
learning: MOSES's own ``hill_climbing``, built exactly as ``-a hc`` builds it,
is called with ``min(max_evals, gomea_params.max_evals_per_deme)``, the same
cap ``--gomea-max-evals-per-deme`` gives GOMEA. Without a cap it is ``-a hc``.
"""

from __future__ import annotations

from pymoses.gomea.optimizer import gomea_parameters
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)

__all__ = ["hc_capped_optimization"]


class hc_capped_optimization(optimizer_base):
    """Hill climbing whose per-deme budget is capped at
    ``gomea_params.max_evals_per_deme`` (``None`` or 0: no cap)."""

    def __init__(
        self,
        op: optim_parameters | None = None,
        hc: hc_parameters | None = None,
        gp: gomea_parameters | None = None,
    ) -> None:
        super().__init__(op)
        self.hc = hill_climbing(self.opt_params, hc if hc is not None else hc_parameters())
        self.cap = (gp if gp is not None else gomea_parameters()).max_evals_per_deme

    def __call__(self, deme: deme_t, iscorer, max_evals: int, max_time: int) -> None:
        budget = min(max_evals, self.cap) if self.cap else max_evals
        self.hc(deme, iscorer, budget, max_time)
        self.over_budget = self.hc.over_budget
