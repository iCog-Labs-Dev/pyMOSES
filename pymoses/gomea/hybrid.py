"""``-a hc-gomea``: hill climbing, then GOMEA on the same deme (Phase 1e).

GOMEA extension (not in classic MOSES). Per deme, behind the same optimizer
seam as every other optimizer (``__call__(deme, iscorer, max_evals,
max_time)``):

1. MOSES's own ``hill_climbing`` runs on the deme with the full budget,
   built from ``opt_params``/``hc_params`` exactly as ``-a hc`` builds it.
2. If hill climbing stopped on the budget, the time limit or the target, the
   deme is hill climbing's and nothing else happens: no RNG draw, no
   evaluation, so the run stays byte-identical to ``-a hc``.
3. Otherwise ``gomea_optimization.run_seeded`` continues the SAME deme with
   what is left of the budget: every instance hill climbing evaluated is
   preloaded into the GOMEA oracle (never charged, scored or appended
   again), the initial population is the best distinct ones among them,
   and new evaluations are appended to the deme. ``deme.n_evals`` ends as
   hill climbing's count plus GOMEA's charged evaluations.
"""

from __future__ import annotations

import time

from pymoses.gomea.optimizer import BUDGET_REASONS, gomea_optimization, gomea_parameters
from pymoses.gomea.schema import FrozenSchema
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)

__all__ = ["HC_STOP_REASONS", "hc_gomea_optimization"]

# Why hill climbing's part of a deme ended. ``local_optimum`` is every stop
# that is not the budget, the time limit or the target (no improvement,
# maximum search distance, single step): the only one that hands over.
HC_STOP_REASONS = ("budget", "time_limit", "target", "local_optimum")


class hc_gomea_optimization(optimizer_base):
    """Hill climbing followed, at a local optimum, by GOMEA on the same deme.

    ``last_result`` describes the most recent call: ``hc_evals``,
    ``hc_stop`` (one of ``HC_STOP_REASONS``), ``gomea_ran``,
    ``gomea_evals`` (charged), ``gomea_termination_reason`` and ``gomea``
    (the GOMEA phase's own ``last_result``, or ``None``).
    """

    def __init__(
        self,
        op: optim_parameters | None = None,
        hc: hc_parameters | None = None,
        gp: gomea_parameters | None = None,
    ) -> None:
        super().__init__(op)
        self.hc = hill_climbing(self.opt_params, hc if hc is not None else hc_parameters())
        self.gomea = gomea_optimization(self.opt_params, gp)
        self.gomea_params = self.gomea.gomea_params
        self.last_result: dict | None = None

    def __call__(self, deme: deme_t, iscorer, max_evals: int, max_time: int) -> None:
        """Fill the empty ``deme`` (hill climbing, then maybe GOMEA) and set
        ``deme.n_evals``. Raises ``UnsupportedRepresentation`` for a
        representation with contin fields before anything is evaluated,
        unless ``gomea_params.contin`` is ``"copy"``."""
        schema = FrozenSchema(iscorer.get_rep(), self.gomea_params.contin)

        t0 = time.monotonic()
        self.hc(deme, iscorer, max_evals, max_time)
        hc_elapsed = time.monotonic() - t0
        hc_evals = deme.n_evals
        hc_stop = self._hc_stop(deme, hc_evals, max_evals)

        gomea_ran = hc_stop == "local_optimum"
        gomea_result = None
        if gomea_ran:
            self.gomea.run_seeded(
                deme, iscorer, schema, max_evals - hc_evals, max_time - hc_elapsed
            )
            gomea_result = self.gomea.last_result

        gomea_evals = gomea_result["ledger"]["requests"] if gomea_result else 0
        gomea_gens = gomea_result["generations"] if gomea_result else 0
        gomea_reason = gomea_result["termination_reason"] if gomea_result else None

        self.nsteps = self.hc.nsteps + gomea_gens
        self.total_steps += self.hc.nsteps + gomea_gens
        self.total_evals += hc_evals + gomea_evals
        self.over_budget = self.hc.over_budget or gomea_reason in BUDGET_REASONS
        self.demeID = deme.getID()

        self.last_result = {
            "hc_evals": hc_evals,
            "hc_stop": hc_stop,
            "gomea_ran": gomea_ran,
            "gomea_evals": gomea_evals,
            "gomea_termination_reason": gomea_reason,
            "gomea": gomea_result,
        }
        TelemetrySink(self.gomea_params.telemetry_path).emit(
            "hybrid",
            deme_id=deme.getID(),
            max_evals=max_evals,
            **{k: v for k, v in self.last_result.items() if k != "gomea"},
        )

    def _hc_stop(self, deme: deme_t, hc_evals: int, max_evals: int) -> str:
        """Why hill climbing stopped, from what it left behind (hill
        climbing itself is not modified): ``over_budget`` covers the
        evaluation budget and the time limit; the target is the deme's best
        raw score reaching ``terminate_if_gte``, the same test hill climbing
        makes."""
        if hc_evals >= max_evals:
            return "budget"
        if self.hc.over_budget:
            return "time_limit"
        best_raw = max((si.second.get_score() for si in deme), default=None)
        if best_raw is not None and self.opt_params.terminate_if_gte <= best_raw:
            return "target"
        return "local_optimum"
