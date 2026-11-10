"""``-a hc --const-opt lm``: hill climbing with LM at the end of each deme (server round 10).

NOT A PORT (pyMoses extension; no C++ counterpart). See
``docs/gomea/SERVER-ROUND10-PREREGISTRATION.md`` ("Code change"). Per deme:

1. MOSES's own ``hill_climbing`` runs on the deme with the deme's budget, built from
   ``opt_params``/``hc_params`` exactly as ``-a hc`` builds it (``--hc-widen-search``
   included).
2. LM runs on the deme's best program, with the same rule and the same code as GOMEA's
   ``--const-opt lm`` (``gomea_optimization._tune_best``, borrowed here, not copied): the
   tuned program replaces the old one in the deme if the training error is lower, and with
   ``--const-opt-guard val`` only if the validation NMSE is not worse.
3. Each LM pass (and each validation pass of the guard, and the final score of a tuned
   program) counts as one evaluation. The evaluations come from the deme's budget, which
   ``deme_expander`` takes from what is left of ``-m``; they are added to ``deme.n_evals``,
   so MOSES's ``n_evals`` includes them and never passes ``-m``.

The deme's best program is the first entry with the highest penalized score, the order
``-a hc`` itself climbs by. With ``--gomea-log`` each deme writes one "end" event (the
fields ``phase_report.py`` and ``moses_sr_runner.const_opt_summary`` read: ``ledger``,
``generations``, ``termination_reason``, ``const_opt``) plus ``hc_evals``.

LM reserve with widen search (user, 2026-11-10): with ``--hc-widen-search 1`` HC ends a deme
only at the max search distance or at the budget, so it can use the whole deme budget and
leave LM nothing. With widen search only, the climb gets ``max_evals - lm_reserve(max_evals)``
and LM gets the rest of the deme budget; the "end" event then also has ``lm_reserve``. If LM
leaves evaluations unused, MOSES starts the next deme as usual. Without widen search the climb
gets the full deme budget, as before.

Without ``--const-opt`` this class is never built and ``-a hc`` is unchanged
(``tests/gomea/test_hc_lm_identity.py``).
"""

from __future__ import annotations

import time

from pymoses.gomea.hybrid import hc_gomea_optimization
from pymoses.gomea.optimizer import (
    _new_const_opt_counts,
    gomea_optimization,
    gomea_parameters,
)
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymoses.moses.representation.instance import instance

__all__ = ["LM_RESERVE", "hc_lm_optimization", "lm_reserve"]

# The largest number of evaluations of a widen-search deme that the climb leaves for LM.
LM_RESERVE = 500


def lm_reserve(max_evals: int) -> int:
    """The evaluations a widen-search deme keeps for LM: min(LM_RESERVE, max_evals // 2)."""
    return min(LM_RESERVE, max_evals // 2)


class _Ledger:
    def __init__(self, requests: int) -> None:
        self.requests = requests
        self.tuning_requests = 0


class _HcDemeOracle:
    """The part of ``EvaluationOracle`` that ``_tune_best`` uses, for one finished HC deme.
    ``best_genotype`` is the deme's best instance; ``charge_tuning`` charges one evaluation
    against the deme's budget and its time limit."""

    def __init__(self, deme: deme_t, iscorer, max_evals: int, max_time: float,
                 start: float) -> None:
        self.iscorer = iscorer
        self.ledger = _Ledger(deme.n_evals)
        self.max_evals = max_evals
        self.max_time = max_time
        self.start = start
        best, best_score = None, None
        for si in deme:
            s = si.second.get_penalized_score()
            if best_score is None or s > best_score:
                best, best_score = si.first, s
        self._best = best

    def best_genotype(self):
        return self._best

    def charge_tuning(self) -> None:
        if self.ledger.requests >= self.max_evals:
            raise BudgetExhausted("global_budget")
        if time.monotonic() - self.start >= self.max_time:
            raise BudgetExhausted("time_limit")
        self.ledger.requests += 1
        self.ledger.tuning_requests += 1


class _HcSchema:
    """The part of ``FrozenSchema`` that ``_tune_best`` uses: the "genotype" is the deme's
    instance itself."""

    def __init__(self, iscorer) -> None:
        self._rep = iscorer.get_rep()

    def candidate(self, inst, reduce: bool = True):
        return self._rep.get_candidate(inst, reduce)

    def to_instance(self, inst):
        return instance(inst)


class hc_lm_optimization(optimizer_base):
    """Hill climbing, then LM on the deme's best program (``gomea_params.const_opt ==
    "lm"``). ``last_result`` is the most recent deme's "end" record."""

    # GOMEA's LM step and its helpers, reused as they are (they read only ``_co``,
    # ``_val_rows`` and ``gomea_params`` of ``self``).
    _tune_best = gomea_optimization._tune_best
    _guard_rows = gomea_optimization._guard_rows
    _close_tuning = gomea_optimization._close_tuning
    _hc_stop = hc_gomea_optimization._hc_stop

    def __init__(
        self,
        op: optim_parameters | None = None,
        hc: hc_parameters | None = None,
        gp: gomea_parameters | None = None,
    ) -> None:
        super().__init__(op)
        self.hc = hill_climbing(self.opt_params, hc if hc is not None else hc_parameters())
        self.gomea_params = gp if gp is not None else gomea_parameters(const_opt="lm")
        if self.gomea_params.const_opt != "lm":
            raise ValueError(f"hc_lm_optimization needs const_opt 'lm', got "
                             f"{self.gomea_params.const_opt!r}")
        self._val_rows = None
        self._co: dict = _new_const_opt_counts()
        self._deme_index = 0
        self.last_result: dict | None = None

    def __call__(self, deme: deme_t, iscorer, max_evals: int, max_time: int) -> None:
        start = time.monotonic()
        widen = bool(self.hc.hc_params.widen_search)
        reserve = lm_reserve(max_evals) if widen else 0
        climb_evals = max_evals - reserve
        self.hc(deme, iscorer, climb_evals, max_time)
        hc_evals = deme.n_evals
        reason = self._hc_stop(deme, hc_evals, climb_evals)
        self._co = _new_const_opt_counts()
        oracle = _HcDemeOracle(deme, iscorer, max_evals, max_time, start)
        self._tune_best(deme, oracle, _HcSchema(iscorer))

        self.nsteps = self.hc.nsteps
        self.total_steps += self.hc.nsteps
        self.total_evals += deme.n_evals
        self.over_budget = self.hc.over_budget
        self.demeID = deme.getID()

        co = {"mode": "lm", **self._co}
        if self.gomea_params.const_opt_guard != "none":
            co.update(
                guard=self.gomea_params.const_opt_guard,
                lm_guard_checked=self._co.get("lm_guard_checked", 0),
                lm_guard_rejected=self._co.get("lm_guard_rejected", 0),
            )
        self.last_result = {
            "deme_id": str(deme.getID()),
            "deme_index": self._deme_index,
            "optimizer": "hc",
            "termination_reason": reason,
            "generations": self.hc.nsteps,
            "max_evals": max_evals,
            "hc_evals": hc_evals,
            "ledger": {"requests": deme.n_evals,
                       "tuning_requests": oracle.ledger.tuning_requests},
            "const_opt": co,
        }
        if widen:
            self.last_result["lm_reserve"] = reserve
        self._deme_index += 1
        TelemetrySink(self.gomea_params.telemetry_path).emit("end", **self.last_result)
