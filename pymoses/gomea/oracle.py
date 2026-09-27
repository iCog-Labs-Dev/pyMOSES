"""The single path through which the GOMEA population optimizer scores a
genotype.

Hill climbing's own contract (``docs/gomea/P1.0-AUDIT.md`` Sec 3) is: every
evaluated instance is appended to the ``instance_set`` deme, and
``deme.n_evals`` counts scorer calls. A population optimizer that samples,
mixes and replaces genotypes needs that same contract enforced from ONE
place, not re-derived at every call site -- so every genotype the optimizer
ever scores goes through ``EvaluationOracle.evaluate``, which:

* checks the budget FIRST (``check()``), before the scorer ever runs, so an
  exhausted budget never spends an extra scorer call;
* calls the wrapped ``iscorer`` (typically a ``complexity_based_scorer``,
  see ``instance_scorer.py``) exactly once per accepted request;
* appends an independent ``scored_instance`` to the deme, so the deme is an
  all-evaluated-trials archive exactly like hill climbing's (Sec 3's
  "matched archive policy" -- neither more nor less than HC would record);
* tracks a small ``EvaluationLedger`` of counters later tasks report and
  reason about (budget accounting, cache effectiveness, skip bookkeeping).

Skips (``record_noop_skip``/``record_phenotype_skip``) are for genotypes the
optimizer chooses NOT to send to the scorer at all -- e.g. a trial identical
to the instance it would replace, or one proven to decode to the same
unreduced tree as another already-scored genotype (``FrozenSchema.
phenotype_key``). They only move ledger counters: no budget check, no
scorer call, no deme append. Later tasks (the GOM operator) decide when
those apply and call the matching method themselves.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.moses.types import composite_score, very_worst_score
from pymoses.moses.representation.instance_scorer import iscorer_base
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.scored_instance import scored_instance


class BudgetExhausted(Exception):
    """Raised by ``EvaluationOracle.evaluate`` when ``check()`` finds the
    budget already spent -- the scorer is never called in that case.

    ``reason`` is one of ``'global_budget'``, ``'inner_budget'`` or
    ``'time_limit'`` (see ``EvaluationOracle.check`` for what each one
    means and the order they are checked in).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class EvaluationLedger:
    """Running counters for one ``EvaluationOracle``'s lifetime.

    ``requests`` is the budget unit: it counts only ``evaluate()`` calls
    that got PAST ``check()`` and actually reached the scorer -- the same
    quantity ``check()`` compares against ``max_evals``/``inner_cap``, and
    what ``finalize()`` writes into ``deme.n_evals``. Skips
    (``noop_skips``/``phenotype_skips``) never touch it.

    ``duplicate_requests`` counts the subset of ``requests`` whose genotype
    this oracle had already evaluated earlier in the same invocation
    (protocol Sec 94: requests reported separately from actual work, since
    the demo problems run with the score cache off). A counter only: a
    duplicate is still scored, archived and charged like any request.

    ``memo_hits`` counts ``evaluate()`` calls answered from the genotype
    memo (``EvaluationOracle(genotype_memo=True)``, the Phase 1d
    hypothesis-C ablation): never scored, charged or archived, so never
    part of ``requests`` or ``duplicate_requests``. Always 0 with the memo
    off.
    """

    requests: int = 0
    invalid: int = 0
    noop_skips: int = 0
    phenotype_skips: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duplicate_requests: int = 0
    memo_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "invalid": self.invalid,
            "noop_skips": self.noop_skips,
            "phenotype_skips": self.phenotype_skips,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "duplicate_requests": self.duplicate_requests,
            "memo_hits": self.memo_hits,
        }


class EvaluationOracle:
    """Budgeted, ledgered wrapper around one ``iscorer`` for one deme.

    Not itself a ported C++ symbol -- classic's optimizers (hill climbing,
    ``hill_climbing.py``) each own their own budget bookkeeping inline; this
    package factors that out into one collaborator so every part of the
    GOMEA optimizer (initial population, GOM variation, later tasks) shares
    identical budget, archival and counting behaviour instead of
    re-implementing it.

    ``genotype_memo`` (off by default) is a GOMEA-only mechanism test for
    Phase 1d (hypothesis C), not a MOSES feature: when on, a genotype this
    oracle already evaluated is answered with a copy of its stored score,
    with no scorer call, no budget charge (so even when the budget is
    spent) and no second deme entry, and counted in ``ledger.memo_hits``.
    The memo belongs to this oracle, i.e. to one deme/invocation; it is
    never shared across demes or schemas.
    """

    def __init__(
        self,
        schema: FrozenSchema,
        iscorer: iscorer_base | Callable,
        deme: instance_set,
        max_evals: int,
        max_time: float,
        inner_cap: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        genotype_memo: bool = False,
    ) -> None:
        self.schema = schema
        self.iscorer = iscorer
        self.deme = deme
        self.max_evals = max_evals
        self.max_time = max_time
        self.inner_cap = inner_cap
        self.clock = clock
        self.start = clock()

        self.ledger = EvaluationLedger()
        self._best: composite_score | None = None
        self._best_evals = 0
        # Every genotype that reached the scorer so far, for
        # ``ledger.duplicate_requests``.
        self._seen: set[tuple[int, ...]] = set()
        # Genotype -> stored score, only when ``genotype_memo`` is on.
        self._memo: dict[tuple[int, ...], composite_score] | None = {} if genotype_memo else None

    def check(self) -> str | None:
        """The budget reason that would stop evaluation right now, or
        ``None`` if none applies. Checked before EVERY evaluation --
        ``evaluate()`` calls this itself, first thing, every time.

        Order: global budget, then the (optional) inner cap, then the
        time limit -- matching the brief's documented precedence, so that
        when several limits are exhausted at once the reported reason is
        always the same one.
        """
        if self.ledger.requests >= self.max_evals:
            return "global_budget"
        if self.inner_cap is not None and self.ledger.requests >= self.inner_cap:
            return "inner_budget"
        if self.clock() - self.start >= self.max_time:
            return "time_limit"
        return None

    def evaluate(self, g: tuple[int, ...]) -> composite_score:
        """Score one genotype through the wrapped ``iscorer``, append it
        to the deme, and update the ledger.

        Raises ``BudgetExhausted`` (without calling the scorer, and
        without touching the deme or the ledger) when ``check()`` finds
        the budget already spent. Returns an independent copy of the
        score -- the caller may freely mutate it; the deme's own copy
        (and this oracle's own ``best_so_far`` bookkeeping) are each
        separate objects, never the same one.

        With ``genotype_memo`` on, a genotype already evaluated by this
        oracle returns a copy of its stored score first, before the budget
        check, and touches nothing but ``ledger.memo_hits``.
        """
        if self._memo is not None:
            stored = self._memo.get(tuple(g))
            if stored is not None:
                self.ledger.memo_hits += 1
                return copy.copy(stored)

        reason = self.check()
        if reason is not None:
            raise BudgetExhausted(reason)

        inst = self.schema.to_instance(g)

        cscorer = getattr(self.iscorer, "_cscorer", None)
        hits_before = getattr(cscorer, "_cache_hits", 0)
        misses_before = getattr(cscorer, "_cache_misses", 0)

        score = self.iscorer(inst)

        hits_after = getattr(cscorer, "_cache_hits", 0)
        misses_after = getattr(cscorer, "_cache_misses", 0)

        self.ledger.requests += 1
        key = tuple(g)
        if key in self._seen:
            self.ledger.duplicate_requests += 1
        else:
            self._seen.add(key)
        self.ledger.cache_hits += hits_after - hits_before
        self.ledger.cache_misses += misses_after - misses_before

        penalized = score.get_penalized_score()
        if math.isnan(penalized) or penalized == very_worst_score:
            self.ledger.invalid += 1

        self.deme.append(scored_instance(self.schema.to_instance(g), copy.copy(score)))
        if self._memo is not None:
            self._memo[key] = copy.copy(score)

        if self._best is None or self._best < score:
            self._best = copy.copy(score)
            self._best_evals = self.ledger.requests

        return copy.copy(score)

    def record_noop_skip(self) -> None:
        """A trial genotype identical to the recipient it would replace
        was never sent to the scorer -- count the skip, nothing else."""
        self.ledger.noop_skips += 1

    def record_phenotype_skip(self) -> None:
        """A trial genotype proven (via ``FrozenSchema.phenotype_key``) to
        decode to the same unreduced tree as an already-scored genotype
        was never sent to the scorer -- count the skip, nothing else."""
        self.ledger.phenotype_skips += 1

    def best_so_far(self) -> composite_score:
        """The best score seen so far, under MOSES's own composite
        comparator (ties keep the earlier evaluation -- see ``evaluate``).
        A defensive copy: mutating the result never affects this oracle's
        own bookkeeping.
        """
        if self._best is None:
            return composite_score()
        return copy.copy(self._best)

    def evals_at_best(self) -> int:
        """The 1-based request count at which the current best-so-far was
        first obtained (0 if nothing has been evaluated yet)."""
        return self._best_evals

    def finalize(self) -> None:
        """Write this oracle's final counts back onto the deme, matching
        what hill climbing itself sets at the end of its own run
        (``hill_climbing.__call__``, ``docs/gomea/P1.0-AUDIT.md`` Sec 3):
        ``deme.n_evals`` is the total scorer-reaching request count,
        ``deme.n_best_evals`` is the request count the current best was
        first found at.
        """
        self.deme.n_evals = self.ledger.requests
        self.deme.n_best_evals = self.evals_at_best()
