"""NOT A PORT, T3.2 (server round 16): the RV-GOMEA core on the constant genes
(``--gomea-rv-step rvg``).

Specification: ``docs/gomea/T3.2-FEASIBILITY.md`` ("Appendix: the RV-GOMEA core") and
``docs/gomea/SERVER-ROUND16-PREREGISTRATION.md`` ("The seed fix", "Arms"). The code is that of
the T3.2 spike 2 (commit be83e08) for the budget rules ``balance`` and ``late``, with the seed
fix (``rv_stream``). The ``cap`` rule of the spike is left out.

The RV-GOMEA core (Bouter et al., GECCO 2017, Algorithms 1-4) with the intron-aware rules
of GP-RV-GOMEA (Koch et al., PPSN 2024, Sec. 3.4), on the constant genes of a regression
deme (``--gomea-contin copy``). Every deme gene is a constant gene; a value of 0 removes
its term, so a gene with value 0 is never sampled (the structure stays GOM's work).

One RV generation (``RVCore.__call__``), after a GOM generation:

* Active genes of an individual: non-zero, "live" under the structural rule
  (``rv_inert.analyse``), and not in the online absorbed cache.
* Univariate FOS: one element per gene with a model, in a random order.
* Model of gene p (intron-aware): the best floor(0.35 m) (at least 1) of the m individuals
  in which p is active, ranked by the deme's comparator; maximum-likelihood mean and
  variance; the mean shift from the previous RV generation of the deme.
* Sampling: y ~ N(mu, c sigma^2); the first floor(0.5 * 0.35 * n) non-elitist individuals
  (deme order) also get c * 2 * mu_shift (AMS). One sample per distinct genotype per gene.
* Acceptance: keep a strict improvement; otherwise keep with probability 0.05 (own stream),
  else restore. A new value becomes a free code only when it is kept.
* AdaptMultiplier (SDR-AVS) after each gene; the second AMS round; the no-improvement
  stretch of each individual and forced improvements (NIS_MAX = 20).
* Elitism: the best individual at the start of the RV generation is not changed.

The budget rule decides, after each GOM generation, if an RV generation runs:

* ``balance``: with probability p_RV = 1 - 0.5 (RV evaluations / GOM evaluations) of the
  deme (1 before the first GOM evaluation);
* ``late``: only after a GOM generation that did not improve the deme best
  (``engine.gom_improved``), then with the same p_RV.

Departures from the appendix are marked "DEPARTURE" in the code. The spike 2 record lists
14 of them (``pyMoses-suite-tmp/t32spike2/REPORT_INTERIM.md``).
"""

from __future__ import annotations

import functools
import math
import statistics
import time
from collections.abc import Sequence

from pymoses.gomea.const_opt import mutation_sigma
from pymoses.gomea.gom import Individual
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.gomea.rv_inert import analyse
from pymoses.moses.moses.types import very_worst_score
from pymoses.util.rng import MT19937RandGen

RV_MODES = ("off", "rvg")  # ``--gomea-rv-step``
RV_BUDGETS = ("balance", "late")  # ``--gomea-rv-budget``

TAU = 0.35  # RV-GOMEA: the selection share
DELTA_AMS = 2.0  # RV-GOMEA: the AMS step factor
P_ACCEPT = 0.05  # RV-GOMEA: keep a non-improving sample with this probability
THETA_SDR = 1.0  # AVS: the SDR threshold
ETA_DEC = 0.9  # AVS: the decrease factor
ETA_INC = 1.0 / 0.9  # AVS: the increase factor
NIS_MAX = 20  # GP-RV-GOMEA's forced-improvement value; also AVS's NIS_MAX (see below)
FI_ALPHA0 = 0.5  # Algorithm 4: the first blend factor
FI_ALPHA_MIN = 0.01  # Algorithm 4: stop when the blend factor is below this
RV_BALANCE = 0.5  # p_RV = 1 - RV_BALANCE * (RV evaluations / GOM evaluations)
NEAR_REL = 1e-6  # telemetry only: "near-equal" raw scores (not used by any rule)

# The seed fix: the salt of the master generator of the RV streams ("RVG1" in ASCII).
RV_SEED_SALT = 0x52564731


def rv_seed(run_seed: int, deme_index: int) -> int:
    """The seed of the RV stream of deme ``deme_index`` (0 for the first deme of the run).

    The rule: make the master generator ``MT19937RandGen((run_seed XOR RV_SEED_SALT) mod
    2**32)``. The seed is its draw number ``deme_index + 1`` of ``randint()``. ``run_seed``
    is the run seed (``-r``). The rule never uses the global generator ``randGen()``."""
    if deme_index < 0:
        raise ValueError(f"deme_index must be >= 0, got {deme_index}")
    master = MT19937RandGen((run_seed ^ RV_SEED_SALT) & 0xFFFFFFFF)
    s = 0
    for _ in range(deme_index + 1):
        s = master.randint()
    return s


def rv_stream(run_seed: int, deme_index: int) -> MT19937RandGen:
    """The seed fix: the RV stream of deme ``deme_index`` of the run with seed ``run_seed``,
    ``MT19937RandGen(rv_seed(run_seed, deme_index))``.

    The stream does not draw from, and does not change, the global generator ``randGen()``.
    So, with ``--gomea-rv-step rvg``, everything up to the first RV generation of a deme
    (the init, linkage and variation streams, the initial population, the first GOM
    generation) is the same as with ``--gomea-rv-step off``. (Spike 2 seeded the RV stream
    from one ``randGen()`` draw. That changed all later draws of the run, also those of the
    first deme's initial population, because the MOSES sampler draws from ``randGen()``.)"""
    return MT19937RandGen(rv_seed(run_seed, deme_index))


def _gauss(rng: MT19937RandGen) -> float:
    """A standard normal draw (Box-Muller, two ``randdouble`` draws; a guard for u1 = 0)."""
    u1 = 1.0 - rng.randdouble()
    if u1 <= 0.0:
        u1 = 2.0**-32
    u2 = rng.randdouble()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _invalid(score) -> bool:
    p = score.get_penalized_score()
    return math.isnan(p) or p == very_worst_score or not math.isfinite(score.get_score())


def _c_stats(c: dict[int, float], genes: Sequence[int]) -> dict:
    vals = [c[p] for p in genes if p in c]
    if not vals:
        return {"c_n": 0}
    return {
        "c_n": len(vals),
        "c_med": statistics.median(vals),
        "c_min": min(vals),
        "c_max": max(vals),
    }


class RVCore:
    """NOT A PORT, T3.2 (server round 16): the engine's ``rv_generation`` hook for
    ``--gomea-rv-step rvg``.

    ``__call__(eng, P, generation)`` runs one RV generation (or skips it, by the budget
    rule) and returns ``(P', record, stop)``. ``converged(P)`` is the structure-only test
    of GP-RV-GOMEA Sec. 4.1 that the engine uses in place of its own test when this hook
    is set. Every evaluation goes through ``ev.evaluate`` (the oracle: budget, deme store).
    ``rng`` is the RV stream of the deme (``rv_stream``).

    Telemetry: ``record()`` gives the totals of the deme (the "rv" record of the "end"
    event); ``__call__`` gives the "rv" record of each "generation" event. The field names
    are those of spike 2. ``skipped_cap`` and ``cap_hits`` stay 0 and ``cap`` is ``None``,
    because the ``cap`` rule of the spike is left out.
    """

    def __init__(self, schema, oracle, ev, rng: MT19937RandGen, keep, budget="balance"):
        if budget not in RV_BUDGETS:
            raise ValueError(f"budget must be one of {RV_BUDGETS}, got {budget!r}")
        self.schema = schema
        self.oracle = oracle
        self.ev = ev
        self.rng = rng
        # Rule 4: the acceptance draws have a stream of their own (seeded from the RV stream,
        # which exists only when the flag is on).
        self.rng_acc = MT19937RandGen(rng.randint())
        self.keep = keep
        self.budget = budget
        self.reduce = getattr(oracle.iscorer, "_reduce", True)
        self.contin = tuple(schema.contin_positions)
        self._contin_set = set(self.contin)
        self._other = tuple(p for p in range(schema.dim) if p not in self._contin_set)
        self._depth = dict(zip(schema.contin_positions, schema.contin_depths, strict=True))
        # The deme's RV state.
        self.c: dict[int, float] = {}  # distribution multiplier per gene (starts at 1.0)
        self.prev_mu: dict[int, float] = {}  # the model mean of the previous RV generation
        self.pop_nis = 0  # AVS: the population no-improvement stretch
        self.nis: list[int] | None = None  # Algorithm 2: NIS per individual (index)
        self.absorbed: set[tuple[int, tuple]] = set()  # online cache: (gene, structure)
        self.init_requests = oracle.ledger.requests
        self.rv_evals = 0
        self.rv_index = 0  # RV generations run in this deme
        self._last_end_req = oracle.ledger.requests
        self._gen_start_req = oracle.ledger.requests
        self._te = 0.0  # the evaluation seconds of the current RV generation
        self.converged_by: str | None = None
        self.totals: dict = {
            "slots": 0,
            "generations": 0,
            "skipped_p_rv": 0,
            "skipped_gom_improved": 0,
            "skipped_cap": 0,  # the spike's field; always 0 (no cap rule)
            "cap_hits": 0,  # the spike's field; always 0 (no cap rule)
            "evals": 0,
            "improving": 0,
            "kept_p": 0,
            "kept_p_worse": 0,
            "rejected": 0,
            "invalid": 0,
            "ams1_evals": 0,
            "ams1_improving": 0,
            "ams1_kept_p": 0,
            "ams2_evals": 0,
            "ams2_improving": 0,
            "ams2_kept_p": 0,
            "fi_triggers": 0,
            "fi_evals": 0,
            "fi_improved": 0,
            "fi_elite_copies": 0,
            "skip_struct": 0,
            "skip_cache": 0,
            "skip_dup": 0,
            "cache_marks": 0,
            "near_equal": 0,
            "sd_degenerate": 0,
            "models": 0,
            "pm1_moves": 0,
            "pm1_dcomplexity": 0,
            "pm1_complexity_up": 0,
            "eval_s": 0.0,
            "analyse_s": 0.0,
            "step_s": 0.0,
        }

    # -- helpers -------------------------------------------------------------------------

    def _cmp(self, a: Individual, b: Individual) -> int:
        if self.ev.worse(a.score, b.score):
            return 1
        if self.ev.worse(b.score, a.score):
            return -1
        return 0

    def pattern(self, g: Sequence[int]) -> tuple:
        """The structure of ``g``: which constant genes are non-zero (decoded values, not
        codes), plus the codes of any other gene."""
        value = self.schema.contin_value
        nz = tuple(p for p in self.contin if value(p, g[p]) != 0.0)
        return (nz, tuple(g[p] for p in self._other))

    def converged(self, P: Sequence[Individual]) -> bool:
        """GP-RV-GOMEA Sec. 4.1: all individuals have the same structure, or the full
        generation (GOM plus RV) made no new evaluation."""
        if self.oracle.ledger.requests == self._gen_start_req:
            self.converged_by = "no_evals"
            return True
        first = self.pattern(P[0].genotype)
        if all(self.pattern(ind.genotype) == first for ind in P[1:]):
            self.converged_by = "structure"
            return True
        return False

    def _trial(self, g: tuple[int, ...], changes: dict[int, float]):
        """``(trial genotype, tentative new codes)``. A value is registered as a free code
        now (the oracle scores a genotype through its codes) and unregistered by
        ``_rollback`` if the trial is not kept."""
        schema = self.schema
        t = list(g)
        new = []
        for p, v in changes.items():
            v = float(v)
            code = schema._free_codes[p].get(v)
            if code is None:
                code = schema.register_contin_value(p, v)
                new.append((p, code, v))
            t[p] = code
        return tuple(t), new

    def _rollback(self, trial: tuple[int, ...], new) -> None:
        """Rule "free codes": undo the tentative codes of a trial that was not kept. The
        oracle forgets the trial genotype (memo, seen set, deme index), because its codes
        will be given to other values."""
        if not new:
            return
        schema = self.schema
        for p, code, v in reversed(new):
            lst = schema._free_values[p]
            if code != 4 ** self._depth[p] + len(lst) - 1:
                raise AssertionError("tentative free code is not the last one")
            lst.pop()
            del schema._free_codes[p][v]
            schema._n_free -= 1
        o = self.oracle
        if o._memo is not None:
            o._memo.pop(trial, None)
        o._seen.discard(trial)
        if o._last_index is not None:
            o._last_index.pop(trial, None)

    def _evaluate(self, g: tuple[int, ...]):
        """One RV evaluation through the oracle."""
        o = self.oracle
        r0 = o.ledger.requests
        t0 = time.perf_counter()
        try:
            return self.ev.evaluate(g)
        finally:
            self._te += time.perf_counter() - t0
            self.rv_evals += o.ledger.requests - r0

    def _try(self, eng, P, i, changes, cnt, kind, pat=None, gene=None):
        """Evaluate ``changes`` on individual ``i`` and apply the acceptance rule.
        Returns ``(improved, score)``; ``(None, None)`` when the trial is a no-op."""
        cur = P[i]
        trial, new = self._trial(cur.genotype, changes)
        if trial == cur.genotype:
            return None, None
        r0 = self.rv_evals
        try:
            score = self._evaluate(trial)
        except BaseException:
            self._rollback(trial, new)
            raise
        charged = self.rv_evals - r0  # 0 for a memo hit (a value kept before)
        cnt["evals"] += charged
        if kind:
            cnt[kind + "_evals"] += charged
        # The online absorbed cache: the same raw score as before.
        if gene is not None and not _invalid(score) and not _invalid(cur.score):
            a, b = score.get_score(), cur.score.get_score()
            if a == b:
                key = (gene, pat)
                if key not in self.absorbed:
                    self.absorbed.add(key)
                    cnt["cache_marks"] += 1
            elif abs(a - b) <= NEAR_REL * max(1.0, abs(b)):
                cnt["near_equal"] += 1
        improved = self.ev.worse(cur.score, score)
        keep = improved
        if not improved:
            # DEPARTURE: an invalid score (NaN or the worst score) is never kept by p_accept.
            if self.rng_acc.randdouble() < P_ACCEPT and not _invalid(score):
                keep = True
                cnt["kept_p"] += 1
                if kind:
                    cnt[kind + "_kept_p"] += 1
                if self.ev.worse(score, cur.score):
                    cnt["kept_p_worse"] += 1
            else:
                cnt["rejected"] += 1
                if _invalid(score):
                    cnt["invalid"] += 1
        else:
            cnt["improving"] += 1
            if kind:
                cnt[kind + "_improving"] += 1
        if not keep:
            self._rollback(trial, new)
            return improved, score
        # A kept sample that moves a value of exactly +-1 (the reducer drops a factor 1).
        value = self.schema.contin_value
        pm1 = sum(1 for p in changes if value(p, cur.genotype[p]) in (1.0, -1.0))
        if pm1:
            cnt["pm1_moves"] += pm1
            d = score.get_complexity() - cur.score.get_complexity()
            cnt["pm1_dcomplexity"] += d
            if d > 0:
                cnt["pm1_complexity_up"] += 1
        ind = Individual(trial, score, self.ev.phenotype_key(trial))
        P[i] = ind
        eng.offer_elite(ind)
        if self.keep is not None:
            self.keep(trial)
        return improved, score

    # -- one RV generation ---------------------------------------------------------------

    def __call__(self, eng, P: list[Individual], generation: int):
        t0 = time.perf_counter()
        tot = self.totals
        o = self.oracle
        self._gen_start_req = self._last_end_req
        tot["slots"] += 1
        rec: dict = {"ran": False}
        try:
            # The budget rule of the arm.
            gom = o.ledger.requests - self.init_requests - self.rv_evals
            if self.budget == "late" and getattr(eng, "gom_improved", False):
                tot["skipped_gom_improved"] += 1
                rec["why"] = "gom_improved"
                return P, rec, None
            p_rv = 1.0 - RV_BALANCE * (self.rv_evals / gom) if gom > 0 else 1.0
            rec["p_rv"] = p_rv
            if self.rng.randdouble() >= p_rv:
                tot["skipped_p_rv"] += 1
                rec["why"] = "p_rv"
                return P, rec, None
            return self._generation(eng, list(P), rec, t0)
        finally:
            self._last_end_req = o.ledger.requests
            tot["step_s"] += time.perf_counter() - t0

    def _generation(self, eng, P: list[Individual], rec: dict, t0: float):
        tot = self.totals
        schema = self.schema
        value = schema.contin_value
        n = len(P)
        self.rv_index += 1
        tot["generations"] += 1
        rec.update(ran=True, idx=self.rv_index, n=n)
        if self.nis is None or len(self.nis) != n:
            self.nis = [0] * n
        self._te = 0.0
        cnt = {
            k: 0
            for k in (
                "evals improving kept_p kept_p_worse rejected invalid ams1_evals ams1_improving "
                "ams1_kept_p ams2_evals ams2_improving ams2_kept_p fi_triggers fi_evals "
                "fi_improved fi_elite_copies skip_struct skip_cache skip_dup cache_marks "
                "near_equal pm1_moves pm1_dcomplexity pm1_complexity_up"
            ).split()
        }
        # Elitism (n_elitist = 1): the best individual now is not changed.
        e = 0
        for i in range(1, n):
            if self.ev.worse(P[e].score, P[i].score):
                e = i
        # Active genes of each individual (structural rule; the cache is applied per use).
        ta = time.perf_counter()
        by_g: dict = {}
        live: list[set[int]] = []
        nonzero: list[set[int]] = []
        pats: list[tuple] = []
        for ind in P:
            got = by_g.get(ind.genotype)
            if got is None:
                a = analyse(schema, ind.genotype, self.reduce)
                lv = {p for p, s in a["genes"].items() if s == "live"}
                nz = {p for p in self.contin if value(p, ind.genotype[p]) != 0.0}
                got = (lv & nz, nz, self.pattern(ind.genotype))
                by_g[ind.genotype] = got
            live.append(got[0])
            nonzero.append(got[1])
            pats.append(got[2])
        tot["analyse_s"] += time.perf_counter() - ta

        def active(i: int, p: int) -> bool:
            return p in live[i] and (p, pats[i]) not in self.absorbed

        # Algorithm 1: the models, from the individuals in which the gene is active.
        order = sorted(range(n), key=functools.cmp_to_key(lambda i, j: self._cmp(P[i], P[j])))
        models: dict[int, tuple[float, float, float, object]] = {}
        degenerate = 0
        for p in self.contin:
            idx = [i for i in order if active(i, p)]
            if not idx:
                continue
            k = max(1, math.floor(TAU * len(idx)))
            vals = [value(p, P[i].genotype[p]) for i in idx[:k]]
            mu = math.fsum(vals) / k
            sd = math.sqrt(math.fsum((v - mu) ** 2 for v in vals) / k)
            if sd == 0.0:
                # DEPARTURE: all selected values are equal (a grid value that GOM copied, or
                # k = 1); the ML variance is 0, so the sample would be the mean. Use the
                # first spike's spread max(0.1 |mu|, 0.1) for this generation only.
                sd = mutation_sigma(mu)
                degenerate += 1
            shift = mu - self.prev_mu[p] if p in self.prev_mu else 0.0
            models[p] = (mu, sd, shift, P[idx[0]].score)
            self.c.setdefault(p, 1.0)
        for p, m in models.items():
            self.prev_mu[p] = m[0]
        tot["models"] += len(models)
        tot["sd_degenerate"] += degenerate
        genes = list(models)
        for a in range(len(genes) - 1, 0, -1):
            b = self.rng.randint_n(a + 1)
            genes[a], genes[b] = genes[b], genes[a]
        non_elite = [i for i in range(n) if i != e]
        n_ams = math.floor(0.5 * TAU * n)
        ams_set = set(non_elite[:n_ams])
        start_scores = [ind.score for ind in P]
        improved_any = [False] * n
        visited = [False] * n
        stop = None
        try:
            # Algorithm 2: for each FOS element (gene), for each non-elitist individual.
            for p in genes:
                mu, sd, shift, best_sel = models[p]
                c = self.c[p]
                seen: set[tuple[int, ...]] = set()
                improvers: list[float] = []
                for i in non_elite:
                    if p not in nonzero[i]:
                        continue
                    if p not in live[i]:
                        cnt["skip_struct"] += 1
                        continue
                    if (p, pats[i]) in self.absorbed:
                        cnt["skip_cache"] += 1
                        continue
                    g = P[i].genotype
                    if g in seen:
                        cnt["skip_dup"] += 1
                        continue
                    seen.add(g)
                    y = mu + math.sqrt(c) * sd * _gauss(self.rng)
                    kind = None
                    if i in ams_set and shift != 0.0:
                        y += c * DELTA_AMS * shift
                        kind = "ams1"
                    if y == 0.0:
                        continue  # never write a 0 (it would remove the term)
                    visited[i] = True
                    imp, score = self._try(eng, P, i, {p: y}, cnt, kind, pats[i], p)
                    if imp:
                        improved_any[i] = True
                        if self.ev.worse(best_sel, score):
                            improvers.append(y)
                self._adapt(p, mu, sd, improvers)
            # The second AMS round: all active genes of the first n_ams non-elitist individuals.
            for i in non_elite[:n_ams]:
                ch = {}
                g = P[i].genotype
                for p in sorted(live[i]):
                    if p in models and models[p][2] != 0.0 and (p, pats[i]) not in self.absorbed:
                        v = value(p, g[p])
                        if v == 0.0:
                            continue
                        y = v + DELTA_AMS * models[p][2]
                        if y != 0.0:
                            ch[p] = y
                if not ch:
                    continue
                visited[i] = True
                imp, _ = self._try(eng, P, i, ch, cnt, "ams2")
                if imp:
                    improved_any[i] = True
            # The no-improvement stretch of each individual, and forced improvements.
            for i in non_elite:
                if not visited[i]:
                    continue  # DEPARTURE: no RV trial in this generation, no count
                if improved_any[i] or self.ev.worse(start_scores[i], P[i].score):
                    self.nis[i] = 0
                else:
                    self.nis[i] += 1
                if self.nis[i] > NIS_MAX:
                    self._forced_improvement(eng, P, i, live, nonzero, cnt)
                    self.nis[i] = 0
        except BudgetExhausted as exc:
            stop = exc.reason
        for k, v in cnt.items():
            tot[k] += v
        tot["eval_s"] += self._te
        rec.update(
            cnt,
            models=len(models),
            sd_degenerate=degenerate,
            shift_nonzero=sum(1 for m in models.values() if m[2] != 0.0),
            pop_nis=self.pop_nis,
            inert_pairs=sum(len(nonzero[i] - live[i]) for i in non_elite),
            mean_live=sum(len(x) for x in live) / n,
            mean_nonzero=sum(len(x) for x in nonzero) / n,
            absorbed_cache=len(self.absorbed),
            free_values=schema._n_free,
            eval_s=self._te,
            step_s=time.perf_counter() - t0,
            **_c_stats(self.c, genes),
        )
        return P, rec, stop

    def _adapt(self, p: int, mu: float, sd: float, improvers: list[float]) -> None:
        """Algorithm 3, AdaptMultiplier (SDR-AVS) for the univariate element ``p``."""
        c = self.c[p]
        if improvers:
            self.pop_nis = 0
            if c < 1.0:
                c = 1.0
            sdr = abs(math.fsum(improvers) / len(improvers) - mu) / sd
            if sdr > THETA_SDR:
                c = ETA_INC * c
        else:
            if c <= 1.0:
                self.pop_nis += 1
            if c > 1.0 or self.pop_nis >= NIS_MAX:
                c = ETA_DEC * c
            if c < 1.0 and self.pop_nis < NIS_MAX:
                c = 1.0
        self.c[p] = c

    def _forced_improvement(self, eng, P, i, live, nonzero, cnt) -> None:
        """Algorithm 4 on the real part: blend the constants of individual ``i`` toward
        the elitist's, on the genes active in ``i`` and non-zero in the elitist."""
        cnt["fi_triggers"] += 1
        elite = eng.elite if eng.elite is not None else eng.best_of(P)
        cur = P[i]
        if cur.genotype == elite.genotype:
            return
        value = self.schema.contin_value
        genes = [p for p in sorted(live[i]) if value(p, elite.genotype[p]) != 0.0]
        for a in range(len(genes) - 1, 0, -1):
            b = self.rng.randint_n(a + 1)
            genes[a], genes[b] = genes[b], genes[a]
        alpha = FI_ALPHA0
        while alpha >= FI_ALPHA_MIN:
            for p in genes:
                g = P[i].genotype
                v, ve = value(p, g[p]), value(p, elite.genotype[p])
                y = alpha * v + (1.0 - alpha) * ve
                if y == v or y == 0.0:
                    continue
                trial, new = self._trial(g, {p: y})
                try:
                    score = self._evaluate(trial)
                except BaseException:
                    self._rollback(trial, new)
                    raise
                cnt["fi_evals"] += 1
                cnt["evals"] += 1
                if self.ev.worse(P[i].score, score):
                    cnt["fi_improved"] += 1
                    ind = Individual(trial, score, self.ev.phenotype_key(trial))
                    P[i] = ind
                    eng.offer_elite(ind)
                    if self.keep is not None:
                        self.keep(trial)
                    return
                self._rollback(trial, new)
            alpha *= 0.5
        # Nothing improved: P_i becomes a copy of the elitist (no evaluation; never worse).
        if self.ev.worse(P[i].score, elite.score):
            P[i] = elite
            cnt["fi_elite_copies"] += 1

    def record(self) -> dict:
        """The deme's totals (the "rv" record of the "end" event)."""
        d = dict(self.totals)
        d["mode"] = "rvg"
        d["budget"] = self.budget
        d["cap"] = None  # the spike's field; no cap rule
        d["rv_evals"] = self.rv_evals
        d["converged_by"] = self.converged_by
        d["absorbed_cache"] = len(self.absorbed)
        d["free_values"] = self.schema._n_free
        d["c_final"] = _c_stats(self.c, list(self.c))
        d["nis_max_seen"] = max(self.nis) if self.nis else 0
        return d
