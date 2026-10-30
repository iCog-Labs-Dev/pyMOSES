"""Instrumented pyMoses GOMEA run (diagnosis only; repo code is not changed).

Monkeypatches pymoses.gomea.engine.perform_generation and engine.gom to dump,
per generation: the population over the model's linked positions (as the
model sees it, symbolized when the model symbolizes), H0, the learned FOS,
GOM trial attribution by subset size, and (optionally) a direct epistasis
probe that scores genotypes through the oracle's iscorer WITHOUT the ledger,
the deme or any RNG stream (so the run itself is unchanged).

Usage: instr.py OUT.jsonl EPI_DEMES(int, 0=off) -- <moses argv...>
"""

from __future__ import annotations

import json
import random
import sys
import math

sys.setrecursionlimit(100000)
REPO = "/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/pyMoses"
sys.path.insert(0, REPO)

import pymoses.gomea.engine as E  # noqa: E402
import pymoses.gomea.gom as G  # noqa: E402
from pymoses.gomea.fos import LinkageModel  # noqa: E402
from pymoses.gomea.oracle import BudgetExhausted  # noqa: E402

OUT = open(sys.argv[1], "w")
EPI_DEMES = int(sys.argv[2])
assert sys.argv[3] == "--"
ARGV = sys.argv[4:]

STATE = {"deme": -1, "cur": None, "model_id": None}
EPI_RNG = random.Random(12345)  # private stream for the probe only


def emit(rec):
    OUT.write(json.dumps(rec) + "\n")
    OUT.flush()


# ---- capture each learned FOS on the model instance ----
_orig_learn = LinkageModel.learn
_orig_learn_subset = LinkageModel.learn_subset


def _learn(self, pop):
    F = _orig_learn(self, pop)
    self._dbg_F = [list(s) for s in F]
    return F


def _learn_subset(self, pop, positions, H_linked=None):
    F = _orig_learn_subset(self, pop, positions, H_linked)
    self._dbg_F = [list(s) for s in F]
    self._dbg_sel = list(positions)
    return F


LinkageModel.learn = _learn
LinkageModel.learn_subset = _learn_subset


# ---- GOM copy with per-trial attribution (same RNG draws, same logic) ----
def _bucket(n):
    return "1" if n == 1 else ("2-3" if n <= 3 else ("4-15" if n <= 15 else "16+"))


def gom_rec(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance="nonworse",
            on_accept=None):
    cur = STATE["cur"]
    strict = acceptance == "strict"
    cur_g, cur_s, cur_k = recipient.genotype, recipient.score, recipient.pkey
    improved = False
    d = cur["d_fos"]
    oracle = ev.oracle
    for idx in G._shuffled_order(len(fos), rng):
        subset = fos[idx]
        donor = donors[rng.randint_n(len(donors))]
        stats.trials += 1
        b = _bucket(len(subset))
        near_root = len(subset) >= max(2, d // 2)
        att = cur["att"].setdefault(b, {"trials": 0, "noop": 0, "memo": 0, "eval": 0,
                                         "rejected": 0, "neutral": 0, "improve": 0,
                                         "changed_pos_sum": 0, "phen": 0})
        att["trials"] += 1
        trial = G._mix(cur_g, donor.genotype, subset)
        if trial == cur_g:
            stats.noop_skips += 1
            ev.record_noop_skip()
            att["noop"] += 1
            continue
        nchg = sum(1 for p in subset if trial[p] != cur_g[p])
        att["changed_pos_sum"] += nchg
        trial_k = ev.phenotype_key(trial)
        if skip_inactive and trial_k is not None and trial_k == cur_k:
            stats.phenotype_skips += 1
            ev.record_phenotype_skip()
            att["phen"] += 1
            if strict:
                stats.rejected += 1
                continue
            stats.accepted_neutral += 1
            cur_g = trial
            continue
        m0 = oracle.ledger.memo_hits
        try:
            trial_s = ev.evaluate(trial)
        except BudgetExhausted as exc:
            raise G.GomInterrupted(G.Individual(cur_g, cur_s, cur_k), exc.reason) from exc
        if oracle.ledger.memo_hits > m0:
            att["memo"] += 1
        else:
            att["eval"] += 1
        stats.evaluated += 1
        if ev.worse(trial_s, cur_s) or (strict and not ev.worse(cur_s, trial_s)):
            stats.rejected += 1
            att["rejected"] += 1
            continue
        if ev.worse(cur_s, trial_s):
            stats.accepted_improving += 1
            improved = True
            att["improve"] += 1
            cur["improvements"].append({"size": len(subset), "nchg": nchg,
                                        "near_root": near_root,
                                        "gain": trial_s.get_penalized_score()
                                        - cur_s.get_penalized_score()})
        else:
            stats.accepted_neutral += 1
            att["neutral"] += 1
        cur_g, cur_s, cur_k = trial, trial_s, trial_k
        if on_accept is not None:
            on_accept(trial)
    return G.Individual(cur_g, cur_s, cur_k), cur_g != recipient.genotype, improved


E.gom = gom_rec


# ---- epistasis probe ----
def _score(oracle, g):
    s = oracle.iscorer(oracle.schema.to_instance(g))
    return s.get_penalized_score(), s.get_score()


def epistasis_probe(eng, P, model, pairs_by_group, domains):
    oracle = eng.ev.oracle
    best = eng.best_of(P).genotype
    cache = {}

    def f(g):
        if g not in cache:
            cache[g] = _score(oracle, g)
        return cache[g]

    f0 = f(best)
    out = {}
    for gname, pairs in pairs_by_group.items():
        res = []
        for (i, j) in pairs:
            # donor-available combos: values seen in the population, both differ from best
            donor_vals = {(ind.genotype[i], ind.genotype[j]) for ind in P
                          if ind.genotype[i] != best[i] and ind.genotype[j] != best[j]}
            combos = []
            for _ in range(2):
                ai = EPI_RNG.choice([v for v in range(domains[i]) if v != best[i]])
                aj = EPI_RNG.choice([v for v in range(domains[j]) if v != best[j]])
                combos.append((ai, aj))
            for (ai, aj) in combos:
                gi = list(best); gi[i] = ai; gi = tuple(gi)
                gj = list(best); gj[j] = aj; gj = tuple(gj)
                gij = list(best); gij[i] = ai; gij[j] = aj; gij = tuple(gij)
                fi, fj, fij = f(gi), f(gj), f(gij)
                rec = {}
                for k, name in ((0, "pen"), (1, "raw")):
                    inter = fij[k] - fi[k] - fj[k] + f0[k]
                    scale = max(1e-9, abs(f0[k]) * 1e-9)
                    rec[name + "_epi"] = abs(inter) > scale and all(
                        math.isfinite(x) for x in (fij[k], fi[k], fj[k], f0[k]))
                    rec[name + "_any_effect"] = any(abs(x[k] - f0[k]) > scale for x in (fi, fj, fij))
                    rec[name + "_joint_only_gain"] = (fij[k] > f0[k] + scale and fi[k] <= f0[k] + scale
                                                      and fj[k] <= f0[k] + scale)
                    rec[name + "_inter"] = inter
                rec["donor_avail"] = len(donor_vals) > 0
                res.append(rec)
        out[gname] = res
    return out


_orig_pg = E.perform_generation


def pg_wrapper(eng, P, model, nis, generation):
    if id(model) != STATE["model_id"]:
        STATE["model_id"] = id(model)
        STATE["deme"] += 1
    deme = STATE["deme"]
    genos = [ind.genotype for ind in P]
    linked = list(model.linked_positions)
    src = genos if model.symbolize is None else model.symbolize(genos)
    pop_l = [[row[p] for p in linked] for row in src]
    H0 = model._H0
    pens = [ind.score.get_penalized_score() for ind in P]
    cur = {"att": {}, "improvements": [], "d_fos": len(model.search_positions)}
    STATE["cur"] = cur
    epi = None
    if deme < EPI_DEMES and generation <= 2 and len(linked) >= 4:
        # rank pairs by LT-B and MI similarity on the CURRENT population
        from pymoses.gomea.stats import biased_similarity, entropy_matrix, mi_similarity
        H = entropy_matrix(pop_l, range(len(linked)))
        S = biased_similarity(H, H0)
        M = mi_similarity(H)
        d = len(linked)
        allpairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
        top_ltb = sorted(allpairs, key=lambda ab: -S[ab[0]][ab[1]])[:15]
        top_mi = sorted(allpairs, key=lambda ab: -M[ab[0]][ab[1]])[:15]
        rnd = EPI_RNG.sample(allpairs, min(15, len(allpairs)))
        mp = lambda L: [(linked[a], linked[b]) for a, b in L]
        epi = epistasis_probe(eng, P, model, {"top_ltb": mp(top_ltb), "top_mi": mp(top_mi),
                                              "random": mp(rnd)}, eng.domains)
    new_pop, record, stop = _orig_pg(eng, P, model, nis, generation)
    emit({
        "deme": deme, "gen": generation, "N": len(P), "mode": model.mode,
        "n_search": len(model.search_positions), "linked": linked,
        "pop": pop_l, "H0": H0, "pens": pens,
        "fos": getattr(model, "_dbg_F", None), "sel": getattr(model, "_dbg_sel", None),
        "strengths": list(getattr(model, "last_strengths", []) or []),
        "att": cur["att"], "improvements": cur["improvements"],
        "stats": record["stats"], "stop": stop, "epi": epi,
        "full_genos_varying": None,
    })
    return new_pop, record, stop


E.perform_generation = pg_wrapper

_orig_finish = E.GomeaEngine._finish


def _finish(self, reason):
    res = _orig_finish(self, reason)
    emit({"term": reason, "deme": STATE["deme"], "gens": res.generations})
    return res


E.GomeaEngine._finish = _finish

from pymoses.moses.main.moses_exec import moses_exec  # noqa: E402

rc = moses_exec(["moses", *ARGV])
emit({"done": True, "rc": rc, "demes": STATE["deme"] + 1})
OUT.close()
