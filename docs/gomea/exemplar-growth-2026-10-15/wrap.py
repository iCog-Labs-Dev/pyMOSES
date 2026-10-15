"""Instrumentation wrapper for pyMoses exemplar-growth analysis.

Usage: wrap.py <out.jsonl> <moses args...>
Monkeypatches (never edits repo files):
  metapopulation.select_exemplar  -> exemplar size/complexity/score
  deme_expander.create_demes      -> knob count
  metapopulation.merge_demes      -> deme best, running-best trajectory,
                                     merged candidates, mp size
  pymoses.gomea.engine.gom        -> exact copy of gom() with per-accept logging
"""
import json
import sys
import time

sys.setrecursionlimit(100000)

from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.deme.deme_expander import deme_expander
from pymoses.moses.metapopulation.metapopulation import metapopulation
import pymoses.moses.metapopulation.merging as mg
import pymoses.gomea.engine as eng_mod
import pymoses.gomea.gom as gom_mod
from pymoses.gomea.gom import Individual, _shuffled_order, _mix, GomInterrupted
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str

out = open(sys.argv[1], "w", buffering=1)
args = sys.argv[2:]
T0 = time.process_time()
cur = {"exp": 0}
gomacc = []  # per-deme GOM accept records


def emit(kind, **kw):
    out.write(json.dumps({"kind": kind, "exp": cur["exp"], "cpu": round(time.process_time() - T0, 2), **kw}) + "\n")


def cs_d(cs):
    return {"raw": float(cs.get_score()), "pen": float(cs.get_penalized_score()), "cx": int(cs.get_complexity())}


# ---- exemplar selection
_sel = metapopulation.select_exemplar


def sel(self):
    r = _sel(self)
    if r is not None:
        tr = r.get_tree()
        # rank of the selected one among the metapop, by penalized score
        ordered = list(self._scored_trees)
        pens = [t.get_penalized_score() for t in ordered]
        best_pen = max(pens)
        cx = [t.get_composite_score().get_complexity() for t in ordered]
        near = [c for p, c in zip(pens, cx) if p >= best_pen - 0.06]
        emit("select", nodes=tr.size(), **cs_d(r.get_composite_score()), mp_size=len(ordered),
             mp_best_pen=float(best_pen), n_near_best=len(near),
             near_best_cx_min=min(near), near_best_cx_max=max(near),
             tree=tree_to_stream_str(tr))
    return r


metapopulation.select_exemplar = sel

# ---- knob building
_cd = deme_expander.create_demes


def cd(self, exemplar, *a, **k):
    t = time.process_time()
    r = _cd(self, exemplar, *a, **k)
    emit("create", exemplar_nodes=exemplar.size(), knobs=[rp.fields().raw_size() for rp in self._reps],
         create_cpu=round(time.process_time() - t, 2))
    return r


deme_expander.create_demes = cd


# ---- deme analysis + merge
def deme_stats(deme, rep):
    fs = rep.fields()
    insts = list(deme)
    if not insts:
        return {}
    best = None
    traj = []  # running best (composite <) in evaluation order
    for si in insts:
        cs = si.second
        if best is None:
            best = si
            continue
        if best.second < cs:
            traj.append({"dcx": int(cs.get_complexity() - best.second.get_complexity()),
                         "draw": float(cs.get_score() - best.second.get_score()),
                         "dpen": float(cs.get_penalized_score() - best.second.get_penalized_score()),
                         "ham": int(fs.hamming_distance(si.first, best.first))})
            best = si
    tr = rep.get_candidate(best.first, True)
    first = insts[0]
    return {"n": len(insts), "n_evals": int(deme.n_evals), "best": {**cs_d(best.second), "nodes": tr.size()},
            "first": cs_d(first.second), "traj": traj}


_md = metapopulation.merge_demes
_gnc = mg.get_new_candidates
_trim = metapopulation.trim_down_deme
mstate = {}


def gnc(self, pot):
    r = _gnc(self, pot)
    cxs = sorted(int(c.get_composite_score().get_complexity()) for c in r)
    pens = [float(c.get_penalized_score()) for c in r]
    mstate["pot"] = len(pot)
    mstate["new"] = len(r)
    mstate["new_cx"] = cxs
    if r:
        bi = max(range(len(pens)), key=lambda i: pens[i])
        mstate["new_best_pen"] = pens[bi]
    return r


metapopulation.get_new_candidates = gnc


def md(self, all_demes, reps):
    ds = [deme_stats(d, reps[j]) for j, demes in enumerate(all_demes) for d in demes]
    pre_best = float(self.best_composite_score().get_penalized_score()) if self.size() else None
    mstate.clear()
    t = time.process_time()
    r = _md(self, all_demes, reps)
    cxs = sorted(int(t_.get_composite_score().get_complexity()) for t_ in self._scored_trees)
    emit("merge", demes=ds, merge_cpu=round(time.process_time() - t, 2), **mstate, mp_size=self.size(),
         mp_cx_median=cxs[len(cxs) // 2] if cxs else None, pre_best_pen=pre_best,
         gom=gom_summary())
    cur["exp"] += 1
    return r


metapopulation.merge_demes = md


# ---- GOM with per-accept logging (exact copy of pymoses/gomea/gom.py gom())
def gom_summary():
    if not gomacc:
        return None
    s = {}
    for rec in gomacc:
        key = rec["type"]
        d = s.setdefault(key, {"n": 0, "cx_up": 0, "cx_eq": 0, "cx_down": 0, "sum_dcx": 0, "sum_ham": 0, "ham_hist": {}})
        d["n"] += 1
        dc = rec.get("dcx")
        if dc is not None:
            d["sum_dcx"] += dc
            if dc > 0:
                d["cx_up"] += 1
            elif dc == 0:
                d["cx_eq"] += 1
            else:
                d["cx_down"] += 1
        h = rec["ham"]
        d["sum_ham"] += h
        hb = "1" if h == 1 else "2" if h == 2 else "3-4" if h <= 4 else "5-8" if h <= 8 else "9+"
        d["ham_hist"][hb] = d["ham_hist"].get(hb, 0) + 1
    gomacc.clear()
    return s


def ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def gom_instr(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance="nonworse"):
    strict = acceptance == "strict"
    cur_g, cur_s, cur_k = recipient.genotype, recipient.score, recipient.pkey
    improved = False
    for idx in _shuffled_order(len(fos), rng):
        subset = fos[idx]
        donor = donors[rng.randint_n(len(donors))]
        stats.trials += 1
        trial = _mix(cur_g, donor.genotype, subset)
        if trial == cur_g:
            stats.noop_skips += 1
            ev.record_noop_skip()
            continue
        trial_k = ev.phenotype_key(trial)
        if skip_inactive and trial_k is not None and trial_k == cur_k:
            stats.phenotype_skips += 1
            ev.record_phenotype_skip()
            if strict:
                stats.rejected += 1
                continue
            stats.accepted_neutral += 1
            gomacc.append({"type": "pheno_skip", "ham": ham(trial, cur_g)})
            cur_g = trial
            continue
        try:
            trial_s = ev.evaluate(trial)
        except BudgetExhausted as exc:
            raise GomInterrupted(Individual(cur_g, cur_s, cur_k), exc.reason) from exc
        stats.evaluated += 1
        h = ham(trial, cur_g)
        dcx = int(trial_s.get_complexity() - cur_s.get_complexity())
        if ev.worse(trial_s, cur_s) or (strict and not ev.worse(cur_s, trial_s)):
            stats.rejected += 1
            gomacc.append({"type": "rejected", "ham": h, "dcx": dcx})
            continue
        if ev.worse(cur_s, trial_s):
            stats.accepted_improving += 1
            improved = True
            draw = float(trial_s.get_score() - cur_s.get_score())
            # would HC (min_score_improv 0.5 on penalized) count this as an improvement?
            dpen = float(trial_s.get_penalized_score() - cur_s.get_penalized_score())
            typ = "acc_improving" if dpen > 0.5 else ("acc_improving_small" if dpen > 0 else "acc_improving_cxtie")
            gomacc.append({"type": typ, "ham": h, "dcx": dcx, "draw": draw})
        else:
            stats.accepted_neutral += 1
            gomacc.append({"type": "acc_neutral", "ham": h, "dcx": dcx})
        cur_g, cur_s, cur_k = trial, trial_s, trial_k
    offspring = Individual(cur_g, cur_s, cur_k)
    return offspring, cur_g != recipient.genotype, improved


eng_mod.gom = gom_instr

rc = 0
try:
    rc = moses_exec(["moses", *args])
except SystemExit as e:
    rc = e.code
emit("done", rc=rc)
