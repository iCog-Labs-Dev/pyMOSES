"""Analysis of instr.py dumps (system python3 + numpy).

Per generation record: diversity, LT-B and MI similarity vs a column-permutation
null, tree stability, GOM attribution, epistasis probe aggregates.
Usage: analyze.py NPERM OUT.json run1.jsonl [run2.jsonl ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np

NPERM = int(sys.argv[1])
OUTP = sys.argv[2]
FILES = sys.argv[3:]
rng = np.random.default_rng(7)


def entropies(X):
    """H (d x d): marginal on diag, joint off diag (natural log)."""
    N, d = X.shape
    K = X.max(axis=0) + 1
    offs = np.concatenate([[0], np.cumsum(K)[:-1]])
    O = np.zeros((N, K.sum()), dtype=np.float64)
    O[np.arange(N)[:, None], offs[None, :] + X] = 1.0
    C = O.T @ O
    P = C / N
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(P > 0, -P * np.log(P), 0.0)
    H = np.add.reduceat(np.add.reduceat(T, offs, axis=0), offs, axis=1)
    # diagonal block of a column with itself = marginal entropy (diag of one-hot outer)
    return H


def sims(H, H0):
    h = np.diag(H)
    h0 = np.diag(H0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ltb = (h / h0)[:, None] + (h / h0)[None, :] - 2.0 * H / H0
    mi = h[:, None] + h[None, :] - H
    return ltb, mi


def curveball_null(X, trades_per_row=5):
    """Null that keeps each row's number of deviations from the column mode and each
    column's multiset of values: curveball trades on the deviation matrix, then the
    column's non-mode values are permuted over its new deviating cells."""
    N, d = X.shape
    modes = np.array([np.bincount(X[:, c]).argmax() for c in range(d)])
    D = X != modes[None, :]
    rows = [set(np.flatnonzero(D[r])) for r in range(N)]
    for _ in range(trades_per_row * N):
        a, b = rng.choice(N, 2, replace=False)
        A, B = rows[a], rows[b]
        only_a, only_b = list(A - B), list(B - A)
        pool = only_a + only_b
        if not pool:
            continue
        perm = rng.permutation(len(pool))
        na = len(only_a)
        new_a = {pool[i] for i in perm[:na]}
        new_b = {pool[i] for i in perm[na:]}
        common = A & B
        rows[a] = common | new_a
        rows[b] = common | new_b
    Xn = np.tile(modes, (N, 1))
    for c in range(d):
        vals = X[D[:, c], c]
        cells = [r for r in range(N) if c in rows[r]]
        Xn[cells, c] = rng.permutation(vals)
    return Xn


def analyze_sim(X, H0, nperm, null="perm"):
    N, d = X.shape
    H = entropies(X)
    ltb, mi = sims(H, H0)
    iu = np.triu_indices(d, 1)
    varying = np.array([len(np.unique(X[:, c])) > 1 for c in range(d)])
    vp = varying[iu[0]] & varying[iu[1]]
    obs_l, obs_m = ltb[iu], mi[iu]
    null_l = np.empty((nperm, len(obs_l)), dtype=np.float32)
    null_m = np.empty((nperm, len(obs_l)), dtype=np.float32)
    for k in range(nperm):
        if null == "perm":
            Xp = np.empty_like(X)
            for c in range(d):
                Xp[:, c] = X[rng.permutation(N), c]
        else:
            Xp = curveball_null(X)
        Hp = entropies(Xp)
        lp, mp = sims(Hp, H0)
        null_l[k] = lp[iu]
        null_m[k] = mp[iu]
    out = {"n_pairs": int(len(obs_l)), "n_pairs_varying": int(vp.sum())}
    for name, obs, null in (("ltb", obs_l, null_l), ("mi", obs_m, null_m)):
        p95 = np.percentile(null, 95, axis=0)
        p99 = np.percentile(null, 99, axis=0)
        # exact-tie safe: strict exceedance, and per-pair empirical p-value
        pval = ((null >= obs[None, :] - 1e-12).sum(axis=0) + 1) / (nperm + 1)
        sel = vp
        nmax = null[:, sel].max(axis=1) if sel.any() else np.array([np.nan])
        o = obs[sel]
        out[name] = {
            "share_gt_p95": float((o > p95[sel] + 1e-12).mean()) if sel.any() else None,
            "share_gt_p99": float((o > p99[sel] + 1e-12).mean()) if sel.any() else None,
            "share_p_le_0.01": float((pval[sel] <= 0.01).mean()) if sel.any() else None,
            "top_obs": float(o.max()) if sel.any() else None,
            "null_max_p50": float(np.median(nmax)),
            "null_max_p95": float(np.percentile(nmax, 95)),
            "top_vs_nullmax_rank": float((nmax < o.max()).mean()) if sel.any() else None,
            "abs_max_all": float(np.abs(obs).max()),
            "n_distinct_values": int(len(np.unique(np.round(o, 9)))) if sel.any() else 0,
        }
    return out


def diversity(X):
    N, d = X.shape
    varying = [len(np.unique(X[:, c])) > 1 for c in range(d)]
    # minority count per varying column (size of non-modal group)
    mins = []
    for c in range(d):
        if varying[c]:
            _, cnt = np.unique(X[:, c], return_counts=True)
            mins.append(N - cnt.max())
    ham = []
    for a in range(N):
        ham.append((X[a + 1:] != X[a]).sum(axis=1))
    ham = np.concatenate(ham) if N > 1 else np.array([0])
    return {
        "d_linked": d,
        "n_varying": int(sum(varying)),
        "share_varying": float(np.mean(varying)) if d else 0.0,
        "mean_hamming": float(ham.mean()),
        "unique_genotypes": int(len({tuple(r) for r in X.tolist()})),
        "share_minority_eq1": float(np.mean([m == 1 for m in mins])) if mins else None,
        "median_minority": float(np.median(mins)) if mins else None,
    }


def subsets(fos):
    return {frozenset(s) for s in (fos or []) if len(s) >= 2}


results = []
agg_att = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
agg_imp = defaultdict(lambda: defaultdict(int))
agg_epi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for fn in FILES:
    tag = fn.split("/")[-1].removesuffix(".jsonl")
    prev = {}
    for line in open(fn):
        r = json.loads(line)
        if "done" in r or "term" in r:
            continue
        X = np.array(r["pop"], dtype=np.int64)
        H0 = np.array(r["H0"])
        rec = {"run": tag, "deme": r["deme"], "gen": r["gen"], "N": r["N"], "mode": r["mode"]}
        rec["div"] = diversity(X)
        if X.shape[1] >= 2:
            rec["sim_all"] = analyze_sim(X, H0, NPERM)
            rec["sim_all_cb"] = analyze_sim(X, H0, max(20, NPERM // 2), null="cb")
            if r.get("sel"):
                idx = [r["linked"].index(p) for p in r["sel"] if p in r["linked"]]
                if len(idx) >= 2:
                    rec["sim_sel"] = analyze_sim(X[:, idx], H0[np.ix_(idx, idx)], NPERM)
                    rec["sim_sel_cb"] = analyze_sim(X[:, idx], H0[np.ix_(idx, idx)], max(20, NPERM // 2), null="cb")
                    rec["div_sel"] = diversity(X[:, idx])
        cur = subsets(r["fos"])
        key = (tag, r["deme"])
        if key in prev and prev[key] and cur:
            rec["stab_retained"] = len(prev[key] & cur) / len(prev[key])
        prev[key] = cur
        sizes = [len(s) for s in (r["fos"] or [])]
        rec["fos_n"] = len(sizes)
        rec["fos_size_hist"] = {b: sum(1 for s in sizes if (s == 1 if b == "1" else (2 <= s <= 3 if b == "2-3" else (4 <= s <= 15 if b == "4-15" else s >= 16)))) for b in ("1", "2-3", "4-15", "16+")}
        rec["att"] = r["att"]
        rec["n_improve"] = len(r["improvements"])
        group = f"{tag.split('_')[0]}|{r['mode']}"
        for b, a in r["att"].items():
            for k, v in a.items():
                agg_att[group][b][k] += v
        for imp in r["improvements"]:
            agg_imp[group]["n"] += 1
            agg_imp[group]["near_root"] += imp["near_root"]
            agg_imp[group]["nchg1"] += imp["nchg"] == 1
            agg_imp[group]["nchg2_3"] += 2 <= imp["nchg"] <= 3
            agg_imp[group]["nchg4p"] += imp["nchg"] >= 4
        if r.get("epi"):
            for gname, lst in r["epi"].items():
                eg = agg_epi[f"{tag.split('_')[0]}|gen{min(r['gen'], 1)}+"][gname]
                for p in lst:
                    eg["n"] += 1
                    for k in ("pen_epi", "raw_epi", "pen_any_effect", "raw_joint_only_gain",
                              "pen_joint_only_gain", "donor_avail"):
                        eg[k] += bool(p[k])
                    eg["pen_epi_given_effect"] += bool(p["pen_epi"] and p["pen_any_effect"])
        results.append(rec)

json.dump({"records": results,
           "att": {g: {b: dict(v) for b, v in d.items()} for g, d in agg_att.items()},
           "imp": {g: dict(v) for g, v in agg_imp.items()},
           "epi": {g: {k: dict(v) for k, v in d.items()} for g, d in agg_epi.items()}},
          open(OUTP, "w"), indent=1)
print("wrote", OUTP, len(results), "records")
