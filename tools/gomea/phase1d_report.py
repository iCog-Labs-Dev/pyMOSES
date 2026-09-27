"""Phase 1d report (PHASE1D-PREREGISTRATION.md): why did MOSES HC beat embedded GOMEA?

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1d_report.py stage1 RESULT_DIR OUT_DIR
    .venv/bin/python tools/gomea/phase1d_report.py stage2 NEW_E2E_DIR[,DIR] PHASE1_DIR[,DIR] OUT_DIR

**stage1** reads a frozen-representation result directory written by
`benchmarks.gomea.runner` (tasks named `<family>-x<k>`). For every arm and task family it
reports:

* the best error reached by 1,000 and 5,000 evaluations, from each run's scorer trace;
* the mechanism metrics, from the GOMEA telemetry and the harness counters;
* paired Wilcoxon tests of each factor arm vs `LT-B` over (fixture, seed) units, with Holm
  correction across arms × checkpoints within each family;
* the pre-registered carry-forward decision.

**stage2** pairs the new end-to-end arms by seed with the Phase 1 `HC`, `LT-B` and `UNI`
runs. It applies the Phase 1 rule: paired Wilcoxon on final error and on censored
evaluations-to-success, Holm within each task, and the practical thresholds of 10
percentage points of success or 20 % fewer evaluations.
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

FACTOR_ARMS = ["LTB-WIDE", "LTB-UNIF", "LTB-N32", "LTB-N64", "LTB-MEMO", "LTB-STRICT"]
CHECKPOINTS = (1000, 5000)


def _summaries(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(glob.glob(str(d / "runs/*/summary.json"))):
            s = json.loads(Path(f).read_text())
            s["_dir"] = str(Path(f).parent)
            out.append(s)
    return out


def error_by(run_dir: str, checkpoint: int) -> float | None:
    """Best error among scorer calls 1..checkpoint (events are logged only on improvement)."""
    best = None
    p = Path(run_dir) / "trace.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if e.get("n", 10**12) <= checkpoint and "raw" in e:
            best = -e["raw"] if best is None else min(best, -e["raw"])
    return best


def mechanisms(run_dir: str) -> dict:
    """GOMEA telemetry for one frozen run (exactly one deme)."""
    p = Path(run_dir) / "events.jsonl"
    m = {
        "unique_end": None,
        "converged": None,
        "entropy_gen1": None,
        "improving_share": None,
        "memo_hits": None,
        "generations": None,
    }
    if not p.exists():
        return m
    gens, end = [], None
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if e["type"] == "generation":
            gens.append(e)
        elif e["type"] == "end":
            end = e
    if gens:
        m["unique_end"] = gens[-1].get("unique_genotypes")
        m["entropy_gen1"] = gens[0].get("mean_entropy_bits")
        ev = sum(g["stats"].get("evaluated", 0) for g in gens)
        im = sum(g["stats"].get("accepted_improving", 0) for g in gens)
        m["improving_share"] = im / ev if ev else None
    if end:
        m["converged"] = end["termination_reason"] == "converged"
        m["memo_hits"] = end["ledger"].get("memo_hits")
        m["generations"] = end["generations"]
    return m


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def stage1(result_dir: Path, out: Path) -> None:
    runs = _summaries([result_dir])
    unit = {}  # (family, arm) -> {(fixture, seed): run}
    for s in runs:
        fam = s["task"].split("-x")[0]
        unit.setdefault((fam, s["arm"]), {})[(s["task"], s["seed"])] = s
    fams = sorted({f for f, _ in unit})
    arms = sorted({a for _, a in unit})
    agg, tests = [], []
    for fam in fams:
        for arm in arms:
            rs = list(unit.get((fam, arm), {}).values())
            if not rs:
                continue
            mech = [mechanisms(s["_dir"]) for s in rs]
            calls = sum(s.get("scorer_calls_total") or 0 for s in rs)
            dups = sum(s.get("scorer_duplicate_calls") or 0 for s in rs)
            agg.append(
                {
                    "family": fam,
                    "arm": arm,
                    "n": len(rs),
                    **{
                        f"median_error_by_{c}": _med([error_by(s["_dir"], c) for s in rs])
                        for c in CHECKPOINTS
                    },
                    "solved_by_5000": sum(1 for s in rs if error_by(s["_dir"], 5000) == 0),
                    "median_evals_used": _med([s.get("n_calls") for s in rs]),
                    "duplicate_call_share": dups / calls if calls else None,
                    "median_unique_genotypes_end": _med([m["unique_end"] for m in mech]),
                    "converged_share": (
                        sum(1 for m in mech if m["converged"])
                        / sum(1 for m in mech if m["converged"] is not None)
                    )
                    if any(m["converged"] is not None for m in mech)
                    else None,
                    "median_entropy_after_gen1": _med([m["entropy_gen1"] for m in mech]),
                    "median_improving_share": _med([m["improving_share"] for m in mech]),
                    "median_memo_hits": _med([m["memo_hits"] for m in mech]),
                    "median_generations": _med([m["generations"] for m in mech]),
                }
            )
        base = unit.get((fam, "LT-B"), {})
        fam_tests = []
        for arm in FACTOR_ARMS:
            other = unit.get((fam, arm), {})
            keys = sorted(set(base) & set(other))
            for c in CHECKPOINTS:
                diffs = []
                for k in keys:
                    a, b = error_by(other[k]["_dir"], c), error_by(base[k]["_dir"], c)
                    if a is not None and b is not None:
                        diffs.append(a - b)
                _, p = wilcoxon_signed_rank(diffs)
                fam_tests.append(
                    {
                        "family": fam,
                        "arm": arm,
                        "checkpoint": c,
                        "n": len(diffs),
                        "wins": sum(d < 0 for d in diffs),
                        "ties": sum(d == 0 for d in diffs),
                        "losses": sum(d > 0 for d in diffs),
                        "median_diff": _med(diffs),
                        "p": p,
                    }
                )
        for t, ph in zip(fam_tests, holm([t["p"] for t in fam_tests]), strict=True):
            t["p_holm"] = ph
            t["direction"] = (
                "better"
                if ph < 0.05 and t["wins"] > t["losses"]
                else "worse"
                if ph < 0.05 and t["losses"] > t["wins"]
                else "ns"
            )
        tests += fam_tests
    carried = []
    for arm in FACTOR_ARMS:
        ts = [t for t in tests if t["arm"] == arm]
        if any(t["direction"] == "better" for t in ts) and not any(
            t["direction"] == "worse" for t in ts
        ):
            carried.append(arm)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stage1_aggregate.csv", agg), ("stage1_tests.csv", tests)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "stage1_decision.json").write_text(
        json.dumps({"carried_factors": carried, "always_carried": ["LTB-CAP"]}, indent=2)
    )
    for r in agg:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for t in tests:
        print(
            f"{t['family']:5s} {t['arm']:11s} @{t['checkpoint']:5d} W/T/L {t['wins']}/{t['ties']}/"
            f"{t['losses']} p_holm {t['p_holm']:.3g} {t['direction']}"
        )
    print("carried:", carried)


def _censored(s: dict, budget: int) -> int:
    e = s.get("evals_to_target")
    return e if (s.get("success") and e is not None) else budget + 1


def stage2(new_dirs: list[Path], p1_dirs: list[Path], out: Path, budget: int = 20000) -> None:
    runs = defaultdict(dict)  # (task, arm) -> {seed: summary}
    for s in _summaries(p1_dirs):
        if s["arm"] in ("HC", "LT-B", "UNI"):
            runs[(s["task"], s["arm"])][s["seed"]] = s
    for s in _summaries(new_dirs):
        runs[(s["task"], s["arm"])][s["seed"]] = s
    tasks = sorted({t for t, _ in runs})
    new_arms = sorted({s["arm"] for s in _summaries(new_dirs)})
    agg, rows = [], []
    for task in tasks:
        for arm in sorted({a for t, a in runs if t == task}):
            rs = list(runs[(task, arm)].values())
            agg.append(
                {
                    "task": task,
                    "arm": arm,
                    "n": len(rs),
                    "successes": sum(1 for s in rs if s.get("success")),
                    "median_error": _med(
                        [-s["best_raw"] for s in rs if s.get("best_raw") is not None]
                    ),
                    "median_evals_censored": _med([_censored(s, budget) for s in rs]),
                    "median_cpu_s": _med([s.get("cpu_s") for s in rs]),
                    "median_deme_count": _med([s.get("deme_count") for s in rs]),
                }
            )
        task_rows = []
        for arm in new_arms:
            for ref in ("LT-B", "HC"):
                a, b = runs.get((task, arm), {}), runs.get((task, ref), {})
                seeds = sorted(set(a) & set(b))
                if not seeds:
                    continue
                ea = [-a[s]["best_raw"] for s in seeds]
                eb = [-b[s]["best_raw"] for s in seeds]
                ca = [_censored(a[s], budget) for s in seeds]
                cb = [_censored(b[s], budget) for s in seeds]
                sa = sum(1 for s in seeds if a[s].get("success")) / len(seeds)
                sb = sum(1 for s in seeds if b[s].get("success")) / len(seeds)
                de = [x - y for x, y in zip(ea, eb, strict=True)]
                task_rows.append(
                    {
                        "task": task,
                        "contrast": f"{arm}:{ref}",
                        "n": len(seeds),
                        "success_a": sa,
                        "success_b": sb,
                        "wins": sum(d < 0 for d in de),
                        "ties": sum(d == 0 for d in de),
                        "losses": sum(d > 0 for d in de),
                        "p_error": wilcoxon_signed_rank(de)[1],
                        "p_evals": wilcoxon_signed_rank(
                            [x - y for x, y in zip(ca, cb, strict=True)]
                        )[1],
                        "median_evals_a": _med(ca),
                        "median_evals_b": _med(cb),
                    }
                )
        adj = holm([p for r in task_rows for p in (r["p_error"], r["p_evals"])])
        for i, r in enumerate(task_rows):
            r["p_error_holm"], r["p_evals_holm"] = adj[2 * i], adj[2 * i + 1]
            sig = min(r["p_error_holm"], r["p_evals_holm"]) < 0.05
            a_p = (
                r["success_a"] >= r["success_b"] + 0.10
                or r["median_evals_a"] <= 0.8 * r["median_evals_b"]
            )
            b_p = (
                r["success_b"] >= r["success_a"] + 0.10
                or r["median_evals_b"] <= 0.8 * r["median_evals_a"]
            )
            r["verdict"] = "better" if sig and a_p else "worse" if sig and b_p else "tie"
        rows += task_rows
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("stage2_aggregate.csv", agg), ("stage2_verdicts.csv", rows)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    for r in agg:
        print(r)
    for r in rows:
        print(
            f"{r['task']:5s} {r['contrast']:20s} {r['verdict']:6s} succ {r['success_a']:.2f}/"
            f"{r['success_b']:.2f} W/T/L {r['wins']}/{r['ties']}/{r['losses']} "
            f"pE {r['p_error_holm']:.3g} pV {r['p_evals_holm']:.3g}"
        )


def main() -> int:
    mode = sys.argv[1]
    if mode == "stage1":
        stage1(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "stage2":
        stage2(
            [Path(p) for p in sys.argv[2].split(",")],
            [Path(p) for p in sys.argv[3].split(",")],
            Path(sys.argv[4]),
        )
    else:
        raise SystemExit("mode must be stage1 or stage2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
