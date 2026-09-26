"""Phase 1 confirmatory report: apply the pre-registered decision rule and build the
machine-readable findings bundle.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1_report.py RESULTS_ROOT OUT_DIR

RESULTS_ROOT is the runner's `<output_root>/1` directory. It must contain the
`confirm-e2e-*`, `confirm-e2e-pa4-*` and `confirm-frozen-*` result directories, each already
analysed by `benchmarks.gomea.analyze` (the driver does that).

The operational form of PREREGISTRATION.md's rule was written before any end-to-end
confirmatory result was seen. For a contrast A:B on a task:

* "significant" means that the Holm-adjusted p of the error test OR of the
  evaluations-to-success test is < 0.05;
* "practically better" means that A's success rate is >= B's + 0.10, OR that A's median
  censored evaluations-to-success is <= 0.8 x B's (failures count as budget + 1);
* the verdict is **better** if significant and practically better; **worse** if significant
  and B is practically better than A (the mirrored rule); and **tie** otherwise.

Frozen comparison (secondary, descriptive): the unit is a (fixture, seed) pair, pooled
within each task family (pa4, maj5, mux6). The tables give a paired Wilcoxon test on final
error with Holm correction across the family's contrasts, W/T/L, and the evaluations used.

Standard library only; it reuses the harness statistics.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import bootstrap_median_ci, holm, wilcoxon_signed_rank  # noqa: E402

CONTRASTS = [
    ("LT-B", "HC"),
    ("LT-B", "LT-MI"),
    ("LT-B", "UNI"),
    ("RT", "LT-B"),
    ("LT-B-FI", "LT-B"),
]
PRIMARY = {("LT-B", "HC"), ("LT-B", "LT-MI"), ("LT-B", "UNI")}
ARMS = ["HC", "UNI", "RT", "LT-MI", "LT-B", "LT-B-FI"]


def _one(pattern: str) -> Path:
    hits = [h for h in glob.glob(pattern) if os.path.isdir(h)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one directory for {pattern}, found {hits}")
    return Path(hits[0])


def _summaries(d: Path) -> list[dict]:
    return [
        json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(d / "runs/*/summary.json")))
    ]


def _read_csv(p: Path) -> list[dict]:
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _checksum(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(str(d / "runs/*/summary.json"))):
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def _budget(d: Path) -> int:
    argv = json.loads((d / "manifest.json").read_text())["budget"]["argv"]
    return int(argv[argv.index("-m") + 1])


def _censored_evals(s: dict, budget: int) -> int:
    e = s.get("evals_to_target")
    return int(e) if (s.get("success") and e is not None) else budget + 1


def e2e_verdicts(d: Path) -> tuple[list[dict], dict]:
    budget = _budget(d)
    agg = {(r["task"], r["arm"]): r for r in _read_csv(d / "analysis/aggregate_results.csv")}
    con = {(r["task"], r["contrast"]): r for r in _read_csv(d / "analysis/contrasts.csv")}
    runs = defaultdict(list)
    for s in _summaries(d):
        runs[(s["task"], s["arm"])].append(s)
    rows = []
    for task in sorted({t for t, _ in agg}):
        for a, b in CONTRASTS:
            c = con.get((task, f"{a}:{b}"))
            if c is None:
                continue
            sr_a, sr_b = (
                float(agg[(task, a)]["success_rate"]),
                float(agg[(task, b)]["success_rate"]),
            )
            me_a = statistics.median(_censored_evals(s, budget) for s in runs[(task, a)])
            me_b = statistics.median(_censored_evals(s, budget) for s in runs[(task, b)])

            def _f(x: str) -> float:
                return float(x) if x not in ("", "nan") else float("nan")

            p1, p2 = _f(c["p_error_holm"]), _f(c["p_evals_to_target_holm"])
            sig = (p1 < 0.05) or (p2 < 0.05)
            a_prac = (sr_a >= sr_b + 0.10) or (me_a <= 0.8 * me_b)
            b_prac = (sr_b >= sr_a + 0.10) or (me_b <= 0.8 * me_a)
            verdict = "better" if (sig and a_prac) else "worse" if (sig and b_prac) else "tie"
            rows.append(
                {
                    "task": task,
                    "contrast": f"{a}:{b}",
                    "primary": (a, b) in PRIMARY,
                    "n_paired": int(c["n_paired"]),
                    "wins": int(c["wins"]),
                    "ties": int(c["ties"]),
                    "losses": int(c["losses"]),
                    "success_a": sr_a,
                    "success_b": sr_b,
                    "median_evals_censored_a": me_a,
                    "median_evals_censored_b": me_b,
                    "p_error_holm": p1,
                    "p_evals_holm": p2,
                    "boot_ci_error": [_f(c["boot_ci_error_lo"]), _f(c["boot_ci_error_hi"])],
                    "verdict": verdict,
                }
            )
    return rows, {"budget": budget, "aggregate": list(agg.values())}


def frozen_pooled(d: Path) -> list[dict]:
    fam_runs = defaultdict(dict)  # family -> {(fixture, seed, arm): summary}
    for s in _summaries(d):
        fam = s["task"].split("-x")[0]
        fam_runs[fam][(s["task"], s["seed"], s["arm"])] = s
    out = []
    for fam in sorted(fam_runs):
        rr = fam_runs[fam]
        units = sorted({(t, sd) for t, sd, _ in rr})
        worst = (
            max((-s["best_raw"] for s in rr.values() if s.get("best_raw") is not None), default=0.0)
            + 1.0
        )

        def err(t, sd, arm, rr=rr, worst=worst):
            s = rr.get((t, sd, arm))
            return worst if (s is None or s.get("best_raw") is None) else -s["best_raw"]

        per_arm = {}
        for arm in ARMS:
            ss = [rr[(t, sd, arm)] for t, sd in units if (t, sd, arm) in rr]
            per_arm[arm] = {
                "n": len(ss),
                "solved": sum(1 for s in ss if s.get("success")),
                "median_error": statistics.median(err(t, sd, arm) for t, sd in units),
                "median_evals_used": statistics.median(s.get("n_calls") or 0 for s in ss)
                if ss
                else None,
                "median_cpu_s": statistics.median(s.get("cpu_s") or 0 for s in ss) if ss else None,
                "status": dict(Counter(s["status"] for s in ss)),
            }
        cons = []
        for a, b in CONTRASTS:
            diffs = [err(t, sd, a) - err(t, sd, b) for t, sd in units]
            _, p = wilcoxon_signed_rank(diffs)
            cons.append(
                {
                    "contrast": f"{a}:{b}",
                    "n_units": len(diffs),
                    "wins": sum(x < 0 for x in diffs),
                    "ties": sum(x == 0 for x in diffs),
                    "losses": sum(x > 0 for x in diffs),
                    "p_error": p,
                    "boot_ci_error": list(bootstrap_median_ci(diffs)) if diffs else None,
                }
            )
        for c, ph in zip(cons, holm([c["p_error"] for c in cons]), strict=True):
            c["p_error_holm"] = ph
        out.append(
            {
                "family": fam,
                "n_fixtures": len({t for t, _ in units}),
                "per_arm": per_arm,
                "contrasts": cons,
            }
        )
    return out


def gomea_behaviour(d: Path) -> dict:
    """Per arm: GOMEA demes per run, generations and evaluations per deme, and end reasons."""
    res = defaultdict(
        lambda: {"demes": [], "gens": [], "evals": [], "reasons": Counter(), "dims": [], "dup": []}
    )
    for ev in glob.glob(str(d / "runs/*/events.jsonl")):
        run = Path(ev).parent.name
        task, arm, _ = run.split("__")
        r = res[(task, arm)]
        n = 0
        for line in Path(ev).read_text().splitlines():
            e = json.loads(line)
            if e["type"] == "invocation_start":
                n += 1
                r["dims"].append(e["dim"])
            elif e["type"] == "end":
                r["gens"].append(e["generations"])
                r["evals"].append(e["ledger"]["requests"])
                r["dup"].append(e["ledger"].get("duplicate_requests", 0))
                r["reasons"][e["termination_reason"]] += 1
        r["demes"].append(n)
    out = {}
    for (task, arm), r in sorted(res.items()):

        def med(xs):
            return statistics.median(xs) if xs else None

        out[f"{task}/{arm}"] = {
            "median_demes_per_run": med(r["demes"]),
            "median_generations_per_deme": med(r["gens"]),
            "median_evals_per_deme": med(r["evals"]),
            "median_dim": med(r["dims"]),
            "duplicate_request_share": (sum(r["dup"]) / sum(r["evals"]))
            if sum(r["evals"])
            else None,
            "end_reasons": dict(r["reasons"]),
        }
    return out


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    e2e = _one(str(root / "confirm-e2e-[0-9a-f]*"))
    pa4 = _one(str(root / "confirm-e2e-pa4-*"))
    frz = _one(str(root / "confirm-frozen-*"))
    verdicts, meta = [], {}
    for d in (e2e, pa4):
        v, m = e2e_verdicts(d)
        verdicts += v
        meta[d.name] = m
    bundle = {
        "phase": 1,
        "code_commit": json.loads((e2e / "manifest.json").read_text())["commit"],
        "result_dirs": {
            d.name: {
                "summaries_sha256": _checksum(d),
                "n_runs": len(_summaries(d)),
                "status": dict(Counter(s["status"] for s in _summaries(d))),
            }
            for d in (e2e, pa4, frz)
        },
        "decision_rule": __doc__.split("The operational form")[1]
        .split("Frozen comparison")[0]
        .strip(),
        "e2e_verdicts": verdicts,
        "e2e_aggregate": meta,
        "frozen_pooled": frozen_pooled(frz),
        "gomea_behaviour": {
            "e2e": {**gomea_behaviour(e2e), **gomea_behaviour(pa4)},
            "frozen": gomea_behaviour(frz),
        },
    }
    (out / "findings.json").write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str))
    with (out / "e2e_verdicts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(verdicts[0].keys()))
        w.writeheader()
        w.writerows(verdicts)
    for d in (e2e, pa4, frz):
        for name in ("aggregate_results.csv", "contrasts.csv", "failures.md"):
            src = d / "analysis" / name
            if src.exists():
                (out / f"{d.name.rsplit('-', 1)[0]}__{name}").write_bytes(src.read_bytes())
    print(f"wrote {out}/findings.json and e2e_verdicts.csv")
    for v in verdicts:
        print(
            f"{v['task']:5s} {v['contrast']:13s} {v['verdict']:6s} succ {v['success_a']:.2f} vs "
            f"{v['success_b']:.2f}  W/T/L {v['wins']}/{v['ties']}/{v['losses']}  "
            f"p_err_holm {v['p_error_holm']:.3g}  p_evals_holm {v['p_evals_holm']:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
