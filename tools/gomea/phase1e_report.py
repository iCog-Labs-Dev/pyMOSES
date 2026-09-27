"""Phase 1e report (PHASE1E-PREREGISTRATION.md).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1e_report.py stage1 RESULT_DIR OUT_DIR
    .venv/bin/python tools/gomea/phase1e_report.py stage2 RESULT_DIR OUT_DIR

**stage1** applies the pre-registered selection rule to the development round:

1. the highest sum over tasks of the success rate;
2. then the lowest sum of median censored evaluations-to-success;
3. then the lowest sum of median final error.

It also writes descriptive comparisons, paired by seed, of every arm vs HC and of the two
memo contrasts.

**stage2** compares the stage 1 winner with HC, paired by seed. It runs a Wilcoxon test on
final error and on censored evaluations, applies Holm correction within each task, and
reports the verdict under the Phase 1 practical rule.

The budget is read from each result directory's manifest.
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
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank, wilson_ci  # noqa: E402


def _load(d: Path):
    runs = defaultdict(dict)
    for f in sorted(glob.glob(str(d / "runs/*/summary.json"))):
        s = json.loads(Path(f).read_text())
        runs[(s["task"], s["arm"])][s["seed"]] = s
    manifest = json.loads((d / "manifest.json").read_text())
    argv = manifest["budget"]["argv"]
    return runs, int(argv[argv.index("-m") + 1])


def _err(s: dict) -> float:
    return -s["best_raw"]


def _cens(s: dict, budget: int) -> int:
    e = s.get("evals_to_target")
    return e if (s.get("success") and e is not None and e <= budget) else budget + 1


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _paired(a: dict, b: dict, budget: int) -> dict:
    seeds = sorted(set(a) & set(b))
    de = [_err(a[s]) - _err(b[s]) for s in seeds]
    dv = [_cens(a[s], budget) - _cens(b[s], budget) for s in seeds]
    return {
        "n": len(seeds),
        "success_a": sum(1 for s in seeds if a[s].get("success")) / len(seeds),
        "success_b": sum(1 for s in seeds if b[s].get("success")) / len(seeds),
        "wins": sum(d < 0 for d in de),
        "ties": sum(d == 0 for d in de),
        "losses": sum(d > 0 for d in de),
        "p_error": wilcoxon_signed_rank(de)[1],
        "p_evals": wilcoxon_signed_rank(dv)[1],
        "median_evals_a": _med([_cens(a[s], budget) for s in seeds]),
        "median_evals_b": _med([_cens(b[s], budget) for s in seeds]),
        "median_error_a": _med([_err(a[s]) for s in seeds]),
        "median_error_b": _med([_err(b[s]) for s in seeds]),
    }


def _write(out: Path, name: str, rows: list[dict]) -> None:
    with (out / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _aggregate(runs, budget):
    agg = []
    for (task, arm), rs in sorted(runs.items()):
        v = list(rs.values())
        k = sum(1 for s in v if s.get("success"))
        lo, hi = wilson_ci(k, len(v))
        agg.append(
            {
                "task": task,
                "arm": arm,
                "n": len(v),
                "successes": k,
                "success_rate": k / len(v),
                "wilson_lo": lo,
                "wilson_hi": hi,
                "median_error": _med([_err(s) for s in v]),
                "median_evals_censored": _med([_cens(s, budget) for s in v]),
                "median_cpu_s": _med([s.get("cpu_s") for s in v]),
                "median_deme_count": _med([s.get("deme_count") for s in v]),
                "duplicate_call_share": (
                    sum(s.get("scorer_duplicate_calls") or 0 for s in v)
                    / max(1, sum(s.get("scorer_calls_total") or 0 for s in v))
                ),
            }
        )
    return agg


def stage1(d: Path, out: Path) -> None:
    runs, budget = _load(d)
    out.mkdir(parents=True, exist_ok=True)
    agg = _aggregate(runs, budget)
    tasks = sorted({t for t, _ in runs})
    arms = sorted({a for _, a in runs if a != "HC"})
    score = []
    for arm in arms:
        rows = [r for r in agg if r["arm"] == arm]
        if len(rows) != len(tasks):
            continue
        score.append(
            {
                "arm": arm,
                "sum_success_rate": sum(r["success_rate"] for r in rows),
                "sum_median_evals": sum(r["median_evals_censored"] for r in rows),
                "sum_median_error": sum(r["median_error"] for r in rows),
            }
        )
    score.sort(key=lambda r: (-r["sum_success_rate"], r["sum_median_evals"], r["sum_median_error"]))
    desc = []
    for task in tasks:
        for arm in arms:
            desc.append(
                {
                    "task": task,
                    "contrast": f"{arm}:HC",
                    **_paired(runs[(task, arm)], runs[(task, "HC")], budget),
                }
            )
        for a, b in (("LTB-C500-M", "LTB-C500"), ("UNI-C500-M", "UNI-C500")):
            if (task, a) in runs and (task, b) in runs:
                desc.append(
                    {
                        "task": task,
                        "contrast": f"{a}:{b}",
                        **_paired(runs[(task, a)], runs[(task, b)], budget),
                    }
                )
    _write(out, "stage1_aggregate.csv", agg)
    _write(out, "stage1_ranking.csv", score)
    _write(out, "stage1_descriptive.csv", desc)
    decision = {"winner": score[0]["arm"], "runner_up": score[1]["arm"], "ranking": score}
    (out / "stage1_decision.json").write_text(json.dumps(decision, indent=2))
    for r in agg:
        print(
            f"{r['task']:5s} {r['arm']:11s} {r['successes']:2d}/{r['n']} err {r['median_error']} "
            f"evals {r['median_evals_censored']} demes {r['median_deme_count']} "
            f"dup {r['duplicate_call_share']:.2f} cpu {r['median_cpu_s']:.0f}"
        )
    for r in score:
        print(r)
    print("winner:", decision["winner"], "runner-up:", decision["runner_up"])


def stage2(d: Path, out: Path) -> None:
    runs, budget = _load(d)
    out.mkdir(parents=True, exist_ok=True)
    agg = _aggregate(runs, budget)
    rows = []
    for task in sorted({t for t, _ in runs}):
        arms = sorted({a for t, a in runs if t == task and a != "HC"})
        for arm in arms:
            r = {
                "task": task,
                "contrast": f"{arm}:HC",
                **_paired(runs[(task, arm)], runs[(task, "HC")], budget),
            }
            r["p_error_holm"], r["p_evals_holm"] = holm([r["p_error"], r["p_evals"]])
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
            rows.append(r)
    _write(out, "stage2_aggregate.csv", agg)
    _write(out, "stage2_verdicts.csv", rows)
    for r in agg:
        print(
            f"{r['task']:5s} {r['arm']:11s} {r['successes']:2d}/{r['n']} err {r['median_error']} "
            f"evals {r['median_evals_censored']} demes {r['median_deme_count']} "
            f"cpu {r['median_cpu_s']:.0f}"
        )
    for r in rows:
        print(
            f"{r['task']:5s} {r['contrast']:15s} {r['verdict']:6s} succ {r['success_a']:.2f}/"
            f"{r['success_b']:.2f} W/T/L {r['wins']}/{r['ties']}/{r['losses']} "
            f"pE {r['p_error_holm']:.3g} pV {r['p_evals_holm']:.3g}"
        )


def main() -> int:
    mode, d, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"stage1": stage1, "stage2": stage2}[mode](d, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
