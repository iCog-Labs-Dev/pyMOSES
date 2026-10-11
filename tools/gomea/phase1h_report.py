"""Phase 1h report: does learned linkage beat random linkage with a bigger sample?

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1h_report.py OUT_DIR RESULT_DIR [RESULT_DIR ...]

Each result directory holds frozen maj5 runs (`benchmarks/gomea/runner.py`) whose arms are
named `<FOS>-N<pop>`, with FOS one of LT-B, LT-MI and RT. Runs pair by (task, seed). The
metric is the final error, i.e. the wrong rows of 32 (`-best_raw`; a run that is not `ok`
counts as 32).

For each population size N, the contrasts LT-B vs RT and LT-MI vs RT (learned vs random
linkage) and LT-B vs LT-MI (the bias correction) are tested with a paired Wilcoxon test.
Holm correction runs over every learned-vs-random contrast across all N together. The
report also gives the mean paired difference with a 95 % bootstrap CI (10,000 resamples,
fixed seed), as the pre-registered equivalence check. See PHASE1H-PREREGISTRATION.md.
"""

from __future__ import annotations

import csv
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

MARGIN = 0.25


def _err(s: dict) -> float:
    if s.get("status") != "ok" or s.get("best_raw") is None:
        return 32.0
    return -s["best_raw"]


def _boot_ci(d: list[float]) -> tuple[float, float]:
    rng = random.Random(1)
    n = len(d)
    means = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(10_000))
    return means[249], means[9_749]


def main() -> int:
    out = Path(sys.argv[1])
    runs: dict = defaultdict(dict)
    meta: dict = defaultdict(list)
    for d in sys.argv[2:]:
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            fos, n = s["arm"].rsplit("-N", 1)
            runs[(int(n), fos)][(s["task"], s["seed"])] = s
            meta[(int(n), fos)].append(s)
    rows = []
    for n in sorted({n for n, _ in runs}):
        for a, b in (("LT-B", "RT"), ("LT-MI", "RT"), ("LT-B", "LT-MI")):
            ra, rb = runs.get((n, a), {}), runs.get((n, b), {})
            keys = sorted(set(ra) & set(rb))
            if not keys:
                continue
            d = [_err(ra[k]) - _err(rb[k]) for k in keys]
            lo, hi = _boot_ci(d)
            rows.append(
                {
                    "N": n,
                    "contrast": f"{a}:{b}",
                    "pairs": len(keys),
                    "a_lower": sum(x < 0 for x in d),
                    "ties": sum(x == 0 for x in d),
                    "b_lower": sum(x > 0 for x in d),
                    "mean_diff": sum(d) / len(d),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
                    "median_err_a": statistics.median(_err(ra[k]) for k in keys),
                    "median_err_b": statistics.median(_err(rb[k]) for k in keys),
                    "solved_a": sum(_err(ra[k]) == 0 for k in keys),
                    "solved_b": sum(_err(rb[k]) == 0 for k in keys),
                }
            )
    learned = [r for r in rows if r["contrast"].endswith(":RT")]
    for r, ph in zip(learned, holm([r["p"] for r in learned]), strict=True):
        r["p_holm"] = ph
    for r in rows:
        r.setdefault("p_holm", None)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = []
    for (n, fos), v in sorted(meta.items()):
        agg.append(
            {
                "N": n,
                "fos": fos,
                "runs": len(v),
                "not_ok": sum(s.get("status") != "ok" for s in v),
                "mean_err": sum(_err(s) for s in v) / len(v),
                "median_err": statistics.median(_err(s) for s in v),
                "solved": sum(_err(s) == 0 for s in v),
                "median_evals": statistics.median(s.get("n_calls") or 0 for s in v),
                "median_wall_s": statistics.median(s.get("wall_s") or 0 for s in v),
            }
        )
    with (out / "aggregate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    sig = [r for r in learned if r["p_holm"] < 0.05]
    better = [r for r in sig if r["mean_diff"] < 0]
    worse = [r for r in sig if r["mean_diff"] > 0]
    top = max(r["N"] for r in learned)
    top_rows = [r for r in learned if r["N"] == top]
    equivalent = (
        not sig
        and len(top_rows) == 2
        and all(-MARGIN <= r["ci_lo"] and r["ci_hi"] <= MARGIN for r in top_rows)
    )
    verdict = (
        "TRUE: learned linkage beats random linkage"
        if better and not worse
        else "FALSE: no meaningful difference at the largest N"
        if equivalent
        else "MIXED: learned linkage significantly worse somewhere"
        if worse
        else "INCONCLUSIVE"
    )
    (out / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "largest_N": top, "margin": MARGIN}, indent=2)
    )
    for r in agg:
        print(
            f"N={r['N']:5d} {r['fos']:6s} runs {r['runs']} mean err {r['mean_err']:.2f} "
            f"median {r['median_err']} solved {r['solved']} evals {r['median_evals']} "
            f"wall {r['median_wall_s']:.0f}"
        )
    for r in rows:
        ph = f"{r['p_holm']:.3g}" if r["p_holm"] is not None else "-"
        print(
            f"N={r['N']:5d} {r['contrast']:12s} pairs {r['pairs']} a<b {r['a_lower']} "
            f"= {r['ties']} a>b {r['b_lower']} mean {r['mean_diff']:+.3f} "
            f"CI [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p {r['p']:.3g} holm {ph}"
        )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
