"""Phase 1b report: stand-alone GP-GOMEA vs MOSES hill climbing (PHASE1B-PREREGISTRATION.md).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1b_report.py GPG_ROOT MOSES_ROOT OUT_DIR

`GPG_ROOT` is the GP-GOMEA runner's output root, which holds `phase1b-confirm`,
`phase1b-confirm-pa4` and `phase1b-secondary-200k`. `MOSES_ROOT` is the MOSES runner's
`<output_root>/1`, which holds `confirm-e2e-*` and `confirm-e2e-pa4-*`.

The two systems' random generators are unrelated, so the comparisons are **unpaired**:

* Fisher's exact test (two-sided) on success;
* Mann–Whitney U (two-sided; normal approximation with tie and continuity correction) on
  final error and on censored evaluations-to-success;
* Holm correction within each task across contrasts × metrics;
* a bootstrap CI of the difference in medians, resampling each group independently.

The decision rule is Phase 1's, with the unpaired tests substituted.

Standard library only.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilson_ci  # noqa: E402

BUDGET = 20000
CONTRASTS = [
    ("GPG-TUNED", "HC", True),
    ("GPG-DEFAULT", "HC", True),
    ("GPG-TUNED", "MOSES-UNI", False),
    ("GPG-TUNED", "MOSES-LT-B", False),
]
MOSES_ARMS = {"HC": "HC", "UNI": "MOSES-UNI", "LT-B": "MOSES-LT-B"}
# Figure series, in fixed categorical order (validated palette: dataviz references/palette.md).
SERIES = [
    ("HC", "MOSES hill climbing"),
    ("GPG-TUNED", "GP-GOMEA tuned"),
    ("GPG-DEFAULT", "GP-GOMEA defaults"),
    ("MOSES-UNI", "MOSES + GOMEA (uni)"),
]


# ----------------------------------------------------------------------------- statistics
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]: the sum of the hypergeometric
    probabilities of every table with the same margins that is no more likely than the
    observed one (the 1e-9 relative tolerance guards against float ties)."""
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    denom = math.comb(n, c1)

    def p(x: int) -> float:
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / denom

    p_obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-9)))


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float]:
    """(U of x, two-sided p). Normal approximation with tie correction and continuity
    correction; p = 1.0 when every value is tied."""
    n1, n2 = len(x), len(y)
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks: list[float] = [0.0] * len(allv)
    ties = []
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        ties.append(j - i + 1)
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv, strict=True) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    tie_term = sum(t**3 - t for t in ties)
    var = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return u1, 1.0
    z = (abs(u1 - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return u1, min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


def boot_median_diff(x: list[float], y: list[float], n: int = 10000, seed: int = 12345):
    rng = random.Random(seed)
    diffs = sorted(
        statistics.median(rng.choices(x, k=len(x))) - statistics.median(rng.choices(y, k=len(y)))
        for _ in range(n)
    )
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


# ----------------------------------------------------------------------------- loading
def _load(pattern: str) -> list[dict]:
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(pattern))]


def load_runs(gpg_root: Path, moses_root: Path) -> dict:
    runs: dict = {}
    for s in _load(str(gpg_root / "phase1b-confirm*/runs/*/summary.json")):
        runs.setdefault((s["task"], s["arm"]), []).append(s)
    for s in _load(str(moses_root / "confirm-e2e-*/runs/*/summary.json")):
        if s["arm"] in MOSES_ARMS:
            runs.setdefault((s["task"], MOSES_ARMS[s["arm"]]), []).append(s)
    return runs


def error_of(s: dict, worst: float) -> float:
    return worst if s.get("best_raw") is None else -s["best_raw"]


def censored(s: dict) -> int:
    e = s.get("evals_to_target")
    return e if (s.get("success") and e is not None and e <= BUDGET) else BUDGET + 1


# ----------------------------------------------------------------------------- figures
def ecdf_svg(task: str, runs: dict, path: Path) -> None:
    """Success probability vs evaluations, one 2px step line per system."""
    w, h, ml, mr, mt, mb = 720, 400, 64, 70, 64, 48
    pw, ph = w - ml - mr, h - mt - mb

    def sx(v: float) -> float:
        return ml + pw * v / BUDGET

    def sy(v: float) -> float:
        return mt + ph * (1 - v)

    # Validated categorical steps (dataviz palette, slots 1-4), light and dark. Literal
    # colours in CSS classes: SVG presentation attributes do not resolve var().
    light = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    dark = ["#3987e5", "#d95926", "#199e70", "#c98500"]

    def swatches(cols: list[str]) -> str:
        return "".join(f".s{i + 1}{{stroke:{c}}}.d{i + 1}{{fill:{c}}}" for i, c in enumerate(cols))

    css = (
        ".bg{fill:#fcfcfb}text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "fill:#52514e;font-size:12px}.t{fill:#0b0b0b;font-size:14px;font-weight:600}"
        ".grid{stroke:#e8e7e3;stroke-width:1}.line{fill:none;stroke-width:2;"
        "stroke-linejoin:round;stroke-linecap:round}"
        + swatches(light)
        + "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}text{fill:#c3c2b7}"
        ".t{fill:#ffffff}.grid{stroke:#383835}" + swatches(dark) + "}"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-labelledby="ttl"><style>{css}</style>',
        f'<rect class="bg" width="{w}" height="{h}"/>',
        f'<title id="ttl">{task}: share of runs solved vs evaluations used</title>',
        f'<text class="t" x="{ml}" y="22">{task}: share of runs solved within N evaluations</text>',
    ]
    lx = ml  # legend row under the title (always present for >= 2 series)
    for i, (_arm, name) in enumerate(SERIES):
        out.append(
            f'<line class="line s{i + 1}" x1="{lx}" x2="{lx + 16}" y1="41" y2="41"/>'
            f'<text x="{lx + 22}" y="45">{name}</text>'
        )
        lx += 22 + 7 * len(name) + 18
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(
            f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
            f'<text x="{ml - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>'
        )
    for v in range(0, BUDGET + 1, 5000):
        out.append(f'<text x="{sx(v):.1f}" y="{mt + ph + 18}" text-anchor="middle">{v:,}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{h - 8}" text-anchor="middle">evaluations</text>')
    labels = []
    for i, (arm, name) in enumerate(SERIES):
        rs = runs.get((task, arm), [])
        if not rs:
            continue
        hits = sorted(censored(s) for s in rs if censored(s) <= BUDGET)
        pts, frac = [(0.0, 0.0)], 0.0
        for k, e in enumerate(hits, 1):
            pts += [(e, frac), (e, k / len(rs))]
            frac = k / len(rs)
        pts.append((BUDGET, frac))
        d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(
            f'<path class="line s{i + 1}" d="{d}"><title>{name}: '
            f"{len(hits)}/{len(rs)} solved within {BUDGET:,}</title></path>"
        )
        labels.append([sy(frac), f"{len(hits)}/{len(rs)}", i + 1])
    labels.sort()
    for j in range(1, len(labels)):  # keep end labels at least 15px apart
        labels[j][0] = max(labels[j][0], labels[j - 1][0] + 15)
    for y, text, k in labels:
        out.append(
            f'<circle class="d{k}" cx="{ml + pw + 8}" cy="{y - 4:.1f}" r="4"/>'
            f'<text x="{ml + pw + 16}" y="{y:.1f}">{text}</text>'
        )
    out.append("</svg>")
    path.write_text("\n".join(out))


# ----------------------------------------------------------------------------- report
def main() -> int:
    gpg_root, moses_root, out = map(Path, sys.argv[1:4])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = load_runs(gpg_root, moses_root)
    tasks = sorted({t for t, _ in runs})
    agg, cons = [], []
    for task in tasks:
        worst = (
            max((error_of(s, 0) for (t, _), rs in runs.items() if t == task for s in rs), default=0)
            + 1
        )
        for (t, arm), rs in sorted(runs.items()):
            if t != task:
                continue
            errs = [error_of(s, worst) for s in rs]
            k = sum(1 for s in rs if censored(s) <= BUDGET)
            lo, hi = wilson_ci(k, len(rs))
            succ_evals = sorted(censored(s) for s in rs if censored(s) <= BUDGET)
            agg.append(
                {
                    "task": task,
                    "arm": arm,
                    "n": len(rs),
                    "successes": k,
                    "success_rate": k / len(rs),
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "error_median": statistics.median(errs),
                    "error_q1": statistics.quantiles(errs, n=4)[0],
                    "error_q3": statistics.quantiles(errs, n=4)[2],
                    "median_evals_censored": statistics.median(censored(s) for s in rs),
                    "median_evals_of_successes": statistics.median(succ_evals)
                    if succ_evals
                    else None,
                    "no_result_within_budget": sum(1 for s in rs if s.get("best_raw") is None),
                    "cpu_s_median": statistics.median(s.get("cpu_s") or 0 for s in rs),
                    "status": dict(Counter(s["status"] for s in rs)),
                }
            )
        rows = []
        for a, b, primary in CONTRASTS:
            ra, rb = runs.get((task, a)), runs.get((task, b))
            if not ra or not rb:
                continue
            ka = sum(1 for s in ra if censored(s) <= BUDGET)
            kb = sum(1 for s in rb if censored(s) <= BUDGET)
            ea, eb = [error_of(s, worst) for s in ra], [error_of(s, worst) for s in rb]
            ca, cb = [censored(s) for s in ra], [censored(s) for s in rb]
            rows.append(
                {
                    "task": task,
                    "contrast": f"{a}:{b}",
                    "primary": primary,
                    "n_a": len(ra),
                    "n_b": len(rb),
                    "success_a": ka / len(ra),
                    "success_b": kb / len(rb),
                    "p_success": fisher_exact(ka, len(ra) - ka, kb, len(rb) - kb),
                    "p_error": mann_whitney(ea, eb)[1],
                    "p_evals": mann_whitney(ca, cb)[1],
                    "median_error_a": statistics.median(ea),
                    "median_error_b": statistics.median(eb),
                    "boot_ci_error_diff": boot_median_diff(ea, eb),
                    "median_evals_a": statistics.median(ca),
                    "median_evals_b": statistics.median(cb),
                }
            )
        flat = [p for r in rows for p in (r["p_success"], r["p_error"], r["p_evals"])]
        adj = holm(flat)
        for i, r in enumerate(rows):
            r["p_success_holm"], r["p_error_holm"], r["p_evals_holm"] = adj[3 * i : 3 * i + 3]
            sig = min(r["p_success_holm"], r["p_error_holm"], r["p_evals_holm"]) < 0.05
            a_prac = (
                r["success_a"] >= r["success_b"] + 0.10
                or r["median_evals_a"] <= 0.8 * r["median_evals_b"]
            )
            b_prac = (
                r["success_b"] >= r["success_a"] + 0.10
                or r["median_evals_b"] <= 0.8 * r["median_evals_a"]
            )
            r["verdict"] = "better" if sig and a_prac else "worse" if sig and b_prac else "tie"
        cons += rows
        ecdf_svg(task, runs, out / "figures" / f"{task}_success_vs_evals.svg")
    sec = {}
    for s in _load(str(gpg_root / "phase1b-secondary-200k/runs/*/summary.json")):
        d = sec.setdefault(s["task"], {"n": 0, "solved": 0, "evals": [], "errors": []})
        d["n"] += 1
        d["solved"] += bool(s["success"])
        d["errors"].append(-s["best_raw"] if s["best_raw"] is not None else None)
        if s["success"]:
            d["evals"].append(s["evals_to_target"])
    secondary = {
        t: {
            "n": d["n"],
            "solved": d["solved"],
            "median_evals_of_successes": statistics.median(d["evals"]) if d["evals"] else None,
            "median_error": statistics.median(e for e in d["errors"] if e is not None),
        }
        for t, d in sorted(sec.items())
    }
    for name, rows in (("aggregate.csv", agg), ("contrasts.csv", cons)):
        with (out / name).open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    (out / "findings.json").write_text(
        json.dumps(
            {"budget": BUDGET, "aggregate": agg, "contrasts": cons, "secondary_200k": secondary},
            indent=2,
            default=str,
        )
    )
    for r in agg:
        print(
            f"{r['task']:5s} {r['arm']:12s} solved {r['successes']:2d}/{r['n']:2d} "
            f"err med {r['error_median']} evals(censored) med {r['median_evals_censored']} "
            f"cpu {r['cpu_s_median']:.1f}s noresult {r['no_result_within_budget']}"
        )
    for r in cons:
        print(
            f"{r['task']:5s} {r['contrast']:22s} {r['verdict']:6s} succ {r['success_a']:.2f}/"
            f"{r['success_b']:.2f} pS {r['p_success_holm']:.3g} pE {r['p_error_holm']:.3g} "
            f"pV {r['p_evals_holm']:.3g}"
        )
    print("secondary 200k:", secondary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
