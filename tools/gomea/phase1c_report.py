"""Phase 1c report: how stand-alone GP-GOMEA scales with Boolean problem size, with classic
C++ MOSES hill climbing as a same-language reference (PHASE1C-PREREGISTRATION.md).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1c_report.py GPG_DIR MOSES_DIR OUT_DIR

`GPG_DIR` and `MOSES_DIR` are result directories (each has `runs/*/summary.json`) written by
`gpgomea_runner.py` and `moses_cpp_runner.py`.

The measures, per (task, system):

* the success rate within the budget, with a Wilson 95 % CI;
* the success rate at the checkpoints 20k / 100k / 500k evaluations;
* the median censored evaluations-to-solve (a failure counts as budget + 1);
* ERT (expected running time), defined as the evaluations a run spent until it succeeded
  or stopped, summed over all runs and divided by the number of successes (infinite with
  no success);
* the median final error, and median CPU and wall seconds.

Scaling is summarised by fitting log10(ERT) against the number of inputs, by least squares,
within each family over the sizes with finite ERT. The report gives the slope as a growth
factor per extra input, and it only fits when at least 2 points are finite.

Per task, the unpaired tests are Fisher's exact test on success and Mann–Whitney U on
censored evaluations. They are Holm-corrected within each family across tasks × metrics.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1b_report import fisher_exact, mann_whitney  # noqa: E402

from benchmarks.gomea.stats import holm, wilson_ci  # noqa: E402

FAMILIES = {
    "parity": ["pa3", "pa4", "pa5", "pa6", "pa7"],
    "majority": ["maj5", "maj7", "maj9"],
    "multiplexer": ["mux6", "mux11"],
}
INPUTS = {
    "pa3": 3,
    "pa4": 4,
    "pa5": 5,
    "pa6": 6,
    "pa7": 7,
    "maj5": 5,
    "maj7": 7,
    "maj9": 9,
    "mux6": 6,
    "mux11": 11,
}
CHECKPOINTS = (20000, 100000, 500000)
SYSTEMS = [("MOSES-CPP", "classic C++ MOSES (HC)"), ("GP-GOMEA", "GP-GOMEA (stand-alone)")]
LIGHT = ["#2a78d6", "#eb6834"]
DARK = ["#3987e5", "#d95926"]


def load(d: Path, system: str) -> dict:
    runs: dict = {}
    for f in sorted(glob.glob(str(d / "runs/*/summary.json"))):
        s = json.loads(Path(f).read_text())
        runs.setdefault((s["task"], system), []).append(s)
    return runs


def spent(s: dict, budget: int) -> int:
    if s.get("success") and s.get("evals_to_target") is not None:
        return s["evals_to_target"]
    used = s.get("evals_within_budget", s.get("n_calls"))
    return min(used if used is not None else budget, budget)


def censored(s: dict, budget: int) -> int:
    e = s.get("evals_to_target")
    return e if (s.get("success") and e is not None and e <= budget) else budget + 1


def summarise(rs: list[dict], budget: int) -> dict:
    n = len(rs)
    k = sum(1 for s in rs if censored(s, budget) <= budget)
    lo, hi = wilson_ci(k, n)
    total = sum(spent(s, budget) for s in rs)
    errs = [-s["best_raw"] if s.get("best_raw") is not None else None for s in rs]
    finite = [e for e in errs if e is not None]
    return {
        "n": n,
        "successes": k,
        "success_rate": k / n,
        "wilson_lo": lo,
        "wilson_hi": hi,
        **{
            f"success_at_{c}": sum(1 for s in rs if censored(s, budget) <= c) / n
            for c in CHECKPOINTS
            if c <= budget
        },
        "median_evals_censored": statistics.median(censored(s, budget) for s in rs),
        "ert": (total / k) if k else math.inf,
        "median_final_error": statistics.median(finite) if finite else None,
        "no_result": sum(1 for e in errs if e is None),
        "median_cpu_s": statistics.median(s.get("cpu_s") or 0 for s in rs),
        "median_wall_s": statistics.median(s.get("wall_s") or 0 for s in rs),
        "status": {st: sum(1 for s in rs if s["status"] == st) for st in {s["status"] for s in rs}},
    }


def fit_growth(points: list[tuple[int, float]]) -> dict | None:
    pts = [(x, math.log10(y)) for x, y in points if math.isfinite(y) and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return {
        "points": len(pts),
        "log10_slope_per_input": slope,
        "growth_factor_per_input": 10**slope,
    }


# ------------------------------------------------------------------------------ figures
def _css() -> str:
    def sw(cols: list[str]) -> str:
        return "".join(f".s{i + 1}{{stroke:{c}}}.d{i + 1}{{fill:{c}}}" for i, c in enumerate(cols))

    return (
        ".bg{fill:#fcfcfb}text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "fill:#52514e;font-size:12px}.t{fill:#0b0b0b;font-size:14px;font-weight:600}"
        ".grid{stroke:#e8e7e3;stroke-width:1}.line{fill:none;stroke-width:2;"
        "stroke-linejoin:round;stroke-linecap:round}.ring{stroke:#fcfcfb;stroke-width:2}"
        + sw(LIGHT)
        + "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}text{fill:#c3c2b7}"
        ".t{fill:#ffffff}.grid{stroke:#383835}.ring{stroke:#1a1a19}" + sw(DARK) + "}"
    )


def family_svg(
    family: str, tasks: list[str], agg: dict, budget: int, path: Path, metric: str
) -> None:
    """metric 'ert' (log y) or 'success' (0-100 %)."""
    w, h, ml, mr, mt, mb = 720, 400, 72, 40, 70, 56
    pw, ph = w - ml - mr, h - mt - mb
    xs = [INPUTS[t] for t in tasks]
    x0, x1 = min(xs) - 0.5, max(xs) + 0.5

    def sx(v: float) -> float:
        return ml + pw * (v - x0) / (x1 - x0)

    if metric == "ert":
        lo_e, hi_e = 2, math.ceil(math.log10(budget * 4))
        title = f"{family}: expected evaluations per success (ERT), log scale"

        def sy(v: float) -> float:
            return mt + ph * (1 - (math.log10(v) - lo_e) / (hi_e - lo_e))

        ticks = [(10**e, f"{10**e:,}") for e in range(lo_e, hi_e + 1)]
    else:
        title = f"{family}: share of runs solved within {budget:,} evaluations"

        def sy(v: float) -> float:
            return mt + ph * (1 - v)

        ticks = [(v, f"{int(v * 100)}%") for v in (0, 0.25, 0.5, 0.75, 1.0)]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-labelledby="ttl"><style>{_css()}</style>',
        f'<rect class="bg" width="{w}" height="{h}"/><title id="ttl">{title}</title>',
        f'<text class="t" x="{ml}" y="22">{title}</text>',
    ]
    lx = ml
    for i, (_s, name) in enumerate(SYSTEMS):
        out.append(
            f'<line class="line s{i + 1}" x1="{lx}" x2="{lx + 16}" y1="42" y2="42"/>'
            f'<text x="{lx + 22}" y="46">{name}</text>'
        )
        lx += 22 + 7 * len(name) + 22
    for v, lab in ticks:
        out.append(
            f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{sy(v):.1f}" '
            f'y2="{sy(v):.1f}"/><text x="{ml - 8}" y="{sy(v) + 4:.1f}" '
            f'text-anchor="end">{lab}</text>'
        )
    for t in tasks:
        out.append(
            f'<text x="{sx(INPUTS[t]):.1f}" y="{mt + ph + 18}" text-anchor="middle">{t}</text>'
        )
    out.append(
        f'<text x="{ml + pw / 2}" y="{h - 10}" text-anchor="middle">problem (number of '
        f"inputs grows left to right)</text>"
    )
    for i, (system, name) in enumerate(SYSTEMS):
        pts = []
        for t in tasks:
            a = agg.get((t, system))
            if a is None:
                continue
            v = a["ert"] if metric == "ert" else a["success_rate"]
            pts.append((t, v, a))
        seg = [(sx(INPUTS[t]), sy(v)) for t, v, _ in pts if metric != "ert" or math.isfinite(v)]
        if len(seg) > 1:
            out.append(
                f'<path class="line s{i + 1}" d="M'
                + " L".join(f"{x:.1f},{y:.1f}" for x, y in seg)
                + '"/>'
            )
        for t, v, a in pts:
            x = sx(INPUTS[t]) + (i - 0.5) * 6
            tip = (
                f"{name} on {t}: {a['successes']}/{a['n']} solved; ERT "
                f"{'none solved' if not math.isfinite(a['ert']) else f'{a["ert"]:,.0f}'}"
            )
            if metric == "ert" and not math.isfinite(v):
                y = mt + 4
                out.append(
                    f'<text x="{x:.1f}" y="{y + 8:.1f}" text-anchor="middle" '
                    f'class="d{i + 1}">×<title>{tip}</title></text>'
                )
                continue
            out.append(
                f'<circle class="d{i + 1} ring" cx="{x:.1f}" cy="{sy(v):.1f}" r="5">'
                f"<title>{tip}</title></circle>"
            )
    if metric == "ert":
        out.append(
            f'<text x="{ml + pw}" y="{mt - 6}" text-anchor="end">× = no run solved '
            f"within {budget:,}</text>"
        )
    out.append("</svg>")
    path.write_text("\n".join(out))


def main() -> int:
    gpg_dir, moses_dir, out = map(Path, sys.argv[1:4])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = {**load(moses_dir, "MOSES-CPP"), **load(gpg_dir, "GP-GOMEA")}
    budget = json.loads((gpg_dir / "manifest.json").read_text())["budget"]
    assert budget == json.loads((moses_dir / "manifest.json").read_text())["budget"]
    agg = {k: summarise(v, budget) for k, v in runs.items()}
    rows, tests, growth = [], [], {}
    for fam, tasks in FAMILIES.items():
        fam_tests = []
        for t in tasks:
            for system, _ in SYSTEMS:
                if (t, system) in agg:
                    rows.append(
                        {
                            "family": fam,
                            "task": t,
                            "inputs": INPUTS[t],
                            "system": system,
                            **agg[(t, system)],
                        }
                    )
            a, b = runs.get((t, "GP-GOMEA")), runs.get((t, "MOSES-CPP"))
            if a and b:
                ka = sum(1 for s in a if censored(s, budget) <= budget)
                kb = sum(1 for s in b if censored(s, budget) <= budget)
                fam_tests.append(
                    {
                        "family": fam,
                        "task": t,
                        "p_success": fisher_exact(ka, len(a) - ka, kb, len(b) - kb),
                        "p_evals": mann_whitney(
                            [censored(s, budget) for s in a], [censored(s, budget) for s in b]
                        )[1],
                        "success_gpg": ka / len(a),
                        "success_moses": kb / len(b),
                        "median_evals_gpg": statistics.median(censored(s, budget) for s in a),
                        "median_evals_moses": statistics.median(censored(s, budget) for s in b),
                    }
                )
        adj = holm([p for r in fam_tests for p in (r["p_success"], r["p_evals"])])
        for i, r in enumerate(fam_tests):
            r["p_success_holm"], r["p_evals_holm"] = adj[2 * i], adj[2 * i + 1]
        tests += fam_tests
        for system, _ in SYSTEMS:
            growth[f"{fam}/{system}"] = fit_growth(
                [(INPUTS[t], agg[(t, system)]["ert"]) for t in tasks if (t, system) in agg]
            )
        for metric in ("ert", "success"):
            family_svg(fam, tasks, agg, budget, out / "figures" / f"{fam}_{metric}.svg", metric)
    for name, data in (("scaling.csv", rows), ("tests.csv", tests)):
        with (out / name).open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)
    (out / "findings.json").write_text(
        json.dumps(
            {"budget": budget, "scaling": rows, "tests": tests, "growth": growth},
            indent=2,
            default=str,
        )
    )
    for r in rows:
        ert = "inf" if not math.isfinite(r["ert"]) else f"{r['ert']:,.0f}"
        print(
            f"{r['task']:6s} {r['system']:10s} solved {r['successes']:2d}/{r['n']:2d} "
            f"ERT {ert:>10s} med_err {r['median_final_error']} cpu {r['median_cpu_s']:.1f}s "
            f"wall {r['median_wall_s']:.1f}s status {r['status']}"
        )
    for r in tests:
        print(
            f"{r['task']:6s} GPG vs MOSES: succ {r['success_gpg']:.2f}/{r['success_moses']:.2f} "
            f"pS {r['p_success_holm']:.3g} pE {r['p_evals_holm']:.3g}"
        )
    print("growth:", json.dumps(growth, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
