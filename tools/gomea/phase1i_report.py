"""Phase 1i report: does a starting population spread evenly over distances help GOMEA?

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1i_report.py --stage {1,2} OUT_DIR RESULT_DIR [...]

Frozen maj5 at N = 256. Arms (by name): `LT-B-N256` and `RT-N256` use the default start
(radius mixture 1:0.4,2:0.3,4:0.2,max:0.1); `LT-B-UNI-N256` and `RT-UNI-N256` use
`--gomea-radii uniform` (every distance 1..all knobs equally likely). Runs pair by (fixture,
seed). The metric is final error, the wrong rows of 32 (`-best_raw`; a run that is not
`ok` counts as 32).

Primary contrasts, Holm-corrected together:

* `start`: LT-B with the uniform start vs LT-B with the default start;
* `interaction`: (LT-B - RT) with the uniform start minus (LT-B - RT) with the default
  start, per pair. Negative means a diverse start widens learned linkage's lead.

Two looks at ALPHA = 0.025 each (0.05 overall). A contrast is decided ("uniform better" /
"uniform worse") when Holm p < ALPHA; at stage 1, ALPHA <= Holm p < EXTEND_BELOW is
"extend" (10 more seeds per fixture); anything else is "no difference". See
PHASE1I-PREREGISTRATION.md.
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

ALPHA = 0.025
EXTEND_BELOW = 0.25
ARMS = ("LT-B-N256", "RT-N256", "LT-B-UNI-N256", "RT-UNI-N256")


def _err(s: dict) -> float:
    if s.get("status") != "ok" or s.get("best_raw") is None:
        return 32.0
    return -s["best_raw"]


def _row(name: str, d: list[float]) -> dict:
    return {
        "contrast": name,
        "pairs": len(d),
        "negative": sum(x < 0 for x in d),
        "zero": sum(x == 0 for x in d),
        "positive": sum(x > 0 for x in d),
        "mean": sum(d) / len(d) if d else float("nan"),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


def main() -> int:
    if len(sys.argv) < 5 or sys.argv[1] != "--stage" or sys.argv[2] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage, out = int(sys.argv[2]), Path(sys.argv[3])
    runs: dict = {a: {} for a in ARMS}
    for d in sys.argv[4:]:
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            if s["arm"] in runs and s["task"].startswith("maj5-"):
                runs[s["arm"]][(s["task"], s["seed"])] = s
    keys = sorted(set.intersection(*(set(v) for v in runs.values())))
    e = {a: [_err(runs[a][k]) for k in keys] for a in ARMS}
    lb, rt, lbu, rtu = (e[a] for a in ARMS)
    primary = [
        _row("start", [u - d for u, d in zip(lbu, lb)]),
        _row("interaction", [(a - b) - (c - d) for a, b, c, d in zip(lbu, rtu, lb, rt)]),
    ]
    for r, ph in zip(primary, holm([r["p"] for r in primary]), strict=True):
        r["p_holm"] = ph
        if ph < ALPHA:
            r["verdict"] = "uniform better" if r["mean"] < 0 else "uniform worse"
        elif stage == 1 and ph < EXTEND_BELOW:
            r["verdict"] = "extend"
        else:
            r["verdict"] = "no difference"
    descriptive = [
        _row("RT uniform - RT default", [u - d for u, d in zip(rtu, rt)]),
        _row("LT-B - RT, uniform start", [a - b for a, b in zip(lbu, rtu)]),
        _row("LT-B - RT, default start", [a - b for a, b in zip(lb, rt)]),
    ]
    agg = []
    for a in ARMS:
        v = [runs[a][k] for k in keys]
        agg.append(
            {
                "arm": a,
                "runs": len(v),
                "not_ok": sum(s.get("status") != "ok" for s in v),
                "mean_err": sum(e[a]) / len(e[a]) if v else float("nan"),
                "solved": sum(x == 0 for x in e[a]),
                "median_evals": statistics.median(s.get("n_calls") or 0 for s in v) if v else 0,
                "median_wall_s": statistics.median(s.get("wall_s") or 0 for s in v) if v else 0,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    rows = primary + descriptive
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "contrast", k))
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out / "aggregate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    (out / "verdict.json").write_text(
        json.dumps({r["contrast"]: r["verdict"] for r in primary}, indent=2) + "\n"
    )
    for r in agg + rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
