"""Phase 1m report: budget-derived demes (population and cap from the budget) in full MOSES runs.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1m_report.py --stage {1,2} OUT_DIR MUX11_DIRS SR_DIRS

MUX11_DIRS and SR_DIRS are comma-separated result directories (stage 2 passes the stage-1 and
extension directories together), or `-` to leave a family out.

Arms: `HC`; `G-C250-N16` (current tuned GOMEA: cap 250, N 16, LT-B); `G-D10-LTB` and
`G-D10-RT` (cap = total / 10, N = 0.2 x cap clamped at 200, LT-B or random tree);
`G-D10-LTB-WIDE` (as G-D10-LTB with starting distances spread over 1, 2, 4, 8).

Metrics, paired by seed (mux11) or split (regression):

* **mux11:** evaluations to a solution (`evals_to_target`); unsolved or failed = 40,000.
* **Con, EnH, EnC:** test NMSE of the validation-selected program; failed = infinite.

Contrasts per problem (A minus B):

* primary (Holm over all 8): `design` = G-D10-LTB vs G-C250-N16, and `linkage` =
  G-D10-LTB vs G-D10-RT;
* secondary (Holm over all 8): `wide` = G-D10-LTB-WIDE vs G-D10-LTB, and `vs-hc` =
  G-D10-LTB vs HC.
Descriptive: G-C250-N16 vs HC.

Rule (each look at ALPHA = 0.025, 0.05 overall): Holm p < ALPHA and a median at least 10 %
lower is "A better" (mirror: "A worse"); for primary contrasts at stage 1,
ALPHA <= Holm p < EXTEND_BELOW is "extend"; otherwise "tie". See PHASE1M-PREREGISTRATION.md.
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
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

ALPHA = 0.025
EXTEND_BELOW = 0.25
MUX_BUDGET = 40000.0
ARMS = ("HC", "G-C250-N16", "G-D10-LTB", "G-D10-RT", "G-D10-LTB-WIDE")
PRIMARY = (("design", "G-D10-LTB", "G-C250-N16"), ("linkage", "G-D10-LTB", "G-D10-RT"))
SECONDARY = (("wide", "G-D10-LTB-WIDE", "G-D10-LTB"), ("vs-hc", "G-D10-LTB", "HC"))
DESCRIPTIVE = (("current-vs-hc", "G-C250-N16", "HC"),)


def _load(dirs: str) -> dict:
    runs: dict = {}
    for d in dirs.split(","):
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            if s["arm"] in ARMS:
                runs.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = s
    return runs


def _mux(s: dict) -> float:
    if s.get("status") == "ok" and s.get("success") and s.get("evals_to_target") is not None:
        return float(s["evals_to_target"])
    return MUX_BUDGET


def _sr(s: dict) -> float:
    if s.get("status") == "ok" and s.get("test_nmse") is not None:
        return float(s["test_nmse"])
    return math.inf


def _contrast(problem, name, a, b, arms, metric) -> dict:
    ra, rb = arms.get(a, {}), arms.get(b, {})
    keys = sorted(set(ra) & set(rb))
    xa = [metric(ra[k]) for k in keys]
    xb = [metric(rb[k]) for k in keys]
    d = [x - y if math.isfinite(x - y) else (1e12 if x > y else -1e12) for x, y in zip(xa, xb)]
    return {
        "problem": problem,
        "contrast": name,
        "a": a,
        "b": b,
        "pairs": len(keys),
        "a_lower": sum(x < 0 for x in d),
        "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "median_a": statistics.median(xa) if xa else float("nan"),
        "median_b": statistics.median(xb) if xb else float("nan"),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


def _decide(rows: list[dict], stage: int, can_extend: bool) -> None:
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        ma, mb = r["median_a"], r["median_b"]
        if ph < ALPHA and ma <= 0.9 * mb:
            r["verdict"] = "A better"
        elif ph < ALPHA and mb <= 0.9 * ma:
            r["verdict"] = "A worse"
        elif can_extend and stage == 1 and ph < EXTEND_BELOW:
            r["verdict"] = "extend"
        else:
            r["verdict"] = "tie"


def main() -> int:
    if len(sys.argv) != 6 or sys.argv[1] != "--stage" or sys.argv[2] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage, out = int(sys.argv[2]), Path(sys.argv[3])
    families = []
    if sys.argv[4] != "-":
        families.append((_load(sys.argv[4]), _mux))
    if sys.argv[5] != "-":
        families.append((_load(sys.argv[5]), _sr))
    primary, secondary, descriptive, agg = [], [], [], []
    for runs, metric in families:
        for problem in sorted(runs):
            arms = runs[problem]
            primary += [_contrast(problem, n, a, b, arms, metric) for n, a, b in PRIMARY]
            secondary += [_contrast(problem, n, a, b, arms, metric) for n, a, b in SECONDARY]
            descriptive += [_contrast(problem, n, a, b, arms, metric) for n, a, b in DESCRIPTIVE]
            for arm in ARMS:
                v = list(arms.get(arm, {}).values())
                agg.append(
                    {
                        "problem": problem,
                        "arm": arm,
                        "runs": len(v),
                        "not_ok": sum(s.get("status") != "ok" for s in v),
                        "median_metric": statistics.median(metric(s) for s in v) if v else None,
                        "solved": sum(metric(s) < MUX_BUDGET for s in v) if metric is _mux else None,
                        "median_cpu_s": statistics.median(s.get("cpu_s") or 0 for s in v) if v else None,
                    }
                )
    _decide(primary, stage, can_extend=True)
    _decide(secondary, stage, can_extend=False)
    for r in descriptive:
        r["p_holm"], r["verdict"] = None, "descriptive"
    rows = primary + secondary + descriptive
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("problem", "contrast"), k))
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out / "aggregate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    (out / "verdict.json").write_text(
        json.dumps({f"{r['problem']}:{r['contrast']}": r["verdict"] for r in primary + secondary},
                   indent=2) + "\n"
    )
    for r in agg + rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
