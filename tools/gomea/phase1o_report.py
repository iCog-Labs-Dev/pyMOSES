"""Phase 1o report: a complexity penalty inside GOMEA's acceptance only (backlog item 13).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1o_report.py --stage {1,2} OUT_DIR MUX11_DIRS CON_DIRS HC_DIRS

MUX11_DIRS / CON_DIRS: comma-separated Phase 1o result directories (or `-`). HC_DIRS:
comma-separated directories from which only the capped hill-climbing arm `HC-C250` is read
(Phase 1n's runs, and Phase 1o's Con extension). All arms use a 250-evaluation cap per deme.

Arms: `LTB-C250-M` (current GOMEA), `PEN` (+ --gomea-accept-complexity-ratio R), `PEN-ACC`
(+ --gomea-deme-store accepted), `HC-C250`.

Metrics, paired by seed (mux11) or split (Con): accuracy = evaluations to a solution on mux11
(unsolved = 40,000) or test NMSE on Con (failed = infinite); cost = CPU seconds per run.

* primary (Holm over 4): per problem, `cpu` and `accuracy` of PEN vs LTB-C250-M;
* secondary (Holm over 6): per problem, `acc-store` and `cpu-store` (PEN-ACC vs PEN) and
  `vs-hc` (PEN accuracy vs HC-C250).

Rule (each look at ALPHA = 0.025): Holm p < ALPHA and a median at least 10 % lower is
"A better" (mirror: "A worse"); primary contrasts at stage 1 with ALPHA <= Holm p <
EXTEND_BELOW are "extend"; otherwise "tie". See PHASE1O-PREREGISTRATION.md.
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
GOMEA_ARMS = ("LTB-C250-M", "PEN", "PEN-ACC")


def _load(dirs: str, arms: tuple[str, ...], into: dict) -> None:
    if dirs == "-":
        return
    for d in dirs.split(","):
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            if s["arm"] in arms:
                into.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = s


def _acc(problem: str, s: dict) -> float:
    if problem == "mux11":
        if s.get("status") == "ok" and s.get("success") and s.get("evals_to_target") is not None:
            return float(s["evals_to_target"])
        return MUX_BUDGET
    if s.get("status") == "ok" and s.get("test_nmse") is not None:
        return float(s["test_nmse"])
    return math.inf


def _cpu(problem: str, s: dict) -> float:
    return float(s.get("cpu_s") or math.inf)


def _contrast(problem, name, a, b, arms, metric) -> dict:
    ra, rb = arms.get(a, {}), arms.get(b, {})
    keys = sorted(set(ra) & set(rb))
    xa = [metric(problem, ra[k]) for k in keys]
    xb = [metric(problem, rb[k]) for k in keys]
    d = [x - y if math.isfinite(x - y) else (1e12 if x > y else -1e12) for x, y in zip(xa, xb)]
    return {
        "problem": problem, "contrast": name, "a": a, "b": b, "pairs": len(keys),
        "a_lower": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "median_a": statistics.median(xa) if xa else float("nan"),
        "median_b": statistics.median(xb) if xb else float("nan"),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


def _decide(rows, stage, can_extend) -> None:
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
    if len(sys.argv) != 7 or sys.argv[1] != "--stage" or sys.argv[2] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage, out = int(sys.argv[2]), Path(sys.argv[3])
    runs: dict = {}
    _load(sys.argv[4], GOMEA_ARMS, runs)
    _load(sys.argv[5], GOMEA_ARMS, runs)
    _load(sys.argv[6], ("HC-C250",), runs)
    primary, secondary, agg = [], [], []
    for problem in sorted(p for p in runs if any(a in runs[p] for a in GOMEA_ARMS)):
        arms = runs[problem]
        primary += [
            _contrast(problem, "cpu", "PEN", "LTB-C250-M", arms, _cpu),
            _contrast(problem, "accuracy", "PEN", "LTB-C250-M", arms, _acc),
        ]
        secondary += [
            _contrast(problem, "acc-store", "PEN-ACC", "PEN", arms, _acc),
            _contrast(problem, "cpu-store", "PEN-ACC", "PEN", arms, _cpu),
            _contrast(problem, "vs-hc", "PEN", "HC-C250", arms, _acc),
        ]
        for arm in (*GOMEA_ARMS, "HC-C250"):
            v = list(arms.get(arm, {}).values())
            agg.append({
                "problem": problem, "arm": arm, "runs": len(v),
                "not_ok": sum(s.get("status") != "ok" for s in v),
                "median_accuracy": statistics.median(_acc(problem, s) for s in v) if v else None,
                "solved": sum(_acc(problem, s) < MUX_BUDGET for s in v) if problem == "mux11" else None,
                "median_cpu_s": statistics.median(_cpu(problem, s) for s in v) if v else None,
                "median_best_complexity": statistics.median(
                    s["best_cpx"] for s in v if s.get("best_cpx") is not None
                ) if any(s.get("best_cpx") is not None for s in v) else None,
            })
    _decide(primary, stage, can_extend=True)
    _decide(secondary, stage, can_extend=False)
    rows = primary + secondary
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
        json.dumps({f"{r['problem']}:{r['contrast']}": r["verdict"] for r in rows}, indent=2) + "\n"
    )
    for r in agg + rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
