"""Phase 1r report: seeding MOSES's metapopulation with GP-GOMEA elites (backlog item 9).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1r_report.py --stage {1,2} OUT_DIR RESULT_DIRS

RESULT_DIRS: comma-separated Phase 1r result directories (`moses_sr_runner.py` output).
All arms: 10,000 evaluations per run in total, a 250-evaluation cap per deme. Seeded arms
(`-S`) spend 20 % of that on GP-GOMEA (height 3, functions + - *, top 5 distinct elites) and
the rest on MOSES started from the seeds plus the empty program.

Arms: `HC-C250`, `HC-C250-S`, `LTB-C250-M`, `LTB-C250-M-S`. Metric: test NMSE, paired by
split (a failed run = infinite).

* primary (Holm over 4): per problem, `seed-hc` (HC-C250-S vs HC-C250) and `seed-gomea`
  (LTB-C250-M-S vs LTB-C250-M);
* secondary (Holm over 2): per problem, `seeded-gomea-vs-hc` (LTB-C250-M-S vs HC-C250-S).

Descriptive: the best seed's own test NMSE, the share of runs whose reported program
descends from a seed, final program size, CPU per run.

Rule (each look at ALPHA = 0.025): Holm p < ALPHA and a median at least 10 % lower is
"A better" (mirror: "A worse"); primary contrasts at stage 1 with ALPHA <= Holm p <
EXTEND_BELOW are "extend"; otherwise "tie". At stage 2 the Holm family stays complete and
only contrasts that were "extend" at stage 1 take a stage-2 verdict (see
PHASE1R-PREREGISTRATION.md).
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
ARMS = ("HC-C250", "HC-C250-S", "LTB-C250-M", "LTB-C250-M-S")


def _load(dirs: str, into: dict) -> None:
    for d in dirs.split(","):
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            if s["arm"] in ARMS:
                into.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = s


def _acc(s: dict) -> float:
    if s.get("status") == "ok" and s.get("test_nmse") is not None:
        return float(s["test_nmse"])
    return math.inf


def _size(s: dict) -> int | None:
    p = s.get("final_program")
    return None if p is None else p.count("(") + len(p.replace("(", " ").replace(")", " ").split())


def _contrast(problem, name, a, b, arms) -> dict:
    ra, rb = arms.get(a, {}), arms.get(b, {})
    keys = sorted(set(ra) & set(rb))
    xa = [_acc(ra[k]) for k in keys]
    xb = [_acc(rb[k]) for k in keys]
    d = [x - y if math.isfinite(x - y) else (0.0 if x == y else (1e12 if x > y else -1e12))
         for x, y in zip(xa, xb)]
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


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[1] != "--stage" or sys.argv[2] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage, out = int(sys.argv[2]), Path(sys.argv[3])
    runs: dict = {}
    _load(sys.argv[4], runs)
    primary, secondary, agg = [], [], []
    for problem in sorted(runs):
        arms = runs[problem]
        primary += [
            _contrast(problem, "seed-hc", "HC-C250-S", "HC-C250", arms),
            _contrast(problem, "seed-gomea", "LTB-C250-M-S", "LTB-C250-M", arms),
        ]
        secondary += [_contrast(problem, "seeded-gomea-vs-hc", "LTB-C250-M-S", "HC-C250-S", arms)]
        for arm in ARMS:
            v = list(arms.get(arm, {}).values())
            seeded = arm.endswith("-S")
            agg.append({
                "problem": problem, "arm": arm, "runs": len(v),
                "not_ok": sum(s.get("status") != "ok" for s in v),
                "infinite_test": sum(not math.isfinite(_acc(s)) for s in v),
                "median_test_nmse": _med(_acc(s) for s in v),
                "median_train_nmse": _med(s.get("train_nmse") for s in v),
                "median_cpu_s": _med(s.get("cpu_s") for s in v),
                "median_final_size": _med(_size(s) for s in v),
                "median_best_seed_test_nmse": _med(s.get("seed_test_nmse") for s in v) if seeded else None,
                "final_from_seed": sum(str(s.get("final_lineage", "")).startswith("seed") for s in v)
                if seeded else None,
                "median_seed_evals": _med(s.get("seed_evals") for s in v) if seeded else None,
            })
    if not agg:
        print("no Phase 1r runs found", file=sys.stderr)
        return 1
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
