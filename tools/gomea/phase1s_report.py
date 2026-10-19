"""Phase 1s report: penalised GOMEA (PEN) vs plain GOMEA on mux11, CPU to a solution.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1s_report.py OUT_DIR FRESH_DIRS POOL_DIRS

FRESH_DIRS: comma-separated Phase 1s result directories (seeds 3021-3030). POOL_DIRS:
comma-separated directories from which the same two arms are read as the pool (Phase 1o's
mux11 runs, seeds 3001-3010). All arms use a 250-evaluation cap per deme and `-m 40000`.

Arms: `LTB-C250-M` (current GOMEA), `PEN` (+ --gomea-accept-complexity-ratio 0.125).

Metric, paired by seed: CPU seconds to a solution (`cpu_to_target`); a run that is not ok,
or did not solve, counts as infinite.

* primary: `pooled`, PEN vs LTB-C250-M over fresh and pool seeds (20 pairs);
* replication check: `fresh`, the same contrast over the fresh seeds only;
* descriptive: `pool-only`, over the pool seeds only.

Rule (primary only, single look, ALPHA = 0.025, no Holm): p < ALPHA and a median at least
10 % lower is "A better" (mirror: "A worse"); otherwise "tie". `replicates` says whether
the fresh contrast leans the same way as the pooled verdict (more seeds where PEN is lower
for "A better", more where it is higher for "A worse"); null for a tie.
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
from benchmarks.gomea.stats import wilcoxon_signed_rank  # noqa: E402

ALPHA = 0.025
ARMS = ("LTB-C250-M", "PEN")
TASK = "mux11"


def _load(dirs: str, into: dict) -> None:
    for d in dirs.split(","):
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            if s["arm"] in ARMS and s["task"] == TASK:
                into.setdefault(s["arm"], {})[s["seed"]] = s


def _cpu_to_target(s: dict) -> float:
    if s.get("status") == "ok" and s.get("success") and s.get("cpu_to_target") is not None:
        return float(s["cpu_to_target"])
    return math.inf


def _med(v: list, key) -> float | None:
    x = [key(s) for s in v]
    x = [y for y in x if y is not None]
    return statistics.median(x) if x else None


def _per_eval(s: dict) -> float | None:
    if s.get("cpu_s") is None or not s.get("n_calls"):
        return None
    return float(s["cpu_s"]) / float(s["n_calls"])


def _contrast(name, a, b, arms) -> dict:
    ra, rb = arms.get(a, {}), arms.get(b, {})
    keys = sorted(set(ra) & set(rb))
    xa = [_cpu_to_target(ra[k]) for k in keys]
    xb = [_cpu_to_target(rb[k]) for k in keys]
    # As phase1q_report._contrast, except that two unsolved runs (inf - inf) are a tie
    # rather than "A lower": with an uncapped metric both sides can be infinite.
    d = [
        x - y if math.isfinite(x - y) else 0.0 if x == y else (1e12 if x > y else -1e12)
        for x, y in zip(xa, xb, strict=True)
    ]
    return {
        "problem": TASK, "contrast": name, "a": a, "b": b, "pairs": len(keys),
        "a_lower": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "median_a": statistics.median(xa) if xa else float("nan"),
        "median_b": statistics.median(xb) if xb else float("nan"),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


def _decide(r: dict) -> None:
    ma, mb = r["median_a"], r["median_b"]
    if r["p"] < ALPHA and ma <= 0.9 * mb:
        r["verdict"] = "A better"
    elif r["p"] < ALPHA and mb <= 0.9 * ma:
        r["verdict"] = "A worse"
    else:
        r["verdict"] = "tie"


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    fresh: dict = {}
    pool: dict = {}
    _load(sys.argv[2], fresh)
    _load(sys.argv[3], pool)
    pooled = {arm: {**pool.get(arm, {}), **fresh.get(arm, {})} for arm in ARMS}
    if set(fresh.get(ARMS[0], {})) & set(pool.get(ARMS[0], {})):
        print("warning: fresh and pool share seeds; fresh runs win", file=sys.stderr)
    rows = [
        _contrast("pooled", "PEN", "LTB-C250-M", pooled),
        _contrast("fresh", "PEN", "LTB-C250-M", fresh),
        _contrast("pool-only", "PEN", "LTB-C250-M", pool),
    ]
    _decide(rows[0])
    prim, rep = rows[0], rows[1]
    if prim["verdict"] == "A better":
        prim["replicates"] = rep["a_lower"] > rep["b_lower"]
    elif prim["verdict"] == "A worse":
        prim["replicates"] = rep["b_lower"] > rep["a_lower"]
    else:
        prim["replicates"] = None
    agg = []
    for source, runs in (("fresh", fresh), ("pool", pool)):
        for arm in ARMS:
            v = list(runs.get(arm, {}).values())
            agg.append({
                "source": source, "arm": arm, "runs": len(v),
                "not_ok": sum(s.get("status") != "ok" for s in v),
                "solved": sum(bool(s.get("success")) for s in v),
                "median_cpu_to_target": _med(v, _cpu_to_target),
                "median_evals_to_target": _med(
                    v, lambda s: s.get("evals_to_target") if s.get("success") else math.inf
                ),
                "median_cpu_s": _med(v, lambda s: s.get("cpu_s")),
                "median_cpu_per_eval": _med(v, _per_eval),
                "median_deme_count": _med(v, lambda s: s.get("deme_count")),
                "median_best_complexity": _med(v, lambda s: s.get("best_cpx")),
            })
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {k for r in rows for k in r}, key=lambda k: (k not in ("problem", "contrast"), k)
    )
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out / "aggregate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    (out / "verdict.json").write_text(
        json.dumps({f"{TASK}:pooled": prim["verdict"], "replicates": prim["replicates"]},
                   indent=2) + "\n"
    )
    for r in agg + rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
