"""Phase 1n report: GOMEA vs hill climbing with the SAME per-deme cap, at equal evaluations.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1j_report.py --stage {1,2} OUT_DIR MUX11_DIRS CON_DIRS

MUX11_DIRS and CON_DIRS are comma-separated result directories (stage 2 passes the stage-1
and extension directories together), or `-` to leave that problem out.

Arms HC-C250 (`-a hc-capped`, 250 evaluations per deme) and LTB-C250-M (GOMEA, LT-B, 250
per deme, memo), at equal evaluations (mux11 `-m 40000`, Con `-m 10000`). Pairing is by
seed (mux11) or split (Con).

* **mux11:** metric is evaluations to a solution (`evals_to_target`); an unsolved or failed
  run counts as 40,000.
* **Con:** metric is test NMSE of the validation-selected program; a failed run counts as
  infinite.

Per problem: paired Wilcoxon signed-rank test of LTB-C250-M vs HC, Holm-corrected over the
problems given. Two looks, each at ALPHA = 0.025 (Bonferroni over the looks, 0.05 overall):

* stage 1 (10 pairs): Holm p < ALPHA and a median at least 10 % lower than HC's is "better"
  (the mirror image is "worse"); ALPHA <= Holm p < EXTEND_BELOW is "extend" (run 10 more
  pairs); anything else is a "tie";
* stage 2 (all 20 pairs, extended problems only): the same rule without "extend".

See PHASE1N-PREREGISTRATION.md.
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

T_MUX = 40000.0
ALPHA = 0.025
EXTEND_BELOW = 0.25
ARMS = ("HC-C250", "LTB-C250-M")


def _load(dirs: str, task: str) -> dict:
    runs: dict = {a: {} for a in ARMS}
    files = [f for d in dirs.split(",") for f in glob.glob(str(Path(d) / "runs/*/summary.json"))]
    for f in files:
        s = json.loads(Path(f).read_text())
        if s["arm"] in runs and s["task"] == task:
            runs[s["arm"]][s["seed"]] = s
    return runs


def _mux_metric(s: dict) -> float:
    if s.get("status") == "ok" and s.get("success") and s.get("evals_to_target") is not None:
        return float(s["evals_to_target"])
    return T_MUX


def _con_metric(s: dict) -> float:
    if s.get("status") == "ok" and s.get("test_nmse") is not None:
        return float(s["test_nmse"])
    return math.inf


def _med(xs: list) -> float:
    return statistics.median(xs) if xs else float("nan")


def _contrast(problem: str, runs: dict, metric, extra) -> dict:
    keys = sorted(set(runs["HC-C250"]) & set(runs["LTB-C250-M"]))
    a = [metric(runs["LTB-C250-M"][k]) for k in keys]
    b = [metric(runs["HC-C250"][k]) for k in keys]
    d = [x - y if math.isfinite(x - y) else (1e12 if x > y else -1e12) for x, y in zip(a, b)]
    row = {
        "problem": problem,
        "pairs": len(keys),
        "ltb_lower": sum(x < 0 for x in d),
        "ties": sum(x == 0 for x in d),
        "hc_lower": sum(x > 0 for x in d),
        "median_ltb": _med(a),
        "median_hc": _med(b),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }
    for arm, tag in (("LTB-C250-M", "ltb"), ("HC-C250", "hc")):
        v = [runs[arm][k] for k in keys]
        row[f"not_ok_{tag}"] = sum(s.get("status") != "ok" for s in v)
        row[f"median_cpu_{tag}"] = _med([s.get("cpu_s") or 0 for s in v])
        row[f"median_evals_{tag}"] = _med([extra(s) for s in v])
    return row


def main() -> int:
    if sys.argv[1] != "--stage" or sys.argv[2] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage = int(sys.argv[2])
    out, mux_dirs, con_dirs = Path(sys.argv[3]), sys.argv[4], sys.argv[5]
    rows = []
    if mux_dirs != "-":
        mux = _load(mux_dirs, "mux11")
        row = _contrast("mux11", mux, _mux_metric, lambda s: s.get("n_calls") or 0)
        row["solved_ltb"] = sum(_mux_metric(s) < T_MUX for s in mux["LTB-C250-M"].values())
        row["solved_hc"] = sum(_mux_metric(s) < T_MUX for s in mux["HC-C250"].values())
        rows.append(row)
    if con_dirs != "-":
        con = _load(con_dirs, "Con")
        rows.append(_contrast("Con", con, _con_metric, lambda s: s.get("n_evals") or 0))
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["stage"] = stage
        r["p_holm"] = ph
        lo, hi = r["median_ltb"], r["median_hc"]
        if ph < ALPHA and lo <= 0.9 * hi:
            r["verdict"] = "better"
        elif ph < ALPHA and hi <= 0.9 * lo:
            r["verdict"] = "worse"
        elif stage == 1 and ALPHA <= ph < EXTEND_BELOW:
            r["verdict"] = "extend"
        else:
            r["verdict"] = "tie"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "problem", k))
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "verdict.json").write_text(
        json.dumps({r["problem"]: r["verdict"] for r in rows}, indent=2) + "\n"
    )
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
