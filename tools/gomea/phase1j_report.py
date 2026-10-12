"""Phase 1j report: does tuned GOMEA still beat hill climbing at equal wall-clock time?

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1j_report.py OUT_DIR MUX11_RESULT_DIR CON_RESULT_DIR

Arms HC and LTB-C250-M, run with `--max-time T -m 1000000000` (time only, as classic
MOSES is run). Pairing is by seed (mux11) or split (Con).

* **mux11:** metric is wall-clock time to a solution (`time_to_target`); an unsolved or
  failed run counts as T = 3,600 s.
* **Con:** metric is test NMSE of the validation-selected program; a failed run counts as
  infinite.

Per problem: paired Wilcoxon signed-rank test of LTB-C250-M vs HC, Holm-corrected over the
two problems. "Better" means Holm p < 0.05 and a median at least 10 % lower than HC's;
"worse" is the mirror image; anything else is a tie. See PHASE1J-PREREGISTRATION.md.
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

T_MUX = 3600.0
ARMS = ("HC", "LTB-C250-M")


def _load(d: str) -> dict:
    runs: dict = {a: {} for a in ARMS}
    for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
        s = json.loads(Path(f).read_text())
        if s["arm"] in runs:
            runs[s["arm"]][s["seed"]] = s
    return runs


def _mux_metric(s: dict) -> float:
    if s.get("status") == "ok" and s.get("success") and s.get("time_to_target") is not None:
        return float(s["time_to_target"])
    return T_MUX


def _con_metric(s: dict) -> float:
    if s.get("status") == "ok" and s.get("test_nmse") is not None:
        return float(s["test_nmse"])
    return math.inf


def _med(xs: list) -> float:
    return statistics.median(xs) if xs else float("nan")


def _contrast(problem: str, runs: dict, metric, extra) -> dict:
    keys = sorted(set(runs["HC"]) & set(runs["LTB-C250-M"]))
    a = [metric(runs["LTB-C250-M"][k]) for k in keys]
    b = [metric(runs["HC"][k]) for k in keys]
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
    for arm, tag in (("LTB-C250-M", "ltb"), ("HC", "hc")):
        v = [runs[arm][k] for k in keys]
        row[f"not_ok_{tag}"] = sum(s.get("status") != "ok" for s in v)
        row[f"median_cpu_{tag}"] = _med([s.get("cpu_s") or 0 for s in v])
        row[f"median_evals_{tag}"] = _med([extra(s) for s in v])
    return row


def main() -> int:
    out = Path(sys.argv[1])
    mux, con = _load(sys.argv[2]), _load(sys.argv[3])
    rows = [
        _contrast("mux11", mux, _mux_metric, lambda s: s.get("n_calls") or 0),
        _contrast("Con", con, _con_metric, lambda s: s.get("n_evals") or 0),
    ]
    rows[0]["solved_ltb"] = sum(_mux_metric(s) < T_MUX for s in mux["LTB-C250-M"].values())
    rows[0]["solved_hc"] = sum(_mux_metric(s) < T_MUX for s in mux["HC"].values())
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        lo, hi = r["median_ltb"], r["median_hc"]
        if ph < 0.05 and lo <= 0.9 * hi:
            r["verdict"] = "better"
        elif ph < 0.05 and hi <= 0.9 * lo:
            r["verdict"] = "worse"
        else:
            r["verdict"] = "tie"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "problem", k))
    with (out / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
