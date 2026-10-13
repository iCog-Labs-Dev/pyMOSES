"""Phase 1k report: with fewer pair knobs, does learned linkage gain on random linkage?

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1k_report.py --stage {1,2} OUT_DIR RESULT_DIR [...]

Frozen maj5. Arms (by name): `LT-B-N<n>` and `RT-N<n>` use the default knob building;
`LT-B-RK-N<n>` and `RT-RK-N<n>` use `--logical-perm-ratio -0.75` (one pair knob per
and/or node instead of five, about 40 % fewer knobs). N is 64 or 256. Runs pair by
(fixture, seed). The metric is final error, the wrong rows of 32 (`-best_raw`; a run that
is not `ok` counts as 32).

Primary contrast, per N, Holm-corrected over the two N: `interaction` = (LT-B - RT) with
reduced knobs minus (LT-B - RT) with default knobs, per pair. Negative means fewer knobs
widen learned linkage's lead. Two looks at ALPHA = 0.025 each (0.05 overall): Holm p < ALPHA
decides ("reduced helps" / "reduced hurts"); at stage 1, ALPHA <= Holm p < EXTEND_BELOW is
"extend"; anything else is "no difference".

Secondary (reported with its own p, not part of the decision): the share of evaluated GOM
trials that improved the individual (`accepted_improving / evaluated`, summed over a run's
generations), LT-B minus RT, for each knob set and N. See PHASE1K-PREREGISTRATION.md.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

ALPHA = 0.025
EXTEND_BELOW = 0.25
NS = (64, 256)


def _err(s: dict) -> float:
    if s.get("status") != "ok" or s.get("best_raw") is None:
        return 32.0
    return -s["best_raw"]


def _improve_rate(run_dir: Path) -> float | None:
    imp = ev = 0
    path = run_dir / "events.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if "fos_size" in e and "stats" in e:
            imp += e["stats"].get("accepted_improving", 0)
            ev += e["stats"].get("evaluated", 0)
    return imp / ev if ev else None


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
    runs: dict = {}
    for d in sys.argv[4:]:
        for f in glob.glob(str(Path(d) / "runs/*/summary.json")):
            s = json.loads(Path(f).read_text())
            s["_dir"] = str(Path(f).parent)
            runs.setdefault(s["arm"], {})[(s["task"], s["seed"])] = s
    primary, secondary, agg = [], [], []
    for n in NS:
        arms = (f"LT-B-RK-N{n}", f"RT-RK-N{n}", f"LT-B-N{n}", f"RT-N{n}")
        keys = sorted(set.intersection(*(set(runs.get(a, {})) for a in arms)))
        e = {a: [_err(runs[a][k]) for k in keys] for a in arms}
        lbr, rtr, lb, rt = (e[a] for a in arms)
        primary.append(
            _row(f"interaction N{n}", [(a - b) - (c - d) for a, b, c, d in zip(lbr, rtr, lb, rt)])
        )
        secondary.append(_row(f"LT-B - RT reduced N{n}", [a - b for a, b in zip(lbr, rtr)]))
        secondary.append(_row(f"LT-B - RT default N{n}", [a - b for a, b in zip(lb, rt)]))
        for tag, (la, ra) in (("reduced", arms[:2]), ("default", arms[2:])):
            rates = [
                (_improve_rate(Path(runs[la][k]["_dir"])), _improve_rate(Path(runs[ra][k]["_dir"])))
                for k in keys
            ]
            d = [a - b for a, b in rates if a is not None and b is not None]
            secondary.append(_row(f"improve-rate LT-B - RT {tag} N{n}", d))
        for a in arms:
            v = [runs[a][k] for k in keys]
            agg.append(
                {
                    "arm": a,
                    "runs": len(v),
                    "not_ok": sum(s.get("status") != "ok" for s in v),
                    "mean_err": sum(e[a]) / len(e[a]) if v else float("nan"),
                    "solved": sum(x == 0 for x in e[a]),
                }
            )
    for r, ph in zip(primary, holm([r["p"] for r in primary]), strict=True):
        r["p_holm"] = ph
        if ph < ALPHA:
            r["verdict"] = "reduced helps" if r["mean"] < 0 else "reduced hurts"
        elif stage == 1 and ph < EXTEND_BELOW:
            r["verdict"] = "extend"
        else:
            r["verdict"] = "no difference"
    out.mkdir(parents=True, exist_ok=True)
    rows = primary + secondary
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
