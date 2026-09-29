"""Phase 1f report (PHASE1F-PREREGISTRATION.md).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1f_report.py A RESULT_DIR PUBLISHED_CSV OUT_DIR
    .venv/bin/python tools/gomea/phase1f_report.py B GPGOMEA_DIR MOSES_DIR PUBLISHED_CSV OUT_DIR
    .venv/bin/python tools/gomea/phase1f_report.py CD RESULT_DIR PART_A_DIR OUT_DIR

* **A** applies the part A criteria to the C++ Table 3 replication (levels and pattern).
* **B** compares GP-GOMEA with classic C++ MOSES on test NMSE: a Wilcoxon test per
  dataset, Holm correction across datasets, and a 10 % practical threshold. It also sets
  the GP-GOMEA medians beside the paper's Table 8 IMS g = 4 medians.
* **CD** applies the part C criteria (LTB vs LTMI and vs RT inside pyMoses) and the part D
  verdicts (LTB vs HC).

PUBLISHED_CSV has the columns `table,dataset,config,metric,value` (the values of the
paper's Tables 3 and 8). A run that did not end `ok`, or has a non-finite NMSE, counts as
the worst value in its pair; the report lists every such run.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]


def _load(d: Path) -> dict:
    runs: dict = defaultdict(dict)
    for f in sorted(glob.glob(str(d / "runs/*/summary.json"))):
        s = json.loads(Path(f).read_text())
        runs[(s["task"], s["arm"])][s["seed"]] = s
    return runs


def _val(s: dict, key: str) -> float:
    v = s.get(key)
    if s.get("status") != "ok" or v is None or not math.isfinite(v):
        return math.inf
    return v


def _bad(runs: dict) -> list[str]:
    return sorted(
        s["run_id"]
        for by_seed in runs.values()
        for s in by_seed.values()
        if s.get("status") != "ok" or not math.isfinite(s.get("test_nmse") or math.inf)
    )


def _med(xs):
    return statistics.median(xs) if xs else None


def _pair(a: dict, b: dict, key: str) -> dict:
    seeds = sorted(set(a) & set(b))
    xa = [_val(a[s], key) for s in seeds]
    xb = [_val(b[s], key) for s in seeds]
    d = []
    for x, y in zip(xa, xb, strict=True):
        if math.isinf(x) and math.isinf(y):
            d.append(0.0)
        elif math.isinf(x):
            d.append(1e300)
        elif math.isinf(y):
            d.append(-1e300)
        else:
            d.append(x - y)
    p = wilcoxon_signed_rank(d)[1] if any(d) else 1.0
    return {
        "n": len(seeds),
        "median_a": _med(xa),
        "median_b": _med(xb),
        "a_lower": sum(x < 0 for x in d),
        "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "p": p,
    }


def _write(out: Path, name: str, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _published(path: Path) -> dict:
    pub = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                value = float(r["value"])
            except ValueError:  # the B/W count rows, e.g. "0/6"
                continue
            pub[(r["table"], r["dataset"], r["config"], r["metric"])] = value
    return pub


def _aggregate(runs: dict) -> list[dict]:
    rows = []
    for (task, arm), by_seed in sorted(runs.items()):
        v = list(by_seed.values())
        rows.append(
            {
                "dataset": task,
                "arm": arm,
                "n": len(v),
                "n_not_ok": sum(s.get("status") != "ok" for s in v),
                "median_train_nmse": _med([_val(s, "train_nmse") for s in v]),
                "median_test_nmse": _med([_val(s, "test_nmse") for s in v]),
                "median_cpu_s": _med([s.get("cpu_s") or 0 for s in v]),
                "median_wall_s": _med([s.get("wall_s") or 0 for s in v]),
                "median_evals": _med(
                    [e for s in v if (e := s.get("n_evals") or s.get("evaluations"))]
                ),
                "max_rss_mb": max(((s.get("ru_maxrss_kb") or 0) for s in v), default=0) / 1024,
            }
        )
    return rows


def part_a(d: Path, published: Path, out: Path) -> None:
    runs = _load(d)
    pub = _published(published)
    names = {"LT-MIb": "LT-MIb~", "LT-MI": "LT-MI", "RT": "RT"}
    level_rows, pattern_rows = [], []
    for arm, pname in names.items():
        for metric in ("train", "test"):
            for ds in DATASETS:
                ours = _med([_val(s, f"{metric}_nmse") for s in runs[(ds, arm)].values()])
                paper = pub[("Table3", ds, pname, f"{metric}_nmse_median")]
                ratio = ours / paper if ours is not None else math.nan
                level_rows.append(
                    {
                        "arm": arm,
                        "metric": metric,
                        "dataset": ds,
                        "ours_median": ours,
                        "paper_median": paper,
                        "ratio": ratio,
                        "within_band": 0.8 <= ratio <= 1.25,
                    }
                )
    for other in ("LT-MI", "RT"):
        for metric in ("train", "test"):
            for ds in DATASETS:
                r = _pair(runs[(ds, "LT-MIb")], runs[(ds, other)], f"{metric}_nmse")
                sig = r["p"] < 0.05 / 2
                better = sig and r["median_a"] < r["median_b"]
                worse = sig and r["median_a"] > r["median_b"]
                pattern_rows.append(
                    {
                        "contrast": f"LT-MIb:{other}",
                        "metric": metric,
                        "dataset": ds,
                        **r,
                        "ltmib_better": better,
                        "ltmib_worse": worse,
                    }
                )
    _write(out, "partA_levels.csv", level_rows)
    _write(out, "partA_pattern.csv", pattern_rows)
    _write(out, "partA_aggregate.csv", _aggregate(runs))
    paper_counts = {
        ("LT-MI", "train"): 8,
        ("LT-MI", "test"): 7,
        ("RT", "train"): 10,
        ("RT", "test"): 9,
    }
    verdict = {"levels": {}, "pattern": {}, "not_ok_runs": _bad(runs)}
    for arm in names:
        for metric in ("train", "test"):
            k = sum(
                r["within_band"] for r in level_rows if r["arm"] == arm and r["metric"] == metric
            )
            verdict["levels"][f"{arm}/{metric}"] = {"within_band": k, "replicates": k >= 8}
    for (other, metric), paper_k in paper_counts.items():
        rows = [
            r for r in pattern_rows if r["contrast"] == f"LT-MIb:{other}" and r["metric"] == metric
        ]
        k = sum(r["ltmib_better"] for r in rows)
        w = sum(r["ltmib_worse"] for r in rows)
        verdict["pattern"][f"LT-MIb:{other}/{metric}"] = {
            "ours_better": k,
            "ours_worse": w,
            "paper_better": paper_k,
            "replicates": abs(k - paper_k) <= 2 and w == 0,
        }
    verdict["replicated"] = all(v["replicates"] for v in verdict["levels"].values()) and all(
        v["replicates"] for v in verdict["pattern"].values()
    )
    (out / "partA_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


def _verdicts(runs: dict, a_arm: str, b_arm: str, threshold: float = 0.10) -> list[dict]:
    rows = []
    for ds in DATASETS:
        rows.append(
            {
                "dataset": ds,
                "contrast": f"{a_arm}:{b_arm}",
                **_pair(runs[(ds, a_arm)], runs[(ds, b_arm)], "test_nmse"),
            }
        )
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        ma, mb = r["median_a"], r["median_b"]
        if ph < 0.05 and ma <= (1 - threshold) * mb:
            r["verdict"] = "better"
        elif ph < 0.05 and mb <= (1 - threshold) * ma:
            r["verdict"] = "worse"
        else:
            r["verdict"] = "tie"
    return rows


def part_b(gdir: Path, mdir: Path, published: Path, out: Path) -> None:
    runs = _load(gdir)
    runs.update(_load(mdir))
    pub = _published(published)
    rows = _verdicts(runs, "GPG-IMS4", "MOSES-CPP")
    _write(out, "partB_verdicts.csv", rows)
    _write(out, "partB_aggregate.csv", _aggregate(runs))
    t8 = []
    for ds in DATASETS:
        ours = _med([_val(s, "test_nmse") for s in runs[(ds, "GPG-IMS4")].values()])
        paper = pub.get(("Table8", ds, "GP-GOMEA|IMSg4", "test_nmse_median"))
        t8.append(
            {
                "dataset": ds,
                "ours_test_median": ours,
                "paper_test_median": paper,
                "ratio": ours / paper if paper else None,
            }
        )
    _write(out, "partB_table8.csv", t8)
    k_b = sum(r["verdict"] == "better" for r in rows)
    k_w = sum(r["verdict"] == "worse" for r in rows)
    verdict = {
        "better": k_b,
        "worse": k_w,
        "tie": len(rows) - k_b - k_w,
        "gpgomea_beats_moses": k_b > len(rows) / 2 and k_w == 0,
        "not_ok_runs": _bad(runs),
    }
    (out / "partB_verdict.json").write_text(json.dumps(verdict, indent=2))
    for r in rows:
        print(
            f"{r['dataset']} {r['median_a']:.3g} vs {r['median_b']:.3g} "
            f"p_holm={r['p_holm']:.3g} {r['verdict']}"
        )
    print(json.dumps(verdict, indent=2))


def part_cd(d: Path, a_dir: Path, out: Path) -> None:
    runs = _load(d)
    runs_a = _load(a_dir)
    c_rows = []
    for other in ("LTMI", "RT"):
        for metric in ("train", "test"):
            for ds in DATASETS:
                r = _pair(runs[(ds, "LTB")], runs[(ds, other)], f"{metric}_nmse")
                sig = r["p"] < 0.05 / 2
                c_rows.append(
                    {
                        "contrast": f"LTB:{other}",
                        "metric": metric,
                        "dataset": ds,
                        **r,
                        "ltb_better": sig and r["median_a"] < r["median_b"],
                        "ltb_worse": sig and r["median_a"] > r["median_b"],
                        "ltb_median_lower": r["median_a"] < r["median_b"],
                    }
                )
    _write(out, "partC_pattern.csv", c_rows)
    worse_any = [f"{r['contrast']}/{r['metric']}/{r['dataset']}" for r in c_rows if r["ltb_worse"]]
    lower = {
        other: sum(
            r["ltb_median_lower"]
            for r in c_rows
            if r["contrast"] == f"LTB:{other}" and r["metric"] == "train"
        )
        for other in ("LTMI", "RT")
    }
    c_verdict = {
        "ltb_significantly_worse": worse_any,
        "train_median_lower_count": lower,
        "consistent": not worse_any and all(k >= 6 for k in lower.values()),
    }
    levels = []
    for ds in DATASETS:
        row = {"dataset": ds}
        for arm in ("LTB", "LTMI", "RT", "HC"):
            row[f"py_{arm}_test"] = _med([_val(s, "test_nmse") for s in runs[(ds, arm)].values()])
        for arm in ("LT-MIb", "LT-MI", "RT"):
            row[f"cpp_{arm}_test"] = _med(
                [_val(s, "test_nmse") for s in runs_a[(ds, arm)].values()]
            )
        levels.append(row)
    _write(out, "partC_levels.csv", levels)
    d_rows = _verdicts(runs, "LTB", "HC")
    for other in ("LTMI", "RT"):
        d_rows += _verdicts(runs, other, "HC")
    _write(out, "partD_verdicts.csv", d_rows)
    _write(out, "partCD_aggregate.csv", _aggregate(runs))
    main_rows = [r for r in d_rows if r["contrast"] == "LTB:HC"]
    k_b = sum(r["verdict"] == "better" for r in main_rows)
    k_w = sum(r["verdict"] == "worse" for r in main_rows)
    d_verdict = {
        "better": k_b,
        "worse": k_w,
        "tie": len(main_rows) - k_b - k_w,
        "gomea_beats_hc": k_b > len(main_rows) / 2 and k_w == 0,
    }
    verdict = {"C": c_verdict, "D": d_verdict, "not_ok_runs": _bad(runs)}
    (out / "partCD_verdict.json").write_text(json.dumps(verdict, indent=2))
    for r in d_rows:
        print(
            f"{r['contrast']:8s} {r['dataset']} {r['median_a']:.3g} vs {r['median_b']:.3g} "
            f"p_holm={r['p_holm']:.3g} {r['verdict']}"
        )
    print(json.dumps(verdict, indent=2))


def main() -> int:
    mode, args = sys.argv[1], [Path(a) for a in sys.argv[2:]]
    if mode == "A":
        part_a(*args)
    elif mode == "B":
        part_b(*args)
    elif mode == "CD":
        part_cd(*args)
    else:
        raise SystemExit(f"unknown mode {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
