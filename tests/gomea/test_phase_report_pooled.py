"""phase_report.py, server round 8: the pooled primary test (r = log(NMSE_A / NMSE_B) over all
(problem, seed) pairs, Holm across the pooled contrasts) and secondary contrasts without
verdicts. The expected p values are computed by hand from the exact Wilcoxon distribution."""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import phase_report  # noqa: E402

SPEC = {
    "phase": "t",
    "metrics": {"P1": "test_nmse", "P2": "test_nmse"},
    "primary": [],
    "pooled_primary": [["f-vs-h", "F", "H", "test_nmse"], ["f-vs-n", "F", "N", "test_nmse"]],
    "secondary": [[p, n, "F", b, "test_nmse"] for p in ("P1", "P2")
                  for n, b in (("f-vs-h", "H"), ("f-vs-n", "N"))],
    "secondary_verdicts": False,
}
FAIL = None


def _write(root: Path, runs: dict) -> None:
    """runs: {(problem, arm, seed): test NMSE or FAIL}."""
    for i, ((p, arm, seed), v) in enumerate(sorted(runs.items())):
        d = root / "runs" / str(i)
        d.mkdir(parents=True)
        s = {"task": p, "arm": arm, "seed": seed, "status": "ok" if v is not FAIL else "error",
             "test_nmse": v, "train_nmse": v}
        (d / "summary.json").write_text(json.dumps(s))


def _report(tmp_path, runs, spec=SPEC):
    res, out = tmp_path / "res", tmp_path / "out"
    _write(res, runs)
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    rc = phase_report.main(["--stage", "1", str(sp), str(out), str(res)])
    rows = list(csv.DictReader((out / "contrasts.csv").open())) if rc == 0 else []
    return rc, {(r["problem"], r["contrast"]): r for r in rows}, out


def _runs():
    """10 pairs: P1 and P2, seeds 1-5. N = 1 everywhere, F = 0.5 * N (r = log 0.5 on all 10).
    H: r = log(F / H) is -3, -2, -1, +0.5 on P1 s1-s4; F failed on P1 s5 (r = +1e12, the largest
    rank against F); both failed on P2 s1 (r = 0); F == H on P2 s2-s5 (r = 0)."""
    runs = {}
    f = 0.5
    rs = {("P1", 1): -3, ("P1", 2): -2, ("P1", 3): -1, ("P1", 4): 0.5}
    for p in ("P1", "P2"):
        for s in range(1, 6):
            runs[(p, "N", s)] = 1.0
            runs[(p, "F", s)] = f
            if (p, s) in rs:
                runs[(p, "H", s)] = f / math.exp(rs[(p, s)])
            else:
                runs[(p, "H", s)] = f
    runs[("P1", "F", 5)] = FAIL
    runs[("P2", "F", 1)] = FAIL
    runs[("P2", "H", 1)] = FAIL
    return runs


def test_pooled_contrast_by_hand(tmp_path):
    """f-vs-h: nonzero r = -3, -2, -1, +0.5, +1e12 with ranks 4, 3, 2, 1, 5: W+ = 6. Of the 32
    sign patterns of ranks 1-5, 13 have W+ <= 6 and 22 have W+ >= 6, so p = 2 * 13 / 32 = 0.8125.
    The median of the 10 values (-3, -2, -1, 0 x 5, 0.5, 1e12) is 0, so the ratio is 1.
    f-vs-n: F failed on P1 s5 and on P2 s1 (N ok): r = +1e12 on those 2 pairs and log 0.5 on
    the other 8. Ranks: 4.5 (8 tied), 9.5 (2 tied); W+ = 2 * 9.5 = 19. Subsets with sum <= 19:
    no 9.5: C(8, 0..4) = 163; one 9.5 (2 ways): C(8, 0..2) = 37, so 74; both: 1. Total 238 of
    1,024, and 787 have sum >= 19, so p = 2 * 238 / 1024 = 0.46484375. The median r is log 0.5.
    Holm across 2: 2 * 0.46484375 = 0.9296875 for both."""
    rc, rows, _ = _report(tmp_path, _runs())
    assert rc == 0
    h = rows[("pooled", "f-vs-h")]
    assert h["family"] == "primary" and h["metric"] == "log_ratio:test_nmse"
    assert int(h["pairs"]) == 10 and int(h["a_lower"]) == 3 and int(h["b_lower"]) == 2
    assert int(h["ties"]) == 5
    assert float(h["p"]) == pytest.approx(26 / 32)
    assert float(h["median_log_ratio"]) == 0.0 and float(h["ratio"]) == 1.0
    n = rows[("pooled", "f-vs-n")]
    assert int(n["pairs"]) == 10 and int(n["a_lower"]) == 8 and int(n["b_lower"]) == 2
    assert float(n["p"]) == pytest.approx(476 / 1024)
    assert float(n["median_log_ratio"]) == pytest.approx(math.log(0.5))
    assert float(n["ratio"]) == pytest.approx(0.5)
    assert float(n["p_holm"]) == pytest.approx(0.9296875)
    assert float(h["p_holm"]) == pytest.approx(0.9296875)
    assert n["verdict"] == "tie" and h["verdict"] == "tie"


def test_pooled_verdicts_better_worse_tie(tmp_path):
    """10 pairs all with r = log 0.5: p = 2 / 1024 per contrast, Holm 2 * 0.00195 = 0.0039 < 0.025.
    F vs N: ratio 0.5 -> "A better". F vs H: ratio 2 -> "A worse"."""
    runs = {}
    for p in ("P1", "P2"):
        for s in range(1, 6):
            runs[(p, "F", s)] = 0.5
            runs[(p, "N", s)] = 1.0
            runs[(p, "H", s)] = 0.25
    rc, rows, out = _report(tmp_path, runs)
    assert rc == 0
    n, h = rows[("pooled", "f-vs-n")], rows[("pooled", "f-vs-h")]
    assert float(n["p"]) == pytest.approx(2 / 1024) and float(h["p"]) == pytest.approx(2 / 1024)
    assert float(n["p_holm"]) == pytest.approx(4 / 1024)
    assert n["verdict"] == "A better" and float(n["ratio"]) == pytest.approx(0.5)
    assert h["verdict"] == "A worse" and float(h["ratio"]) == pytest.approx(2.0)
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:f-vs-n"] == "A better" and v["pooled:f-vs-h"] == "A worse"


def test_pooled_ratio_inside_the_band_is_a_tie(tmp_path):
    """Holm p < 0.025 but exp(median r) = 0.95 > 0.9: "tie", never "extend"."""
    runs = {}
    for p in ("P1", "P2"):
        for s in range(1, 6):
            runs[(p, "F", s)] = 0.95
            runs[(p, "N", s)] = 1.0
            runs[(p, "H", s)] = 1.0
    rc, rows, _ = _report(tmp_path, runs)
    assert rc == 0
    assert rows[("pooled", "f-vs-n")]["verdict"] == "tie"
    assert float(rows[("pooled", "f-vs-n")]["p_holm"]) < 0.025


def test_secondary_have_holm_but_no_verdict(tmp_path):
    """Per problem: 5 pairs with F lower than N: p = 2 / 32 = 0.0625; Holm across the 4
    secondary contrasts. No verdict (empty in the CSV, null in verdict.json)."""
    runs = {}
    for p in ("P1", "P2"):
        for s in range(1, 6):
            runs[(p, "F", s)] = 0.5 * s
            runs[(p, "N", s)] = 1.0 * s
            runs[(p, "H", s)] = 0.5 * s
    rc, rows, out = _report(tmp_path, runs)
    assert rc == 0
    sec = [r for r in rows.values() if r["family"] == "secondary"]
    assert len(sec) == 4 and all(r["verdict"] == "" for r in sec)
    assert float(rows[("P1", "f-vs-n")]["p"]) == pytest.approx(2 / 32)
    # Holm over [0.0625, 0.0625, 1, 1]: 4 * 0.0625 = 0.25, then max(0.25, 3 * 0.0625) = 0.25
    assert float(rows[("P1", "f-vs-n")]["p_holm"]) == pytest.approx(0.25)
    assert float(rows[("P1", "f-vs-h")]["p_holm"]) == 1.0
    v = json.loads((out / "verdict.json").read_text())
    assert v["P1:f-vs-n"] is None


def test_pooled_and_per_problem_primary_together_are_refused(tmp_path):
    spec = {**SPEC, "primary": [["P1", "x", "F", "N"]]}
    rc, _, _ = _report(tmp_path, _runs(), spec)
    assert rc == 2


def test_spec_without_the_new_keys_is_unchanged(tmp_path):
    """A spec with no pooled contrasts and no secondary_verdicts key: per-problem verdicts."""
    spec = {"phase": "t", "metrics": {"P1": "test_nmse"},
            "primary": [["P1", "f-vs-n", "F", "N"]], "secondary": [["P1", "f-vs-h", "F", "H"]]}
    runs = {}
    for s in range(1, 11):
        runs[("P1", "F", s)] = 0.5 * s
        runs[("P1", "N", s)] = 1.0 * s
        runs[("P1", "H", s)] = 1.0 * s
    rc, rows, _ = _report(tmp_path, runs, spec)
    assert rc == 0
    assert rows[("P1", "f-vs-n")]["verdict"] == "A better"
    assert rows[("P1", "f-vs-h")]["verdict"] == "A better"
    assert "median_log_ratio" not in rows[("P1", "f-vs-n")]
