"""Server round 15 (NOT A PORT: the IT linkage study, BACKLOG 13.9): the report additions of
``phase_report.py`` and ``linkage_report.py`` (``SERVER-ROUND15-PREREGISTRATION.md``,
"Metrics" and "Contrasts").

* the metric ``train_nmse`` (before, an unknown metric needed the key ``success``, so it was
  always infinite for a regression run);
* the metric ``anytime_area`` (exp of the trapezoid area under ln(best training NMSE) over
  log10 of the evaluation fraction, divided by the width);
* the pooled interaction contrast d = log(A1 / B1) - log(A2 / B2), its failure rule and its
  verdicts;
* ``"contrast_families"``: primary, secondary and descriptive Holm families of their own, and
  per-dataset families;
* ``"evals_to_reference"``: the evaluations to reach the median final training NMSE of the
  reference arm;
* byte identity: for every spec.json of ``configs/gomea/`` at 3b18ca6 (the commit before these
  changes), the report files and the stdout of the report at 3b18ca6 and now are the same, on
  made-up Boolean and regression runs (also runs with the new summary fields ``anytime`` and an
  "end" record with a trace).
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import linkage_report as lr  # noqa: E402
import phase_report  # noqa: E402
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

OLD = "3b18ca6"  # the commit before the round 15 report changes
FRACS = lr.ANYTIME_FRACTIONS


def _anytime(values):
    return [{"evals": max(1, round(f * 1000)), "frac": f, "best_train_nmse": v, "reached": True}
            for f, v in zip(FRACS, values, strict=True)]


# ---- the metrics


def test_train_nmse_is_a_metric_and_a_failed_run_is_infinite():
    s = {"status": "ok", "train_nmse": 12.5, "test_nmse": 20.0}
    assert phase_report._value("train_nmse", s) == 12.5
    assert phase_report._value("train_nmse", {**s, "status": "error"}) == math.inf
    assert phase_report._value("train_nmse", {"status": "ok", "test_nmse": 1.0}) == math.inf
    assert "train_nmse" in phase_report.METRICS and "anytime_area" in phase_report.METRICS


def test_the_old_report_had_no_train_nmse_metric(tmp_path):
    """Why the metric is new: at 3b18ca6 a regression run had an infinite train_nmse value."""
    old = _old_tools(tmp_path)
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, sys.argv[1]); import phase_report as p; "
                        "print(p._value('train_nmse', {'status': 'ok', 'train_nmse': 1.0}))",
                        str(old)], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.stdout.strip() == "inf"


def test_anytime_area_of_a_constant_curve_is_its_level():
    assert lr.anytime_area({"anytime": _anytime([7.0] * 7)}) == pytest.approx(7.0, rel=1e-12)


def test_anytime_area_of_a_curve_linear_in_the_log_axes():
    """NMSE = 10 / frac: ln(NMSE) is linear in log10(frac), so the trapezoid rule is exact and
    the value is the geometric mean of the end points, sqrt(1000 * 10) = 100."""
    v = lr.anytime_area({"anytime": _anytime([10.0 / f for f in FRACS])})
    assert v == pytest.approx(100.0, rel=1e-12)


def test_anytime_area_is_the_trapezoid_rule_by_hand():
    vals = [50.0, 40.0, 30.0, 20.0, 15.0, 12.0, 11.0]
    xs = [math.log10(f) for f in FRACS]
    ys = [math.log(v) for v in vals]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(6))
    assert lr.anytime_area({"anytime": _anytime(vals)}) == pytest.approx(
        math.exp(area / 2.0), rel=1e-12)
    # the order of the records does not matter
    assert lr.anytime_area({"anytime": _anytime(vals)[::-1]}) == lr.anytime_area(
        {"anytime": _anytime(vals)})


def test_anytime_area_edge_cases():
    assert lr.anytime_area({"anytime": _anytime([5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])}) == 0.0
    assert lr.anytime_area({"anytime": _anytime([None] + [1.0] * 6)}) is None
    assert lr.anytime_area({"anytime": _anytime([1.0] * 7)[:6]}) is None  # a checkpoint missing
    assert lr.anytime_area({}) is None
    s = {"status": "ok", "anytime": _anytime([None] + [1.0] * 6)}
    assert phase_report._value("anytime_area", s) == math.inf
    ok = {"status": "ok", "anytime": _anytime([2.0] * 7)}
    assert phase_report._value("anytime_area", ok) == pytest.approx(2.0)
    assert phase_report._value("anytime_area", {**ok, "status": "error"}) == math.inf


# ---- made-up runs of a 2 x 5 phase


ARMS = ["T-UNI", "T-RT", "T-LT", "T-LTB", "T-NP", "IT-UNI", "IT-RT", "IT-LT", "IT-LTB", "IT-NP"]


def _write_run(root: Path, task, arm, seed, train, status="ok", trace=None, anytime=None):
    rd = root / "runs" / f"{task}__{arm}__s{seed}"
    rd.mkdir(parents=True)
    s = {"task": task, "arm": arm, "seed": seed, "status": status,
         "train_nmse": train if status == "ok" else None,
         "val_nmse": 1.1 * train if status == "ok" else None,
         "test_nmse": 1.2 * train if status == "ok" else None,
         "test_nmse_clipped": 1.2 * train if status == "ok" else None,
         "n_evals": 1000 if status == "ok" else None, "cpu_s": 3.0,
         "final_program": "+($1 $2)" if status == "ok" else None,
         "anytime": anytime if anytime is not None else _anytime([train * 2] * 6 + [train])}
    (rd / "summary.json").write_text(json.dumps(s))
    t = trace if trace is not None else [[1, train * 3], [500, train * 2], [900, train]]
    end = {"type": "end", "n_evals": 1000, "generations": 5, "trace": t, "anytime": s["anytime"]}
    (rd / "events.jsonl").write_text(json.dumps({"type": "start"}) + "\n" + json.dumps(end) + "\n")
    (rd / "candidates.json").write_text(json.dumps([{"program": "+($1 $2)"}]))


def _runs(root: Path, factor, datasets=("Con", "Dow"), seeds=range(71, 81), fail=()):
    for d in datasets:
        for k in seeds:
            for arm in ARMS:
                base = 10.0 + k / 10 + (5 if d == "Dow" else 0)
                f = factor(arm, d, k)
                _write_run(root, d, arm, k, base * f,
                           status="error" if (d, arm, k) in fail else "ok")


def _spec(tmp_path, families, ref=None, name="spec.json"):
    spec = {"phase": "t15", "metrics": {"Con": "train_nmse", "Dow": "train_nmse"},
            "primary": [], "extend": False, "contrast_families": families}
    if ref is not None:
        spec["evals_to_reference"] = ref
    p = tmp_path / name
    p.write_text(json.dumps(spec))
    return p


def _report(tmp_path, spec, root, out="out"):
    o = tmp_path / out
    rc = phase_report.main(["--stage", "1", str(spec), str(o), str(root)])
    return rc, o


def _rows(path):
    return list(csv.DictReader(path.open()))


P1 = [[f"{f.lower()}-gain", f"IT-{f}", "IT-RT", f"T-{f}", "T-RT", "train_nmse"]
      for f in ("UNI", "LT", "LTB", "NP")]
P2 = [[f"it-{f.lower()}-vs-rt", f"IT-{f}", "IT-RT", "train_nmse"]
      for f in ("UNI", "LT", "LTB", "NP")] + [["it-np-vs-ltb", "IT-NP", "IT-LTB", "train_nmse"]]


def test_the_interaction_d_by_hand(tmp_path):
    """d = log(A1 / B1) - log(A2 / B2) per pair; ratio = exp(median d)."""
    def factor(arm, d, k):
        return {"IT-LTB": 0.5, "IT-RT": 1.0, "T-LTB": 0.9, "T-RT": 1.0}.get(arm, 1.0) * (
            1 + (k % 3) / 100 if arm == "IT-LTB" else 1.0)
    root = tmp_path / "r"
    _runs(root, factor)
    runs = {}
    phase_report._load(str(root), runs)
    row = lr.interaction_contrast("x", "IT-LTB", "IT-RT", "T-LTB", "T-RT", runs, ["Con", "Dow"],
                                  "train_nmse", phase_report._value)
    want = []
    for d in ("Con", "Dow"):
        for k in range(71, 81):
            v = {a: runs[d][a][k]["train_nmse"] for a in ("IT-LTB", "IT-RT", "T-LTB", "T-RT")}
            want.append(math.log(v["IT-LTB"] / v["IT-RT"]) - math.log(v["T-LTB"] / v["T-RT"]))
    assert row["pairs"] == 20 and row["a_lower"] == 20 and row["b_lower"] == 0
    med = sorted(want)[9:11]
    assert row["median_log_ratio"] == pytest.approx(sum(med) / 2, rel=1e-12)
    assert row["ratio"] == pytest.approx(math.exp(sum(med) / 2), rel=1e-12)
    assert row["p"] == wilcoxon_signed_rank(want)[1]
    assert row["metric"] == "interaction:train_nmse"
    assert (row["a"], row["b"], row["a2"], row["b2"]) == ("IT-LTB", "IT-RT", "T-LTB", "T-RT")


def test_a_failed_run_is_infinitely_bad_in_the_interaction():
    def s(v):
        return {"status": "ok", "train_nmse": v} if v is not None else {"status": "error"}

    def d_of(a1, b1, a2, b2):
        runs = {"Con": {"A1": {1: s(a1)}, "B1": {1: s(b1)}, "A2": {1: s(a2)}, "B2": {1: s(b2)}}}
        r = lr.interaction_contrast("x", "A1", "B1", "A2", "B2", runs, ["Con"], "train_nmse",
                                    phase_report._value)
        return r["median_log_ratio"]

    assert d_of(None, 1.0, 1.0, 1.0) == lr.BIG  # A1 failed: the A2 side gains more
    assert d_of(1.0, None, 1.0, 1.0) == -lr.BIG  # B1 failed: the A1 side gains more
    assert d_of(None, None, 2.0, 1.0) == pytest.approx(-math.log(2.0))  # log(A1/B1) = 0
    assert d_of(None, 1.0, None, 1.0) == 0.0
    assert d_of(None, 1.0, 1.0, None) == 2 * lr.BIG
    assert d_of(None, 1.0, 2.0, 1.0) == pytest.approx(lr.BIG - math.log(2.0))
    assert lr._ratio(lr.BIG) == math.inf and lr._ratio(-lr.BIG) == 0.0


def test_families_verdicts_and_holm(tmp_path):
    """P1 and P2 are primary families of their own; S1 secondary; S3 descriptive; D per
    dataset. IT-LTB gains much more over IT-RT than T-LTB over T-RT, IT-NP gains less."""
    def factor(arm, d, k):
        jit = 1 + ((k * 7 + len(arm)) % 5) / 1000
        return {"IT-LTB": 0.5, "IT-NP": 1.0, "T-NP": 0.5}.get(arm, 1.0) * jit
    root = tmp_path / "r"
    _runs(root, factor)
    fams = [{"name": "P1", "role": "primary", "contrasts": P1},
            {"name": "P2", "role": "primary", "contrasts": P2},
            {"name": "S1", "role": "secondary",
             "contrasts": [[c[0].replace("it-", "t-"), c[1].replace("IT-", "T-"),
                            c[2].replace("IT-", "T-"), c[3]] for c in P2]},
            {"name": "S3", "role": "descriptive",
             "contrasts": [[c[0] + "-test"] + c[1:-1] + ["test_nmse"] for c in P1 + P2]
             + [[c[0] + "-anytime"] + c[1:-1] + ["anytime_area"] for c in P1 + P2]},
            {"name": "D", "role": "descriptive", "per_dataset": True, "contrasts": P1 + P2}]
    rc, out = _report(tmp_path, _spec(tmp_path, fams), root)
    assert rc == 0
    rows = _rows(out / "contrasts.csv")
    by = {(r["problem"], r["contrast"]): r for r in rows}
    assert len(rows) == 4 + 5 + 5 + 18 + 9 * 2
    assert {r["holm_family"] for r in rows} == {"P1", "P2", "S1", "S3", "D"}
    for fam, n in (("P1", 4), ("P2", 5), ("S1", 5), ("S3", 18), ("D", 18)):
        fr = [r for r in rows if r["holm_family"] == fam]
        assert len(fr) == n
        want = holm([float(r["p"]) for r in fr])
        assert [float(r["p_holm"]) for r in fr] == pytest.approx(want, rel=1e-12)
    assert by[("pooled", "ltb-gain")]["verdict"] == "A1 gains more"
    assert by[("pooled", "np-gain")]["verdict"] == "A2 gains more"
    assert by[("pooled", "uni-gain")]["verdict"] == "tie"
    assert by[("pooled", "it-ltb-vs-rt")]["verdict"] == "A better"
    assert by[("pooled", "it-np-vs-ltb")]["verdict"] == "A worse"
    assert by[("pooled", "t-np-vs-rt")]["verdict"] == "A better"
    assert by[("pooled", "ltb-gain")]["family"] == "primary"
    assert by[("pooled", "t-uni-vs-rt")]["family"] == "secondary"
    assert by[("pooled", "ltb-gain-anytime")]["family"] == "descriptive"
    assert by[("pooled", "ltb-gain-anytime")]["verdict"] == "descriptive"
    assert by[("pooled", "ltb-gain-anytime")]["metric"] == "interaction:anytime_area"
    assert by[("Con", "ltb-gain")]["pairs"] == "10" and by[("Dow", "it-np-vs-ltb")]["pairs"] == "10"
    assert by[("Con", "ltb-gain")]["verdict"] == "descriptive"
    assert all(r["pairs"] == "20" for r in rows if r["problem"] == "pooled")
    v = json.loads((out / "verdict.json").read_text())
    assert set(v) == {f"pooled:{c[0]}" for c in P1 + P2 + fams[2]["contrasts"]}
    assert v["pooled:ltb-gain"] == "A1 gains more"


def test_the_anytime_metric_in_a_contrast(tmp_path):
    """A faster curve with the same final value: the anytime contrast sees it, the final
    training NMSE does not."""
    root = tmp_path / "r"
    for d in ("Con", "Dow"):
        for k in range(71, 81):
            for arm in ("A", "B"):
                curve = ([3.0, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0] if arm == "A"
                         else [9.0, 8.0, 6.0, 4.0, 2.0, 1.5, 1.0])
                _write_run(root, d, arm, k, 1.0, anytime=_anytime([c * (1 + k / 100)
                                                                   for c in curve]))
    fams = [{"name": "X", "role": "secondary",
             "contrasts": [["a-vs-b", "A", "B", "anytime_area"], ["a-vs-b-final", "A", "B",
                                                                    "train_nmse"]]}]
    rc, out = _report(tmp_path, _spec(tmp_path, fams), root)
    assert rc == 0
    by = {r["contrast"]: r for r in _rows(out / "contrasts.csv")}
    assert by["a-vs-b"]["verdict"] == "A better" and by["a-vs-b"]["a_lower"] == "20"
    assert by["a-vs-b-final"]["verdict"] == "tie" and by["a-vs-b-final"]["ties"] == "20"


@pytest.mark.parametrize("families, ref, msg", [
    ([], None, "bad contrast_families"),
    ([{"name": "P1", "role": "main", "contrasts": P1}], None, "bad role"),
    ([{"name": "P1", "role": "primary", "contrasts": [["x", "A", "B", "nope"]]}], None,
     "unknown metric"),
    ([{"name": "P1", "role": "primary", "contrasts": [["x", "A", "B", "C", "train_nmse"]]}],
     None, "bad contrast"),
    ([{"name": "P1", "role": "primary", "contrasts": P1},
      {"name": "P1", "role": "secondary", "contrasts": P2}], None, "bad family name"),
    ([{"name": "P1", "role": "primary", "contrasts": P1},
      {"name": "P2", "role": "secondary", "contrasts": P1}], None, "must be distinct"),
    ([{"name": "P1", "role": "primary", "contrasts": P1, "extra": 1}], None, "bad family"),
    ([{"name": "P1", "role": "primary", "contrasts": P1, "per_dataset": "yes"}], None,
     "per_dataset"),
    ([{"name": "P1", "role": "primary", "contrasts": []}], None, "has no contrasts"),
    ([{"name": "P1", "role": "primary", "contrasts": P1}], {"T-UNI": 3}, "bad evals_to_reference"),
])  # fmt: skip
def test_a_malformed_round15_key_is_refused(families, ref, msg, tmp_path, capsys):
    root = tmp_path / "r"
    _runs(root, lambda a, d, k: 1.0, seeds=[71, 72])
    rc, _ = _report(tmp_path, _spec(tmp_path, families, ref), root)
    assert rc == 2
    assert msg in capsys.readouterr().err


def test_evals_to_reference(tmp_path):
    """The target is the median final training NMSE of the reference arm on the dataset; a run
    reaches it at the first trace point at or below it."""
    root = tmp_path / "r"
    finals = {71: 10.0, 72: 20.0, 73: 30.0}  # RT finals; median 20
    for k, f in finals.items():
        _write_run(root, "Con", "IT-RT", k, f, trace=[[1, 100.0], [400, f]])
    _write_run(root, "Con", "IT-LTB", 71, 5.0, trace=[[1, 90.0], [300, 25.0], [350, 20.0],
                                                      [800, 5.0]])
    _write_run(root, "Con", "IT-LTB", 72, 21.0, trace=[[1, 90.0], [600, 21.0]])  # never
    _write_run(root, "Con", "IT-LTB", 73, 1.0, status="error")
    fams = [{"name": "P2", "role": "primary", "contrasts": [P2[2]]}]
    ref = {"IT-LTB": "IT-RT", "IT-RT": "IT-RT"}
    spec = _spec(tmp_path, fams, ref)
    s = json.loads(spec.read_text())
    s["metrics"] = {"Con": "train_nmse"}
    spec.write_text(json.dumps(s))
    rc, out = _report(tmp_path, spec, root)
    assert rc == 0
    rows = {(r["problem"], r["arm"]): r for r in _rows(out / "evals_to_reference.csv")}
    assert list(_rows(out / "evals_to_reference.csv")[0]) == list(lr.EVALS_FIELDS)
    r = rows[("Con", "IT-LTB")]
    assert float(r["target_train_nmse"]) == 20.0 and r["reference"] == "IT-RT"
    assert (r["runs"], r["failed"], r["reached"]) == ("3", "1", "1")
    assert r["median_evals_to_reference"] == "inf" and r["median_evals_reached"] == "350.0"
    rt = rows[("Con", "IT-RT")]
    assert (rt["reached"], rt["median_evals_to_reference"]) == ("2", "400.0")
    assert rows[("pooled", "IT-LTB")]["runs"] == "3" and rows[("pooled", "IT-LTB")][
        "target_train_nmse"] == ""


def test_evals_to_target_tolerates_rounding():
    s = {"status": "ok", "_events_path": None}
    assert lr.evals_to_target(s, 1.0) == math.inf  # no trace
    assert lr.evals_to_target({"status": "error"}, 1.0) == math.inf


# ---- byte identity against 3b18ca6 (git)


def _old_tools(tmp_path: Path) -> Path:
    d = tmp_path / f"old-{OLD}"
    d.mkdir(exist_ok=True)
    for n in ("phase_report.py", "run_table.py"):
        r = subprocess.run(["git", "show", f"{OLD}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
            pytest.fail(f"git show {OLD}:tools/gomea/{n} failed: {r.stderr.decode().strip()!r}. "
                        f"The byte-identity test needs the reference commit {OLD}; use a clone "
                        f"with the full history.")
        (d / n).write_bytes(r.stdout)
    return d


def _reference_specs() -> list[str]:
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", OLD, "configs/gomea"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"git ls-tree {OLD} failed: {r.stderr.strip()!r}")
    return sorted(p for p in r.stdout.split() if p.endswith("/spec.json"))


def _arms_of(spec: dict) -> list[str]:
    arms = []
    for k in ("primary", "secondary", "descriptive"):
        for c in spec.get(k, []):
            arms += [c[2], c[3]]
    for k in ("pooled_primary", "pooled_secondary"):
        for c in spec.get(k, []):
            arms += [c[1], c[2]]
    for a, b in spec.get("same_search", []):
        arms += [a, b]
    return list(dict.fromkeys(arms))


BOOL_METRICS = {"cpu_to_target", "evals_to_target"}


def _fixture_runs(root: Path, spec: dict) -> None:
    """Made-up runs for one spec: Boolean summaries for a problem with a Boolean metric,
    regression summaries otherwise (with per-term records, LM records, the round 15 fields
    `anytime` and an "end" record with a trace, failed and unsolved runs, demes)."""
    problems = sorted(set(spec["metrics"]) | {c[0] for k in ("primary", "secondary",
                                                             "descriptive")
                                              for c in spec.get(k, [])})
    arms = _arms_of(spec) or ["A", "B"]  # a spec without contrasts still aggregates its runs
    for pi, p in enumerate(problems):
        boolean = spec["metrics"].get(p) in BOOL_METRICS
        for ai, arm in enumerate(arms):
            for k in range(1, 9):
                rd = root / "runs" / f"{p}__{arm}__s{k}"
                rd.mkdir(parents=True)
                x = (1.0 + 0.15 * ai) * (1.0 + k / 37.0) * (1.0 + pi / 11.0)
                failed = (ai + k + pi) % 13 == 0
                status = "error" if failed else "ok"
                if boolean:
                    success = not failed and (ai * 3 + k) % 7 != 0
                    s = {"arm": arm, "task": p, "seed": k, "status": status, "success": success,
                         "cpu_to_target": 2.0 * x if success else None,
                         "evals_to_target": int(1000 * x) if success else None,
                         "cpu_s": 3.0 * x, "n_calls": 5000, "deme_count": 2 + ai % 3,
                         "best_cpx": 5 + k % 4, "best_raw": 0 if success else -2}
                else:
                    s = {"arm": arm, "task": p, "seed": k, "status": status,
                         "test_nmse": None if failed else 10.0 * x,
                         "train_nmse": None if failed else 8.0 * x,
                         "val_nmse": None if failed else 9.0 * x,
                         "test_nmse_clipped": None if failed else 9.5 * x,
                         "cpu_s": 4.0 * x, "n_evals": 20000,
                         "final_program": "+(*($1 $2) sin($3))",
                         "anytime": _anytime([20.0 * x] * 6 + [8.0 * x]),
                         "generations": 7, "cache_hits": 11}
                    if ai % 2:
                        s["terms"] = {"n_terms": 2, "fallback": False, "reason": None,
                                      "weights": [0.0, 1.0, 2.0]}
                    if ai % 3 == 0:
                        s["const_opt"] = {"mode": "lm", "lm_calls": 3 + k, "tuning_evals": 50,
                                          "tuning_share": 0.01, "lm_guard_checked": 2,
                                          "lm_guard_rejected": 1}
                    if pi == 0 and k == 1 and not failed:
                        s["test_nmse"] = 150.0  # a run above 100
                (rd / "summary.json").write_text(json.dumps(s))
                end = {"type": "end", "generations": 3 + k % 2, "termination_reason": "stall",
                       "ledger": {"requests": 10}, "dim": 30 + ai, "pop_size_effective": 64}
                st_end = {"type": "end", "n_evals": 20000, "trace": [[1, 30.0], [900, 8.0 * x]]}
                (rd / "events.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in (end, end, st_end)) + "\n")
                (rd / "candidates.json").write_text(json.dumps(
                    [{"program": "$1"}, {"program": f"+($1 {ai % 2})"}]))


DRIVER = r"""import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
import phase_report
for spec, out, root, log in json.load(open(sys.argv[2])):
    for stage in ("1", "2"):
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            rc = phase_report.main(["--stage", stage, spec, out + "-stage" + stage, root])
        with open(log + "-stage" + stage, "w") as fh:
            json.dump({"rc": rc, "stdout": o.getvalue(), "stderr": e.getvalue()}, fh)
"""


def test_every_old_spec_gives_a_byte_identical_report(tmp_path):
    """Every spec.json of configs/gomea/ at 3b18ca6 (70 phases and smoke phases): the files
    and the stdout of the report at 3b18ca6 and now are the same, at stage 1 and stage 2."""
    specs = _reference_specs()
    assert len(specs) >= 70
    old = _old_tools(tmp_path)
    jobs = {"old": [], "new": []}
    for rel in specs:
        spec = REPO / rel
        name = spec.parent.name
        sj = json.loads(spec.read_text())
        assert "contrast_families" not in sj and "evals_to_reference" not in sj
        root = tmp_path / "runs" / name
        _fixture_runs(root, sj)
        for tag in jobs:
            jobs[tag].append([str(spec), str(tmp_path / tag / name), str(root),
                              str(tmp_path / tag / f"{name}.log")])
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for tag, tool in (("old", old), ("new", TOOLS)):
        (tmp_path / tag).mkdir()
        jf = tmp_path / f"{tag}-jobs.json"
        jf.write_text(json.dumps(jobs[tag]))
        subprocess.run([sys.executable, "-c", DRIVER, str(tool), str(jf)], env=env, check=True,
                       cwd=tmp_path)
    compared = 0
    for (_, o_out, _, o_log), (_, n_out, _, n_log) in zip(jobs["old"], jobs["new"], strict=True):
        for stage in ("1", "2"):
            lo = json.loads(Path(o_log + "-stage" + stage).read_text())
            ln = json.loads(Path(n_log + "-stage" + stage).read_text())
            assert ln == lo, o_out
            od, nd = Path(o_out + "-stage" + stage), Path(n_out + "-stage" + stage)
            files = sorted(p.name for p in od.iterdir()) if od.exists() else []
            assert (sorted(p.name for p in nd.iterdir()) if nd.exists() else []) == files
            for f in files:
                assert (nd / f).read_bytes() == (od / f).read_bytes(), (o_out, f)
                compared += 1
            if stage == "1":
                assert lo["rc"] == 0, (o_out, lo["stderr"])
                assert {"contrasts.csv", "aggregate.csv", "verdict.json"} <= set(files)
    assert compared >= 3 * 2 * len(specs)
