"""Tests for the generic phase report, tools/gomea/phase_report.py, on synthetic summaries."""

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import phase_report  # noqa: E402


def _bool(arm, seed, cpu, task="mux11", success=True, status="ok", evals=None):
    return {"arm": arm, "task": task, "seed": seed, "status": status, "success": success,
            "cpu_to_target": cpu if success else None,
            "evals_to_target": (evals if evals is not None else 100 * seed) if success else None,
            "cpu_s": 2.0 * (cpu or 1.0), "n_calls": 1000, "deme_count": 3, "best_cpx": 7,
            "best_raw": 0 if success else -4}


def _reg(arm, split, nmse, task="Con", status="ok"):
    return {"arm": arm, "task": task, "seed": split, "status": status,
            "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse,
            "cpu_s": 10.0, "n_evals": 500, "final_program": "(+ $x1 (* $x2 2))"}


def _write(root: Path, name: str, summaries: list[dict]) -> str:
    d = root / name
    for s in summaries:
        rd = d / "runs" / f"{s['task']}__{s['arm']}__s{s['seed']}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "summary.json").write_text(json.dumps(s))
    return str(d)


def _run(tmp_path, spec, dirs, stage=1):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / f"out{stage}"
    rc = phase_report.main(["--stage", str(stage), str(sp), str(out), ",".join(dirs)])
    verdict = json.loads((out / "verdict.json").read_text()) if rc == 0 else None
    return rc, out, verdict


def _rows(out: Path, name: str) -> list[dict]:
    with (out / name).open() as fh:
        return list(csv.DictReader(fh))


SPEC = {"phase": "t", "metrics": {"mux11": "cpu_to_target"},
        "primary": [["mux11", "ab", "A", "B"]]}
SEEDS = range(3001, 3011)


def test_better_worse_tie(tmp_path):
    d = _write(tmp_path, "r", [_bool("A", s, 1.0 + 0.01 * i) for i, s in enumerate(SEEDS)]
               + [_bool("B", s, 5.0 + i) for i, s in enumerate(SEEDS)]
               + [_bool("C", s, 1.0 + i) for i, s in enumerate(SEEDS)]
               + [_bool("D", s, 1.0 + i + (0.5 if i % 2 else -0.5)) for i, s in enumerate(SEEDS)])
    spec = {"phase": "t", "metrics": {"mux11": "cpu_to_target"},
            "primary": [["mux11", "ab", "A", "B"], ["mux11", "ba", "B", "A"],
                        ["mux11", "cd", "C", "D"]]}
    rc, _, v = _run(tmp_path, spec, [d])
    assert rc == 0
    assert v == {"mux11:ab": "A better", "mux11:ba": "A worse", "mux11:cd": "tie"}


def test_median_rule_blocks_small_gain(tmp_path):
    # A always lower, but its median only 5 % lower: significant, not "better".
    d = _write(tmp_path, "r", [_bool("A", s, 95.0 + i) for i, s in enumerate(SEEDS)]
               + [_bool("B", s, 100.0 + i) for i, s in enumerate(SEEDS)])
    _, out, v = _run(tmp_path, SPEC, [d], stage=2)
    assert float(_rows(out, "contrasts.csv")[0]["p_holm"]) < phase_report.ALPHA
    assert v == {"mux11:ab": "tie"}


def test_extend_only_primary_stage1(tmp_path):
    # 6 of 10 lower by a lot, 4 higher by a little: raw p = 0.066.
    a = [1.0] * 6 + [12.0, 14.0, 16.0, 18.0]
    b = [10.0] * 6 + [11.0, 12.0, 13.0, 14.0]
    d = _write(tmp_path, "r", [_bool("A", s, x) for s, x in zip(SEEDS, a)]
               + [_bool("B", s, x) for s, x in zip(SEEDS, b)])
    spec = dict(SPEC, secondary=[["mux11", "ab2", "A", "B"]],
                descriptive=[["mux11", "ab3", "A", "B"]])
    rc, out, v = _run(tmp_path, spec, [d])
    rows = {r["contrast"]: r for r in _rows(out, "contrasts.csv")}
    assert ALPHA_OK(float(rows["ab"]["p_holm"]))
    assert v == {"mux11:ab": "extend", "mux11:ab2": "tie"}
    assert rows["ab3"]["verdict"] == "" and rows["ab3"]["family"] == "descriptive"
    _, _, v2 = _run(tmp_path, spec, [d], stage=2)
    assert v2["mux11:ab"] == "tie"


def ALPHA_OK(p):  # noqa: N802
    return phase_report.ALPHA <= p < phase_report.EXTEND_BELOW


def test_unsolved_and_failed_are_infinite(tmp_path):
    a = [_bool("A", s, 1.0) for s in SEEDS]
    b = ([_bool("B", s, None, success=False) for s in list(SEEDS)[:8]]
         + [_bool("B", s, None, status="error", success=False) for s in list(SEEDS)[8:]])
    d = _write(tmp_path, "r", a + b)
    rc, out, v = _run(tmp_path, SPEC, [d])
    row = _rows(out, "contrasts.csv")[0]
    assert row["a_lower"] == "10" and math.isinf(float(row["median_b"]))
    assert v["mux11:ab"] == "A better"
    agg = {r["arm"]: r for r in _rows(out, "aggregate.csv")}
    assert agg["B"]["solved"] == "0" and agg["B"]["not_ok"] == "2"
    assert agg["A"]["solved"] == "10"


def test_both_infinite_is_a_tie_pair(tmp_path):
    d = _write(tmp_path, "r", [_bool(arm, s, None, success=False) for arm in "AB" for s in SEEDS])
    _, out, v = _run(tmp_path, SPEC, [d])
    row = _rows(out, "contrasts.csv")[0]
    assert row["ties"] == "10" and float(row["p"]) == 1.0 and v["mux11:ab"] == "tie"


def test_later_dir_overrides_earlier(tmp_path):
    d1 = _write(tmp_path, "r1", [_bool("A", s, 50.0) for s in SEEDS]
                + [_bool("B", s, 10.0 + i) for i, s in enumerate(SEEDS)])
    d2 = _write(tmp_path, "r2", [_bool("A", s, 1.0) for s in SEEDS])
    _, _, v = _run(tmp_path, SPEC, [d1])
    assert v["mux11:ab"] == "A worse"
    _, _, v = _run(tmp_path, SPEC, [d1, d2])
    assert v["mux11:ab"] == "A better"
    _, _, v = _run(tmp_path, SPEC, [d2, d1])
    assert v["mux11:ab"] == "A worse"


def test_holm_over_family_and_mixed_dirs(tmp_path):
    # Two primary contrasts with raw p ~ 0.002; Holm doubles the smaller one.
    db = _write(tmp_path, "b", [_bool("A", s, 1.0 + i) for i, s in enumerate(SEEDS)]
                + [_bool("B", s, 20.0 + 2 * i) for i, s in enumerate(SEEDS)])
    dr = _write(tmp_path, "r", [_reg("A", k, 0.1 + 0.01 * k) for k in range(1, 11)]
                + [_reg("B", k, 0.5 + 0.02 * k) for k in range(1, 11)])
    spec = {"phase": "t", "metrics": {"mux11": "cpu_to_target", "Con": "test_nmse"},
            "primary": [["mux11", "ab", "A", "B"], ["Con", "ab", "A", "B"]],
            "secondary": [["Con", "ba", "B", "A"]]}
    rc, out, v = _run(tmp_path, spec, [db, dr])
    assert rc == 0
    rows = {(r["problem"], r["contrast"]): r for r in _rows(out, "contrasts.csv")}
    p1, p2 = float(rows["mux11", "ab"]["p"]), float(rows["Con", "ab"]["p"])
    h1, h2 = float(rows["mux11", "ab"]["p_holm"]), float(rows["Con", "ab"]["p_holm"])
    assert math.isclose(min(h1, h2), 2 * min(p1, p2))
    # the secondary family has one member: Holm leaves its p alone
    assert math.isclose(float(rows["Con", "ba"]["p_holm"]), float(rows["Con", "ba"]["p"]))
    assert v == {"mux11:ab": "A better", "Con:ab": "A better", "Con:ba": "A worse"}
    agg = {(r["problem"], r["arm"]): r for r in _rows(out, "aggregate.csv")}
    assert agg["Con", "A"]["solved"] == "" and float(agg["Con", "A"]["median_size"]) == 7
    assert float(agg["Con", "A"]["median_cpu_per_eval"]) == 10.0 / 500
    assert float(agg["mux11", "A"]["median_size"]) == 7


def test_failed_regression_run_is_infinite(tmp_path):
    d = _write(tmp_path, "r", [_reg("A", k, 0.1) for k in range(1, 11)]
               + [_reg("B", k, 0.2, status="error") for k in range(1, 11)])
    spec = {"phase": "t", "metrics": {"Con": "test_nmse"}, "primary": [["Con", "ab", "A", "B"]]}
    _, out, v = _run(tmp_path, spec, [d])
    assert v["Con:ab"] == "A better"
    assert math.isinf(float(_rows(out, "contrasts.csv")[0]["median_b"]))


def test_seed_fields(tmp_path):
    a = [dict(_reg("A", k, 0.1), seed_test_nmse=0.3, final_lineage="seed2") for k in range(1, 11)]
    a[0]["final_lineage"] = "empty"
    d = _write(tmp_path, "r", a)
    spec = {"phase": "t", "metrics": {"Con": "test_nmse"}, "descriptive": [["Con", "x", "A", "A"]]}
    _, out, v = _run(tmp_path, spec, [d])
    assert v == {}
    agg = _rows(out, "aggregate.csv")[0]
    assert agg["final_from_seed"] == "9" and float(agg["median_seed_test_nmse"]) == 0.3


def test_no_runs_exits_1(tmp_path):
    rc, _, _ = _run(tmp_path, SPEC, [str(tmp_path / "nothing")])
    assert rc == 1


def _gp(arm, split, nmse, task="Con", status="ok"):
    """A `gpgomea_sr_runner.py` summary (infix program, its own node count)."""
    return {"arm": arm, "task": task, "seed": split, "status": status, "exit_code": 0,
            "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse,
            "val_nmse": None, "cpu_s": 0.5, "evaluations": 17869.0,
            "final_program": "((x1+x2)*x3)", "final_program_nodes": 5.0}


def test_gpgomea_summaries_pair_with_moses_by_split(tmp_path):
    # Phase 1ai: pyMoses GOMEA (moses_sr_runner) vs stand-alone GP-GOMEA in its own dir.
    dm = _write(tmp_path, "sr", [_reg("GOMEA", k, 0.1 + 0.01 * k) for k in range(1, 11)])
    dg = _write(tmp_path, "gp", [_gp("GPGOMEA", k, 0.5 + 0.02 * k) for k in range(1, 9)]
                + [_gp("GPGOMEA", k, None, status="timeout") for k in (9, 10)])
    spec = {"phase": "t", "metrics": {"Con": "test_nmse"},
            "primary": [["Con", "vs-gpgomea", "GOMEA", "GPGOMEA"]]}
    rc, out, v = _run(tmp_path, spec, [dm, dg])
    assert rc == 0 and v == {"Con:vs-gpgomea": "A better"}
    row = _rows(out, "contrasts.csv")[0]
    assert row["pairs"] == "10" and row["a_lower"] == "10"
    # the two timed-out GP-GOMEA runs count as inf: median of 0.52..0.66 plus inf, inf
    assert math.isclose(float(row["median_b"]), 0.61)
    agg = {r["arm"]: r for r in _rows(out, "aggregate.csv")}
    assert float(agg["GPGOMEA"]["median_size"]) == 5  # final_program_nodes, not the infix
    assert float(agg["GPGOMEA"]["median_evals_used"]) == 17869
    assert float(agg["GPGOMEA"]["median_cpu_per_eval"]) == 0.5 / 17869
    assert agg["GPGOMEA"]["not_ok"] == "2"
    assert float(agg["GOMEA"]["median_size"]) == 7 and float(agg["GOMEA"]["median_evals_used"]) == 500


def test_scores_now_shows_gpgomea_runs(tmp_path):
    import os
    import subprocess

    res = tmp_path / "res"
    _write(res / "phase1ai", "phase1ai-sr", [_reg("GOMEA", k, 0.1, task="Air") for k in (1, 2)])
    _write(res / "phase1ai", "phase1ai-gp", [_gp("GPGOMEA", k, 0.3, task="Air") for k in (1, 2)])
    script = Path(__file__).resolve().parents[2] / "tools" / "gomea" / "scores_now.py"
    out = subprocess.run([sys.executable, str(script), "phase1ai"], capture_output=True,
                         text=True, env={**os.environ, "GOMEA_RESULTS": str(res)}).stdout
    rows = {line.split()[0]: line.split() for line in out.splitlines()
            if line.strip().startswith(("GOMEA", "GPGOMEA"))}
    # arm runs fail test train gap size evals cpu
    assert rows["GPGOMEA"][1:] == ["2", "0", "0.3", "0.3", "0.0", "5", "17,869", "0"]
    assert rows["GOMEA"][1:] == ["2", "0", "0.1", "0.1", "0.0", "7", "500", "10"]
    import importlib.util

    spec = importlib.util.spec_from_file_location("scores_now", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT == ["phase1ah", "phase1ai", "phase1af-pilot", "phase1af", "phase1aj",
                           "phase1ag", "phase1ae"]


def test_program_size_metric(tmp_path):
    # Server round 3, 1ap: program size as a primary metric (smaller is better).
    small = "(+ $x1 2)"  # 1 "(" + 3 atoms = 4
    big = "(+ $x1 (* $x2 (+ $x3 (* $x4 2))))"  # 4 "(" + 9 atoms = 13
    a = [dict(_reg("A", k, 0.2), final_program=small) for k in range(1, 11)]
    b = [dict(_reg("B", k, 0.1), final_program=big) for k in range(1, 11)]
    b[0]["status"] = "error"  # a failed run is infinitely large
    d = _write(tmp_path, "r", a + b)
    spec = {"phase": "t", "metrics": {"Con": "program_size"},
            "primary": [["Con", "size", "A", "B"]],
            "secondary": [["Con", "nmse", "A", "B"]]}
    rc, out, v = _run(tmp_path, spec, [d])
    assert rc == 0 and v["Con:size"] == "A better"
    row = _rows(out, "contrasts.csv")[0]
    assert row["metric"] == "program_size" and float(row["median_a"]) == 4.0
    assert float(row["median_b"]) == 13.0


def test_const_opt_columns(tmp_path):
    co = {"lm_calls": 4, "lm_improved": 1, "lm_improved_share": 0.25, "lm_singular": 1,
          "lm_failed": 0, "tuning_evals": 50, "tuning_share": 0.05, "polish_accepted": True}
    a = [dict(_reg("A", k, 0.2), const_opt=co) for k in range(1, 11)]
    d = _write(tmp_path, "r", a + [_reg("B", k, 0.3) for k in range(1, 11)])
    spec = {"phase": "t", "metrics": {"Con": "test_nmse"}, "primary": [["Con", "x", "A", "B"]]}
    _, out, _ = _run(tmp_path, spec, [d])
    agg = {r["arm"]: r for r in _rows(out, "aggregate.csv")}
    assert float(agg["A"]["median_lm_improved_share"]) == 0.25
    assert float(agg["A"]["median_tuning_evals"]) == 50 and agg["A"]["polish_accepted"] == "10"
    assert agg["B"]["median_lm_calls"] == ""


def test_contrast_metric_override(tmp_path):
    # a fifth element sets the contrast's metric (1ap: size primary, NMSE secondary)
    a = [dict(_reg("A", k, 0.3), final_program="(+ $x1 2)") for k in range(1, 11)]
    b = [_reg("B", k, 0.1) for k in range(1, 11)]
    d = _write(tmp_path, "r", a + b)
    spec = {"phase": "t", "metrics": {"Con": "program_size"},
            "primary": [["Con", "size", "A", "B"]],
            "secondary": [["Con", "nmse", "A", "B", "test_nmse"]]}
    rc, out, v = _run(tmp_path, spec, [d])
    assert rc == 0 and v == {"Con:size": "A better", "Con:nmse": "A worse"}
    spec["secondary"][0][4] = "nope"
    rc, _, _ = _run(tmp_path, spec, [d])
    assert rc == 2
