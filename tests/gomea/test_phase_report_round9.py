"""Server round 9 (SERVER-ROUND9-PREREGISTRATION.md): the new contrast metrics of
tools/gomea/phase_report.py (`train_test_gap`, `cpu_s`, `lm_calls`) and the spec key
`"extend": false` (1bc has no extension)."""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import phase_report  # noqa: E402

SPLITS = range(1, 11)


def _reg(arm, split, nmse=0.1, train=0.1, cpu=10.0, lm=None, status="ok", task="Con"):
    s = {"arm": arm, "task": task, "seed": split, "status": status,
         "test_nmse": nmse if status == "ok" else None,
         "train_nmse": train if status == "ok" else None,
         "cpu_s": cpu, "n_evals": 500, "final_program": "(+ $x1 (* $x2 2))"}
    if lm is not None:
        s["const_opt"] = {"lm_calls": lm}
    return s


def _write(root: Path, summaries: list[dict]) -> str:
    d = root / "r"
    for s in summaries:
        rd = d / "runs" / f"{s['task']}__{s['arm']}__s{s['seed']}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "summary.json").write_text(json.dumps(s))
    return str(d)


def _run(tmp_path, spec, d, stage=1):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / f"out{stage}"
    rc = phase_report.main(["--stage", str(stage), str(sp), str(out), d])
    assert rc == 0
    rows = {r["contrast"]: r for r in csv.DictReader((out / "contrasts.csv").open())}
    return json.loads((out / "verdict.json").read_text()), rows


def _spec(metric, **kw):
    return dict({"phase": "t", "metrics": {"Con": "test_nmse"},
                 "primary": [["Con", "ab", "A", "B", metric]]}, **kw)


@pytest.mark.parametrize("metric", ["train_test_gap", "cpu_s", "lm_calls"])
def test_new_metrics_are_registered(metric):
    assert metric in phase_report.METRICS


def test_train_test_gap_is_test_minus_train(tmp_path):
    # A: gap 0.01, B: gap 0.2 + i/100; A lower on every split.
    d = _write(tmp_path, [_reg("A", k, nmse=0.31, train=0.3) for k in SPLITS]
               + [_reg("B", k, nmse=0.3 + 0.2 + k / 100, train=0.3) for k in SPLITS])
    v, rows = _run(tmp_path, _spec("train_test_gap"), d)
    assert v == {"Con:ab": "A better"}
    assert float(rows["ab"]["median_a"]) == pytest.approx(0.01)
    assert rows["ab"]["metric"] == "train_test_gap"


def test_cpu_s_is_the_summary_cpu(tmp_path):
    # a seeded summary's cpu_s already holds GP-GOMEA's CPU (moses_sr_runner.py)
    d = _write(tmp_path, [_reg("A", k, cpu=100.0 + k) for k in SPLITS]
               + [_reg("B", k, cpu=10.0 + k) for k in SPLITS])
    v, rows = _run(tmp_path, _spec("cpu_s"), d)
    assert v == {"Con:ab": "A worse"}
    assert float(rows["ab"]["median_a"]) == pytest.approx(105.5)


def test_lm_calls_come_from_const_opt(tmp_path):
    d = _write(tmp_path, [_reg("A", k, lm=50 + k) for k in SPLITS]
               + [_reg("B", k, lm=500 + k) for k in SPLITS])
    v, rows = _run(tmp_path, _spec("lm_calls"), d)
    assert v == {"Con:ab": "A better"}
    assert float(rows["ab"]["median_b"]) == pytest.approx(505.5)


@pytest.mark.parametrize("metric", ["train_test_gap", "cpu_s", "lm_calls"])
def test_a_failed_run_is_infinite_for_the_new_metrics(tmp_path, metric):
    d = _write(tmp_path, [_reg("A", k, lm=5) for k in SPLITS]
               + [_reg("B", k, lm=5, status="error") for k in SPLITS])
    v, rows = _run(tmp_path, _spec(metric), d)
    assert math.isinf(float(rows["ab"]["median_b"])) and rows["ab"]["a_lower"] == "10"


def test_lm_calls_missing_is_infinite(tmp_path):
    d = _write(tmp_path, [_reg("A", k, lm=5) for k in SPLITS] + [_reg("B", k) for k in SPLITS])
    _, rows = _run(tmp_path, _spec("lm_calls"), d)
    assert math.isinf(float(rows["ab"]["median_b"]))


# 6 of 10 lower by a lot, 4 higher by a little: raw p = 0.066 (test_phase_report.py)
A_EXT = [1.0] * 6 + [12.0, 14.0, 16.0, 18.0]
B_EXT = [10.0] * 6 + [11.0, 12.0, 13.0, 14.0]


def _ext_dir(tmp_path):
    return _write(tmp_path, [_reg("A", k, nmse=x) for k, x in zip(SPLITS, A_EXT)]
                  + [_reg("B", k, nmse=x) for k, x in zip(SPLITS, B_EXT)])


def test_default_spec_still_extends(tmp_path):
    v, rows = _run(tmp_path, _spec("test_nmse"), _ext_dir(tmp_path))
    assert phase_report.ALPHA <= float(rows["ab"]["p_holm"]) < phase_report.EXTEND_BELOW
    assert v == {"Con:ab": "extend"}


def test_extend_false_gives_a_tie(tmp_path):
    v, rows = _run(tmp_path, _spec("test_nmse", extend=False), _ext_dir(tmp_path))
    assert phase_report.ALPHA <= float(rows["ab"]["p_holm"]) < phase_report.EXTEND_BELOW
    assert v == {"Con:ab": "tie"}


def test_extend_false_keeps_better(tmp_path):
    d = _write(tmp_path, [_reg("A", k, nmse=0.1 + k / 100) for k in SPLITS]
               + [_reg("B", k, nmse=0.5 + k / 100) for k in SPLITS])
    v, _ = _run(tmp_path, _spec("test_nmse", extend=False), d)
    assert v == {"Con:ab": "A better"}
