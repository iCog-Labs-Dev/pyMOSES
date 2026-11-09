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


# ---- the spec key "secondary_families" (1bc: N16-C250 vs N16 is its own Holm family)

D6 = ["Con", "Tow", "Yac", "WiR", "Bos", "EnC"]


def _bc_spec(**kw):
    def c(name, a, b, metric):
        return [[d, name, a, b, metric] for d in D6]
    return dict({"phase": "t", "metrics": dict.fromkeys(D6, "test_nmse"),
                 "primary": c("c250-vs-base", "BASE-C250", "BASE", "test_nmse"),
                 "secondary": c("n16c250-vs-n16", "N16-C250", "N16", "test_nmse")
                 + c("c250-vs-base-size", "BASE-C250", "BASE", "program_size")
                 + c("c250-vs-base-lm", "BASE-C250", "BASE", "lm_calls"),
                 "extend": False}, **kw)


def _bc_dir(tmp_path):
    """N16-C250 has a 50 % lower test NMSE than N16 on every split of every dataset (raw p
    2/1024). BASE-C250 and BASE are equal in NMSE and size, and BASE-C250 has fewer LM calls
    on 6 of 10 splits and more on 4 (a large raw p)."""
    out = []
    for d in D6:
        for k in SPLITS:
            out += [_reg("N16-C250", k, nmse=0.1 + k / 100, task=d),
                    _reg("N16", k, nmse=0.3 + k / 100, task=d),
                    _reg("BASE", k, nmse=0.2, lm=100, task=d),
                    _reg("BASE-C250", k, nmse=0.2, lm=90 if k <= 6 else 120, task=d)]
    return _write(tmp_path, out)


def _bc_run(tmp_path, spec):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(sp), str(out), _bc_dir(tmp_path)]) == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    return json.loads((out / "verdict.json").read_text()), rows


def test_default_keeps_one_secondary_family_of_18(tmp_path):
    v, rows = _bc_run(tmp_path, _bc_spec())
    fam = [r for r in rows if r["contrast"] == "n16c250-vs-n16"]
    assert len(fam) == 6
    for r in fam:
        assert float(r["p"]) == pytest.approx(2 / 1024)
        assert float(r["p_holm"]) == pytest.approx(18 * 2 / 1024)  # 0.035: no verdict
        assert v[f"{r['problem']}:n16c250-vs-n16"] == "tie"
    assert "holm_family" not in rows[0]


def test_own_family_of_6_gives_a_verdict(tmp_path):
    v, rows = _bc_run(tmp_path, _bc_spec(secondary_families=[["n16c250-vs-n16"]]))
    fam = [r for r in rows if r["contrast"] == "n16c250-vs-n16"]
    assert len(fam) == 6
    for r in fam:
        assert float(r["p_holm"]) == pytest.approx(6 * 2 / 1024)  # 0.0117 < ALPHA
        assert float(r["p_holm"]) < phase_report.ALPHA
        assert r["family"] == "secondary" and r["holm_family"] == "secondary/n16c250-vs-n16"
        assert v[f"{r['problem']}:n16c250-vs-n16"] == "A better"


def test_the_other_12_secondary_contrasts_form_the_second_family(tmp_path):
    _, rows = _bc_run(tmp_path, _bc_spec(secondary_families=[["n16c250-vs-n16"]]))
    rest = [r for r in rows if r["family"] == "secondary" and r["contrast"] != "n16c250-vs-n16"]
    assert len(rest) == 12 and {r["holm_family"] for r in rest} == {"secondary"}
    want = phase_report.holm([float(r["p"]) for r in rest])
    assert [float(r["p_holm"]) for r in rest] == pytest.approx(want)
    prim = [r for r in rows if r["family"] == "primary"]
    assert len(prim) == 6
    assert [float(r["p_holm"]) for r in prim] == pytest.approx(
        phase_report.holm([float(r["p"]) for r in prim]))


def test_the_row_order_and_the_verdict_keys_do_not_change(tmp_path):
    v0, r0 = _bc_run(tmp_path, _bc_spec())
    v1, r1 = _bc_run(tmp_path, _bc_spec(secondary_families=[["n16c250-vs-n16"]]))
    assert [(r["problem"], r["contrast"]) for r in r0] == [(r["problem"], r["contrast"]) for r in r1]
    assert list(v0) == list(v1)


@pytest.mark.parametrize("bad", [[["nope"]], [["c250-vs-base"]],
                                 [["n16c250-vs-n16"], ["n16c250-vs-n16"]], ["n16c250-vs-n16"]])
def test_bad_secondary_families_are_refused(tmp_path, bad):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(_bc_spec(secondary_families=bad)))
    assert phase_report.main(["--stage", "1", str(sp), str(tmp_path / "o"),
                              _bc_dir(tmp_path)]) == 2
