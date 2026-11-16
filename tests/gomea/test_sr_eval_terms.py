"""Server round 12 (T2.2): the report side of `--linear-scaling terms`.

`tools/gomea/sr_eval.py` refits the per-term weights on the training rows
(`evaluate(..., terms=True)`, CLI `--terms`) with the split, the solver and the fallback
rule of the scorer, and `tools/gomea/moses_sr_runner.py` uses that refit for an arm whose
argv has `--linear-scaling terms`.
"""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.moses.scoring.bscores import contin_bscore

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner  # noqa: E402
import sr_eval  # noqa: E402

TRAIN = REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"

PROGRAMS = [
    "+(sin(*(1.3 $1)) *(0.5 $2))",  # fitted
    "+(sin($1) $2 *($1 $2))",  # fitted, 3 terms
    "+(/(1 +(4 $1)) sin($2) exp(*(0.1 $1)))",  # fitted, 3 terms
    "+($1 *(2 $1))",  # collinear
    "+(sin($1) $2 0.25)",  # a numeric leaf
    "*(2.5 sin(*(1.3 $1)))",  # one term
    "$2",  # one term
]


def _splits(dst: Path, f=None) -> Path:
    """train/val/test CSVs from the test table (row i: train if i % 5 < 3, val if 3, test if
    4); `f(x1, x2)` replaces the target when given."""
    dst.mkdir(parents=True, exist_ok=True)
    lines = TRAIN.read_text().splitlines()
    head, rows = lines[0], lines[1:]
    if f is not None:
        rows = [f"{a},{b},{f(float(a), float(b))!r}" for a, b, _ in (r.split(",") for r in rows)]
    for name, keep in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
        body = [r for i, r in enumerate(rows) if i % 5 in keep]
        (dst / f"{name}.csv").write_text("\n".join([head, *body]) + "\n")
    return dst


@pytest.mark.parametrize("prog", PROGRAMS)
def test_sr_eval_and_the_scorer_fit_the_same_weights(prog, tmp_path):
    d = _splits(tmp_path)
    scorer = contin_bscore(loadTable(str(d / "train.csv"), "y"))
    scorer.set_linear_scaling_terms(True)
    k, fit = scorer.term_fit(str_to_combo_tree(prog))
    train = sr_eval.read_table(str(d / "train.csv"))
    res = sr_eval.evaluate(prog, train, terms=True)
    rec = res["terms"]
    assert rec["n_terms"] == k
    assert rec["fallback"] == (fit is not None and fit.weights is None)
    assert rec["reason"] == (None if fit is None else fit.reason)
    if fit is not None and fit.weights is not None:
        assert rec["weights"] == list(fit.weights)  # exactly the same numbers
    else:
        assert rec["weights"] == [res["a"], res["b"]]
    # the training NMSE of the report is the scorer's error (narrowed to float32 there)
    bs = scorer(str_to_combo_tree(prog))
    n = len(train.y)
    mean = math.fsum(train.y) / n
    var = math.fsum((y - mean) ** 2 for y in train.y) / n
    assert 100.0 * -math.fsum(bs) / n / var == pytest.approx(
        res["train"]["nmse_scaled"], rel=1e-5
    )


def test_a_known_combination_has_nmse_about_zero_on_every_split(tmp_path):
    d = _splits(tmp_path, lambda a, b: 1.0 + 2.0 * math.sin(a) - 0.5 * b * b)
    train, test = (sr_eval.read_table(str(d / f"{s}.csv")) for s in ("train", "test"))
    prog = "+(sin($1) *($2 $2))"
    plain = sr_eval.evaluate(prog, train, test)
    res = sr_eval.evaluate(prog, train, test, terms=True)
    assert plain["train"]["nmse_scaled"] > 1.0 and plain["test"]["nmse_scaled"] > 1.0
    assert res["train"]["nmse_scaled"] < 1e-20
    assert res["test"]["nmse_scaled"] < 1e-20  # the training weights, used on test
    assert res["terms"]["weights"] == pytest.approx([1.0, 2.0, -0.5], abs=1e-9)
    assert res["terms"]["fallback"] is False
    # everything else is the default result
    for k in ("program", "a", "b"):
        assert res[k] == plain[k]
    for split in ("train", "test"):
        assert res[split]["nmse_raw"] == plain[split]["nmse_raw"]


@pytest.mark.parametrize(
    ("prog", "n_terms", "reason"),
    [
        ("+($1 *(2 $1))", 2, "singular"),
        ("+(sin($1) $2 0.25)", 3, "constant"),
        ("+(log($1) $2)", 2, "nonfinite"),  # nan on the rows where log overflows
        ("*(2.5 sin($1))", 1, None),
    ],
)
def test_one_term_or_a_fallback_keeps_the_default_nmse(prog, n_terms, reason, tmp_path):
    d = _splits(tmp_path)
    train, test = (sr_eval.read_table(str(d / f"{s}.csv")) for s in ("train", "test"))
    plain = sr_eval.evaluate(prog, train, test)
    res = sr_eval.evaluate(prog, train, test, terms=True)
    rec = res.pop("terms")
    assert json.dumps(res) == json.dumps(plain)
    assert rec["n_terms"] == n_terms and rec["reason"] == reason
    assert rec["fallback"] is (reason is not None)
    assert json.dumps(rec["weights"]) == json.dumps([plain["a"], plain["b"]])


def test_cli_terms_flag(tmp_path):
    d = _splits(tmp_path)
    argv = ["--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
            "--program", "+(sin($1) $2 *($1 $2))"]  # fmt: skip
    outs = []
    for extra in ([], ["--terms"]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert sr_eval.main([*argv, *extra]) == 0
        outs.append(json.loads(buf.getvalue()))
    assert "terms" not in outs[0]
    assert outs[1]["terms"]["n_terms"] == 3 and len(outs[1]["terms"]["weights"]) == 4
    assert outs[1]["train"]["nmse_scaled"] < outs[0]["train"]["nmse_scaled"]


# ------------------------------------------------------------ the runner


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        ([], "0"),
        (["-a", "gomea", "--linear-scaling", "1"], "1"),
        (["--linear-scaling", "terms", "--const-opt", "none"], "terms"),
        (["--linear-scaling=terms"], "terms"),
        (["--linear-scaling", "1", "--linear-scaling", "terms"], "terms"),
    ],
)
def test_runner_reads_the_linear_scaling_value(argv, want):
    assert moses_sr_runner._linear_scaling(argv) == want


_CANDIDATES = "".join(f"-{i + 10} {p} \n" for i, p in enumerate(PROGRAMS)) + "n_evals: 99\n"


def _run_job(work: Path, argv: list[str]) -> dict:
    split_dir = _splits(work / "splits" / "s1")
    fake = work / "fake_moses.sh"
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n" + _CANDIDATES + "EOF\n")
    fake.chmod(0o755)
    job = {
        "run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1,
        "binary": str(fake), "argv": argv, "budget": {"evals": 99}, "seed_spec": None,
        "split_dir": str(split_dir), "wall_cap_s": 60, "mem_limit_mb": None,
        "run_dir": str(work / "run"),
    }  # fmt: skip
    return moses_sr_runner._run_one(job)


def test_runner_scores_a_terms_arm_with_the_refit(tmp_path):
    argv = ["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "none"]
    summary = _run_job(tmp_path, argv)
    assert summary["status"] == "ok" and "const_opt" not in summary
    cands = json.loads((tmp_path / "run" / "candidates.json").read_text())
    d = tmp_path / "splits" / "s1"
    train, val, test = (sr_eval.read_table(str(d / f"{s}.csv")) for s in ("train", "val", "test"))
    for c in cands:
        on_val = sr_eval.evaluate(c["program"], train, val, terms=True)
        on_test = sr_eval.evaluate(c["program"], train, test, terms=True)
        assert c["terms"] == on_val["terms"]
        assert c["val_nmse"] == on_val["test"]["nmse_scaled"]
        assert c["test_nmse"] == on_test["test"]["nmse_scaled"]
    best = min(cands, key=lambda c: c["val_nmse"])
    assert summary["final_program"] == best["program"]
    assert summary["terms"] == best["terms"]
    assert set(summary["terms"]) == {"n_terms", "fallback", "reason", "weights"}
    # the per-term refit changes what the runner selects or reports on this list
    plain = _run_job(tmp_path / "plain", ["-a", "gomea", "--linear-scaling", "1"])
    assert "terms" not in plain
    assert (plain["final_program"], plain["val_nmse"]) != (
        summary["final_program"],
        summary["val_nmse"],
    )
