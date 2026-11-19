"""Server round 14: the linear baselines LIN-OLS and LIN-RIDGE in the runner
(``SERVER-ROUND14-PREREGISTRATION.md``, "LIN-OLS and LIN-RIDGE: linear baselines").

* ``sr_linear.py``: OLS by ``least_squares.fit_terms`` with the fallback for aliased inputs, and
  ridge on standardized inputs (intercept not penalized) for each lambda of the grid.
* ``moses_sr_runner.py``: an arm ``{"linear": "ols"}`` or ``{"linear": "ridge"}`` runs no
  MOSES process. The runner fits the model to the training target scaled as for the MOSES arms,
  scores it with the NMSE functions of ``sr_eval.py``, keeps the ridge lambda with the lowest
  validation NMSE (a tie keeps the larger lambda), and writes the summary (``linear`` and
  ``terms`` records, ``final_program`` ``+($x1 ... $xk)``, ``n_evals`` null).
* ``expand()`` refuses a wrong value and a linear arm with an argv (or any other key).
* ``phase_report.py`` and ``run_table.py`` read the summaries; the size of a linear model is that
  of the TLS program ``+($x1 ... $xk)`` with k terms.
* Byte identity against ad0c8d8 (the commit before this change, from git): the runner output of
  a plain, a TLS and an HC arm, and the jobs of every committed regression manifest, do not
  change.
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
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402
import run_table  # noqa: E402
import sr_eval  # noqa: E402
import sr_linear  # noqa: E402

OLD = "ad0c8d8"  # the commit before the linear baselines
VOLATILE = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")


def _write(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([",".join(header), *(",".join(map(str, r)) for r in rows)]) + "\n")


def _split(tmp: Path, train, val, test, header=("x1", "x2", "y")) -> Path:
    d = tmp / "splits" / "s1"
    for name, rows in (("train", train), ("val", val), ("test", test)):
        _write(d / f"{name}.csv", header, rows)
    return d


def _job(tmp: Path, method: str, split_dir: Path, **kw) -> dict:
    return {"run_id": "T__LIN__s1", "task": "T", "arm": "LIN", "seed": 1, "linear": method,
            "split_dir": str(split_dir), "wall_cap_s": 60, "mem_limit_mb": None,
            "run_dir": str(tmp / "run"), **kw}


def _plane(xs, w=(3.0, 2.0, -0.5), noise=None):
    """Rows (x1, x2, y) of y = w0 + w1 x1 + w2 x2 (+ noise[i])."""
    out = []
    for i, (a, b) in enumerate(xs):
        y = w[0] + w[1] * a + w[2] * b + (noise[i] if noise else 0.0)
        out.append((a, b, repr(y)))
    return out


GRID = [(float(a), float(b)) for a in range(-3, 4) for b in (0, 1, 3, 4)]
NOISE = [((7 * i) % 5 - 2) * 0.1 for i in range(len(GRID))]


def _no_process(monkeypatch):
    """Make every attempt to start a process fail the test."""
    def boom(*a, **k):
        raise AssertionError("a linear job started a process")
    monkeypatch.setattr(runner.subprocess, "run", boom)
    monkeypatch.setattr(runner.subprocess, "Popen", boom)


# ---- sr_linear: OLS and ridge on tables with known solutions


def test_ols_recovers_an_exact_plane():
    rows = _plane(GRID)
    cols = sr_linear.columns([[r[0], r[1]] for r in rows])
    w, rec = sr_linear.fit_ols(cols, [float(r[2]) for r in rows])
    assert rec == {"fallback": None, "fit_reason": None, "dropped": []}
    assert w == pytest.approx([3.0, 2.0, -0.5], abs=1e-12)


def test_ols_of_one_input_is_the_textbook_line():
    # x = 0..3, y = 1, 3, 2, 5: slope Sxy / Sxx = 5.5 / 5 = 1.1, intercept 2.75 - 1.65 = 1.1
    w, _ = sr_linear.fit_ols([[0.0, 1.0, 2.0, 3.0]], [1.0, 3.0, 2.0, 5.0])
    assert w == pytest.approx([1.1, 1.1], abs=1e-12)


def test_ridge_of_one_input_has_the_closed_form():
    # one input, population sd s (s^2 = 1.25), z = (x - 1.5) / s: beta = (Sxy / s) / (n + lam),
    # so w1 = beta / s = 5.5 / (1.25 (4 + lam)) = 4.4 / (4 + lam); the intercept is not penalized
    x, y = [0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 5.0]
    fits, constant = sr_linear.fit_ridge([x], y, lambdas=(0.0, 1.0, 4.0))
    assert constant == []
    for lam, w in zip((0.0, 1.0, 4.0), fits, strict=True):
        w1 = 4.4 / (4.0 + lam)
        assert w == pytest.approx([2.75 - 1.5 * w1, w1], abs=1e-12)


def test_ridge_with_a_tiny_lambda_is_ols_and_shrinks_with_a_large_one():
    rows = _plane(GRID, noise=NOISE)
    cols = sr_linear.columns([[r[0], r[1]] for r in rows])
    y = [float(r[2]) for r in rows]
    ols, _ = sr_linear.fit_ols(cols, y)
    fits, _ = sr_linear.fit_ridge(cols, y)
    assert fits[0] == pytest.approx(ols, abs=1e-6)  # lambda 1e-6
    norms = [math.hypot(*w[1:]) for w in fits]
    assert all(a >= b for a, b in zip(norms, norms[1:], strict=False))  # more shrinkage
    assert norms[-1] < norms[0]


def test_the_grid_is_ten_to_the_minus_6_up_to_ten_to_the_3():
    assert sr_linear.RIDGE_LAMBDAS == tuple(10.0**k for k in range(-6, 4))
    assert len(sr_linear.RIDGE_LAMBDAS) == 10
    assert sr_linear.METHODS == ("ols", "ridge")


def test_to_target_units_undoes_the_target_scaling():
    mean, sd = 4.0, 2.5
    w = sr_linear.to_target_units([1.0, 2.0, -3.0], mean, sd)
    for x1, x2 in ((0.0, 0.0), (1.5, -2.0)):
        y_scaled = 1.0 + 2.0 * x1 - 3.0 * x2
        assert w[0] + w[1] * x1 + w[2] * x2 == pytest.approx(mean + sd * y_scaled / 10.0)


# ---- the singular table: the OLS fallback


def test_a_singular_table_drops_the_aliased_and_the_constant_inputs():
    """x2 = 2 x1 (collinear) and x3 = 5 (constant): fit_terms gives no weights; the fallback drops
    x2 and x3 (weight 0) and fits x1 alone, so its predictions are those of the fit on x1."""
    x1 = [float(i) for i in range(10)]
    y = [1.0 + 0.7 * a + (0.3 if i % 3 == 0 else -0.1) for i, a in enumerate(x1)]
    cols = [x1, [2.0 * a for a in x1], [5.0] * 10]
    w, rec = sr_linear.fit_ols(cols, y)
    assert rec["fallback"] == "aliased" and rec["dropped"] == [1, 2]
    assert rec["fit_reason"] in ("singular", "constant")
    alone, _ = sr_linear.fit_ols([x1], y)
    assert w == pytest.approx([alone[0], alone[1], 0.0, 0.0], abs=1e-12)
    assert sr_linear.independent_columns(cols) == [0]
    # only the collinear pair: fit_terms says "singular"
    w2, rec2 = sr_linear.fit_ols(cols[:2], y)
    assert rec2 == {"fallback": "aliased", "fit_reason": "singular", "dropped": [1]}
    assert w2 == pytest.approx([alone[0], alone[1], 0.0], abs=1e-12)


def test_ridge_gives_a_constant_input_the_weight_0():
    x1 = [float(i) for i in range(10)]
    fits, constant = sr_linear.fit_ridge([x1, [5.0] * 10], [2.0 * a for a in x1])
    assert constant == [1]
    assert all(w is not None and w[2] == 0.0 for w in fits)


def test_the_runner_records_the_ols_fallback(tmp_path, monkeypatch):
    _no_process(monkeypatch)
    rows = [(float(i), 2.0 * i, repr(1.0 + 0.7 * i + (0.3 if i % 3 == 0 else -0.1)))
            for i in range(12)]
    d = _split(tmp_path, rows, rows[:4], rows[4:8])
    s = runner._run_one(_job(tmp_path, "ols", d))
    assert s["status"] == "ok"
    li = s["linear"]
    assert li["fallback"] == "aliased" and li["fit_reason"] == "singular"
    assert li["dropped_inputs"] == ["x2"] and li["weights"][2] == 0.0
    assert s["terms"] == {"n_terms": 2, "fallback": False, "reason": None,
                          "weights": li["weights"]}


# ---- the runner: the summary of a linear job


def _tables(d: Path):
    return [sr_eval.read_table(str(d / f"{p}.csv")) for p in ("train", "val", "test")]


@pytest.mark.parametrize("method", ["ols", "ridge"])
@pytest.mark.parametrize("clip", [False, True])
def test_the_summary_of_a_linear_job(method, clip, tmp_path, monkeypatch):
    _no_process(monkeypatch)
    rows = _plane(GRID, noise=NOISE)
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    kw = {"clip_test_nmse": True} if clip else {}
    s = runner._run_one(_job(tmp_path, method, d, plain_nmse=True, **kw))
    disk = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert disk == s
    assert s["status"] == "ok" and s["error"] is None and s["n_evals"] is None
    assert s["exit_code"] is None and s["cpu_s"] >= 0 and s["wall_s"] >= 0
    assert s["ru_maxrss_kb"] > 0 and "argv" not in s and "const_opt" not in s
    assert s["final_program"] == "+($x1 $x2)"
    assert s["n_candidates"] == 1 and s["n_candidates_unscorable"] == 0
    li = s["linear"]
    assert li["method"] == method and li["inputs"] == ["x1", "x2"] and li["bool_inputs"] == []
    assert s["terms"] == {"n_terms": 2, "fallback": False, "reason": None,
                          "weights": li["weights"]}
    assert "plain_val_nmse" not in s  # plain_nmse is for --linear-scaling terms arms only
    train, val, test = _tables(d)
    w = li["weights"]
    pred = {n: [w[0] + w[1] * r[0] + w[2] * r[1] for r in t.X]
            for n, t in (("train", train), ("val", val), ("test", test))}
    for name, t in (("train", train), ("val", val), ("test", test)):
        assert s[f"{name}_nmse"] == pytest.approx(sr_eval.nmse(t.y, pred[name]), rel=1e-12)
    assert s["first_train_nmse"] == s["train_nmse"] and s["first_test_nmse"] == s["test_nmse"]
    if clip:
        lo, hi = sr_eval.clip_range(train.y)
        assert s["test_nmse_clipped"] == pytest.approx(
            sr_eval.nmse_clipped(test.y, pred["test"], lo, hi), rel=1e-12)
    else:
        assert "test_nmse_clipped" not in s
    if method == "ols":
        assert li["fallback"] is None and li["dropped_inputs"] == [] and "lambda" not in li
        assert w == pytest.approx(sr_linear.to_target_units(
            sr_linear.fit_ols(sr_linear.columns(train.X),
                              [10 * (t - s["target_mean"]) / s["target_sd"] for t in train.y])[0],
            s["target_mean"], s["target_sd"]), rel=1e-12)
    else:
        assert li["lambda"] in sr_linear.RIDGE_LAMBDAS and li["constant_inputs"] == []
        assert li["lambdas"] == list(sr_linear.RIDGE_LAMBDAS)
        assert len(li["val_nmse_by_lambda"]) == 10
        assert li["val_nmse_by_lambda"][sr_linear.RIDGE_LAMBDAS.index(li["lambda"])] == \
            pytest.approx(s["val_nmse"], rel=1e-12)
        assert "fallback" not in li
    cands = json.loads((tmp_path / "run" / "candidates.json").read_text())
    assert cands == [{"program": "+($x1 $x2)", "train_nmse": s["train_nmse"],
                      "val_nmse": s["val_nmse"], "test_nmse": s["test_nmse"]}]
    assert (tmp_path / "run" / "train_scaled.csv").exists()
    assert not (tmp_path / "run" / "stdout.txt").exists()


def test_the_target_scaling_does_not_change_the_model(tmp_path, monkeypatch):
    """The fit uses y' = 10 (y - mean) / sd; the weights in target units are those of a fit
    on y itself (OLS and ridge are equivariant under an affine map of the target)."""
    _no_process(monkeypatch)
    rows = _plane(GRID, noise=NOISE)
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    train = sr_eval.read_table(str(d / "train.csv"))
    cols = sr_linear.columns(train.X)
    ols = runner._run_one(_job(tmp_path / "o", "ols", d))
    assert ols["linear"]["weights"] == pytest.approx(sr_linear.fit_ols(cols, train.y)[0],
                                                     rel=1e-9)
    ridge = runner._run_one(_job(tmp_path / "r", "ridge", d))
    i = sr_linear.RIDGE_LAMBDAS.index(ridge["linear"]["lambda"])
    assert ridge["linear"]["weights"] == pytest.approx(
        sr_linear.fit_ridge(cols, train.y)[0][i], rel=1e-9)


def test_a_boolean_input_counts_as_0_or_1(tmp_path, monkeypatch):
    """x2 holds only 0 and 1 (MOSES types it as boolean, as Bos x4): the model uses it as the
    number 0 or 1."""
    _no_process(monkeypatch)
    xs = [(float(a), float(b)) for a in range(-3, 4) for b in (0, 1)] * 2
    rows = _plane(xs, w=(1.0, 0.5, 2.0), noise=[((5 * i) % 3 - 1) * 0.05 for i in range(len(xs))])
    rows = [(a, int(b), y) for a, b, y in rows]
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    train = sr_eval.read_table(str(d / "train.csv"))
    assert train.bool_cols == frozenset({1})
    s = runner._run_one(_job(tmp_path, "ols", d))
    assert s["status"] == "ok" and s["linear"]["bool_inputs"] == ["x2"]
    ref = sr_linear.fit_ols(sr_linear.columns(train.X), train.y)[0]
    assert s["linear"]["weights"] == pytest.approx(ref, rel=1e-9)


# ---- the lambda choice


def test_the_lambda_with_the_lowest_validation_nmse_and_a_tie_keeps_the_larger(monkeypatch):
    """Made-up validation NMSE values for the 10 lambdas: the lowest value 2.0 is at 10^-3 and
    10^-2; the larger lambda, 10^-2, wins."""
    rows = _plane(GRID, noise=NOISE)
    train = sr_eval.Table(["x1", "x2"], [[r[0], r[1]] for r in rows], [float(r[2]) for r in rows])
    vals = iter([5.0, 4.0, 3.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    monkeypatch.setattr(runner.sr_eval, "nmse", lambda y, p: next(vals))
    w, rec = runner._fit_linear("ridge", train, train, 0.0, 10.0)
    assert rec["lambda"] == 1e-2
    assert rec["val_nmse_by_lambda"] == [5.0, 4.0, 3.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    fits, _ = sr_linear.fit_ridge(sr_linear.columns(train.X), train.y)
    assert w == pytest.approx(fits[4], rel=1e-12)  # y' = y with mean 0 and sd 10


def test_a_validation_set_at_the_training_mean_picks_the_largest_lambda(tmp_path):
    """Validation targets at the training mean: the more shrinkage, the lower the validation
    NMSE, so lambda = 10^3."""
    rows = _plane(GRID, noise=NOISE)
    train = rows[:20]
    my = sum(float(r[2]) for r in train) / len(train)
    val = [(a, b, repr(my + (0.01 if i % 2 else -0.01))) for i, (a, b, _) in enumerate(rows[20:])]
    d = _split(tmp_path, train, val, rows[20:])
    s = runner._run_one(_job(tmp_path, "ridge", d))
    assert s["linear"]["lambda"] == 1e3
    v = s["linear"]["val_nmse_by_lambda"]
    assert v[-1] == min(v)


def test_a_noise_free_plane_picks_the_smallest_lambda(tmp_path):
    rows = _plane(GRID)
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    s = runner._run_one(_job(tmp_path, "ridge", d))
    assert s["linear"]["lambda"] == 1e-6


def test_equal_validation_nmse_for_every_lambda_picks_the_largest(tmp_path):
    """A constant input: every lambda gives the mean, the same validation NMSE; the tie keeps
    the largest lambda."""
    rows = [(1.0, repr(float(i % 4))) for i in range(12)]
    d = _split(tmp_path, rows, rows[:5], rows[5:], header=("x1", "y"))
    s = runner._run_one(_job(tmp_path, "ridge", d))
    assert s["status"] == "ok" and s["linear"]["constant_inputs"] == ["x1"]
    assert len(set(s["linear"]["val_nmse_by_lambda"])) == 1
    assert s["linear"]["lambda"] == 1e3


def test_a_failed_fit_gives_status_error(tmp_path, monkeypatch):
    _no_process(monkeypatch)
    rows = _plane(GRID)
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    _write(d / "val.csv", ("x1", "x3", "y"), rows[16:22])  # other feature columns
    s = runner._run_one(_job(tmp_path, "ols", d, clip_test_nmse=True))
    assert s["status"] == "error" and "feature columns differ" in s["error"]
    assert s["test_nmse"] is None and s["test_nmse_clipped"] is None and s["n_evals"] is None
    assert json.loads((tmp_path / "run" / "candidates.json").read_text()) == []


def test_an_existing_summary_is_kept(tmp_path):
    rows = _plane(GRID)
    d = _split(tmp_path, rows[:16], rows[16:22], rows[22:])
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "summary.json").write_text('{"kept": true}')
    assert runner._run_one(_job(tmp_path, "ols", d)) == {"kept": True}


# ---- expand(): the manifest shape and the refusals


def _manifest(tmp_path, arms, **kw):
    return {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con", "Dow"],
            "splits": [61, 62], "wall_cap_s": 10, "mem_limit_mb": 5120, "arms": arms, **kw}


def test_expand_gives_a_linear_job_no_binary_no_argv_and_no_budget(tmp_path):
    moses = {"binary": "/x/.venv/bin/moses", "argv": ["-a", "hc"], "budget": {"evals": 10}}
    m = _manifest(tmp_path, {"HC": moses, "LIN-OLS": {"linear": "ols"},
                             "LIN-RIDGE": {"linear": "ridge"}},
                  clip_test_nmse=True, plain_nmse=True)
    jobs = runner.expand(m, tmp_path / "o", tmp_path)
    assert len(jobs) == 12
    lin = [j for j in jobs if j["arm"].startswith("LIN")]
    assert len(lin) == 8
    for j in lin:
        assert set(j) == {"run_id", "task", "arm", "seed", "linear", "split_dir", "wall_cap_s",
                          "mem_limit_mb", "run_dir", "clip_test_nmse", "plain_nmse"}
        assert j["linear"] == {"LIN-OLS": "ols", "LIN-RIDGE": "ridge"}[j["arm"]]
        assert j["run_id"] == f"{j['task']}__{j['arm']}__s{j['seed']}"
        assert j["split_dir"] == str(tmp_path / j["task"] / "splits" / f"s{j['seed']}")
        assert j["run_dir"] == str(tmp_path / "o" / "t" / "runs" / j["run_id"])
        assert j["mem_limit_mb"] == 5120 and j["wall_cap_s"] == 10
        with pytest.raises(KeyError):
            runner.moses_argv(j, tmp_path, 100)  # no MOSES command line
    off = runner.expand({**m, "clip_test_nmse": False, "plain_nmse": False}, tmp_path, tmp_path)
    assert all("clip_test_nmse" not in j and "plain_nmse" not in j for j in off)


@pytest.mark.parametrize("arm, why", [
    ({"linear": "lasso"}, 'linear must be "ols" or "ridge"'),
    ({"linear": "OLS"}, 'linear must be "ols" or "ridge"'),
    ({"linear": None}, 'linear must be "ols" or "ridge"'),
    ({"linear": True}, 'linear must be "ols" or "ridge"'),
    ({"linear": ""}, 'linear must be "ols" or "ridge"'),
    ({"linear": "ols", "argv": []}, "a linear arm has no argv"),
    ({"linear": "ridge", "argv": ["-a", "hc"]}, "a linear arm has no argv"),
    ({"linear": "ols", "binary": ".venv/bin/moses"}, 'only the key "linear"'),
    ({"linear": "ols", "budget": {"evals": 20000}}, 'only the key "linear"'),
    ({"linear": "ols", "terms_guard": "val"}, 'only the key "linear"'),
    ({"linear": "ols", "seed": {"binary": "x"}}, 'only the key "linear"'),
])
def test_expand_refuses_a_wrong_linear_arm(arm, why, tmp_path):
    moses = {"binary": "/x/.venv/bin/moses", "argv": ["-a", "hc"], "budget": {"evals": 10}}
    with pytest.raises(ValueError, match=why):
        runner.expand(_manifest(tmp_path, {"HC": moses, "X": arm}), tmp_path, tmp_path)


# ---- the report reads the summaries; the size rule


def test_the_size_of_a_linear_model_is_that_of_the_tls_program(tmp_path):
    """`+($x1 ... $xk)` with n_terms k: the report gives it the size of a TLS summary with the
    same program and k terms (node count "(" + atoms = k + 2, plus k terms)."""
    for k in (5, 13, 57):
        prog = "+(" + " ".join(f"$x{i}" for i in range(1, k + 1)) + ")"
        lin = {"final_program": prog, "_size_terms": True,
               "terms": {"n_terms": k, "fallback": False, "reason": None, "weights": [0] * (k + 1)}}
        tls = {"final_program": "+(" + " ".join(f"${i}" for i in range(1, k + 1)) + ")",
               "_size_terms": True,
               "terms": {"n_terms": k, "fallback": False, "reason": None, "weights": [0] * (k + 1)}}
        assert phase_report._nodes(lin) == k + 2
        assert phase_report._size(lin) == phase_report._size(tls) == 2 * k + 2
        assert run_table.terms_size(lin) == k


def _moses_summary(task, arm, split, nmse):
    return {"arm": arm, "task": task, "seed": split, "status": "ok", "test_nmse": nmse,
            "train_nmse": nmse / 2, "val_nmse": nmse, "test_nmse_clipped": nmse, "cpu_s": 10.0,
            "n_evals": 20000, "final_program": "+(*($1 $2) $3)",
            "terms": {"n_terms": 2, "fallback": False, "reason": None, "weights": [0, 1, 1]}}


def test_phase_report_and_run_table_read_the_linear_summaries(tmp_path):
    """Real linear summaries (from the runner) beside made-up TLS summaries: the report of a spec
    with size_terms, run_table and a descriptive pooled contrast against LIN-OLS ends with rc 0;
    runs.csv has the linear rows with the size 2k + 2, n_terms k and no evaluations."""
    root = tmp_path / "res"
    for k in (1, 2, 3):
        rows = _plane(GRID, noise=[x * k for x in NOISE])
        d = _split(tmp_path / f"d{k}", rows[:16], rows[16:22], rows[22:])
        for arm, method in (("LIN-OLS", "ols"), ("LIN-RIDGE", "ridge")):
            j = {**_job(tmp_path, method, d, clip_test_nmse=True), "task": "T", "arm": arm,
                 "seed": k, "run_id": f"T__{arm}__s{k}",
                 "run_dir": str(root / "runs" / f"T__{arm}__s{k}")}
            assert runner._run_one(j)["status"] == "ok"
        rd = root / "runs" / f"T__BASE-TLS__s{k}"
        rd.mkdir(parents=True)
        (rd / "summary.json").write_text(json.dumps(_moses_summary("T", "BASE-TLS", k, 0.5 * k)))
    spec = {"phase": "t14", "metrics": {"T": "test_nmse"}, "primary": [],
            "pooled_primary": [["basetls-vs-linols", "BASE-TLS", "LIN-OLS", "test_nmse"]],
            "pooled_secondary": [["basetls-vs-linridge", "BASE-TLS", "LIN-RIDGE", "test_nmse"],
                                 ["basetls-vs-linols-size", "BASE-TLS", "LIN-OLS",
                                  "program_size"]],
            "pooled_descriptive": ["basetls-vs-linridge"], "secondary": [],
            "secondary_verdicts": False, "extend": False, "size_terms": True, "run_table": True}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(tmp_path / "spec.json"), str(out), str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    assert {r["contrast"]: r["pairs"] for r in rows} == {
        "basetls-vs-linols": "3", "basetls-vs-linridge": "3", "basetls-vs-linols-size": "3"}
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:basetls-vs-linridge"] == "descriptive"
    size = next(r for r in rows if r["contrast"] == "basetls-vs-linols-size")
    assert float(size["median_b"]) == 6.0  # k = 2 inputs: 2 * 2 + 2
    table = list(csv.DictReader((out / "runs.csv").open()))
    lin = [r for r in table if r["arm"].startswith("LIN")]
    assert len(lin) == 6 and len(table) == 9
    for r in lin:
        assert r["status"] == "ok" and r["size"] == "6" and r["nodes"] == "4"
        assert r["n_terms"] == "2" and r["fallback"] == "False" and r["evals"] == ""
        assert r["lm_calls"] == "" and r["demes"] == "" and r["test_nmse_clipped"] != ""
    agg = {r["arm"]: r for r in csv.DictReader((out / "aggregate.csv").open())}
    assert agg["LIN-OLS"]["median_size"] == "6" and agg["LIN-OLS"]["median_nodes"] == "4"
    assert agg["LIN-OLS"]["median_evals_used"] == "" and agg["LIN-OLS"]["not_ok"] == "0"


# ---- byte identity against ad0c8d8 (git)


def _old_tool(tmp_path: Path, names) -> Path:
    """The files `names` of tools/gomea/ at commit OLD, in a directory of their own. A missing
    commit fails the test (no skip)."""
    d = tmp_path / f"old-{OLD}"
    d.mkdir()
    for n in names:
        r = subprocess.run(["git", "show", f"{OLD}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:
            pytest.fail(f"git show {OLD}:tools/gomea/{n} failed: {r.stderr.decode().strip()!r}. "
                        f"The byte-identity tests need the reference commit {OLD}; use a clone "
                        f"with the full history.")
        (d / n).write_bytes(r.stdout)
    return d


TRAIN = REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"
PROGRAMS = ["+(sin(*(1.3 $1)) *(0.5 $2))", "+(sin($1) $2 *($1 $2))", "+($1 *(2 $1))",
            "*(2.5 sin(*(1.3 $1)))", "$2", "+($1 $9)"]
PLAIN = ["-a", "gomea", "--linear-scaling", "1"]
TLS = ["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "none"]
HC = ["-a", "hc", "--linear-scaling", "1"]


def _moses_job(work: Path, argv, **kw) -> dict:
    """A job of a fake MOSES that prints PROGRAMS, on splits of the const_opt_none table."""
    d = work / "splits" / "s1"
    d.mkdir(parents=True)
    lines = TRAIN.read_text().splitlines()
    for name, keep in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
        body = [r for i, r in enumerate(lines[1:]) if i % 5 in keep]
        (d / f"{name}.csv").write_text("\n".join([lines[0], *body]) + "\n")
    out = "".join(f"-{i + 10} {p} \n" for i, p in enumerate(PROGRAMS)) + "n_evals: 99\n"
    fake = work / "fake_moses.sh"
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n" + out + "EOF\n")
    fake.chmod(0o755)
    return {"run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1, "binary": str(fake),
            "argv": argv, "budget": {"evals": 99}, "seed_spec": None, "split_dir": str(d),
            "wall_cap_s": 60, "mem_limit_mb": None, "run_dir": str(work / "run"), **kw}


RUN_ONE = ("import json, sys; sys.path.insert(0, sys.argv[1]); import moses_sr_runner as r; "
           "r._run_one(json.loads(sys.argv[2]))")


def _run_in(tool_dir: Path, job: dict) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run([sys.executable, "-c", RUN_ONE, str(tool_dir), json.dumps(job)], env=env,
                   check=True, cwd=job["run_dir"].rsplit("/", 1)[0])


@pytest.mark.parametrize("argv", [PLAIN, TLS, HC], ids=["plain", "tls", "hc"])
@pytest.mark.parametrize("keys", [{}, {"clip_test_nmse": True, "plain_nmse": True}],
                         ids=["no-keys", "keys"])
def test_the_runner_output_of_an_old_arm_is_byte_identical(argv, keys, tmp_path):
    old = _old_tool(tmp_path, ["moses_sr_runner.py", "sr_eval.py"])
    jo = _moses_job(tmp_path / "o", argv, **keys)
    jn = _moses_job(tmp_path / "n", argv, **keys)
    _run_in(old, jo)
    _run_in(TOOLS, jn)
    co, cn = (Path(j["run_dir"]) / "candidates.json" for j in (jo, jn))
    assert cn.read_bytes() == co.read_bytes()
    so, sn = (json.loads((Path(j["run_dir"]) / "summary.json").read_text()) for j in (jo, jn))
    assert set(sn) == set(so)

    def text(s):
        return json.dumps({k: v for k, v in s.items() if k not in VOLATILE}, indent=2,
                          sort_keys=True)
    assert text(sn) == text(so)


EXPAND = ("import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
          "import moses_sr_runner as r; out = {}\n"
          "for f in sys.argv[3:]:\n"
          "    try:\n"
          "        m = json.loads(Path(f).read_text())\n"
          "        out[f] = r.expand(m, Path('/o'), Path(sys.argv[2]))\n"
          "    except Exception as e:\n"
          "        out[f] = 'error: ' + repr(e)\n"
          "print(json.dumps(out, sort_keys=True))")


def test_expand_gives_the_jobs_of_ad0c8d8_for_every_committed_manifest(tmp_path):
    """Every regression manifest under configs/gomea that has no linear arm (all manifests before
    round 14): the jobs of expand() are those of the runner at ad0c8d8."""
    old = _old_tool(tmp_path, ["moses_sr_runner.py", "sr_eval.py"])
    files = sorted(str(f) for f in (REPO / "configs" / "gomea").glob("*/sr*.json")
                   if not any("linear" in a for a in json.loads(f.read_text())["arms"].values()))
    assert len(files) > 150
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    got = [json.loads(subprocess.run([sys.executable, "-c", EXPAND, str(d), str(REPO), *files],
                                     env=env, check=True, capture_output=True, text=True).stdout)
           for d in (old, TOOLS)]
    assert got[0] == got[1]
    assert sum(isinstance(v, list) for v in got[1].values()) == len(files)
