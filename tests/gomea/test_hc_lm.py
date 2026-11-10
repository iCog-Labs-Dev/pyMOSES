"""Server round 10: LM for hill climbing (``-a hc --const-opt lm``, with or without
``--const-opt-guard val``), as ``docs/gomea/SERVER-ROUND10-PREREGISTRATION.md`` ("Code
change") registers it: at the end of each HC deme, LM on the deme's best program, kept if
the training error is lower (and, with the guard, if the validation NMSE is not worse).
Each LM pass counts as one evaluation against ``-m``. The identity of ``-a hc`` without the
flags is ``test_hc_lm_identity.py``."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

_HC = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
_LM = ["--const-opt", "lm"]
# The training target is y = x1 (x2 + C). Hill climbing finds *(+(1.5 $2) $1) (a knob
# value near C), and LM tunes the constant to C. Linear scaling cannot absorb C.
C = 1.37


def _table(path: Path, c: float) -> Path:
    """40 rows, y = x1 (x2 + c)."""
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x1", "x2", "y"])
        for i in range(40):
            x1, x2 = -3 + 6 * i / 39, ((i * 7) % 11) / 5.0
            wr.writerow([repr(x1), repr(x2), repr(x1 * (x2 + c))])
    return path


def _reg(tmp_path, m="800"):
    """The fixed regression setup of tools/gomea/moses_sr_runner.py on the training table."""
    train = _table(tmp_path / "train.csv", C)
    return ["-i", str(train), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


def _guard(tmp_path, c=C):
    """The guard, with validation rows y = x1 (x2 + c)."""
    return ["--const-opt-guard", "val", "--const-opt-val", str(_table(tmp_path / "v.csv", c))]


def _run(argv, tmp_path):
    """(stdout, events of --gomea-log) of one in-process run."""
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    log.unlink(missing_ok=True)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    events = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return buf.getvalue(), events


def _ends(events):
    return [e for e in events if e["type"] == "end"]


def _n_evals(out):
    return int(out.rsplit("n_evals:", 1)[1])


def _run_error(argv, tmp_path, capsys):
    """The error text of a run that the command line must refuse."""
    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        try:
            rc = moses_exec(["moses", *argv])
        except (ValueError, SystemExit) as exc:
            text = str(exc)
        else:
            assert rc != 0
            text = ""
    finally:
        os.chdir(old)
    cap = capsys.readouterr()
    return text + cap.out + cap.err


# ---- flags


@pytest.mark.parametrize("extra, message", [
    (["--const-opt", "mutate"], "needs -a gomea"),
    (["--const-opt", "polish", "--const-opt-val", "v.csv"], "needs -a gomea"),
])
def test_hc_still_refuses_mutate_and_polish(extra, message, tmp_path, capsys):
    assert message in _run_error(_reg(tmp_path) + _HC + extra, tmp_path, capsys)


@pytest.mark.parametrize("algo", ["sa", "un"])
def test_other_classic_optimizers_still_refuse_lm(algo, tmp_path, capsys):
    argv = _reg(tmp_path) + ["-a", algo, "--linear-scaling", "1"] + _LM
    assert "needs -a gomea" in _run_error(argv, tmp_path, capsys)


def test_hc_lm_guard_still_needs_the_validation_table(tmp_path, capsys):
    argv = _reg(tmp_path) + _HC + _LM + ["--const-opt-guard", "val"]
    assert "needs --const-opt-val" in _run_error(argv, tmp_path, capsys)


def test_hc_lm_is_regression_only(tmp_path, capsys):
    argv = ["-H", "pa", "-k", "3", "-r", "5", "-m", "300", "-a", "hc"] + _LM
    assert "regression" in _run_error(argv, tmp_path, capsys)


# ---- LM at each deme end


def test_lm_runs_at_each_hc_deme_end(tmp_path):
    out, events = _run(_reg(tmp_path) + _HC + _LM, tmp_path)
    ends = _ends(events)
    assert len(ends) >= 2  # several demes in 800 evaluations
    assert [e["deme_index"] for e in ends] == list(range(len(ends)))
    for e in ends:
        co = e["const_opt"]
        assert co["mode"] == "lm" and co["lm_calls"] == 1
        assert co["lm"]["status"] in ("improved", "not_improved", "singular", "failed",
                                      "no_constants", "budget")
        assert e["ledger"]["requests"] == e["hc_evals"] + co["tuning_evals"]
        assert e["ledger"]["tuning_requests"] == co["tuning_evals"] == co["lm"]["evals"]
        assert e["termination_reason"] in ("budget", "time_limit", "target", "local_optimum")
        assert isinstance(e["generations"], int)
    assert any(e["const_opt"]["lm"]["status"] == "improved" for e in ends)
    assert any(e["const_opt"]["lm"].get("written_back") for e in ends)


def test_the_evaluation_count_includes_the_lm_passes(tmp_path):
    out, events = _run(_reg(tmp_path) + _HC + _LM, tmp_path)
    ends = _ends(events)
    n = _n_evals(out)
    assert n <= 800
    assert n == sum(e["ledger"]["requests"] for e in ends)
    tuning = sum(e["const_opt"]["tuning_evals"] for e in ends)
    assert tuning > 0
    assert n == sum(e["hc_evals"] for e in ends) + tuning


def test_lm_changes_the_hc_run(tmp_path):
    """The tuned programs reach the output: LM is not a no-op."""
    plain = _run(_reg(tmp_path) + _HC, tmp_path)
    tuned = _run(_reg(tmp_path) + _HC + _LM, tmp_path)
    assert plain[1] == []  # -a hc without --const-opt writes no log
    assert plain[0] != tuned[0]


def test_lm_stops_at_the_budget(tmp_path):
    """With a small -m the budget ends during LM or before it; the count never passes -m."""
    argv = _reg(tmp_path, "120") + _HC + _LM
    out, events = _run(argv, tmp_path)
    assert _n_evals(out) <= 120
    assert _n_evals(out) == sum(e["ledger"]["requests"] for e in _ends(events))


# ---- the guard


def test_the_guard_rejects_a_change_that_makes_the_validation_error_worse(tmp_path):
    """Validation rows y = x1 (x2 + 2.5): LM moves the constant from the knob value 1.5 to the
    training value 1.37, away from 2.5, so the tuned program is worse on validation, and the
    guard keeps the old program."""
    out, events = _run(_reg(tmp_path) + _HC + _LM + _guard(tmp_path, c=2.5), tmp_path)
    ends = _ends(events)
    rejected = 0
    for e in ends:
        co, lm = e["const_opt"], e["const_opt"]["lm"]
        assert co["guard"] == "val"
        if lm.get("guard") == "rejected":
            rejected += 1
            assert lm["val_nmse_new"] > lm["val_nmse_old"]
            assert lm["written_back"] is False
            assert co["lm_guard_rejected"] == 1
        elif lm.get("guard") == "accepted":
            assert lm["val_nmse_new"] <= lm["val_nmse_old"]
    assert rejected > 0
    assert _n_evals(out) == sum(e["ledger"]["requests"] for e in ends)


def test_the_guard_accepts_when_validation_agrees(tmp_path):
    out, events = _run(_reg(tmp_path) + _HC + _LM + _guard(tmp_path, c=C), tmp_path)
    ends = _ends(events)
    checked = sum(e["const_opt"]["lm_guard_checked"] for e in ends)
    accepted = [e for e in ends if e["const_opt"]["lm"].get("guard") == "accepted"]
    assert checked > 0 and accepted
    assert any(e["const_opt"]["lm"]["written_back"] for e in accepted)


def test_a_rejected_result_is_never_written_back(tmp_path, monkeypatch):
    import pymoses.gomea.optimizer as co

    written = []
    real = co.scored_tree_instance

    def spy(*args):
        written.append(args)
        return real(*args)

    monkeypatch.setattr(co, "scored_tree_instance", spy)
    monkeypatch.setattr(co, "guard_accepts", lambda *a: {
        "accepted": False, "val_nmse_old": 1.0, "val_nmse_new": 2.0})
    _run(_reg(tmp_path) + _HC + _LM + _guard(tmp_path), tmp_path)
    assert written == []


# ---- widen search


def test_widen_search_with_lm(tmp_path):
    out, events = _run(_reg(tmp_path) + _HC + ["--hc-widen-search", "1"] + _LM + _guard(tmp_path),
                       tmp_path)
    ends = _ends(events)
    assert ends and all(e["const_opt"]["lm_calls"] == 1 for e in ends)
    assert _n_evals(out) <= 800
    assert _n_evals(out) == sum(e["ledger"]["requests"] for e in ends)


# ---- the runner


def _job(argv):
    return {"binary": "/x/.venv/bin/moses", "seed": 3, "budget": {"evals": 1000}, "argv": argv}


def test_runner_gives_hc_with_lm_the_events_log(tmp_path):
    out = runner.moses_argv(_job(_HC + _LM), tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    assert runner.moses_argv(_job(_HC), tmp_path, 100)[-1] == "1"  # plain HC: no log
    # the runner adds --const-opt-val for the guard whatever the optimizer
    assert runner._const_opt_guard(_HC + _LM + ["--const-opt-guard", "val"]) == "val"


def test_runner_summarizes_hc_lm_events(tmp_path):
    _, events = _run(_reg(tmp_path) + _HC + _LM + _guard(tmp_path, c=2.5), tmp_path)
    s = runner.const_opt_summary(tmp_path / "events.jsonl", 800)
    ends = _ends(events)
    assert s["mode"] == "lm" and s["lm_calls"] == len(ends)
    assert s["guard"] == "val"
    assert s["lm_guard_rejected"] == sum(e["const_opt"]["lm_guard_rejected"] for e in ends) > 0
