"""The LM validation guard (``--const-opt lm --const-opt-guard val``, server round 5): accept
and reject, the evaluation count, the flags, the runner, and the identity of the guard's off
state with the code before it."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.gomea.const_opt import RegressionRows, guard_accepts, validation_nmse
from pymoses.gomea.optimizer import gomea_parameters
from pymoses.gomea.oracle import BudgetExhausted

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
# Reference: `--const-opt lm` and `polish` at commit ed7e383, the commit before the guard
# (`make_reference.py` in that folder wrote these files there).
LM_REF = Path(__file__).resolve().parent / "data" / "const_opt_lm_ed7e383"

_ARGV = [
    "-i", str(DATA / "train.csv"), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "3", "-V1", "-l", "WARN", "-m", "700", "-a", "gomea",
    "--gomea-pop-size", "16", "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "1",
]  # fmt: skip
CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def _run(argv, tmp_path):
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    if log.exists():
        log.unlink()
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


def _canonical(events):
    lines = []
    for ev in events:
        ev = {k: v for k, v in ev.items() if k not in CLOCK_KEYS}
        if "telemetry_path" in ev.get("params", {}):
            ev["params"]["telemetry_path"] = "LOG"
        lines.append(json.dumps(ev, sort_keys=True))
    return "\n".join(lines) + "\n"


def _noisy_val_table(path: Path, slope: float) -> Path:
    """A validation table with the training columns: y = slope * sin(1.3 x1) + 0.5 x2."""
    rows = list(csv.reader((DATA / "train.csv").open()))
    head, body = rows[0], rows[1:]
    k, x1, x2 = head.index("y"), head.index("x1"), head.index("x2")
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in body:
            r = list(r)
            r[k] = repr(slope * math.sin(1.3 * float(r[x1])) + 0.5 * float(r[x2]))
            w.writerow(r)
    return path


# ---- the check itself


def _rows(ys, scaled=True):
    xs = [-3 + 6 * i / (len(ys) - 1) for i in range(len(ys))]
    return RegressionRows([[x] for x in xs], ys, scaled)


def _sine(w, n=40):
    return [math.sin(w * (-3 + 6 * i / (n - 1))) for i in range(n)]


def test_guard_accepts_a_program_that_is_better_on_validation():
    train = _rows(_sine(1.3))
    val = _rows(_sine(1.3))
    charges = []
    v = guard_accepts(str_to_combo_tree("sin(*(1.0 $1))"), str_to_combo_tree("sin(*(1.3 $1))"),
                      train, val, lambda: charges.append(1))
    assert v["accepted"] and v["val_nmse_new"] < v["val_nmse_old"]
    assert len(charges) == 2


def test_guard_rejects_a_program_that_is_worse_on_validation():
    """Better on the training rows, worse on the validation rows."""
    train = _rows(_sine(1.3))
    val = _rows(_sine(1.0))
    old, new = str_to_combo_tree("sin(*(1.0 $1))"), str_to_combo_tree("sin(*(1.3 $1))")
    v = guard_accepts(old, new, train, val, lambda: None)
    assert not v["accepted"] and v["val_nmse_new"] > v["val_nmse_old"]


def test_guard_accepts_an_equal_validation_nmse():
    """The rule is "not worse"."""
    train = _rows(_sine(1.3))
    t = str_to_combo_tree("sin(*(1.3 $1))")
    assert guard_accepts(t, t, train, _rows(_sine(1.0)), lambda: None)["accepted"]


def test_guard_scales_with_the_training_fit():
    """The guard's NMSE is `validation_nmse`, whose scaling comes from the training rows: a
    validation target with another offset and slope is not refitted."""
    train = _rows(_sine(1.3))
    val = _rows([3 * y + 5 for y in _sine(1.3)])
    t = str_to_combo_tree("sin(*(1.3 $1))")
    v = guard_accepts(t, t, train, val, lambda: None)
    assert v["val_nmse_new"] == validation_nmse(t, train, val)
    assert v["val_nmse_new"] > 0.5  # a refit on the validation rows gives about 0


@pytest.mark.parametrize("limit", [0, 1])
def test_guard_stops_at_the_budget(limit):
    count = [0]

    def charge():
        if count[0] >= limit:
            raise BudgetExhausted("global_budget")
        count[0] += 1

    t = str_to_combo_tree("sin(*(1.3 $1))")
    with pytest.raises(BudgetExhausted):
        guard_accepts(t, t, _rows(_sine(1.3)), _rows(_sine(1.3)), charge)
    assert count[0] == limit


# ---- flags


def test_parameters_validation():
    with pytest.raises(ValueError):
        gomea_parameters(const_opt_guard="test")
    with pytest.raises(ValueError):
        gomea_parameters(const_opt_guard="val", const_opt_val="v.csv")  # needs lm
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="lm", const_opt_guard="val")  # needs the table
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="lm", const_opt_val="v.csv")  # a table without the guard
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="polish", const_opt_val="v.csv", const_opt_guard="val")
    d = gomea_parameters(const_opt="lm", const_opt_val="v.csv", const_opt_guard="val").as_dict()
    assert d["const_opt_guard"] == "val"
    assert "const_opt_guard" not in gomea_parameters().as_dict()
    assert "const_opt_guard" not in gomea_parameters(const_opt="lm").as_dict()
    assert "const_opt_guard" not in gomea_parameters(const_opt="polish",
                                                     const_opt_val="v.csv").as_dict()


@pytest.mark.parametrize("extra, message", [
    (["--const-opt-guard", "bad"], "not recognized"),
    (["--const-opt-guard", "val"], "needs --const-opt lm"),
    (["--const-opt", "lm", "--const-opt-guard", "val"], "needs --const-opt-val"),
    (["--const-opt", "mutate", "--const-opt-guard", "val", "--const-opt-val", "v"],
     "needs --const-opt lm"),
    (["--const-opt", "lm", "--const-opt-val", "v"], "--const-opt-guard val only"),
])
def test_command_line_validation(extra, message, tmp_path, capsys):
    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        try:
            rc = moses_exec(["moses", *_ARGV, *extra])
        except (ValueError, SystemExit) as exc:
            text = str(exc)
        else:
            cap = capsys.readouterr()
            text = cap.out + cap.err
            assert rc != 0
    finally:
        os.chdir(old)
    cap = capsys.readouterr()
    assert message in text + cap.out + cap.err


# ---- end to end


def _ends(events):
    return [e for e in events if e["type"] == "end"]


def test_end_to_end_cost_accounting(tmp_path):
    val = _noisy_val_table(tmp_path / "val.csv", 2.5)
    out, events = _run(
        _ARGV + ["--const-opt", "lm", "--const-opt-guard", "val", "--const-opt-val", str(val)],
        tmp_path)
    ends = _ends(events)
    n_evals = int(out.rsplit("n_evals:", 1)[1])
    assert n_evals <= 700
    assert n_evals == sum(e["ledger"]["requests"] for e in ends)
    assert sum(e["const_opt"]["tuning_evals"] for e in ends) == sum(
        e["ledger"].get("tuning_requests", 0) for e in ends)
    n_checked = 0
    for e in ends:
        co, lm = e["const_opt"], e["const_opt"]["lm"]
        assert co["guard"] == "val"
        n_checked += co["lm_guard_checked"]
        if lm is None or lm["status"] != "improved":
            assert co["lm_guard_checked"] == 0
            continue
        if "guard" not in lm:  # the budget ended in the guard
            assert lm["budget"] and not lm["written_back"]
            continue
        accepted = lm["guard"] == "accepted"
        assert accepted == (lm["val_nmse_new"] <= lm["val_nmse_old"])
        assert lm["written_back"] == (accepted and not lm["budget"])
        # LM's passes, two validation passes, and the final score if the guard accepted
        assert co["lm_guard_rejected"] == (0 if accepted else 1)
    assert n_checked > 0


def _first_improved(events):
    return next(e["const_opt"]["lm"] for e in _ends(events)
                if e["const_opt"]["lm"] and e["const_opt"]["lm"]["status"] == "improved")


@pytest.mark.parametrize("accepted, extra_passes", [(True, 3), (False, 2)])
def test_each_validation_pass_costs_one_evaluation(accepted, extra_passes, tmp_path, monkeypatch):
    """The first deme's search is the same with or without the guard, so its LM call is too.
    Without the guard, LM's passes and one score. With it: two validation passes, and the
    score only if the guard accepts. The guard's verdict is forced here."""
    import pymoses.gomea.optimizer as opt

    _, plain = _run(_ARGV + ["--const-opt", "lm"], tmp_path)
    lm_only = _first_improved(plain)["evals"] - 1
    real = opt.guard_accepts

    def forced(*args):
        return {**real(*args), "accepted": accepted}

    monkeypatch.setattr(opt, "guard_accepts", forced)
    val = _noisy_val_table(tmp_path / "val.csv", 2.5)
    out, events = _run(
        _ARGV + ["--const-opt", "lm", "--const-opt-guard", "val", "--const-opt-val", str(val)],
        tmp_path)
    rec = _first_improved(events)
    assert rec["evals"] == lm_only + extra_passes
    assert rec["written_back"] is accepted
    assert rec["guard"] == ("accepted" if accepted else "rejected")
    ends = _ends(events)
    checked = sum(e["const_opt"]["lm_guard_checked"] for e in ends)
    rejected = sum(e["const_opt"]["lm_guard_rejected"] for e in ends)
    assert checked > 0 and rejected == (0 if accepted else checked)
    assert int(out.rsplit("n_evals:", 1)[1]) == sum(e["ledger"]["requests"] for e in ends)


def test_a_rejected_result_leaves_the_old_program_in_the_output(tmp_path, monkeypatch):
    """With every LM result rejected, no tuned constant reaches the merge: the printed best
    program is one that the search itself found, as with no tuning on the same demes."""
    import pymoses.gomea.optimizer as opt

    written = []
    real = opt.scored_tree_instance

    def spy(*args):
        written.append(args)
        return real(*args)

    monkeypatch.setattr(opt, "scored_tree_instance", spy)
    monkeypatch.setattr(opt, "guard_accepts", lambda *a: {
        "accepted": False, "val_nmse_old": 1.0, "val_nmse_new": 2.0})
    val = _noisy_val_table(tmp_path / "val.csv", 2.5)
    _run(_ARGV + ["--const-opt", "lm", "--const-opt-guard", "val", "--const-opt-val", str(val)],
         tmp_path)
    assert written == []
    monkeypatch.setattr(opt, "guard_accepts", lambda *a: {
        "accepted": True, "val_nmse_old": 2.0, "val_nmse_new": 1.0})
    _run(_ARGV + ["--const-opt", "lm", "--const-opt-guard", "val", "--const-opt-val", str(val)],
         tmp_path)
    assert written != []


def test_guard_none_is_the_same_as_no_flag(tmp_path):
    a = _run(_ARGV + ["--const-opt", "lm"], tmp_path)
    b = _run(_ARGV + ["--const-opt", "lm", "--const-opt-guard", "none"], tmp_path)
    assert a[0] == b[0]
    assert _canonical(a[1]) == _canonical(b[1])


@pytest.mark.parametrize("name", ["lm", "polish"])
def test_lm_and_polish_are_byte_identical_to_ed7e383(name, tmp_path):
    extra = {"lm": ["--const-opt", "lm"],
             "polish": ["--const-opt", "polish", "--const-opt-val", str(DATA / "train.csv")]}[name]
    out, events = _run(_ARGV + extra, tmp_path)
    assert out == (LM_REF / f"{name}.out").read_text()
    assert _canonical(events).replace(str(DATA), "DATA") == (
        LM_REF / f"{name}.events.jsonl").read_text()


# ---- the runner


def test_runner_gives_the_validation_table_only_to_the_guard_and_polish():
    assert runner._const_opt_guard(["--const-opt", "lm", "--const-opt-guard", "val"]) == "val"
    assert runner._const_opt_guard(["--const-opt-guard=val"]) == "val"
    assert runner._const_opt_guard(["--const-opt", "lm"]) == "none"


def test_runner_passes_the_scaled_validation_split(tmp_path, monkeypatch):
    split = tmp_path / "split"
    split.mkdir()
    for f in ("train", "val", "test"):
        (split / f"{f}.csv").write_text((DATA / "train.csv").read_text())
    calls = {}

    class Done:
        stdout, stderr, returncode = "", "", 1

    def fake_run(argv, **kw):
        calls["argv"] = argv
        return Done()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for name, argv, has_val in [
        ("guard", ["-a", "gomea", "--const-opt", "lm", "--const-opt-guard", "val"], True),
        ("plain", ["-a", "gomea", "--const-opt", "lm"], False),
        ("off", ["-a", "gomea"], False),
    ]:
        run_dir = tmp_path / name
        runner._run_one({"run_id": name, "task": "T", "arm": name, "seed": 1, "argv": argv,
                         "binary": ".venv/bin/moses", "budget": {"evals": 10},
                         "split_dir": str(split), "wall_cap_s": 10, "mem_limit_mb": None,
                         "run_dir": str(run_dir)})
        assert ("--const-opt-val" in calls["argv"]) == has_val, name
        assert (run_dir / "val_scaled.csv").exists() == has_val, name
        if has_val:
            i = calls["argv"].index("--const-opt-val")
            assert calls["argv"][i + 1] == str(run_dir / "val_scaled.csv")


def test_summary_carries_the_rejected_count(tmp_path):
    ev = tmp_path / "events.jsonl"
    base = {"lm_calls": 1, "lm_improved": 1, "tuning_evals": 9, "mode": "lm"}
    lines = [
        {"type": "end", "const_opt": {**base, "guard": "val", "lm_guard_checked": 1,
                                      "lm_guard_rejected": 1}},
        {"type": "end", "const_opt": {**base, "guard": "val", "lm_guard_checked": 1,
                                      "lm_guard_rejected": 0}},
        {"type": "end", "const_opt": {**base, "guard": "val", "lm_guard_checked": 2,
                                      "lm_guard_rejected": 2}},
    ]
    ev.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    s = runner.const_opt_summary(ev, 100)
    assert s["guard"] == "val"
    assert s["lm_guard_checked"] == 4 and s["lm_guard_rejected"] == 3
    assert s["lm_guard_rejected_share"] == 0.75


def test_summary_without_the_guard_has_no_guard_keys(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text(json.dumps({"type": "end", "const_opt": {"mode": "lm", "lm_calls": 1}}) + "\n")
    s = runner.const_opt_summary(ev, 100)
    assert not [k for k in s if "guard" in k]

