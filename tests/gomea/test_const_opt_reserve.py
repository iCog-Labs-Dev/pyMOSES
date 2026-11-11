"""Server round 11: the LM reserve for GOMEA (``-a gomea --const-opt lm --const-opt-reserve
R``), as ``docs/gomea/SERVER-ROUND11-PREREGISTRATION.md`` ("Code change") registers it: GOMEA
in each deme stops at the deme budget minus min(R, deme budget // 2), and LM on the deme's
best program can then use the evaluations that are left. A deme that ends by stagnation
before that point does not change. The identity of the arms without the option is
``test_const_opt_reserve_identity.py``."""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
TRAIN = DATA / "train.csv"

# The round 10 BASE flags (configs/gomea/phase1bd/), with the training table as the
# validation table of the guard.
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
    "--const-opt-val", str(TRAIN),
]  # fmt: skip
CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def _reg(m: str) -> list[str]:
    return ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


def _reserve(r: int) -> list[str]:
    return ["--const-opt-reserve", str(r)]


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
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


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


def _demes(events):
    """(invocation_start, end) of each deme, in order."""
    starts = [e for e in events if e["type"] == "invocation_start"]
    ends = [e for e in events if e["type"] == "end"]
    assert len(starts) == len(ends)
    return list(zip(starts, ends, strict=True))


def _n_evals(out):
    return int(out.rsplit("n_evals:", 1)[1])


def _search(end):
    """The evaluations of the GOMEA search of a deme (LM passes left out)."""
    return end["ledger"]["requests"] - end["ledger"].get("tuning_requests", 0)


# ---- the flag


@pytest.mark.parametrize("extra, message", [
    (_BASE + _reserve(-1), "--const-opt-reserve must be >= 0"),
    (["-a", "gomea", "--gomea-contin", "copy", "--linear-scaling", "1"] + _reserve(500),
     "--const-opt-reserve needs --const-opt lm"),
    (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "mutate"] + _reserve(500),
     "--const-opt-reserve needs --const-opt lm"),
    (["-a", "hc", "--linear-scaling", "1", "--const-opt", "lm"] + _reserve(500),
     "--const-opt-reserve needs -a gomea"),
    (["-a", "hc-gomea", "--gomea-contin", "copy", "--const-opt", "lm"] + _reserve(500),
     "--const-opt-reserve needs -a gomea"),
], ids=["negative", "no-const-opt", "mutate", "hc", "hc-gomea"])
def test_the_flag_is_refused(extra, message, tmp_path, capsys):
    assert message in _run_error(_reg("300") + extra, tmp_path, capsys)


def test_the_parameters_refuse_a_bad_reserve():
    from pymoses.gomea.optimizer import gomea_parameters

    with pytest.raises(ValueError, match="const_opt_reserve must be"):
        gomea_parameters(const_opt="lm", const_opt_reserve=-1)
    with pytest.raises(ValueError, match="const_opt_reserve needs const_opt 'lm'"):
        gomea_parameters(const_opt_reserve=500)


def test_the_hybrid_refuses_a_reserve():
    from pymoses.gomea.hybrid import hc_gomea_optimization
    from pymoses.gomea.optimizer import gomea_parameters

    with pytest.raises(ValueError, match="const_opt_reserve"):
        hc_gomea_optimization(gp=gomea_parameters(const_opt="lm", const_opt_reserve=500))


def test_the_reserve_value():
    """min(R, deme budget // 2), 0 when off; the parameter is left out of the telemetry
    while it is 0."""
    from pymoses.gomea.optimizer import gomea_parameters

    gp = gomea_parameters(const_opt="lm", const_opt_reserve=500)
    assert [gp.lm_reserve(b) for b in (20000, 1000, 999, 600, 1, 0)] == [500, 500, 499, 300,
                                                                         0, 0]
    assert gp.as_dict()["const_opt_reserve"] == 500
    off = gomea_parameters(const_opt="lm")
    assert off.lm_reserve(20000) == 0 and "const_opt_reserve" not in off.as_dict()


# ---- LM gets evaluations


def test_without_the_reserve_one_deme_uses_the_whole_budget_and_lm_gets_none(tmp_path):
    """The case of BASE on Dow and Tow: at -m 600 the one deme ends at the budget, so the LM
    call at its end gets 0 evaluations. The "end" event has no const_opt_reserve."""
    out, events = _run(_reg("600") + _BASE, tmp_path)
    [(start, end)] = _demes(events)
    assert end["termination_reason"] == "global_budget"
    assert end["const_opt"]["tuning_evals"] == 0
    assert end["const_opt"]["lm"]["status"] == "budget"
    assert "const_opt_reserve" not in end
    assert _n_evals(out) == 600


def test_with_the_reserve_lm_gets_evaluations(tmp_path):
    """The same run with R = 500: the deme budget is 600, so the reserve is 300. GOMEA stops
    at 300 evaluations and LM uses some of the 300 that are left."""
    out, events = _run(_reg("600") + _BASE + _reserve(500), tmp_path)
    (start, end), *_ = _demes(events)
    assert start["max_evals"] == 600
    assert end["const_opt_reserve"] == 300
    assert end["termination_reason"] == "global_budget"
    assert _search(end) == 300
    assert end["const_opt"]["tuning_evals"] > 0
    assert end["const_opt"]["lm"]["status"] != "budget"
    assert _n_evals(out) <= 600


@pytest.mark.parametrize("m, r", [("600", 500), ("1500", 500), ("200", 5000), ("1500", 1)])
def test_the_evaluations_never_pass_m(m, r, tmp_path):
    out, events = _run(_reg(m) + _BASE + _reserve(r), tmp_path)
    demes = _demes(events)
    n = _n_evals(out)
    assert n <= int(m)
    assert n == sum(end["ledger"]["requests"] for _, end in demes)
    for start, end in demes:
        budget = start["max_evals"]  # no deme cap: the deme budget is what is left of -m
        reserve = min(r, budget // 2)
        assert end["const_opt_reserve"] == reserve
        assert _search(end) <= budget - reserve
        assert end["ledger"]["requests"] <= budget


def test_a_deme_that_stagnates_before_the_stop_does_not_change(tmp_path):
    """At -m 1500 the first deme ends by stagnation after fewer than 1000 evaluations, so a
    reserve of 500 (stop at 1000) does not change it."""
    _, plain = _run(_reg("1500") + _BASE, tmp_path)
    _, reserved = _run(_reg("1500") + _BASE + _reserve(500), tmp_path)

    def first_deme(events):
        out, ends = [], 0
        for e in events:
            if ends:
                break
            e = {k: v for k, v in e.items() if k not in CLOCK_KEYS | {"const_opt_reserve"}}
            e.get("params", {}).pop("const_opt_reserve", None)
            out.append(e)
            ends += e["type"] == "end"
        return out

    a, b = first_deme(plain), first_deme(reserved)
    assert a[-1]["termination_reason"] == "stagnation"
    assert _search(a[-1]) < 1000
    assert a == b
    assert [e["const_opt_reserve"] for e in reserved if e["type"] == "end"][0] == 500


def test_with_a_deme_cap_the_search_stops_below_the_cap(tmp_path):
    """With a per-deme cap of 400 the deme budget is 400 and the reserve is 200: the search
    stops at 200 (reason inner_budget), and LM can use the rest of the run's budget."""
    argv = _reg("1500") + _BASE + ["--gomea-max-evals-per-deme", "400"] + _reserve(500)
    out, events = _run(argv, tmp_path)
    demes = _demes(events)
    assert _n_evals(out) <= 1500
    for start, end in demes:
        budget = min(start["max_evals"], 400)
        assert end["const_opt_reserve"] == min(500, budget // 2)
        assert _search(end) <= budget - end["const_opt_reserve"]
    assert any(end["termination_reason"] == "inner_budget" for _, end in demes)
    assert any(end["const_opt"]["tuning_evals"] > 0 for _, end in demes)
