"""Budget-derived deme sizes: ``--gomea-pop-fraction``/``--gomea-pop-max``
(population N from the deme's budget) and ``--gomea-deme-share`` (the
per-deme cap from ``-m``).

With every option off the parameters, telemetry and runs are unchanged.
"""

from __future__ import annotations

import argparse
import json

import pytest

from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.gomea.capped_hc import hc_capped_optimization
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen
from tests.gomea.test_integration import _cli_parser, _run
from tests.gomea.test_optimizer import _deme_contents, _iscorer


@pytest.fixture
def pa3(make_boolean_rep):
    """A fresh ``pa`` -3 problem (as in ``test_optimizer``): (rep, iscorer, deme)."""

    def _make():
        rep = make_boolean_rep("and", 3)
        return rep, _iscorer(rep, 3), instance_set(rep.fields())

    return _make


def _gp(*argv: str) -> gomea_parameters:
    return _gomea_params_from_args(_cli_parser(True).parse_args(list(argv)))


# ---------------------------------------------------------------------------
# N from the fraction, the cap, the evaluations left and the clamp.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fraction", "cap", "max_evals", "pop_max", "expected"),
    [
        (0.1, None, 400, 0, 40),  # no cap: the evaluations left
        (0.1, 100, 400, 0, 10),  # cap smaller than what is left
        (0.1, 1000, 400, 0, 40),  # what is left smaller than the cap
        (0.04, 1000, 4000, 0, 40),
        (0.05, 250, 4000, 0, 12),  # floor(12.5)
        (0.29, None, 100, 0, 29),  # exact product, not 28.999...
        (0.01, 100, 100, 0, 2),  # floor(1) raised to the minimum 2
        (0.1, 100, 0, 0, 2),  # nothing left: still a valid population
        (0.1, None, 4000, 64, 64),  # clamped
        (0.1, None, 400, 64, 40),  # under the clamp
    ],
)
def test_deme_pop_size(fraction, cap, max_evals, pop_max, expected):
    gp = gomea_parameters(max_evals_per_deme=cap, pop_fraction=fraction, pop_max=pop_max)
    assert gp.deme_pop_size(max_evals) == expected


def test_pop_fraction_off_keeps_pop_size():
    gp = gomea_parameters(pop_size=24, max_evals_per_deme=100)
    assert all(gp.deme_pop_size(m) == 24 for m in (0, 50, 10**6))
    assert gp.deme_budget(400) == 100 and gp.deme_budget(60) == 60


@pytest.mark.parametrize(
    "kw", [{"pop_fraction": -0.1}, {"pop_max": 1}, {"pop_max": -2}, {"deme_share": -1}]
)
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        gomea_parameters(**kw)


# ---------------------------------------------------------------------------
# The CLI: deme-share cap from -m, precedence, validation, off = unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("m", "share", "cap"), [(4000, 4, 1000), (4000, 16, 250), (4001, 4, 1000), (3, 16, 1)]
)
def test_deme_share_cap_from_total_budget(m, share, cap):
    gp = _gp("-m", str(m), "--gomea-deme-share", str(share))
    assert gp.max_evals_per_deme == cap and gp.deme_share == share


def test_deme_share_with_agreeing_explicit_cap_is_accepted():
    gp = _gp("-m", "4000", "--gomea-deme-share", "16", "--gomea-max-evals-per-deme", "250")
    assert gp.max_evals_per_deme == 250


def test_deme_share_with_conflicting_explicit_cap_is_an_error():
    with pytest.raises(ValueError, match="conflicts"):
        _gp("-m", "4000", "--gomea-deme-share", "4", "--gomea-max-evals-per-deme", "250")


@pytest.mark.parametrize(
    "argv",
    [
        ["--gomea-pop-fraction", "-0.1"],
        ["--gomea-pop-fraction", "1.5"],
        ["--gomea-pop-fraction", "0.1", "--gomea-pop-max", "1"],
        ["--gomea-pop-max", "32"],  # needs --gomea-pop-fraction
        ["--gomea-deme-share", "-1"],
    ],
)
def test_invalid_cli_values_rejected(argv):
    with pytest.raises(ValueError):
        _gp(*argv)


def test_cli_options_reach_parameters():
    gp = _gp("-m", "4000", "--gomea-pop-fraction", "0.04", "--gomea-pop-max", "32")
    assert (gp.pop_fraction, gp.pop_max, gp.deme_share) == (0.04, 32, 0)
    assert gp.max_evals_per_deme is None and gp.deme_pop_size(4000) == 32


def test_options_off_leave_parameters_unchanged():
    gp = _gp("-m", "4000", "--gomea-max-evals-per-deme", "250")
    ref = gomea_parameters(max_evals_per_deme=250)
    assert gp.as_dict() == ref.as_dict()
    assert not {"pop_fraction", "pop_max", "deme_share"} & gp.as_dict().keys()
    assert not gp.budget_derived() and gp.deme_pop_size(4000) == 16
    # A namespace without the new options (an older caller) still works.
    assert _gomea_params_from_args(argparse.Namespace()).as_dict() == gomea_parameters().as_dict()


def test_deme_share_caps_hc_capped():
    gp = _gp("-m", "4000", "--gomea-deme-share", "16")
    opt = hc_capped_optimization(optim_parameters(), hc_parameters(), gp)
    assert opt.cap == 250


# ---------------------------------------------------------------------------
# The optimizer draws N programs and logs the effective sizes.
# ---------------------------------------------------------------------------


def test_optimizer_uses_derived_population(pa3, tmp_path):
    log = tmp_path / "g.jsonl"
    rep, isc, deme = pa3()
    gp = gomea_parameters(max_evals_per_deme=200, pop_fraction=0.1, telemetry_path=str(log))
    randGen().seed(5)
    gomea_optimization(optim_parameters(), gp)(deme, isc, 10**4, 10**6)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = next(e for e in events if e["type"] == "invocation_start")
    init = next(e for e in events if e["type"] == "init")
    assert (start["deme_budget"], start["deme_cap"], start["deme_pop_size"]) == (200, 200, 20)
    assert init["n"] == 20 and start["params"]["pop_fraction"] == 0.1


def test_fraction_equal_to_fixed_size_gives_the_same_deme(pa3):
    """A fraction giving N = 16 searches exactly like --gomea-pop-size 16."""
    out = []
    for gp in (
        gomea_parameters(max_evals_per_deme=160),
        gomea_parameters(max_evals_per_deme=160, pop_fraction=0.1),
    ):
        rep, isc, deme = pa3()
        randGen().seed(9)
        gomea_optimization(optim_parameters(), gp)(deme, isc, 10**4, 10**6)
        out.append((_deme_contents(deme), randGen().randint()))
    assert out[0] == out[1]


def test_off_path_telemetry_has_no_new_keys(pa3, tmp_path):
    log = tmp_path / "g.jsonl"
    rep, isc, deme = pa3()
    randGen().seed(5)
    gp = gomea_parameters(max_evals_per_deme=100, telemetry_path=str(log))
    gomea_optimization(optim_parameters(), gp)(deme, isc, 10**4, 10**6)
    start = json.loads(log.read_text().splitlines()[0])
    assert not {"deme_budget", "deme_cap", "deme_pop_size"} & start.keys()
    assert not {"pop_fraction", "pop_max", "deme_share"} & start["params"].keys()


# ---------------------------------------------------------------------------
# End to end through the CLI.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["gomea", "hc-gomea"])
def test_end_to_end_logs_cap_and_population(algo, tmp_path):
    log = tmp_path / "g.jsonl"
    rc, stdout = _run(["-H", "maj", "-k", "3", "-a", algo, "--gomea-deme-share", "4",
                       "--gomea-pop-fraction", "0.1", "-m", "400", "-r", "1",
                       "--gomea-log", str(log)])  # fmt: skip
    assert rc == 0 and stdout
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [e for e in events if e["type"] == "invocation_start"]
    inits = [e for e in events if e["type"] == "init"]
    ends = [e for e in events if e["type"] == "end"]
    assert starts and len(starts) == len(ends)
    for s in starts:
        assert s["deme_cap"] == 100 and s["params"]["deme_share"] == 4
        assert s["deme_budget"] == min(100, s["max_evals"])
        assert s["deme_pop_size"] == max(2, s["deme_budget"] // 10)
    if algo == "gomea":
        assert [i["n"] for i in inits] == [s["deme_pop_size"] for s in starts]
    for e in ends:
        assert e["ledger"]["requests"] <= 100
