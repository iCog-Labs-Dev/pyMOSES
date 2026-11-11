"""``--gomea-pop-size dynamic`` (server round 6, ``docs/gomea/SERVER-ROUND6-PREREGISTRATION.md``,
"New code"): each deme's population N is the number of new instances classic hill climbing
takes at distance 1, from ``--hc-fraction-of-nn`` and ``--hc-max-nn-evals``, the classic
minimum of 100 and the evaluations left. The fixed-N identity is in
``test_pop_size_identity.py``."""

from __future__ import annotations

import json

import pytest

from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.gomea.optimizer import DYNAMIC_POP_SIZE, dynamic_pop_size
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.field_set import disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_scorer import distance_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen
from tests.gomea.test_integration import _cli_parser, _run
from tests.gomea.test_optimizer import _deme_contents, _iscorer

FRACTIONS = [2.0, 0.25, 0.05]


def _gp(*argv: str) -> gomea_parameters:
    return _gomea_params_from_args(_cli_parser(True).parse_args(list(argv)))


def _fields(arity: int, n: int, extra: tuple[int, int] | None = None) -> field_set:
    specs = [disc_spec(arity)] * n
    if extra is not None:
        specs += [disc_spec(extra[0])] * extra[1]
    return field_set(specs)


class _Stop(Exception):
    pass


def _classic_count(fs: field_set, fraction: float, max_evals: int, max_nn: int = 20000):
    """(count, current_number_of_evals) that classic hill climbing computes at its first
    distance-1 step on ``fs``, with ``--hc-fraction-of-nn fraction``. The climb stops there."""
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    hcp = hc_parameters()
    hcp.fraction_of_nn = fraction
    hcp.max_nn_evals = max_nn
    hc = hill_climbing(optim_parameters(), hcp)
    seen = {}
    real = hill_climbing.n_new_instances

    def spy(self, distance, me, cur, total):
        n = real(self, distance, me, cur, total)
        if distance == 1:
            seen["count"], seen["cur"] = n, cur
            raise _Stop
        return n

    hc.n_new_instances = spy.__get__(hc)
    randGen().seed(1)
    with pytest.raises(_Stop):
        hc(instance_set(fs), scorer, max_evals, 86400)
    return seen["count"], seen["cur"]


FIELD_SETS = {
    "bin3": (2, 3, None),
    "bin40": (2, 40, None),
    "ter120": (3, 120, None),
    "mixed": (2, 300, (5, 60)),
    "bin7000": (2, 7000, None),
}


@pytest.mark.parametrize("name", sorted(FIELD_SETS))
@pytest.mark.parametrize("fraction", [*FRACTIONS, 1.0])
@pytest.mark.parametrize("max_evals", [10**9, 1000, 60])
def test_count_equals_the_classic_count(name, fraction, max_evals):
    """For a field set, a fraction and a budget, N is what classic hill climbing takes at
    distance 1 with the same evaluations left."""
    fs = _fields(*FIELD_SETS[name])
    classic, cur = _classic_count(fs, fraction, max_evals)
    gp = gomea_parameters(pop_size=DYNAMIC_POP_SIZE, hc_fraction_of_nn=fraction)
    assert gp.deme_pop_size(max_evals - cur, fs) == max(2, classic)


@pytest.mark.parametrize("max_nn", [20000, 500])
def test_max_nn_evals_is_the_classic_cap(max_nn):
    fs = _fields(2, 7000)
    classic, cur = _classic_count(fs, 2.0, 10**9, max_nn)
    assert classic == max_nn
    assert (
        gomea_parameters(pop_size="dynamic", hc_max_nn_evals=max_nn).deme_pop_size(10**9, fs)
        == max_nn
    )


def test_known_values():
    """information_theoretic_bits of 40 binary fields is 40: 2 x 40 neighbours x F, at
    least 100, at most the evaluations left."""
    fs = _fields(2, 40)
    assert dynamic_pop_size(fs, 10**9, 2.0, 20000) == 160
    assert dynamic_pop_size(fs, 10**9, 0.25, 20000) == 100  # the classic minimum
    assert dynamic_pop_size(fs, 50, 2.0, 20000) == 50  # the evaluations left
    assert dynamic_pop_size(fs, 1, 2.0, 20000) == 2  # GOM needs a donor
    big = _fields(2, 1000)
    assert dynamic_pop_size(big, 10**9, 0.05, 20000) == 100
    assert dynamic_pop_size(big, 10**9, 0.25, 20000) == 500


def test_the_deme_cap_limits_the_evaluations_left():
    fs = _fields(2, 1000)
    gp = gomea_parameters(pop_size="dynamic", max_evals_per_deme=300)
    assert gp.deme_pop_size(10**9, fs) == 300


def test_cli_parses_dynamic_and_the_classic_options():
    gp = _gp("--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.25",
             "--hc-max-nn-evals", "777")  # fmt: skip
    assert gp.pop_size == "dynamic" and gp.dynamic()
    assert (gp.hc_fraction_of_nn, gp.hc_max_nn_evals) == (0.25, 777)
    d = gp.as_dict()
    assert d["pop_size"] == "dynamic" and d["hc_fraction_of_nn"] == 0.25
    assert d["hc_max_nn_evals"] == 777
    fixed = _gp("--gomea-pop-size", "16", "--hc-fraction-of-nn", "0.25")
    assert fixed.pop_size == 16 and not fixed.dynamic()
    assert not {"hc_fraction_of_nn", "hc_max_nn_evals"} & fixed.as_dict().keys()


def test_cli_refuses_a_bad_value():
    with pytest.raises(SystemExit):
        _cli_parser(True).parse_args(["--gomea-pop-size", "dynamc"])


def test_dynamic_refuses_pop_fraction():
    with pytest.raises(ValueError, match="both size the population"):
        _gp("--gomea-pop-size", "dynamic", "--gomea-pop-fraction", "0.1")
    with pytest.raises(ValueError, match="both size the population"):
        gomea_parameters(pop_size="dynamic", pop_fraction=0.1)
    with pytest.raises(ValueError, match="needs the deme's field set"):
        gomea_parameters(pop_size="dynamic").deme_pop_size(100)


@pytest.mark.parametrize("bad", ["16", 1, 2.0, True])
def test_a_bad_pop_size_is_refused(bad):
    with pytest.raises(ValueError):
        gomea_parameters(pop_size=bad)


def test_optimizer_draws_and_logs_the_dynamic_population(make_boolean_rep, tmp_path):
    log = tmp_path / "g.jsonl"
    rep = make_boolean_rep("and", 3)
    isc = _iscorer(rep, 3)
    deme = instance_set(rep.fields())
    gp = gomea_parameters(pop_size="dynamic", hc_fraction_of_nn=2.0, telemetry_path=str(log))
    randGen().seed(5)
    gomea_optimization(optim_parameters(), gp)(deme, isc, 10**4, 10**6)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = next(e for e in events if e["type"] == "invocation_start")
    init = next(e for e in events if e["type"] == "init")
    end = next(e for e in events if e["type"] == "end")
    n = dynamic_pop_size(rep.fields(), 10**4, 2.0, 20000)
    assert n >= 100
    assert start["deme_pop_size"] == n and start["deme_budget"] == 10**4
    assert start["params"]["pop_size"] == "dynamic"
    assert end["pop_size_dynamic"] == n
    assert init["n"] == end["pop_size_effective"]
    assert init["n"] == n or init["feasible_short"]


def test_a_dynamic_n_of_16_searches_like_a_fixed_16(make_boolean_rep):
    """The dynamic size only changes N: with the same N the deme is the same."""
    out = []
    for gp in (
        gomea_parameters(max_evals_per_deme=16 * 10),
        gomea_parameters(pop_size="dynamic", max_evals_per_deme=16 * 10),
    ):
        rep = make_boolean_rep("and", 3)
        isc, deme = _iscorer(rep, 3), instance_set(rep.fields())
        n = gp.deme_pop_size(16, rep.fields())
        assert n == 16
        randGen().seed(9)
        gomea_optimization(optim_parameters(), gp)(deme, isc, 16, 10**6)
        out.append((_deme_contents(deme), randGen().randint()))
    assert out[0] == out[1]


@pytest.mark.parametrize("algo", ["gomea", "hc-gomea"])
@pytest.mark.parametrize("fraction", ["2.0", "0.05"])
def test_end_to_end_runs_and_logs_each_deme(algo, fraction, tmp_path):
    """Through the CLI, also through ``run_seeded`` (``-a hc-gomea``)."""
    log = tmp_path / "g.jsonl"
    rc, stdout = _run(["-H", "maj", "-k", "3", "-a", algo, "--gomea-pop-size", "dynamic",
                       "--hc-fraction-of-nn", fraction, "-m", "1500", "-r", "1",
                       "--gomea-log", str(log)])  # fmt: skip
    assert rc == 0 and stdout
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [e for e in events if e["type"] == "invocation_start"]
    ends = [e for e in events if e["type"] == "end"]
    assert starts and len(starts) == len(ends)
    for s, e in zip(starts, ends, strict=True):
        assert s["params"]["pop_size"] == "dynamic"
        assert s["params"]["hc_fraction_of_nn"] == float(fraction)
        assert e["pop_size_dynamic"] == s["deme_pop_size"] >= 2
        assert s["deme_pop_size"] <= max(2, s["deme_budget"])


def test_dynamic_with_a_deme_share_runs_and_logs_the_capped_population(tmp_path):
    """Server round 11 (the arms BASE-D5, -D10 and -D20): ``--gomea-pop-size dynamic`` with
    ``--gomea-deme-share D``. Before the fix, the telemetry of the budget-derived options
    asked for N without the deme's field set and the run stopped with a ValueError. N is the
    dynamic N of the capped deme budget, as with ``--gomea-max-evals-per-deme``."""
    log = tmp_path / "g.jsonl"
    rc, stdout = _run(["-H", "maj", "-k", "3", "-a", "gomea", "--gomea-pop-size", "dynamic",
                       "--gomea-deme-share", "5", "-m", "1500", "-r", "1",
                       "--gomea-log", str(log)])  # fmt: skip
    assert rc == 0 and stdout
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [e for e in events if e["type"] == "invocation_start"]
    ends = [e for e in events if e["type"] == "end"]
    assert len(starts) == len(ends) >= 2
    for s, e in zip(starts, ends, strict=True):
        assert s["params"]["deme_share"] == 5 and s["deme_cap"] == 300
        assert s["deme_budget"] == min(300, s["max_evals"])
        assert e["pop_size_dynamic"] == s["deme_pop_size"] <= max(2, s["deme_budget"])
        assert e["ledger"]["requests"] <= 300
