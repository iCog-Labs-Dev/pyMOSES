"""Long demes for promising exemplars (Phase 1af, backlog item 19):
``--gomea-long-deme-cap``/``--gomea-long-deme-rule``/``--gomea-long-deme-prob``.

With the cap at 0 (the default) the parameters, telemetry and runs are unchanged;
``random`` with p = 0 draws nothing and is the same run as the option off.
"""

from __future__ import annotations

import json

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.gomea.long_deme import LongDemePolicy
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen
from tests.gomea.test_integration import _cli_parser, _events, _run
from tests.gomea.test_optimizer import _iscorer


def _gp(*argv: str) -> gomea_parameters:
    return _gomea_params_from_args(_cli_parser(True).parse_args(list(argv)))


class _Tree:
    """A metapopulation entry: a combo tree and its penalized score."""

    def __init__(self, tree: str, penalized: float) -> None:
        self.tree = str_to_combo_tree(tree)
        self.penalized = penalized

    def get_tree(self):
        return self.tree

    def get_penalized_score(self) -> float:
        return self.penalized


class _Metapop:
    def __init__(self, *trees: _Tree) -> None:
        self.trees = list(trees)

    def get_trees(self):
        return self.trees


# ---------------------------------------------------------------------------
# Option parsing.
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_left_out_of_the_telemetry():
    gp = _gp("-a", "gomea", "--gomea-max-evals-per-deme", "250")
    assert (gp.long_deme_cap, gp.long_deme_rule, gp.long_deme_prob) == (0, "top", 0.0)
    d = gp.as_dict()
    assert not any(k.startswith("long_deme") for k in d)
    assert gomea_optimization(optim_parameters(), gp).long_deme is None


def test_options_parse_and_reach_gomea_parameters():
    gp = _gp(
        "-a", "gomea", "--gomea-max-evals-per-deme", "250", "--gomea-long-deme-cap", "2000",
        "--gomea-long-deme-rule", "random", "--gomea-long-deme-prob", "0.25",
    )  # fmt: skip
    assert (gp.long_deme_cap, gp.long_deme_rule, gp.long_deme_prob) == (2000, "random", 0.25)
    assert gp.as_dict()["long_deme_cap"] == 2000
    pol = gomea_optimization(optim_parameters(), gp).long_deme
    assert (pol.cap, pol.rule, pol.prob) == (2000, "random", 0.25)


@pytest.mark.parametrize(
    "bad",
    [
        ["--gomea-long-deme-cap", "-1"],
        ["--gomea-long-deme-cap", "2000", "--gomea-long-deme-rule", "best"],
        ["--gomea-long-deme-cap", "2000", "--gomea-long-deme-rule", "random",
         "--gomea-long-deme-prob", "1.5"],
        # a probability without the random rule, or without the cap
        ["--gomea-long-deme-cap", "2000", "--gomea-long-deme-prob", "0.2"],
        ["--gomea-long-deme-rule", "random", "--gomea-long-deme-prob", "0.2"],
    ],
)  # fmt: skip
def test_bad_values_are_rejected(bad):
    with pytest.raises(ValueError):
        _gp("-a", "gomea", "--gomea-max-evals-per-deme", "250", *bad)


def test_long_cap_needs_a_normal_cap_and_gomea():
    with pytest.raises(ValueError, match="normal per-deme cap"):
        _gp("-a", "gomea", "--gomea-long-deme-cap", "2000")
    with pytest.raises(ValueError, match="needs -a gomea"):
        _gp("-a", "hc-capped", "--gomea-max-evals-per-deme", "250",
            "--gomea-long-deme-cap", "2000")  # fmt: skip
    # the cap from --gomea-deme-share counts as a normal cap
    gp = _gp("-a", "gomea", "-m", "1000", "--gomea-deme-share", "4",
             "--gomea-long-deme-cap", "600")  # fmt: skip
    assert (gp.max_evals_per_deme, gp.long_deme_cap) == (250, 600)


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------


def test_top_rule_picks_the_best_exemplar_once_only():
    a, b, c = _Tree("and($1 $2)", -2.0), _Tree("or($1 $2)", -1.0), _Tree("$1", float("nan"))
    mp = _Metapop(a, b, c)
    pol = LongDemePolicy(2000, "top")
    assert pol.decide(mp, a) is False  # not the best
    assert pol.decide(mp, b) is True  # the best (NaN scores are ignored)
    assert pol.decide(mp, b) is False  # it already had its long deme
    # a new best gets one; an equal best (a tie) counts as best
    d = _Tree("or($1 $3)", -0.5)
    e = _Tree("or($2 $3)", -0.5)
    mp.trees += [d, e]
    assert pol.decide(mp, b) is False
    assert pol.decide(mp, d) is True
    assert pol.decide(mp, e) is True
    assert pol.decide(mp, c) is False  # a NaN exemplar is never best
    assert (pol.n_demes, pol.n_long) == (7, 3)


def test_top_rule_keys_exemplars_by_their_tree():
    pol = LongDemePolicy(2000, "top")
    first = _Tree("and($1 $2)", -1.0)
    assert pol.decide(_Metapop(first), first) is True
    # another object holding the same program is the same exemplar
    same = _Tree("and($1 $2)", -1.0)
    assert pol.decide(_Metapop(same), same) is False


def _random_choices(seed: int, prob: float, n: int = 200) -> tuple[list[bool], int]:
    randGen().seed(seed)
    pol = LongDemePolicy(2000, "random", prob)
    t = _Tree("and($1 $2)", -1.0)
    out = [pol.decide(_Metapop(t), t) for _ in range(n)]
    return out, randGen().randint()


def test_random_rule_is_deterministic_for_a_seed():
    a, ra = _random_choices(5, 0.3)
    b, rb = _random_choices(5, 0.3)
    c, _ = _random_choices(6, 0.3)
    assert a == b and ra == rb
    assert a != c
    assert 30 < sum(a) < 90  # about 0.3 * 200


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_random_rule_draws_nothing_at_p_0_or_1(prob):
    choices, after = _random_choices(9, prob)
    randGen().seed(9)
    assert after == randGen().randint()  # the run's RNG was not touched
    assert choices == [prob == 1.0] * len(choices)


def test_off_draws_nothing_and_keeps_the_normal_cap():
    opt = gomea_optimization(optim_parameters(), gomea_parameters(max_evals_per_deme=20))
    randGen().seed(3)
    t = _Tree("and($1 $2)", -1.0)
    opt.prepare_deme(_Metapop(t), t)
    after = randGen().randint()
    randGen().seed(3)
    assert after == randGen().randint()
    assert opt.deme_cap() == 20


# ---------------------------------------------------------------------------
# The cap a long deme runs with.
# ---------------------------------------------------------------------------


def _run_deme(make_boolean_rep, tmp_path, max_evals: int, long: bool) -> tuple[int, dict]:
    rep = make_boolean_rep("and", 3)
    isc, deme = _iscorer(rep, 3), instance_set(rep.fields())
    log = tmp_path / f"ev-{max_evals}-{long}.jsonl"
    # max_gens/stall high enough that the deme stops on a budget, not on a stall
    gp = gomea_parameters(
        max_evals_per_deme=20, long_deme_cap=120, stall_generations=1000,
        max_generations=1000, telemetry_path=str(log),
    )  # fmt: skip
    opt = gomea_optimization(optim_parameters(), gp)
    t = _Tree("and($1 $2)", -1.0 if long else -2.0)
    best = _Tree("or($1 $2)", -1.0)
    opt.prepare_deme(_Metapop(t, best), t)
    randGen().seed(1)
    opt(deme, isc, max_evals, 10**6)
    return deme.n_evals, _events(log)


def test_long_deme_uses_the_long_cap(make_boolean_rep, tmp_path):
    n_long, ev_long = _run_deme(make_boolean_rep, tmp_path, 10**4, True)
    n_norm, ev_norm = _run_deme(make_boolean_rep, tmp_path, 10**4, False)
    assert n_norm <= 20 < n_long <= 120
    start, end = ev_long[0], ev_long[-1]
    assert start["type"] == "invocation_start" and end["type"] == "end"
    for e in (start, end):
        assert e["long"] is True and e["deme_cap_used"] == 120
        assert (e["n_demes"], e["n_long_demes"], e["long_deme_rule"]) == (1, 1, "top")
    assert ev_norm[0]["long"] is False and ev_norm[0]["deme_cap_used"] == 20
    if end["termination_reason"] == "inner_budget":
        assert n_long == 120


def test_long_cap_is_bounded_by_the_evaluations_left(make_boolean_rep, tmp_path):
    n, ev = _run_deme(make_boolean_rep, tmp_path, 50, True)
    assert n <= 50
    assert ev[-1]["long"] is True and ev[-1]["deme_cap_used"] == 120


# ---------------------------------------------------------------------------
# End to end through the CLI.
# ---------------------------------------------------------------------------

_PA3 = ["-H", "pa", "-k", "3", "-a", "gomea", "-m", "1500", "-r", "1",
        "--gomea-max-evals-per-deme", "100"]  # fmt: skip


def test_top_rule_end_to_end(tmp_path):
    log = tmp_path / "events.jsonl"
    rc, _ = _run([*_PA3, "--gomea-long-deme-cap", "600", "--gomea-log", str(log)])
    assert rc == 0
    starts = [e for e in _events(log) if e["type"] == "invocation_start"]
    ends = [e for e in _events(log) if e["type"] == "end"]
    assert len(starts) == len(ends) >= 2
    # the first exemplar is the only program, so the best: its deme is long
    assert starts[0]["long"] is True and starts[0]["deme_cap_used"] == 600
    assert [e["n_demes"] for e in starts] == list(range(1, len(starts) + 1))
    assert sum(e["long"] for e in starts) == starts[-1]["n_long_demes"]
    assert all(e["deme_cap_used"] == (600 if e["long"] else 100) for e in starts)
    assert [e["long"] for e in ends] == [e["long"] for e in starts]


def test_random_p0_is_the_same_run_as_off(tmp_path):
    off_log, p0_log = tmp_path / "off.jsonl", tmp_path / "p0.jsonl"
    rc1, off = _run([*_PA3, "--gomea-log", str(off_log)])
    rc2, p0 = _run([*_PA3, "--gomea-long-deme-cap", "600", "--gomea-long-deme-rule", "random",
                    "--gomea-log", str(p0_log)])  # fmt: skip
    assert rc1 == rc2 == 0 and off == p0
    # the long-deme keys, and the wall-clock ones (as `tools/gomea/identity_check.py`
    # strips them: `max_time` is what is left of the run's time limit)
    strip = ("long", "deme_cap_used", "long_deme_rule", "n_demes", "n_long_demes", "params",
             "max_time")  # fmt: skip
    a = [{k: v for k, v in e.items() if k not in strip and not k.endswith("_s")}
         for e in _events(off_log)]  # fmt: skip
    b = [{k: v for k, v in e.items() if k not in strip and not k.endswith("_s")}
         for e in _events(p0_log)]  # fmt: skip
    for x in (*a, *b):
        x.pop("timings", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert not any(e.get("long") for e in _events(p0_log))
