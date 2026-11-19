"""Tests for `--exemplar-selection lexicase` (NOT A PORT, pyMoses extension, server round 13,
arm BASE-LEX): epsilon-lexicase exemplar selection (`metapopulation/lexicase.py`).

The identity of every setting without the option is pinned by
`tests/gomea/test_exemplar_selection_identity.py`.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.table.table import ITable, OTable
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.metapopulation import merging
from pymoses.moses.metapopulation.lexicase import (
    EXEMPLAR_SELECTION_CHOICES,
    lexicase_select,
    lexicase_stats,
    mad,
    median,
)
from pymoses.moses.metapopulation.metapop_params import metapop_parameters
from pymoses.moses.metapopulation.metapopulation import metapopulation
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import contin_bscore
from pymoses.moses.scoring.scoring_base import bscore_base
from pymoses.util.rng import MT19937RandGen, randGen

_REPO = Path(__file__).resolve().parents[3]
_TRAIN = _REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"
_N_ROWS = 40  # rows of _TRAIN


@pytest.fixture(autouse=True)
def _restore_logger(tmp_path, monkeypatch):
    """`moses_exec` (the CLI tests) sets the process-wide logger's level and file; restore
    them so later test modules see the defaults. Keep `moses.log` out of the repo."""
    from pymoses.util.logger import logger

    monkeypatch.chdir(tmp_path)
    level, fname = logger().get_level(), logger().get_filename()
    yield
    logger().set_level(level)
    logger().set_filename(fname)


class _ScriptedRng:
    """`randint_n` returns the scripted values (0 when the script is empty) and records
    each call's `n`."""

    def __init__(self, *values: int) -> None:
        self.values = list(values)
        self.calls: list[int] = []

    def randint_n(self, n: int) -> int:
        self.calls.append(n)
        v = self.values.pop(0) if self.values else 0
        assert 0 <= v < n
        return v


def _bs(*errors: float) -> list[float]:
    """A behavioural score from per-row errors (bscore entry = -error)."""
    return [-e for e in errors]


# ------------------------------------------------------------ median and MAD


def test_median_and_mad():
    assert median([3.0]) == 3.0
    assert median([5.0, 1.0, 3.0]) == 3.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert mad([0.0, 10.0, 10.0, 11.0, 11.0]) == 1.0
    assert mad([1.0, 1.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        median([])


# ------------------------------------------------------------ the selection rule

# A is the best on cases 0-3 by far; B to E are close to each other, and better than A
# on case 4. On cases 0-3 epsilon is MAD([0, 10, 10, 11, 11]) = 1, so only A survives;
# on case 4 epsilon is MAD([1, 0, 0, 0, 0]) = 0, so B to E survive.
_MOST = [
    _bs(0, 0, 0, 0, 1),
    _bs(10, 10, 10, 10, 0),
    _bs(10, 10, 10, 10, 0),
    _bs(11, 11, 11, 11, 0),
    _bs(11, 11, 11, 11, 0),
]


def test_the_candidate_best_on_most_cases_is_selected_and_the_search_stops():
    rng = _ScriptedRng(0)  # the first case is case 0
    assert lexicase_select(_MOST, rng) == (0, 1)
    # one draw for the case; no draw for the pick, since one candidate remains
    assert rng.calls == [5]


def test_a_case_where_the_others_are_better_keeps_them_all_then_picks_at_random():
    # case 4 first: B to E survive; on the other cases all four are within
    # (lowest = 10) + 1, so all cases are used and the pick is random among the four
    rng = _ScriptedRng(4, 0, 0, 0, 0, 2)
    assert lexicase_select(_MOST, rng) == (3, 5)
    assert rng.calls == [5, 4, 3, 2, 1, 4]  # five cases, then the pick among four


def test_the_candidate_best_on_most_cases_wins_about_four_in_five_with_the_run_rng():
    rng = MT19937RandGen(7)
    wins = sum(lexicase_select(_MOST, rng)[0] == 0 for _ in range(2000))
    assert 0.76 < wins / 2000 < 0.84


def test_epsilon_keeps_the_candidates_within_the_mad_of_the_best_one():
    # one case: median 3, deviations [3, 2, 1, 0, 7, 7, 7], MAD 3: keep errors <= 0 + 3
    pool = [_bs(e) for e in (0, 1, 2, 3, 10, 10, 10)]
    for k in range(4):
        rng = _ScriptedRng(0, k)
        assert lexicase_select(pool, rng) == (k, 1)
        assert rng.calls == [1, 4]  # the case, then the pick among the 4 survivors


def test_epsilon_is_the_mad_of_the_whole_pool_not_of_the_survivors():
    # case 0: MAD([0, 0, 100, 100, 100]) = 0 keeps candidates 0 and 1.
    # case 1: MAD of the whole pool [3, 0, 0, 10, 20] is 3, so candidate 0 (error 3)
    # stays with candidate 1 (error 0). The MAD of the survivors only (1.5) would drop it.
    pool = [_bs(0, 3), _bs(0, 0), _bs(100, 0), _bs(100, 10), _bs(100, 20)]
    rng = _ScriptedRng(0, 0, 0)
    assert lexicase_select(pool, rng) == (0, 2)
    assert rng.calls == [2, 1, 2]


def test_ties_end_with_a_random_pick_after_the_last_case():
    pool = [_bs(1, 2, 3)] * 4
    for k in range(4):
        rng = _ScriptedRng(0, 0, 0, k)
        assert lexicase_select(pool, rng) == (k, 3)
    rng = MT19937RandGen(3)
    picks = {lexicase_select(pool, rng)[0] for _ in range(200)}
    assert picks == {0, 1, 2, 3}


def test_a_pool_of_one_uses_no_case_and_no_draw():
    rng = _ScriptedRng()
    assert lexicase_select([_bs(1, 2, 3)], rng) == (0, 0)
    assert rng.calls == []


def test_the_pool_needs_full_behavioural_scores():
    with pytest.raises(AssertionError, match="one entry for each training row"):
        lexicase_select([_bs(1, 2), _bs(1)], _ScriptedRng())
    with pytest.raises(AssertionError, match="one entry for each training row"):
        lexicase_select([[], []], _ScriptedRng())
    with pytest.raises(AssertionError, match="not finite"):
        lexicase_select([_bs(1.0), _bs(math.inf)], _ScriptedRng())


def test_the_stats_report():
    s = lexicase_stats()
    assert s.report()["selections"] == 0 and s.report()["cases_used_mean"] is None
    s.note_selection(3, 10, 40)
    s.note_selection(5, 20, 40)
    s.note_recompute(7, 0.5)
    assert s.report() == {
        "n_cases": 40, "selections": 2, "cases_used_mean": 4.0, "cases_used_max": 5,
        "pool_size_mean": 15.0, "bscore_recomputes": 7, "bscore_recompute_cpu_s": 0.5,
    }  # fmt: skip


# ------------------------------------------------------------ the metapopulation


class _RowBscore(bscore_base):
    """Per-row errors for each program text; any other program gets errors of 50."""

    def __init__(self, errors: dict[str, list[float]], n: int) -> None:
        super().__init__()
        self._errors = errors
        self._n = n

    def __call__(self, tr):
        e = self._errors.get(combo_tree_to_str(tr).strip(), [50.0] * self._n)
        return [-x for x in e]

    def best_possible_bscore(self):
        return [0.0] * self._n


_P = ["and($1 $2)", "or($1 $2)", "and($1 !$2)"]
_ERR = {_P[0]: [0.0, 0.0, 5.0], _P[1]: [3.0, 3.0, 0.0], _P[2]: [4.0, 4.0, 0.0]}


def _mp(selection="lexicase", revisit=0, programs=_P, errors=None):
    pa = metapop_parameters()
    pa.exemplar_selection = selection
    pa.revisit = revisit
    sc = behave_cscore(_RowBscore(errors or _ERR, 3))
    return metapopulation([str_to_combo_tree(p) for p in programs], sc, pa)


def _text(sct):
    return combo_tree_to_str(sct.get_tree()).strip()


def test_off_by_default():
    assert metapop_parameters().exemplar_selection == "boltzmann"
    assert EXEMPLAR_SELECTION_CHOICES == ("boltzmann", "lexicase")
    assert _mp("boltzmann")._lexicase is None
    assert _mp()._lexicase is not None


def test_the_revisit_rule_and_the_visit_count(monkeypatch):
    # with revisit 0 each candidate is selected once, then no exemplar is left
    monkeypatch.setattr(randGen(), "randint_n", lambda n: 0)
    mp = _mp()
    chosen = [mp.select_exemplar() for _ in range(3)]
    assert sorted(_text(c) for c in chosen) == sorted(_P)
    assert all(mp._visited_exemplars.get(c) == 1 for c in chosen)
    assert mp.select_exemplar() is None
    rep = mp._lexicase.report()
    # the pool shrinks by one at each selection; the "no exemplar" call is not one
    assert rep["selections"] == 3 and rep["pool_size_mean"] == 2.0
    # revisit 1: each candidate twice
    mp = _mp(revisit=1)
    chosen = [_text(mp.select_exemplar()) for _ in range(6)]
    assert sorted(chosen) == sorted(_P * 2)
    assert mp.select_exemplar() is None


def test_the_first_case_decides_with_the_run_rng(monkeypatch):
    # case 0 first: MAD([0, 3, 4]) = 1 keeps candidate 0 alone (one draw)
    draws = iter([0])
    monkeypatch.setattr(randGen(), "randint_n", lambda n: next(draws))
    assert _text(_mp().select_exemplar()) == _P[0]
    # case 2 first: MAD([5, 0, 0]) = 0 keeps candidates 1 and 2; on cases 1 and 0 the
    # MAD is 1 and both stay (errors 3 and 4 <= 3 + 1); after the last case the pick
    # (draw 1 of 2) is candidate 2
    draws = iter([2, 0, 0, 1])
    monkeypatch.setattr(randGen(), "randint_n", lambda n: next(draws))
    mp = _mp()
    assert _text(mp.select_exemplar()) == _P[2]
    assert mp._lexicase.report()["cases_used_max"] == 3


def test_the_size_one_shortcut_is_unchanged(monkeypatch):
    def no_draw(*a):
        raise AssertionError("the size() == 1 shortcut draws nothing")

    monkeypatch.setattr(randGen(), "randint_n", no_draw)
    monkeypatch.setattr(randGen(), "randdouble", no_draw)
    mp = _mp(programs=[_P[0]])
    assert _text(mp.select_exemplar()) == _P[0]
    assert mp.select_exemplar() is None
    assert mp._lexicase.report()["selections"] == 0


def test_the_pool_is_the_boltzmann_pool(monkeypatch):
    # a candidate whose score is not finite is never in the pool
    errors = {**_ERR, _P[0]: [math.inf, 0.0, 0.0]}
    monkeypatch.setattr(randGen(), "randint_n", lambda n: 0)
    mp = _mp(errors=errors)
    chosen = {_text(mp.select_exemplar()) for _ in range(2)}
    assert chosen == {_P[1], _P[2]}
    assert mp.select_exemplar() is None
    assert mp._lexicase.report()["pool_size_mean"] == 1.5


def test_the_pool_keeps_a_candidate_whose_boltzmann_weight_underflows(monkeypatch):
    """Backlog 12.2: the float32 Boltzmann weight exp((p - best) * 100 / T) is 0 for a large
    score gap, so the roulette never picks that candidate; the lexicase pool (the finite
    entries of ``probs``) still has it."""
    errors = {**_ERR, _P[2]: [10.0, 10.0, 10.0]}  # a score gap of more than 20
    seen = {}
    mod = sys.modules[metapopulation.__module__]
    real = mod.roulette_select

    def spy(probs, total, rng):
        seen["probs"] = list(probs)
        return real(probs, total, rng)

    monkeypatch.setattr(mod, "roulette_select", spy)
    mp = _mp("boltzmann", errors=errors)
    order = [_text(sct) for sct in mp._scored_trees]
    mp.select_exemplar()
    w = dict(zip(order, seen["probs"], strict=True))
    assert w[_P[2]] == 0.0 and w[_P[0]] > 0.0 and w[_P[1]] > 0.0
    monkeypatch.setattr(randGen(), "randint_n", lambda n: 0)
    mp = _mp(errors=errors)
    mp.select_exemplar()
    assert mp._lexicase.report()["pool_size_mean"] == 3.0


def test_boltzmann_draws_from_the_roulette_only(monkeypatch):
    def no_draw(*a):
        raise AssertionError("boltzmann draws no case")

    monkeypatch.setattr(randGen(), "randint_n", no_draw)
    mp = _mp("boltzmann")
    assert mp.select_exemplar() is not None


# ------------------------------------------------------------ the scorer's errors


def test_the_regression_bscore_has_one_scaled_entry_per_row_and_nothing_else():
    rows = [(0.5 * i - 2.0,) for i in range(9)]
    ys = [3.0 + 2.0 * x * x for (x,) in rows]
    it = ITable()
    for r in rows:
        it.push_back(r)
    b = contin_bscore(OTable(ys), it)
    b.set_linear_scaling(True)
    bs = b(str_to_combo_tree("*($1 $1)"))
    # a + b*f fits y exactly: one entry per row, all zero; no complexity entry
    assert len(bs) == len(rows)
    assert all(abs(x) < 1e-5 for x in bs)
    bs = b(str_to_combo_tree("$1"))
    assert len(bs) == len(rows) and all(x <= 0 for x in bs)


# ------------------------------------------------------------ the CLI


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = moses_exec(["moses", *argv])
    return rc, buf.getvalue()


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_REG = ["-i", str(_TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", "2400", "-c", "3", "-V1", "-l", "WARN"]  # fmt: skip
_HC_LM = ["-m", "1500", "-a", "hc", "--linear-scaling", "1", "--const-opt", "lm",
          "--const-opt-guard", "val", "--const-opt-val", str(_TRAIN)]  # fmt: skip
_GOMEA = ["-m", "1000", "-a", "gomea", "--gomea-pop-size", "dynamic", "--hc-fraction-of-nn",
          "0.05", "--gomea-contin", "copy", "--gomea-deme-store", "accepted",
          "--linear-scaling", "1", "--gomea-max-evals-per-deme", "200"]  # fmt: skip
_LEX = ["--exemplar-selection", "lexicase"]


def _n_evals(stdout: str) -> int:
    return int(next(x for x in stdout.splitlines() if x.startswith("n_evals:")).split()[1])


@pytest.mark.parametrize("algo", [_HC_LM, _GOMEA], ids=["hc-lm", "gomea"])
def test_every_metapopulation_member_has_its_behavioural_score_when_on(algo, monkeypatch):
    seen = {"merges": 0, "empty": 0}
    orig = merging.metapopulation.merge_demes

    def merge_demes(self, all_demes, reps, *a, **k):
        r = orig(self, all_demes, reps, *a, **k)
        seen["merges"] += 1
        seen["empty"] += sum(len(c.get_bscore()) != _N_ROWS for c in self.get_trees())
        return r

    monkeypatch.setattr(merging.metapopulation, "merge_demes", merge_demes)
    rc, _ = _run([*_REG, *algo, *_LEX])
    assert rc == 0 and seen["merges"] >= 2 and seen["empty"] == 0
    # without the option the merge keeps no behavioural score (the option is what adds them)
    seen.update(merges=0, empty=0)
    rc, _ = _run([*_REG, *algo])
    assert rc == 0 and seen["merges"] >= 2 and seen["empty"] > 0


def test_recomputes_are_not_evaluations_and_the_budget_holds(tmp_path):
    ev = tmp_path / "events.jsonl"
    rc, out = _run([*_REG, *_GOMEA, *_LEX, "--gomea-log", str(ev)])
    assert rc == 0
    events = _events(ev)
    lex = [e for e in events if e["type"] == "lexicase"]
    assert len(lex) == 1 and lex[0]["bscore_recomputes"] > 0 and lex[0]["selections"] >= 2
    # n_evals is exactly the evaluations of the demes' ledgers: the recomputes are not in it
    ledger = sum(e["ledger"]["requests"] for e in events if e["type"] == "end")
    assert _n_evals(out) == ledger <= 1000


def test_hc_lm_keeps_the_budget(tmp_path):
    ev = tmp_path / "events.jsonl"
    rc, out = _run([*_REG, *_HC_LM, *_LEX, "--gomea-log", str(ev)])
    assert rc == 0 and _n_evals(out) <= 1500
    (lex,) = [e for e in _events(ev) if e["type"] == "lexicase"]
    assert lex["n_cases"] == _N_ROWS and lex["selections"] >= 2
    assert 1 <= lex["cases_used_mean"] <= lex["cases_used_max"] <= _N_ROWS
    assert lex["pool_size_mean"] >= 1 and lex["bscore_recompute_cpu_s"] >= 0


def test_plain_hc_runs_without_a_log():
    rc, out = _run([*_REG, "-m", "600", "-a", "hc", "--linear-scaling", "1", *_LEX])
    assert rc == 0 and _n_evals(out) <= 600


def test_the_same_seed_gives_the_same_run(tmp_path):
    runs = []
    for i in range(2):
        ev = tmp_path / f"events{i}.jsonl"
        rc, out = _run([*_REG, *_HC_LM, *_LEX, "--gomea-log", str(ev)])
        (lex,) = [e for e in _events(ev) if e["type"] == "lexicase"]
        lex.pop("bscore_recompute_cpu_s")
        runs.append((rc, out, lex))
    assert runs[0] == runs[1]
    # and the option changes the run
    rc, out = _run([*_REG, *_HC_LM])
    assert out != runs[0][1]


def test_no_lexicase_event_when_off(tmp_path):
    ev = tmp_path / "events.jsonl"
    rc, _ = _run([*_REG, *_HC_LM, "--gomea-log", str(ev)])
    assert rc == 0 and not [e for e in _events(ev) if e["type"] == "lexicase"]


def test_the_runner_reads_the_event(tmp_path):
    sys.path.insert(0, str(_REPO / "tools" / "gomea"))
    import moses_sr_runner as runner

    ev = tmp_path / "events.jsonl"
    assert runner.lexicase_summary(ev) is None
    ev.write_text('{"type": "end", "ledger": {"requests": 5}}\n')
    assert runner.lexicase_summary(ev) is None
    ev.write_text(
        '{"type": "end"}\n{"type": "lexicase", "selections": 4, "bscore_recomputes": 9}\n'
    )
    assert runner.lexicase_summary(ev) == {"selections": 4, "bscore_recomputes": 9}


# ------------------------------------------------------------ refusals and the option name


def test_refuses_other_values():
    with pytest.raises(ValueError, match="--exemplar-selection must be one of"):
        _run([*_REG, "-m", "50", "--exemplar-selection", "tournament"])


def test_refuses_a_problem_other_than_it():
    with pytest.raises(ValueError, match="lexicase applies to regression only"):
        _run(["-H", "pa", "-k", "3", "-m", "50", *_LEX])


def test_refuses_a_boolean_table(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("y,x1\n0,0\n1,1\n")
    with pytest.raises(ValueError, match="lexicase applies to regression only"):
        _run([f"-i{p}", "-uy", "-m50", *_LEX])


def test_refuses_a_discretized_target():
    with pytest.raises(ValueError, match="scorer discretize_contin_bscore"):
        _run([*_REG, "-m", "50", "--discretize-threshold", "0.5", *_LEX])


def test_boltzmann_is_accepted_everywhere():
    rc, _ = _run(["-H", "pa", "-k", "3", "-m", "200", "--exemplar-selection", "boltzmann"])
    assert rc == 0


def test_exemplar_still_parses_as_before(monkeypatch):
    """`--exemplar` (classic `-e`) and its abbreviations still name the exemplar option."""
    import pymoses.moses.main.problem_params as pp_mod

    seen = []
    orig = pp_mod.problem_params.parse_options

    def spy(self, args):
        seen.append((list(args.exemplars_str), args.exemplar_selection))
        return orig(self, args)

    monkeypatch.setattr(pp_mod.problem_params, "parse_options", spy)
    for flag in (["--exemplar", "or($1 $2)"], ["--exemplar=or($1 $2)"], ["--exempl", "or($1 $2)"],
                 ["-e", "or($1 $2)"]):  # fmt: skip
        seen.clear()
        rc, _ = _run(["-H", "pa", "-k", "3", "-m", "50", *flag])
        assert rc == 0 and seen == [(["or($1 $2)"], "boltzmann")], flag
    seen.clear()
    rc, _ = _run(["-H", "pa", "-k", "3", "-m", "50", "--exemplar-sel", "boltzmann"])
    assert rc == 0 and seen == [([], "boltzmann")]
