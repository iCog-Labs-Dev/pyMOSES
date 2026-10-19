"""Tests for pymoses.moses.metapopulation.lineage (NOT A PORT, pyMoses
extension): exemplar lineage and `--reserve-empty-every K`."""

import contextlib
import io

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.moses.metapopulation.lineage import exemplar_lineage
from pymoses.moses.metapopulation.metapop_params import metapop_parameters
from pymoses.moses.metapopulation.metapopulation import metapopulation
from pymoses.moses.moses.types import scored_combo_tree, scored_combo_tree_set
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.scoring_base import bscore_base
from pymoses.util.rng import randGen


@pytest.fixture(autouse=True)
def _restore_logger():
    """`moses_exec` (the CLI tests) sets the process-wide logger's level and
    file; restore them so later test modules see the defaults."""
    from pymoses.util.logger import logger

    level, fname = logger().get_level(), logger().get_filename()
    yield
    logger().set_level(level)
    logger().set_filename(fname)


GOOD = str_to_combo_tree("and($1 $2)")
WEAK = str_to_combo_tree("or($1 $2)")
EMPTY = str_to_combo_tree("and")


class _Bscore(bscore_base):
    def __init__(self, scores):
        super().__init__()
        self._scores = scores

    def __call__(self, tr):
        return [self._scores.get(combo_tree_to_str(tr).strip(), -50.0)]

    def best_possible_bscore(self):
        return [0.0]


def _cscorer():
    return behave_cscore(_Bscore({"and($1 $2)": 5.0, "or($1 $2)": 1.0}))


def _text(sct):
    return combo_tree_to_str(sct.get_tree()).strip()


def _mp(k, empty=None):
    pa = metapop_parameters()
    pa.reserve_empty_every = k
    return metapopulation([GOOD, WEAK, EMPTY], _cscorer(), pa, empty_exemplars=empty)


def test_off_by_default():
    assert metapop_parameters().reserve_empty_every == 0
    assert metapopulation([GOOD, EMPTY], _cscorer())._lineage is None


def test_schedule_and_initial_labels():
    lin = exemplar_lineage(3)
    lin.add_initial([GOOD, EMPTY])
    assert lin.initial == {"exemplar1": "and($1 $2)", "exemplar2": "and"}
    assert lin.empty_labels == {"exemplar2"}
    turns = []
    for _ in range(7):
        turns.append(lin.reserved_turn())
        lin.expansion_done()
    assert turns == [False, False, True, False, False, True, False]
    # explicit indices win over the literal check (reduction turns `and` into `true`)
    lin = exemplar_lineage(1)
    lin.add_initial([GOOD, str_to_combo_tree("true")], empty=[1])
    assert lin.empty_labels == {"exemplar2"}
    # no empty exemplar: never reserved
    lin = exemplar_lineage(1)
    lin.add_initial([GOOD, WEAK])
    assert not lin.reserved_turn()
    with pytest.raises(ValueError):
        exemplar_lineage(-1)


def test_reserved_turn_picks_the_empty_lineage_then_falls_back(monkeypatch):
    monkeypatch.setattr(randGen(), "randdouble", lambda: 0.1)
    mp = _mp(1)
    # the empty program scores -50 against GOOD's 5: unrestricted it is never picked
    assert _text(mp.select_exemplar()) == "and"
    assert mp._lineage.current == "exemplar3"
    # the empty program is now visited (revisit 0): the subset is exhausted, so the
    # reserved turn makes the normal choice
    assert _text(mp.select_exemplar()) == "and($1 $2)"
    rep = mp._lineage.report()
    assert rep["reserved_turns"] == 2
    assert rep["reserved_fallbacks"] == 1


def test_every_second_expansion(monkeypatch):
    monkeypatch.setattr(randGen(), "randdouble", lambda: 0.1)
    mp = _mp(2)
    assert _text(mp.select_exemplar()) == "and($1 $2)"
    mp._lineage.expansion_done()
    assert _text(mp.select_exemplar()) == "and"
    mp._lineage.expansion_done()
    rep = mp._lineage.report()
    assert rep["expansions_by_lineage"] == {"exemplar1": 1, "exemplar3": 1}
    assert rep["empty_lineage_expansions"] == 1


def test_merged_programs_inherit_and_injected_get_their_label(monkeypatch):
    monkeypatch.setattr(randGen(), "randdouble", lambda: 0.1)
    mp = _mp(1)
    assert _text(mp.select_exemplar()) == "and"
    child = str_to_combo_tree("and($3 $4)")
    cands = scored_combo_tree_set()
    cs = mp.get_cscorer()
    cands.insert(scored_combo_tree(child, cs=cs.get_cscore(child), bs=list(cs.get_bscore(child))))
    mp.merge_candidates(cands)
    assert mp._lineage.lineage["and($3 $4)"] == "exemplar3"
    # a descendant of the empty program is in the reserved subset
    assert mp._lineage.in_empty_lineage(next(c for c in mp.get_trees() if _text(c) == "and($3 $4)"))
    n = mp.inject_candidates([str_to_combo_tree("or($5 $6)")], "reseed1")
    assert n == 1
    assert mp._lineage.lineage["or($5 $6)"] == "reseed1"
    assert any(_text(c) == "or($5 $6)" for c in mp.get_trees())


def test_inject_without_lineage():
    mp = metapopulation([GOOD, EMPTY], _cscorer())
    size = mp.size()
    mp.inject_candidates([str_to_combo_tree("or($5 $6)")], "reseed1")
    assert mp.size() == size + 1


def _run_mux(tmp_path, *extra):
    from pymoses.moses.main.moses_exec import moses_exec
    from pymoses.moses.metapopulation import metapopulation as mpmod

    got = {}
    orig = mpmod.metapopulation.init

    def init(self, ex):
        got["mp"] = self
        return orig(self, ex)

    mpmod.metapopulation.init = init
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            moses_exec(
                [
                    "moses",
                    "-H",
                    "mux",
                    "-k",
                    "2",
                    "-m",
                    "1500",
                    "-r",
                    "1",
                    "-l",
                    "WARN",
                    "-f",
                    str(tmp_path / "m.log"),
                    "-e",
                    "or(and($3 !$1) $4)",
                    "-e",
                    "and($1 $2 $6)",
                    "-e",
                    "and",
                    *extra,
                ]
            )
    finally:
        mpmod.metapopulation.init = orig
    return got["mp"]


def test_cli_end_to_end(tmp_path):
    assert _run_mux(tmp_path)._lineage is None
    rep = _run_mux(tmp_path, "--reserve-empty-every", "1")._lineage.report()
    # `and` reduces to `true`; moses_main still marks it as the empty program
    assert rep["initial"]["exemplar3"] == "true"
    assert rep["empty_labels"] == ["exemplar3"]
    assert rep["n_expansions"] >= 1
    assert rep["empty_lineage_expansions"] == rep["n_expansions"]


def _parse(argv):
    import argparse

    from pymoses.moses.main.problem_params import problem_params

    pp = problem_params()
    parser = argparse.ArgumentParser()
    pp.add_options(parser)
    pp.add_gomea_options(parser)
    pp.parse_options(parser.parse_args(argv))
    return pp


def test_cli_option():
    assert _parse(["-H", "mux"]).meta_params.reserve_empty_every == 0
    assert _parse(["-H", "mux", "--reserve-empty-every", "4"]).meta_params.reserve_empty_every == 4
    with pytest.raises(ValueError, match="reserve-empty-every"):
        _parse(["-H", "mux", "--reserve-empty-every", "-1"])
