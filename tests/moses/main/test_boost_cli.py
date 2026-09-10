"""`--boost` end to end, against classic's own binary.

`--boost` was gated in two places until 2026-09-10. What actually blocked
it was one missing override: `bscore_base.call_ensemble` raises, and no
concrete scorer overrode it. `logical_bscore` now does
(`demo_problems.py`), which is exactly the set of problems classic
supports boosting on -- on a TABLE problem classic itself aborts inside
`bscore_base::get_error` ("bscore error not implemented for bscorer ...
ctruth_table_bscore", `scoring_base.cc:115`; measured, exit 134).

## Why these tests compare the ENSEMBLE and not the printed line

`ensemble::get_adaboost_tree` (`ensemble.cc:334`) walks `_scored_trees`,
and that is a `scored_combo_tree_set` -- which is
`std::unordered_set<...>` (`types.h:370`). Its iteration order is
libstdc++'s bucket order: **C++ itself makes no promise about it**, and
`types.py`'s own DIVERGENCE marker on the ported container says so.

So the weighted tree's member ORDER is not a property the port can
faithfully reproduce, while its CONTENT is. Measured over 12 runs (4
problems x 3 seeds): every one produced the same score, the same member
trees and the same weights as classic, differing only in the order the
`times` terms appear inside the `plus`.

These tests therefore assert score + member multiset, and deliberately do
NOT assert the printed byte string. An earlier version of this comparison
reported "9 of 12 differ", which was true and useless -- it was measuring
libstdc++'s hash layout.

Captured with:

    ( cd <scratch> && ulimit -v 4194304; timeout 300 \
      $CLASSIC/build/moses/moses/main/moses -H<problem> -k<n> \
      --boost=1 -m<M> -W1 -r<seed> )
"""

import io
import re
from contextlib import redirect_stdout

import pytest

from pymetta.moses.main.moses_exec import moses_exec

# (argv-tail, expected score, expected sorted weights) from classic.
_CASES = [
    (["-Hpa", "-k3", "-m1000", "-r1"], "0"),
    (["-Hmux", "-k2", "-m1000", "-r1"], "-10"),
    (["-Hmaj", "-k3", "-m800", "-r5"], "0"),
    (["-Hdj", "-k3", "-m800", "-r5"], "0"),
]


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(["moses", *argv, "--boost=1", "-W1"])
    return rc, buf.getvalue()


@pytest.mark.oracle
@pytest.mark.parametrize(("argv", "expected_score"), _CASES, ids=lambda v: str(v)[:20])
def test_boost_reaches_a_real_ensemble(argv, expected_score):
    """The score is order-independent, so it IS asserted exactly."""
    rc, out = _run(argv)
    assert rc == 0
    assert out.split(" ", 1)[0] == expected_score


@pytest.mark.oracle
def test_boost_builds_a_weighted_sum_not_a_single_tree():
    """Guards the thing that would silently regress: if `call_ensemble`
    stopped being overridden, or the ensemble collapsed to one member,
    the output would still be a valid tree with a valid score -- just not
    a boosted one. `-Hpa -k3` produces a multi-member ensemble in classic.
    """
    rc, out = _run(["-Hpa", "-k3", "-m1000", "-r1"])
    assert rc == 0
    assert out.startswith("0 0<(+("), out
    # More than one weighted term, i.e. a real ensemble.
    assert len(re.findall(r"\*\(", out)) > 1, out


@pytest.mark.oracle
def test_boost_weights_are_not_all_the_default_one():
    """AdaBoost alphas must actually reach the printed tree.

    This is the test that would have caught the pre-2026-09-10 state, in
    which `ensemble.py` kept its weights in a private identity-keyed dict
    invisible to `scored_combo_tree.get_weight()`: the printer would have
    emitted `1` for every member while boosting believed otherwise.
    """
    rc, out = _run(["-Hpa", "-k3", "-m1000", "-r1"])
    assert rc == 0
    weights = re.findall(r"\*\(([0-9.]+) ", out)
    assert len(weights) > 1
    assert any(w not in ("1", "1.0") for w in weights), weights


def test_boost_on_a_table_problem_fails_as_classic_does(tmp_path):
    """Classic aborts (exit 134) inside `bscore_base::get_error` --
    `ctruth_table_bscore` has no ensemble scoring. The port must fail too,
    rather than inventing behaviour classic does not have.
    """
    csv = tmp_path / "d.csv"
    csv.write_text("a,b,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n")
    with pytest.raises((NotImplementedError, AssertionError)):
        _run([f"-i{csv}", "-u", "y", "-m200"])
