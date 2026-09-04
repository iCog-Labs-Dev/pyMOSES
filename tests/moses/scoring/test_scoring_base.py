"""C++: moses/moses/scoring/scoring_base.h and scoring_base.cc

No cxxtest exists for scoring_base.h's free-standing pieces in classic.
Direct unit tests of `discrete_complexity_coef` and both
`set_complexity_coef` overloads (the `ctruth_table_bscore` oracle in
test_bscores.py only ever exercises the one-argument `complexity_ratio`
form, since that is all tools/cpp_probe/score_probe.cc's CLI supports --
see its own module comment) plus `bscore_base`'s abstractness.
"""

import math

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import ITable, OTable, Table, type_node
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.moses.scoring.scoring_base import bscore_base, discrete_complexity_coef


def test_bscore_base_cannot_be_instantiated_directly():
    """`operator()`/`best_possible_bscore` are pure virtual in C++
    (scoring_base.h:61,72, `= 0`) -- `abc.ABC` + `@abstractmethod` is the
    Python equivalent, so instantiating the base class directly must
    fail exactly like attempting to instantiate an abstract C++ class
    would fail to compile.
    """
    with pytest.raises(TypeError):
        bscore_base()


def test_discrete_complexity_coef_returns_a_negative_value():
    """scoring_base.h:39-42's own doc comment: "Note: this returns
    NEGATIVE values" is contradicted by scoring_base.cc:38's comment
    ("returns a POSITIVE number") -- the ACTUAL formula (`-log(alphabet)
    / log(p/(1-p))`) returns positive for 0<p<0.5 (log(p/(1-p)) is
    negative there, so the leading `-` flips a negative into a
    positive) -- verified numerically here since the two comments in
    scoring_base.h/.cc disagree with each other.
    """
    coef = discrete_complexity_coef(2, 0.1)
    assert coef > 0.0
    assert coef == pytest.approx(-math.log(2) / math.log(0.1 / 0.9))


def test_set_complexity_coef_one_argument_form():
    """C++: moses/moses/scoring/scoring_base.cc:63 bscore_base::set_complexity_coef(score_t)"""
    ctable = _boolean_ctable()
    bsc = ctruth_table_bscore(ctable)

    assert bsc.get_complexity_coef() == pytest.approx(0.0)  # class default

    bsc.set_complexity_coef(2.0)
    assert bsc.get_complexity_coef() == pytest.approx(0.5)

    # complexity_ratio <= 0.0 is the formula's singularity -- resets to 0.0.
    bsc.set_complexity_coef(0.0)
    assert bsc.get_complexity_coef() == pytest.approx(0.0)
    bsc.set_complexity_coef(-1.0)
    assert bsc.get_complexity_coef() == pytest.approx(0.0)


def test_set_complexity_coef_two_argument_form():
    """C++: moses/moses/scoring/scoring_base.cc:50
    bscore_base::set_complexity_coef(unsigned, float)
    """
    ctable = _boolean_ctable()
    bsc = ctruth_table_bscore(ctable)

    bsc.set_complexity_coef(2, 0.1)
    assert bsc.get_complexity_coef() == pytest.approx(discrete_complexity_coef(2, 0.1))

    # p==0.0 and p==0.5 are the formula's two singularities -- both reset
    # the coefficient to 0.0 rather than evaluating log(0)/log(1) (scoring_
    # base.cc:52-56's own comment).
    bsc.set_complexity_coef(2, 0.0)
    assert bsc.get_complexity_coef() == pytest.approx(0.0)
    bsc.set_complexity_coef(2, 0.5)
    assert bsc.get_complexity_coef() == pytest.approx(0.0)


def test_set_complexity_coef_rejects_other_argument_counts():
    ctable = _boolean_ctable()
    bsc = ctruth_table_bscore(ctable)
    with pytest.raises(TypeError):
        bsc.set_complexity_coef()
    with pytest.raises(TypeError):
        bsc.set_complexity_coef(1, 2, 3)


def _boolean_ctable():
    """A minimal hand-built boolean CTable -- avoids a filesystem fixture
    for these formula-only tests (contrast test_bscores.py's oracle,
    which needs real committed datasets).
    """
    T, F = builtin.logical_true, builtin.logical_false
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type])
    it.push_back((T, T))
    it.push_back((F, F))
    ot = OTable([T, F])
    return Table(ot, it).compressed()
