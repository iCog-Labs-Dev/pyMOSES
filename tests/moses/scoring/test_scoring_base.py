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
from pymetta.moses.moses.types import scored_combo_tree_set
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


def test_call_ensemble_raises_boosting_is_out_of_scope():
    """S10 Task 0: `bscore_base::operator()(const scored_combo_tree_set&)`
    (scoring_base.h:66) -- the base-class body is an unconditional
    `OC_ASSERT(false, ...)` (scoring_base.cc:74-75) that no bscorer in
    this port overrides (boosting stays out of scope). Ported as
    `call_ensemble`, not `__call__` -- see that method's own docstring.
    """
    ctable = _boolean_ctable()
    bsc = ctruth_table_bscore(ctable)
    with pytest.raises(NotImplementedError, match="Ensemble scoring not implemented"):
        bsc.call_ensemble(scored_combo_tree_set())


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


# ---------------------------------------------------------------------------
# bscore_ctable_base.ignore_rows / ignore_rows_at_times
#
# Both were raising gates until the `ss-*` subsample task; both are on the
# real search path now (deme_expander.cc:483,487,508 and ss_bscore.cc:
# 82,109,117). `CTable::remove_rows` itself (table.cc:646) has its own C++
# oracle -- tests/feature_selection/test_ctable_subsample.py against
# tests/fixtures/fs_diversity.json -- so what is tested HERE is the five
# statements of scoring_base.cc:267-281 around it, and above all the
# restore-from-`_all_rows_wrk_ctable` that makes repeated calls with
# DIFFERENT subsets work.
# ---------------------------------------------------------------------------


def _eight_row_ctable():
    """Eight uncompressed rows over two boolean inputs, compressing to four
    distinct input rows of two observations each -- so a row-index subset
    can straddle a compressed row and actually exercise the per-observation
    count arithmetic, not just whole-row deletion.
    """
    T, F = builtin.logical_true, builtin.logical_false
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type])
    outs = []
    for a in (F, T):
        for b in (F, T):
            for out in (T, F):
                it.push_back((a, b))
                outs.append(out)
    return Table(OTable(outs), it).compressed()


def test_ignore_rows_shrinks_the_working_table_and_recomputes():
    """C++: scoring_base.cc:267-281."""
    bsc = ctruth_table_bscore(_eight_row_ctable())
    assert bsc.get_ctable_usize() == 8
    assert bsc.size() == 4
    assert bsc._ctable_weight == 8.0

    bsc.ignore_rows({0, 1, 2, 3})

    assert bsc.get_ctable_usize() == 4
    assert bsc._ctable_weight == 4.0
    assert bsc.size() == len(bsc._wrk_ctable)
    # `_orig_ctable` is the caller's own table and is never touched
    # (scoring_base.cc:323 returns it verbatim).
    assert bsc.get_ctable().uncompressed_size() == 8.0


def test_ignore_rows_restores_all_rows_before_each_removal():
    """`_wrk_ctable = _all_rows_wrk_ctable;` (scoring_cc:274) is the whole
    reason `ignore_rows` can be called once per ss-deme with a DIFFERENT
    subset each time (deme_expander.cc:487) and once more with the empty
    set to restore the whole table (cc:512). Without it the second call
    would subset an already-subset table.
    """
    bsc = ctruth_table_bscore(_eight_row_ctable())

    bsc.ignore_rows({0, 1, 2, 3})
    assert bsc.get_ctable_usize() == 4
    bsc.ignore_rows({4, 5, 6, 7})
    assert bsc.get_ctable_usize() == 4, "second call must start from the FULL table"
    bsc.ignore_rows(set())
    assert bsc.get_ctable_usize() == 8
    assert bsc._ctable_weight == 8.0


def test_ignore_rows_disjoint_subsets_keep_disjoint_data():
    """The two halves an `--ss-n-subsample-demes=2` run discards are
    complementary, so what SURVIVES them must be complementary too.
    """
    bsc = ctruth_table_bscore(_eight_row_ctable())

    bsc.ignore_rows({0, 1, 2, 3})
    first = {row: dict(c) for row, c in bsc._wrk_ctable.items()}
    bsc.ignore_rows({4, 5, 6, 7})
    second = {row: dict(c) for row, c in bsc._wrk_ctable.items()}

    assert first != second
    total = 0.0
    for counts in (first, second):
        for c in counts.values():
            total += sum(c.values())
    assert total == 8.0


def test_ignore_rows_is_a_no_op_when_returning_weighted_scores():
    """`if (_return_weighted_score) return;` (scoring_base.cc:272) -- "Must
    not compress if boosting: the booster keeps track of a weight for each
    row, and so altering the number of rows will just confuse the
    mechanisn" (cc:269-271's own comment, typo included).
    """
    bsc = ctruth_table_bscore(_eight_row_ctable())
    bsc.use_weighted_scores()

    bsc.ignore_rows({0, 1, 2, 3})

    assert bsc.get_ctable_usize() == 8
    assert bsc._ctable_weight == 8.0


def test_ignore_rows_at_times_with_the_empty_set_restores_everything():
    """An untimestamped table yields `get_timestamps() == set()`, so every
    ss-deme on it gets the EMPTY set and this degenerates to "restore all
    rows, recompute" -- classic's behaviour too, which is why
    `--ss-by-time=1` on such a table scores every ss-deme over the whole
    table in both languages.
    """
    bsc = ctruth_table_bscore(_eight_row_ctable())
    bsc.ignore_rows({0, 1, 2, 3})
    assert bsc.get_ctable_usize() == 4

    bsc.ignore_rows_at_times(set())

    assert bsc.get_ctable_usize() == 8
    assert bsc._ctable_weight == 8.0


def test_ignore_rows_at_times_drops_only_the_named_dates():
    """C++: scoring_base.cc:290-305, over `CTable::remove_rows_at_times`
    (table.cc:707).
    """
    from pymetta.comboreduct.table.table import CTable, TimedCounter
    from pymetta.comboreduct.type_checker.type_signature import gen_signature

    T, F = builtin.logical_true, builtin.logical_false
    ct = CTable(olabel="y", ilabels=["x1"], signature=gen_signature(type_node.boolean_type, 1))
    for row, obs in (((T,), ((T, "d1", 2.0), (F, "d2", 1.0))), ((F,), ((T, "d2", 3.0),))):
        cell = TimedCounter()
        for value, date, count in obs:
            cell.add(value, date, count)
        ct[row] = cell

    bsc = ctruth_table_bscore(ct)
    assert bsc.get_ctable_usize() == 6

    bsc.ignore_rows_at_times({"d2"})

    assert bsc.get_ctable_usize() == 2  # only (T,)'s two d1 observations survive
    assert bsc.size() == 1
    assert bsc._ctable_weight == 2.0

    bsc.ignore_rows_at_times(set())
    assert bsc.get_ctable_usize() == 6


def test_ignore_cols_is_still_a_gate():
    """The FEATURE half is still unported -- `CTable.filtered_preserve_idxs`
    does not exist (scoring_base.py's module docstring). Pinned so that
    lifting the row gates is not mistaken for lifting this one.
    """
    bsc = ctruth_table_bscore(_eight_row_ctable())
    with pytest.raises(NotImplementedError, match="filtered_preserve_idxs"):
        bsc.ignore_cols({0})
