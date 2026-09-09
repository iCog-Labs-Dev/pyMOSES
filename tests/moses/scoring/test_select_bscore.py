"""C++: moses/moses/scoring/select_bscore.h and select_bscore.cc

Differential oracle against tests/fixtures/scorers.json's "select" cases,
plus direct unit/boundary/mutation-proof tests.

`__call__` (`operator()`) requires the "mixed" interpreter (contin
inputs, boolean output). Whether that is available is decided at
COLLECTION TIME by `_MIXED_INTERPRETER_AVAILABLE` (probed directly,
below -- not hardcoded), because this file does not own
`pymetta/comboreduct/` and cannot assume which side of a concurrent
mixed-interpreter port landed first:

* Unavailable: `test_call_raises_on_the_missing_mixed_interpreter` runs,
  proving the gate raises cleanly rather than returning a silently wrong
  bscore.
* Available (S14 Task 1 ported `contin_interpreter`/`mixed_interpreter`,
  comboreduct/interpreter/interpreter.py, and wired `interpreter_visitor`
  to route a contin-input row to it): `test_call_matches_recorded_cpp_
  bscore_rows` runs instead, proving `__call__` does not merely complete
  but reproduces classic's own recorded per-row values (`tests/fixtures/
  scorers.json`'s "select" cases carry `bscore_rows`/
  `sum_bscore_weighted`, recorded from classic, previously unused because
  `__call__` used to raise before reaching them).

Both tests are `skipif`-gated on the SAME probe, in opposite directions,
so they can never both run (double-counting) or both skip (silent
coverage loss) for one process -- see the probe's own docstring.

Every OTHER case here tests the constructor's own computed state and the
tree-free methods (`best_possible_bscore`/`worst_possible_bscore`/
`min_improv`), independent of interpreter availability.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table import CTable, type_node
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.scoring.select_bscore import select_bscore

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_CASES = json.loads((_FIXTURE_DIR / "scorers.json").read_text())["select"]
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"


def _contin_ctable(values: list[float]) -> CTable:
    """A small hand-built contin CTable: one input column, weight 1.0 per
    row, each row its own compressed key (all distinct values). Each
    `value` is used as BOTH the row's sole input and its output value --
    fine for these guard/boundary tests, which never inspect the input
    side, only the output distribution `select_bscore` ranks by.
    """
    ct = CTable("y", ["x1"])
    for x in values:
        ct[(x,)] = Counter({x: 1.0})
    ct.set_signature(TypeSignature([type_node.contin_type], type_node.contin_type))
    return ct


# ---------------------------------------------------------------------------
# Differential oracle: everything NOT dependent on tree evaluation
# ---------------------------------------------------------------------------


def test_scorers_fixture_has_select_cases():
    assert len(_CASES) >= 4


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_CASES), ids=sorted(_CASES))
def test_select_bscore_matches_cpp_probe(case_name):
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / case["file"]), case["target_feature"])
    ct = table.compressed()

    bsc = select_bscore(
        ct, case["lower_percentile"], case["upper_percentile"], case["hardness"], case["positive"]
    )

    assert bsc._lower_bound == case["lower_bound"], case_name
    assert bsc._upper_bound == case["upper_bound"], case_name
    assert bsc._effective_length == case["effective_length"], case_name
    assert bsc._weights == case["weights"], case_name

    best = bsc.best_possible_bscore()
    worst = bsc.worst_possible_bscore()
    expected_best = sorted(r["value"] for r in case["best_possible_bscore_rows"])
    expected_worst = sorted(r["value"] for r in case["worst_possible_bscore_rows"])
    assert sorted(best) == expected_best, case_name
    assert sorted(worst) == expected_worst, case_name
    assert bsc.min_improv() == case["min_improv"], case_name


# ---------------------------------------------------------------------------
# __call__ -- gated on a live capability probe, not a hardcoded assumption
# about merge order (see module docstring). [Important 3, branch review]
# ---------------------------------------------------------------------------


def _mixed_interpreter_available() -> bool:
    """Probes the EXACT call both tests below care about -- a genuinely
    contin-input row through `select_bscore.__call__` -- rather than
    inspecting `pymetta.comboreduct.interpreter` internals this file does
    not own. Evaluated once, at collection time, and cached in
    `_MIXED_INTERPRETER_AVAILABLE` below, so
    `test_call_raises_on_the_missing_mixed_interpreter` and
    `test_call_matches_recorded_cpp_bscore_rows` key off the identical
    boolean and can never disagree about which regime they are in.
    """
    ct = _contin_ctable([0.1, 0.5, 0.9])
    bsc = select_bscore(ct, 0.0, 1.0, 1.0, True)
    tr = str_to_combo_tree("0<($1)")
    try:
        bsc(tr)
    except NotImplementedError:
        return False
    return True


_MIXED_INTERPRETER_AVAILABLE = _mixed_interpreter_available()


@pytest.mark.skipif(
    _MIXED_INTERPRETER_AVAILABLE,
    reason="mixed interpreter is available (S14 Task 1 landed); superseded "
    "by test_call_matches_recorded_cpp_bscore_rows below",
)
def test_call_raises_on_the_missing_mixed_interpreter():
    """Proves the gate raises cleanly rather than returning a silently
    wrong bscore -- see select_bscore.py's own module docstring. Runs
    ONLY while `_mixed_interpreter_available()` measures `False`; once
    the mixed interpreter lands for real, `test_call_matches_recorded_
    cpp_bscore_rows` below takes over -- see this module's own docstring
    for why the two cannot both run or both skip.
    """
    ct = _contin_ctable([0.1, 0.5, 0.9])
    bsc = select_bscore(ct, 0.0, 1.0, 1.0, True)
    tr = str_to_combo_tree("0<($1)")
    with pytest.raises(NotImplementedError):
        bsc(tr)


@pytest.mark.skipif(
    not _MIXED_INTERPRETER_AVAILABLE,
    reason="mixed interpreter not yet available (whole-project gap; see "
    "module docstring) -- __call__ still raises NotImplementedError, "
    "covered instead by test_call_raises_on_the_missing_mixed_interpreter",
)
@pytest.mark.parametrize("case_name", sorted(_CASES), ids=sorted(_CASES))
def test_call_matches_recorded_cpp_bscore_rows(case_name):
    """`__call__` was 0% covered before this test: a mutant negating the
    entire returned bscore (`return [-x for x in bs]`) passed the whole
    module, because the only test that ever exercised `__call__`
    (`test_call_now_evaluates_via_mixed_interpreter_instead_of_raising`,
    S14 Task 1's own commit, now superseded by this one) asserted only
    `len(bs) == len(ct)` / `all(isinstance(x, float) for x in bs)` --
    never a single VALUE.

    `tests/fixtures/scorers.json`'s own "select" cases already carry
    `bscore_rows`/`sum_bscore_weighted`, recorded from classic and
    unused until now (`__call__` used to raise before ever reaching
    them). Verified independently against those recorded C++ values: all
    4 "select" cases match exactly.

    Row order is compared as a sorted MULTISET, not positionally: this
    port's `CTable` does not reproduce `std::map`'s sorted iteration
    order -- the same, already-established caveat
    `test_discriminating_bscore.py`'s own oracle documents and works
    around for `prerec`'s row-keyed bscore prefix. `sum_bscore_weighted`
    is checked too, as an order-independent second measurement (the
    negate-everything mutant this test was written to kill flips the sign
    of the sum as well as of every element, so either assertion alone
    already kills it -- both are kept because they are cheap and prove
    different things: per-row values vs. the aggregate `local_moses`
    actually reads).
    """
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / case["file"]), case["target_feature"])
    ct = table.compressed()
    bsc = select_bscore(
        ct, case["lower_percentile"], case["upper_percentile"], case["hardness"], case["positive"]
    )
    tr = str_to_combo_tree(case["tree"])
    bs = bsc(tr)
    assert sorted(bs) == sorted(r["value"] for r in case["bscore_rows"]), case_name
    assert bsc.sum_bscore(bs) == pytest.approx(case["sum_bscore_weighted"]), case_name


# ---------------------------------------------------------------------------
# The output-type guard
# ---------------------------------------------------------------------------


def test_select_bscore_rejects_a_boolean_ctable():
    from pymetta.comboreduct.combo.vertex import builtin

    ct = CTable("y", ["x1"])
    ct[(builtin.logical_true,)] = Counter({builtin.logical_true: 1.0})
    ct.set_signature(TypeSignature([type_node.boolean_type], type_node.boolean_type))
    with pytest.raises(AssertionError, match="contin-valued"):
        select_bscore(ct)


# ---------------------------------------------------------------------------
# hardness is accepted but never used
# ---------------------------------------------------------------------------


def test_hardness_argument_has_no_effect():
    ct = _contin_ctable([x / 10.0 for x in range(1, 10)])
    bsc1 = select_bscore(ct, 0.2, 0.8, hardness=1.0, positive=True)
    bsc2 = select_bscore(ct, 0.2, 0.8, hardness=999.0, positive=True)
    assert bsc1._lower_bound == bsc2._lower_bound
    assert bsc1._upper_bound == bsc2._upper_bound
    assert bsc1.best_possible_bscore() == bsc2.best_possible_bscore()


# ---------------------------------------------------------------------------
# Invalid percentile/bound guards
# ---------------------------------------------------------------------------


def test_invalid_percentiles_raise():
    ct = _contin_ctable([x / 10.0 for x in range(1, 10)])
    with pytest.raises(AssertionError, match="invalid percentiles"):
        select_bscore(ct, 0.8, 0.2)  # lower > upper


def test_percentiles_are_clamped_to_0_1():
    """`lower_percentile`/`upper_percentile` are clamped via
    `fmax(0.0, ...)`/`fmin(1.0, ...)` BEFORE the ordering check -- an
    out-of-range pair that still orders correctly after clamping must
    construct without raising.

    [Minor, branch review] Used to assert nothing beyond "did not raise",
    so a mutant clamping to `(0.1, 0.9)` instead of `(0.0, 1.0)` survived.
    20 evenly-weighted rows (not the 9-row helper call other tests in
    this module use) are needed to DISCRIMINATE the two clamps: with 9
    rows, `0.1*9=0.9` and `0.0*9=0` both land in the SAME first row
    (`ceil` to the first row whose cumulative weight is >= the
    threshold), so a (0.1, 0.9) mutant would have been invisible even
    with `_lower_bound`/`_upper_bound` asserted -- checked directly
    against a live run before writing this. With 20 rows, `0.1*20=2` and
    `0.9*20=18` land on the SECOND and 18th rows respectively, distinct
    from the first and 20th -- so `_lower_bound`/`_upper_bound` under the
    (0.1, 0.9) mutant would be the data's 2nd-smallest/19th-smallest
    values, not its true min/max.
    """
    vals = [x / 100.0 for x in range(1, 21)]
    ct = _contin_ctable(vals)
    bsc = select_bscore(ct, -5.0, 5.0)  # clamps to (0.0, 1.0) -- must not raise
    assert bsc._lower_bound == pytest.approx(min(vals))
    assert bsc._upper_bound == pytest.approx(max(vals))


# ---------------------------------------------------------------------------
# The uninitialized-_lower_bound/_upper_bound pin (constructor's own
# premature reset_weights() call) -- mutation-proof for the DO NOT FIX pin.
# ---------------------------------------------------------------------------


def test_reset_weights_effective_length_is_the_measured_value():
    """NOT a mutation proof on its own: `worst_possible_bscore`'s
    per-row `fabs(n_inside - n_outside)` is IDENTICAL whether a
    single-observation row is classified inside or outside its window
    (the difference is always +/-that row's own weight either way) --
    so `_effective_length` equals the row count for ANY interval choice
    on data where every compressed row has exactly one observation,
    including both the measured pin (`0.0`/`0.0`) and, say, an
    "everything is inside" interval. This is confirmed directly: mutating
    the pin to `(very_worst_score, very_best_score)` does NOT turn this
    assertion red on this test's own data. The REAL proof that `0.0`/
    `0.0` is the right pin is the differential oracle above
    (`test_select_bscore_matches_cpp_probe`'s `_effective_length ==
    case["effective_length"]`), which compares against classic's OWN
    measured (bugged) constructor output, captured with a genuine
    multi-row compressed table where the choice of pin is NOT degenerate
    in this way. This test is kept only as a documented, human-readable
    regression of the actual numeric outcome on simple data.
    """
    ct = _contin_ctable([1.0, 2.0, 3.0, -1.0, -2.0])
    bsc = select_bscore(ct, 0.2, 0.8, 1.0, True)
    assert bsc._effective_length == len(ct)
    assert bsc._weights == [1.0] * len(ct)
