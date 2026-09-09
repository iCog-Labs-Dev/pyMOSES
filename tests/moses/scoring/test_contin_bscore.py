"""Oracle and guard tests for pymetta.moses.scoring.bscores.contin_bscore
(S14 Task 3).

The oracle fixture (tests/fixtures/contin_bscore.json) is produced by
tools/cpp_probe/contin_bscore_probe.cc against the real classic library
(see tools/capture_contin_bscore.py) -- not reimplemented by hand. Four
sections: "synthetic" (a small hand-built OTable/ITable, both
`err_function_type`s), "table_default" (classic's OWN default
err_function_type -- constructed with NO explicit eft argument at all --
over a 20-row vendored sample of classic's own trainingRegression.dat),
"table_squared"/"table_abs" (the same table with an EXPLICIT eft), and
"complexity"/"complexity_ratio" (both `set_complexity_coef` overloads).

Like test_bscores.py's own score.json oracle, every numeric comparison
below is EXACT (`==`), not `pytest.approx`: `score_t` is genuinely
narrowed to IEEE-754 binary32 at every point classic's own arithmetic
narrows (`_to_score_t`), so this port's double-precision Python floats
hold the exact same bit pattern (widened) as the values the probe's
`%.17g` JSON round-trips.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table import ITable, OTable, Table
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.moses.types import _to_score_t
from pymetta.moses.scoring.bscores import _contin_complexity_coef, _sq, contin_bscore
from pymetta.moses.scoring.scoring_base import discrete_complexity_coef

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = json.loads((_FIXTURE_DIR / "contin_bscore.json").read_text())
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"
_SAMPLE_CSV = _TABLE_DATA_DIR / "contin_regression_sample.csv"


def _synthetic_it_ot():
    it = ITable()
    for v in (1.0 / 3.0, 2.0 / 3.0, 0.1, -0.7, 1.23456789):
        it.push_back((v,))
    ot = OTable([0.5, 1.5, 0.05, -1.4, 2.469])
    return it, ot


# ---------------------------------------------------------------------------
# synthetic: both err_function_types, hand-built OTable/ITable
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("label,eft", [("squared_error", "squared"), ("abs_error", "abs")])
def test_synthetic_matches_cpp(label, eft):
    it, ot = _synthetic_it_ot()
    tr = str_to_combo_tree("*(2 $1)")
    bsc = contin_bscore(
        ot, it, contin_bscore.squared_error if eft == "squared" else contin_bscore.abs_error
    )

    case = _GOLDEN["synthetic"][label]
    bs = bsc(tr)
    assert bs == case["bscore"]
    assert bsc.best_possible_bscore() == case["best_possible_bscore"]
    assert bsc.min_improv() == case["min_improv"]
    assert bsc.sum_bscore(bs) == case["sum_bscore"]


def test_synthetic_zero_residual_row_is_negative_zero():
    """Row index 3 (it=-0.7, ot=-1.4, tr=*(2 $1)) has res == tar EXACTLY,
    so both err functions produce a genuine `-0.0`, not merely `0.0` --
    pins the JSON-round-trip fix in contin_bscore_probe.cc's own
    `float_json` (a bare "-0" with no decimal point parses back as
    Python's int `0`, silently discardin the sign of zero; see that
    probe's own comment) by checking the SIGN explicitly, which a plain
    `==` against `0.0` cannot do (`-0.0 == 0.0` is `True` in Python).
    """
    it, ot = _synthetic_it_ot()
    tr = str_to_combo_tree("*(2 $1)")
    bsc = contin_bscore(ot, it)
    bs = bsc(tr)
    import math

    assert bs[3] == 0.0
    assert math.copysign(1.0, bs[3]) == -1.0


# ---------------------------------------------------------------------------
# table_default: classic's OWN default err_function_type, no eft argument
# ---------------------------------------------------------------------------


def _load_sample_table() -> Table:
    return loadTable(str(_SAMPLE_CSV), "y")


@pytest.mark.oracle
def test_table_default_matches_cpp_own_default():
    """Acceptance criterion #3: at least one test runs classic's OWN
    defaults, diffed against classic's binary. `contin_bscore(table)` is
    called here with NO `eft` argument at all -- exactly what
    contin_bscore_probe.cc's `table_default` mode does in C++
    (`contin_bscore bsc(table);`), so the default this test exercises is
    classic's real, unconfigured default (squared_error), not a value
    either this test or the port's own code chose.
    """
    table = _load_sample_table()
    case = _GOLDEN["table_default"]
    assert table.size() == case["n_rows"]

    bsc = contin_bscore(table)  # <-- no eft: the actual default
    tr = str_to_combo_tree("+($1 $2)")
    bs = bsc(tr)

    golden = case["default_eft_squared"]
    assert bs == golden["bscore"]
    assert bsc.best_possible_bscore() == golden["best_possible_bscore"]
    assert bsc.min_improv() == golden["min_improv"]
    assert bsc.sum_bscore(bs) == golden["sum_bscore"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "section,eft",
    [("table_squared", contin_bscore.squared_error), ("table_abs", contin_bscore.abs_error)],
)
def test_table_explicit_eft_matches_cpp(section, eft):
    table = _load_sample_table()
    bsc = contin_bscore(table, eft)
    tr = str_to_combo_tree("+($1 $2)")
    bs = bsc(tr)

    label = "squared" if eft == contin_bscore.squared_error else "abs"
    golden = _GOLDEN[section][label]
    assert bs == golden["bscore"]
    assert bsc.sum_bscore(bs) == golden["sum_bscore"]


# ---------------------------------------------------------------------------
# set_complexity_coef: both overloads
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_set_complexity_coef_two_arg_matches_cpp():
    it = ITable()
    it.push_back((1.0,))
    ot = OTable([1.0])
    bsc = contin_bscore(ot, it)

    alphabet_size = _GOLDEN["complexity"]["alphabet_size"]
    for case in _GOLDEN["complexity"]["results"]:
        bsc.set_complexity_coef(alphabet_size, case["stdev"])
        assert bsc.get_complexity_coef() == case["complexity_coef"]


@pytest.mark.oracle
def test_set_complexity_coef_one_arg_matches_cpp():
    it = ITable()
    it.push_back((1.0,))
    ot = OTable([1.0])
    bsc = contin_bscore(ot, it)

    for case in _GOLDEN["complexity_ratio"]["results"]:
        bsc.set_complexity_coef(case["ratio"])
        assert bsc.get_complexity_coef() == case["complexity_coef"]


def test_set_complexity_coef_two_arg_uses_contin_formula_not_discrete():
    """DIVERGENCE-style guard, proven by mutation: `contin_bscore`
    OVERRIDES `bscore_base`'s own two-argument `set_complexity_coef`
    (`discrete_complexity_coef`, gated on `0.0 < p < 0.5`) with a
    COMPLETELY DIFFERENT formula (`_contin_complexity_coef`, gated on
    `stdev > 0.0` alone) -- see `contin_bscore.set_complexity_coef`'s own
    docstring. Picks (alphabet_size=3, stdev=0.4) specifically because it
    lies inside BOTH gates (`0.4 > 0` and `0.0 < 0.4 < 0.5`), so a
    regression that silently delegates to the base formula would not be
    caught by the `stdev <= 0`/`p not in (0, 0.5)` boundary cases alone --
    those return 0.0 under EITHER formula, masking the difference.

    MUTATION: changing this override's `if len(args) == 2:` body to
    `super().set_complexity_coef(*args)` unconditionally (i.e. deleting
    the override's own formula) makes `get_complexity_coef()` return
    `discrete_complexity_coef(3, 0.4)` instead -- confirmed by hand,
    reverted after confirming red/green (see the two computed values
    below differ).
    """
    it = ITable()
    it.push_back((1.0,))
    ot = OTable([1.0])
    bsc = contin_bscore(ot, it)

    bsc.set_complexity_coef(3, 0.4)
    got = bsc.get_complexity_coef()

    contin_formula = _contin_complexity_coef(3, 0.4)
    discrete_formula = discrete_complexity_coef(3, 0.4)
    assert contin_formula != discrete_formula, "test inputs must discriminate the two formulas"
    assert got == contin_formula
    assert got != discrete_formula


# ---------------------------------------------------------------------------
# Constructor-overload dispatch (type-based, matching C++ static resolution)
# ---------------------------------------------------------------------------


def test_table_overload_matches_otable_itable_overload():
    """Not an oracle case (no separate C++ probe run for this beyond the
    table_default/table_squared cases already covered above) -- a direct
    equivalence check that the `Table` overload and the `(OTable,
    ITable)` overload agree on the SAME underlying data, confirming the
    merge itself.
    """
    table = _load_sample_table()
    bsc_from_table = contin_bscore(table, contin_bscore.abs_error)
    bsc_from_parts = contin_bscore(table.otable, table.itable, contin_bscore.abs_error)
    tr = str_to_combo_tree("+($1 $2)")
    assert bsc_from_table(tr) == bsc_from_parts(tr)


def test_scoring_functor_overload():
    """The templated `contin_bscore(const Scoring&, const ITable&, eft)`
    overload (bscores.h:203-209) has ZERO real callers within this port's
    scope (only `demo-problems.cc`'s `sr` problem uses it in classic, and
    `pymetta/moses/main/` is out of this task's scope) -- see the module
    docstring on bscores.py. Direct, non-oracle proof that it builds the
    same `target` a manually-scored `(OTable, ITable)` construction would.
    """
    it, _ = _synthetic_it_ot()

    def score(row):
        return row[0] * 3.0

    bsc = contin_bscore(score, it)
    manual_ot = OTable([score(row) for row in it])
    bsc_manual = contin_bscore(manual_ot, it)

    tr = str_to_combo_tree("$1")
    assert bsc(tr) == bsc_manual(tr)


def test_scoring_functor_overload_rejects_non_itable_second_argument():
    def score(row):
        return row[0]

    with pytest.raises(TypeError, match="ITable"):
        contin_bscore(score, [(1.0,), (2.0,)])


def test_otable_overload_rejects_missing_itable():
    _, ot = _synthetic_it_ot()
    with pytest.raises(TypeError, match="ITable"):
        contin_bscore(ot, None)


def test_unrecognized_first_argument_raises_type_error():
    with pytest.raises(TypeError, match="contin_bscore"):
        contin_bscore(42)


# ---------------------------------------------------------------------------
# best_possible_bscore reads target.size(), not the input row count
# ---------------------------------------------------------------------------


def test_best_possible_bscore_uses_target_size_not_input_size():
    """DIVERGENCE-shaped guard, proven by construction: bscores.cc:163-166
    reads `target.size()`, not `_size` (the INPUT row count) -- see
    `best_possible_bscore`'s own docstring. Builds an `OTable` DELIBERATELY
    longer than its paired `ITable` (never happens for a REAL caller, but
    C++ does not forbid it at construction time either -- see `__init__`'s
    own docstring) so the two sizes are observably different, and checks
    which one `best_possible_bscore` actually reports.

    MUTATION: changing `[0.0] * len(self._target)` to `[0.0] * self._size`
    would make this assert `4 == 2`, failing -- confirmed by hand,
    reverted after confirming red/green.
    """
    it = ITable()
    it.push_back((1.0,))
    it.push_back((2.0,))
    ot = OTable([1.0, 2.0, 3.0, 4.0])
    bsc = contin_bscore(ot, it)
    assert bsc._size == 2
    assert len(bsc.best_possible_bscore()) == 4


def test_call_raises_on_mismatched_cti_target_lengths():
    """DIVERGENCE(C++ moses/moses/scoring/bscores.cc:151): `boost::
    transform(cti, target, ...)` iterates only as many times as `cti` has
    elements and is undefined behaviour if `target` is SHORTER -- this
    port's `zip(..., strict=True)` raises `ValueError` instead, the same
    disclosed strengthening `ctruth_table_bscore.__call__`'s own
    `_best_possible_score` zip already uses (bscores.py).
    """
    it = ITable()
    it.push_back((1.0,))
    it.push_back((2.0,))
    ot = OTable([1.0])  # shorter than it
    bsc = contin_bscore(ot, it)
    tr = str_to_combo_tree("$1")
    with pytest.raises(ValueError, match="shorter"):
        bsc(tr)


# ---------------------------------------------------------------------------
# init: bad err_function_type
# ---------------------------------------------------------------------------


def test_init_rejects_bad_err_function_type():
    """DIVERGENCE(C++ moses/moses/scoring/bscores.h:194-195):
    `default: OC_ASSERT(false);` inside `init`'s `switch` becomes
    `AssertionError` -- unreachable from any real caller (only the two
    declared enum members are ever passed), same convention as every
    other direct `OC_ASSERT(false, ...)` in this port.
    """
    it, ot = _synthetic_it_ot()
    bsc = contin_bscore(ot, it)
    with pytest.raises(AssertionError):
        bsc.init(99)


# ---------------------------------------------------------------------------
# _sq / _contin_complexity_coef: narrowing discriminates a real bug shape
# ---------------------------------------------------------------------------


def test_sq_helper():
    assert _sq(3.0) == 9.0
    assert _sq(-2.5) == 6.25


def test_contin_complexity_coef_narrows_stdev_before_squaring_not_after():
    """Fidelity guard: the CALLER (`contin_bscore.set_complexity_coef`)
    must narrow `stdev` to float32 BEFORE calling `_contin_complexity_
    coef`, mirroring C++'s own `float stdev` PARAMETER, narrowed at the
    call boundary before `contin_complexity_coef`'s body ever runs -- see
    that method's own docstring. Both variants below apply the SAME final
    `_to_score_t` narrowing (via `_contin_complexity_coef` itself in both
    cases), isolating the one difference this test cares about to whether
    `stdev` was pre-narrowed -- NOT whether the result is narrowed (both
    are). `alphabet_size=3, stdev=0.3` is a MEASURED discriminating pair
    (found by search, not assumed): naively trying `stdev=0.1` first, as
    this test originally did, turned out to be a live instance of the
    exact trap S14's own brief warns about -- "the value alone could not
    tell the faithful and broken forms apart" -- the two computations
    happened to round to the SAME float32 result for that particular
    input, which would have made this "guard" pass whether or not the
    narrowing order was implemented correctly at all.
    """
    pre_narrowed = _contin_complexity_coef(3, _to_score_t(0.3))
    not_pre_narrowed = _contin_complexity_coef(3, 0.3)
    assert pre_narrowed != not_pre_narrowed, "0.3 must discriminate narrow-first from not"
    assert pre_narrowed == 0.19775022566318512
    assert not_pre_narrowed == 0.19775021076202393

    # The oracle case for stdev=0.1 (tests/fixtures/contin_bscore.json,
    # "complexity" section, captured via `contin_bscore.set_complexity_
    # coef(3, 0.1)`'s own float32 parameter) still pins this port's real
    # call path bit-for-bit against classic, even though 0.1 itself does
    # not happen to discriminate the two narrowing orders.
    matches_golden = [
        c for c in _GOLDEN["complexity"]["results"] if c["stdev"] == 0.10000000149011612
    ]
    assert len(matches_golden) == 1
    assert _contin_complexity_coef(3, _to_score_t(0.1)) == matches_golden[0]["complexity_coef"]
