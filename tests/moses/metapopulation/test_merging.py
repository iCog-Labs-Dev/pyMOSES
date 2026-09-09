"""Unit + oracle tests for pymetta.moses.metapopulation.merging.

`metapopulationUTest.cxxtest` covers none of this file's scope (see
CARRY-FORWARD.md: its two tests exercise `remove_dominated`/
`get_nondominated_disjoint`, both `domination.cc`, outside this whole S10
slice). `tools/cpp_probe/merging_probe.cc` fills that gap: it drives the
REAL classic pipeline (a real `Table`/`CTable`/`ctruth_table_bscore`/
`behave_cscore` stack, a real `deme_expander` + `hill_climbing`, and a real
`metapopulation`) over `trainingBooleanXor.dat` with exemplar `"and()"`,
arity 2, seed 1, max_evals 200 -- the IDENTICAL scenario
`tests/fixtures/deme_expander.json`'s own `"xor_empty_and"` case already
captures for S10 Task 4's own oracle -- then calls the real
`metapopulation::merge_demes` and dumps the resulting pool. `_build_dex`
below reproduces that same construction in Python (not by loading that
JSON fixture -- this file needs the LIVE `deme_expander`/`representation`
objects `merge_demes` consumes, not their already-captured trajectory).

## Measured: `keep_top_unique_candidates`'s own deme-size now matches C++

The probe's own `"deme_size"` field (measured: 13, after `merge_demes`
internally sorts+dedups the original 17-instance deme) IS now asserted
against in `test_single_merge_matches_cpp_probe` below.

CORRECTED (sort_demes review, 2026-09-08): the two paragraphs that used to
stand here recorded a measured divergence (17 in this port vs. 13 in
classic) and argued it was harmless. The divergence's ROOT CAUSE has since
been fixed, not merely waived: `sort_demes` used to be ported as
`deme.sort(reverse=True)` (a STABLE sort) on the argument that
`boost::sort`/`std::sort` gives no C++ tie-order for a stable sort to
diverge from. That argument was measured and found FALSE (see `merging.py`
-- the block comment above `_sort` -- and docs/CARRY-FORWARD.md Sec 5b.1):
libstdc++'s `std::sort` deterministically reorders ties, and `sort_demes`
now reproduces that real permutation via `_sort`, a transliteration of
libstdc++'s introsort.

Consequence for THIS scenario: with the old stable sort, hill-climbing's
eight `!$1`/`!$2`-scoring instances and eight `$1`/`$2`-scoring instances
(discovered in an ALTERNATING pattern) stayed in that alternating order, so
`keep_top_unique_candidates`'s purely-ADJACENT-pair dedup scan (`prev_it`/
`it` are neighbours only) found zero adjacent ties to collapse (17 stayed
17). With `_sort` reproducing classic's real `std::sort` permutation, which
groups same-tree instances adjacently often enough to remove 4, this port's
own deme size is now measured to be 13 as well -- an EXACT match, not
merely a smaller upper bound.

The truncation-order-dependence finding from the PRIOR correction (S10
branch review, [Important 2]) still stands, unrelated to the fix above:
`keep_top_unique_candidates`'s own final truncation, `del deme[top_cnd:]`
(see that function's own docstring in `merging.py` for its current line),
runs AFTER the adjacent-dedup scan and keeps only the first `top_cnd`
survivors -- so once `top_cnd` is smaller than the naturally-deduplicated
length, tie-order decides WHICH DISTINCT trees are still present when
`deme_to_trees` reads the deme, before `pot_candidates` ever sees it. Same
multiset, same `top_cnd`: order `$1 $1 $2 $3` -> `['$1','$2','$3']`, but
order `$1 $2 $1 $3` -> `['$1','$2','$1']` -- tree `$3` is lost outright, not
merely reordered. `test_keep_top_unique_candidates_truncation_is_tie_order_dependent`
below pins exactly this.

`top_cnd` in THIS scenario starts at `min(deme.n_evals, len(deme))` =
`min(200, 17)` = 17 (the probe's own `max_evals=200` comfortably exceeds
the 17-instance deme), and neither `max_candidates_per_deme` (default,
unset -- the `>= 0` gate never fires) nor `n_subsample_demes` (default 0)
clips it further; `top_cnd` only ever shrinks afterward via `top_cnd =
min(top_cnd, len(deme))` as the dedup scan itself removes instances, so
the final `del deme[top_cnd:]` never truncates past what dedup already did
on its own -- truncation is a no-op here regardless of which sort produced
the dedup count. The FINAL merged pool -- this task's actual exit
criterion ("merged pool contents and ordering match C++") -- is measured
below to match the probe EXACTLY: same 5 trees, same scores, same order,
same `best_composite_score`/`best_tree`, same `merge_demes` return value,
for BOTH the first merge and a second, repeated merge of an
independently-rebuilt identical deme -- and now the same intermediate deme
size too.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation import merging  # noqa: F401  (binds merge_demes et al.)
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import (
    _to_score_t,
    composite_score,
    scored_combo_tree,
    scored_combo_tree_set,
    very_worst_score,
)
from pymetta.moses.representation.field_set import disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.moses.scoring.scoring_base import bscore_base
from pymetta.util.rng import randGen

_XOR_DATA = "tests/fixtures/table_data/trainingBooleanXor.dat"
_PROBE_PATH = "tools/cpp_probe/merging_probe.cc"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SORT_DEMES_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "sort_demes.json"
_SORT_DEMES_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "sort_demes.manifest.json"
_SORT_DEMES_GOLDEN = json.loads(_SORT_DEMES_FIXTURE_PATH.read_text())
_SORT_DEMES_MANIFEST = json.loads(_SORT_DEMES_MANIFEST_PATH.read_text())


# ---------------------------------------------------------------------------
# Shared real-pipeline helper -- the SAME scenario merging_probe.cc drives.
# ---------------------------------------------------------------------------


def _build_cscorer() -> behave_cscore:
    table = loadTable(_XOR_DATA, "y")
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    bsc.set_complexity_coef(3.5)
    return behave_cscore(bsc)


def _build_dex(cscorer: behave_cscore, seed: int) -> deme_expander:
    """Same recipe as tests/moses/deme/test_deme_expander.py's own
    `_build_dex`, hardcoded to the "xor_empty_and" scenario's own
    parameters (dataset, exemplar "and()", arity 2, max_evals 200) rather
    than reading that file's golden fixture -- this module needs the LIVE
    deme_expander object `merge_demes` consumes, not a captured trajectory.
    """
    rule = logical_reduction(())(2)
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters()
    hc = hill_climbing(optim_parameters(), hc_parameters(widen=False, step=False, cross=False))
    sig = gen_signature(type_node.boolean_type, 2)
    dex = deme_expander(sig, rule, rule, cscorer, hc, params, filter_params)

    exemplar = str_to_combo_tree("and()")
    randGen().seed(seed)
    dex.create_demes(exemplar, 0)
    dex.optimize_demes(200, 86400)
    return dex


class _FixedBscore(bscore_base):
    """Not itself a ported C++ symbol -- a controllable per-tree bscore
    table (same shape as test_metapopulation.py's own `_ProbeBscore` and
    test_domination.py's own `_FixedBscore`), used ONLY by the
    `discard_dominated` end-to-end test below: that test needs specific,
    hand-picked bscore VECTORS to make a domination relationship exact,
    which `_build_cscorer`'s real `ctruth_table_bscore` over the XOR
    dataset does not give direct control over.
    """

    def __init__(self, scores: dict[str, list[float]]) -> None:
        super().__init__()
        self._scores = scores

    def __call__(self, tr):
        return list(self._scores.get(combo_tree_to_str(tr).strip(), []))

    def best_possible_bscore(self):
        return [0.0]


def _pool_trees(mp: metapopulation) -> list[tuple[str, float, int, float]]:
    return [
        (
            combo_tree_to_str(sct.get_tree()).strip(),
            sct.get_score(),
            sct.get_complexity(),
            sct.get_penalized_score(),
        )
        for sct in mp.get_trees()
    ]


# ---------------------------------------------------------------------------
# Oracle: merge_demes, matched against tools/cpp_probe/merging_probe.cc.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_single_merge_matches_cpp_probe():
    """tools/cpp_probe/merging_probe.cc's "single_merge" scenario, captured
    2026-09-08 (build/run recipe in that file's own header comment):

        "pre_size":1, "merge_demes_returned":false, "post_size":5,
        "pool":[
          {"tree":"true","cscore":{"score":-1,...,"penalized_score":-1}},
          {"tree":"!$1", score -1, complexity 1, penalized -1.2857142686843872},
          {"tree":"!$2", score -1, complexity 1, penalized -1.2857142686843872},
          {"tree":"$1",  score -3, complexity 1, penalized -3.2857143878936768},
          {"tree":"$2",  score -3, complexity 1, penalized -3.2857143878936768}],
        "best_composite_score": score -1, complexity 0, penalized -1,
        "best_tree":"true"

    See the module docstring for why the probe's own "deme_size" (13) IS
    now asserted here: it used to be a tie-order artifact this port's old
    stable `sort_demes` didn't reproduce (17 vs. 13), which provably could
    not change the final 5-tree pool either way -- but `sort_demes` now
    reproduces libstdc++'s real permutation, so the intermediate size
    matches exactly too.

    Exact (`==`), not `pytest.approx`: `complexity_penalty`/
    `penalized_score` are genuinely-fractional float32 values
    (`1/3.5`-derived, same reasoning test_deme_expander.py's own module
    docstring gives), and both sides compute them through the SAME
    `score_t`-narrowing arithmetic.
    """
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    assert mp.size() == 1

    dex = _build_dex(cscorer, seed=1)
    assert len(dex._demes[0][0]) == 17  # pre-merge_demes deme size, both languages

    done = mp.merge_demes(dex._demes, dex._reps)

    # Now an EXACT match against the C++ probe's own "deme_size":13 (module
    # docstring) -- sort_demes reproduces libstdc++'s real std::sort
    # permutation, so keep_top_unique_candidates' adjacent-pair dedup scan
    # collapses the same 4 instances classic's own run does.
    assert len(dex._demes[0][0]) == 13

    assert done is False
    assert mp.size() == 5
    assert _pool_trees(mp) == [
        ("true", -1.0, 0, -1.0),
        ("!$1", -1.0, 1, -1.2857142686843872),
        ("!$2", -1.0, 1, -1.2857142686843872),
        ("$1", -3.0, 1, -3.2857143878936768),
        ("$2", -3.0, 1, -3.2857143878936768),
    ]

    best = mp.best_composite_score()
    assert best.get_score() == -1.0
    assert best.get_complexity() == 0
    assert best.get_penalized_score() == -1.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "true"


@pytest.mark.oracle
def test_repeat_merge_of_an_identical_deme_finds_nothing_new():
    """tools/cpp_probe/merging_probe.cc's "repeat_merge" scenario: a
    SECOND, independently-built deme from the identical seed (so identical
    content, a different object) merged into a metapopulation that
    already contains every tree that deme can produce. Captured:
    `"pre_size":5, "merge_demes_returned":false, "post_size":5`.
    Exercises `get_new_candidates`' dedup end to end: `pot_candidates`
    would again be the same 5 trees, all already present, so
    `get_new_candidates` returns an EMPTY set and `merge_demes` takes the
    `candidates.empty(): return False` early exit without touching
    `_scored_trees`/`_best_candidates` again.
    """
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    dex1 = _build_dex(cscorer, seed=1)
    mp.merge_demes(dex1._demes, dex1._reps)
    assert mp.size() == 5

    dex2 = _build_dex(cscorer, seed=1)
    pre = mp.size()
    done = mp.merge_demes(dex2._demes, dex2._reps)

    assert pre == 5
    assert done is False
    assert mp.size() == 5


def test_merging_probe_source_is_the_file_this_oracle_cites():
    """Lightweight provenance guard (no fixture/manifest pipeline for a
    probe this small -- same "probe without a fixture pipeline" precedent
    `metapop_core_probe.cc`/`hcextra_max_count_probe.cc` already set):
    the probe this test file's docstrings cite must actually exist.
    """
    import pathlib

    path = pathlib.Path(_PROBE_PATH)
    assert path.is_file()
    assert "merge_demes" in path.read_text()


# ---------------------------------------------------------------------------
# useful_score_range
# ---------------------------------------------------------------------------


def test_useful_score_range_narrows_to_score_t():
    """S10 branch review, [Important 4]: `useful_score_range`'s own
    `_to_score_t` call (`merging.py:231`) is unguarded -- deleting it
    survives the whole suite, because no existing caller happens to hit a
    `complexity_temperature` where the double-precision result and its
    float32 narrowing diverge in an OBSERVABLE way. `complexity_
    temperature = 6.0` gives `6.0 * 30.0 / 100.0 == 1.8` exactly in
    double precision, but float32's nearest representable value is
    `1.7999999523162842` -- decidable directly by `==`, no
    `pytest.approx` needed (contrast `1.8`, which is expressible as an
    exact double literal here purely because Python parses source
    literals as double in the first place).
    """
    mp = _mp()
    mp._params.complexity_temperature = 6.0
    assert mp.useful_score_range() == 1.7999999523162842
    assert mp.useful_score_range() != 1.8


# ---------------------------------------------------------------------------
# trim_down_deme
# ---------------------------------------------------------------------------


class _Deme(list):
    """Not itself a ported C++ symbol -- a bare `list` of `scored_instance`s
    with a settable `n_evals` (a plain `list` has no `__dict__`, so a bare
    `deme.n_evals = N` raises `AttributeError`). `trim_down_deme`/
    `keep_top_unique_candidates` only ever read `len(deme)`/
    `deme[i].second`/`deme[i].first`/`deme.pop()`/`deme.n_evals`/slice
    assignment -- all satisfied here without building a real
    `instance_set` (and the `field_set` it would need) at all.
    """

    n_evals = 0


def _plain_deme(scores: list[tuple[float, float]]) -> _Deme:
    """Each `(score, complexity_penalty)` pair becomes one entry, in the
    given order -- callers are responsible for handing this an already
    score-sorted (descending) sequence, `trim_down_deme`'s own documented
    precondition.
    """
    return _Deme(
        scored_instance(instance(), composite_score(score, 0, complexity_penalty))
        for score, complexity_penalty in scores
    )


def _mp(cscorer: behave_cscore | None = None) -> metapopulation:
    cscorer = cscorer if cscorer is not None else _build_cscorer()
    return metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())


def test_trim_down_deme_is_a_no_op_below_min_pool_size():
    mp = _mp()
    deme = _plain_deme([(0.0, 0.0)] * 20)  # == _min_pool_size, "not >" (`>=` guard)
    mp.trim_down_deme(deme)
    assert len(deme) == 20


def test_trim_down_deme_keeps_a_score_exactly_at_the_cutoff():
    """Places a value EXACTLY on the `< bot_sc` boundary: `bot_sc` itself
    must be KEPT (strict `<`, not `<=`), and only survives past index
    `_min_pool_size` (index 0 of a real deme can never be evaluated at all
    -- see trim_down_deme's own docstring).
    """
    mp = _mp()
    top_sc = 0.0
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng  # un-narrowed here on purpose -- see below

    # 21 entries: 20 at the top score (never individually checked -- they
    # sit at indices 0..19, and the loop only ever inspects index >= ...
    # no, ACTUALLY every index from size-1 down to 1 is inspected in this
    # port (see trim_down_deme's own docstring) -- what index 0 alone is
    # exempt from. `composite_score`'s own constructor narrows `bot_sc`
    # via `_to_score_t` exactly as `trim_down_deme` narrows its own
    # `top_sc - useful_score_range()` -- both compute the identical
    # double-precision subtraction before narrowing, so the two bit
    # patterns are guaranteed equal.
    deme = _plain_deme([(top_sc, 0.0)] * 20 + [(bot_sc, 0.0)])
    mp.trim_down_deme(deme)

    assert len(deme) == 21  # the boundary value survives -- not `< bot_sc`


def test_trim_down_deme_removes_a_score_strictly_below_the_cutoff():
    mp = _mp()
    top_sc = 0.0
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng

    deme = _plain_deme([(top_sc, 0.0)] * 20 + [(bot_sc - 1.0, 0.0)])
    mp.trim_down_deme(deme)

    assert len(deme) == 20  # the too-low tail entry is gone


def test_trim_down_deme_never_evaluates_index_zero():
    """Index 0 is exempt from the `i > 0` scan (trim_down_deme's own
    docstring) -- but that exemption is UNOBSERVABLE whenever
    `useful_score_range()` is non-negative: `top_sc` (index 0's own
    score) can never itself be `< bot_sc = top_sc - range` when `range >=
    0`, so checking index 0 anyway would never have removed it either
    (confirmed by mutation: `while i > 0` widened to `while i >= 0` did
    NOT change this test's outcome until `complexity_temperature` was
    made negative here). A NEGATIVE `complexity_temperature` makes
    `useful_score_range()` negative, so `bot_sc = top_sc - range > top_sc`
    -- index 0's own score now DOES satisfy `< bot_sc`, and would be
    removed if the scan ever reached it. It must survive anyway.
    """
    mp = _mp()
    mp._params.complexity_temperature = -6.0
    rng = mp.useful_score_range()
    assert rng < 0.0

    # top_sc, read from index 0, is -100.0; bot_sc = -100.0 - rng > -100.0,
    # so index 0's OWN score satisfies "< bot_sc" -- it must survive only
    # because index 0 is never re-checked, not because its score clears
    # the cutoff. Every other entry, well above bot_sc, is unaffected
    # either way -- this test isolates index 0's own exemption alone.
    deme = _plain_deme([(-100.0, 0.0)] + [(0.0, 0.0)] * 20)
    mp.trim_down_deme(deme)
    assert len(deme) == 21  # nothing removed: index 0 exempt despite < bot_sc


def test_trim_down_deme_bot_sc_narrowing_matters_at_the_boundary():
    """S10 branch review, [Important 4]: `bot_sc`'s own `_to_score_t` call
    (`merging.py:264`) is unguarded -- deleting it survives the whole
    suite. `test_trim_down_deme_keeps_a_score_exactly_at_the_cutoff`
    above does NOT catch it: that test's own `top_sc`/`rng` pair happens
    to round UP when narrowed (`_to_score_t(D) > D` there, per its own
    comment), so an entry placed at `_to_score_t(D)` compares the same
    way against both the narrowed and the raw `bot_sc` -- no
    contradiction is possible either way.

    This picks the OPPOSITE rounding direction: `top_sc = -5.0`,
    `complexity_temperature = 0.1` gives `rng == 0.029999999329447746`
    and the raw double `D = top_sc - rng == -5.029999999329448`, whose
    narrowed value `_to_score_t(D) == -5.03000020980835` is STRICTLY LESS
    than `D` -- decidable by `==`/`<`, not `pytest.approx`. An entry
    scored at exactly `_to_score_t(D)` (already float32-exact, so
    `composite_score`'s own narrowing at construction is a no-op on it)
    is `< bot_sc` FALSE when `bot_sc` is correctly narrowed to that SAME
    value (survives), but TRUE when `bot_sc` is left as the raw, less
    negative `D` (removed) -- proven by mutation.
    """
    mp = _mp()
    mp._params.complexity_temperature = 0.1
    top_sc = -5.0
    rng = mp.useful_score_range()
    assert rng == 0.029999999329447746  # pins the exact float32 rng this test depends on

    raw_bot_sc = top_sc - rng
    narrowed_bot_sc = _to_score_t(raw_bot_sc)
    assert raw_bot_sc == -5.029999999329448
    assert narrowed_bot_sc == -5.03000020980835
    assert narrowed_bot_sc < raw_bot_sc  # the rounding direction this test needs

    deme = _plain_deme([(top_sc, 0.0)] * 20 + [(narrowed_bot_sc, 0.0)])
    mp.trim_down_deme(deme)

    assert len(deme) == 21  # survives only because bot_sc is narrowed to this SAME value


# ---------------------------------------------------------------------------
# sort_demes
# ---------------------------------------------------------------------------


def test_sort_demes_sorts_each_deme_descending_in_place():
    mp = _mp()
    deme_a = _plain_deme([(1.0, 0.0), (5.0, 0.0), (3.0, 0.0)])
    deme_b = _plain_deme([(-1.0, 0.0), (-5.0, 0.0)])
    all_demes = [[deme_a], [deme_b]]

    mp.sort_demes(all_demes)

    assert [si.second.get_score() for si in deme_a] == [5.0, 3.0, 1.0]
    assert [si.second.get_score() for si in deme_b] == [-1.0, -5.0]


# ------------------------------------ oracle: sort_demes's std::sort order


def _bit_fs(n: int) -> field_set:
    """`tools/cpp_probe/sort_demes_probe.cc`'s `make_bit_fs`: n single-bit
    knobs, matching `crossover_probe.cc`'s own helper of the same shape
    (see `test_crossover.py`).
    """
    return field_set([disc_spec(2) for _ in range(n)])


def _id_instance(fs: field_set, ident: int) -> instance:
    """Writes `ident` into the bit knobs the comparator never looks at, so
    an instance carries a label that survives any permutation.
    """
    inst = instance(fs.packed_width())
    for i in range(fs.n_bits()):
        fs.set_raw(inst, fs.begin_bit_raw_idx() + i, (ident >> i) & 1)
    return inst


def _instance_id(fs: field_set, inst: instance) -> int:
    ident = 0
    for i in range(fs.n_bits()):
        if fs.get_raw(inst, fs.begin_bit_raw_idx() + i):
            ident |= 1 << i
    return ident


def _bits_for(n: int) -> int:
    """`sort_demes_probe.cc`'s own `bits_for`: enough bits for `n` distinct
    ids.
    """
    return max(n, 2).bit_length() + 1


def _cs(sample: dict) -> composite_score:
    return composite_score(
        sample["score"],
        sample["complexity"],
        sample["complexity_penalty"],
        sample["uniformity_penalty"],
    )


def test_sort_demes_fixture_manifest_matches_current_probe_source():
    """Compiler-free rot detection, the same guard
    `test_crossover.py::test_fixture_manifest_matches_current_probe_source`
    provides for its own fixture: editing `sort_demes_probe.cc` without
    re-running `tools/capture_sort_demes.py` fails here.
    """
    assert _SORT_DEMES_MANIFEST["schema_version"] == 1
    assert _SORT_DEMES_MANIFEST["probe_source"] == "tools/cpp_probe/sort_demes_probe.cc"
    assert _SORT_DEMES_MANIFEST["capture_script"] == "tools/capture_sort_demes.py"
    probe = _REPO_ROOT / "tools" / "cpp_probe" / "sort_demes_probe.cc"
    expected_sha256 = hashlib.sha256(probe.read_bytes()).hexdigest()
    assert _SORT_DEMES_MANIFEST["probe_source_sha256"] == expected_sha256


@pytest.mark.oracle
@pytest.mark.parametrize("case", _SORT_DEMES_GOLDEN["full_sort"], ids=lambda c: c["label"])
def test_sort_matches_libstdcxx_order(case):
    """The measurement that settles this file's own `sort_demes` question
    (module docstring): `_sort` -- `merging.py`'s transliteration of
    libstdc++'s introsort, exercised here through the actual
    `metapopulation.sort_demes` entry point -- reproduces libstdc++'s
    `std::sort(v.begin(), v.end(), std::greater<deme_inst_t>())`
    permutation element for element, at every measured size.
    """
    fs = _bit_fs(_bits_for(case["size"]))
    deme = _Deme(scored_instance(_id_instance(fs, s["id"]), _cs(s)) for s in case["input"])

    _mp().sort_demes([[deme]])

    assert [_instance_id(fs, si.first) for si in deme] == case["order"], case["label"]


def test_sort_is_not_a_stable_sort_above_libstdcxx_threshold():
    """Proves the substitution `sort_demes` used to make (a STABLE
    `deme.sort(reverse=True)`) was wrong for realistic deme sizes, rather
    than asserting it: 17 all-tied elements is one past libstdc++'s own
    `_S_threshold == 16` (bits/stl_algo.h:1807, gcc 16.2.1) -- the smallest
    size at which a quicksort partition runs before the closing insertion
    sort -- and libstdc++'s measured order there genuinely differs from a
    stable descending sort.
    """
    case = next(c for c in _SORT_DEMES_GOLDEN["full_sort"] if c["label"] == "all_tied_17")
    assert case["order"] != list(range(17))  # measured C++: [8, 16, 15, 14, ...]

    fs = _bit_fs(_bits_for(case["size"]))
    stable = _Deme(scored_instance(_id_instance(fs, s["id"]), _cs(s)) for s in case["input"])
    stable.sort(reverse=True)
    assert [_instance_id(fs, si.first) for si in stable] == list(range(17))
    assert [_instance_id(fs, si.first) for si in stable] != case["order"]


def test_sort_matches_a_stable_sort_at_or_below_libstdcxx_threshold():
    """The other side of the same boundary: for `n <= 16`,
    `__introsort_loop`'s `while (last - first > 16)` never runs a
    partition, and the closing `__final_insertion_sort` makes zero moves
    when every comparison ties -- so libstdc++'s real order happens to
    equal a stable sort's for these two sizes. Confirms the boundary is
    exactly 16/17, not merely "small enough in practice".
    """
    for label in ("all_tied_15", "all_tied_16"):
        case = next(c for c in _SORT_DEMES_GOLDEN["full_sort"] if c["label"] == label)
        assert case["order"] == list(range(case["size"])), label


def test_sort_breaks_equal_penalized_scores_by_complexity():
    """`composite_score::operator<` (types.cc:139-148) falls back to
    `complexity > r.complexity` when penalized scores are equal, so a
    HIGHER complexity sorts EARLIER under `<` and therefore LATER under
    `std::greater` -- true regardless of algorithm, since unequal
    complexity means these are NOT ties. Pinned against the measured order
    rather than assumed.
    """
    case = next(
        c for c in _SORT_DEMES_GOLDEN["full_sort"] if c["label"] == "equal_score_diff_complexity_40"
    )
    assert [s["complexity"] for s in case["input"]] == list(range(40))
    assert case["order"] == list(range(40))  # lowest complexity first, exactly


def test_sort_demes_truncation_survivors_match_cpp_probe():
    """Traces the reordering forward to an observable outcome, exactly the
    way S12 did for `resize_deme`'s `std::partial_sort` (module docstring):
    `keep_top_unique_candidates`'s own closing line is an UNCONDITIONAL
    `deme.erase(deme.begin() + top_cnd, deme.end())` (a truncation run
    directly here, matching `tools/cpp_probe/sort_demes_probe.cc`'s own
    `keep_top_unique_forward` group, which needs no combo trees to make
    this point). 100 instances tie completely on score; truncating to the
    top 17 after a REAL `sort_demes` keeps a completely different id set
    than truncating after a stable sort would.
    """
    case = _SORT_DEMES_GOLDEN["keep_top_unique_forward"]
    fs = _bit_fs(_bits_for(case["deme_size"]))
    deme = _Deme(
        scored_instance(_id_instance(fs, i), composite_score(-3.0, 0, 0.0, 0.0))
        for i in range(case["deme_size"])
    )

    _mp().sort_demes([[deme]])
    del deme[case["top_cnd"] :]

    assert [_instance_id(fs, si.first) for si in deme] == case["survivors"]
    assert case["survivors"] != list(range(case["top_cnd"]))  # a stable sort's answer


# ---------------------------------------------------------------------------
# deme_to_trees
# ---------------------------------------------------------------------------


def test_deme_to_trees_dedups_by_tree_structure():
    """17 raw instances, only 5 structurally distinct reduced trees (the
    same "xor_empty_and" scenario the oracle tests above use) --
    `pot_candidates` (a `scored_combo_tree_set`) must collapse them to
    exactly 5, independent of `keep_top_unique_candidates`'s own,
    tie-order-sensitive adjacent dedup (module docstring)."""
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees(real_deme, rep, pot_candidates)

    assert len(pot_candidates) == 5
    assert {combo_tree_to_str(sct.get_tree()).strip() for sct in pot_candidates} == {
        "true",
        "!$1",
        "!$2",
        "$1",
        "$2",
    }


def test_deme_to_trees_skips_very_worst_and_non_finite_scores():
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]

    good = real_deme[0]
    bad_worst = scored_instance(instance(real_deme[1].first), composite_score(very_worst_score, 1))
    bad_nan = scored_instance(instance(real_deme[2].first), composite_score(math.nan, 1))
    deme = [good, bad_worst, bad_nan]

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees(deme, rep, pot_candidates)

    assert len(pot_candidates) == 1


def test_deme_to_trees_copies_the_composite_score_not_the_deme_entry():
    """`inst_csc = inst.second` is a reference into the deme's own
    scored_instance; `copy.copy(inst_csc)` must break that alias --
    mutating the deme's own entry afterward must not reach the stored
    candidate.
    """
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    entry = real_deme[0]
    original_score = entry.second.get_score()

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees([entry], rep, pot_candidates)

    entry.second.set_score(-12345.0)

    (stored,) = list(pot_candidates)
    assert stored.get_score() == original_score
    assert stored.get_score() != -12345.0


# ---------------------------------------------------------------------------
# get_new_candidates
# ---------------------------------------------------------------------------


def test_get_new_candidates_keeps_only_trees_not_already_in_the_pool():
    mp = _mp()
    already = scored_combo_tree_set()
    already.insert(scored_combo_tree(str_to_combo_tree("$1"), composite_score(2.0, 1)))
    mp.merge_candidates(already)  # populate _scored_trees with "$1"

    candidates = scored_combo_tree_set()
    candidates.insert(scored_combo_tree(str_to_combo_tree("$1"), composite_score(2.0, 1)))
    candidates.insert(scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1)))

    fresh = mp.get_new_candidates(candidates)

    assert len(fresh) == 1
    (only,) = list(fresh)
    assert combo_tree_to_str(only.get_tree()).strip() == "$2"


def test_get_new_candidates_copies_not_aliases():
    """`res.insert(cnd)` (merging.cc:483) copies via
    `scored_combo_tree_set`'s own copy-on-insert contract -- mutating the
    ORIGINAL candidate object after the call must not reach the returned
    set's own entry.
    """
    mp = _mp()
    candidates = scored_combo_tree_set()
    original = scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1))
    candidates.insert(original)

    fresh = mp.get_new_candidates(candidates)
    original.get_composite_score().set_score(-999.0)

    (only,) = list(fresh)
    assert only.get_score() == 3.0


# ---------------------------------------------------------------------------
# merge_candidates
# ---------------------------------------------------------------------------


def test_merge_candidates_inserts_independent_copies():
    mp = _mp()  # already seeds _scored_trees with "$1" -- merge_candidates adds a 2nd
    candidates = scored_combo_tree_set()
    original = scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1))
    candidates.insert(original)

    mp.merge_candidates(candidates)
    original.get_composite_score().set_score(-999.0)

    (stored,) = [sct for sct in mp.get_trees() if combo_tree_to_str(sct.get_tree()).strip() == "$2"]
    assert stored.get_score() == 3.0


def test_merge_candidates_discard_dominated_calls_merge_nondominated():
    """P2 metapopulation task: `discard_dominated` is now real
    (`domination.py`) -- this used to be `test_merge_candidates_
    discard_dominated_raises`, pinning a `NotImplementedError`. Reaches
    `merge_nondominated` with an EMPTY candidate set (mirrors the gate's
    own former test shape): must not raise, and an empty merge must leave
    the pool untouched. `merge_nondominated`'s own domination LOGIC (a
    candidate set that actually dominates existing pool members) is
    covered directly, against metapopulationUTest.cxxtest's own oracle,
    by tests/moses/metapopulation/test_domination.py -- this test isolates
    only "does merge_candidates's gate reach it without raising"."""
    mp = _mp()
    before = mp.size()
    mp._params.discard_dominated = True
    mp.merge_candidates(scored_combo_tree_set())
    assert mp.size() == before


# ---------------------------------------------------------------------------
# keep_top_unique_candidates
# ---------------------------------------------------------------------------


def test_keep_top_unique_candidates_removes_an_adjacent_exact_duplicate():
    """Two literally-identical instances (same object, same score) placed
    adjacently must collapse to one: `is_approx_eq` trivially holds (exact
    tie) and the decoded, reduced trees are trivially identical too (same
    instance, same representation).
    """
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    dup = real_deme[1]  # a real "!$1" or "!$2" -scoring instance
    deme = _Deme([dup, scored_instance(instance(dup.first), dup.second)])
    deme.n_evals = len(deme)

    mp.keep_top_unique_candidates([[deme]], [rep])

    assert len(deme) == 1


def test_keep_top_unique_candidates_keeps_tied_score_different_trees():
    """The REAL "xor_empty_and" deme's own indices 1/2 (`!$1`/`!$2`) tie
    EXACTLY on penalized score but decode to structurally different
    trees -- must NOT be removed, even though `is_approx_eq` says they
    tie."""
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    a, b = real_deme[1], real_deme[2]
    assert a.second.get_penalized_score() == b.second.get_penalized_score()
    tree_a = combo_tree_to_str(rep.get_candidate(a.first, True)).strip()
    tree_b = combo_tree_to_str(rep.get_candidate(b.first, True)).strip()
    assert tree_a != tree_b

    deme = _Deme([a, b])
    deme.n_evals = 2
    mp.keep_top_unique_candidates([[deme]], [rep])

    assert len(deme) == 2


def test_keep_top_unique_candidates_top_cnd_zero_empties_the_deme_safely():
    """MEASURED HAZARD in keep_top_unique_candidates' own docstring: real
    C++'s iterator/`!=` loop is undefined behaviour here (`it` at index 1
    is never EQUAL to a target at index 0 while only ever increasing).
    This port's `while it_idx < top_cnd` bound is well-defined for
    `top_cnd == 0` without any special-casing (`1 < 0` is simply `False`),
    falling straight through to the final truncation, `del deme[0:]`.
    `reps=[None]` -- `rep` is bound but never read: the pairwise loop body
    (the only place that calls `get_candidate`) never runs.
    """
    mp = _mp()
    mp._params.max_candidates_per_deme = 0
    deme = _plain_deme([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 3

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 0


def test_keep_top_unique_candidates_top_cnd_one_keeps_only_the_first():
    mp = _mp()
    mp._params.max_candidates_per_deme = 1
    deme = _plain_deme([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 3

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 1
    assert deme[0].second.get_score() == 2.0


def test_keep_top_unique_candidates_n_evals_below_deme_size_caps_top_cnd():
    """"It can happen that the true number of evals is less than the deme
    size" (merging.cc's own comment) -- `n_evals` below `len(deme)` must
    cap `top_cnd`, truncating the excess even with no duplicates at all.
    """
    mp = _mp()
    deme = _plain_deme([(3.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 2

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 2
    assert [si.second.get_score() for si in deme] == [3.0, 2.0]


def test_keep_top_unique_candidates_max_candidates_per_deme_gate():
    """Pins the `max_candidates_per_deme >= 0` gate: the default (-1)
    must NOT limit top_cnd; a non-negative value must."""
    mp = _mp()
    deme = _plain_deme([(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)])
    deme.n_evals = 3
    assert mp._params.max_candidates_per_deme == -1

    mp.keep_top_unique_candidates([[deme]], [None])
    assert len(deme) == 3  # untouched: default gate off


class _FakeRepByIdentity:
    """Not itself a ported C++ symbol -- stands in for a real
    `representation` when a test needs `keep_top_unique_candidates`'
    pairwise dedup scan to see a FEW SPECIFIC, controllable trees rather
    than a live decode off a real knob/field_set setup. Maps each
    `instance` object (by IDENTITY, via `id()` -- a bare `instance` is an
    unhashable `list` subtype, see `instance.py`'s own docstring) to a
    preset, already-reduced combo tree.
    """

    def __init__(self, tree_by_instance_id: dict[int, object]) -> None:
        self._tree_by_instance_id = tree_by_instance_id

    def get_candidate(self, inst: instance, reduce: bool = True) -> object:
        return self._tree_by_instance_id[id(inst)]


def _tied_deme_with_trees(tree_strs: list[str], top_cnd: int) -> tuple[_Deme, _FakeRepByIdentity]:
    """Builds a `_Deme` of `len(tree_strs)` entries, all tied on penalized
    score (so `_is_approx_eq` holds for EVERY adjacent pair -- only tree
    structure decides which pairs collapse), each entry's `.first` a
    distinct `instance()` object decoding (via the returned fake rep) to
    the combo tree named at the same position in `tree_strs`.
    `n_evals` is left at the `_Deme` default (0); `top_cnd` is forced
    directly via `max_candidates_per_deme` so the caller controls
    truncation independently of `len(tree_strs)`.

    `scored_instance.__init__` copies its `i` argument (`instance(i)`,
    the same value-copy `scored_instance.py`'s own docstring documents),
    so the object actually stored at `.first` is NOT the `instance()`
    handed to the constructor -- the identity map below is keyed off each
    entry's OWN `.first` (post-copy), not off a throwaway pre-construction
    instance.
    """
    trees = [str_to_combo_tree(s) for s in tree_strs]
    deme = _Deme(scored_instance(instance(), composite_score(1.0, 0)) for _ in tree_strs)
    deme.n_evals = len(deme)
    rep = _FakeRepByIdentity(
        {id(entry.first): tree for entry, tree in zip(deme, trees, strict=True)}
    )
    return deme, rep


def test_keep_top_unique_candidates_truncation_is_tie_order_dependent():
    """S10 branch review, [Important 2]: pins the counterexample to the
    (corrected, see this module's own docstring) claim that dedup makes
    tie-order structurally irrelevant. Same multiset of four tied-score
    trees (`$1 $1 $2 $3`), same `top_cnd` (3, forced below the
    4-instance deme via `max_candidates_per_deme`) -- only the ORDER of
    the two `$1` duplicates relative to `$2`/`$3` differs between the two
    scenarios below, and that alone decides whether the final
    `del deme[top_cnd:]` (`merging.py:766`) discards a genuine DUPLICATE
    or a DISTINCT tree.

    Order `$1 $1 $2 $3`: the two `$1`s are ADJACENT, so the pairwise scan
    collapses them first (`del deme[1]`, `top_cnd = min(3, 3) = 3`),
    leaving exactly `[$1, $2, $3]` -- the final truncation to 3 removes
    nothing further. Order `$1 $2 $1 $3`: the two `$1`s are NOT adjacent
    (a `$2` sits between them), so the scan finds no adjacent structural
    tie at all (`it_idx` only ever compares NEIGHBOURS) and reaches
    `it_idx == top_cnd == 3` with the deme still at its original length
    4 -- the final `del deme[3:]` then discards `$3` outright, leaving
    `[$1, $2, $1]` with the duplicate `$1` intact instead.
    """
    mp = _mp()

    adjacent_deme, adjacent_rep = _tied_deme_with_trees(["$1", "$1", "$2", "$3"], top_cnd=3)
    mp._params.max_candidates_per_deme = 3
    mp.keep_top_unique_candidates([[adjacent_deme]], [adjacent_rep])
    adjacent_result = [
        combo_tree_to_str(adjacent_rep.get_candidate(si.first, True)).strip()
        for si in adjacent_deme
    ]
    assert adjacent_result == ["$1", "$2", "$3"]

    separated_deme, separated_rep = _tied_deme_with_trees(["$1", "$2", "$1", "$3"], top_cnd=3)
    mp._params.max_candidates_per_deme = 3
    mp.keep_top_unique_candidates([[separated_deme]], [separated_rep])
    separated_result = [
        combo_tree_to_str(separated_rep.get_candidate(si.first, True)).strip()
        for si in separated_deme
    ]
    assert separated_result == ["$1", "$2", "$1"]  # $3 is LOST, not merely reordered


# ---------------------------------------------------------------------------
# recompute_scores_over_whole_dataset / ss_filter -- subsample gate
# ---------------------------------------------------------------------------


def test_ss_filter_default_passes_every_breadth_first_deme():
    mp = _mp()
    assert mp.ss_filter([[], [], []], []) == [True, True, True]


def test_ss_filter_n_subsample_demes_gate_raises():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.ss_filter([[]], [])


def test_recompute_scores_over_whole_dataset_default_is_a_no_op():
    mp = _mp()
    mp.recompute_scores_over_whole_dataset([], [])  # must not raise


def test_recompute_scores_over_whole_dataset_n_subsample_demes_gate_raises():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.recompute_scores_over_whole_dataset([], [])


# ---------------------------------------------------------------------------
# merge_demes gates -- every one pinned with an EMPTY all_demes/reps
# (sort_demes/keep_top_unique_candidates/the deme loop all no-op on an
# empty list, so these isolate exactly the gate under test) except
# merge_callback, which needs at least one real candidate to be reached at
# all (it is checked AFTER the `candidates.empty(): return False` exit).
# ---------------------------------------------------------------------------


def test_merge_demes_do_boosting_gate_raises():
    mp = _mp()
    mp._params.do_boosting = True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_discard_dominated_no_longer_raises_on_empty_demes():
    """P2 metapopulation task: `discard_dominated` is now real
    (`domination.py`) -- this used to be `test_merge_demes_discard_
    dominated_gate_raises`, pinning a `NotImplementedError`. With an EMPTY
    `all_demes`/`reps`, `candidates` ends up empty regardless of
    `discard_dominated`, so this now reaches the exact same
    `candidates.empty(): return False` exit
    `test_merge_demes_default_gates_off_returns_false_on_no_candidates`
    already pins for the all-defaults case -- `discard_dominated` alone no
    longer changes that outcome. The real recompute+remove_dominated LOGIC
    (non-empty candidates that actually get filtered) is covered by
    test_domination.py and
    test_merge_candidates_discard_dominated_calls_merge_nondominated
    above."""
    mp = _mp()
    mp._params.discard_dominated = True
    assert mp.merge_demes([], []) is False


def test_merge_demes_discard_dominated_recompute_and_remove_dominated_end_to_end():
    """Exercises `merge_demes`'s OWN new code (the recompute block + its
    second `remove_dominated` call, merging.cc:298-328) end to end, not
    just `merge_nondominated` in isolation (test_domination.py) or
    `merge_candidates`'s own gate (the test above): one breadth-first deme
    with THREE candidates, "$2" (dominates "$3"), "$3" (dominated by "$2",
    must be trimmed by merge_demes's own `remove_dominated` call before
    ever reaching the cross-metapopulation check) and "$4" (incomparable
    with everything). The pre-existing pool member "$1" is dominated by
    "$2" and must be evicted by the SUBSEQUENT `merge_nondominated` call
    (`merge_candidates`, reached from `merge_demes`)."""
    scores = {
        "$1": [0.0, 0.0],
        "$2": [2.0, 2.0],  # dominates $1 and $3
        "$3": [1.0, 1.0],  # dominated by $2 -- trimmed within merge_demes itself
        "$4": [-1.0, 5.0],  # incomparable with everything
    }
    cscorer = behave_cscore(_FixedBscore(scores))
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    mp._params.discard_dominated = True

    deme = _Deme(
        scored_instance(instance(), composite_score(score, 0))
        for score in (3.0, 2.0, 1.0)  # $2, $3, $4 -- already descending
    )
    deme.n_evals = len(deme)
    trees = [str_to_combo_tree(s) for s in ("$2", "$3", "$4")]
    rep = _FakeRepByIdentity(
        {id(entry.first): tree for entry, tree in zip(deme, trees, strict=True)}
    )

    done = mp.merge_demes([[deme]], [rep])

    assert done is False  # merge_callback is None -- merge_demes's own default return
    trees_in_pool = {combo_tree_to_str(sct.get_tree()).strip() for sct in mp.get_trees()}
    assert trees_in_pool == {"$2", "$4"}  # $1 evicted, $3 trimmed, $2/$4 survive


def test_merge_demes_keep_bscore_gate_raises():
    mp = _mp()
    mp._params.keep_bscore = True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_diversity_enabled_no_longer_raises_on_empty_demes():
    """P2 metapopulation task: `diversity_enabled()` is now real
    (`diversity.py`) -- this used to be `test_merge_demes_diversity_
    enabled_gate_raises`, pinning a `NotImplementedError`. With an EMPTY
    `all_demes`/`reps`, `candidates` ends up empty regardless of
    `diversity_enabled()`, so this now reaches the exact same
    `candidates.empty(): return False` exit
    `test_merge_demes_default_gates_off_returns_false_on_no_candidates`
    already pins for the all-defaults case -- `set_diversity()` itself is
    only ever reached AFTER `merge_candidates`, which an empty
    `candidates` never gets to. Real `set_diversity()` behaviour (an
    actual pool to penalize) is covered by test_diversity.py."""
    mp = _mp()
    mp._params.diversity.pressure = 0.5
    assert mp.diversity_enabled() is True
    assert mp.merge_demes([], []) is False


def test_merge_demes_default_gates_off_returns_false_on_no_candidates():
    """Pins the NEGATIVE case for all four gates at once: with everything
    left at its default, an empty `all_demes` must reach the
    `candidates.empty()` early exit and return False, not raise."""
    mp = _mp()
    assert mp.merge_demes([], []) is False


def test_merge_demes_merge_callback_gate_raises():
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    dex = _build_dex(cscorer, seed=1)
    mp._params.merge_callback = object()  # anything not None
    with pytest.raises(NotImplementedError):
        mp.merge_demes(dex._demes, dex._reps)


def test_merge_demes_n_subsample_demes_gate_raises_before_processing_demes():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.merge_demes([[]], [None])


# ---------------------------------------------------------------------------
# resize_metapop
# ---------------------------------------------------------------------------


def _grow_pool(
    mp: metapopulation, entries: list[tuple[float, int] | tuple[float, int, float]]
) -> None:
    """`scored_combo_tree_set` (`candidates`, below) dedups by TREE
    STRUCTURE, not by score (types.py's own `scored_combo_tree_equal`) --
    every entry here needs its OWN structurally distinct tree
    (`$2`, `$3`, ...; index 1 is the metapopulation's own seed exemplar,
    "$1", already in the pool) or all but the first would collapse into
    one on insert, silently defeating every caller that wants N distinct
    pool entries.

    Each entry is `(score, complexity)` -- `complexity_penalty` defaults
    0.0, so `penalized_score == score` -- or `(score, complexity,
    complexity_penalty)` when a caller specifically needs the two to
    differ (`resize_metapop` sorts/cuts by PENALIZED score throughout;
    S9/S10's own recurring failure mode was a test suite that could never
    have told a raw-vs-penalized-score bug apart, per the task brief's own
    Testing bar).
    """
    candidates = scored_combo_tree_set()
    for i, entry in enumerate(entries, start=2):
        score, complexity, *rest = entry
        complexity_penalty = rest[0] if rest else 0.0
        tree = str_to_combo_tree(f"${i}")
        cs = composite_score(score, complexity, complexity_penalty)
        candidates.insert(scored_combo_tree(tree, cs))
    mp.merge_candidates(candidates)


def test_resize_metapop_is_a_no_op_at_exactly_min_pool_size():
    mp = _mp()
    _grow_pool(mp, [(float(-i), 0) for i in range(19)])  # + the seed "$1" == 20
    assert mp.size() == 20

    mp.resize_metapop()

    assert mp.size() == 20
    assert not hasattr(mp, "_merge_count")  # never even entered the body


def test_resize_metapop_phase1_keeps_a_score_exactly_at_the_cutoff():
    mp = _mp()
    top_sc = mp.best_composite_score().get_score()  # the seed "$1"'s own score
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng

    # 20 padding entries (fill positions 1..20, never individually
    # checked -- phase 1 only scans from index _min_pool_size==20 onward)
    # plus one AT the cutoff (index 20) and one BELOW it (index 21).
    _grow_pool(mp, [(top_sc, 0)] * 20 + [(bot_sc, 0), (bot_sc - 1.0, 0)])
    assert mp.size() == 23

    mp.resize_metapop()

    assert mp.size() == 22  # only the strictly-below-cutoff entry is gone
    scores = sorted(sct.get_penalized_score() for sct in mp.get_trees())
    assert min(scores) == pytest.approx(bot_sc)  # the boundary value, not the one below it


def test_resize_metapop_phase1_uses_penalized_score_not_raw_score():
    """Distinguishes "compares get_penalized_score()" (correct,
    merging.cc:386) from a hypothetical "compares get_score() instead"
    bug: a RAW score high enough to look safe by itself, paired with a
    complexity_penalty large enough to push its PENALIZED score well
    below the cutoff. Removed only if the code truly reads
    `get_penalized_score()`.
    """
    mp = _mp()
    top_sc = mp.best_composite_score().get_score()
    rng = mp.useful_score_range()

    trap_raw_score = top_sc + 1000.0  # would look like the new BEST by raw score alone
    trap_penalty = 1000.0 + rng + 1.0  # penalized_score lands well below bot_sc
    _grow_pool(mp, [(top_sc, 0)] * 20 + [(trap_raw_score, 0, trap_penalty)])
    assert mp.size() == 22

    mp.resize_metapop()

    assert mp.size() == 21  # the high-raw/low-penalized trap entry is gone
    assert all(sct.get_score() != trap_raw_score for sct in mp.get_trees())


def test_resize_metapop_min_pool_size_protects_below_cutoff_scores_inside_the_top_20():
    """S10 branch review, [Important 3]: `_min_pool_size`'s entire purpose
    -- "never trim the top 20" -- is unpinned by every OTHER phase-1 test
    above, because in each of them the first 20 ranked entries are all
    the SAME tied top score, so a mutant `idx = 0` scans exactly the same
    entries the real `idx = self._min_pool_size` does and no test
    notices. This puts a genuinely BELOW-CUTOFF score inside the
    protected window instead: the seed "$1" alone at the top (ranked 0 by
    its own PENALIZED score -- no padding entry ties or beats it here,
    unlike the other phase-1 tests above), then 25 entries all at one
    shared LOW score, strictly below `bot_sc` -- so ranks 1..25 (by
    `sct_score_greater`, best-first) are ALL below the cutoff, and ranks
    1..19 of them sit INSIDE the first-20 window the scan is never
    supposed to enter.

    Correct code (`idx = self._min_pool_size` == 20): the scan starts at
    rank 20, which is already `< worst_score`, so it breaks immediately
    with `idx == 20` and removes ranks 20.. (6 entries) -- keeping the top
    entry plus 19 below-cutoff entries alive purely because they are
    inside the floor, final size 20. Mutant (`idx = 0`): the scan starts
    at rank 0 (the top entry, not below cutoff, `idx` advances to 1), then
    immediately finds rank 1 below cutoff and breaks there, removing ranks
    1.. (25 entries) -- final size 1. The two are far enough apart (20 vs
    1) that no tolerance/approx issue can blur them.
    """
    mp = _mp()
    # The seed's PENALIZED score (not `.get_score()`'s raw score -- this
    # exemplar has a non-zero complexity penalty, `-3.0` raw vs `-3.2857..`
    # penalized here) is what `resize_metapop`'s own `ordered[0].
    # get_penalized_score()` actually reads as `top_score`.
    seed_penalized = mp.best_composite_score().get_penalized_score()
    rng = mp.useful_score_range()
    bot_sc = seed_penalized - rng
    low_sc = bot_sc - 1.0  # strictly below the cutoff, unlike every other phase-1 test's padding

    _grow_pool(mp, [(low_sc, 0)] * 25)  # + seed "$1" == 26, ranks 1..25 all below cutoff
    assert mp.size() == 26

    mp.resize_metapop()

    assert mp.size() == 20  # the first-20 floor, not the cutoff, decided this
    scores = [sct.get_penalized_score() for sct in mp.get_trees()]
    assert scores.count(pytest.approx(low_sc)) == 19  # 19 below-cutoff survivors, inside the floor
    assert max(scores) == pytest.approx(seed_penalized)  # the seed itself, still present


def test_resize_metapop_phase2_evicts_down_to_the_computed_cap():
    """Phase 2 (the cap-based random eviction) never compares scores at
    all -- it evicts a uniformly random SURVIVING pool member regardless
    of value -- so, unlike phase 1's own tests above, a raw-vs-penalized
    score distinction has nothing to exercise here; every padding entry
    below is tied at the same raw score with zero complexity penalty,
    comfortably above the phase-1 cutoff, so ONLY phase 2 runs.
    """
    mp = _mp()
    _grow_pool(mp, [(0.0, 0, ) for _ in range(219)])  # + seed == 220, all tied
    # Use a distinct, nonzero complexity so this scenario is not
    # accidentally penalty-blind (S9/S10's own recurring failure mode --
    # see the task brief's own Testing bar).
    assert mp.size() == 220

    randGen().seed(7)
    mp._params.cap_coef = 0.16
    mp.resize_metapop()

    merge_count = 1  # lazy-initialized from 0 on this, the FIRST call
    cap = mp._params.cap_coef
    cap *= merge_count + 250.0
    cap *= 1 + 2.0 * math.exp(-merge_count / 500.0)
    expected = int(cap)

    assert mp._merge_count == 1
    assert 50 < expected < 220  # sanity: exercises phase 2, no wraparound
    assert mp.size() == expected


def test_resize_metapop_signed_int32_narrowing_matches_cpp_measured_value():
    """CORRECTED (S10 branch review, [Important 1]): this test used to pin
    an explicit UNSIGNED 64-bit mask as "the SAME enormous value ... fed
    to `randint_n`" C++ supposedly received -- MEASURED FALSE, see
    `resize_metapop`'s own docstring for the full correction.
    `MT19937RandGen::randint` takes a plain `int` (`mt19937ar.h:54`), so
    C++'s `randint(popsz - offset)` (both `size_t`, merging.cc:427) hands
    the callee the underflowed subtraction NARROWED to a signed 32-bit
    int, not the raw 64-bit underflow. `_min_pool_size` (20) is strictly
    less than `offset` (50), so a population that survives phase 1 at,
    say, 25 members reaches phase 2 needing to evict below 50 -- reachable,
    not contrived away by construction elsewhere in this module.

    All entries here share one tied score, so phase 1 (whose cutoff logic
    is independently covered above) removes nothing, isolating phase 2 --
    keeps this test from accidentally exercising phase-1 behaviour instead
    of the narrowing it targets.

    The `IndexError` alone does NOT distinguish a correctly-narrowed
    subtraction from an incorrectly-masked one here: for ANY `popsz <
    offset`, `which = offset + randint_n(diff)` lands outside `[0, popsz)`
    either way (`which >= offset > popsz` always, regardless of `diff`'s
    magnitude) -- confirmed by mutation: reverting the mask back to
    `(popsz - offset) & 0xFFFFFFFFFFFFFFFF` leaves this test's
    `pytest.raises(IndexError)` passing unchanged. What DOES distinguish
    them is the actual argument fed to `randint_n`, and the value it
    returns -- both captured directly below, by wrapping (not replacing)
    the real `randGen()` singleton's method, so the RNG stream itself is
    consumed exactly as `resize_metapop` would consume it either way.

    Pinned against the measured C++ probe end to end, not just at the
    `rng.py` layer (contrast `test_rng.py`'s own
    `test_randint_n_matches_cpp_for_negative_n`, which pins all three of
    the probe's values off one continuous stream): seed 1 here makes
    `resize_metapop`'s own single draw the SAME first draw the probe's
    `MT19937RandGen(1).randint(-25)` consumes, so `captured_results` must
    match the probe's measured `22` exactly, through this module's own
    mask and `rng.py`'s own `abs(n)` fix together -- reverting EITHER one
    changes this value.
    """
    mp = _mp()
    # `_mp()`'s own seed ("$1") is scored by the REAL cscorer, not 0.0 --
    # every padding entry is pinned to that SAME score (not a literal
    # 0.0) so the whole 25-entry pool is genuinely tied and phase 1
    # (independently covered above) removes nothing, isolating phase 2.
    seed_score = mp.best_composite_score().get_score()
    _grow_pool(mp, [(seed_score, 0) for _ in range(24)])  # + seed == 25, all tied
    assert mp.size() == 25

    randGen().seed(1)  # matches the compiled probe's own seed
    mp._params.cap_coef = 1e-6  # forces popsz_cap == 0 -- any positive popsz evicts

    rng = randGen()
    original_randint_n = rng.randint_n
    captured_diffs: list[int] = []
    captured_results: list[int] = []

    def spy(n: int) -> int:
        captured_diffs.append(n)
        result = original_randint_n(n)
        captured_results.append(result)
        return result

    rng.randint_n = spy
    try:
        with pytest.raises(IndexError):
            mp.resize_metapop()
    finally:
        rng.randint_n = original_randint_n

    # popsz(25) - offset(50) = -25, narrowed to a SIGNED 32-bit int -- small
    # and negative, not the unsigned-64-bit wraparound this test used to
    # pin (`(25 - 50) & 0xFFFFFFFFFFFFFFFF`, > 2**63).
    assert captured_diffs == [-25]
    # Matches the compiled probe's measured `MT19937RandGen(1).randint(-25)
    # == 22` end to end, through the real `resize_metapop` code path.
    assert captured_results == [22]


# ============================================================ log_best_candidates
#
# No C++ probe of its own: `log_best_candidates`'s ONLY real effect, once
# every `logger()` call is dropped (module docstring's "port control flow,
# not text" section), is WHICH already-ported, already-oracle-verified
# accessor it reaches -- `best_composite_score()` or `_ensemble.
# get_ensemble()` -- based on `best_candidates().empty()`/`do_boosting`.
# Pinned here via spies/stubs monkeypatched onto the instance: a stub that
# raises if called proves a branch is NOT reached; a spy that records a
# call and delegates proves it IS.


def test_log_best_candidates_empty_pool_short_circuits():
    """`best_candidates().empty()` must return before EITHER accessor is
    touched -- even with `do_boosting=True` set, which would otherwise
    select the `_ensemble.get_ensemble()` arm.
    """
    mp = _mp()
    mp._best_candidates.clear()
    assert mp.best_candidates().empty()
    mp._params.do_boosting = True

    def boom():
        raise AssertionError("must not be reached when best_candidates() is empty")

    mp.best_composite_score = boom
    mp._ensemble.get_ensemble = boom

    assert mp.log_best_candidates() is None  # returns quietly, touches neither


def test_log_best_candidates_non_boosting_reaches_best_composite_score_only():
    mp = _mp()
    assert not mp.best_candidates().empty()
    assert mp._params.do_boosting is False

    cscore_calls = []
    original_cscore = mp.best_composite_score

    def cscore_spy():
        cscore_calls.append(True)
        return original_cscore()

    def ensemble_boom():
        raise AssertionError("must not reach the boosting arm when do_boosting is False")

    mp.best_composite_score = cscore_spy
    mp._ensemble.get_ensemble = ensemble_boom

    mp.log_best_candidates()
    assert cscore_calls == [True]


def test_log_best_candidates_boosting_reaches_get_ensemble_only():
    mp = _mp()
    assert not mp.best_candidates().empty()
    mp._params.do_boosting = True

    ensemble_calls = []
    original_get_ensemble = mp._ensemble.get_ensemble

    def ensemble_spy():
        ensemble_calls.append(True)
        return original_get_ensemble()

    def cscore_boom():
        raise AssertionError("must not reach the non-boosting arm when do_boosting is True")

    mp._ensemble.get_ensemble = ensemble_spy
    mp.best_composite_score = cscore_boom

    mp.log_best_candidates()
    assert ensemble_calls == [True]
