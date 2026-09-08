"""Unit + oracle tests for pymetta.moses.moses.local_moses.

## Oracle: `tools/cpp_probe/local_moses_probe.cc`

`MOSESUTest` is S11's real oracle but belongs to Task 2's wiring (full CLI
parameter parsing via `moses_main`), not this task. This module instead
drives the REAL classic `local_moses()` directly, via
`tools/cpp_probe/local_moses_probe.cc` (build/run recipe in its own header
comment) -- a real `Table`/`CTable`/`ctruth_table_bscore`/`behave_cscore`/
`deme_expander`/`hill_climbing`/`metapopulation` stack, `trainingBoolean.dat`
(arity 3), seed exemplar `"$1"`, seed 42, run for 5 generations by calling
`local_moses()` repeatedly with `max_gens` bumped by one each time (see that
probe's own header comment for why per-generation snapshots need this
repeated-call construction, and the measured `advanced` field confirming
`premature_termination` never fires in this scenario, so the construction
is faithful). Its measured JSON output is reproduced as literal expected
values below (`_EXPECTED_GENERATIONS`), the same "probe without a
checked-in fixture" precedent `test_metapopulation.py` already established
for `metapop_core_probe.cc` -- a single fixed 5-generation run is small
enough that a fixture+capture-script pair would be pure overhead.

`test_local_moses_matches_cpp_probe_step_for_step` is the exit-criterion
test: for the fixed seed, the sequence of expansions, `n_evals` after each,
and the metapopulation's best (composite) score after each are compared
by EXACT `==` against the probe's own measured output -- not
`pytest.approx` (`score_t` is genuinely narrowed to IEEE-754 binary32 at
every point classic's own arithmetic narrows, the same reasoning
`test_deme_expander.py`'s own module docstring gives for its identical
choice).

## Unit + mutation tests

The rest of this file drills into `expand_deme`'s own control flow (empty
metapopulation, exhausted exemplars, retry-until-create_demes-succeeds,
stats bookkeeping, `done or mp.empty()`) with lightweight fakes -- no real
C++ pipeline needed to prove these branches, since they are pure
orchestration over whatever `mp`/`dex` return -- and `local_moses`'s own
four termination predicates plus `premature_termination`, each pinned by a
scenario that blocks the loop from ever calling `expand_deme`, with
`expand_deme` itself replaced by a call-counting spy (monkeypatched onto
the module, matching this port's convention of testing free-function
orchestration via monkeypatch rather than dependency injection classic
itself has no equivalent seam for).
"""

from __future__ import annotations

import pymetta.moses.moses.local_moses as lm
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.local_moses import expand_deme, local_moses
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.util.rng import randGen

# ===========================================================================
# Oracle: tools/cpp_probe/local_moses_probe.cc, measured 2026-09-08.
# ===========================================================================

_TABLE_PATH = "tests/fixtures/table_data/trainingBoolean.dat"
_SEED = 42
_ARITY = 3

# Literal, measured output of `( ulimit -v 2097152; timeout 120
# tools/cpp_probe/build/local_moses_probe )`. Every `advanced: True` below
# confirms `stats.n_expansions` grew by EXACTLY one on that call, i.e.
# `premature_termination` never fired -- the probe's own repeated-call
# construction (bumping `max_gens` by one before each `local_moses()` call)
# is therefore observably identical, generation by generation, to one
# continuous run.
_EXPECTED_GENERATIONS = [
    {
        "n_expansions": 1,
        "n_evals": 45,
        "metapop_size": 4,
        "best_score": 0.0,
        "best_complexity": 3,
        "best_complexity_penalty": 0.85714292526245117,
        "best_penalized_score": -0.85714292526245117,
        "best_tree": "or(and($1 $2) !$3)",
    },
    {
        "n_expansions": 2,
        "n_evals": 58,
        "metapop_size": 15,
        "best_score": 0.0,
        "best_complexity": 3,
        "best_complexity_penalty": 0.85714292526245117,
        "best_penalized_score": -0.85714292526245117,
        "best_tree": "or(and($1 $2) !$3)",
    },
    {
        "n_expansions": 3,
        "n_evals": 74,
        "metapop_size": 20,
        "best_score": 0.0,
        "best_complexity": 3,
        "best_complexity_penalty": 0.85714292526245117,
        "best_penalized_score": -0.85714292526245117,
        "best_tree": "or(and($1 $2) !$3)",
    },
    {
        "n_expansions": 4,
        "n_evals": 97,
        "metapop_size": 20,
        "best_score": 0.0,
        "best_complexity": 3,
        "best_complexity_penalty": 0.85714292526245117,
        "best_penalized_score": -0.85714292526245117,
        "best_tree": "or(and($1 $2) !$3)",
    },
    {
        "n_expansions": 5,
        "n_evals": 113,
        "metapop_size": 20,
        "best_score": 0.0,
        "best_complexity": 3,
        "best_complexity_penalty": 0.85714292526245117,
        "best_penalized_score": -0.85714292526245117,
        "best_tree": "or(and($1 $2) !$3)",
    },
]


def _build_pipeline() -> tuple[metapopulation, deme_expander]:
    """Mirrors tools/cpp_probe/local_moses_probe.cc's own construction,
    field for field: `trainingBoolean.dat`, complexity_coef 3.5, seed
    exemplar "$1" (metapopulation's own seed must be directly scorable --
    see the probe's own comment for why a bare "and()" crashes here),
    `hc_parameters(widen=False, step=False, cross=False)`, `perm_ratio=-1.0`.
    """
    table = loadTable(_TABLE_PATH, "y")
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    bsc.set_complexity_coef(3.5)
    cscorer = behave_cscore(bsc)

    exemplar = "$1"
    from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree

    mp = metapopulation(
        str_to_combo_tree(exemplar),
        cscorer,
        metapop_parameters(),
        subsample_deme_filter_parameters(),
    )

    sig = gen_signature(type_node.boolean_type, _ARITY)

    from pymetta.comboreduct.reduct.logical_reduction import logical_reduction

    rule = logical_reduction(())(2)

    dparams = deme_parameters()
    dparams.perm_ratio = -1.0
    fparams = subsample_deme_filter_parameters()

    hc = hill_climbing(optim_parameters(), hc_parameters(widen=False, step=False, cross=False))

    dex = deme_expander(sig, rule, rule, cscorer, hc, dparams, fparams)
    return mp, dex


def test_local_moses_matches_cpp_probe_step_for_step():
    mp, dex = _build_pipeline()
    randGen().seed(_SEED)

    pa = moses_parameters(max_evals=20000, max_score=1000.0)
    stats = moses_statistics()

    for gen, expected in enumerate(_EXPECTED_GENERATIONS, start=1):
        pa.max_gens = gen
        local_moses(mp, dex, pa, stats)

        assert stats.n_expansions == expected["n_expansions"], f"gen {gen}: n_expansions"
        assert stats.n_evals == expected["n_evals"], f"gen {gen}: n_evals"
        assert mp.size() == expected["metapop_size"], f"gen {gen}: metapop_size"

        csc = mp.best_composite_score()
        assert csc.get_score() == expected["best_score"], f"gen {gen}: score"
        assert csc.get_complexity() == expected["best_complexity"], f"gen {gen}: complexity"
        assert (
            csc.get_complexity_penalty() == expected["best_complexity_penalty"]
        ), f"gen {gen}: complexity_penalty"
        assert (
            csc.get_penalized_score() == expected["best_penalized_score"]
        ), f"gen {gen}: penalized_score"
        assert combo_tree_to_str(mp.best_tree()).strip() == expected["best_tree"], (
            f"gen {gen}: best_tree"
        )


_PROBE_SOURCE_SHA256 = "24aa22582c6d5076a6775b97b31ad3d731c62118e740fa6d7aff48b4e0963aea"
# Measured directly (`sha256sum tools/cpp_probe/local_moses_probe.cc`)
# against the EXACT probe binary that produced `_EXPECTED_GENERATIONS`
# above -- if this ever mismatches, the probe was edited without being
# rebuilt/rerun/re-captured, and `_EXPECTED_GENERATIONS` is now stale.


def test_fixture_manifest_matches_current_probe_source():
    """CARRY-FORWARD.md Sec 5: "ci.sh cannot rebuild the probes" -- pin the
    probe source's hash so a future edit that silently changes the probe
    without re-running it (and updating `_EXPECTED_GENERATIONS` above) is
    caught, the same convention test_deme_expander.py/test_hill_climbing.py
    already establish for their own checked-in JSON fixtures' manifests.
    """
    import hashlib
    from pathlib import Path

    probe_path = (
        Path(__file__).resolve().parents[3] / "tools" / "cpp_probe" / "local_moses_probe.cc"
    )
    actual_sha256 = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    assert actual_sha256 == _PROBE_SOURCE_SHA256


# ===========================================================================
# expand_deme -- unit tests over lightweight fakes
# ===========================================================================


class _FakeExemplar:
    """Not itself a ported C++ symbol -- the minimum surface `expand_deme`
    dereferences off a `select_exemplar()` result: `.get_tree()`, and
    `==` against `mp.end()` (`None`, see `metapopulation.end`'s own
    docstring) via ordinary identity/`None`-comparison (no custom
    `__eq__` needed here -- this fake is never compared against a REAL
    `scored_combo_tree`, only ever against `None`).
    """

    def __init__(self, tree: str = "tree") -> None:
        self.tree = tree

    def get_tree(self):
        return self.tree


class _FakeMp:
    def __init__(self, empty: bool = False, exemplar=None, merge_result: bool = False) -> None:
        self._empty = empty
        self._exemplar = exemplar
        self._merge_result = merge_result
        self.select_exemplar_calls = 0
        self.merge_calls: list[tuple] = []

    def empty(self) -> bool:
        return self._empty

    def select_exemplar(self):
        self.select_exemplar_calls += 1
        return self._exemplar

    def end(self):
        return None

    def merge_demes(self, demes, reps) -> bool:
        self.merge_calls.append((demes, reps))
        return self._merge_result


class _FakeDex:
    def __init__(self, create_demes_results=(True,), total_evals: int = 7) -> None:
        self._create_demes_results = list(create_demes_results)
        self._total_evals = total_evals
        self._demes = ["demes-sentinel"]
        self._reps = ["reps-sentinel"]
        self.create_demes_calls: list[tuple] = []
        self.optimize_calls: list[tuple] = []
        self.free_called = False

    def create_demes(self, tree, n_expansions) -> bool:
        self.create_demes_calls.append((tree, n_expansions))
        return self._create_demes_results.pop(0)

    def optimize_demes(self, max_evals, max_time) -> None:
        self.optimize_calls.append((max_evals, max_time))

    def total_evals(self) -> int:
        return self._total_evals

    def free_demes(self) -> None:
        self.free_called = True


def test_expand_deme_empty_metapopulation_returns_true_immediately():
    mp = _FakeMp(empty=True)
    dex = _FakeDex()
    stats = moses_statistics()

    assert expand_deme(mp, dex, 100, 100, stats) is True
    assert mp.select_exemplar_calls == 0
    assert dex.create_demes_calls == []
    assert stats.n_evals == 0
    assert stats.n_expansions == 0


def test_expand_deme_no_exemplar_returns_true():
    """`select_exemplar()` returning `mp.end()` (`None`) -- "all exemplars
    have been visited" (local_moses.cc's own warning text, dropped as
    logging) -- must return `True` without ever touching `dex`.
    """
    mp = _FakeMp(empty=False, exemplar=None)
    dex = _FakeDex()
    stats = moses_statistics()

    assert expand_deme(mp, dex, 100, 100, stats) is True
    assert mp.select_exemplar_calls == 1
    assert dex.create_demes_calls == []


def test_expand_deme_retries_until_create_demes_succeeds():
    """`while (1)` (local_moses.cc:57) -- a `create_demes` failure must
    retry with a FRESH `select_exemplar()` call, not give up.
    """
    exemplar = _FakeExemplar("t1")
    mp = _FakeMp(empty=False, exemplar=exemplar, merge_result=False)
    dex = _FakeDex(create_demes_results=(False, False, True), total_evals=3)
    stats = moses_statistics()

    result = expand_deme(mp, dex, 100, 100, stats)

    assert mp.select_exemplar_calls == 3
    assert dex.create_demes_calls == [("t1", 0), ("t1", 0), ("t1", 0)]
    assert result is False  # done=False (merge_result), mp.empty()=False


def test_expand_deme_normal_path_updates_stats_and_frees_demes():
    exemplar = _FakeExemplar("t1")
    mp = _FakeMp(empty=False, exemplar=exemplar, merge_result=False)
    dex = _FakeDex(create_demes_results=(True,), total_evals=42)
    stats = moses_statistics()
    stats.n_evals = 10
    stats.n_expansions = 2

    result = expand_deme(mp, dex, 500, 3600, stats)

    assert dex.optimize_calls == [(500, 3600)]
    assert stats.n_evals == 10 + 42
    assert stats.n_expansions == 3
    assert mp.merge_calls == [(dex._demes, dex._reps)]
    assert dex.free_called is True
    assert result is False


def test_expand_deme_return_is_done_or_mp_empty_true_via_done():
    exemplar = _FakeExemplar("t1")
    mp = _FakeMp(empty=False, exemplar=exemplar, merge_result=True)
    dex = _FakeDex(create_demes_results=(True,))
    stats = moses_statistics()

    assert expand_deme(mp, dex, 100, 100, stats) is True


def test_expand_deme_return_is_done_or_mp_empty_true_via_mp_empty():
    """`mp.empty()` is read a SECOND time at the very end, AFTER merging --
    a fake whose `empty()` reports `True` unconditionally exercises the
    `or mp.empty()` half even with `done=False`.
    """
    exemplar = _FakeExemplar("t1")
    mp = _FakeMp(empty=True, exemplar=exemplar, merge_result=False)
    # NOTE: `mp.empty()` is checked at entry too (returns True there) --
    # use a two-stage fake so entry sees non-empty but the FINAL check
    # sees empty, exercising line 92's own `mp.empty()`, not line 51's.
    calls = {"n": 0}

    def empty():
        calls["n"] += 1
        return calls["n"] > 1  # False on the entry check, True thereafter

    mp.empty = empty
    dex = _FakeDex(create_demes_results=(True,))
    stats = moses_statistics()

    assert expand_deme(mp, dex, 100, 100, stats) is True


# ===========================================================================
# local_moses -- the four termination predicates + premature_termination
# ===========================================================================


class _PredicateMp:
    def __init__(self, best_score: float = -1.0) -> None:
        self._best_score = best_score

    def best_score(self) -> float:
        return self._best_score


def test_not_max_evals_blocks_the_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(lm, "expand_deme", lambda *a, **k: calls.append(1))

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=5, max_gens=-1, max_score=1000.0)
    stats = moses_statistics()
    stats.n_evals = 5  # already AT the budget -- `5 < 5` is False

    local_moses(mp, object(), pa, stats)
    assert calls == []


def test_not_max_gens_blocks_the_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(lm, "expand_deme", lambda *a, **k: calls.append(1))

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=10000, max_gens=3, max_score=1000.0)
    stats = moses_statistics()
    stats.n_expansions = 3  # `3 != 3` is False

    local_moses(mp, object(), pa, stats)
    assert calls == []


def test_not_max_score_blocks_the_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(lm, "expand_deme", lambda *a, **k: calls.append(1))

    mp = _PredicateMp(best_score=100.0)
    pa = moses_parameters(max_evals=10000, max_gens=-1, max_score=100.0)
    stats = moses_statistics()

    local_moses(mp, object(), pa, stats)  # `100.0 < 100.0` is False
    assert calls == []


def test_not_max_time_blocks_the_loop(monkeypatch):
    """`stats.elapsed_secs = 0.0` is reset UNCONDITIONALLY at the very
    start of `local_moses` (local_moses.cc:122) -- a caller cannot
    pre-load `elapsed_secs` past `max_time` before the call. Setting
    `max_time <= 0` is the only way to make `not_max_time()` false from
    the FIRST check onward.
    """
    calls = []
    monkeypatch.setattr(lm, "expand_deme", lambda *a, **k: calls.append(1))

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=10000, max_gens=-1, max_score=1000.0)
    pa.max_time = 0

    stats = moses_statistics()
    local_moses(mp, object(), pa, stats)
    assert calls == []


def test_premature_termination_stops_after_exactly_one_call(monkeypatch):
    """`expand_deme` returning `True` must stop the loop immediately,
    even though every OTHER termination predicate would still allow
    another iteration.
    """
    calls = []

    def fake_expand_deme(mp, dex, max_evals, max_time, stats):
        calls.append(1)
        return True

    monkeypatch.setattr(lm, "expand_deme", fake_expand_deme)

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=10000, max_gens=-1, max_score=1000.0)
    stats = moses_statistics()

    local_moses(mp, object(), pa, stats)
    assert calls == [1]


def test_not_premature_termination_keeps_looping_until_another_predicate_stops(monkeypatch):
    """The mirror image of the above: `expand_deme` returning `False`
    every time must let the OTHER predicates be what eventually stops the
    loop, not `premature_termination`.
    """
    calls = []

    def fake_expand_deme(mp, dex, max_evals, max_time, stats):
        calls.append(1)
        stats.n_evals += 1
        return False

    monkeypatch.setattr(lm, "expand_deme", fake_expand_deme)

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=4, max_gens=-1, max_score=1000.0)
    stats = moses_statistics()

    local_moses(mp, object(), pa, stats)
    assert calls == [1, 1, 1, 1]
    assert stats.n_evals == 4


def test_not_max_gens_uses_bang_equal_not_less_than(monkeypatch):
    """Pins `pa.max_gens != stats.n_expansions` VERBATIM (local_moses.cc:
    132) against the specific shape that distinguishes it from `<`: a
    fake `expand_deme` that advances `n_expansions` by 2 every call SKIPS
    OVER `max_gens=3` entirely (0 -> 2 -> 4 -> ...), so `!=` never
    naturally becomes false there and the loop keeps running (bounded
    here by `max_evals`, which advances by 1 per call) -- a `<` in its
    place would have stopped as soon as `n_expansions` first reached or
    passed 3 (i.e. after the SECOND call, at `n_expansions == 4`), one
    call earlier than the verbatim `!=` actually runs.
    """
    calls = []

    def fake_expand_deme(mp, dex, max_evals, max_time, stats):
        calls.append(1)
        stats.n_expansions += 2
        stats.n_evals += 1
        return False

    monkeypatch.setattr(lm, "expand_deme", fake_expand_deme)

    mp = _PredicateMp(best_score=-1.0)
    pa = moses_parameters(max_evals=3, max_gens=3, max_score=1000.0)
    stats = moses_statistics()

    local_moses(mp, object(), pa, stats)

    # Verbatim `!=`: stops only once `not_max_evals()` does (n_evals hits
    # 3), by which point n_expansions has overshot max_gens to 6.
    assert calls == [1, 1, 1]
    assert stats.n_expansions == 6
