"""Oracle and guard tests for hill-climbing crossover (pyMeTTa S12).

## Oracle: `tools/cpp_probe/crossover_probe.cc` -> `tests/fixtures/crossover.json`

Five groups, all measured against the REAL classic library (see
`tools/capture_crossover.py`):

* `partial_sort` -- 15 `std::partial_sort(..., std::greater<deme_inst_t>())`
  samples covering ties, NaN, +/-inf, `mid == 0`, `mid == size` and
  `size == 1`. This is the group that settles the slice's central judgment
  call: `std::partial_sort` has NO stability guarantee, so C++ defines no
  tie order, but libstdc++'s actual order decides which instances get
  crossed. It was MEASURED and is reproduced exactly by
  `hill_climbing`'s `_partial_sort` (a transliteration of libstdc++'s
  `__heap_select` + `__sort_heap`), and
  `test_partial_sort_is_not_a_stable_sort` shows a stable Python sort would
  genuinely differ -- so the "a stable sort can only be at least as
  well-ordered" argument used elsewhere in this port is false here.
* `direct` -- 24 deme scenarios run through all four of `cross_top_one`,
  `cross_top_two`, `cross_top_three` and `crossover`, dumping the whole
  deme afterwards plus the return value: 96 comparisons, each element for
  element. Includes every clamp boundary (`sample_size - 1`, `C(n,2)`,
  `C(n,3)`), `num_to_make` 0 and 1, sample sizes 1/2/3, all-tied samples
  (where the `partial_sort` tie order alone decides the result), a
  non-zero `base` (so `merge_instance` must CLEAR bits as well as set
  them), penalties that invert raw-score order, and the 120/121/122
  budget split.
* `hill_climbing` -- 7 full `hill_climbing::operator()` runs with crossover
  at classic's own default (on), covering BOTH gate entry paths (see
  below), with per-iteration ground truth for whether the `if (xover)`
  branch was taken, read out of classic's own DEBUG log rather than
  inferred from a matching score.
* `local_moses` -- the S11 end-to-end five-generation scenario re-run with
  `--hc-crossover=1` (classic's default) and, as a control, with it off.
  This is the slice's exit criterion.

## Both gate entry paths are covered, and one inventory claim is corrected

`docs/CROSSOVER-INVENTORY.md` says a small Boolean problem "reaches
crossover only via `last_chance`". Measured, that is not quite right, and
the difference matters for coverage: `distance` is incremented at
hill-climbing.cc:367-368 BEFORE the `last_chance` block at :434 runs, so a
`last_chance` crossover always happens at `distance >= 2`, where
`estimate_neighborhood` returns `safe_binomial_coefficient(nn, distance)`
rather than `2 * nn`. For a Boolean field set that is already `>= 400` at
distance 2 once `nn >= 29` (`C(29,2) == 406`), so `large_nbh` -- not
`last_chance` -- is what actually opens the gate on most "small" problems.
The fixture therefore carries:

* `last_chance_only` (arity 4 => 25 info bits => `C(25,2) == 300 < 400`) --
  the only case where `last_chance` is the deciding conjunct. Classic's own
  log says `last chance` for it, and `last chance` for nothing else.
* `large_nbh_above` (404 neighbours) vs `large_nbh_below` (392) -- the
  `>= crossover_min_neighbors` boundary at distance 1, straddled exactly.
* `last_chance`/`last_chance_immediate` -- armed by `last_chance` but
  admitted by `large_nbh` at distance 2; classic logs "n'bhood too large"
  for them, which is how this was found.
* `large_nbh_widen` -- the `already_xover` branch, reachable only when
  `widen_search` keeps a non-improving crossover iteration alive.
* `no_crossover` -- the `hc_params.crossover == False` control.

The `local_moses` oracle reaches crossover through `last chance` in a real
end-to-end run, and changes the trajectory measurably: `n_evals` goes
45/58/74/**198**/**214** with crossover on against 45/58/74/97/113 with it
off (the latter reproducing S11's own pinned numbers exactly).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.local_moses import local_moses
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import (
    _c_div2,
    _cbrtf_plus_3,
    _partial_sort,
    _sqrtf_plus_3,
    hc_parameters,
    hill_climbing,
)
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.field_set import disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import distance_based_scorer
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.util.rng import randGen

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "crossover.json"
_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "crossover.manifest.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())
_MANIFEST = json.loads(_MANIFEST_PATH.read_text())

_XOVER_FUNCS = ("cross_top_one", "cross_top_two", "cross_top_three", "crossover")


# ------------------------------------------------------------------ helpers


def _bit_fs(n: int) -> field_set:
    """The probe's `make_bit_fs`: n single-bit knobs."""
    return field_set([disc_spec(2) for _ in range(n)])


def _id_instance(fs: field_set, ident: int) -> instance:
    """The probe's `id_instance`: writes `ident` into the bit knobs, so an
    instance carries a label the score comparator never looks at.
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


def _cs(sample: dict) -> composite_score:
    return composite_score(
        sample["score"],
        sample["complexity"],
        sample["complexity_penalty"],
        sample["uniformity_penalty"],
    )


def _build_xdeme(case: dict, fs: field_set) -> instance_set:
    """The probe's `build_xdeme`, field for field."""
    deme = instance_set(fs)
    deme.resize(case["sample_start"] + case["sample_size"])
    for i in range(case["sample_start"]):
        deme[i].first = _id_instance(fs, 32 + i)
        deme[i].second = composite_score(-1000.0 - i, 0, 0.0, 0.0)
    for i, sample in enumerate(case["samples"]):
        deme[case["sample_start"] + i].first = _id_instance(fs, sample["id"])
        deme[case["sample_start"] + i].second = _cs(sample)
    return deme


def _assert_deme_matches(fs: field_set, deme, expected: list[dict], label: str) -> None:
    assert len(deme) == len(expected), f"{label}: deme size"
    for i, want in enumerate(expected):
        got_raw = [fs.get_raw(deme[i].first, k) for k in range(fs.raw_size())]
        assert got_raw == want["raw"], f"{label}: slot {i} raw"
        cs = deme[i].second
        wsc = want["score"]
        assert cs.get_score() == wsc["score"], f"{label}: slot {i} score"
        assert cs.get_complexity() == wsc["complexity"], f"{label}: slot {i} complexity"
        assert cs.get_complexity_penalty() == wsc["complexity_penalty"], f"{label}: slot {i} cpen"
        assert cs.get_uniformity_penalty() == wsc["uniformity_penalty"], f"{label}: slot {i} upen"
        assert cs.get_penalized_score() == wsc["penalized_score"], f"{label}: slot {i} penalized"


def _fs_from_shape(shape: dict) -> field_set:
    specs = [disc_spec(m) for m in shape["disc_multiplicities"]]
    specs += [disc_spec(2) for _ in range(shape["n_bits"])]
    return field_set(specs)


def _target_for(fs: field_set, null_target: bool) -> instance:
    """The probe's `run_hc_case` target: the all-zero instance when
    `null_target` (which makes the 4-arg overload's own `init_inst` the
    global optimum), else each knob at its own top value.
    """
    target = instance(fs.packed_width())
    if null_target:
        return target
    for i in range(fs.raw_size()):
        if fs.begin_disc_raw_idx() <= i < fs.end_disc_raw_idx():
            multy = fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy
            fs.set_raw(target, i, multy - 1)
        elif fs.begin_bit_raw_idx() <= i < fs.end_bit_raw_idx():
            fs.set_raw(target, i, 1)
    return target


class _TracedHillClimbing(hill_climbing):
    """Records one entry per iteration of `__call__`, marking the ones on
    which `crossover` actually ran.

    `estimate_neighborhood` is called exactly once per iteration, at the top
    (hill-climbing.cc:105), and `crossover` only from inside `if (xover)`
    (:165) later in that same iteration -- so this reproduces exactly the
    `{"iteration": n, "xover": bool}` sequence the probe reads out of
    classic's own DEBUG log ("Iteration: %u" at :108, "Crossover
    enabled"/"Crossover disabled" at :162/:184). This is what makes
    "crossover fired, on these iterations" an assertion rather than an
    inference from a matching score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xover_log: list[dict] = []
        self._iteration = 0

    def __call__(self, *args, **kwargs):
        self._iteration = 0
        return super().__call__(*args, **kwargs)

    def estimate_neighborhood(self, distance, fields):
        self._iteration += 1
        self.xover_log.append({"iteration": self._iteration, "xover": False})
        return super().estimate_neighborhood(distance, fields)

    def crossover(self, *args, **kwargs):
        self.xover_log[-1]["xover"] = True
        return super().crossover(*args, **kwargs)

    @property
    def fired_on(self) -> list[int]:
        return [e["iteration"] for e in self.xover_log if e["xover"]]


def _expected_xover_log(case: dict) -> list[dict]:
    """The probe's `why` string is dropped logging (see hill_climbing.py's
    module docstring); only iteration + fired-or-not is port-observable.
    """
    return [{"iteration": e["iteration"], "xover": e["xover"]} for e in case["xover_log"]]


# ------------------------------------------------------------------- fixture


def test_fixture_has_expected_shape():
    assert set(_GOLDEN) == {"partial_sort", "direct", "hill_climbing", "local_moses"}
    assert len(_GOLDEN["partial_sort"]) == 15
    assert set(_GOLDEN["direct"]) == set(_XOVER_FUNCS)
    for func in _XOVER_FUNCS:
        assert len(_GOLDEN["direct"][func]) == 24
    assert set(_GOLDEN["hill_climbing"]) == {
        "last_chance",
        "last_chance_immediate",
        "last_chance_only",
        "no_crossover",
        "large_nbh_below",
        "large_nbh_above",
        "large_nbh_widen",
    }
    assert set(_GOLDEN["local_moses"]) == {"crossover_on", "crossover_off"}


def test_fixture_manifest_matches_current_probe_source():
    """Compiler-free rot detection, the same guard
    `test_hill_climbing.py::test_fixture_manifest_matches_current_probe_source`
    provides for its own fixture: editing `crossover_probe.cc` without
    re-running `tools/capture_crossover.py` fails here.
    """
    assert _MANIFEST["schema_version"] == 1
    assert _MANIFEST["probe_source"] == "tools/cpp_probe/crossover_probe.cc"
    assert _MANIFEST["capture_script"] == "tools/capture_crossover.py"
    probe = _REPO_ROOT / "tools" / "cpp_probe" / "crossover_probe.cc"
    assert _MANIFEST["probe_source_sha256"] == hashlib.sha256(probe.read_bytes()).hexdigest()


# -------------------------------------------------------- std::partial_sort


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["partial_sort"], ids=lambda c: c["label"])
def test_partial_sort_matches_libstdcxx_order(case):
    """The measurement that settles the tie-order question: `_partial_sort`
    reproduces libstdc++'s permutation element for element, including for
    inputs where every element ties.
    """
    fs = _bit_fs(6)
    v = [scored_instance(_id_instance(fs, s["id"]), _cs(s)) for s in case["input"]]
    _partial_sort(v, 0, case["mid"], len(v))
    assert [_instance_id(fs, si.first) for si in v] == case["order"], case["label"]


def test_partial_sort_is_not_a_stable_sort():
    """Proves the substitution this port used to make elsewhere is wrong,
    rather than asserting it: for an all-tied sample, libstdc++'s measured
    order differs from a stable descending sort, and it is the FRONT of the
    range -- the part `cross_top_one` crosses and `resize_deme` keeps --
    that differs.
    """
    case = next(c for c in _GOLDEN["partial_sort"] if c["label"] == "all_tied_8_mid4")
    assert case["order"] != sorted(range(8))  # measured C++: [1, 3, 0, 2, ...]

    fs = _bit_fs(6)
    stable = [scored_instance(_id_instance(fs, s["id"]), _cs(s)) for s in case["input"]]
    stable.sort(reverse=True)
    assert [_instance_id(fs, si.first) for si in stable] == list(range(8))
    assert [_instance_id(fs, si.first) for si in stable] != case["order"]


def test_partial_sort_breaks_equal_penalized_scores_by_complexity():
    """`composite_score::operator<` (types.cc:139-148) falls back to
    `complexity > r.complexity` when penalized scores are equal, so a
    HIGHER complexity sorts EARLIER under `<` and therefore LATER under
    `std::greater`. Pinned against the measured order rather than assumed.
    """
    case = next(
        c for c in _GOLDEN["partial_sort"] if c["label"] == "equal_score_diff_complexity_6_mid3"
    )
    assert [s["complexity"] for s in case["input"]] == [0, 1, 2, 3, 4, 5]
    assert case["order"][:3] == [0, 1, 2]  # lowest complexity first


def test_partial_sort_ranks_by_penalized_not_raw_score():
    """A discriminating case: raw score ascends with id while the penalties
    make the penalized score descend, so a comparator reading `get_score()`
    would produce the exact reverse of the measured order.
    """
    case = next(c for c in _GOLDEN["partial_sort"] if c["label"] == "penalties_invert_order_8_mid4")
    assert [s["score"] for s in case["input"]] == list(range(8))
    assert case["order"][:4] == [0, 1, 2, 3]

    fs = _bit_fs(6)
    v = [scored_instance(_id_instance(fs, s["id"]), _cs(s)) for s in case["input"]]
    penalized = [si.second.get_penalized_score() for si in v]
    assert penalized == sorted(penalized, reverse=True)


# ------------------------------------------------- cross_top_* and crossover


@pytest.mark.oracle
@pytest.mark.parametrize("func", _XOVER_FUNCS)
@pytest.mark.parametrize(
    "index", range(len(_GOLDEN["direct"]["cross_top_one"])), ids=lambda i: str(i)
)
def test_cross_functions_match_cpp(func, index):
    case = _GOLDEN["direct"][func][index]
    fs = _bit_fs(case["n_bits"])
    deme = _build_xdeme(case, fs)
    base = _id_instance(fs, case["base_id"])
    hc = hill_climbing(optim_parameters(), hc_parameters())

    returned = getattr(hc, func)(
        deme,
        case["deme_size"],
        case["num_to_make"],
        case["sample_start"],
        case["sample_size"],
        base,
    )
    label = f"{func}/{case['label']}"
    assert returned == case["returned"], f"{label}: return value"
    _assert_deme_matches(fs, deme, case["deme"], label)


def test_cross_top_one_rejects_an_empty_sample():
    """`OC_ASSERT(sample_size > 0, ...)` (hill-climbing.cc:596). Without it,
    `sample_size - 1` on the next line is `size_t` arithmetic that wraps to
    2**64-1 and the clamp never fires.
    """
    fs = _bit_fs(4)
    deme = instance_set(fs)
    deme.resize(3)
    hc = hill_climbing(optim_parameters(), hc_parameters())
    with pytest.raises(ValueError, match="Cross-over sample size must be positive"):
        hc.cross_top_one(deme, 3, 2, 0, 0, instance(fs.packed_width()))


def test_crossover_splits_the_budget_with_the_remainder_to_cross_top_one():
    """hill-climbing.cc:703-731: the budget is capped at
    `crossover_pop_size` and then split `n/3 + n%3` / `n/3` / `n/3`, so the
    default 120 gives 40/40/40 and the WHOLE remainder goes to
    `cross_top_one`. Recorded per stage rather than only checking the total,
    which 40/40/40 and 41/40/39 would both satisfy.
    """
    fs = _bit_fs(24)
    calls = []

    class Spy(hill_climbing):
        def cross_top_one(self, deme, deme_size, num_to_make, *rest):
            calls.append(("one", num_to_make))
            return super().cross_top_one(deme, deme_size, num_to_make, *rest)

        def cross_top_two(self, deme, deme_size, num_to_make, *rest):
            calls.append(("two", num_to_make))
            return super().cross_top_two(deme, deme_size, num_to_make, *rest)

        def cross_top_three(self, deme, deme_size, num_to_make, *rest):
            calls.append(("three", num_to_make))
            return super().cross_top_three(deme, deme_size, num_to_make, *rest)

    for budget, expected in (
        (120, [("one", 40), ("two", 40), ("three", 40)]),
        # The remainder goes entirely to cross_top_one -- 41/40/40, not
        # 40/40/41 and not 41/40/39.
        (118, [("one", 40), ("two", 39), ("three", 39)]),
        (119, [("one", 41), ("two", 39), ("three", 39)]),
        (100, [("one", 34), ("two", 33), ("three", 33)]),
        (101, [("one", 35), ("two", 33), ("three", 33)]),
        # At or above crossover_pop_size the cap applies FIRST, so 121 and
        # 122 are indistinguishable from 120 -- the remainder rule never
        # sees them. (Measured: min(budget, 120) at hill-climbing.cc:704.)
        (121, [("one", 40), ("two", 40), ("three", 40)]),
        (122, [("one", 40), ("two", 40), ("three", 40)]),
        (500, [("one", 40), ("two", 40), ("three", 40)]),
    ):
        calls.clear()
        case = {"sample_start": 0, "sample_size": 24, "samples": []}
        case["samples"] = [
            {
                "id": 1 << i,
                "score": -float(i),
                "complexity": 0,
                "complexity_penalty": 0.0,
                "uniformity_penalty": 0.0,
            }
            for i in range(24)
        ]
        deme = _build_xdeme(case, fs)
        hc = Spy(optim_parameters(), hc_parameters())
        hc.crossover(deme, 24, budget, 0, 24, _id_instance(fs, 0))
        assert calls == expected, f"budget {budget}"


def test_cross_top_three_index_collisions_are_reproduced_not_fixed():
    """DO NOT FIX (hill-climbing.cc:687): `k < i` with an index formula that
    assumes `k < j`.

    Rather than merely asserting the fixture matches, this reconstructs the
    write sequence directly from the ported loop bounds and pins what the
    bug actually DOES, so the marker's claims are checkable:

    * nothing below `num_to_make` is left unwritten (every slot is covered);
    * slots are written repeatedly, last write wins;
    * block `i`'s final offset spills into block `i+1`'s slot 0.

    The `expected_pairs` below are derived from `k < i`; recomputing them
    with `k < j` (the "fixed" bound) makes this test fail, which is the
    point.
    """
    writes: dict[int, list[tuple[int, int, int]]] = {}
    num_to_make = 40
    sample_size = 10
    stopped_at = None
    for i in range(2, sample_size):
        for j in range(1, i):
            for k in range(i):  # the bug: `k < i`, not `k < j`
                n = i * (i - 1) * (i - 2) // 6 + j * (j - 1) // 2 + k
                if num_to_make <= n:
                    stopped_at = (i, j, k, n)
                    break
                writes.setdefault(n, []).append((i, j, k))
            else:
                continue
            break
        else:
            continue
        break

    assert stopped_at is not None
    # No hole below num_to_make.
    assert sorted(writes) == list(range(num_to_make))
    # Repeated writes really happen, and the last one wins.
    multiply_written = {n: ws for n, ws in writes.items() if len(ws) > 1}
    assert multiply_written, "the k<i bug must produce colliding indices"
    assert writes[3] == [(3, 1, 2), (3, 2, 1)]  # doc/cross_top_three_explanation.md's own example
    # `k >= j` triples are exactly the ones a `k < j` loop would never make.
    assert any(k >= j for ws in writes.values() for (_, j, k) in ws)
    # Block i's last offset spills into block i+1's slot 0: C(4,3) == 4 is
    # written by block i=3 (offset i*(i-1)/2 == 3) before block i=4 starts.
    assert (3, 2, 2) in writes[4]
    assert writes[4][-1] == (4, 1, 0)


def test_cross_top_three_fixture_matches_the_collision_derivation():
    """Ties the derivation above to the measured C++: for the all-distinct
    `flips10_start0_make40` case, every crossed slot equals the OR of the
    ids of its LAST writer's (i, j, k) triple -- because `base` is 0 and
    each sample flips exactly one distinct bit, so `merge_instance` ORs.
    """
    case = next(
        c for c in _GOLDEN["direct"]["cross_top_three"] if c["label"] == "flips10_start0_make40"
    )
    order = [s["id"] for s in case["samples"]]  # strictly descending score => already sorted
    assert case["num_to_make"] == 40 and case["sample_size"] == 10

    last_writer: dict[int, tuple[int, int, int]] = {}
    for i in range(2, case["sample_size"]):
        for j in range(1, i):
            for k in range(i):
                n = i * (i - 1) * (i - 2) // 6 + j * (j - 1) // 2 + k
                if case["num_to_make"] <= n:
                    break
                last_writer[n] = (i, j, k)
            else:
                continue
            break
        else:
            continue
        break

    for n, (i, j, k) in last_writer.items():
        want_id = order[k] | order[i] | order[j]
        got = case["deme"][case["deme_size"] + n]
        got_id = sum(bit << pos for pos, bit in enumerate(got["raw"]))
        assert got_id == want_id, f"slot {n}: last writer (i={i}, j={j}, k={k})"
    # The sample really is in descending score order, so `order[x]` above is
    # the x-th best -- i.e. this test is reading the same ranking the C++
    # partial_sort produced, not a coincidence of insertion order.
    assert [s["score"] for s in case["samples"]] == sorted(
        (s["score"] for s in case["samples"]), reverse=True
    )


def test_c_div2_truncates_toward_zero():
    """`_c_div2` exists because C's `/` truncates toward zero where Python's
    `//` floors, and libstdc++'s heap code really does hand it negatives:
    `__push_heap` computes `(holeIndex - 1) / 2` with `holeIndex == 0`, and
    `__adjust_heap` computes `(len - 1) / 2` / `(len - 2) / 2` with `len`
    as small as 0.

    Measured caveat, stated rather than glossed: in the reachable state
    space those negative results are never DECISIVE -- `__push_heap`'s loop
    guard `holeIndex > topIndex` is already false whenever `holeIndex == 0`
    and `topIndex >= 0`, and `__adjust_heap`'s `(len - 2) / 2` is only read
    under `(len & 1) == 0`, which excludes `len == 1`, the one length where
    C++ and Python disagree there. So swapping `_c_div2` for `//` does not
    change any output this suite can produce; this test pins the helper's
    own contract directly instead, which is the only place the difference
    is observable.
    """
    assert _c_div2(-1) == 0  # C++: -1 / 2 == 0; Python: -1 // 2 == -1
    assert _c_div2(-2) == -1
    assert _c_div2(-3) == -1
    assert _c_div2(0) == 0
    assert _c_div2(1) == 0
    assert _c_div2(4) == 2


def test_sqrtf_and_cbrtf_helpers_narrow_to_binary32_like_c():
    """The `float`-vs-`double` half of `sqrtf`/`cbrtf`, with MEASURED
    counterexamples rather than an argument that it "cannot matter".

    C converts the `size_t` argument to `float` before the call and keeps
    the result in `float`; computing the same expression in Python's
    binary64 gives a DIFFERENT truncated integer at these inputs, each
    confirmed against a C++ snippet compiled with the same toolchain:

        n = 8380417: sqrtf(2n) + 3 -> 4097 (C++ and narrowed Python),
                     but binary64 sqrt gives 4096.
        n = 26244:   cbrtf(6n) + 3 -> 57 (C++ and narrowed Python),
                     but binary64 cbrt gives 56.
    """
    assert _sqrtf_plus_3(8380417) == 4097
    assert _sqrtf_plus_3(8384512) == 4098
    assert _sqrtf_plus_3(8388607) == 4099
    assert _cbrtf_plus_3(26244) == 57
    assert _cbrtf_plus_3(36000) == 63
    assert _cbrtf_plus_3(209952) == 111


def test_cross_top_two_early_return_is_equivalent_to_skipping():
    """Documents, with proof, one mutation this suite deliberately does NOT
    catch: turning `cross_top_two`'s `return num_to_make` (hill-climbing.cc:
    652) into a `continue` changes nothing.

    `n = i*(i-1)/2 + j` is STRICTLY INCREASING in the loop's own iteration
    order (within a block `j` rises; between blocks the last `n` of block
    `i` is `(i^2+i-2)/2` and the first of block `i+1` is `(i^2+i)/2`), so
    once `num_to_make <= n` every later `n` is larger too and a `continue`
    would write nothing more. The same mutation applied to
    `cross_top_three` IS caught, because its `n` is not monotone -- which
    is exactly the upstream bug this slice reproduces.
    """
    for sample_size in (2, 5, 10, 40):
        seen = []
        for i in range(1, sample_size):
            for j in range(i):
                seen.append(i * (i - 1) // 2 + j)
        assert seen == sorted(seen) and len(set(seen)) == len(seen), sample_size

    # ... and cross_top_three's really is not monotone, at the same sizes.
    three = []
    for i in range(2, 10):
        for j in range(1, i):
            for k in range(i):
                three.append(i * (i - 1) * (i - 2) // 6 + j * (j - 1) // 2 + k)
    assert three != sorted(three)


def test_iteration_above_two_is_redundant_with_the_deme_size_conjunct():
    """Documents a second deliberately-uncaught mutation, with the
    measurement behind it: relaxing `(iteration > 2)`
    (hill-climbing.cc:148) to `(iteration > 1)` changes nothing, because
    `current_number_of_instances >= 3` already excludes iterations 1 and 2.

    Iteration 1 always runs at distance 0 and therefore appends exactly one
    instance (hill-climbing.cc:521-522's `if (distance == 0) return 1;`),
    so the deme holds 0 instances entering iteration 1 and exactly 1
    entering iteration 2 -- both below `xover_min_deme` (3). Recorded from
    a real run rather than argued: `n_new_instances`' third parameter is
    `current_number_of_instances` at the top of each iteration.
    """
    seen = []

    class Spy(_TracedHillClimbing):
        def n_new_instances(self, distance, max_evals, current, total):
            seen.append(current)
            return super().n_new_instances(distance, max_evals, current, total)

    hcp = hc_parameters(cross=True)
    hcp.crossover_min_neighbors = 1
    fs = _bit_fs(6)
    scorer = distance_based_scorer(fs, instance(fs.packed_width()))
    randGen().seed(1)
    deme = instance_set(fs)
    hc = Spy(optim_parameters(terminate_if_gte=1.0), hcp)
    hc(deme, scorer, 500, 86400)

    assert len(seen) >= 3
    assert seen[0] == 0  # entering iteration 1
    assert seen[1] == 1  # entering iteration 2 -- below xover_min_deme (3)
    assert seen[2] >= 3  # entering iteration 3 -- the first that can qualify


class _TwoBitBonusScorer:
    """A deliberately deceptive landscape, needed to make `last_chance`'s
    one-shot RESET observable.

    Single-bit flips off the all-zero centre score WORSE than the centre
    (so iteration 2 plateaus and arms `last_chance`), while any instance
    with two or more bits set scores BETTER (so the crossover on iteration
    3 improves, adopts a new centre, and clears `already_xover`). Without
    the `last_chance = False` at hill-climbing.cc:457, the stale flag would
    then open the gate again on iteration 4, where `large_nbh` is false.
    """

    def __call__(self, inst) -> composite_score:
        bits = sum(1 for v in self._fs.iter_bit(inst) if v)
        if bits == 0:
            return composite_score(0.0, 0, 0.0, 0.0)
        if bits == 1:
            return composite_score(-1.0, 0, 0.0, 0.0)
        return composite_score(1.0, 0, 0.0, 0.0)

    def __init__(self, fs: field_set) -> None:
        self._fs = fs

    def get_rep(self):
        return None


def test_last_chance_is_cleared_after_the_crossover_iteration():
    """hill-climbing.cc:457's `last_chance = false;`. On the deceptive
    landscape above, iteration 3's crossover IMPROVES, so a new centre is
    adopted and `already_xover` is cleared -- leaving the stale
    `last_chance` as the only thing that could reopen the gate on
    iteration 4, where the neighbourhood is far too small for `large_nbh`.

    Measured: crossover fires on iterations 3 and 5 -- iteration 5 is a
    SECOND plateau arming `last_chance` again, which is correct ("one shot
    per plateau", not "one shot per run"). Iteration 4 is the one that must
    stay clear: it directly follows the improving crossover, so only a
    stale `last_chance` could open the gate there.
    """
    fs = _bit_fs(4)
    scorer = _TwoBitBonusScorer(fs)
    randGen().seed(1)
    deme = instance_set(fs)
    hc = _TracedHillClimbing(optim_parameters(terminate_if_gte=10.0), hc_parameters(cross=True))
    hc(deme, scorer, 400, 86400)

    assert len(hc.xover_log) >= 4, "the run must survive past the crossover iteration"
    assert hc.fired_on and hc.fired_on[0] == 3
    assert 4 not in hc.fired_on, "a stale last_chance reopened the gate"


def test_sqrtf_and_cbrtf_helpers_truncate_like_c():
    """`unsigned num_to_sort = sqrtf(2 * num_to_make) + 3;`
    (hill-climbing.cc:632) and its `cbrtf(6 * num_to_make) + 3` twin
    (:673). Values chosen ON the truncation boundaries: `2n` a perfect
    square and one either side, `6n` a perfect cube and one either side.
    """
    # sqrtf(2n)+3: 2n == 0, 1, 4, 9, 16, 25 (n == 0, 2, 8, 12(->24 no), ...)
    assert _sqrtf_plus_3(0) == 3  # sqrt(0) + 3
    assert _sqrtf_plus_3(2) == 5  # sqrt(4) == 2 exactly -> 5, not 4
    assert _sqrtf_plus_3(1) == 4  # sqrt(2) == 1.414 -> 4
    assert _sqrtf_plus_3(8) == 7  # sqrt(16) == 4 -> 7
    assert _sqrtf_plus_3(7) == 6  # sqrt(14) == 3.741 -> 6
    assert _sqrtf_plus_3(40) == 11  # sqrt(80) == 8.944 -> 11
    # cbrtf(6n)+3: 6n == 0, 8, 27, 64, 216
    assert _cbrtf_plus_3(0) == 3
    assert _cbrtf_plus_3(1) == 4  # cbrt(6) == 1.817 -> 4
    assert _cbrtf_plus_3(36) == 9  # cbrt(216) == 6 exactly -> 9, not 8
    assert _cbrtf_plus_3(35) == 8  # cbrt(210) == 5.943 -> 8
    assert _cbrtf_plus_3(40) == 9  # cbrt(240) == 6.214 -> 9


# ------------------------------------------- full hill_climbing, crossover on


@pytest.mark.oracle
@pytest.mark.parametrize("label", sorted(_GOLDEN["hill_climbing"]))
def test_hill_climbing_with_crossover_matches_cpp_step_for_step(label):
    case = _GOLDEN["hill_climbing"][label]
    fs = _fs_from_shape(case["field_shape"])
    scorer = distance_based_scorer(fs, _target_for(fs, case["null_target"]))

    randGen().seed(case["seed"])
    deme = instance_set(fs)
    hc = _TracedHillClimbing(
        optim_parameters(terminate_if_gte=case["terminate_if_gte"]),
        hc_parameters(
            widen=case["hc_widen_search"],
            cross=case["hc_crossover"],
            max_evals=case["hc_max_nn_evals"],
        ),
    )
    hc(deme, scorer, case["max_evals"], 86400)

    # Crossover FIRED, on exactly the iterations classic's own debug log
    # says it did -- asserted, not inferred from a matching score.
    assert hc.xover_log == _expected_xover_log(case), f"{label}: crossover firing pattern"

    assert deme.n_evals == case["n_evals"], f"{label}: n_evals"
    assert hc.nsteps == case["nsteps"], f"{label}: nsteps"
    assert hc.over_budget == case["over_budget"], f"{label}: over_budget"
    _assert_deme_matches(fs, deme, case["trajectory"], label)


def test_the_fixture_covers_both_gate_entry_paths():
    """Guards the suite itself: if a future edit made every crossover case
    reach the gate the same way, the "both paths" claim in this module's
    docstring would quietly become false.
    """
    reasons = {
        why
        for case in _GOLDEN["hill_climbing"].values()
        for why in (e["why"] for e in case["xover_log"] if e["xover"])
    }
    assert any(r == "last chance" for r in reasons), "no case reaches the gate via last_chance"
    assert any(r.startswith("n'bhood too large") for r in reasons), (
        "no case reaches the gate via large_nbh"
    )

    only = _GOLDEN["hill_climbing"]["last_chance_only"]
    assert [e["why"] for e in only["xover_log"] if e["xover"]] == ["last chance"]
    # ... and at that moment large_nbh really is false: 25 info bits at
    # distance 2 gives C(25,2) == 300 < crossover_min_neighbors (400).
    assert only["nn_estimate_d1"] == 50

    above = _GOLDEN["hill_climbing"]["large_nbh_above"]
    below = _GOLDEN["hill_climbing"]["large_nbh_below"]
    assert above["nn_estimate_d1"] == 404 and below["nn_estimate_d1"] == 392
    assert [e["iteration"] for e in above["xover_log"] if e["xover"]] == [3, 4, 5]
    assert [e["iteration"] for e in below["xover_log"] if e["xover"]] == []


# ------------------------------------------------------------- the gate rules


def _plateau_run(
    n_knobs: int,
    hcp: hc_parameters,
    max_evals: int = 500,
    terminate_if_gte: float = 1.0,
) -> _TracedHillClimbing:
    """A run over `n_knobs` single-bit knobs whose target is the all-zero
    instance -- which is exactly the `init_inst` the 4-arg overload builds
    -- so the search is at its optimum from iteration 1 and plateaus on
    iteration 2. That is the shortest path to the `last_chance` block
    (hill-climbing.cc:434-458).
    """
    fs = _bit_fs(n_knobs)
    scorer = distance_based_scorer(fs, instance(fs.packed_width()))
    randGen().seed(1)
    deme = instance_set(fs)
    hc = _TracedHillClimbing(optim_parameters(terminate_if_gte=terminate_if_gte), hcp)
    hc(deme, scorer, max_evals, 86400)
    return hc


def test_gate_requires_iteration_above_two():
    """hill-climbing.cc:148's `(iteration > 2)`. With
    `crossover_min_neighbors` dropped to 1, `large_nbh` is true from the
    first iteration, so this conjunct is the only thing holding crossover
    off on iterations 1 and 2.
    """
    hcp = hc_parameters(cross=True)
    hcp.crossover_min_neighbors = 1
    hc = _plateau_run(6, hcp)
    assert hc.fired_on and min(hc.fired_on) == 3
    assert [e["xover"] for e in hc.xover_log[:2]] == [False, False]


def test_gate_requires_the_deme_to_hold_at_least_three_instances():
    """hill-climbing.cc:144-149's hard-coded `xover_min_deme = 3`. A
    one-knob field set only ever reaches two instances, so with every other
    conjunct satisfied (`crossover_min_neighbors = 1` makes `large_nbh`
    true, and `last_chance` gets armed) crossover still never fires.
    """
    hcp = hc_parameters(cross=True)
    hcp.crossover_min_neighbors = 1
    hc = _plateau_run(1, hcp)
    assert len(hc.xover_log) >= 3, "the run must actually reach an iteration past 2"
    assert hc.fired_on == []


def test_gate_uses_a_greater_or_equal_neighbourhood_test():
    """`total_number_of_neighbors >= xover_min_neighbors`
    (hill-climbing.cc:147). A 6-bit field set estimates 2*6 == 12
    neighbours at distance 1, so a threshold of exactly 12 must admit and
    13 must refuse -- pinning `>=` rather than `>`.

    Four single-bit knobs, not six, for a measured reason: the estimate is
    `2 * nn` at distance 1 but `safe_binomial_coefficient(nn, distance)`
    beyond it (hill-climbing.cc:504-517), and only for `nn <= 4` is the
    distance-1 value the LARGEST of them (8, then C(4,2)=6, C(4,3)=4,
    C(4,4)=1). With six knobs a threshold of 13 still admits crossover
    once the run plateaus and distance reaches 2 (C(6,2) = 15) -- measured,
    it fired on iteration 9 -- which would make the "refuses" half pass for
    the wrong reason.

    The assertion is about ITERATION 3 specifically, not about crossover
    never happening: at iteration 3 the search is still climbing at
    distance 1, `last_chance` has not been armed (it needs a plateau), and
    `large_nbh` is the only conjunct in play. The threshold-9 run does
    eventually cross over (measured: iteration 7), but only after the
    plateau arms `last_chance` -- the OTHER entry path, which this test
    cannot suppress without also suppressing the plateau that ends the run.
    """
    fired = {}
    for threshold in (8, 9):
        hcp = hc_parameters(cross=True)
        hcp.crossover_min_neighbors = threshold
        fs = _bit_fs(4)
        # A target the search can climb towards keeps distance at 1 for the
        # first few iterations, which is where 2 * nn is the estimate.
        target = instance(fs.packed_width())
        for i in range(4):
            fs.set_raw(target, i, 1)
        scorer = distance_based_scorer(fs, target)
        randGen().seed(1)
        deme = instance_set(fs)
        hc = _TracedHillClimbing(optim_parameters(terminate_if_gte=1.0), hcp)
        hc(deme, scorer, 500, 86400)
        fired[threshold] = hc.fired_on
    assert fired[8] and fired[8][0] == 3, (
        "threshold == estimate must admit crossover at distance 1 (>=, not >)"
    )
    assert 3 not in fired[9], "threshold one above the estimate must refuse it at distance 1"


def test_last_chance_is_one_shot_per_plateau():
    """`last_chance` is set at hill-climbing.cc:450 and cleared at :457 on
    the very next pass, and `already_xover` (:168) blocks a second attempt
    from the same centre -- so a plateau yields exactly one crossover.
    """
    case = _GOLDEN["hill_climbing"]["last_chance_only"]
    assert [e["iteration"] for e in case["xover_log"] if e["xover"]] == [3]
    assert len(case["xover_log"]) == 3


def test_already_xover_blocks_a_second_crossover_from_the_same_centre():
    """hill-climbing.cc:100/168/288: `already_xover` is set when crossover
    runs and cleared only when a NEW centre is adopted. `large_nbh_widen`
    is the only fixture case that stays alive after a non-improving
    crossover (widen_search suppresses the "No improvement" exit), and it
    shows the gate refusing on every subsequent iteration.
    """
    case = _GOLDEN["hill_climbing"]["large_nbh_widen"]
    assert [e["iteration"] for e in case["xover_log"] if e["xover"]] == [3]
    assert [e["why"] for e in case["xover_log"][3:]] == ["already done from that center"] * 3


def test_crossover_reads_the_previous_iteration_block_not_the_current_one():
    """The first of the inventory's two asymmetries
    (hill-climbing.cc:165-166 vs :190): the crossover branch is handed
    `prev_start`/`prev_size`/`prev_center`, captured at :204-206 on the
    PREVIOUS pass, while the sampling branch uses `center_inst` and the
    CURRENT `current_number_of_instances`.

    Recorded directly from a real run: on the crossing iteration,
    `sample_start + sample_size` equals `deme_size` (the whole
    previously-generated block, ending exactly where the new instances will
    be appended), and `base` is the centre as of that previous pass.
    """
    seen = []

    class Spy(_TracedHillClimbing):
        def crossover(self, deme, deme_size, budget, sample_start, sample_size, base):
            seen.append(
                {
                    "deme_size": deme_size,
                    "len_deme": len(deme),
                    "sample_start": sample_start,
                    "sample_size": sample_size,
                    "base": list(base),
                }
            )
            return super().crossover(deme, deme_size, budget, sample_start, sample_size, base)

    fs = _bit_fs(6)
    scorer = distance_based_scorer(fs, instance(fs.packed_width()))
    hcp = hc_parameters(cross=True)
    hcp.crossover_min_neighbors = 1
    randGen().seed(1)
    deme = instance_set(fs)
    hc = Spy(optim_parameters(terminate_if_gte=1.0), hcp)
    hc(deme, scorer, 500, 86400)

    assert seen, "crossover must actually have run"
    first = seen[0]
    assert first["deme_size"] == first["len_deme"]
    assert first["sample_start"] + first["sample_size"] == first["deme_size"]
    # Iteration 1 contributes exactly one instance (distance 0), so the
    # previous block starts at 1 -- NOT at 0, and not at deme_size.
    assert first["sample_start"] == 1
    assert first["sample_size"] == first["deme_size"] - 1


def test_distance_is_not_incremented_after_a_crossover_iteration():
    """The second asymmetry: `} else if (!xover) distance++;`
    (hill-climbing.cc:367-368). Recorded from `large_nbh_widen`, the one
    configuration where a non-improving crossover iteration is followed by
    another iteration: the distance handed to `estimate_neighborhood` after
    the crossover is the SAME as the one handed to it before.
    """
    distances = []

    class Spy(_TracedHillClimbing):
        def estimate_neighborhood(self, distance, fields):
            result = super().estimate_neighborhood(distance, fields)
            distances.append(distance)
            return result

    fs = _bit_fs(6)
    scorer = distance_based_scorer(fs, instance(fs.packed_width()))
    hcp = hc_parameters(widen=True, cross=True)
    hcp.crossover_min_neighbors = 1
    randGen().seed(1)
    deme = instance_set(fs)
    hc = Spy(optim_parameters(terminate_if_gte=1.0), hcp)
    hc(deme, scorer, 400, 86400)

    assert hc.fired_on, "crossover must actually have run"
    fire = hc.fired_on[0] - 1  # 0-based index into `distances`
    assert fire + 1 < len(distances), "a later iteration must exist to observe"
    assert distances[fire + 1] == distances[fire]


def test_crossover_disabled_never_calls_the_crossover_machinery():
    """The `hc_params.crossover` conjunct itself (hill-climbing.cc:147) --
    and the `if (hc_params.crossover)` guard around the whole last_chance
    block at :434. With it off, neither fires no matter how permissive the
    other conjuncts are.
    """
    hcp = hc_parameters(cross=False)
    hcp.crossover_min_neighbors = 1
    hc = _plateau_run(6, hcp)
    assert hc.fired_on == []


def test_resize_deme_uses_the_measured_partial_sort_order():
    """`resize_deme`'s own `std::partial_sort` (hill-climbing.cc:815-817)
    truncates immediately afterwards, so tie order decides WHICH tied
    instances survive -- the case S12 re-checked rather than inheriting the
    "a stable sort is at least as well-ordered" argument.

    All ten instances tie on score, and each carries a distinct id in its
    bits; the five that survive are libstdc++'s five, not the first five.
    """
    fs = _bit_fs(6)
    deme = instance_set(fs)
    deme.resize(10)
    for i in range(10):
        deme[i].first = _id_instance(fs, i)
        deme[i].second = composite_score(-1.0, 0, 0.0, 0.0)

    hcp = hc_parameters()
    hcp.max_allowed_instances = 5
    hc = hill_climbing(optim_parameters(), hcp)
    assert hc.resize_deme(deme, 0.0) is True

    survivors = [_instance_id(fs, si.first) for si in deme]
    assert len(survivors) == 5
    assert survivors != list(range(5)), "a stable sort would keep ids 0..4"

    reference = [scored_instance(_id_instance(fs, i), composite_score(-1.0, 0)) for i in range(10)]
    _partial_sort(reference, 0, 5, 10)
    assert survivors == [_instance_id(fs, si.first) for si in reference[:5]]


def test_scored_instance_assignment_in_crossover_is_a_value_copy():
    """CARRY-FORWARD.md 2.4's rule, at the three `deme[ntgt] = deme[src]`
    sites: C++'s `scored_instance` assignment copies both members, and
    `merge_instance` then mutates only the copy. An aliasing port would
    corrupt the SOURCE instance, which is still inside the sample window
    the next iteration of the same loop reads.
    """
    fs = _bit_fs(6)
    case = {
        "sample_start": 0,
        "sample_size": 4,
        "samples": [
            {
                "id": 1 << i,
                "score": -float(i),
                "complexity": 0,
                "complexity_penalty": 0.0,
                "uniformity_penalty": 0.0,
            }
            for i in range(4)
        ],
    }
    deme = _build_xdeme(case, fs)
    before = [copy.deepcopy(list(deme[i].first)) for i in range(4)]
    hc = hill_climbing(optim_parameters(), hc_parameters())
    hc.cross_top_one(deme, 4, 3, 0, 4, _id_instance(fs, 0))
    # The sample window is permuted by partial_sort but its CONTENTS must
    # be untouched -- same multiset of instances, none merged into.
    after = sorted(tuple(deme[i].first) for i in range(4))
    assert after == sorted(tuple(b) for b in before)


# ---------------------------------------------- end to end, crossover default


@pytest.mark.oracle
@pytest.mark.parametrize("label", ["crossover_on", "crossover_off"])
def test_local_moses_with_crossover_matches_cpp_step_for_step(label):
    """S12's exit criterion: with `--hc-crossover=1` (classic's DEFAULT),
    the five-generation trajectory matches C++ step for step, and crossover
    is asserted to have fired -- on the iteration classic's own log names.
    """
    case = _GOLDEN["local_moses"][label]

    table = loadTable("tests/fixtures/table_data/trainingBoolean.dat", "y")
    bsc = ctruth_table_bscore(table.compressed())
    bsc.set_complexity_coef(3.5)
    cscorer = behave_cscore(bsc)
    mp = metapopulation(
        str_to_combo_tree("$1"),
        cscorer,
        metapop_parameters(),
        subsample_deme_filter_parameters(),
    )
    sig = gen_signature(type_node.boolean_type, 3)
    rule = logical_reduction(())(2)
    dparams = deme_parameters()
    dparams.perm_ratio = -1.0
    hc = _TracedHillClimbing(
        optim_parameters(),
        hc_parameters(widen=False, step=False, cross=case["hc_crossover"]),
    )
    dex = deme_expander(sig, rule, rule, cscorer, hc, dparams, subsample_deme_filter_parameters())

    randGen().seed(case["seed"])
    pa = moses_parameters(max_evals=20000, max_score=1000.0)
    stats = moses_statistics()

    for expected in case["generations"]:
        pa.max_gens = expected["gen"]
        local_moses(mp, dex, pa, stats)
        gen = expected["gen"]
        assert stats.n_expansions == expected["n_expansions"], f"{label} gen {gen}: n_expansions"
        assert stats.n_evals == expected["n_evals"], f"{label} gen {gen}: n_evals"
        assert mp.size() == expected["metapop_size"], f"{label} gen {gen}: metapop_size"
        csc = mp.best_composite_score()
        want = expected["best_composite_score"]
        assert csc.get_score() == want["score"], f"{label} gen {gen}: score"
        assert csc.get_complexity() == want["complexity"], f"{label} gen {gen}: complexity"
        assert csc.get_complexity_penalty() == want["complexity_penalty"], f"{label} gen {gen}"
        assert csc.get_penalized_score() == want["penalized_score"], f"{label} gen {gen}"
        assert combo_tree_to_str(mp.best_tree()).strip() == expected["best_tree"]

    assert hc.xover_log == _expected_xover_log(case), f"{label}: crossover firing pattern"


def test_crossover_changes_the_end_to_end_trajectory():
    """Guards against the whole crossover suite passing vacuously: if
    crossover ON and OFF produced the same run, `test_local_moses_..._on`
    would prove nothing about the new code. Measured, they differ from
    generation 4 onwards.
    """
    on = _GOLDEN["local_moses"]["crossover_on"]
    off = _GOLDEN["local_moses"]["crossover_off"]
    assert [g["n_evals"] for g in off["generations"]] == [45, 58, 74, 97, 113]
    assert [g["n_evals"] for g in on["generations"]] == [45, 58, 74, 198, 214]
    assert [e["iteration"] for e in on["xover_log"] if e["xover"]] == [3]
    assert [e["why"] for e in on["xover_log"] if e["xover"]] == ["last chance"]
    assert [e["iteration"] for e in off["xover_log"] if e["xover"]] == []
    # The crossover-OFF half reproduces S11's own pinned local_moses numbers,
    # confirming this probe's scenario is the same one, not a lookalike.
