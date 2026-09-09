"""Oracle tests for the continuous branches of neighborhood_sampling.

Three places in `neighborhood_sampling.{h,cc}` see continuous fields on
the ENUMERATION side (as opposed to the three SAMPLING helpers, which have
their own fixture in `test_neighborhood_sampling.py`):

  * `count_neighborhood_size_from_index`'s contin branch (cc:117-160)
  * `vary_n_knobs`' contin branch (h:289-349)
  * `randomize` / `generate_initial_sample` over a contin field_set

All three raised `NotImplementedError` until 2026-09-09. This file is the
oracle that replaced those raises.

Fixture: `tests/fixtures/contin_enumeration.json`, produced by
`tools/capture_contin_enumeration.py` from
`tools/cpp_probe/contin_enumeration_probe.cc` -- classic's own functions,
linked out of `libmoses.so`, not a reimplementation. The probe's module
comment lists which C++ arm each case reaches.

WHAT THE CORPUS ACTUALLY COVERS, and how that was established. Every
numeric trap below was verified by MUTATION: the "obvious wrong" version
was written, the suite re-run, and the number of failing rows recorded.
A trap whose mutation killed nothing is NOT covered, and this project's
most-repeated defect is a corpus where the faithful and the broken version
happen to agree.

    mutation                                              rows killed
    ---------------------------------------------------- -----------
    exact math.comb instead of boost's double arithmetic     8 / 115 binomial
    drop boost's `ceil(result - 0.5f)` rounding step          7 / 96  count
    `_double_to_unsigned` without the int64-range check       2 / 115 binomial
    `cni` accumulated without the unsigned-32 wrap (loop 1)   1 / 96  count
    `cni` accumulated without the unsigned-32 wrap (loop 2)   1 / 96  count
    `>= max_count` instead of `> max_count`                   2 / 96  count
    `remRLs` without the 32-bit truncation                    4 / 48  vary
    `pow2`'s shift-range assert removed                       2 / 2   throws
    recurse to `starting_index + 1` not `+ depth`            79 / 96  count
    `next_contin` without the `- relative_raw_idx` term      15 / 48  vary
    vary's Stop arm exploring Right before Left              21 / 48  vary
    `randomize` drawing disc before contin              10/15 order + 6/6 sample

Two notes on that table, both measured rather than assumed:

  * The first three mutations kill ZERO count/vary rows. They are caught
    only by the `binomial_double` section, because no reachable contin
    depth in the count corpus produces a binomial large enough to tell the
    implementations apart. Dropping that section would leave three traps
    silently uncovered.
  * One mutation is deliberately absent: rewriting C++'s
    `for(; relative_raw_idx < length; --length, ++itr)` to decrement
    `relative_raw_idx` instead kills nothing, and analysis agrees -- both
    forms run exactly `length - relative_raw_idx` times over the same
    `itr` range. The odd-looking loop is not a trap, so no test claims to
    cover it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymetta.moses.eda.initialization import randomize
from pymetta.moses.moses.neighborhood_sampling import (
    _boost_binomial_coefficient_double,
    _double_to_unsigned,
    count_neighborhood_size_from_index,
    generate_initial_sample,
    vary_n_knobs,
)
from pymetta.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "contin_enumeration.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


def _make_fs(specs) -> field_set:
    return field_set(
        [contin_spec(0.0, 1.0, 2.0, a) if kind == "contin" else disc_spec(a) for kind, a in specs]
    )


def _inst_from_raw(fs: field_set, raw) -> instance:
    inst = instance(fs.packed_width())
    for idx, value in enumerate(raw):
        fs.set_raw(inst, idx, value)
    return inst


def _raw(fs: field_set, inst: instance) -> list[int]:
    return [fs.get_raw(inst, k) for k in range(fs.raw_size())]


def _assert_layout(fs: field_set, layout, label: str) -> None:
    """`field_set` sorts its specs internally, so the probe dumps the raw
    layout it actually got and every case re-asserts it here. A silent
    layout difference would make every other assertion in the case
    meaningless while still looking like a pass.
    """
    assert fs.raw_size() == layout["raw_size"], label
    assert fs.begin_contin_raw_idx() == layout["begin_contin"], label
    assert fs.end_contin_raw_idx() == layout["end_contin"], label
    assert fs.begin_disc_raw_idx() == layout["begin_disc"], label
    assert fs.end_disc_raw_idx() == layout["end_disc"], label
    assert fs.begin_bit_raw_idx() == layout["begin_bit"], label
    assert fs.end_bit_raw_idx() == layout["end_bit"], label


# --------------------------------------------------------------- binomial


@pytest.mark.parametrize("case", _GOLDEN["binomial_double"], ids=lambda c: f"{c['n']}c{c['k']}")
def test_boost_binomial_coefficient_matches_cpp(case):
    """`(unsigned)boost::math::binomial_coefficient<double>(n, k)` --
    both the double and the cast, since the count branch only ever sees
    the cast result.
    """
    n, k = case["n"], case["k"]
    if n > 170:
        # boost's beta-function branch; the port gates rather than
        # approximating it. See `_boost_binomial_coefficient_double`.
        with pytest.raises(NotImplementedError):
            _boost_binomial_coefficient_double(n, k)
        return
    got = _boost_binomial_coefficient_double(n, k)
    assert got.hex() == float.fromhex(case["hex"]).hex()
    assert _double_to_unsigned(got) == case["trunc_unsigned"]


# ------------------------------------------------------------------ count


@pytest.mark.parametrize(
    "case", _GOLDEN["count"], ids=lambda c: f"{c['label']}-d{c['dist']}-i{c['starting_index']}"
)
def test_count_neighborhood_size_from_index_matches_cpp(case):
    fs = _make_fs([tuple(x) for x in case["specs"]])
    _assert_layout(fs, case["layout"], case["label"])
    center = case["center"] + [0] * (fs.raw_size() - len(case["center"]))
    inst = _inst_from_raw(fs, center)
    kwargs = {"max_count": case["max_count"]} if case["use_max_count"] else {}
    got = count_neighborhood_size_from_index(
        fs, inst, case["dist"], case["starting_index"], **kwargs
    )
    assert got == case["result"]


@pytest.mark.parametrize("case", _GOLDEN["count_throws"], ids=lambda c: c["label"])
def test_count_neighborhood_size_from_index_fails_where_cpp_fails(case):
    """`pow2`'s `OC_ASSERT(32 > x)` aborts classic for a contin field with
    `depth - length >= 32` at `dist >= 32`. The port raises there rather
    than returning a number classic never produces.
    """
    assert case["threw"], "fixture row is not a throwing case"
    fs = _make_fs([tuple(x) for x in case["specs"]])
    center = case["center"] + [0] * (fs.raw_size() - len(case["center"]))
    inst = _inst_from_raw(fs, center)
    with pytest.raises(ValueError, match="out of range"):
        count_neighborhood_size_from_index(fs, inst, case["dist"], case["starting_index"])


# ------------------------------------------------------------------- vary


@pytest.mark.parametrize(
    "case", _GOLDEN["vary"], ids=lambda c: f"{c['label']}-d{c['dist']}-i{c['starting_index']}"
)
def test_vary_n_knobs_matches_cpp(case):
    """Element for element, in emission order -- not set-wise. The order
    is what `generate_all_in_neighborhood`'s callers consume.
    """
    fs = _make_fs([tuple(x) for x in case["specs"]])
    _assert_layout(fs, case["layout"], case["label"])
    inst = _inst_from_raw(fs, case["center"])
    out: list[instance] = []
    vary_n_knobs(fs, inst, case["dist"], case["starting_index"], out, 10**9)
    assert [_raw(fs, i) for i in out] == case["instances"]


# -------------------------------------------------------------- RNG order


@pytest.mark.parametrize(
    "case", _GOLDEN["randomize_order"], ids=lambda c: f"{c['label']}-s{c['seed']}"
)
def test_randomize_matches_cpp_including_stream_position(case):
    """`post_randint` is one extra draw taken AFTER the call in both
    implementations, so a port that draws a different NUMBER of values is
    caught even when the resulting instance happens to match.
    """
    fs = _make_fs([tuple(x) for x in case["specs"]])
    _assert_layout(fs, case["layout"], case["label"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])
    randomize(fs, inst)
    post = randGen().randint_n(1000000)
    assert _raw(fs, inst) == case["raw"]
    assert post == case["post_randint"]


@pytest.mark.parametrize(
    "case", _GOLDEN["generate_initial_sample"], ids=lambda c: f"{c['label']}-s{c['seed']}"
)
def test_generate_initial_sample_contin_matches_cpp(case):
    """`generate_initial_sample` is the real caller whose path
    `_randomize`'s contin raise was blocking.
    """
    fs = _make_fs([tuple(x) for x in case["specs"]])
    _assert_layout(fs, case["layout"], case["label"])
    randGen().seed(case["seed"])
    out: list[instance] = []
    generate_initial_sample(fs, case["n"], out, case["n"])
    post = randGen().randint_n(1000000)
    assert [_raw(fs, i) for i in out] == case["instances"]
    assert post == case["post_randint"]
