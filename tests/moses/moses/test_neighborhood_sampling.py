"""Oracle and guard tests for pymoses.moses.moses.neighborhood_sampling.

The oracle fixture (tests/fixtures/neighborhood.json) is produced by
tools/cpp_probe/neighborhood_probe.cc against the real classic library (see
tools/capture_neighborhood.py) -- not reimplemented by hand. It carries four
sections: "deterministic" (count_neighborhood_size +
generate_all_in_neighborhood, no RNG), "safe_binomial_coefficient",
"generate_initial_sample", and "sample_from_neighborhood" (the latter two are
RNG-drawn; reproducing them element-for-element is this task's whole point).

`sample_new_instances` (both overloads) is tested separately, near the
bottom of this file, against tests/fixtures/hillclimbing.json -- S9 Task 2's
own fixture (tools/cpp_probe/hillclimbing_probe.cc), since it needs
`instance_set`, which did not exist when this file's original oracle
fixture was captured.

S9 fix-round coverage additions (2026-09): the max_count parameter of
`count_neighborhood_size_from_index` and the strict `<` routing boundary
inside `sample_new_instances` were both previously untested branches --
see `test_count_neighborhood_size_from_index_max_count_stops_recursion_early`
and `test_sample_new_instances_takes_exhaustive_path_at_the_equality_boundary`
below for the mutation each one pins.
"""

import json
from pathlib import Path

import pytest

from pymoses.moses.moses.neighborhood_sampling import (
    _SIZE_T_MAX,
    _randomize,
    count_neighborhood_size,
    count_neighborhood_size_from_index,
    flip_LR,
    generate_all_in_neighborhood,
    generate_contin_neighbor,
    generate_initial_sample,
    safe_binomial_coefficient,
    sample_from_neighborhood,
    sample_new_instances,
    twiddle_contin_bit,
    vary_n_knobs,
)
from pymoses.moses.moses.types import worst_composite_score
from pymoses.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "neighborhood.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

_HC_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "hillclimbing.json"
_HC_GOLDEN = json.loads(_HC_FIXTURE_PATH.read_text())

_CONTIN_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "contin_neighborhood.json"
_CONTIN_GOLDEN = json.loads(_CONTIN_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


def test_golden_fixture_has_expected_shape(golden):
    assert set(golden) == {
        "deterministic",
        "safe_binomial_coefficient",
        "generate_initial_sample",
        "sample_from_neighborhood",
    }
    assert len(golden["deterministic"]) == 11
    assert len(golden["safe_binomial_coefficient"]) == 8
    assert len(golden["generate_initial_sample"]) == 12
    assert len(golden["sample_from_neighborhood"]) == 7
    for case in golden["deterministic"]:
        assert len(case["neighbors"]) == case["count" if case["has_center"] else "count_null"]
    for case in golden["generate_initial_sample"]:
        assert len(case["instances"]) == case["n"]
    for case in golden["sample_from_neighborhood"]:
        assert len(case["instances"]) == case["sample_size"]


def _uniform_fs(multiplicity: int, num_knobs: int) -> field_set:
    return field_set(disc_spec(multiplicity), num_knobs)


def _mixed_fs(multiplicities: list[int]) -> field_set:
    return field_set([disc_spec(m) for m in multiplicities])


def _raw_tuple(fs: field_set, inst: instance) -> tuple[int, ...]:
    return tuple(fs.get_raw(inst, i) for i in range(fs.raw_size()))


# ---------------------------------------------------------------- deterministic


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["deterministic"],
    ids=lambda c: f"m{c['multiplicity']}_k{c['num_knobs']}_d{c['distance']}_c{c['has_center']}",
)
def test_deterministic_matches_cpp(case):
    fs = _uniform_fs(case["multiplicity"], case["num_knobs"])

    if case["has_center"]:
        center = instance(fs.packed_width())
        for idx, value in enumerate(case["center"]):
            fs.set_raw(center, idx, value)
    else:
        center = None

    count_null = count_neighborhood_size(fs, case["distance"])
    assert count_null == case["count_null"]

    if case["has_center"]:
        count_with = count_neighborhood_size(fs, case["distance"], inst=center)
        assert count_with == case["count"]

    out: list[instance] = []
    if case["has_center"]:
        generate_all_in_neighborhood(fs, case["distance"], out, case["count"], center_inst=center)
    else:
        generate_all_in_neighborhood(fs, case["distance"], out, case["count_null"])

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(n) for n in case["neighbors"]]
    assert got == expected


# ---------------------------------------------------------- safe_binomial_coefficient


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["safe_binomial_coefficient"], ids=lambda c: f"k{c['k']}_n{c['n']}"
)
def test_safe_binomial_coefficient_matches_cpp(case):
    assert safe_binomial_coefficient(case["k"], case["n"]) == case["result"]


def test_safe_binomial_coefficient_k_less_than_n_returns_zero():
    # DIVERGENCE: no C++ oracle value exists for this input -- see the
    # function's own docstring marker. k < n never occurs at the real call
    # site (count_neighborhood_size_from_index guards dist <= rb first).
    assert safe_binomial_coefficient(3, 4) == 0


# --------------------------------------------------- count_neighborhood_size max_count


def test_count_neighborhood_size_from_index_max_count_stops_recursion_early():
    """S9 fix-round coverage: no test in this port called
    `count_neighborhood_size_from_index` (neighborhood_sampling.cc:99) with
    a non-default `max_count` before this one, leaving its early-stop
    guard (neighborhood_sampling.cc:169-170, "stop prematurely if above
    max_count") completely unpinned. This early return is NOT a pure
    performance optimisation with an unchanged result -- it makes the
    function return the "skip this knob" recursive sub-count ALONE,
    omitting the additive "use this knob" term, so the returned value
    genuinely differs (and is smaller) whenever the guard fires.

    Needs a genuinely multi-valued disc field (multiplicity 3): a
    multiplicity-2 `disc_spec` is folded into `field_set`'s bit range
    (width == 1, see `field_set.py`'s `_build_disc_spec`/`_nbool`), which
    takes the separate, non-recursive bit branch
    (neighborhood_sampling.cc:172-179) that has no `max_count` guard at
    all -- a bit-only field would silently fail to exercise this line.

    C++-observable: measured directly against classic via
    `tools/cpp_probe/hcextra_max_count_probe.cc` (5 knobs, multiplicity 3,
    distance 3) --

        g++ -O2 -std=c++17 -I $CLASSIC -I $PREFIX/include \\
            -o tools/cpp_probe/build/hcextra_max_count_probe \\
            tools/cpp_probe/hcextra_max_count_probe.cc \\
            -L $PREFIX/lib -L $PREFIX/lib/opencog \\
            -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$PREFIX/lib/opencog \\
            -lmoses -lcomboreduct -lcogutil
        ( ulimit -v 2097152; timeout 300 ./tools/cpp_probe/build/hcextra_max_count_probe )

    printed (2026-09-08, run under this session's standing OOM bound):
    `{"begin_disc_raw_idx":0,"end_disc_raw_idx":5,"dist":3,"uncapped":80,
    "capped":[{"max_count":1,"result":8},{"max_count":5,"result":8},
    {"max_count":10,"result":32},{"max_count":20,"result":32},
    {"max_count":50,"result":80},{"max_count":100,"result":80}]}` --
    bit-identical to this port's own values below (the function is
    deterministic; no RNG is involved).

    MUTATION: delete the early-return (or force its condition `False`) --
    verified every capped call then returns the full uncapped 80 instead
    of 8/8/32/32/80/80. Confirmed by hand: reverting the mutation restores
    the values asserted below.
    """
    fs = _mixed_fs([3, 3, 3, 3, 3])
    inst = instance(fs.packed_width())
    assert fs.begin_disc_raw_idx() == 0
    assert fs.end_disc_raw_idx() == 5

    uncapped = count_neighborhood_size_from_index(fs, inst, 3, 0)
    assert uncapped == 80

    expected_by_cap = {1: 8, 5: 8, 10: 32, 20: 32, 50: 80, 100: 80}
    for cap, expected in expected_by_cap.items():
        assert count_neighborhood_size_from_index(fs, inst, 3, 0, cap) == expected


# ------------------------------------------------------------- generate_initial_sample


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["generate_initial_sample"],
    ids=lambda c: f"{c['label']}_seed{c['seed']}",
)
def test_generate_initial_sample_matches_cpp(case):
    fs = _mixed_fs(case["multiplicities"])
    randGen().seed(case["seed"])

    out: list[instance] = []
    generate_initial_sample(fs, case["n"], out, case["n"])

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(row) for row in case["instances"]]
    assert got == expected


# ---------------------------------------------------------- sample_from_neighborhood


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["sample_from_neighborhood"],
    ids=lambda c: f"seed{c['seed']}_d{c['dist']}",
)
def test_sample_from_neighborhood_matches_cpp(case):
    fs = _mixed_fs(case["multiplicities"])
    center = instance(fs.packed_width())
    for idx, value in enumerate(case["center_raw"]):
        fs.set_raw(center, idx, value)

    randGen().seed(case["seed"])
    out: list[instance] = []
    sample_from_neighborhood(
        fs, case["dist"], case["sample_size"], out, case["sample_size"], center
    )

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(row) for row in case["instances"]]
    assert got == expected


# --------------------------------------------------------------------- guards


def test_emit_raises_when_buffer_full():
    fs = _uniform_fs(2, 3)
    inst = instance(fs.packed_width())
    with pytest.raises(IndexError):
        vary_n_knobs(fs, inst, 0, 0, [], 0)


def test_generate_all_in_neighborhood_rejects_wrong_center_width():
    fs = _uniform_fs(2, 3)
    bad_center = instance(fs.packed_width() + 1)
    with pytest.raises(ValueError):
        generate_all_in_neighborhood(fs, 1, [], 10, center_inst=bad_center)


def test_sample_from_neighborhood_rejects_wrong_center_width():
    fs = _uniform_fs(2, 3)
    bad_center = instance(fs.packed_width() + 1)
    with pytest.raises(ValueError):
        sample_from_neighborhood(fs, 1, 5, [], 5, bad_center)


def test_sample_from_neighborhood_rejects_distance_over_dimension():
    # match= pins this to the explicit dist > dim guard's own message, not
    # merely "some ValueError somewhere": without the guard, lazy_selector's
    # own "selector is empty" ValueError still fires once all dim indices
    # are exhausted, which would let a bare pytest.raises(ValueError) pass
    # vacuously even with the guard deleted (measured directly: removing the
    # `if dist > dim` check leaves this test green using a bare check).
    fs = _uniform_fs(2, 3)
    center = instance(fs.packed_width())
    with pytest.raises(ValueError, match="cannot be greater than the field dimension"):
        sample_from_neighborhood(fs, fs.dim_size() + 1, 5, [], 5, center)


def test_sample_from_neighborhood_zero_distance_leaves_instance_unchanged():
    fs = _mixed_fs([2, 2, 2, 3, 4, 5])
    center = instance(fs.packed_width())
    fs.set_raw(center, 0, 2)
    randGen().seed(0)
    out: list[instance] = []
    sample_from_neighborhood(fs, 0, 4, out, 4, center)
    for inst in out:
        assert _raw_tuple(fs, inst) == _raw_tuple(fs, center)


def _fs_with_contin() -> field_set:
    return field_set([contin_spec(0.0, 1.0, 2.0, 3), disc_spec(3)])


# These two used to assert that `count_neighborhood_size_from_index` and
# `vary_n_knobs` RAISE on a contin field. Both branches are ported now
# (2026-09-09); the oracle comparison lives in
# tests/moses/moses/test_contin_enumeration.py, and what remains here is
# the direct check that they run and produce something sane, which is what
# these two tests were positioned to cover.


def test_count_neighborhood_size_handles_a_contin_field():
    fs = _fs_with_contin()
    # depth-3 contin + a multiplicity-3 disc. At distance 1 from the
    # all-Stop center: 2 settings for the contin's first pseudo-bit
    # (Left/Right) plus 2 non-identity settings of the disc knob.
    got = count_neighborhood_size_from_index(fs, instance(fs.packed_width()), 1, 0)
    assert got == 4


def test_vary_n_knobs_handles_a_contin_field():
    fs = _fs_with_contin()
    out: list[instance] = []
    vary_n_knobs(fs, instance(fs.packed_width()), 1, 0, out, 100)
    # Same four neighbours, now enumerated rather than counted -- and all
    # distinct, which is what makes the count above a count of anything.
    assert len(out) == 4
    assert len({_raw_tuple(fs, i) for i in out}) == 4


def test_sample_from_neighborhood_contin_field_no_longer_raises():
    """S14 Task 3: `sample_from_neighborhood`'s "modify contin" branch now
    delegates to `generate_contin_neighbor` instead of raising -- see the
    oracle tests below (`test_sample_from_neighborhood_contin_matches_cpp`)
    for the element-for-element comparison against classic. This is a
    direct, non-oracle smoke check that it runs to completion, and that a
    small distance-1 sample (which can land on either the disc knob or the
    contin knob but never both) sometimes actually reaches the contin
    field -- i.e. this is not accidentally still a no-op for it.
    """
    fs = _fs_with_contin()
    center = instance(fs.packed_width())
    fs.set_raw(center, 3, 2)
    randGen().seed(0)
    out: list[instance] = []
    sample_from_neighborhood(fs, 1, 20, out, 20, center)
    assert len(out) == 20
    contin_before = tuple(fs.get_raw(center, i) for i in range(3))
    touched_contin = any(
        tuple(fs.get_raw(inst, i) for i in range(3)) != contin_before for inst in out
    )
    assert touched_contin, "20 distance-1 draws never once landed on the contin knob"


# ---------------------------------------------------- S14 Task 3: contin sampling
#
# Oracle fixture: tests/fixtures/contin_neighborhood.json, produced by
# tools/cpp_probe/contin_neighborhood_probe.cc (see tools/capture_contin_
# neighborhood.py) against the real classic library -- covering flip_LR,
# twiddle_contin_bit, generate_contin_neighbor directly, plus
# sample_from_neighborhood's own contin branch end to end. The latter is
# this task's actual exit criterion ("for a fixed seed, contin
# neighbourhoods match C++ element for element").


def _contin_only_fs(depth: int) -> field_set:
    return field_set(contin_spec(0.0, 1.0, 2.0, depth), 1)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _CONTIN_GOLDEN["flip_LR"],
    ids=lambda c: f"raw{c['raw_idx']}_before{c['before']}",
)
def test_flip_LR_matches_cpp(case):
    fs = _contin_only_fs(3)
    inst = instance(fs.packed_width())
    fs.set_raw(inst, case["raw_idx"], case["before"])
    flip_LR(fs, inst, case["raw_idx"])
    assert fs.get_raw(inst, case["raw_idx"]) == case["after"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _CONTIN_GOLDEN["twiddle_contin_bit"],
    ids=lambda c: (
        f"cur{c['cur']}_nextinvalid{c['next_invalid']}_next{c['next_val']}_seed{c['seed']}"
    ),
)
def test_twiddle_contin_bit_matches_cpp(case):
    """Also checks a `randbool()` drawn immediately AFTER the call, so a
    case that should draw NOTHING (the plain-flip branch) is distinguished
    from one that draws exactly one `randbool()` -- a value-only comparison
    of ``after`` cannot tell those apart when the flip and the Stop-branch
    happen to choose the same resulting raw value.
    """
    fs = _contin_only_fs(6)
    inst = instance(fs.packed_width())
    fs.set_raw(inst, 0, case["cur"])
    if not case["next_invalid"]:
        fs.set_raw(inst, 1, case["next_val"])

    randGen().seed(case["seed"])
    next_idx = fs.raw_size() if case["next_invalid"] else 1
    twiddle_contin_bit(fs, inst, 0, next_idx)

    assert fs.get_raw(inst, 0) == case["after"]
    assert randGen().randbool() == case["post_randbool"]


def _contin_fixture_fs(depths: list[int], trailing_disc: bool) -> field_set:
    specs: list = [contin_spec(0.0, 1.0, 2.0, d) for d in depths]
    if trailing_disc:
        specs.append(disc_spec(3))
    return field_set(specs)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _CONTIN_GOLDEN["generate_contin_neighbor"],
    ids=lambda c: f"{c['label']}_dist{c['dist']}_seed{c['seed']}",
)
def test_generate_contin_neighbor_matches_cpp(case):
    fs = _contin_fixture_fs(case["depths"], case["trailing_disc"])
    inst = instance(fs.packed_width())
    for i, v in enumerate(case["raw"]):
        fs.set_raw(inst, i, v)
    if case["trailing_disc"]:
        fs.set_raw(inst, len(case["raw"]), 2)

    randGen().seed(case["seed"])
    generate_contin_neighbor(fs, inst, case["idx"], case["dist"])

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["result"]


def _sfn_contin_fs() -> field_set:
    return field_set(
        [
            contin_spec(0.0, 1.0, 2.0, 5),
            contin_spec(0.0, 1.0, 2.0, 3),
            disc_spec(3),
            disc_spec(4),
            disc_spec(2),
            disc_spec(2),
        ]
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _CONTIN_GOLDEN["sample_from_neighborhood_contin"],
    ids=lambda c: f"seed{c['seed']}_d{c['dist']}",
)
def test_sample_from_neighborhood_contin_matches_cpp(case):
    """The task's own exit criterion: element-for-element agreement with
    classic for `sample_from_neighborhood` over a field_set that includes
    contin fields, at a fixed seed.
    """
    fs = _sfn_contin_fs()
    center = instance(fs.packed_width())
    for idx, value in enumerate(case["center_raw"]):
        fs.set_raw(center, idx, value)

    randGen().seed(case["seed"])
    out: list[instance] = []
    sample_from_neighborhood(
        fs, case["dist"], case["sample_size"], out, case["sample_size"], center
    )

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(row) for row in case["instances"]]
    assert got == expected


def test_generate_contin_neighbor_zero_distance_is_noop():
    """neighborhood_sampling.cc:64-65's early return: `dist == 0` leaves
    `inst` completely unchanged and draws nothing from the RNG.

    MUTATION: deleting the `if dist == 0: return` guard falls through into
    `OC_ASSERT(dist <= depth)` -- still true for dist=0 -- and then into
    `lazy_random_selector(uppest_low, lowest_low, rng)()`, which DOES draw
    from the RNG and can mutate `inst` (uppest_low/lowest_low still bracket
    a valid, nonempty range at dist=0: `lowest_low=length`,
    `uppest_low=min(length+1,depth+1)=length+1`, a 1-wide range -- so it
    does not even raise, it silently draws and, depending on that draw,
    can `twiddle_contin_bit` a real pseudo-bit). Verified directly:
    commenting out the guard changes `randGen().randbool()` drawn
    immediately after the call from a value identical across both seeds
    below to one that depends on the seed, and can flip `inst`'s raw
    values -- confirmed by hand, reverted after confirming red/green.
    """
    fs = _contin_only_fs(4)
    inst = instance(fs.packed_width())
    fs.set_raw(inst, 0, 1)
    fs.set_raw(inst, 1, 2)
    fs.set_raw(inst, 2, 1)
    fs.set_raw(inst, 3, 0)
    before = [fs.get_raw(inst, i) for i in range(fs.raw_size())]

    for seed in (0, 1, 999):
        randGen().seed(seed)
        generate_contin_neighbor(fs, inst, 0, 0)
        assert [fs.get_raw(inst, i) for i in range(fs.raw_size())] == before
        # No RNG draw either: the very next randbool() must depend only on
        # the seed, not on any hidden draw generate_contin_neighbor made.
        randGen().seed(seed)
        expected_next = randGen().randbool()
        randGen().seed(seed)
        generate_contin_neighbor(fs, inst, 0, 0)
        assert randGen().randbool() == expected_next


def test_generate_contin_neighbor_rejects_dist_over_depth():
    """DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.cc:67):
    `OC_ASSERT(dist <= depth)` becomes `ValueError`. `match=` pins this to
    the explicit guard's own message, not merely "some ValueError
    somewhere": for dist > depth, `uppest_low <= lowest_low` always holds
    too (`depth + 1 - dist <= 0 <= lowest_low`), so `lazy_random_selector`'s
    own "cannot select from an empty list" ValueError would ALSO fire if
    this guard were deleted -- a bare `pytest.raises(ValueError)` would
    pass vacuously either way. Measured directly: removing the explicit
    guard here still raises, but with lazy_selector's message instead of
    this function's own.
    """
    fs = _contin_only_fs(3)
    inst = instance(fs.packed_width())
    with pytest.raises(ValueError, match="must not exceed depth"):
        generate_contin_neighbor(fs, inst, 0, 4)


def test_twiddle_contin_bit_short_circuits_before_reading_invalid_next():
    """Pins the `or` short-circuit documented in twiddle_contin_bit's own
    docstring: when `next_idx` is the `fs.raw_size()` "invalid" sentinel
    AND `idx + 1 != next_idx` (the ordinary case -- more fields exist past
    this contin one), `fs.get_raw(inst, next_idx)` must NEVER be evaluated,
    since it is genuinely out of range.

    MUTATION: replacing the `or` short-circuit with an unconditional read
    (e.g. `if fs.get_raw(inst, next_idx) == contin_spec.Stop or idx + 1 !=
    next_idx:`, which evaluates the read FIRST) makes this raise
    `IndexError` instead of completing -- confirmed by hand, reverted
    after confirming red/green.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 3), disc_spec(3)])
    inst = instance(fs.packed_width())
    fs.set_raw(inst, 2, contin_spec.Left)  # a non-Stop pseudo-bit at the top
    randGen().seed(0)
    # idx=2 is this contin field's LAST raw position; idx+1=3 is a REAL
    # disc-field position, not fs.raw_size()=4 -- so passing the sentinel
    # (raw_size()) as next_idx must take the "invalid" branch via the `or`
    # short-circuit, never reading raw index 4 (which does not exist).
    twiddle_contin_bit(fs, inst, 2, fs.raw_size())
    assert fs.get_raw(inst, 2) in (contin_spec.Stop, contin_spec.Left, contin_spec.Right)


def test_twiddle_contin_bit_treats_end_sentinel_as_invalid_when_it_collides_with_idx_plus_one():
    """DIVERGENCE (see twiddle_contin_bit's own docstring): when a contin
    field IS the very last thing in the whole raw layout, `idx + 1` at its
    last position genuinely EQUALS `fs.raw_size()` -- the same value used
    as the "invalid" sentinel -- so the `idx + 1 != next_idx` short-circuit
    does not fire. C++ then dereferences `end_raw` (`_fields[size()]`, UB,
    heap-dependent; see contin_neighborhood_probe.cc's own comment on why
    no oracle value exists). The port treats the sentinel as "invalid",
    the header's documented rule and C++'s own behaviour whenever any
    field follows: one `randbool()` is drawn, choosing Stop or a flip --
    the same outcome/draw count as the oracle-pinned `next_invalid` cases
    (cur=Left, seed=s) of `test_twiddle_contin_bit_matches_cpp`.

    MUTATION: dropping the `next_idx == fs.raw_size()` clause makes this
    raise `IndexError` (field_set.get_raw's bounds check) again.
    """
    for case in _CONTIN_GOLDEN["twiddle_contin_bit"]:
        if not case["next_invalid"] or case["cur"] == contin_spec.Stop:
            continue
        fs = _contin_only_fs(3)  # contin is the ENTIRE raw layout: raw_size() == 3
        inst = instance(fs.packed_width())
        fs.set_raw(inst, 2, case["cur"])
        randGen().seed(case["seed"])
        twiddle_contin_bit(fs, inst, 2, fs.raw_size())
        assert fs.get_raw(inst, 2) == case["after"]
        assert randGen().randbool() == case["post_randbool"]


def test_generate_contin_neighbor_last_field_at_full_depth_does_not_raise():
    """Regression for a crash seen in `moses -a hc` on a contin-only
    regression deme (20 contin knobs of depth 5, no disc/bit fields):
    `generate_contin_neighbor` picked `up == depth` on the LAST contin
    field, passed the `fs.raw_size()` sentinel as `next_idx`, which
    equalled `idx + 1`, and `fs.get_raw(inst, raw_size())` raised
    `IndexError: raw field index out of range`.

    Here the contin field is fully used (length == depth == 3) and
    dist=1, so `low` is drawn by `lazy_random_selector(3, 2)` -- which,
    faithfully to CogUtil, draws over `[0, 3)` -- and `low == 2` gives
    `up == 3 == depth`, the crash path; the seed loop below is checked to
    hit it. Each result must match what C++ produces (deterministically)
    for the same field followed by a disc field, where `end_raw` is not
    adjacent to the last pseudo-bit.
    """
    hit_last_bit = False
    for seed in range(16):
        results = []
        for trailing_disc in (False, True):
            fs = _contin_fixture_fs([3], trailing_disc)
            inst = instance(fs.packed_width())
            for i in range(3):
                fs.set_raw(inst, i, contin_spec.Left)
            randGen().seed(seed)
            generate_contin_neighbor(fs, inst, 0, 1)
            results.append(([fs.get_raw(inst, i) for i in range(3)], randGen().randbool()))
        assert results[0] == results[1]
        if results[0][0][2] != contin_spec.Left:
            hit_last_bit = True
    assert hit_last_bit


def test_sample_from_neighborhood_contin_branch_hardcodes_distance_one():
    """DO NOT FIX (see the inline comment at the "modify contin" branch in
    sample_from_neighborhood): C++ hardcodes `1` at
    neighborhood_sampling.h:178's `generate_contin_neighbor(fs, new_inst,
    itc, 1)` call, ignoring `sample_from_neighborhood`'s own `dist`
    parameter entirely -- confirmed by the oracle fixture itself already
    (`test_sample_from_neighborhood_contin_matches_cpp` includes seed=
    12345/dist=4 cases that only match because the port ALSO hardcodes 1).
    This is a direct, isolated proof of the same point: a single contin
    hit changes exactly one raw pseudo-bit position, never more, no matter
    what the OUTER `dist` argument is -- verified by hand (this session)
    that threading `dist` through instead diverges seed=12345/dist=4 from
    the golden fixture, then reverted to restore the match.
    """
    fs = _sfn_contin_fs()
    center = instance(fs.packed_width())
    center_raw = [1, 2, 1, 0, 0, 2, 1, 0, 1, 2, 0]
    for idx, value in enumerate(center_raw):
        fs.set_raw(center, idx, value)

    # Raw ranges: contin #1 depth 5 -> [0, 5); contin #2 depth 3 -> [5, 8).
    contin1_range = range(0, 5)
    contin2_range = range(5, 8)

    randGen().seed(12345)
    out: list[instance] = []
    sample_from_neighborhood(fs, 4, 30, out, 30, center)

    saw_contin1_hit = saw_contin2_hit = False
    for inst in out:
        diffs1 = [i for i in contin1_range if fs.get_raw(inst, i) != fs.get_raw(center, i)]
        diffs2 = [i for i in contin2_range if fs.get_raw(inst, i) != fs.get_raw(center, i)]
        # If `dist` (4) were threaded through instead of the hardcoded 1:
        # contin #2 has depth 3 < 4, so generate_contin_neighbor would
        # immediately raise ValueError ("dist must not exceed depth") the
        # moment a sample lands on it -- this loop would never reach this
        # assertion at all for such a sample. Contin #1 has depth 5 >= 4,
        # so it would NOT raise, but would touch up to 4 raw positions in
        # one hit instead of exactly 1 -- caught below.
        assert len(diffs1) <= 1, f"contin #1 changed {len(diffs1)} positions in one hit: {diffs1}"
        assert len(diffs2) <= 1, f"contin #2 changed {len(diffs2)} positions in one hit: {diffs2}"
        saw_contin1_hit = saw_contin1_hit or bool(diffs1)
        saw_contin2_hit = saw_contin2_hit or bool(diffs2)

    # Sanity: this seed/sample_size combination must actually exercise
    # BOTH contin fields at least once, or the assertions above would be
    # vacuously true.
    assert saw_contin1_hit
    assert saw_contin2_hit


def test_randomize_delegates_to_the_eda_port():
    """`_randomize` used to raise on any contin field and is now a thin
    forwarder to `eda.initialization.randomize`, which is the real port.

    Asserted by delegation rather than by re-checking values: the values
    have their own oracle test (test_contin_enumeration.py's
    `randomize_order` section, which also pins the RNG draw ORDER). What
    this test protects is that the forwarding still happens, so the two
    cannot silently drift back into being separate implementations.
    """
    import pymoses.moses.eda.initialization as eda_init

    fs = _fs_with_contin()
    seen = []

    real = eda_init.randomize

    def spy(fs_, inst_, rng_=None):
        seen.append((fs_, inst_))
        return real(fs_, inst_, rng_)

    eda_init.randomize = spy
    try:
        inst = instance(fs.packed_width())
        _randomize(fs, inst)
    finally:
        # Restore by DELETING the shadow rather than reassigning: an
        # earlier test in this project "restored" a patched name by
        # assignment and left a permanent shadow behind.
        eda_init.randomize = real

    assert len(seen) == 1
    assert seen[0][0] is fs


def test_generate_initial_sample_handles_a_contin_field():
    fs = _fs_with_contin()
    randGen().seed(0)
    out: list[instance] = []
    generate_initial_sample(fs, 4, out, 4)
    assert len(out) == 4
    # Not all-zero: the contin field was actually randomized, not skipped.
    assert any(_raw_tuple(fs, i) != (0,) * fs.raw_size() for i in out)


def test_size_t_max_is_64_bit():
    assert _SIZE_T_MAX == 2**64 - 1


# ------------------------------------------------------------ sample_new_instances


def _sni_center(fs: field_set, n_knobs: int) -> instance:
    center = instance(fs.packed_width())
    for i in range(0, n_knobs, 3):
        fs.set_raw(center, i, 1)
    return center


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _HC_GOLDEN["sample_new_instances"],
    ids=lambda c: (
        f"m{c['multiplicity']}_k{c['n_knobs']}_d{c['dist']}_seed{c['seed']}_total{c['give_total']}"
    ),
)
def test_sample_new_instances_matches_cpp(case):
    fs = _uniform_fs(case["multiplicity"], case["n_knobs"])
    center = _sni_center(fs, case["n_knobs"])
    deme = instance_set(case["current_number_of_instances"], fs)

    randGen().seed(case["seed"])
    if case["give_total"]:
        total = count_neighborhood_size(
            fs, case["dist"], case["number_of_new_instances"], inst=center
        )
        returned = sample_new_instances(
            total,
            case["number_of_new_instances"],
            case["current_number_of_instances"],
            center,
            deme,
            case["dist"],
        )
    else:
        returned = sample_new_instances(
            None,
            case["number_of_new_instances"],
            case["current_number_of_instances"],
            center,
            deme,
            case["dist"],
        )

    assert returned == case["returned"]
    assert len(deme) == case["deme_size"]

    got_new_slice = [
        _raw_tuple(fs, deme[i].first) for i in range(case["current_number_of_instances"], len(deme))
    ]
    expected = [tuple(row) for row in case["new_slice"]]
    assert got_new_slice == expected

    # Every newly-written slot must carry a FRESH default (worst)
    # composite_score, matching scored_instance(instance) -- see
    # neighborhood_sampling.py's _DemeWriter docstring -- not whatever
    # score happened to occupy that slot before deme.resize() grew it.
    for i in range(case["current_number_of_instances"], len(deme)):
        assert deme[i].second == worst_composite_score


def test_sample_new_instances_overload_2_matches_overload_1_without_precomputed_total():
    """Not itself an oracle case (no separate C++ value exists beyond the
    fixture's own give_total=false rows already covered above) -- a direct
    unit check that passing total_number_of_neighbours=None recomputes it
    identically to the caller precomputing it, for a case NOT already in
    the fixture, confirming the merge itself (not just the two fixture-
    covered shapes) is correct.
    """
    fs = _uniform_fs(2, 6)
    center = instance(fs.packed_width())

    randGen().seed(7)
    deme_a = instance_set(0, fs)
    returned_a = sample_new_instances(None, 3, 0, center, deme_a, 1)

    randGen().seed(7)
    deme_b = instance_set(0, fs)
    total = count_neighborhood_size(fs, 1, 3, inst=center)
    returned_b = sample_new_instances(total, 3, 0, center, deme_b, 1)

    assert returned_a == returned_b
    assert [_raw_tuple(fs, si.first) for si in deme_a] == [
        _raw_tuple(fs, si.first) for si in deme_b
    ]


def test_sample_new_instances_grows_deme_in_place():
    fs = _uniform_fs(2, 10)
    center = instance(fs.packed_width())
    deme = instance_set(4, fs)  # 4 pre-existing (unrelated) instances
    randGen().seed(1)
    returned = sample_new_instances(None, 5, 4, center, deme, 1)
    assert returned == 5
    assert len(deme) == 9  # 4 pre-existing + 5 new, not replaced/truncated


def test_sample_new_instances_takes_exhaustive_path_at_the_equality_boundary():
    """S9 fix-round coverage: pins neighborhood_sampling.cc:230's `if
    (number_of_new_instances < total_number_of_neighbours)` -- STRICTLY
    less-than -- as the routing condition between the RNG-sampling branch
    (`sample_from_neighborhood`) and the exhaustive branch
    (`generate_all_in_neighborhood`). No existing test exercises the exact
    equality boundary (`number_of_new_instances == total_number_of_
    neighbours`), where the two branches disagree: at strict equality the
    correct code must take the EXHAUSTIVE, deterministic branch (there is
    nothing left to sample), not draw via RNG.

    Both underlying functions are independently oracle-verified elsewhere
    in this file (`test_deterministic_matches_cpp`,
    `test_sample_from_neighborhood_matches_cpp`) -- this test does not
    re-derive their fidelity, only that `sample_new_instances` ROUTES to
    the right one of the two at this exact boundary, by comparing its
    output at the boundary against a direct, independent call to
    `generate_all_in_neighborhood` for the identical (fs, dist, center,
    total) inputs.

    MUTATION (`<` -> `<=`): verified this routes the equality case into
    the RNG-sampling branch instead, consuming `randGen()` and producing a
    result that no longer matches the exhaustive enumeration (measured
    directly, same seed: `[(0,0,1),(1,0,0),(2,0,0),(3,0,0),(0,1,0),
    (0,2,0)]` from the correct/exhaustive path vs `[(0,2,0),(0,0,1),
    (2,0,0),(0,1,0),(0,1,0),(0,2,0)]` -- note the repeated `(0,1,0)` --
    from the mutated/sampling path). Confirmed by hand: reverting the
    mutation restores the match asserted below.
    """
    fs = _mixed_fs([2, 3, 4])
    center = instance(fs.packed_width())
    total = count_neighborhood_size(fs, 1, inst=center)
    assert total == 6  # (multy-1) summed over the 3 knobs: 1 + 2 + 3

    randGen().seed(5)
    deme = instance_set(0, fs)
    returned = sample_new_instances(total, total, 0, center, deme, 1)
    assert returned == total
    assert len(deme) == total

    expected: list[instance] = []
    generate_all_in_neighborhood(fs, 1, expected, total, center_inst=center)

    got = [_raw_tuple(fs, si.first) for si in deme]
    want = [_raw_tuple(fs, inst) for inst in expected]
    assert got == want
