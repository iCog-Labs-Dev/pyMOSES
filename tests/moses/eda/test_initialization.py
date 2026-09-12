"""Oracle and guard tests for pymoses.moses.eda.initialization.

The oracle fixture (tests/fixtures/eda_initialization.json) is produced by
tools/cpp_probe/eda_initialization_probe.cc against the real classic
library (see tools/capture_eda_initialization.py) -- not reimplemented by
hand. It carries five sections: "uniform_randomize_bit",
"uniform_randomize_disc", "occam_randomize_contin_single",
"occam_randomize_contin_all", and "randomize" (the combining wrapper,
proving contin -> disc -> bit draw order end to end). Every case brackets
the call with a seeded pre/post `randbool()` marker so an extra or missing
draw is caught even when the resulting raw value happens to coincide.
"""

import json
from pathlib import Path

import pytest

from pymoses.moses.eda.initialization import (
    occam_randomize_contin,
    randomize,
    uniform_randomize_bit,
    uniform_randomize_disc,
)
from pymoses.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "eda_initialization.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


def _bit_fs(n_bits: int) -> field_set:
    return field_set([disc_spec(2) for _ in range(n_bits)])


def _disc_fs(multys: list[int]) -> field_set:
    return field_set([disc_spec(m) for m in multys])


def _contin_single_fs(depth: int) -> field_set:
    # Trailing disc(3), matching the probe: keeps the contin field off the
    # raw-array boundary (see the probe's own comment on the C++ UB this
    # avoids in generate_contin_neighbor's callee).
    return field_set([contin_spec(0.0, 1.0, 2.0, depth), disc_spec(3)])


def _contin_all_fs(depths: list[int]) -> field_set:
    specs = [contin_spec(0.0, 1.0, 2.0, d) for d in depths]
    specs.append(disc_spec(3))
    return field_set(specs)


def _mixed_fs(contin_depths: list[int], disc_multys: list[int], n_bits: int) -> field_set:
    specs = [contin_spec(0.0, 1.0, 2.0, d) for d in contin_depths]
    specs += [disc_spec(m) for m in disc_multys]
    specs += [disc_spec(2) for _ in range(n_bits)]
    return field_set(specs)


# ---------------------------------------------------------------- uniform_randomize_bit


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["uniform_randomize_bit"],
    ids=lambda c: f"nbits{c['n_bits']}_seed{c['seed']}",
)
def test_uniform_randomize_bit_matches_cpp(case):
    fs = _bit_fs(case["n_bits"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])

    uniform_randomize_bit(fs, inst)

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["raw"]
    assert randGen().randbool() == case["post_randbool"]


# --------------------------------------------------------------- uniform_randomize_disc


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["uniform_randomize_disc"],
    ids=lambda c: f"multys{c['multys']}_seed{c['seed']}",
)
def test_uniform_randomize_disc_matches_cpp(case):
    fs = _disc_fs(case["multys"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])

    uniform_randomize_disc(fs, inst)

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["raw"]
    assert randGen().randbool() == case["post_randbool"]


# --------------------------------------------------------- occam_randomize_contin (single)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["occam_randomize_contin_single"],
    ids=lambda c: f"depth{c['depth']}_seed{c['seed']}",
)
def test_occam_randomize_contin_single_matches_cpp(case):
    fs = _contin_single_fs(case["depth"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])

    occam_randomize_contin(fs, inst, 0)

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["raw"]
    assert randGen().randbool() == case["post_randbool"]


# ------------------------------------------------------------ occam_randomize_contin (all)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["occam_randomize_contin_all"],
    ids=lambda c: f"depths{c['depths']}_seed{c['seed']}",
)
def test_occam_randomize_contin_all_matches_cpp(case):
    fs = _contin_all_fs(case["depths"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])

    occam_randomize_contin(fs, inst)

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["raw"]
    assert randGen().randbool() == case["post_randbool"]


# ------------------------------------------------------------------------------ randomize


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["randomize"],
    ids=lambda c: (
        f"contin{c['contin_depths']}_disc{c['disc_multys']}_bits{c['n_bits']}_seed{c['seed']}"
    ),
)
def test_randomize_matches_cpp(case):
    fs = _mixed_fs(case["contin_depths"], case["disc_multys"], case["n_bits"])
    inst = instance(fs.packed_width())
    randGen().seed(case["seed"])

    randomize(fs, inst)

    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
    assert got == case["raw"]
    assert randGen().randbool() == case["post_randbool"]


# --------------------------------------------------------------------------- guard/mutation


def test_occam_randomize_contin_all_matches_manual_forward_calls():
    """Direct (non-oracle, implementation-independent) check that the
    all-fields overload is exactly equivalent to driving the single-field
    overload once per spec in `fs.contin()` order -- not reversed, not by
    raw position. `test_occam_randomize_contin_all_matches_cpp` above
    already pins this against the real C++ oracle; this test additionally
    proves the manual-forward-calls reference itself actually depends on
    order (mutation: swap the two manual calls below to reversed and this
    assertion goes red, since forward != reversed for these depths).
    """
    fs = _contin_all_fs([2, 6])
    inst = instance(fs.packed_width())
    randGen().seed(7)
    occam_randomize_contin(fs, inst)
    got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]

    fs2 = _contin_all_fs([2, 6])
    inst2 = instance(fs2.packed_width())
    randGen().seed(7)
    occam_randomize_contin(fs2, inst2, 0)
    occam_randomize_contin(fs2, inst2, 1)
    manual_forward = [fs2.get_raw(inst2, i) for i in range(fs2.raw_size())]

    fs3 = _contin_all_fs([2, 6])
    inst3 = instance(fs3.packed_width())
    randGen().seed(7)
    occam_randomize_contin(fs3, inst3, 1)
    occam_randomize_contin(fs3, inst3, 0)
    manual_reversed = [fs3.get_raw(inst3, i) for i in range(fs3.raw_size())]

    assert got == manual_forward
    assert manual_forward != manual_reversed, (
        "forward and reversed manual calls produced the same result -- "
        "this case cannot discriminate draw order; pick different depths"
    )


def test_randomize_runs_contin_before_disc_before_bit():
    """Mutation proof for `randomize`'s call order: reordering to
    bit -> disc -> contin (a plausible transcription slip) desyncs the
    draw stream from C++'s contin -> disc -> bit order and changes the
    resulting raw array for a mixed field_set, for at least one seed in
    the oracle fixture that actually exercises all three domains.
    """
    mixed_cases = [
        c
        for c in _GOLDEN["randomize"]
        if c["contin_depths"] and c["disc_multys"] and c["n_bits"]
    ]
    assert mixed_cases, "fixture has no case exercising contin+disc+bit together"

    def _randomize_wrong_order(fs, inst):
        # The mutant: bit, then disc, then contin -- reversed from C++.
        uniform_randomize_bit(fs, inst)
        uniform_randomize_disc(fs, inst)
        occam_randomize_contin(fs, inst)

    mismatches = 0
    for case in mixed_cases:
        fs = _mixed_fs(case["contin_depths"], case["disc_multys"], case["n_bits"])
        inst = instance(fs.packed_width())
        randGen().seed(case["seed"])
        _randomize_wrong_order(fs, inst)
        got = [fs.get_raw(inst, i) for i in range(fs.raw_size())]
        if got != case["raw"]:
            mismatches += 1
    assert mismatches > 0, "reversed call order still matched C++ for every case -- weak fixture"


def test_uniform_randomize_bit_empty_field_set_draws_nothing():
    fs = _bit_fs(0)
    inst = instance(fs.packed_width())
    randGen().seed(123)
    marker_before = randGen().randbool()
    randGen().seed(123)
    uniform_randomize_bit(fs, inst)
    assert randGen().randbool() == marker_before


def test_occam_randomize_term_has_no_equivalent_and_is_not_exported():
    """Documents the scope decision directly: there is no
    `occam_randomize_term` in this module (see the module docstring) --
    guards against a future re-import silently reappearing without the
    same review this port gave it.
    """
    import pymoses.moses.eda.initialization as mod

    assert not hasattr(mod, "occam_randomize_term")
    assert "occam_randomize_term" not in mod.__all__
