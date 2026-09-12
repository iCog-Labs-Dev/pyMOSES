"""Regression tests for the S9 fix round's CRITICAL 1 finding: C++'s `size_t`
underflow in `hill_climbing.n_new_instances` (hill-climbing.cc:520-546) was
not reproduced by the initial port, which let an unbounded Python `int`
subtraction go negative instead of wrapping the way `size_t` does.

Also covers the IMPORTANT 4 finding (free-vs-total RAM, and
`sizeof(instance)`), both dormant behind `hc_parameters.resize_to_fit_ram`
(default `False`).

These are deliberately kept separate from `test_hill_climbing.py` (owned by
another agent in this fix round) rather than added there.
"""

import pymoses.moses.moses.optimization.hill_climbing as hill_climbing_mod
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.field_set import disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_scorer import distance_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen


def _hc() -> hill_climbing:
    return hill_climbing(optim_parameters(), hc_parameters(cross=False))


# --------------------------------------------------------------------- CRITICAL 1


def test_n_new_instances_reproduces_measured_cpp_size_t_underflow():
    """Pins the review's own measured C++ value, not merely "not negative":
    a standalone probe reproducing `n_new_instances`'s exact `size_t`/
    `unsigned` arithmetic (hill-climbing.cc:520-546, hc_parameters()
    defaults: fraction_of_nn=2.0, max_nn_evals=20000) gives
    `n_new_instances(distance=1, max_evals=1, current_number_of_evals=3,
    total_number_of_neighbors=6) == 100` -- the wrapped (huge) `nleft`
    that the subsequent `min(nleft, MINIMUM_DEME_SIZE)` clamps down to
    exactly `_MINIMUM_DEME_SIZE`. Before the fix, Python's unbounded `int`
    subtraction gave `nleft = -2`, and `min(-2, 100)` propagated that
    negative value straight through as the method's return value.
    """
    hc = _hc()
    assert hc.n_new_instances(1, 1, 3, 6) == 100


def test_n_new_instances_underflow_is_reachable_with_max_evals_zero():
    """A second measured underflow shape from the review: `max_evals=0`
    with `current_number_of_evals=1` (i.e. the budget is already exhausted
    when this call happens) -- reachable through `__call__` itself, see
    `test_call_with_zero_budget_does_not_truncate_the_deme` below. Also
    wraps to a huge `nleft`, also clamped to `_MINIMUM_DEME_SIZE`.
    """
    hc = _hc()
    assert hc.n_new_instances(1, 0, 1, 6) == 100


def test_n_new_instances_never_returns_negative_when_max_evals_is_exceeded():
    """Broader property alongside the two pinned measurements above: for a
    spread of already-over-budget calls (`current_number_of_evals >
    max_evals`), the wrapped `nleft` must never let a negative value reach
    the return value -- `sample_new_instances`'s caller feeds this
    straight into `deme.resize(current + number_of_new_instances)`, and a
    negative count there truncates already-scored instances.
    """
    hc = _hc()
    for max_evals, current in ((1, 3), (0, 1), (5, 1000), (0, 0)):
        result = hc.n_new_instances(1, max_evals, current, 6)
        assert result >= 0


def test_call_with_zero_budget_does_not_truncate_the_deme():
    """The reachable-through-`__call__` scenario the review measured
    directly: `hc(deme, distance_based_scorer(fs, target), max_evals=0,
    86400)` on a 3-knob all-disc `field_set`. The loop's own `if
    (max_evals <= current_number_of_evals) break;` guard
    (hill-climbing.cc:413-417) is skipped on the very first "distance
    0 -> 1" transition by the earlier `if (number_of_new_instances == 1
    and distance == 1) continue;` (hill-climbing.cc:408-409), so
    `n_new_instances` IS called again with the budget already exhausted.
    Before the fix, the resulting negative `number_of_new_instances` made
    `sample_new_instances` call `deme.resize(current + negative)`,
    truncating the deme back down to empty: `len(deme) == 0` and
    `deme.n_evals == 0`, even though at least one instance (the initial
    center) was already scored. After the fix, the deme retains every
    instance actually scored.
    """
    randGen().seed(1)
    fs = field_set(disc_spec(2), 3)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    hc = _hc()

    hc(deme, scorer, 0, 86400)

    assert len(deme) > 0
    assert deme.n_evals == len(deme)
    assert hc.over_budget is True


# --------------------------------------------------------------------- IMPORTANT 4


def test_instance_bytes_matches_cpp_sizeof_instance_plus_packed_width():
    """C++: hill-climbing.cc:72 `_instance_bytes = sizeof(instance) +
    sizeof(packed_t) * fields.packed_width();`. `sizeof(instance)` --
    `instance` is `std::vector<unsigned long>` -- is 24 bytes on the
    64-bit ABI this port pins (three 8-byte pointers), NOT 8. Before the
    fix, this was computed as `8 * (1 + packed_width())` == `8 + 8w`.
    """
    fs = field_set(disc_spec(2), 5)
    scorer = distance_based_scorer(fs, instance(fs.packed_width()))
    deme = instance_set(fs)
    hc = _hc()

    hc(deme, scorer, 1, 86400)

    assert hc._instance_bytes == 24 + 8 * fs.packed_width()


def test_resize_to_fit_ram_caps_against_free_ram_not_total_ram(monkeypatch):
    """C++: hill-climbing.cc:577 `uint64_t free_ram = getFreeRAM();` (NOT
    `getTotalRAM()`). Before the fix, this port called
    `_get_total_ram_bytes()` here too, so `free_ram == total_RAM_bytes`
    always -- the `min(ACCEPTABLE_RAM_FRACTION * total, MAX_RAM_LIMIT *
    free)` cap could then never observe actual RAM pressure, always
    preferring the `ACCEPTABLE_RAM_FRACTION` arm. Mocks `total` and `free`
    to distinguishable values and drives `n_new_instances` into the
    RAM-pressure branch directly (bypassing the OS-dependent real RAM
    query), so this test's result depends only on which function is
    actually called, not on the host machine's real RAM.
    """
    hc = _hc()
    hc.hc_params.resize_to_fit_ram = True
    hc._total_RAM_bytes = 1000
    hc._instance_bytes = 1

    monkeypatch.setattr(hill_climbing_mod, "_get_total_ram_bytes", lambda: 1000)
    monkeypatch.setattr(hill_climbing_mod, "_get_free_ram_bytes", lambda: 100)

    # distance=1, fraction_of_nn=2.0 (default) -> 1000 * 2 = 2000 new
    # instances requested; instance_bytes=1 makes new_usage == 2000,
    # which exceeds ACCEPTABLE_RAM_FRACTION(0.5) * total(1000) == 500,
    # tripping the RAM-pressure branch.
    result = hc.n_new_instances(distance=1, max_evals=1_000_000, current_number_of_evals=0,
                                 total_number_of_neighbors=1000)

    # Correct (uses free_ram=100): cap = min(0.5*1000, 0.9*100) = 90.
    # Buggy (uses total_RAM_bytes=1000 as "free" too): cap = min(500, 900)
    # = 500 -- distinguishable from the correct answer.
    assert result == 90
