"""C++: moses/moses/moses/moses_params.h

Pins every documented default of `moses_parameters`/`moses_statistics` --
these are the outer loop's ENTIRE termination surface (see
`moses_params.py`'s own module docstring), so an unpinned default here is
exactly the class of gap S9's review found in every `hc_parameters`
default. Each default below was verified by mutation: change it in
`moses_params.py`, rerun this file, confirm the corresponding assertion
(and only that one) fails, then revert -- see the task report for the
full mutation list and results.
"""

import pytest

from pymetta.moses.moses.moses_params import _INT32_MAX, moses_parameters, moses_statistics
from pymetta.moses.moses.types import _to_score_t

# 1/3.5 is not exactly representable in binary32 -- the same fixture
# `test_types.py` uses to prove a `score_t` narrowing site is not a
# passthrough. See that module's own comment for why the assertion below
# guards the fixture itself, not just the code under test.
_UNNARROWED = 1.0 / 3.5
assert _UNNARROWED != _to_score_t(_UNNARROWED), (
    "fixture value must actually require float32 rounding, or these tests "
    "prove nothing"
)


# --------------------------------------------------------------------------
# moses_parameters
# --------------------------------------------------------------------------


def test_moses_parameters_pins_every_documented_default():
    """MUTATION: change any default below in `moses_parameters.__init__`
    (e.g. `max_evals: int = 10000` -> `10001`, or `self.mpi = False` ->
    `True`) -- verified each one, independently, makes exactly its own
    assertion fail while every other assertion here still passes.
    """
    pa = moses_parameters()
    assert pa.local is True
    assert pa.mpi is False
    assert pa.force_feed is False
    assert pa.jobs == {}
    assert pa.vm is None
    assert pa.max_evals == 10000
    assert pa.max_gens == -1
    assert pa.max_score == 0
    assert pa.max_time == _INT32_MAX
    assert pa.max_time == 2147483647
    assert pa.max_cnd_output == -1


def test_moses_parameters_constructor_overrides_its_seven_real_parameters():
    """`vm`, `jobs`, `local`, `max_evals`, `max_gens`, `max_score` and
    `max_cnd_output` are real C++ constructor parameters
    (moses_params.h:42-49) -- verify each one actually reaches its field
    when the caller supplies a non-default value, in C++'s own parameter
    order.
    """
    sentinel_vm = object()
    pa = moses_parameters(
        vm=sentinel_vm,
        jobs={"host1": 4},
        local=False,
        max_evals=5,
        max_gens=3,
        max_score=0.75,
        max_cnd_output=2,
    )
    assert pa.vm is sentinel_vm
    assert pa.jobs == {"host1": 4}
    assert pa.local is False
    assert pa.max_evals == 5
    assert pa.max_gens == 3
    assert pa.max_score == _to_score_t(0.75)
    assert pa.max_cnd_output == 2


def test_moses_parameters_mpi_force_feed_max_time_ignore_constructor_args():
    """`mpi`, `force_feed` and `max_time` are NOT constructor parameters in
    C++ (moses_params.h:42-49 lists only seven) -- they are fixed
    member-initializer values (`mpi(false), force_feed(false), ...
    max_time(INT_MAX)`, moses_params.h:50-52). `moses_parameters.__init__`
    does not even accept them as keyword arguments, so passing one is a
    `TypeError`, which itself proves they cannot be overridden through the
    constructor the way the seven real parameters above can.
    """
    with pytest.raises(TypeError):
        moses_parameters(mpi=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        moses_parameters(force_feed=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        moses_parameters(max_time=5)  # type: ignore[call-arg]


def test_moses_parameters_max_score_narrows_to_score_t():
    """`max_score` is `score_t` (`float`, binary32) in C++
    (moses_params.h:70) -- pins the narrowing site in
    `moses_parameters.__init__`.
    """
    pa = moses_parameters(max_score=_UNNARROWED)
    assert pa.max_score == _to_score_t(_UNNARROWED)
    assert pa.max_score != _UNNARROWED


def test_moses_parameters_jobs_default_is_a_fresh_empty_dict_each_time():
    """C++'s own default argument, `const jobs_t& _jobs = jobs_t()`
    (moses_params.h:44), constructs a FRESH empty map at every call site
    that omits `_jobs` -- mirrored here by never sharing one mutable
    default object across instances (the same trap
    `metapop_parameters.__init__`'s own `diversity` parameter avoids).
    """
    pa1 = moses_parameters()
    pa2 = moses_parameters()
    pa1.jobs["host1"] = 2
    assert pa2.jobs == {}
    assert pa1.jobs is not pa2.jobs


def test_moses_parameters_jobs_is_copied_not_aliased():
    """C++ copies `_jobs` BY VALUE into the `jobs` member
    (`jobs(_jobs)`, moses_params.h:50, a `jobs_t` member initialized from a
    `const jobs_t&` parameter -- a real map copy, not a reference member).
    Mutating the caller's dict after construction must not be visible
    through `pa.jobs`, and mutating `pa.jobs` must not leak back to the
    caller's own dict.
    """
    caller_jobs = {"host1": 1}
    pa = moses_parameters(jobs=caller_jobs)
    caller_jobs["host2"] = 9
    assert pa.jobs == {"host1": 1}

    pa.jobs["host3"] = 3
    assert caller_jobs == {"host1": 1, "host2": 9}


# --------------------------------------------------------------------------
# moses_statistics
# --------------------------------------------------------------------------


def test_moses_statistics_pins_every_documented_default():
    """MUTATION: change any default below in `moses_statistics.__init__`
    (e.g. `self.n_evals = 0` -> `1`, or `self.elapsed_secs = 0.0` -> `1.0`)
    -- verified each one, independently, makes exactly its own assertion
    fail while the other two still pass.
    """
    stats = moses_statistics()
    assert stats.n_evals == 0
    assert stats.n_expansions == 0
    assert stats.elapsed_secs == 0.0
    assert isinstance(stats.elapsed_secs, float)


def test_moses_statistics_instances_do_not_share_counters():
    """Each `moses_statistics()` call must produce independent counters --
    a regression guard against a class-level (shared) attribute mistake,
    which plain ints assigned in `__init__` do not have but is cheap to
    pin directly.
    """
    s1 = moses_statistics()
    s2 = moses_statistics()
    s1.n_evals += 5
    s1.n_expansions += 1
    s1.elapsed_secs += 2.5
    assert s2.n_evals == 0
    assert s2.n_expansions == 0
    assert s2.elapsed_secs == 0.0
