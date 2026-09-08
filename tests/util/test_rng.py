import json  # noqa: I001
import pytest
from pymetta.util.rng import MT19937RandGen, randGen

@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "rng.json").read_text())["draws"]

# The oracle tests below derive their draw count from the fixture itself
# (`[... for _ in golden[seed][key]]`), so a truncated or empty rng.json would
# silently reduce every one of them to `[] == []` and pass. This test pins
# down the fixture's shape independently of what any test happens to consume,
# so a corrupted golden file fails loudly here instead of nowhere.
_ALL_SEEDS = ["0", "1", "7", "42", "12345", "4294967295"]
_EXPECTED_DRAWS = 2000
_SERIES_SEEDS = {
    "raw": _ALL_SEEDS,
    "randint": _ALL_SEEDS,
    "randint_7": ["0", "42"],
    "randbool": ["0", "42"],
    "randdouble": ["0", "42"],
}

def test_golden_fixture_has_expected_shape(golden):
    assert sorted(golden) == sorted(_ALL_SEEDS)
    for series, seeds in _SERIES_SEEDS.items():
        for seed in seeds:
            assert series in golden[seed], f"seed {seed!r} is missing series {series!r}"
            n = len(golden[seed][series])
            assert n == _EXPECTED_DRAWS, (
                f"seed {seed!r} series {series!r} has {n} draws, expected {_EXPECTED_DRAWS}"
            )

@pytest.mark.oracle
@pytest.mark.parametrize("seed", ["0", "1", "7", "42", "12345", "4294967295"])
def test_raw_stream_matches_cpp(golden, seed):
    g = MT19937RandGen(int(seed))
    assert [g.genrand_int32() for _ in golden[seed]["raw"]] == golden[seed]["raw"]

@pytest.mark.oracle
@pytest.mark.parametrize("seed", ["0", "1", "7", "42", "12345", "4294967295"])
def test_randint_matches_cpp(golden, seed):
    g = MT19937RandGen(int(seed))
    assert [g.randint() for _ in golden[seed]["randint"]] == golden[seed]["randint"]

@pytest.mark.oracle
@pytest.mark.parametrize("seed", ["0", "42"])
def test_randint_n_matches_cpp(golden, seed):
    g = MT19937RandGen(int(seed))
    assert [g.randint_n(7) for _ in golden[seed]["randint_7"]] == golden[seed]["randint_7"]

@pytest.mark.oracle
@pytest.mark.parametrize("seed", ["0", "42"])
def test_randbool_matches_cpp(golden, seed):
    g = MT19937RandGen(int(seed))
    assert [int(g.randbool()) for _ in golden[seed]["randbool"]] == golden[seed]["randbool"]

@pytest.mark.oracle
@pytest.mark.parametrize("seed", ["0", "42"])
def test_randdouble_matches_cpp(golden, seed):
    g = MT19937RandGen(int(seed))
    assert [g.randdouble() for _ in golden[seed]["randdouble"]] == golden[seed]["randdouble"]

def test_randint_n_zero_returns_zero():
    # C++: n == 0 short-circuits before the modulo, avoiding division by zero
    assert MT19937RandGen(0).randint_n(0) == 0

def test_randint_n_matches_cpp_for_negative_n():
    """S10 branch review, [Important 1]: `randint_n`'s modulo used to take
    the sign of Python's DIVISOR (`self.randint() % n`), while C's
    truncating `%` takes the sign of the (here always non-negative)
    DIVIDEND -- silently wrong for any negative `n`, unreached until
    `merging.py`'s `resize_metapop` could pass one (its own `size_t`
    underflow, narrowed to a signed 32-bit `int` exactly as C++'s `int`
    parameter does).

    Not part of the `rng.json` golden fixture (which has no negative-`n`
    series): pinned directly against a compiled probe on the real cogutil
    `MT19937RandGen`, `-O2 -std=c++17`, linked against `libcogutil.so`
    (the same recipe `rng_probe.cc` documents) --

        MT19937RandGen g(1u);
        g.randint(-25);  // == 22
        g.randint(-1);   // == 0
        g.randint(-29);  // == 6

    ONE generator, three sequential calls -- each `randint(n)` consumes
    exactly one draw regardless of `n`, so this is the same
    RNG-stream-consumption contract `randint_n`'s own docstring already
    relies on. Before the fix (`self.randint() % n`, sign of the divisor),
    this sequence returned -3/0/-23 instead.
    """
    g = MT19937RandGen(1)
    assert [g.randint_n(-25), g.randint_n(-1), g.randint_n(-29)] == [22, 0, 6]

def test_randint_is_raw_shifted_right_by_one():
    a, b = MT19937RandGen(0), MT19937RandGen(0)
    assert [b.randint() for _ in range(64)] == [a.genrand_int32() >> 1 for _ in range(64)]

def test_unused_methods_raise_rather_than_guess():
    g = MT19937RandGen(0)
    for name in ("randfloat", "randdouble_one_excluded", "rand_discrete"):
        with pytest.raises(NotImplementedError):
            getattr(g, name)() if name != "rand_discrete" else g.rand_discrete([1.0, 1.0])

def test_randgen_singleton_is_stable_and_reseedable():
    assert randGen() is randGen()
    randGen().seed(42)
    first = [randGen().randint() for _ in range(4)]
    randGen().seed(42)
    assert [randGen().randint() for _ in range(4)] == first
