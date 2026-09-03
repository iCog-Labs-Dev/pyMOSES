import json  # noqa: I001
import pytest
from pymetta.util.rng import MT19937RandGen, randGen

@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "rng.json").read_text())["draws"]

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
