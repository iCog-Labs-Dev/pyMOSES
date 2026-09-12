# pyMoses S0–S6 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pyMoses foundation and first runnable slice — a Python port of classic OpenCog MOSES that reproduces the C++ implementation's RNG stream bit-for-bit and produces identical behavioral, complexity, and composite scores for a corpus of (CSV, combo tree) pairs.

**Architecture:** Mirror classic's C++ source tree file-for-file under `pymoses/`. Every ported symbol carries a `C++: <path>:<line> <symbol>` breadcrumb that CI validates. Correctness is established by differential testing against small C++ probe binaries linked to the real MOSES libraries, with golden outputs committed as fixtures so the suite runs with no C++ toolchain present.

**Tech Stack:** Python 3.14 (stdlib only for the runtime package), pytest 9.1, ruff for lint. C++ probes built with the local g++ 16.2.1 against `$HOME/opt/moses-legacy`.

## Global Constraints

- **Runtime dependencies: Python standard library only.** No numpy, no hyperon, nothing. Dev dependencies are `pytest` and `ruff` only.
- **Python 3.14.** `match` statements are used for `vertex` dispatch.
- **C++ identifiers are preserved verbatim, including class names.** `field_set` not `FieldSet`, `combo_tree` not `ComboTree`, `build_logical` not `buildLogical`. ruff naming rules N801/N802/N803/N806 are disabled repo-wide for this reason.
- **Every ported symbol carries a breadcrumb** as the first line of its docstring: `C++: <path relative to a source root>:<line> <C++ qualified symbol>`. There are **two** source roots and `check_alignment.py` tries each: `$CLASSIC` for paths beginning `moses/`, and `$COGUTIL` for paths beginning `opencog/` (`pymoses/util/rng.py` and `tree.py` port CogUtil, not Moses). The prefixes are disjoint, so there is no ambiguity.
- **Every deliberate deviation carries a marker:** `# DIVERGENCE(C++ <path>:<line>): <reason and observable impact>`. A divergence is legitimate only when it has no observable effect on results. "It was easier" is not a reason — raise it for a decision instead.
- **Classic repo root** (referred to as `$CLASSIC`): `/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses`
- **CogUtil root** (`$COGUTIL`): `/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/CogUtil`
- **Every commit is authored `Yab <yabuman46@gmail.com>` and nobody else.** No `Co-Authored-By`, no `Generated with`, no session trailers, ever.
- **Never claim a step passes without running it and reading the real output.**

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, pytest config, ruff config with naming exemptions |
| `tools/check_alignment.py` | Parse breadcrumbs + divergences, validate against `$CLASSIC`, emit `docs/ALIGNMENT.md` |
| `tools/cpp_probe/CMakeLists.txt` | Build the C++ oracle probes against `$HOME/opt/moses-legacy` |
| `tools/cpp_probe/*.cc` | One probe per slice; each dumps internal state as JSON to stdout |
| `tools/capture_golden.py` | Run probes, write `tests/fixtures/*.json` |
| `pymoses/util/rng.py` | MT19937 + the `RandGen` method surface |
| `pymoses/util/tree.py` | `tree<T>` — n-ary linked tree with the splice operations MOSES uses |
| `pymoses/comboreduct/combo/vertex.py` | `builtin` enum, `argument`, `vertex` union, `combo_tree` alias |
| `pymoses/comboreduct/combo/iostream_combo.py` | Combo tree parse and print |
| `pymoses/comboreduct/interpreter/interpreter.py` | `boolean_interpreter`, `interpreter_visitor` |
| `pymoses/comboreduct/table/table.py` | `ITable`, `OTable`, `Table`, `CTable` |
| `pymoses/comboreduct/table/table_io.py` | `loadTable` — CSV parsing and type inference |
| `pymoses/moses/moses/types.py` | `composite_score`, `behavioral_score`, `scored_combo_tree` |
| `pymoses/moses/moses/complexity.py` | `tree_complexity` |
| `pymoses/moses/scoring/scoring_base.py` | `bscore_base` |
| `pymoses/moses/scoring/bscores.py` | `ctruth_table_bscore` |
| `pymoses/moses/scoring/behave_cscore.py` | `behave_cscore` |

---

### Task 0: Scaffold, conventions, and the alignment checker

**Files:**
- Create: `pyproject.toml`, `README.md`, `pymoses/__init__.py`, `tools/check_alignment.py`, `tests/test_scaffold.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `tests/conftest.py` exports fixtures `classic_root` (Path to `$CLASSIC`) and `fixture_dir` (Path to `tests/fixtures`). `tools/check_alignment.py` exposes `parse_breadcrumbs(path) -> list[Breadcrumb]` and `main(argv) -> int`.

- [ ] **Step 1: Create the package skeleton and pyproject**

```toml
# pyproject.toml
[project]
name = "pymoses"
version = "0.0.1"
description = "Python port of classic OpenCog MOSES, aligned to the C++ source"
requires-python = ">=3.14"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=9.0", "ruff>=0.6"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
  "oracle: differential test comparing against committed C++ golden fixtures",
  "live: requires rebuilding the C++ probes; excluded from the default run",
]

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B"]
# C++ identifiers are preserved verbatim so the Python can be diffed against the
# C++ by eye. See docs/superpowers/specs/2026-09-02-pymoses-foundation-design.md
# section 5.1. This is the single most load-bearing convention in the project.
ignore = ["N801", "N802", "N803", "N806", "N815", "N818"]
```

Create empty `pymoses/__init__.py` and package `__init__.py` files for every directory in the File Structure table.

- [ ] **Step 2: Write the failing test for the alignment checker**

```python
# tests/test_alignment_checker.py
import subprocess, sys, textwrap
from pathlib import Path
from tools.check_alignment import parse_breadcrumbs, Breadcrumb

def test_parses_a_wellformed_breadcrumb(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent('''
        def build_logical(self, it):
            """C++: moses/moses/representation/build_knobs.cc:602 build_knobs::build_logical

            Decorate a logical subtree with knobs.
            """
    '''))
    crumbs = parse_breadcrumbs(f)
    assert crumbs == [Breadcrumb(
        py_file=f, py_symbol="build_logical",
        cpp_file="moses/moses/representation/build_knobs.cc",
        cpp_line=602, cpp_symbol="build_knobs::build_logical",
    )]

def test_reports_symbol_missing_from_cpp(tmp_path, classic_root):
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: moses/comboreduct/combo/vertex.h:69 no_such_symbol_xyz"""\n')
    from tools.check_alignment import validate
    problems = validate(parse_breadcrumbs(f), classic_root)
    assert len(problems) == 1
    assert "no_such_symbol_xyz" in problems[0]

def test_accepts_symbol_within_drift_tolerance(tmp_path, classic_root):
    # builtin enum really is at vertex.h:69; cite it 10 lines off and it must still pass
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: moses/comboreduct/combo/vertex.h:79 builtin"""\n')
    from tools.check_alignment import validate
    assert validate(parse_breadcrumbs(f), classic_root) == []
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `python3 -m pytest tests/test_alignment_checker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.check_alignment'`

- [ ] **Step 4: Implement `tools/check_alignment.py`**

Requirements, precisely:
- `Breadcrumb` is a frozen dataclass with fields `py_file: Path`, `py_symbol: str`, `cpp_file: str`, `cpp_line: int`, `cpp_symbol: str`.
- `parse_breadcrumbs(path: Path) -> list[Breadcrumb]` walks the file with `ast`, and for every `FunctionDef`/`AsyncFunctionDef`/`ClassDef` whose docstring's first line matches `^C\+\+: (?P<file>\S+):(?P<line>\d+) (?P<symbol>.+)$`, yields a `Breadcrumb`. Nodes without a breadcrumb are skipped here and reported separately by `coverage()`.
- `validate(crumbs, classic_root) -> list[str]` returns a human-readable problem per breadcrumb where either the C++ file does not exist, or the **last** `::`-separated component of `cpp_symbol` does not appear as a whole word within ±25 lines of `cpp_line`. Tolerance constant `DRIFT_TOLERANCE = 25`.
- `parse_divergences(path) -> list[Divergence]` scans source lines for `# DIVERGENCE\((?P<ref>C\+\+ [^)]+)\): (?P<reason>.+)`.
- `coverage(pkg_root) -> dict[str, tuple[int,int]]` maps module path to (symbols with breadcrumbs, total public symbols).
- `main(argv)` walks `pymoses/`, runs `validate`, writes `docs/ALIGNMENT.md` (a table of every breadcrumb plus every divergence plus per-module coverage), prints problems, and returns `1` if any problem was found else `0`.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_alignment_checker.py -v`
Expected: 3 passed.

- [ ] **Step 6: Add conftest and a CI entry point**

```python
# tests/conftest.py
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASSIC = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses")
COGUTIL = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/CogUtil")

@pytest.fixture(scope="session")
def classic_root() -> Path:
    if not CLASSIC.is_dir():
        pytest.skip(f"classic MOSES checkout not found at {CLASSIC}")
    return CLASSIC

@pytest.fixture(scope="session")
def cogutil_root() -> Path:
    if not COGUTIL.is_dir():
        pytest.skip(f"CogUtil checkout not found at {COGUTIL}")
    return COGUTIL

@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
```

Create `tools/ci.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m ruff check pymoses tools tests
python3 -m pytest
python3 tools/check_alignment.py
```

- [ ] **Step 7: Run the full CI entry point**

Run: `bash tools/ci.sh`
Expected: ruff clean, pytest passes, alignment checker exits 0 and writes `docs/ALIGNMENT.md`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
build(scaffold): package skeleton, pytest/ruff config, alignment checker

tools/check_alignment.py parses the `C++: <file>:<line> <symbol>` breadcrumbs
and DIVERGENCE markers, validates each against the classic checkout with a
+/-25 line drift tolerance, and generates docs/ALIGNMENT.md.
EOF
)"
```

---

### Task 1: `pymoses/util/rng.py` — MT19937 and the RandGen surface

This task is the gate on the project's bit-identity guarantee. Do it before anything that consumes randomness.

**Files:**
- Create: `pymoses/util/rng.py`, `tools/cpp_probe/rng_probe.cc`, `tests/util/test_rng.py`, `tests/fixtures/rng.json`
- Modify: `tools/capture_golden.py`

**Interfaces:**
- Consumes: nothing
- Produces: `class MT19937RandGen` with `__init__(self, seed: int = 0)`, `seed(self, s: int) -> None`, `genrand_int32(self) -> int`, `randint(self) -> int`, `randint_n(self, n: int) -> int`, `randdouble(self) -> float`, `randbool(self) -> bool`. Module-level `randGen() -> MT19937RandGen` returning a process-global singleton seeded 0, mirroring classic's thread-local singleton.

**Reference:** `$COGUTIL/opencog/util/mt19937ar.cc:44-93`, `$COGUTIL/opencog/util/RandGen.h:49`.

Verified facts this task must reproduce (see spec section 6):
- `RandGen : public std::mt19937`, seeded via single-value `init_genrand`, **not** `init_by_array`. Python's `random` module cannot be used — its `seed(int)` uses `init_by_array` and diverges from the first draw.
- `randint()` is `std::uniform_int_distribution<int>()` over `[0, INT_MAX]`, which for this instantiation reduces to `raw >> 1` with an unreachable rejection branch — one draw consumed, deterministic on any libstdc++.
- `randint(n)` is `n == 0 ? 0 : randint() % n`.
- `randdouble()` is `operator()() / (double)max()` where `max()` is `0xFFFFFFFF`.
- `randbool()` is `randint() % 2 == 0`.
- `randfloat()`, `randdouble_one_excluded()`, `rand_discrete()` must raise `NotImplementedError` citing spec section 6.2 — they have zero call sites in the MOSES core, and a plausible-but-wrong stub is worse than a crash.

- [ ] **Step 1: Write the C++ probe**

```cpp
// tools/cpp_probe/rng_probe.cc
// Dumps cogutil RandGen draw sequences as JSON for differential testing.
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const std::vector<unsigned> seeds = {0u, 1u, 7u, 42u, 12345u, 4294967295u};
    const int N = 2000;
    printf("{\n  \"draws\": {\n");
    bool first_seed = true;
    for (unsigned s : seeds) {
        if (!first_seed) printf(",\n");
        first_seed = false;
        printf("    \"%u\": {\n", s);

        std::mt19937 g_raw(s);
        printf("      \"raw\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%u" : "%u", g_raw());
        printf("],\n");

        std::mt19937 g_ri(s);
        printf("      \"randint\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", d(g_ri));
        }
        printf("],\n");

        std::mt19937 g_rn(s);
        printf("      \"randint_7\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", d(g_rn) % 7);
        }
        printf("],\n");

        std::mt19937 g_rb(s);
        printf("      \"randbool\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", (d(g_rb) % 2 == 0) ? 1 : 0);
        }
        printf("],\n");

        std::mt19937 g_rd(s);
        printf("      \"randdouble\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%.17g" : "%.17g", g_rd() / (double)g_rd.max());
        printf("]\n    }");
    }
    printf("\n  }\n}\n");
    return 0;
}
```

- [ ] **Step 2: Build the probe and capture the golden fixture**

Run:
```bash
mkdir -p tools/cpp_probe/build
g++ -O2 -std=c++17 -o tools/cpp_probe/build/rng_probe tools/cpp_probe/rng_probe.cc
tools/cpp_probe/build/rng_probe > tests/fixtures/rng.json
python3 -c "import json;d=json.load(open('tests/fixtures/rng.json'));print(sorted(d['draws']), len(d['draws']['0']['raw']))"
```
Expected: six seeds listed, `2000` draws each. If the JSON fails to parse, fix the probe before continuing.

- [ ] **Step 3: Write the failing test**

```python
# tests/util/test_rng.py
import json
import pytest
from pymoses.util.rng import MT19937RandGen, randGen

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
```

- [ ] **Step 4: Run to confirm it fails**

Run: `python3 -m pytest tests/util/test_rng.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymoses.util.rng'`

- [ ] **Step 5: Implement `pymoses/util/rng.py`**

```python
"""MT19937 random generator, bit-compatible with cogutil's RandGen.

C++: opencog/util/mt19937ar.cc:44 MT19937RandGen
"""

_N, _M = 624, 397
_MATRIX_A, _UPPER_MASK, _LOWER_MASK = 0x9908B0DF, 0x80000000, 0x7FFFFFFF
_UINT32_MAX = 0xFFFFFFFF


class MT19937RandGen:
    """C++: opencog/util/RandGen.h:49 RandGen

    cogutil's RandGen *is* std::mt19937 with convenience methods layered on.
    Python's `random` module cannot be substituted: random.seed(int) routes
    through init_by_array, while std::mt19937::seed(unsigned) uses
    init_genrand, and the two streams diverge from the first draw.
    """

    def __init__(self, seed: int = 0) -> None:
        self._mt = [0] * _N
        self._mti = _N + 1
        self.seed(seed)

    def seed(self, s: int) -> None:
        """C++: opencog/util/RandGen.h:49 std::mt19937::seed (init_genrand)"""
        mt = self._mt
        mt[0] = s & _UINT32_MAX
        for i in range(1, _N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & _UINT32_MAX
        self._mti = _N

    def genrand_int32(self) -> int:
        """C++: opencog/util/RandGen.h:49 std::mt19937::operator()"""
        mt = self._mt
        if self._mti >= _N:
            for i in range(_N):
                y = (mt[i] & _UPPER_MASK) | (mt[(i + 1) % _N] & _LOWER_MASK)
                mt[i] = mt[(i + _M) % _N] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & _UINT32_MAX

    def randint(self) -> int:
        """C++: opencog/util/mt19937ar.cc:44 MT19937RandGen::randint

        std::uniform_int_distribution<int> over [0, INT_MAX] against a 2^32
        generator: scaling is exactly 2 and the rejection branch is
        unreachable, so this is `raw >> 1` on any libstdc++, consuming one
        draw. Verified against a C++ probe over 12000 draws / 6 seeds.
        """
        return self.genrand_int32() >> 1

    def randint_n(self, n: int) -> int:
        """C++: opencog/util/mt19937ar.cc:66 MT19937RandGen::randint(int n)"""
        return 0 if n == 0 else self.randint() % n

    def randdouble(self) -> float:
        """C++: opencog/util/mt19937ar.cc:55 MT19937RandGen::randdouble"""
        return self.genrand_int32() / float(_UINT32_MAX)

    def randbool(self) -> bool:
        """C++: opencog/util/mt19937ar.cc:78 MT19937RandGen::randbool"""
        return self.randint() % 2 == 0

    def randfloat(self) -> float:
        """C++: opencog/util/mt19937ar.cc:50 MT19937RandGen::randfloat"""
        raise NotImplementedError(
            "randfloat() returns a C float32 and is only reachable via the combo "
            "`rand` builtin (P2). Reproducing it needs struct.pack('f', ...). "
            "See spec section 6.2."
        )

    def randdouble_one_excluded(self) -> float:
        """C++: opencog/util/mt19937ar.cc:60 MT19937RandGen::randdouble_one_excluded"""
        raise NotImplementedError(
            "Zero call sites in the MOSES core, and std::uniform_real_distribution "
            "is implementation-defined. See spec section 6.2."
        )

    def rand_discrete(self, weights) -> int:
        """C++: opencog/util/mt19937ar.cc:84 MT19937RandGen::rand_discrete"""
        raise NotImplementedError(
            "Zero call sites in the MOSES core, and std::discrete_distribution "
            "is implementation-defined. See spec section 6.2."
        )


_instance: MT19937RandGen | None = None


def randGen() -> MT19937RandGen:
    """C++: opencog/util/mt19937ar.cc:90 opencog::randGen

    DIVERGENCE(C++ opencog/util/mt19937ar.cc:90): classic's singleton is
    thread_local; ours is process-global. The MOSES core paths in scope are
    single-threaded, and a process-global singleton is what makes a seeded run
    reproducible at all in Python.
    """
    global _instance
    if _instance is None:
        _instance = MT19937RandGen(0)
    return _instance
```

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/util/test_rng.py -v`
Expected: all pass. **If any oracle test fails, stop and escalate** — the project's bit-identity guarantee is void and spec section 6 must be rewritten before proceeding.

- [ ] **Step 7: Run the alignment checker and full CI**

Run: `bash tools/ci.sh`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
port(util/rng): MT19937 bit-compatible with cogutil RandGen

Mirrors opencog/util/mt19937ar.cc:44-93 and RandGen.h:49. randint() is
raw >> 1 (uniform_int_distribution over [0,INT_MAX] against a 2^32
generator: scaling 2, rejection branch unreachable). randfloat,
randdouble_one_excluded and rand_discrete raise NotImplementedError --
zero call sites in the core, and both distributions are
implementation-defined.

Verified against tools/cpp_probe/rng_probe.cc over 2000 draws x 6 seeds
for raw, randint, randint(7), randbool and randdouble.
EOF
)"
```

---

### Task 2: `pymoses/util/tree.py` — the `tree<T>` container

**Files:**
- Create: `pymoses/util/tree.py`, `tools/cpp_probe/tree_probe.cc`, `tests/util/test_tree.py`, `tests/fixtures/tree.json`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class tree_node[T]` with attributes `data: T`, `parent`, `first_child`, `last_child`, `prev_sibling`, `next_sibling`.
  - `class tree[T]` with: `__init__(self, root_data=None)`, `begin()`, `end()`, `__iter__` (pre-order), `size()`, `empty()`, `is_valid(node)`, `child(node, i)`, `children(node)`, `number_of_children(node)`, `number_of_siblings(node)`, `next_sibling(node)`, `previous_sibling(node)`, `append_child(node, data_or_subtree)`, `prepend_child(node, data)`, `append_children(node, iterable)`, `insert(node, data)`, `insert_after(node, data)`, `insert_subtree(node, subtree)`, `insert_subtree_after(node, subtree)`, `insert_above(node, data)`, `erase(node)`, `erase_children(node)`, `replace(node, data_or_subtree)`, `flatten(node)`, `wrap(node, data)`, `reparent(new_parent, from_node)`, `move_after(target, source)`, `move_before(target, source)`, `move_ontop(target, source)`, `equal_subtree(a, b)`, `max_depth(node=None)`, `sort(node, key)`, `count(node)`, `begin_leaf()`, `end_leaf()`, `begin_post()`, `end_post()`.
  - Iteration yields `tree_node` objects, which are usable directly as dict keys (Python identity replaces C++'s `obj_ptr_cmp<It>` at `$COGUTIL/opencog/util/hashing.h:81`).

**Reference:** `$COGUTIL/opencog/util/tree.h` (3420 lines, Kasper Peeters `tree.hh` v2.3). The API surface above is scoped by measured call-site frequency in classic — do **not** port the whole header.

**Critical hazard to preserve:** classic's `build_knobs` mutates the exemplar while iterating it, and sibling loops can observe siblings replaced by `null_vertex` sentinels mid-iteration (classic commit `98ed6fe0f` fixed one such premature `break`). A Python port over live object references has exactly the same hazard. Do not "fix" it by snapshotting children lists — the semantics must match, including the hazard.

- [ ] **Step 1: Write the C++ probe**

`tools/cpp_probe/tree_probe.cc` builds `tree<std::string>` fixtures and, after each splice operation, dumps the whole tree as a nested JSON s-expression. It must cover, each as a named case: `append_child`, `prepend_child`, `insert`, `insert_after`, `insert_above`, `erase`, `erase_children`, `replace`, `flatten`, `wrap`, `reparent`, `move_after`, `move_before`, `move_ontop`, `sort`, and pre-order / post-order / leaf traversal orders. Include a case that mutates a node mid-sibling-iteration and records what the iterator subsequently sees.

Build with:
```bash
g++ -O2 -std=c++17 -I"$COGUTIL" -o tools/cpp_probe/build/tree_probe tools/cpp_probe/tree_probe.cc
```

- [ ] **Step 2: Capture the fixture**

Run: `tools/cpp_probe/build/tree_probe > tests/fixtures/tree.json && python3 -m json.tool tests/fixtures/tree.json > /dev/null && echo OK`
Expected: `OK`.

- [ ] **Step 3: Write the failing tests**

`tests/util/test_tree.py` loads `tree.json` and, for each named case, replays the same operation sequence through the Python `tree` and asserts the serialised shape matches exactly. Serialisation helper:

```python
def to_sexpr(t, node=None):
    node = node if node is not None else t.begin()
    if node is None:
        return None
    kids = [to_sexpr(t, c) for c in t.children(node)]
    return {"data": node.data, "children": kids}
```

Plus non-oracle unit tests for each method's edge cases: erasing the root, flattening a childless node, `insert_above` on the root, `move_ontop` where source is a descendant of target, `equal_subtree` on structurally-equal-but-distinct subtrees, and `max_depth` on a single node.

- [ ] **Step 4: Run to confirm failure**

Run: `python3 -m pytest tests/util/test_tree.py -v`
Expected: FAIL — module not found.

- [ ] **Step 5: Implement `pymoses/util/tree.py`**

Build it as a real linked object graph — `parent` / `first_child` / `last_child` / `prev_sibling` / `next_sibling` — matching `tree.h:102-108`. Do **not** use a Python list of children: `insert_above`, `flatten`, `wrap` and `reparent` are pointer-splicing operations that are natural on links and error-prone on arrays, and matching classic's exact relinking is the whole point.

Every public method carries a breadcrumb citing its `tree.h` line.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/util/test_tree.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "port(util/tree): tree<T> container with the splice ops MOSES uses

Mirrors CogUtil opencog/util/tree.h (Kasper Peeters tree.hh v2.3). Linked
node graph, not an array, so insert_above/flatten/wrap/reparent relink
exactly as the C++ does. API scoped by measured call-site frequency in
classic rather than porting all 3420 lines.

Verified against tools/cpp_probe/tree_probe.cc for every splice operation
and traversal order."
```

---

### Task 3: `pymoses/comboreduct/combo/` — vertex, argument, combo_tree, I/O

**Files:**
- Create: `pymoses/comboreduct/combo/vertex.py`, `argument.py`, `common_def.py`, `iostream_combo.py`, `tests/comboreduct/combo/test_vertex.py`, `test_argument.py`, `test_iostream_combo.py`, `tests/fixtures/combo_roundtrip.json`

**Interfaces:**
- Consumes: `pymoses.util.tree.tree`
- Produces:
  - `common_def.py`: `arity_t = int`, `contin_t = float`.
  - `argument.py`: `class argument` with `idx: int`, `negate()`, `is_negated()`, `abs_idx()`, `abs_idx_from_zero()`, `__lt__`, `__eq__`, `__hash__`.
  - `vertex.py`: `class builtin(IntEnum)` mirroring `vertex.h:69-105` exactly including `null_vertex = 0`; `vertex = builtin | argument | contin_t`; `combo_tree = tree[vertex]`; helpers `is_builtin`, `is_argument`, `is_contin`, `negate_vertex`.
  - `iostream_combo.py`: `str_to_combo_tree(s: str) -> combo_tree`, `combo_tree_to_str(t: combo_tree) -> str`.

**Two `argument` details are load-bearing** — reduct depends on both, and getting either wrong changes reduction results silently:

1. **Sign encodes negation.** `negate()` is `idx = -idx`; `is_negated()` is `idx < 0` (`$CLASSIC/moses/comboreduct/combo/argument.h:54-59`). Indices are 1-based; `idx == 0` is asserted impossible.
2. **`operator<` orders by _absolute_ value** via `absolute_value_order<int>` (`argument.h:62-65`), so `$1` and `!$1` sort adjacent rather than at opposite ends. A natural `__lt__` on the signed int would produce different clause orderings in `flat_normal_form` and therefore different reductions. Port the comparator; do not infer it.

- [ ] **Step 1: Write the failing tests for `argument`**

```python
# tests/comboreduct/combo/test_argument.py
from pymoses.comboreduct.combo.argument import argument

def test_sign_encodes_negation():
    a = argument(3)
    assert not a.is_negated() and a.abs_idx() == 3
    a.negate()
    assert a.idx == -3 and a.is_negated() and a.abs_idx() == 3

def test_index_zero_is_rejected():
    import pytest
    with pytest.raises(AssertionError):
        argument(0)

def test_ordering_is_by_absolute_value():
    # C++: argument.h:62 uses absolute_value_order<int>, so $1 and !$1 are
    # adjacent. A signed-int comparison would sort !$1 before !$2 before $1.
    args = [argument(2), argument(-1), argument(1), argument(-2)]
    assert [a.idx for a in sorted(args)] == [-1, 1, -2, 2] or \
           [a.idx for a in sorted(args)] == [1, -1, 2, -2]
    assert [a.abs_idx() for a in sorted(args)] == [1, 1, 2, 2]

def test_abs_idx_from_zero():
    assert argument(1).abs_idx_from_zero() == 0
    assert argument(-1).abs_idx_from_zero() == 0
    assert argument(2).abs_idx_from_zero() == 1
```

Before implementing, read `$CLASSIC/moses/comboreduct/combo/argument.h:62-65` **and** `$COGUTIL/opencog/util/numeric.h` for `absolute_value_order`'s exact tie-breaking, and make the ordering assertion above exact rather than an `or` of two possibilities.

- [ ] **Step 2: Run to confirm failure, then implement `common_def.py` and `argument.py`**

Run: `python3 -m pytest tests/comboreduct/combo/test_argument.py -v` → FAIL, then implement, then re-run → PASS.

- [ ] **Step 3: Port the `builtin` enum verbatim**

Transcribe `$CLASSIC/moses/comboreduct/combo/vertex.h:69-105` into an `IntEnum`, preserving both names and implicit ordinal values (`null_vertex = 0` first). Write a test that asserts the member count and the ordinal of at least `null_vertex`, `logical_and`, `logical_or`, `logical_not`, `plus`, `times`, `greater_than_zero`, `cond`, matching the C++ header. Do not reorder or omit members even where unimplemented — ordinals are used in comparisons.

- [ ] **Step 4: Implement `vertex.py` and the combo I/O round-trip test**

The round-trip corpus is the 29 trees in `$CLASSIC/testTrees.sh` plus every tree literal in `$CLASSIC/tests/comboreduct/combo/iostream_comboUTest.cxxtest`. Capture C++ output with the already-built `combo-fmt-converter` at `~/opt/moses-legacy/bin/combo-fmt-converter` into `tests/fixtures/combo_roundtrip.json`, then assert Python `str_to_combo_tree` → `combo_tree_to_str` reproduces it exactly.

- [ ] **Step 5: Run tests, run CI, commit**

```bash
python3 -m pytest tests/comboreduct/combo -v && bash tools/ci.sh
git add -A && git commit -m "port(comboreduct/combo): vertex, argument, combo_tree and I/O

Mirrors vertex.h:69-105 (builtin enum, verbatim ordinals), argument.h:44-80
(sign-encoded negation, absolute-value ordering) and iostream_combo.

Round-trip verified against combo-fmt-converter over testTrees.sh and the
iostream_comboUTest corpus."
```

---

### Task 4: `pymoses/comboreduct/interpreter/` — boolean evaluation

**Files:**
- Create: `pymoses/comboreduct/interpreter/interpreter.py`, `tests/comboreduct/interpreter/test_interpreter.py`

**Interfaces:**
- Consumes: `combo_tree`, `builtin`, `argument`, `tree`
- Produces: `class boolean_interpreter` with `__call__(self, tr: combo_tree, inputs: list[builtin]) -> builtin`; `class interpreter_visitor` with `__call__(self, inputs) -> vertex` dispatching on the tree's type.

**Reference:** `$CLASSIC/moses/comboreduct/interpreter/interpreter.cc:47` (`boolean_interpreter`), and `$CLASSIC/moses/comboreduct/table/table.h:304` (`interpreter_visitor` — note it lives in `table.h`, not the interpreter directory).

- [ ] **Step 1: Port `interpreterUTest`**

Transliterate `$CLASSIC/tests/comboreduct/combo/interpreterUTest.cxxtest` (112 lines) to pytest, preserving test names. Run it — expect failure.

- [ ] **Step 2: Implement `boolean_interpreter`**

Handle `logical_and` (identity `true` on empty), `logical_or` (identity `false` on empty), `logical_not`, `argument` (1-based index into `inputs`, negated when `is_negated()`), and the `id::logical_true`/`logical_false` constants. Raise on any other builtin with a message naming the unsupported vertex — silence here would surface as a wrong score much later.

- [ ] **Step 3: Add the oracle test against `eval-table`**

For each tree in the round-trip corpus from Task 3 and each row of `$CLASSIC/trainingBoolean.dat`, compare Python's output against `~/opt/moses-legacy/bin/eval-table`. Capture into `tests/fixtures/eval_boolean.json`.

- [ ] **Step 4: Run tests, run CI, commit**

```bash
python3 -m pytest tests/comboreduct/interpreter -v && bash tools/ci.sh
git add -A && git commit -m "port(comboreduct/interpreter): boolean_interpreter

Mirrors interpreter.cc:47 and the interpreter_visitor dispatcher at
table.h:304. Ported interpreterUTest plus a differential test against
eval-table over the testTrees corpus and trainingBoolean.dat."
```

---

### Task 5: `pymoses/comboreduct/table/` — Table, CTable, loadTable

**Files:**
- Create: `pymoses/comboreduct/table/table.py`, `table_io.py`, `tools/cpp_probe/table_probe.cc`, `tests/comboreduct/table/test_table.py`, `test_table_io.py`, `tests/fixtures/table.json`

**Interfaces:**
- Consumes: `builtin`, `contin_t`
- Produces:
  - `class ITable` — input rows, `__len__`, `__getitem__`, `get_labels()`, `get_types()`, `get_arity()`.
  - `class OTable` — output column, `__len__`, `__getitem__`, `get_label()`, `get_type()`.
  - `class Table` — `itable`, `otable`, `target_pos`, `get_signature()`, `compressed(weighting_feature="") -> CTable`.
  - `class CTable` — `dict[tuple[vertex, ...], Counter[vertex]]` semantics with `uncompressed_size()`, `get_labels()`, `get_signature()`.
  - `loadTable(path, target_feature="", timestamp_feature="", ignore_features=()) -> Table`.

**Reference:** `$CLASSIC/moses/comboreduct/table/table_io.cc:1296` (`loadTable`), `table.cc:501` (`Table::compressed`), `table.h:550` (`CTable`), `table.h:738` (`ITable`), `table.h:866` (`OTable`).

`CTable` is what makes scoring O(distinct rows) rather than O(all rows) — it groups identical input rows and counts output occurrences per group. It is required for both correctness of the score arithmetic and realistic performance.

- [ ] **Step 1: Write the C++ probe**

`table_probe.cc` takes a CSV path plus optional target-column name, calls `loadTable` then `.compressed("")`, and dumps: inferred column labels, inferred column types, arity, the full ITable/OTable contents, and the CTable as a list of `{"inputs": [...], "counts": {"true": n, "false": m}}`.

Build against the installed MOSES libraries:
```bash
g++ -O2 -std=c++17 -I"$CLASSIC" -I"$HOME/opt/moses-legacy/include" \
    -o tools/cpp_probe/build/table_probe tools/cpp_probe/table_probe.cc \
    -L"$HOME/opt/moses-legacy/lib" -lcomboreduct -lcogutil
```
If linking fails, rebuild classic first with `$CLASSIC/scripts/rebuild-moses-local.sh -j4` — it forces `RelWithDebInfo` and strips LTO, which is required for these probes to link.

- [ ] **Step 2: Capture fixtures over the classic corpus**

Inputs: `$CLASSIC/tests/comboreduct/table/dataset.csv`, `dataset_force_feature.csv`, `sparse.data`, `weighted.csv`, `timestamp_table.csv`, plus `$CLASSIC/trainingBoolean.dat` and `trainingBooleanXor.dat`.

- [ ] **Step 3: Port `tableUTest` and write the oracle tests**

Transliterate `$CLASSIC/tests/comboreduct/table/tableUTest.cxxtest` (437 lines). Add an oracle test per fixture file asserting label inference, type inference, arity, row contents and CTable grouping all match.

- [ ] **Step 4: Implement, run, and commit**

Note the expected divergence: classic parses rows with `OMP_ALGO::for_each` (`table_io.cc:382`); we parse serially. Record it with a `DIVERGENCE` marker — parse order does not affect the resulting `ITable`.

```bash
python3 -m pytest tests/comboreduct/table -v && bash tools/ci.sh
git add -A && git commit -m "port(comboreduct/table): ITable, OTable, Table, CTable, loadTable

Mirrors table_io.cc:1296, table.cc:501 and table.h:550/738/866. Ported
tableUTest plus differential tests against tools/cpp_probe/table_probe.cc
over the classic table fixtures and the local training datasets."
```

---

### Task 6: scoring — types, complexity, ctruth_table_bscore, behave_cscore

**Files:**
- Create: `pymoses/moses/moses/types.py`, `complexity.py`, `pymoses/moses/scoring/scoring_base.py`, `bscores.py`, `behave_cscore.py`, `tools/cpp_probe/score_probe.cc`, `tests/moses/scoring/test_bscores.py`, `test_behave_cscore.py`, `tests/moses/moses/test_complexity.py`, `tests/fixtures/score.json`

**Interfaces:**
- Consumes: `CTable`, `combo_tree`, `boolean_interpreter`
- Produces:
  - `types.py`: `class composite_score` with `score`, `complexity`, `complexity_penalty`, `uniformity_penalty`, `get_penalized_score()`, and classic's comparison semantics; `behavioral_score = list[float]`; `class scored_combo_tree`.
  - `complexity.py`: `tree_complexity(tr) -> int`.
  - `scoring_base.py`: `class bscore_base` with `__call__`, `get_complexity`, `best_possible_bscore`, `min_improv`, `set_complexity_coef`, `get_complexity_coef`.
  - `bscores.py`: `class ctruth_table_bscore(bscore_base)` — `__init__(self, ctable)`.
  - `behave_cscore.py`: `class behave_cscore` — `__init__(self, bscorer, cache_size=0)`, `get_cscore(tr) -> composite_score`, `get_bscore(tr) -> behavioral_score`.

**Reference:** `$CLASSIC/moses/moses/scoring/bscores.cc:276-296` and `bscores.h:360-379` (`ctruth_table_bscore`), `behave_cscore.cc:51` and `:63` (`get_cscore` / `get_cscore_nocache`), `moses/moses/moses/types.h` (`composite_score`), `complexity.cc` (`tree_complexity`).

The `behave_cscore` cache is an optimisation, not semantics — use a plain `dict` or omit it entirely, and record the choice as a `DIVERGENCE` if omitted.

- [ ] **Step 1: Port `scoringUTest`, write the failing tests**

Transliterate `$CLASSIC/tests/moses/scoringUTest.cxxtest` (85 lines).

- [ ] **Step 2: Write the C++ probe**

`score_probe.cc` takes a CSV path and a combo tree string, builds the `CTable`, constructs `ctruth_table_bscore` and `behave_cscore`, and dumps the behavioral score vector, `sum_bscore`, `tree_complexity`, the complexity coefficient, and the resulting `composite_score` fields.

- [ ] **Step 3: Capture fixtures**

Cross every dataset from Task 5 with every tree from the Task 3 corpus. This is the cycle's exit criterion, so make the corpus wide — at minimum 20 trees × 4 datasets.

- [ ] **Step 4: Implement, run, and commit**

```bash
python3 -m pytest tests/moses -v && bash tools/ci.sh
git add -A && git commit -m "port(moses/scoring): ctruth_table_bscore, behave_cscore, composite_score

Mirrors bscores.cc:276-296, behave_cscore.cc:51-98, types.h composite_score
and complexity.cc tree_complexity. Ported scoringUTest plus differential
tests against tools/cpp_probe/score_probe.cc.

Closes the S0-S6 cycle: Python and C++ now produce identical behavioral
scores, complexity and composite scores across the corpus."
```

---

## Cycle exit criterion

All of the following must hold, each verified by running the command and reading the output:

1. `bash tools/ci.sh` — ruff clean, all tests pass, alignment checker exits 0.
2. `tests/fixtures/rng.json` oracle tests pass across all six seeds.
3. Scoring oracle tests pass for the full (dataset × tree) corpus.
4. `docs/ALIGNMENT.md` shows every in-scope C++ symbol as ported or explicitly diverged, with no unexplained gaps.

## Self-review notes

**Spec coverage.** Spec §4 layout → Task 0. §5.1 naming → Task 0 ruff config. §5.2–5.4 breadcrumbs/divergences/enforcement → Task 0. §6 RNG → Task 1. §7 Tier 1 ported cxxtests → Tasks 3–6; Tier 2 probes → Tasks 1, 2, 5, 6; Tier 3 trajectory diff → out of scope, S9+. §8 dependencies → Task 0 pyproject. §9 S0–S6 → Tasks 0–6 respectively. §10 workflow → applies to every task. §12 backlog → not in this plan by design.

**Known gap, deliberate:** the spec lists `type_checker/` and `eval.py` under the layout but neither is needed for the boolean scoring path. `type_treeUTest` is therefore not ported in this cycle; the type signature is constructed directly as `gen_signature(boolean_type, arity)` in Task 5. This is flagged for the S7+ plan rather than silently dropped.
