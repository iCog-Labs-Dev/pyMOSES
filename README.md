# pyMoses

A Python port of classic [OpenCog MOSES](https://github.com/opencog/moses) — the C++
evolutionary program learner that evolves short `combo` programs to fit a dataset.

The port is a line-for-line transliteration of the C++ rather than a reimplementation,
and it is held to a hard standard: **given the same seed and the same flags, it prints
what the C++ prints, byte for byte.**

```console
$ moses -i tests/fixtures/table_data/trainingBooleanXor.dat -u y -m 2000 -r 1 -W1
0 or(!$x1 !$x2)
-1 true
-1 !$x1
-1 !$x2
```

Classic MOSES, given those same arguments, prints those same four lines.

---

## Requirements

- **Python 3.14 or newer.** This is enforced by `requires-python`, so `pip` will refuse
  to install on an older interpreter rather than fail mysteriously later.
- **Nothing else.** The port has no runtime dependencies — it is pure standard library.

You do **not** need a C++ toolchain, CMake, Boost, or a MOSES/CogUtil checkout to
install or run pyMoses. Those are needed only to re-derive test fixtures from the C++
(see [Development](#development)).

Check your interpreter first:

```bash
python3 --version      # must be 3.14 or newer
```

## Install

```bash
git clone https://github.com/iCog-Labs-Dev/pyMOSES.git
cd pyMOSES

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install .
```

That installs three commands, deliberately named after the binaries classic MOSES
installs (classic's CMake *target* names differ from its installed *binary* names):

| Command | Classic CMake target | What it does |
|---|---|---|
| `moses` | `moses-exec` | the learner |
| `feature-selection` | `fs-main` | standalone feature selection |
| `eval-features` | `eval-features` | score a feature set |

Confirm the install:

```bash
moses --help
```

One wrinkle to know about up front: **`moses --help` exits with status 1, not 0.** That
is not a bug in the port — classic MOSES does the same, and the port reproduces it
deliberately. It does mean `moses --help && echo ok` will not print `ok`, so don't wire
`--help` into a health check.

### Running without installing

The package is importable straight from a clone, so you can skip `pip install`
entirely. Every `moses` command below also works spelled this way:

```bash
python3 -m pymoses.moses.main --help
```

## Run it

**On a dataset.** Point `-i` at a CSV and name the target column with `-u`. The repo
ships small datasets under `tests/fixtures/table_data/` that you can use right away:

```bash
moses -i tests/fixtures/table_data/trainingBooleanXor.dat -u y -m 2000 -r 1 -W1
```

```
0 or(!$x1 !$x2)
-1 true
-1 !$x1
-1 !$x2
```

Reading the output: each line is `<score> <program>`. For a Boolean problem the score
counts rows got wrong, so `0` is a perfect fit and `-1` is one row wrong. `-W1` labels
the arguments with the CSV's own column names (`$x1`) rather than positions (`$1`).

Results are ordered best-first by the score MOSES actually optimises, which is the score
shown *minus a complexity penalty* — so a short program can outrank a longer one that
fits better. Pass `-x1` to see the breakdown, which is where that ordering stops looking
surprising (digits trimmed here for width — the real output prints them in full):

```console
$ moses -i tests/fixtures/table_data/regression_sample.csv -u y -m 2000 -r 1 -W1 -x1
-0.0733… 0  [score=-0.0733…, penalized score=-0.359…, complexity=1, complexity penalty=0.2857…]
-0.0455… *(+(0.25 *(-0.25 $x1)) $x1)  [score=-0.0455…, penalized score=-1.188…, complexity=4, …]
```

The second program fits better (`-0.0455` beats `-0.0733`) yet ranks lower, because its
complexity costs it more than the fit gains.

That last example is also the continuous case: pyMoses handles real-valued targets as
well as Boolean ones, evolving arithmetic programs instead of logical ones.

Try the three-input Boolean set too:

```bash
moses -i tests/fixtures/table_data/trainingBoolean.dat -u y -m 2000 -r 1 -W1
# 0 or(and($x1 $x2) !$x3)
```

**On a built-in demo problem,** with no data file at all. `-H` selects the problem and
`-k` its arity — here, 3-input parity:

```bash
moses -H pa -k 3 -m 10000 -r 1
# 0 or(and(or(!$1 !$2) or($1 $2) $3) and(or(!$1 $2) or($1 !$2) !$3))
```

The flags worth knowing when you start: `-m` caps the number of evaluations (the main
cost knob), `-r` sets the RNG seed and makes a run reproducible, `-c` asks for more than
one result, and `-l DEBUG -f run.log` writes a trace. `moses --help` lists all 138
options; they are the same options, with the same names and defaults, as classic's.

Available `-H` problems: `pa` (parity), `dj` (disjunction), `mux` (multiplexer),
`maj` (majority), `sr` (polynomial regression), `cp` (combo program), plus the
table-driven ones — `it`, `ip`, `pre`, `pre-conj`, `prerec`, `recall`, `bep`, `f_one`,
`select`, `cluster`. These are classic's own names, so `maj` is spelled `maj`, not
`majority`.

### A word on speed

This is pure Python doing what optimised C++ does, so expect roughly **20× the
runtime**. The parity run above takes about 100 s here, against 4.4 s for the C++
binary. Start with a small `-m` and raise it. Fidelity, not speed, is the point of this
port — if you need MOSES to be fast, run the C++.

## Extension: GOMEA inner optimizer (experimental)

Beyond the classic port, this branch adds one **non-classic** option: a GP-GOMEA-style
population optimizer that can replace MOSES's hill climber *inside* each deme. It leaves
exemplar selection, knob building and the metapopulation unchanged. It is Boolean-only
for now.

```bash
moses -H pa -k 3 -m 3000 -r 1 -a gomea --gomea-pop-size 16
```

Without `-a gomea`, output is byte-identical to classic, as before. In Phase 1, under equal evaluation budgets, it did **not** beat
the stock hill climber on the tested Boolean tasks. `--gomea-fos uni` was its best variant. For the options, the
telemetry, the Python API and the measured comparison with hill climbing, see
[`docs/gomea/USAGE.md`](docs/gomea/USAGE.md) and
[`docs/gomea/PHASE1-FINDINGS.md`](docs/gomea/PHASE1-FINDINGS.md). A follow-up comparing the
authors' stand-alone GP-GOMEA with MOSES hill climbing is in
[`docs/gomea/PHASE1B-FINDINGS.md`](docs/gomea/PHASE1B-FINDINGS.md), how it scales to
larger problems in [`docs/gomea/PHASE1C-FINDINGS.md`](docs/gomea/PHASE1C-FINDINGS.md), and why hill
climbing won in [`docs/gomea/PHASE1D-FINDINGS.md`](docs/gomea/PHASE1D-FINDINGS.md): capping GOMEA's
per-deme evaluations (`--gomea-max-evals-per-deme 500`) closes most of the gap. With a cap of
250, the lt-b FOS and the genotype memo, GOMEA needed about half of hill climbing's
evaluations on 11-input multiplexer and tied on majority-7 and parity-5, at several times
the CPU time ([`docs/gomea/PHASE1E-FINDINGS.md`](docs/gomea/PHASE1E-FINDINGS.md)).

## Run the tests

The test suite is self-contained: it diffs against C++ **golden fixtures committed to
this repo**, so it needs no C++ checkout and no compiler.

```bash
pip install -e ".[dev]"
pytest
```

Expect **8291 passed, 5 skipped** in about 10 minutes on one core: the classic port's
7898 (verified from a clean checkout and a fresh virtualenv) plus 393 added on this branch
for the GOMEA extension (measured on this branch).

On a machine without a classic MOSES/CogUtil checkout you should see 7 more tests skip,
roughly `8284 passed, 12 skipped`. Those are the tests that read the C++ sources
directly; they skip rather than fail, so a green run is still a green run. Everything
else compares against golden fixtures committed here.

## What is and isn't ported

The port covers MOSES's **Boolean and continuous** domains — that is the whole product
scope. The legacy embodiment/action vocabulary is deliberately excluded, as is dead
`#ifdef` code and the C++-only side binaries.

Against classic's 247 source files / 62,423 lines, 159 files (50,211 lines, **80.4 %**)
have Python behind them; 89 of those are complete with no named gap, and the rest carry
a documented, per-file gap. `docs/MODULE-COVERAGE.md` accounts for **every** classic
file with a status and a reason — read that rather than the headline percentage.

## How it stays honest

The port's value rests entirely on being auditable against the C++, so alignment is
mechanically enforced rather than promised:

- Every ported symbol carries a `C++: <path>:<line> <symbol>` breadcrumb as the first
  line of its docstring, pointing at the exact C++ definition it came from.
- Every deliberate difference is marked in-source with `# DIVERGENCE(C++ <ref>): <why>`
  — including the handful of places where classic's own behaviour is unreproducible by
  design (iteration order of `std::unordered_set`, for instance).
- `tools/check_alignment.py` verifies every breadcrumb still points at a real, nearby
  line in a classic checkout, and regenerates `docs/ALIGNMENT.md`.

Further reading: `docs/ALIGNMENT.md` (the breadcrumb and divergence inventory),
`docs/MODULE-COVERAGE.md` (per-file port status), `docs/CARRY-FORWARD.md` (the C++
semantics that cost the most to reproduce — `score_t` being `float` while `contin_t` is
`double`, `std::mt19937` reproduction, x86-64 float→int conversion, and the rest).

## Layout

| Path | Contents |
|---|---|
| `pymoses/` | the port, mirroring classic's tree file-for-file (`comboreduct/…`, `moses/…`) |
| `tests/` | the pytest suite, plus committed C++ golden fixtures under `tests/fixtures/` |
| `tools/` | alignment checker, fixture-capture scripts, C++ oracle probes |
| `docs/` | alignment, coverage, and carry-forward notes |

## License

Same as classic MOSES, which is **dual-licensed**: you may use it under the
**Apache License 2.0** *or* the **GNU Affero General Public License v3**, at your
option. Classic's three licence files are reproduced here verbatim — `LICENSE`
(the one-line statement of the dual grant), `LICENSE.Apache`, and `LICENSE.AGPL`.

Note that AGPL-3.0 and Apache-2.0 pull in very different obligations, so which one
you are taking the code under is worth deciding deliberately rather than by default.

## Development

Contributors additionally need local checkouts of classic
[MOSES](https://github.com/opencog/moses) and
[CogUtil](https://github.com/opencog/cogutil), built, to re-derive fixtures with the
`tools/capture_*.py` scripts and to run the alignment checker.

<!-- AGENTS / MAINTAINERS -- the full local gate is:

         bash tools/ci.sh      # ruff + pytest + alignment + coverage + fixture manifests

     It self-bootstraps a local .venv with pytest and ruff on first run and is
     idempotent thereafter. Do NOT put this in the human-facing instructions above:
     ci.sh CANNOT PASS on any machine but the original author's, and fixing that is
     an open task.

     Why it can't: the classic-checkout locations are hardcoded absolute paths, so
     the last three stages fail anywhere else.

       - tools/check_alignment.py:78  CLASSIC_ROOTS = [Path("/home/yab/Documents/
         Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses"), .../CogUtil,
         Path("/usr/include")]
       - tests/conftest.py:9          CLASSIC  = Path("/home/yab/.../OpenCog/Moses")
       - tests/conftest.py:10         COGUTIL  = Path("/home/yab/.../OpenCog/CogUtil")

     Also hardcoded in ~74 committed tests/fixtures/*.manifest.json provenance
     fields, which are records of where a fixture was captured and are fine as-is.

     The fix: read both roots from environment variables (e.g. $CLASSIC_MOSES_ROOT,
     $COGUTIL_ROOT) falling back to sibling-directory discovery, and make
     check_alignment.py degrade to a skip-with-warning when neither resolves --
     matching how tests/conftest.py already skips. `pytest` alone is unaffected and
     passes anywhere; only ci.sh's alignment/coverage/manifest stages are blocked.

     Licensing is settled: classic's LICENSE, LICENSE.AGPL and LICENSE.Apache are
     copied in verbatim (byte-identical) and declared in pyproject.toml. Note the
     ported modules carry NO per-file licence header, unlike classic's .cc/.h --
     deliberately, because check_alignment.py parses the FIRST line of each
     docstring as the `C++: <path>:<line> <symbol>` breadcrumb. If headers are
     ever wanted, they must go in `#` comments ABOVE the docstring, never inside
     it, or every breadcrumb in the repo stops resolving.
-->
