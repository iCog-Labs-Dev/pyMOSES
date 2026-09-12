"""Build and capture the `nth_element` oracle fixture.

Mirrors tools/capture_contin_enumeration.py's shape: compile
tools/cpp_probe/nth_element_probe.cc against the installed cogutil headers,
run it, and save its JSON stdout as tests/fixtures/nth_element.json. Every
subprocess is resource-bounded (RLIMIT_AS + a wall-clock timeout) per this
project's standing OOM caution.

Why the fixture exists: `replace_the_worst` (`moses/moses/eda/
replacement.h:54`) calls `OMP_ALGO::nth_element`, and the permutation it
leaves in the TAIL of the range is observed -- by `sort_demes`' non-stable
`std::sort`, by `keep_top_unique_candidates`' adjacent-duplicate scan, and
by `tournament_selection`'s positional indexing. The fixture pins:

  * `omp_algo_is_std` -- which macro branch cogutil's `oc_omp.h` actually
    compiles to when built with classic's own `-fopenmp` flag;
  * `omp_matches_std` -- whether `OMP_ALGO::nth_element` and
    `std::nth_element` agree element for element over every case;
  * `cases` -- the full input and output arrays for 133 cases: every
    (size, nth) pair for sizes 0..17, pseudo-random key streams with 1..50
    distinct values at sizes up to the real `-aun` deme's 597, and the
    sorted / reversed / organ-pipe / all-equal shapes at that size.

Elements are `[key, tag]` pairs compared on `key` alone, so ties -- the
only place the permutation is observable at all -- are readable.

The probe compiles with `-fopenmp` deliberately: that is the flag classic's
own build uses (`build/moses/moses/CMakeFiles/moses.dir/flags.make`), and
`OMP_ALGO`'s definition is what is being measured.
"""

import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "nth_element_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "nth_element_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "nth_element.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    command = [
        "g++",
        "-O2",
        "-std=gnu++17",
        "-fopenmp",
        "-I",
        str(CLASSIC),
        "-I",
        str(PREFIX / "include"),
        "-o",
        str(PROBE_BIN),
        str(PROBE_SRC),
        "-L",
        str(lib),
        "-L",
        str(lib_opencog),
        "-Wl,-rpath," + str(lib),
        "-Wl,-rpath," + str(lib_opencog),
        "-lcogutil",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMPILE_TIMEOUT_S,
            preexec_fn=_bound_resources,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"nth_element probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"nth_element probe compilation failed:\n{result.stderr}")


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN)],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"nth_element probe run exceeded {_RUN_TIMEOUT_S}s") from error
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_nth_element.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/nth_element_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "133 OMP_ALGO::nth_element cases over vectors of "
                    "[key, tag] pairs compared on key alone: every (size, nth) "
                    "pair for sizes 0..17, pseudo-random key streams with 1, 2, "
                    "3, 7 and 50 distinct keys at sizes 20/33/64/100/298/597, "
                    "and sorted / reversed / organ-pipe / all-equal shapes at "
                    "597 (the real -aun deme size). Key streams come from an "
                    "LCG hardcoded in the probe, so the fixture depends on no "
                    "library RNG and no external dataset."
                ),
                "host_dependency": (
                    "libstdc++'s std::__introselect (median-of-3 introselect); "
                    "see pymoses/moses/eda/replacement.py's module docstring"
                ),
            },
        ),
    )


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
