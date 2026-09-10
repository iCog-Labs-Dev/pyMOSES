"""Build and capture the univariate-RNG oracle fixture.

Mirrors tools/capture_contin_enumeration.py's shape: compile
tools/cpp_probe/univariate_rng_probe.cc against the installed cogutil
headers and library, run it, and save its JSON stdout as
tests/fixtures/univariate_rng.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this project's standing OOM caution.

The fixture covers the two random streams the `-aun` (univariate) optimizer
reads that nothing else in MOSES reads, and that between them accounted for
the whole measured `-aun` divergence above `-m1000`:

  * glibc's `rand()` -- consumed by the BARE `std::random_shuffle` at
    `opencog/util/digraph.h:96`, which `randomized_topological_sort` uses to
    order a `local_structure_model`'s fields. Sections `rand_default`,
    `rand_srand1`, `rand_srand0` and `rand_seeded` pin the raw generator;
    `topo_sort_edgeless` and `topo_sort_chains` pin the whole
    shuffle+topological-sort composition for the two graph shapes
    `local_structure_model`'s constructor can build (no edges at all for a
    Boolean-only field_set; one chain per contin field otherwise).

  * `opencog::roulette_select` at BOTH live instantiations. Sections
    `roulette_int` (`ScoreT = int`, `local_structure.cc:146`'s real one,
    since `dtree_node` is `vector<int>`) and `roulette_double` dump the
    SAME (counts, randdouble) grid through the real C++ template, so the
    disagreement between the two is a measurement rather than an assertion.

The probe's own module comment lists which C++ arm each section reaches.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "univariate_rng_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "univariate_rng_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "univariate_rng.json"
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
        # gnu++17 matches classic's own CXX_FLAGS
        # (build/moses/moses/CMakeFiles/moses.dir/flags.make), which matters
        # here: `std::random_shuffle` is deprecated from C++14 and removed
        # from C++17, and survives only because libstdc++ keeps it behind
        # `_GLIBCXX_USE_DEPRECATED` (on by default).
        "-std=gnu++17",
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
            f"univariate_rng probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"univariate_rng probe compilation failed:\n{result.stderr}")


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
        raise RuntimeError(f"univariate_rng probe run exceeded {_RUN_TIMEOUT_S}s") from error
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_univariate_rng.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/univariate_rng_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "raw std::rand() streams (process default, srand(1), "
                    "srand(0) and six explicit seeds); randomized_topological_"
                    "sort over edgeless graphs of 15 sizes and over four "
                    "dependency-chain graphs, 12 consecutive calls each from a "
                    "freshly srand(1)'d state; and roulette_select over a fixed "
                    "26 x 239 (leaf-counts x randdouble) grid at BOTH "
                    "ScoreT = int and ScoreT = double. All hardcoded in "
                    "univariate_rng_probe.cc; no external dataset, and the "
                    "roulette sections drive a scripted randdouble rather than "
                    "MT19937 so they isolate roulette_select's own arithmetic."
                ),
                "host_dependency": (
                    "glibc's rand() (TYPE_3 additive feedback) and libstdc++'s "
                    "two-argument std::random_shuffle; see "
                    "pymetta/util/digraph.py's module docstring"
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
