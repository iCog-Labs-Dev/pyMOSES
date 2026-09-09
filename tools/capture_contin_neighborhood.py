"""Build and capture the S14 Task 3 contin-neighborhood oracle fixture.

Mirrors tools/capture_neighborhood.py's shape (S9's own neighborhood
fixture, which covers the Boolean/discrete-only functions in
neighborhood_sampling.{h,cc}): compile
tools/cpp_probe/contin_neighborhood_probe.cc against the installed classic
headers and libraries, run it, and save its JSON stdout as
tests/fixtures/contin_neighborhood.json. Every subprocess is
resource-bounded (RLIMIT_AS + a wall-clock timeout) per this session's
standing OOM caution.

This fixture is for S14 Task 3's three continuous sampling helpers
(`flip_LR`, `twiddle_contin_bit`, `generate_contin_neighbor`,
neighborhood_sampling.cc:32,38,55) and `sample_from_neighborhood`'s own
"modify contin" branch that calls the last of them -- see that probe's own
module comment for the full case breakdown and for why every
`generate_contin_neighbor` case appends a trailing disc(3) field (avoiding
a genuine C++ UB edge case at the raw-array boundary; see the probe's own
comment on this).
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_neighborhood_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_neighborhood_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_neighborhood.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 60


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    command = [
        "g++",
        "-O2",
        "-std=c++17",
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
        "-lmoses",
        "-lcomboreduct",
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
            f"contin_neighborhood probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"contin_neighborhood probe compilation failed:\n{result.stderr}")


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
        raise RuntimeError(f"contin_neighborhood probe run exceeded {_RUN_TIMEOUT_S}s") from error
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_neighborhood.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_neighborhood_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed flip_LR/twiddle_contin_bit/generate_contin_neighbor/"
                    "sample_from_neighborhood(contin) test cases hardcoded in "
                    "contin_neighborhood_probe.cc; no external dataset, seeds fixed "
                    "per case for RNG-drawing sections"
                )
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
