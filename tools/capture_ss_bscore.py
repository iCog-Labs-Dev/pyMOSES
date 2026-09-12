"""Build and capture the ss_bscore.py oracle fixture.

Mirrors tools/capture_eda_initialization.py's shape: compile
tools/cpp_probe/ss_bscore_probe.cc against the installed classic headers
and libraries, run it, and save its JSON stdout as
tests/fixtures/ss_bscore.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this session's standing OOM caution.

This fixture is the oracle for `pymoses.moses.scoring.ss_bscore`'s port of
moses/moses/scoring/ss_bscore.{h,cc}'s three self-contained, RNG-drawing
computations (the accumulator variance recurrence, the low-deviation
penalty, and the two subsample-partitioning branches) -- see the probe's
own module comment for why the full class cannot be exercised end to end
in this port.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "ss_bscore_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "ss_bscore_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ss_bscore.json"
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
        raise RuntimeError(f"ss_bscore probe compilation exceeded {_COMPILE_TIMEOUT_S}s") from error
    if result.returncode != 0:
        raise RuntimeError(f"ss_bscore probe compilation failed:\n{result.stderr}")


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
        raise RuntimeError(f"ss_bscore probe run exceeded {_RUN_TIMEOUT_S}s") from error
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_ss_bscore.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/ss_bscore_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed sample sequences, shuffle/partition configs and "
                    "seeds hardcoded in ss_bscore_probe.cc; no external dataset"
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
