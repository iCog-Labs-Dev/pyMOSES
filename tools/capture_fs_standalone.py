"""Build and capture the P2 feature-selection STANDALONE oracle fixture.

Compiles tools/cpp_probe/fs_standalone_probe.cc against the installed
classic headers and libraries, runs it against the two boolean fixture
tables (tests/fixtures/table_data/trainingBoolean.dat and
trainingBooleanXor.dat), and saves its JSON stdout as
tests/fixtures/fs_standalone.json, with a fs_standalone.manifest.json
sidecar (tools/_manifest.py).

Mirrors tools/capture_feature_selection.py's shape (same probe link line:
`-lfeature_selection -lmoses -lcomboreduct -lcogutil
-lboost_program_options`).

Covers "random"/"smd"/"inc" (algo/random.{h,cc},
algo/stochastic_max_dependency.{h,cc}, algo/incremental.{h,cc}) -- NOT
"hc" (algo/deme_optimize.{h,cc}): classic's own "--algo hc" path segfaults
unconditionally (confirmed against the real installed binary and a gdb
backtrace -- see fs_standalone_probe.cc's own header comment), so there is
no live classic output to capture for it.

Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) per this
project's standing OOM caution.
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, sidecar_path, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "fs_standalone_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "fs_standalone_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "fs_standalone.json"
BOOLEAN_CSV = REPO_ROOT / "tests" / "fixtures" / "table_data" / "trainingBoolean.dat"
XOR_CSV = REPO_ROOT / "tests" / "fixtures" / "table_data" / "trainingBooleanXor.dat"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 120
_SEED = 1


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
        "-lfeature_selection",
        "-lmoses",
        "-lcomboreduct",
        "-lcogutil",
        "-lboost_program_options",
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
            f"fs_standalone probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"fs_standalone probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_fs_standalone.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/fs_standalone_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "random_select_feature_sets() and smd_select_feature_sets() "
                "at target_size 0-3 (boolean) / 0-2 (xor), seed reset to 1 "
                "immediately before each call (matching a fresh -r 1 "
                "process); incremental_select_feature_sets() for seven "
                "target_size/threshold/inc_interaction_terms combinations "
                "on both tables (fully deterministic, no RNG); "
                "smd_select_feature_sets() with scorer='pre' (pre_scorer, "
                "moses_matrix.h) at two target_sizes, with the four "
                "otherwise-uninitialized pre_* fields set to their real "
                "CLI defaults. 'hc' "
                "(algo/deme_optimize.{h,cc}) is deliberately NOT captured "
                "-- classic's own --algo hc path segfaults unconditionally, "
                "confirmed against the real installed binary and a gdb "
                "backtrace (hill_climbing::operator() dereferences a null "
                "iscorer.get_rep() in an unconditional debug-log line, "
                "hill-climbing.cc:69) -- see fs_standalone_probe.cc's own "
                "header comment."
            ),
            "seed": _SEED,
            "boolean_csv": "tests/fixtures/table_data/trainingBoolean.dat",
            "xor_csv": "tests/fixtures/table_data/trainingBooleanXor.dat",
        },
    )


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN), str(BOOLEAN_CSV), str(XOR_CSV), str(_SEED)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"fs_standalone probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_sidecar(FIXTURE_PATH, _build_manifest())


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH} and {sidecar_path(FIXTURE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
