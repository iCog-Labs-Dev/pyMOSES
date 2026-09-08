"""Build and capture the S10 Task 0 scored_combo_tree containers oracle
fixture (sct_score_greater/sct_tree_greater ordering,
scored_combo_tree_hash/scored_combo_tree_equal-driven deduplication).

Mirrors tools/capture_containers.py (S9 Task 0's instance_set/
scored_instance oracle capture) exactly, pointed at the new
containers_sct_probe.cc instead.
"""

import json
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "containers_sct_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "containers_sct_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "containers_sct.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 20


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
            f"containers_sct probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode:
        raise RuntimeError(
            "containers_sct probe compilation failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN)],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"containers_sct probe exceeded {_RUN_TIMEOUT_S}s") from error
    if result.returncode:
        raise RuntimeError(
            "containers_sct probe failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    fixture = json.loads(result.stdout)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_containers_sct.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/containers_sct_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed scored_combo_tree ordering/dedup test cases hardcoded "
                    "in containers_sct_probe.cc; no external dataset or seed"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_containers_sct.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
