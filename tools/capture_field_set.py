"""Build and capture the S8 classic field-set oracle fixture."""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIC = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses")
PREFIX = Path.home() / "opt" / "moses-legacy"
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "field_set_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "field_set_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "field_set.json"
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
            f"field-set probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode:
        raise RuntimeError(
            "field-set probe compilation failed\n"
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
        raise RuntimeError(f"field-set probe exceeded {_RUN_TIMEOUT_S}s") from error
    if result.returncode:
        raise RuntimeError(
            "field-set probe failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    fixture = json.loads(result.stdout)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_field_set.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
