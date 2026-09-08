"""Build and capture the S8 classic type-signature oracle fixture."""

import json
import resource
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "type_signature_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "type_signature_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "type_signature.json"
_MEM_LIMIT_KIB = 2097152
_MEM_LIMIT_BYTES = _MEM_LIMIT_KIB * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 20


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    cmd = [
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
        "-lcomboreduct",
        "-lcogutil",
    ]
    shell_cmd = (
        f"ulimit -v {_MEM_LIMIT_KIB}; timeout {_COMPILE_TIMEOUT_S} {shlex.join(cmd)}"
    )
    result = subprocess.run(["bash", "-c", shell_cmd], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"g++ build failed (exit {result.returncode})")


def capture_fixture() -> None:
    result = subprocess.run(
        [str(PROBE_BIN)],
        capture_output=True,
        check=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        preexec_fn=_bound_resources,
    )
    fixture = json.loads(result.stdout)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_type_signature.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/type_signature_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed type-signature test cases hardcoded in "
                    "type_signature_probe.cc; no external dataset or seed"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_type_signature.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
