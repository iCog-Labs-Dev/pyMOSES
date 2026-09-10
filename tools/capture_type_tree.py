"""Capture the combo type-inference oracle into tests/fixtures/type_tree.json.

Mirrors tools/capture_contin_enumeration.py's shape. Compiles and runs
tools/cpp_probe/type_tree_probe.cc, which links classic's OWN
`moses/comboreduct/type_checker/type_tree.cc` and
`moses/comboreduct/type_checker/type_tree_def.cc` out of
libcomboreduct.so -- so the fixture is classic's behaviour, not a
reimplementation of it.

The probe's own module comment lists every section, the C++ definition it
is an oracle for, and which arm of that definition each case reaches. Its
program corpus is deliberately half ILL-TYPED: a corpus of only well-typed
programs would let a stub that always answers "well formed" pass every
row, which is this project's most-repeated defect.

Run it as:

    PYMETTA_CLASSIC_DIR=/path/to/OpenCog/Moses \\
    PYMETTA_MOSES_PREFIX=$HOME/opt/moses-legacy \\
    .venv/bin/python3 tools/capture_type_tree.py
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "type_tree_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "type_tree_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "type_tree.json"
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
        raise RuntimeError(f"type_tree probe compilation exceeded {_COMPILE_TIMEOUT_S}s") from error
    if result.returncode != 0:
        raise RuntimeError(f"type_tree probe compilation failed:\n{result.stderr}")


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
        raise RuntimeError(f"type_tree probe run exceeded {_RUN_TIMEOUT_S}s") from error
    if not result.stdout.strip():
        raise RuntimeError(
            "type_tree probe produced no output -- refusing to write an empty fixture"
        )
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_type_tree.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/type_tree_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed corpora hardcoded in type_tree_probe.cc; no external "
                    "dataset and no RNG. All 26 id::builtin and all 14 id::action "
                    "values through builtins_properties/actions_properties; 35 "
                    "single-vertex cases; 62 combo programs (roughly half of them "
                    "deliberately ILL-TYPED, covering the arity, operand-type and "
                    "cross-argument mismatch families separately); 39 type-tree "
                    "pairs for inherit/equal/intersection; 32 reduce_type_tree "
                    "cases with and without an arg_types vector; the full "
                    "type_node stream round trip; and convert_index / "
                    "abs_min_arity / get_type_node / default_vertex_value / "
                    "is_argument_type / arg_to_idx tables"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    del argv  # no CLI options -- one fixed probe run to (re)capture
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
