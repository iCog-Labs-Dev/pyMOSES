"""Build tools/cpp_probe/tree_probe.cc and capture its JSON output into
tests/fixtures/tree.json.

docs/CARRY-FORWARD.md Sec 5 named this fixture's missing capture script as
a standing gap: "tree.json has no capture script in-repo -- the recipe
lives only in the plan document" (docs/superpowers/plans/
2026-09-03-pymoses-s0-s6-foundation.md:538-547). This script is that
recipe, made runnable and path-independent.

tree_probe.cc is deliberately header-only (`#define IGNORE_OC_ASSERT`
before `#include <opencog/util/tree.h>`, per that file's own header
comment) so it needs only a raw CogUtil source checkout on the include
path -- no libcogutil link, no installed moses-legacy prefix, no classic
MOSES checkout at all. That is resolved via PYMOSES_COGUTIL_DIR
(tools/_capture_env.py's `cogutil_dir()`), matching the plan document's
own recipe (`g++ ... -I"$COGUTIL" ... tree_probe.cc`).

Usage (from the repo root, with `.venv` already bootstrapped by
tools/ci.sh):

    PYMOSES_COGUTIL_DIR=/path/to/CogUtil .venv/bin/python3 tools/capture_tree.py
"""

import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import cogutil_dir  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "tree_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "tree_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "tree.json"

# CLAUDE.md: "this machine has been OOM-killed twice. Bound every C++
# run." Generous but bounded -- this probe is small and dependency-free
# beyond cogutil headers, but the compile still needs a real bound.
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 60


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-I",
        str(cogutil_dir()),
        "-o",
        str(PROBE_BIN),
        str(PROBE_SRC),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_COMPILE_TIMEOUT_S,
        preexec_fn=_bound_resources,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tree_probe compilation failed:\n{result.stderr}")


def capture_fixture() -> None:
    result = subprocess.run(
        [str(PROBE_BIN)],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_tree.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/tree_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            inputs={
                "cogutil_dir": str(cogutil_dir()),
                "description": (
                    "fixed splice-operation and traversal-order test cases "
                    "hardcoded in tree_probe.cc; no external dataset or seed. "
                    "Needs only a raw CogUtil source checkout ($COGUTIL) on "
                    "the include path -- no libcogutil link, no moses-legacy "
                    "prefix, no classic MOSES checkout."
                ),
            },
        ),
    )


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed probe run to (re)capture
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
