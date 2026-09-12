"""Build tools/cpp_probe/subsample_selector_probe.cc and write its JSON
output to tests/fixtures/subsample_selector.json, with a sidecar manifest.

Separate from tools/capture_golden.py (which owns rng_probe.cc) because
this probe needs its own `_INPUTS` description and is a one-off P2
metapopulation deliverable, not part of that script's fixed table -- see
docs/CARRY-FORWARD.md Sec 5 for why capture scripts stay small and
single-purpose rather than growing a shared registry of every probe ever
added.

Usage (see subsample_selector_probe.cc's own header comment for the
equivalent manual g++ invocation):

    export PYMOSES_MOSES_PREFIX=$HOME/opt/moses-legacy
    python3 tools/capture_subsample_selector.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

PROBE_SOURCE = REPO_ROOT / "tools" / "cpp_probe" / "subsample_selector_probe.cc"
BUILD_DIR = REPO_ROOT / "tools" / "cpp_probe" / "build"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "subsample_selector.json"


def build_probe(binary: Path) -> None:
    prefix = moses_prefix()
    include = prefix / "include"
    lib = prefix / "lib"
    lib_opencog = lib / "opencog"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-I",
        str(include),
        "-o",
        str(binary),
        str(PROBE_SOURCE),
        "-L",
        str(lib),
        "-L",
        str(lib_opencog),
        "-Wl,-rpath," + str(lib),
        "-Wl,-rpath," + str(lib_opencog),
        "-lcogutil",
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    del argv
    binary = BUILD_DIR / "subsample_selector_probe"
    build_probe(binary)
    result = subprocess.run([str(binary)], capture_output=True, check=True, text=True)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(result.stdout)
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_subsample_selector.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/subsample_selector_probe.cc",
            probe_source_path=PROBE_SOURCE,
            probe_kind="cpp_probe",
            prefix_dir=moses_prefix(),
            inputs={
                "configs": [
                    "exhaustive_n4_demes1_base4",
                    "exhaustive_n64_demes3_base4",
                    "exhaustive_n9_demes2_base3",
                    "partial_n64_demes3_base4",
                    "partial_n1000_demes2_base10",
                    "edge_demes0",
                    "base1_demes5",
                ],
                "note": (
                    "each config drives a real opencog::MT19937RandGen + "
                    "opencog::lazy_random_selector through the exact draw/"
                    "decompose loop shape subsample.cc's three tuple-"
                    "sampling functions share (see probe header comment)"
                ),
            },
        ),
    )
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
