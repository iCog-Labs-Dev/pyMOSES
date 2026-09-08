"""Build tools/cpp_probe/table_probe.cc and capture its JSON output, over
the classic table-loading fixture corpus, into tests/fixtures/table.json.

The committed tests/fixtures/table.json is what the default test suite
actually reads (see tests/comboreduct/table/test_table_io.py), so the suite
runs without a C++ toolchain or the classic checkout at all. This script is
a developer tool, not part of tools/ci.sh.

table_probe.cc links against classic's libcomboreduct/libcogutil (see its
own module comment for the exact build recipe and why the *installed*
header tree alone is not enough) -- unlike tools/capture_golden.py's
rng_probe, which needs nothing beyond the standard library.

Every probe run is resource-bounded (`ulimit -v` + `timeout`): this machine
has been OOM-killed twice by unbounded probe runs (see CLAUDE.md).
"""

import json
import resource
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "table_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "table_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "table.json"
# Local, checked-in copies of the classic input files this fixture is
# captured from, so tests/comboreduct/table/test_table_io.py can call
# loadTable() on the exact bytes the fixture was captured from without
# needing $CLASSIC present at test time (see tests/conftest.py's
# classic_root fixture, which the *capture* step below still needs, but
# the test suite itself does not).
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

TABLE_DIR = CLASSIC / "tests" / "comboreduct" / "table"

# (case name, file, target_feature, timestamp_feature, ignore_features)
CASES: list[tuple[str, Path, str, str, list[str]]] = [
    ("dataset.csv", TABLE_DIR / "dataset.csv", "", "", []),
    ("dataset_force_feature.csv", TABLE_DIR / "dataset_force_feature.csv", "", "", []),
    ("sparse.data:target1", TABLE_DIR / "sparse.data", "target1", "", ["a", "b", "c", "target2"]),
    ("sparse.data:target2", TABLE_DIR / "sparse.data", "target2", "", ["a", "b", "c", "target1"]),
    ("weighted.csv", TABLE_DIR / "weighted.csv", "r", "", []),
    ("timestamp_table.csv", TABLE_DIR / "timestamp_table.csv", "length", "time", []),
    ("timestamp_table2.csv", TABLE_DIR / "timestamp_table2.csv", "target", "time", ["input1"]),
    ("timestamp_table3.csv", TABLE_DIR / "timestamp_table3.csv", "target", "time", []),
    ("trainingBoolean.dat", CLASSIC / "trainingBoolean.dat", "y", "", []),
    ("trainingBooleanXor.dat", CLASSIC / "trainingBooleanXor.dat", "y", "", []),
]

# ulimit -v 1048576 (KiB) / timeout 20 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 20


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
            str(PREFIX / "lib"),
            "-L",
            str(PREFIX / "lib" / "opencog"),
            "-Wl,-rpath," + str(PREFIX / "lib"),
            "-Wl,-rpath," + str(PREFIX / "lib" / "opencog"),
            "-lcomboreduct",
            "-lcogutil",
        ],
        check=True,
    )


def run_case(file: Path, target: str, timestamp: str, ignore: list[str]) -> dict:
    args = [str(PROBE_BIN), str(file), target, timestamp, *ignore]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    return json.loads(result.stdout)


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus of cases to (re)capture
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {}
    for name, file, target, timestamp, ignore in CASES:
        if not file.is_file():
            raise FileNotFoundError(f"{name}: fixture input not found: {file}")
        case = run_case(file, target, timestamp, ignore)
        shutil.copyfile(file, DATA_DIR / file.name)
        case["file"] = file.name
        case["target_feature"] = target
        case["timestamp_feature"] = timestamp
        case["ignore_features"] = ignore
        fixture[name] = case
        print(
            f"{name}: arity={case['arity']} rows={len(case['itable'])} "
            f"ctable={len(case['ctable'])}"
        )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_table.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/table_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={"cases": [name for name, *_ in CASES]},
        ),
    )
    print(f"wrote {FIXTURE_PATH} ({len(fixture)} cases)")
    print(f"copied {len(CASES)} input files to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
