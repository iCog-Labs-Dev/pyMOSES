"""Build tools/cpp_probe/ctable_balance_probe.cc and capture its JSON output
into tests/fixtures/ctable_balance.json.

Two sections:

* `balance` -- every cell's counts before and after a real
  `CTable::balance()` call (table.cc:806), on class-imbalanced tables where
  a working balance would visibly rescale them. This is the evidence for
  the "balance() is a no-op" finding: the fixture records classic's own
  answer rather than an argument about the source.
* `discretize` -- `discretize_contin_feature` (table.cc:573) and
  `get_discrete_bin` (table.cc:583) over a fixed grid of ranges and values,
  including the exact `min + i*interval` rounding (which a `linspace`-style
  or compensated implementation would not reproduce) and the `(builtin)`
  cast of the bin ordinal.

The committed fixture is what the test suite reads, so the suite runs
without a C++ toolchain or the classic checkout. This is a developer tool,
not part of tools/ci.sh.

Every probe run is resource-bounded (`ulimit -v` + `timeout`): this machine
has been OOM-killed twice by unbounded probe runs.
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

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "ctable_balance_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "ctable_balance_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ctable_balance.json"
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

# (case name, file under tests/fixtures/table_data, target_feature)
BALANCE_CASES: list[tuple[str, str, str]] = [
    ("imbalanced_bool.csv", "imbalanced_bool.csv", "y"),
    ("trainingBoolean.dat", "trainingBoolean.dat", "y"),
    ("trainingBooleanXor.dat", "trainingBooleanXor.dat", "y"),
    ("weighted.csv", "weighted.csv", "r"),
    ("contin_regression_sample.csv", "contin_regression_sample.csv", "y"),
]

_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 60


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


def run_probe(args: list[str]) -> dict:
    result = subprocess.run(
        [str(PROBE_BIN), *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    return json.loads(result.stdout)


def main(argv: list[str]) -> int:
    del argv
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()

    balance: dict[str, dict] = {}
    for name, filename, target in BALANCE_CASES:
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"{name}: fixture input not found: {path}")
        case = run_probe([str(path), target, ""])
        case["file"] = filename
        case["target_feature"] = target
        balance[name] = case
        print(
            f"{name}: otype={case['output_type']} classes={case['n_classes']} "
            f"changed={case['balance_changed_anything']}"
        )

    discretize = run_probe(["discretize"])
    print(f"discretize: {len(discretize['cases'])} ranges")

    fixture = {"balance": balance, "discretize": discretize}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_ctable_balance.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/ctable_balance_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={"balance_cases": [name for name, *_ in BALANCE_CASES]},
        ),
    )
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
