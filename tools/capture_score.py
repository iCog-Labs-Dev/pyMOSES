"""Build tools/cpp_probe/score_probe.cc and capture its JSON output, over
the cross product of Task 5's dataset fixtures and Task 3's combo tree
corpus, into tests/fixtures/score.json.

The committed tests/fixtures/score.json is what the default test suite
actually reads (see tests/moses/scoring/test_bscores.py), so the suite
runs without a C++ toolchain or the classic checkout at all -- same
division of labor as tools/capture_table.py/table.json.

Every probe run is resource-bounded (`ulimit -v` + `timeout`): this
machine has been OOM-killed twice by unbounded probe runs (see CLAUDE.md).
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

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "score_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "score_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "score.json"
COMBO_ROUNDTRIP_PATH = REPO_ROOT / "tests" / "fixtures" / "combo_roundtrip.json"
# Local, checked-in copy of precision.csv (scoringUTest.cxxtest's own
# fixture -- not part of Task 5's own corpus, so not already copied in),
# alongside Task 5's already-copied table_data/ fixtures -- see
# tests/comboreduct/table/test_table_io.py's oracle for the same pattern.
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

# ulimit -v 1048576 (KiB) / timeout 20 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 20


class Dataset:
    """One dataset from Task 5's fixtures: everything score_probe needs to
    build a boolean-output CTable from it, plus the max argument index
    ($N) a combo tree may reference without indexing past this dataset's
    own arity.
    """

    __slots__ = ("name", "path", "target", "weighting_feature", "arity")

    def __init__(self, name: str, path: Path, target: str, weighting_feature: str, arity: int):
        self.name = name
        self.path = path
        self.target = target
        self.weighting_feature = weighting_feature
        self.arity = arity


# All four boolean-only datasets among Task 5's fixtures (every input AND
# output column 0/1-valued -- ctruth_table_bscore requires a
# boolean-output CTable, see bscores.py). Every other Task 5 fixture
# (dataset.csv, timestamp_table*.csv, sparse.data, ...) has a contin or
# enum column somewhere and so is out of scope for this boolean-only
# scorer (see scoring_base.py's module docstring).
DATASETS = [
    Dataset("trainingBoolean.dat", CLASSIC / "trainingBoolean.dat", "y", "", 3),
    Dataset("trainingBooleanXor.dat", CLASSIC / "trainingBooleanXor.dat", "y", "", 2),
    Dataset(
        "weighted.csv", CLASSIC / "tests" / "comboreduct" / "table" / "weighted.csv", "r", "wgt", 5
    ),
    Dataset("precision.csv", CLASSIC / "tests" / "moses" / "precision.csv", "target", "", 2),
]

# Purpose-built trees with much higher complexity than anything in the
# Task 3 corpus (max 3): boolean `and`/`or`/`not` themselves cost 0
# (complexity.cc:92-99), so the ONLY way to inflate a boolean tree's
# complexity is to give it more argument leaves (each leaf costs 1) --
# repeating the same in-range argument N times is a valid, if redundant,
# combo tree that the boolean interpreter evaluates exactly like a single
# reference would (`and(x, x, ..., x) == x`). Sized to 80: a fix-round
# review computed |1/3.5 - f32(1/3.5)| = 1.28e-8, so complexity_penalty =
# complexity * complexity_coef first exceeds the 1e-6 double-vs-float32
# comparison tolerance at complexity >= 79 (1e-6 / 1.28e-8 = 78.125) when
# complexity_ratio=3.5 -- these trees exist specifically to make that
# double-vs-float32 gap observable, not to model anything realistic.
_HIGH_COMPLEXITY_TREE_ARITY3 = "and(" + " ".join(["$1"] * 80) + ")"
_HIGH_COMPLEXITY_TREE_ARITY5 = "and(" + " ".join(["$1"] * 80) + ")"

# (dataset, tree, complexity_ratio) cases, to exercise
# bscore_base.set_complexity_coef/get_complexity_coef and
# composite_score.complexity_penalty with non-zero coefficients --
# every other case in this file leaves the coefficient at its class
# default (0.0, so complexity_penalty is trivially 0.0 regardless of
# complexity, and the double-vs-float32 gap this fix round exists to
# catch is invisible: 0.0 is exactly representable in both).
#
# 3.5 is MOSES's own command-line default for --complexity-ratio
# (moses_exec's problem-params.cc); 7.0 is a second, arbitrary point
# for coverage. 2.0's reciprocal (0.5) is EXACTLY representable in
# binary32 -- deliberately kept as a negative-control case (a ratio
# whose narrowing is a no-op) alongside the two that are not.
_COMPLEXITY_RATIO_CASES = [
    ("trainingBoolean.dat", "or(and($1 !$3) or())", 2.0),
    ("weighted.csv", "and($1 $2)", 2.0),
    ("trainingBoolean.dat", "or(and($1 !$3) or())", 3.5),
    ("weighted.csv", "and($1 $2)", 3.5),
    ("trainingBoolean.dat", "or(and($1 !$3) or())", 7.0),
    ("weighted.csv", "and($1 $2)", 7.0),
    ("trainingBoolean.dat", _HIGH_COMPLEXITY_TREE_ARITY3, 3.5),
    ("trainingBoolean.dat", _HIGH_COMPLEXITY_TREE_ARITY3, 7.0),
    ("weighted.csv", _HIGH_COMPLEXITY_TREE_ARITY5, 3.5),
    ("weighted.csv", _HIGH_COMPLEXITY_TREE_ARITY5, 7.0),
]


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
            "-lmoses",
            "-lcogutil",
        ],
        check=True,
    )


def run_case(
    dataset: Dataset, tree: str, complexity_ratio: float | None
) -> dict:
    args = [str(PROBE_BIN), str(dataset.path), dataset.target, tree, dataset.weighting_feature]
    if complexity_ratio is not None:
        args.append(str(complexity_ratio))
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    return json.loads(result.stdout)


def _max_arg_index(tree: str) -> int:
    """Highest $N referenced by `tree` -- used to skip trees that would
    index past a lower-arity dataset's own inputs (see interpreter.py's
    boolean_eval: an out-of-range argument index is a plain Python
    IndexError, not a graceful skip).
    """
    n = 0
    i = 0
    while True:
        i = tree.find("$", i)
        if i < 0:
            return n
        j = i + 1
        while j < len(tree) and tree[j].isdigit():
            j += 1
        n = max(n, int(tree[i + 1 : j]))
        i = j


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus of cases to (re)capture
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()

    for dataset in DATASETS:
        if not dataset.path.is_file():
            raise FileNotFoundError(f"{dataset.name}: fixture input not found: {dataset.path}")

    trees = [case["input"] for case in json.loads(COMBO_ROUNDTRIP_PATH.read_text())]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {}
    n_pairs = 0
    for dataset in DATASETS:
        shutil.copyfile(dataset.path, DATA_DIR / dataset.path.name)
        n_used = 0
        for tree in trees:
            if _max_arg_index(tree) > dataset.arity:
                continue
            n_used += 1
            case_name = f"{dataset.name}::{tree}"
            case = run_case(dataset, tree, None)
            case["file"] = dataset.path.name
            case["target_feature"] = dataset.target
            case["weighting_feature"] = dataset.weighting_feature
            case["complexity_ratio"] = None
            case["tree"] = tree
            fixture[case_name] = case
            n_pairs += 1
        print(f"{dataset.name}: {n_used}/{len(trees)} trees fit arity={dataset.arity}")

    for dataset_name, tree, ratio in _COMPLEXITY_RATIO_CASES:
        dataset = next(d for d in DATASETS if d.name == dataset_name)
        case_name = f"{dataset.name}::{tree}::ratio={ratio}"
        case = run_case(dataset, tree, ratio)
        case["file"] = dataset.path.name
        case["target_feature"] = dataset.target
        case["weighting_feature"] = dataset.weighting_feature
        case["complexity_ratio"] = ratio
        case["tree"] = tree
        fixture[case_name] = case
        n_pairs += 1

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_score.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/score_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "datasets": [d.name for d in DATASETS],
                "combo_corpus_fixture": "tests/fixtures/combo_roundtrip.json",
                "complexity_ratio_cases": len(_COMPLEXITY_RATIO_CASES),
                "n_cases": len(fixture),
                "n_probe_runs": n_pairs,
            },
        ),
    )
    distinct_trees = {case["tree"] for case in fixture.values()}
    print(
        f"wrote {FIXTURE_PATH} ({len(fixture)} cases, {n_pairs} probe runs, "
        f"{len(distinct_trees)} distinct trees, {len(DATASETS)} distinct datasets)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
