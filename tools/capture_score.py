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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "score_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "score_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "score.json"
COMBO_ROUNDTRIP_PATH = REPO_ROOT / "tests" / "fixtures" / "combo_roundtrip.json"
# Local, checked-in copy of precision.csv (scoringUTest.cxxtest's own
# fixture -- not part of Task 5's own corpus, so not already copied in),
# alongside Task 5's already-copied table_data/ fixtures -- see
# tests/comboreduct/table/test_table_io.py's oracle for the same pattern.
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"

CLASSIC = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses")
PREFIX = Path.home() / "opt" / "moses-legacy"

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

# One extra (dataset, tree, complexity_ratio) case per dataset, to
# exercise bscore_base.set_complexity_coef/get_complexity_coef and
# composite_score.complexity_penalty with a non-zero coefficient too --
# every other case below leaves the coefficient at its class default
# (0.0, so complexity_penalty is trivially 0.0 regardless of complexity).
_COMPLEXITY_RATIO_CASES = [
    ("trainingBoolean.dat", "or(and($1 !$3) or())", 2.0),
    ("weighted.csv", "and($1 $2)", 2.0),
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
    print(
        f"wrote {FIXTURE_PATH} ({len(fixture)} cases, {n_pairs} probe runs, "
        f"{len(trees)} distinct trees, {len(DATASETS)} distinct datasets)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
