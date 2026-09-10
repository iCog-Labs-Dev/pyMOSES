"""Build and capture the feature-selection diversity/subsampling oracle.

Compiles tools/cpp_probe/fs_diversity_probe.cc against the installed
classic headers and libraries, runs it against three boolean fixture
tables, and saves its JSON stdout as tests/fixtures/fs_diversity.json,
with an fs_diversity.manifest.json sidecar (tools/_manifest.py).

Ground truth for the two gates
``pymetta/moses/deme/feature_selector.py`` used to raise on:

* ``--fs-diversity-pressure > 0`` -- ``feature_selector::
  rank_feature_sets`` (feature_selector.cc:343), ``feature_selector::mi``
  (cc:434) and the ranked branch of ``select_top_feature_sets``
  (cc:235-245);
* ``--fs-subsampling-ratio < 1.0`` -- ``subsampleCTable``
  (moses/comboreduct/table/table.cc:899) and the ``CTable::remove_rows``
  it drives (table.cc:646), ported at
  ``pymetta/feature_selection/ctable_subsample.py``.

Mirrors tools/capture_feature_selection.py's shape, including the
``-lfeature_selection`` link (feature_selector.cc, in libmoses, calls into
libfeature_selection).

Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) per this
project's standing OOM caution.
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, sidecar_path, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "fs_diversity_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "fs_diversity_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "fs_diversity.json"
TABLE_DATA = REPO_ROOT / "tests" / "fixtures" / "table_data"
UNSORTED_CSV = TABLE_DATA / "ctable_unsorted.csv"
BIG_CSV = TABLE_DATA / "ctable_unsorted_big.csv"
BOOLEAN_CSV = TABLE_DATA / "trainingBoolean.dat"
A5_CSV = TABLE_DATA / "bool_a5_maj.csv"
SKEW_CSV = TABLE_DATA / "fs_mi_skewed.csv"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 120


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
        "-lfeature_selection",
        "-lmoses",
        "-lcomboreduct",
        "-lcogutil",
        "-lboost_program_options",
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
            f"fs_diversity probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"fs_diversity probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_fs_diversity.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/fs_diversity_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "CTable::remove_rows() over 12 index sets on a compressed "
                "table deliberately built so std::map order differs from "
                "insertion order for both the rows and two of the four "
                "rows' TimedCounters, plus 10 more on a 4-input/30-row table "
                "whose index sets contain elements >= 8 (so a Python set's "
                "own iteration order is not ascending there); "
                "lazy_random_selector draw SEQUENCES "
                "for 5 (u, seed, n) triples; subsampleCTable() for 10 "
                "(ratio, seed) pairs, 8 more at ratios where the binary32 "
                "narrowing of the `float ratio` parameter changes nremove, "
                "and 10 on the 30-row table; rank_feature_sets() plus "
                "select_top_feature_sets() for "
                "33 hand-built feature_set_pops carrying deliberate ties "
                "(equal scores, duplicated sets, and the fully symmetric "
                "5-input majority table where every same-size feature set "
                "has identical mutual information), across jaccard/mi, four "
                "diversity_interaction values and several n_demes; and "
                "feature_selector::mi() at full double precision for 22 "
                "(table, fs_l, fs_r, diversity_interaction) cases -- most on "
                "fs_mi_skewed.csv, which is NOT a full truth-table enumeration "
                "and therefore has irrational mutual informations where the "
                "other tables have whole numbers of bits; and "
                "feature_selector::operator()() end to end with "
                "diversity_pressure > 0 and/or subsampling_ratio < 1. "
                "Seeds are set with randGen().seed(...) immediately before "
                "each drawing call. NOT captured: the 'hc' algorithm, the "
                "only one whose feature_set_pop can exceed one entry -- "
                "classic's own 'hc' feature-selection path segfaults "
                "unconditionally (measured again here, rc 139; see "
                "tools/cpp_probe/fs_standalone_probe.cc)."
            ),
            "unsorted_csv": "tests/fixtures/table_data/ctable_unsorted.csv",
            "big_csv": "tests/fixtures/table_data/ctable_unsorted_big.csv",
            "boolean_csv": "tests/fixtures/table_data/trainingBoolean.dat",
            "a5_csv": "tests/fixtures/table_data/bool_a5_maj.csv",
            "skew_csv": "tests/fixtures/table_data/fs_mi_skewed.csv",
        },
    )


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [
                str(PROBE_BIN),
                str(UNSORTED_CSV),
                str(BIG_CSV),
                str(BOOLEAN_CSV),
                str(A5_CSV),
                str(SKEW_CSV),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"fs_diversity probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_sidecar(FIXTURE_PATH, _build_manifest())


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH} and {sidecar_path(FIXTURE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
