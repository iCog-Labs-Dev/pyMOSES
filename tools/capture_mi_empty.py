"""Build and capture the EMPTY / ZERO-TOTAL mutual-information oracle fixture.

Compiles ``tools/cpp_probe/mi_empty_probe.cc`` against the classic MOSES
checkout and the installed moses-legacy prefix, runs it against two boolean
CSVs, and saves its JSON stdout as ``tests/fixtures/mi_empty.json`` with a
``mi_empty.manifest.json`` sidecar (``tools/_manifest.py``).

What it is ground truth for
---------------------------

What ``mutualInformation`` (``moses/comboreduct/table/table.h:1132`` and the
ITable/OTable overload at ``:1074``), ``mutualInformationBtwSets``
(``:1311``) and ``OTEntropy`` (``table.cc:916``) return when there is
nothing to divide by -- the case
``pymetta/feature_selection/scorers/mutual_info.py`` used to short-circuit
with ``if total == 0.0: return NaN``.

Three groups, so no single constant answer can satisfy the fixture:

* ``empty`` -- every row removed via ``CTable::remove_rows`` over the whole
  uncompressed index range, i.e. exactly what ``subsampleCTable(0.0, ct)``
  (``table.cc:899``) does, minus the RNG. Every counter is empty, so C++
  performs NO division and ``entropy()`` returns 0.0.
* ``zero_counts`` -- the same rows kept with every ``count_t`` set to 0.0:
  counters non-empty, ``total`` still 0.0, so ``div_total`` really does
  divide by zero and the IEEE-754 result flows into
  ``weighted_information``'s ``p > PROB_EPSILON`` test (``numeric.h:207``).
  This is the branch a bare ``c / total`` in Python cannot reproduce at
  all -- it raises ``ZeroDivisionError``.
* ``baseline`` -- the untouched table, so a port that returned 0.0
  unconditionally would still fail.

Why it was captured: ``--enable-fs=1 --fs-subsampling-ratio=0`` reaches the
``empty`` case from the command line, and the NaN guess made
``simple_selection``'s ``threshold <= sc`` (``simple.h:121``) reject every
feature where classic accepts them all.

Bound every run: ``( ulimit -v 2097152; timeout 300 <cmd> )``, per this
project's standing OOM caution.

Run it as::

    PYMETTA_CLASSIC_DIR=... PYMETTA_MOSES_PREFIX=... \\
        .venv/bin/python3 tools/capture_mi_empty.py
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "mi_empty_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "mi_empty_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mi_empty.json"

_TABLE_DATA = REPO_ROOT / "tests" / "fixtures" / "table_data"
# Two tables, so the `baseline` group is not uniformly zero: parity has
# zero single-feature MI with the target (every singleton is 0.0, only the
# full set reaches 1.0), while the mixed 6-input table has a spread of
# non-zero singleton values. A port returning a constant fails both.
_CSVS = [
    _TABLE_DATA / "bool_a4_par.csv",
    _TABLE_DATA / "bool_a6_mix.csv",
]

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
        "-O1",
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
        raise RuntimeError(f"mi_empty probe compilation exceeded {_COMPILE_TIMEOUT_S}s") from error
    if result.returncode != 0:
        raise RuntimeError(f"mi_empty probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_mi_empty.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/mi_empty_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "mutualInformation()/mutualInformationBtwSets() over every "
                "singleton, adjacent pair and the full feature set of two "
                "boolean CTables (4-input parity, 6-input mixed), in three states: untouched "
                "(baseline), emptied by CTable::remove_rows over the whole "
                "uncompressed index range (the subsampleCTable(0.0) case), "
                "and with every count_t zeroed but the rows kept (total == "
                "0.0 with non-empty counters). Plus OTEntropy() and the "
                "ITable/OTable mutualInformation overload on an empty "
                "table. No RNG is involved anywhere in this probe, so no "
                "seed is recorded."
            ),
            "csvs": [str(p.relative_to(REPO_ROOT)) for p in _CSVS],
        },
    )


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN), *[str(p) for p in _CSVS]],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"mi_empty probe run exceeded {_RUN_TIMEOUT_S}s") from error
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
