"""Build and capture the CONTINUOUS-branch mutual-information oracle fixture.

Compiles ``tools/cpp_probe/mi_contin_probe.cc`` against the classic MOSES
checkout and the installed moses-legacy prefix, runs it against five CSV
fixtures, and saves its JSON stdout as ``tests/fixtures/mi_contin.json``
with a ``mi_contin.manifest.json`` sidecar (``tools/_manifest.py``).

What it is ground truth for
---------------------------

``pymoses/feature_selection/scorers/mutual_info.py``'s two formerly-gated
continuous branches of ``mutualInformation(const CTable&, const
FeatureSet&)`` (``moses/comboreduct/table/table.h:1132``), and the new
``pymoses/feature_selection/scorers/kld.py``:

* ``discretize_contin_feature`` / ``get_discrete_bin``
  (``table.cc:573`` / ``:583``), including the two degenerate interval
  lists the ``min = 100000.0`` / ``max = 0.0`` seeds produce
  (``table.h:1174-1175``) and values landing exactly on a bin boundary.
* ``KLD``/``KLDS`` (``opencog/util/KLD.h:202`` / ``:59``), through both the
  ``KLD(p,q)`` helper and ``KLDS``'s own public API with a hand-built
  ``q_counter`` -- the only route that makes ``boost::accumulate(q_counter
  | map_values, 0)``'s ``int`` init observable.
* ``mutualInformation`` on boolean-input/contin-output tables, on
  contin-input tables, and on a contin-input table whose ``count_t``
  counts have been scaled off the integers so that ``unsigned flt_count =
  val_pair.second;`` (``table.h:1266``) truncates.
* ``mutualInformationBtwSets`` (``table.h:1311``) on the same tables --
  which is how the port learned that function has NO contin branch and
  needs no input-type guard.

Bound every run: ``( ulimit -v 2097152; timeout 300 <cmd> )``, per this
project's standing OOM caution.

Run it as::

    PYMOSES_CLASSIC_DIR=... PYMOSES_MOSES_PREFIX=... \\
        .venv/bin/python3 tools/capture_mi_contin.py
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "mi_contin_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "mi_contin_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mi_contin.json"

_TABLE_DATA = REPO_ROOT / "tests" / "fixtures" / "table_data"
_CSVS = [
    _TABLE_DATA / "mi_contin_out_bins.csv",
    _TABLE_DATA / "mi_contin_out_neg.csv",
    _TABLE_DATA / "mi_contin_out_big.csv",
    _TABLE_DATA / "mi_contin_in_ties.csv",
    _TABLE_DATA / "contin_a4.csv",
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
        raise RuntimeError(f"mi_contin probe compilation exceeded {_COMPILE_TIMEOUT_S}s") from error
    if result.returncode != 0:
        raise RuntimeError(f"mi_contin probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_mi_contin.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/mi_contin_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "discretize_contin_feature/get_discrete_bin over five "
                "(min,max) pairs including both degenerate seeds "
                "(min stuck at 100000.0, max stuck at 0.0), a constant "
                "column, and max<min; KLD(p,q) over ten hand-built vector "
                "pairs (ties in p, ties in q, q wholly below/above p, "
                "unequal sizes, negatives, a scrambled restatement of an "
                "earlier case, a wide dynamic range); KLDS<contin_t> driven "
                "directly with hand-built q_counters, two of them carrying "
                "non-integral counts so boost::accumulate's `int` init "
                "truncates; mutualInformation() for every feature subset of "
                "three boolean-input/contin-output tables; "
                "mutualInformation() per single feature of two "
                "contin-input tables; the same after scaling every count_t "
                "off the integers (so `unsigned flt_count` truncates); and "
                "mutualInformationBtwSets() on all of them. No RNG is "
                "involved anywhere in this probe, so no seed is recorded."
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
        raise RuntimeError(f"mi_contin probe run exceeded {_RUN_TIMEOUT_S}s") from error
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
