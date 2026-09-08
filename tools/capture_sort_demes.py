"""Build and capture the sort_demes oracle fixture.

Settles the question S12 raised but left open (docs/CARRY-FORWARD.md
Sec 5b.1, `.superpowers/sdd/2026-09-08-pymetta-s12-crossover/task-report.md`
Sec 9 item 1): does `metapopulation::sort_demes`'s
`boost::sort(deme, std::greater<scored_instance<composite_score>>())`
(merging.cc:507) reorder ties the way `std::partial_sort` measurably does?

Compiles tools/cpp_probe/sort_demes_probe.cc against the installed classic
headers and libraries and saves its JSON stdout as
tests/fixtures/sort_demes.json, with a `<name>.manifest.json` sidecar
(tools/_manifest.py) recording exactly what produced it.

Mirrors tools/capture_crossover.py's shape; simpler, since this probe needs
no logging capture and no boost_program_options link (no local_moses group).

Every subprocess is resource-bounded (RLIMIT_AS + a wall-clock timeout) per
this session's standing OOM caution; the probe's largest case is 2000
`deme_inst_t` elements and the whole run is well under a second.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "sort_demes_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "sort_demes_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "sort_demes.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
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
        raise RuntimeError(
            f"sort_demes probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"sort_demes probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_sort_demes.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/sort_demes_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "fixed cases hardcoded in sort_demes_probe.cc: std::sort tie-order "
                "samples straddling libstdc++'s _S_threshold=16 boundary and at "
                "realistic MOSES deme sizes (up to 2000), plus one forward-path "
                "truncation case showing the reordering reaches keep_top_unique_"
                "candidates' surviving prefix"
            ),
            "full_sort_sizes": [15, 16, 17, 20, 32, 50, 100, 703, 2000],
            "libstdcxx_S_threshold": 16,
            "gcc_version": "16.2.1",
        },
    )


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"sort_demes probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    write_sidecar(FIXTURE_PATH, _build_manifest())


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH} and {sidecar_path(FIXTURE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
