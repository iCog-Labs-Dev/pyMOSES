"""Build and capture the bounded classic contin-output representation oracle
(S14 Task 2's exit criterion: "the knobs built for a contin exemplar match
C++ in count, order and spec").

Unlike ``capture_representation.py``'s Boolean fixture (``perm_ratio=-1``,
chosen so pair sampling never fires), every case here is captured at BOTH
``linear_contin=True`` (the constructor's own literal default,
representation.h:62) AND ``linear_contin=False`` (the real ``moses`` CLI's
actual default, ``--linear-regression``, problem-params.cc:292) -- see
CARRY-FORWARD.md Sec 5.4: an oracle configured only by the port's own
constructor defaults cannot detect the port's defaults being wrong.

Every probe run is resource-bounded (`ulimit -v` + `timeout`): CLAUDE.md
requires this project-wide, and `rec_canonize`'s own non-linear expansion
(the ``linear_contin=False`` cases) is specifically named there as a real
combinatorial-explosion hazard. The 8 exemplar shapes below were chosen and
measured to stay in the tens of knobs, not hundreds -- see the S14 Task 2
report for the full per-case counts.
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

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_representation_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_representation_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_representation.json"

# ulimit -v 2097152 (KiB) / timeout 300 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 60

_EXPECTED_NAMES = tuple(
    f"{base}-{mode}"
    for base in (
        "bare-argument",
        "sum-two-args",
        "sin-of-arg",
        "impulse-of-boolean-arg",
        "mixed-sum",
        "div-two-args",
        "times-arg-and-constant",
        "bare-childless-plus",
    )
    for mode in ("linear", "nonlinear")
)
# Measured against the compiled classic library (see the S14 Task 2 report):
# a bounded, non-vacuous corpus check that also pins the combinatorial-
# explosion hazard stays bounded, not merely "some positive number of knobs".
_EXPECTED_CONTIN_COUNTS = (
    19, 26,  # bare-argument: linear, nonlinear
    28, 50,  # sum-two-args
    19, 42,  # sin-of-arg
    19, 26,  # impulse-of-boolean-arg
    28, 50,  # mixed-sum
    28, 50,  # div-two-args
    19, 26,  # times-arg-and-constant
    26, 26,  # bare-childless-plus (linear/nonlinear coincide -- see report)
)


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def _run(command: list[str], timeout: int, label: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_bound_resources,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} exceeded {timeout}s") from error
    if result.returncode:
        raise RuntimeError(
            f"{label} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _require_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"contin representation probe returned invalid {label} schema")
    return value


def _validate_fixture(fixture: object) -> None:
    fixture = _require_keys(fixture, {"stepsize", "expansion", "depth", "cases"}, "top-level")
    if (fixture["stepsize"], fixture["expansion"], fixture["depth"]) != (1.0, 2.0, 5):
        raise ValueError(
            "contin representation probe did not use classic's own "
            "representation.cc statics (stepsize=1.0, expansion=2.0, depth=5)"
        )
    cases = fixture["cases"]
    if not isinstance(cases, list) or len(cases) != len(_EXPECTED_NAMES):
        raise ValueError("contin representation probe returned invalid case count")
    case_keys = {
        "name",
        "source",
        "itypes",
        "linear_contin",
        "decorated_exemplar",
        "prototype",
        "disc_knob_count",
        "contin_knobs",
    }
    knob_keys = {
        "order",
        "mean",
        "step_size",
        "expansion",
        "depth",
        "text",
        "location_preorder",
        "location_subtree",
    }
    for case, expected_name, expected_count in zip(
        cases, _EXPECTED_NAMES, _EXPECTED_CONTIN_COUNTS, strict=True
    ):
        case = _require_keys(case, case_keys, "case")
        if case["name"] != expected_name:
            raise ValueError(f"contin representation probe case mismatch: {case['name']!r}")
        if case["disc_knob_count"] != 0:
            raise ValueError("a pure contin-output exemplar must build zero disc knobs")
        if len(case["contin_knobs"]) != expected_count:
            raise ValueError(
                f"{expected_name}: expected {expected_count} contin knobs, "
                f"got {len(case['contin_knobs'])}"
            )
        for order, knob in enumerate(case["contin_knobs"]):
            knob = _require_keys(knob, knob_keys, "knob")
            if knob["order"] != order:
                raise ValueError("contin representation probe returned invalid knob order")


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    _run(
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
            str(lib),
            "-L",
            str(lib_opencog),
            "-Wl,-rpath," + str(lib),
            "-Wl,-rpath," + str(lib_opencog),
            "-lmoses",
            "-lcomboreduct",
            "-lcogutil",
        ],
        _COMPILE_TIMEOUT_S,
        "contin representation probe compilation",
    )


def capture_fixture() -> None:
    result = _run([str(PROBE_BIN)], _RUN_TIMEOUT_S, "contin representation probe")
    fixture = json.loads(result.stdout)
    _validate_fixture(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_representation.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_representation_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "8 fixed small contin-output exemplar shapes (bare "
                    "argument, sum, sin, impulse-of-boolean, mixed contin/"
                    "boolean inputs, div, times-with-constant, bare "
                    "childless plus), each run at both linear_contin=true "
                    "and linear_contin=false; no RNG or external dataset"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_contin_representation.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
