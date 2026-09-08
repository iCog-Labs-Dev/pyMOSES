"""Build and capture the bounded classic Boolean candidate oracle."""

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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "candidate_roundtrip_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "candidate_roundtrip_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "representation.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 120
_EXPECTED_CASES = (
    ("constant-true", "true", 0, []),
    ("bare-argument", "$1", 1, []),
    ("small-and", "and($1 $2)", 2, []),
    ("small-or-negated", "or($1 !$2)", 2, []),
    ("nested-three", "and(or($1 $2) !$3)", 3, []),
    ("ignored-argument", "and($1 $2)", 2, [2]),
    ("and-wraps-or", "and($1 or($2 $3))", 3, []),
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
        raise ValueError(f"representation probe returned invalid {label} schema")
    return value


def _validate_fixture(fixture: object) -> None:
    fixture = _require_keys(fixture, {"perm_ratio", "cases"}, "top-level")
    if fixture["perm_ratio"] != -1:
        raise ValueError("representation probe did not use perm_ratio=-1")
    cases = fixture["cases"]
    if not isinstance(cases, list) or len(cases) != len(_EXPECTED_CASES):
        raise ValueError("representation probe returned invalid case count")
    case_keys = {
        "name",
        "source",
        "arity",
        "ignored_arguments",
        "decorated_exemplar",
        "packed_width",
        "raw_size",
        "exhaustive",
        "knobs",
        "rows",
    }
    knob_keys = {"order", "multy", "text", "location_preorder", "raw_idx"}
    row_keys = {"raw", "packed", "unreduced", "reduced", "unreduced_tt", "reduced_tt"}
    for case, expected in zip(cases, _EXPECTED_CASES, strict=True):
        case = _require_keys(case, case_keys, "case")
        actual = (
            case["name"],
            case["source"],
            case["arity"],
            case["ignored_arguments"],
        )
        if actual != expected or not case["exhaustive"]:
            raise ValueError(f"representation probe case mismatch: {actual!r}")
        if case["raw_size"] != len(case["knobs"]):
            raise ValueError("representation raw size does not match knob count")
        combinations = 1
        for order, knob in enumerate(case["knobs"]):
            knob = _require_keys(knob, knob_keys, "knob")
            if knob["order"] != order or knob["raw_idx"] != order:
                raise ValueError("representation probe returned invalid knob indexes")
            combinations *= knob["multy"]
        if len(case["rows"]) != combinations:
            raise ValueError("representation probe did not exhaust legal settings")
        for row in case["rows"]:
            row = _require_keys(row, row_keys, "row")
            if len(row["raw"]) != case["raw_size"]:
                raise ValueError("representation row has invalid raw width")
            if len(row["packed"]) != case["packed_width"]:
                raise ValueError("representation row has invalid packed width")
            for key in ("unreduced_tt", "reduced_tt"):
                truth = row[key]
                expected_keys = (
                    {"computed", "bits"}
                    if truth.get("computed")
                    else {"computed", "error"}
                )
                _require_keys(truth, expected_keys, "truth-table result")
                if truth["computed"] and len(truth["bits"]) != 1 << case["arity"]:
                    raise ValueError("representation truth table has invalid arity")


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
        "representation probe compilation",
    )


def capture_fixture() -> None:
    result = _run([str(PROBE_BIN)], _RUN_TIMEOUT_S, "representation probe")
    fixture = json.loads(result.stdout)
    _validate_fixture(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_representation.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/candidate_roundtrip_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed Boolean candidate/representation round-trip cases "
                    "hardcoded in candidate_roundtrip_probe.cc; no external "
                    "dataset or seed"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_representation.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
