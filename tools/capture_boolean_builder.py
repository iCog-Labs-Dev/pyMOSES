"""Build and capture the bounded classic deterministic Boolean-builder oracle."""

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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "representation_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "representation_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "boolean_builder.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 30

_EXPECTED_CASES = (
    ("constant-true", "true", 0, [], 0, 0),
    ("bare-argument", "$1", 1, [], 10, 1),
    ("small-and", "and($1 $2)", 2, [], 24, 2),
    ("small-or-negated", "or($1 !$2)", 2, [], 24, 2),
    ("nested-three", "and(or($1 $2) !$3)", 3, [], 54, 7),
    ("ignored-argument", "and($1 $2)", 2, [2], 12, 1),
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
        raise ValueError(f"Boolean-builder probe returned invalid {label} schema")
    return value


def _validate_fixture(fixture: object) -> None:
    fixture = _require_keys(fixture, {"perm_ratio", "cases"}, "top-level")
    if fixture["perm_ratio"] != -1:
        raise ValueError("Boolean-builder probe did not use perm_ratio=-1")
    cases = fixture["cases"]
    if not isinstance(cases, list) or len(cases) != len(_EXPECTED_CASES):
        raise ValueError("Boolean-builder probe returned invalid case count")
    expected_keys = {
        "name",
        "source",
        "arity",
        "ignored_arguments",
        "decorated_exemplar",
        "prototype",
        "probe_calls",
        "knobs",
    }
    knob_keys = {
        "order",
        "multy",
        "text",
        "in_exemplar",
        "location_preorder",
        "location_subtree",
    }
    for case, expected in zip(cases, _EXPECTED_CASES, strict=True):
        case = _require_keys(case, expected_keys, "case")
        actual = (
            case["name"],
            case["source"],
            case["arity"],
            case["ignored_arguments"],
            len(case["probe_calls"]),
            len(case["knobs"]),
        )
        if actual != expected:
            raise ValueError(f"Boolean-builder probe case mismatch: {actual!r}")
        if not all(
            isinstance(case[key], str)
            for key in ("name", "source", "decorated_exemplar", "prototype")
        ):
            raise ValueError("Boolean-builder probe returned invalid case strings")
        if not isinstance(case["probe_calls"], list) or not isinstance(case["knobs"], list):
            raise ValueError("Boolean-builder probe returned invalid event or knob list")
        for event in case["probe_calls"]:
            event = _require_keys(event, {"before", "after"}, "probe event")
            if not isinstance(event["before"], str) or not isinstance(event["after"], str):
                raise ValueError("Boolean-builder probe returned invalid probe event")
        for order, knob in enumerate(case["knobs"]):
            knob = _require_keys(knob, knob_keys, "knob")
            if (
                knob["order"] != order
                or not isinstance(knob["multy"], int)
                or knob["multy"] not in (2, 3)
                or not isinstance(knob["text"], str)
                or not isinstance(knob["in_exemplar"], bool)
                or not isinstance(knob["location_preorder"], int)
                or not isinstance(knob["location_subtree"], str)
            ):
                raise ValueError("Boolean-builder probe returned invalid knob row")
    if sum(len(case["probe_calls"]) for case in cases) != 124:
        raise ValueError("Boolean-builder oracle must contain exactly 124 probe calls")
    if sum(len(case["knobs"]) for case in cases) != 13:
        raise ValueError("Boolean-builder oracle must contain exactly 13 retained knobs")
    if not any(knob["multy"] == 2 for case in cases for knob in case["knobs"]):
        raise ValueError("Boolean-builder oracle did not capture a disallowed setting")


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
        "Boolean-builder probe compilation",
    )


def capture_fixture() -> None:
    result = _run([str(PROBE_BIN)], _RUN_TIMEOUT_S, "Boolean-builder probe")
    fixture = json.loads(result.stdout)
    _validate_fixture(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_boolean_builder.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/representation_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed deterministic Boolean-builder exemplars/arities hardcoded "
                    "in representation_probe.cc; no external dataset or seed"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_boolean_builder.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
