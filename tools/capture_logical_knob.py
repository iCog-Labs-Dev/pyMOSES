"""Build and capture the bounded classic logical-knob oracle."""

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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "logical_knob_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "logical_knob_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "logical_knob.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 20


def _require_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"logical-knob probe returned invalid {label} schema")
    return value


def _require_list(value: object, count: int, label: str) -> list:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"logical-knob probe returned invalid {label} count")
    return value


def _validate_fixture(fixture: object) -> None:
    fixture = _require_keys(
        fixture,
        {"active", "transitions", "append", "disallow", "copy", "clear"},
        "top-level",
    )
    active = _require_keys(
        fixture["active"], {"base", "prototype", "knobs", "turns"}, "active"
    )
    knobs = _require_list(active["knobs"], 3, "active knob")
    turns = _require_list(active["turns"], 9, "active turn")
    for row in turns:
        _require_keys(row, {"knob", "idx", "tree", "reduced"}, "active turn row")
    if (
        active["base"] != "and($1 not($2))"
        or knobs != ["[$1 nil !$1]", "[!$2 nil $2]", "[nil $3 !$3]"]
        or [(row["knob"], row["idx"]) for row in turns]
        != [(knob, idx) for knob in range(1, 4) for idx in range(3)]
    ):
        raise ValueError("logical-knob probe returned invalid active inputs")
    pinned = [
        ("existing-positive", "and($1 $9)", 1),
        ("existing-negated", "and(not($1) $9)", 1),
        ("missing", "and($1 $9)", 2),
    ]
    transitions = _require_list(fixture["transitions"], 3, "transition")
    append = _require_list(fixture["append"], 3, "append")
    for row in transitions:
        _require_keys(
            row,
            {
                "name",
                "base",
                "arg",
                "initial_tree",
                "knob",
                "in_exemplar",
                "multiplicity",
                "spec_multy",
                "states",
            },
            "transition row",
        )
        states = _require_list(row["states"], 7, "transition state")
        for state in states:
            _require_keys(
                state,
                {"idx", "tree", "reduced", "complexity"},
                "transition state row",
            )
    if [(row["name"], row["base"], row["arg"]) for row in transitions] != pinned:
        raise ValueError("logical-knob probe returned invalid transition inputs")
    for row in append:
        _require_keys(row, {"name", "base", "arg", "rows"}, "append row")
        rows = _require_list(row["rows"], 6, "append result")
        for result in rows:
            _require_keys(
                result,
                {
                    "idx",
                    "nested",
                    "candidate",
                    "source_valid",
                    "source_is_exemplar_end",
                    "source_is_candidate_end",
                    "source",
                    "parent",
                },
                "append result row",
            )
    if [(row["name"], row["base"], row["arg"]) for row in append] != pinned:
        raise ValueError("logical-knob probe returned invalid append inputs")
    if any(
        [state["idx"] for state in row["states"]] != [0, 1, 2, 0, 2, 1, 0]
        for row in transitions
    ):
        raise ValueError("logical-knob probe returned incomplete transition coverage")
    expected_append = [(idx, nested) for idx in range(3) for nested in (False, True)]
    if any(
        [(item["idx"], item["nested"]) for item in row["rows"]] != expected_append
        for row in append
    ):
        raise ValueError("logical-knob probe returned incomplete append coverage")
    expected_disallow = [
        (name, "and($1 $9)", arg_idx, setting)
        for name, arg_idx in (("missing", 2), ("existing-positive", 1))
        for setting in range(3)
    ]
    disallow = _require_list(fixture["disallow"], 6, "disallow")
    for row in disallow:
        _require_keys(
            row,
            {"name", "base", "arg", "setting", "multiplicity", "spec_multy", "knob"},
            "disallow row",
        )
    if [
        (row["name"], row["base"], row["arg"], row["setting"])
        for row in disallow
    ] != expected_disallow:
        raise ValueError("logical-knob probe returned incomplete disallow coverage")

    copies = _require_list(fixture["copy"], 2, "copy")
    for row in copies:
        _require_keys(
            row,
            {
                "name",
                "base",
                "arg",
                "source_turn",
                "disallowed",
                "destination_mode",
                "source_tree",
                "destination_before",
                "destination_after",
                "knob",
                "in_exemplar",
                "multiplicity",
                "spec_multy",
                "states",
            },
            "copy row",
        )
        states = _require_list(row["states"], 2, "copy state")
        for state in states:
            _require_keys(
                state,
                {"idx", "tree", "reduced", "complexity"},
                "copy state row",
            )
    expected_copies = [
        ("existing-absent-disallowed-negated", "and($1 $9)", 1, 1, 2, "source-copy"),
        ("missing-negated-disallowed-present", "and($1 $9)", 2, 2, 1, "fresh-base"),
    ]
    if [
        (
            row["name"],
            row["base"],
            row["arg"],
            row["source_turn"],
            row["disallowed"],
            row["destination_mode"],
        )
        for row in copies
    ] != expected_copies or any(
        [state["idx"] for state in row["states"]] != [0, 1] for row in copies
    ):
        raise ValueError("logical-knob probe returned invalid copy inputs")

    clear = _require_list(fixture["clear"], 5, "clear")
    for row in clear:
        _require_keys(
            row,
            {
                "name",
                "base",
                "arg",
                "turn",
                "disallowed",
                "before",
                "after",
                "retained",
                "after_knob",
            },
            "clear row",
        )
    expected_clear = [
        ("existing-negated", "and(not($1) $9)", 1, 2, -1),
        ("existing-disallowed-absent", "and($1 $9)", 1, -1, 0),
        ("missing-absent", "and($1 $9)", 2, 0, -1),
        ("missing-present", "and($1 $9)", 2, 1, -1),
        ("missing-negated-disallowed-absent", "and($1 $9)", 2, 1, 0),
    ]
    if [
        (row["name"], row["base"], row["arg"], row["turn"], row["disallowed"])
        for row in clear
    ] != expected_clear:
        raise ValueError("logical-knob probe returned invalid clear inputs")


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


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    _run(
        [
            "g++", "-O2", "-std=c++17", "-I", str(CLASSIC),
            "-I", str(PREFIX / "include"), "-o", str(PROBE_BIN),
            str(PROBE_SRC), "-L", str(lib), "-L", str(lib_opencog),
            "-Wl,-rpath," + str(lib), "-Wl,-rpath," + str(lib_opencog),
            "-lmoses", "-lcomboreduct", "-lcogutil",
        ],
        _COMPILE_TIMEOUT_S,
        "logical-knob probe compilation",
    )


def capture_fixture() -> None:
    result = _run([str(PROBE_BIN)], _RUN_TIMEOUT_S, "logical-knob probe")
    fixture = json.loads(result.stdout)
    _validate_fixture(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_logical_knob.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/logical_knob_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "fixed logical-knob active/transitions/append/disallow/copy/"
                    "clear test cases hardcoded in logical_knob_probe.cc; no "
                    "external dataset or seed"
                )
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_logical_knob.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
