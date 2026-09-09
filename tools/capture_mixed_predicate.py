"""Build and capture the mixed-predicate oracle fixture.

Compiles tools/cpp_probe/mixed_predicate_probe.cc against the installed
classic headers and libraries, runs it, and saves its JSON stdout as
tests/fixtures/mixed_predicate.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this project's standing OOM caution.

The fixture has two independent halves, both produced by the one probe run:

* ``cases`` -- 40 classic ``representation`` constructions. 34 of them
  (17 shapes) have a BOOLEAN output type over contin (and mixed
  contin/Boolean) inputs: that is the mixed path -- ``build_logical``'s own
  ``is_predicate`` branch (build_knobs.cc:640) and ``logical_canonize``'s
  (build_knobs.cc:160), plus the contin branches of ``insert_typed_arg``
  (build_knobs.cc:386) and ``sample_logical_perms`` (build_knobs.cc:425).
  The other 6 (3 shapes) have a CONTIN output and a predicate nested under
  ``impulse``, reaching ``logical_canonize``'s predicate branch from
  ``rec_canonize`` (build_knobs.cc:903-905) instead -- and pinning what
  classic does NOT do there: the contin path never calls ``build_logical``,
  so those cases carry contin knobs and ZERO disc knobs.
  Every shape is run at both ``perm_ratio=-1`` (no pairs; no RNG draws) and
  ``perm_ratio=0`` (classic's own real default, which runs the pair loop --
  the ONLY caller of ``insert_typed_arg``), with ``randGen()`` reseeded to
  1 immediately before construction.

* ``logical_reduce`` -- 40 rows: 20 predicate-bearing combo trees reduced
  at efforts 2 and 3 with classic's own real default (empty) ``ignore_ops``.
  This is the oracle for ``simplify_predicates``' contin/mixed tail
  (logical_rules.cc:76-81), which is the first element of ``p_medium`` and
  therefore runs on both efforts. ``logical-reductor``
  (moses/comboreduct/main/) cannot serve here: its ``complete_truth_table``
  equivalence check aborts on ``0<`` (measured live, exit 134).
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "mixed_predicate_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "mixed_predicate_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mixed_predicate.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300

_CASE_KEYS = {
    "name",
    "otype",
    "source",
    "itypes",
    "ignored_arguments",
    "ignore_greater_than_zero",
    "perm_ratio",
    "seed",
    "decorated_exemplar",
    "prototype",
    "disc_knobs",
    "contin_knobs",
}
_DISC_KEYS = {
    "order",
    "multy",
    "text",
    "in_exemplar",
    "location_preorder",
    "location_subtree",
}
_CONTIN_KEYS = {
    "order",
    "mean",
    "step_size",
    "expansion",
    "depth",
    "text",
    "location_preorder",
    "location_subtree",
}
_REDUCE_KEYS = {"input", "effort", "ok", "reduced", "size_before", "size_after"}

_EXPECTED_CASE_COUNT = 40
_EXPECTED_REDUCE_COUNT = 40


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
        raise RuntimeError(f"{label} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _require_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"mixed predicate probe returned invalid {label} schema")
    return value


def _validate_fixture(fixture: object) -> None:
    fixture = _require_keys(
        fixture,
        {
            "stepsize",
            "expansion",
            "depth",
            "linear_contin",
            "reduct_effort",
            "cases",
            "logical_reduce",
        },
        "top-level",
    )
    if (fixture["stepsize"], fixture["expansion"], fixture["depth"]) != (1.0, 2.0, 5):
        raise ValueError(
            "mixed predicate probe did not use classic's own representation.cc "
            "statics (stepsize=1.0, expansion=2.0, depth=5)"
        )
    if fixture["linear_contin"] is not False:
        raise ValueError(
            "mixed predicate probe must run at linear_contin=False, the real "
            "moses CLI default (--linear-regression, problem-params.cc:292)"
        )

    cases = fixture["cases"]
    if not isinstance(cases, list) or len(cases) != _EXPECTED_CASE_COUNT:
        raise ValueError("mixed predicate probe returned invalid case count")
    predicate_cases = 0
    pair_cases = 0
    contin_output_cases = 0
    for case in cases:
        case = _require_keys(case, _CASE_KEYS, "case")
        if case["otype"] == "contin":
            contin_output_cases += 1
            if case["disc_knobs"]:
                raise ValueError(
                    "a contin-output exemplar must build zero disc knobs -- "
                    "classic's contin path never calls build_logical"
                )
            if not case["contin_knobs"]:
                raise ValueError("a contin-output exemplar must build contin knobs")
        elif case["otype"] != "boolean":
            raise ValueError(f"unexpected output type {case['otype']!r}")
        for order, knob in enumerate(case["disc_knobs"]):
            knob = _require_keys(knob, _DISC_KEYS, "disc knob")
            if knob["order"] != order:
                raise ValueError("mixed predicate probe returned invalid disc knob order")
        for order, knob in enumerate(case["contin_knobs"]):
            knob = _require_keys(knob, _CONTIN_KEYS, "contin knob")
            if knob["order"] != order:
                raise ValueError("mixed predicate probe returned invalid contin knob order")
        # The whole point of this fixture: a Boolean-output representation
        # that also carries CONTIN knobs, i.e. build_logical's predicate
        # branch actually ran contin_canonize/build_contin.
        if case["contin_knobs"]:
            predicate_cases += 1
        if case["perm_ratio"] == 0:
            pair_cases += 1
    if predicate_cases == 0:
        raise ValueError(
            "no case built a contin knob -- the fixture cannot discriminate "
            "build_logical's own is_predicate branch from a no-op"
        )
    if pair_cases * 2 != _EXPECTED_CASE_COUNT:
        raise ValueError("mixed predicate probe must run every shape at both perm ratios")
    if contin_output_cases != 6:
        raise ValueError(
            "mixed predicate probe must carry the 6 contin-output predicate-under-impulse cases"
        )

    rows = fixture["logical_reduce"]
    if not isinstance(rows, list) or len(rows) != _EXPECTED_REDUCE_COUNT:
        raise ValueError("mixed predicate probe returned invalid logical_reduce row count")
    changed = 0
    for row in rows:
        row = _require_keys(row, _REDUCE_KEYS, "logical_reduce row")
        if row["ok"] is not True:
            raise ValueError(f"classic failed to reduce {row['input']!r}")
        if row["effort"] not in (2, 3):
            raise ValueError("logical_reduce rows must be captured at effort 2 and 3")
        if row["reduced"].strip() != row["input"]:
            changed += 1
    if changed < _EXPECTED_REDUCE_COUNT // 2:
        raise ValueError(
            "fewer than half the logical_reduce rows actually changed -- a corpus "
            "of no-ops cannot discriminate simplify_predicates' tail from its stub"
        )


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
        "mixed predicate probe compilation",
    )


def capture_fixture() -> None:
    result = _run([str(PROBE_BIN)], _RUN_TIMEOUT_S, "mixed predicate probe")
    fixture = json.loads(result.stdout)
    _validate_fixture(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_mixed_predicate.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/mixed_predicate_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "17 fixed Boolean-output exemplar shapes over contin "
                    "(and mixed contin/Boolean) inputs -- bare `and`, bare "
                    "and negated root predicates, predicates as junctor "
                    "children, predicates over +/*//sin/constants, mixed "
                    "inputs, an ignored argument and an ignored "
                    "greater_than_zero -- plus 3 contin-output shapes with a "
                    "predicate nested under impulse; each run at perm_ratio "
                    "-1 and 0 with randGen() reseeded to 1; plus 20 "
                    "predicate-bearing combo trees reduced by logical_reduce "
                    "at efforts 2 and 3 with empty ignore_ops. No external "
                    "dataset"
                ),
                "case_count": _EXPECTED_CASE_COUNT,
                "logical_reduce_row_count": _EXPECTED_REDUCE_COUNT,
                "rng_seed": 1,
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_mixed_predicate.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
