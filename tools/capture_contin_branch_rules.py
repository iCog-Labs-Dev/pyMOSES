"""Build and capture the S14 Task 4 contin_rules.cc/branch_rules.cc
per-rule oracle fixture.

Runs `tools/cpp_probe/contin_rule_probe.cc` once per (rule, tree) case --
that probe applies exactly one named rule, pre-order (`downwards`), to a
freshly-parsed combo tree and prints the reduced tree text (or
`EXCEPTION: <what>` if the rule raised, e.g. an `OC_ASSERT`). Each case
becomes one entry of `tests/fixtures/contin_branch_rules.json`, read by
`tests/comboreduct/reduct/test_contin_rules_oracle.py`.
"""

import json
import resource
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_rule_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_rule_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_branch_rules.json"
_MEM_LIMIT_KIB = 1048576
_MEM_LIMIT_BYTES = _MEM_LIMIT_KIB * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 20

# (rule_name, input tree text) -- see the corresponding class docstrings
# in pymetta/comboreduct/reduct/{contin_rules,branch_rules}.py for what
# each case is meant to exercise.
CASES: list[tuple[str, str]] = [
    ("reduce_plus_zero", "+(0.0 $1 0.0)"),
    ("reduce_plus_zero", "+(0.0 0.0)"),
    ("reduce_times_one_zero", "*(1.0 $1 1.0)"),
    ("reduce_times_one_zero", "*($1 0.0 $2)"),
    ("reduce_plus_times_one_child", "+($1)"),
    ("reduce_plus_times_one_child", "*(+($1 $2))"),
    ("reduce_plus_times_one_child", "+($1 $2)"),
    ("reduce_invert_constant", "/(0.0 $1)"),
    ("reduce_invert_constant", "/($1 2.0)"),
    ("reduce_invert_constant", "/($1 0.0)"),
    ("reduce_invert_constant", "/($1 *(2.0 $2))"),
    ("reduce_invert_constant", "/($1 *(2.0 $2 $3))"),
    ("reduce_fraction", "/(*($1 $2) *($1 $3))"),
    ("reduce_fraction", "/(*($1 $2) $1)"),
    ("reduce_fraction", "/(*($1 $2) *($3 $4))"),
    ("reduce_fraction", "/(*($1 $1 $2) *($1 $1))"),
    ("reduce_factorize", "+(*($1 $2) *($1 $3))"),
    ("reduce_factorize", "+($1 *($1 $3))"),
    ("reduce_factorize", "+(*($1 $2) *($3 $4))"),
    ("reduce_factorize_fraction", "+(/($1 $3) /($2 $3))"),
    ("reduce_factorize_fraction", "+(/($1 $3) /($2 $4))"),
    ("reduce_times_div", "/(/($1 $2) $3)"),
    ("reduce_times_div", "/($1 /($2 $3))"),
    ("reduce_times_div", "*($1 /($2 $3))"),
    ("reduce_times_div", "*($1 $2 /($3 $4) /($5 $6))"),
    ("reduce_times_div", "*($1 $2)"),
    ("reduce_sum_log", "+(log($1) log($2))"),
    ("reduce_sum_log", "+(log($1) *(-1.0 log($2)))"),
    ("reduce_sum_log", "+(log($1) *(-1.0 $2))"),
    ("reduce_sum_log", "+($1 $2)"),
    ("reduce_log_div_times", "log(/(2.0 $1))"),
    ("reduce_log_div_times", "log(/(0.0 $1))"),
    ("reduce_log_div_times", "log(exp($1))"),
    ("reduce_log_div_times", "log(*(exp($1) $2))"),
    ("reduce_log_div_times", "log(*(exp($1) exp($2)))"),
    ("reduce_exp_times", "*(exp($1) exp($2))"),
    ("reduce_exp_times", "*(exp($1) $3 exp($2))"),
    ("reduce_exp_times", "*(exp($1) $2)"),
    ("reduce_exp_div", "/($1 exp($2))"),
    ("reduce_exp_div", "/($1 exp(*($2 $3)))"),
    ("reduce_exp_div", "/($1 $2)"),
    ("reduce_exp_log", "exp(log($1))"),
    ("reduce_exp_log", "exp(+(log($1) $2))"),
    ("reduce_exp_log", "exp(+(log($1) log($3) $2))"),
    ("reduce_exp_log", "exp(+(log($1) log($2)))"),
    ("reduce_sin", "sin(+($1 4.0))"),
    ("reduce_sin", "sin(+($1 1.0))"),
    ("reduce_sin", "sin(+($1 1.0 2.0))"),
    ("reduce_sin", "sin(inf)"),
    ("reduce_sin", "sin(4.0)"),
    ("reduce_cond_else", "cond($1 $2 $2)"),
    ("reduce_cond_else", "cond($1 $2 $3)"),
    ("reduce_cond_else", "cond($1 $2 $3 $4 $4)"),
    ("reduce_cond_else", "contin_if($1 $2 $2)"),
    ("reduce_cond_const", "cond(true $1 $2)"),
    ("reduce_cond_const", "cond(false $1 $2)"),
    ("reduce_cond_const", "cond(false $1 $2 $3 $4)"),
    ("reduce_cond_const", "cond($1 $2 true $3 $4)"),
    ("reduce_cond_const", "contin_if(true $1 $2)"),
    ("reduce_cond_adjacent", "cond($1 $2 $3 $2 $4)"),
    ("reduce_cond_adjacent", "cond($1 $2 $3 $4 $5)"),
    ("reduce_cond_adjacent", "cond($1 $2 $3)"),
    ("reduce_factorize", "$1"),  # childless-plus AssertionError case below
]


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    cmd = [
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
        "-lcomboreduct",
        "-lcogutil",
    ]
    shell_cmd = f"ulimit -v {_MEM_LIMIT_KIB}; timeout {_COMPILE_TIMEOUT_S} {shlex.join(cmd)}"
    result = subprocess.run(["bash", "-c", shell_cmd], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"g++ build failed (exit {result.returncode})")


def _run_one(rule: str, tree: str) -> str:
    result = subprocess.run(
        [str(PROBE_BIN), rule, tree],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        preexec_fn=_bound_resources,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"probe failed for {rule!r} {tree!r}: rc={result.returncode} "
            f"stderr={result.stderr!r}"
        )
    return result.stdout.strip()


def capture_fixture() -> None:
    cases = []
    for rule, tree in CASES:
        # reduce_factorize on a bare $1 forced to `plus` by mutating the
        # tree AFTER parsing is a Python-only test (test_contin_rules.py's
        # own childless-plus AssertionError case); the probe has no CLI
        # hook to reproduce that exact construction, so skip capturing an
        # oracle row for it here (it needs no oracle: it is a restored
        # OC_ASSERT, not divergence-prone tree surgery).
        if rule == "reduce_factorize" and tree == "$1":
            continue
        output = _run_one(rule, tree)
        cases.append({"rule": rule, "input": tree, "output": output})
    fixture = {"cases": cases}
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_branch_rules.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_rule_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "hand-picked (rule, tree) cases hardcoded in this "
                    "capture script, one per contin_rules.py/branch_rules.py "
                    "rule class and its documented edge cases; no external "
                    "dataset or seed"
                ),
                "case_count": len(cases),
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_contin_branch_rules.py takes no arguments")
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
