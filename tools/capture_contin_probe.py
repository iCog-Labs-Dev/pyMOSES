"""Build tools/cpp_probe/contin_interpreter_probe.cc and capture its plain
`label=STATUS:value` output into tests/fixtures/contin_probe.json.

The committed tests/fixtures/contin_probe.json is what the default test
suite actually reads (tests/comboreduct/interpreter/test_interpreter.py),
so the suite runs without a C++ toolchain or the classic checkout at all
-- same division of labor as tools/capture_score.py/score.json.

This probe exists for exactly the cases interpreterUTest.cxxtest's own
three tests (already transliterated byte-for-byte) do NOT reach: overflow
reconstruction (div-by-zero, log domain errors, exp overflow), sin's
silent-nan-no-throw, a bare nan/inf contin_t leaf, the childless
plus()/times() builtin->contin_t ordinal leak, and mixed_interpreter's
greater_than_zero recovering a sign from a caught OverflowException.

Every probe run is resource-bounded (`ulimit -v` + `timeout`): this
machine has been OOM-killed twice by unbounded probe runs (see CLAUDE.md).
"""

import json
import re
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_interpreter_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_interpreter_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_probe.json"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

# ulimit -v 1048576 (KiB) / timeout 20 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 20

_LINE_RE = re.compile(r"^(?P<label>[a-z0-9_]+)=(?P<status>VALUE|OVERFLOW):(?P<value>.+)$")


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
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
            str(PREFIX / "lib"),
            "-L",
            str(PREFIX / "lib" / "opencog"),
            "-Wl,-rpath," + str(PREFIX / "lib"),
            "-Wl,-rpath," + str(PREFIX / "lib" / "opencog"),
            "-lcomboreduct",
            "-lcogutil",
        ],
        check=True,
    )


def run_probe() -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [str(PROBE_BIN)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    cases: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            raise RuntimeError(f"unparseable probe output line: {line!r}")
        cases[m["label"]] = {"status": m["status"], "value": m["value"]}
    return cases


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed set of hardcoded probe cases
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()

    cases = run_probe()
    if not cases:
        raise RuntimeError("probe produced no cases -- check its own stdout format")

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_probe.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_interpreter_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={"n_cases": len(cases)},
        ),
    )
    print(f"wrote {FIXTURE_PATH} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
