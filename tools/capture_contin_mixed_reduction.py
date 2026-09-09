"""Build and capture the S14 Task 4 end-to-end contin_reduce/mixed_reduce
oracle fixture.

Runs `tools/cpp_probe/contin_mixed_reduct_probe.cc` over `tools/cpp_probe/
contin_mixed_corpus.txt` (53 hand-picked contin/mixed trees) at contin
effort 2 and in mixed mode, both with classic's own real default
`ignore_ops` (empty) -- CLAUDE.md's own "at least one oracle must run
classic's own defaults" requirement. `tests/comboreduct/reduct/
test_contin_mixed_reduction_oracle.py` replays every row through this
port's own `contin_reduce`/`mixed_reduce` and asserts byte-identical
output; four additional assumption-bearing cases (not expressible in the
corpus's plain combo-tree text format -- assumptions are not part of
combo-tree syntax) are captured separately via a second, small probe.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_mixed_reduct_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_mixed_reduct_probe"
CORPUS_PATH = REPO_ROOT / "tools" / "cpp_probe" / "contin_mixed_corpus.txt"
ASSUM_PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "mixed_assumption_probe.cc"
ASSUM_PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "mixed_assumption_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_mixed_reduction.json"
_MEM_LIMIT_KIB = 2097152
_MEM_LIMIT_BYTES = _MEM_LIMIT_KIB * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def _build(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
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
        str(out),
        str(src),
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
        raise RuntimeError(f"g++ build failed for {src.name} (exit {result.returncode})")


def _run_probe(args: list[str]) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        preexec_fn=_bound_resources,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"probe failed (exit {result.returncode})")
    return result.stdout


def capture_fixture() -> None:
    contin_rows = json.loads(_run_probe([str(PROBE_BIN), str(CORPUS_PATH), "contin", "2"]))
    mixed_rows = json.loads(_run_probe([str(PROBE_BIN), str(CORPUS_PATH), "mixed"]))
    assumption_rows = json.loads(_run_probe([str(ASSUM_PROBE_BIN)]))

    fixture = {
        "contin_effort_2": contin_rows,
        "mixed": mixed_rows,
        "mixed_with_assumptions": assumption_rows,
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_mixed_reduction.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_mixed_reduct_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "53 hand-picked contin/mixed combo trees in "
                    "tools/cpp_probe/contin_mixed_corpus.txt, reduced at "
                    "contin effort 2 and in mixed mode with classic's own "
                    "real default (empty) ignore_ops, plus 4 assumption-"
                    "bearing cases from tools/cpp_probe/"
                    "mixed_assumption_probe.cc (assumptions are not "
                    "expressible in combo-tree text syntax, so that probe "
                    "is separate); no external dataset or seed"
                ),
                "corpus_size": len(contin_rows),
                "assumption_case_count": len(assumption_rows),
                "second_probe_source": "tools/cpp_probe/mixed_assumption_probe.cc",
            },
        ),
    )


def main(argv: list[str]) -> int:
    if argv:
        raise SystemExit("capture_contin_mixed_reduction.py takes no arguments")
    _build(PROBE_SRC, PROBE_BIN)
    _build(ASSUM_PROBE_SRC, ASSUM_PROBE_BIN)
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
