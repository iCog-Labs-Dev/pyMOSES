"""Task 6 oracle capture: build tools/cpp_probe/reduct_effort_probe.cc
against the real, prebuilt classic MOSES `libcomboreduct.so`/`libcogutil.so`
(scripts/rebuild-moses-local.sh's install prefix), run it over the corpus
assembled by tools/cpp_probe/reduct_corpus.py at logical-reduction efforts
1, 2 and 3, and write tests/fixtures/reduct.json.

The committed tests/fixtures/reduct.json is what the default test suite
actually reads -- Task 6 wires up the actual byte-identical-reduction and
truth-table/ENF-axiom tests against it -- so the suite runs without a C++
toolchain or the classic checkout at all. This script (like
tools/capture_combo_roundtrip.py, tools/capture_eval_boolean.py,
tools/capture_score.py, tools/capture_table.py, and
tools/cpp_probe/run_differential.py) is a developer tool, not part of
tools/ci.sh, and is the thing to rerun if reduct_corpus.py's corpus or
reduct_effort_probe.cc's probe ever change.

Every C++ invocation here -- the g++ compile AND every probe run -- is
resource-bounded (`ulimit -v` + `timeout`): this machine has been
OOM-killed twice by unbounded runs (see CLAUDE.md). The compile is bounded
via a `bash -c` wrapper (mirrors tools/cpp_probe/run_differential.py's
build_probe); each probe run is bounded via `resource.setrlimit` in a
`preexec_fn` (mirrors tools/capture_combo_roundtrip.py/capture_eval_boolean.py
-- no shell re-quoting needed since args here are plain filenames/ints,
not combo strings full of `$`/`!`/`(`/`)`).

Usage (from the repo root, with `.venv` already bootstrapped by
tools/ci.sh):

    .venv/bin/python3 tools/capture_reduct.py
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402
from tools.cpp_probe.reduct_corpus import CORPUS  # noqa: E402

REDUCT_CORPUS_PATH = REPO_ROOT / "tools" / "cpp_probe" / "reduct_corpus.py"
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "reduct_effort_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "reduct_effort_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "reduct.json"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

EFFORTS = (1, 2, 3)

# CLAUDE.md: "this machine has been OOM-killed twice. Bound every C++ run:
# ( ulimit -v 2097152; timeout 300 <cmd> ). Never unbounded."
_MEM_LIMIT_KIB = 2097152
_MEM_LIMIT_BYTES = _MEM_LIMIT_KIB * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe(force: bool = False) -> None:
    if PROBE_BIN.exists() and not force:
        if PROBE_BIN.stat().st_mtime >= PROBE_SRC.stat().st_mtime:
            return
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        f"-I{CLASSIC}",
        f"-I{PREFIX / 'include'}",
        "-o",
        str(PROBE_BIN),
        str(PROBE_SRC),
        f"-L{PREFIX / 'lib'}",
        f"-L{PREFIX / 'lib' / 'opencog'}",
        f"-Wl,-rpath,{PREFIX / 'lib'}",
        f"-Wl,-rpath,{PREFIX / 'lib' / 'opencog'}",
        "-lcomboreduct",
        "-lcogutil",
    ]
    shell_cmd = f"ulimit -v {_MEM_LIMIT_KIB}; timeout {_COMPILE_TIMEOUT_S} " + " ".join(cmd)
    bounded = ["bash", "-c", shell_cmd]
    result = subprocess.run(bounded, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"g++ build failed (exit {result.returncode})")


def run_probe(corpus_path: Path, effort: int) -> list[dict]:
    cmd = [str(PROBE_BIN), str(corpus_path), str(effort)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
        preexec_fn=_bound_resources,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"reduct_effort_probe failed at effort {effort} (exit {result.returncode})"
        )
    return json.loads(result.stdout)


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus x effort cross to (re)capture
    build_probe()
    print(f"probe: {PROBE_BIN}")

    runnable = [c for c in CORPUS if c["runnable"]]
    skipped = [c for c in CORPUS if not c["runnable"]]

    corpus_txt_path = PROBE_BIN.parent / "_reduct_corpus.txt"
    corpus_txt_path.write_text("\n".join(c["combo"] for c in runnable) + "\n")

    cases: list[dict] = []
    for effort in EFFORTS:
        results = run_probe(corpus_txt_path, effort)
        if len(results) != len(runnable):
            raise RuntimeError(
                f"effort {effort}: probe returned {len(results)} results for "
                f"{len(runnable)} runnable corpus entries"
            )
        n_ok = 0
        n_tt_bad = 0
        for entry, result in zip(runnable, results, strict=True):
            if result["input"] != entry["combo"]:
                raise RuntimeError(
                    f"effort {effort}: probe result/corpus order mismatch: "
                    f"{result['input']!r} != {entry['combo']!r}"
                )
            case = {
                "id": entry["id"],
                "source": entry["source"],
                "effort": effort,
                "input": entry["combo"],
                "known_tt_divergent": entry["known_tt_divergent"],
            }
            if "enf_original" in entry:
                case["enf_original"] = entry["enf_original"]
            if result["ok"]:
                n_ok += 1
                case["ok"] = True
                case["reduced"] = result["reduced"]
                case["size_before"] = result["size_before"]
                case["size_after"] = result["size_after"]
                case["tt_computed"] = result.get("tt_computed", True)
                if case["tt_computed"]:
                    case["tt_preserved"] = result["tt_preserved"]
                else:
                    case["tt_error"] = result["tt_error"]
                if case["tt_computed"] and not result["tt_preserved"]:
                    n_tt_bad += 1
            else:
                case["ok"] = False
                case["error"] = result["error"]
            cases.append(case)
        print(
            f"effort {effort}: {n_ok}/{len(runnable)} ok, "
            f"{n_tt_bad} truth-table mismatches reported by C++"
        )

    fixture = {
        "efforts": list(EFFORTS),
        "sources": [
            "testTrees.sh",
            "BooleanReductUTest.cxxtest",
            "BooleanFullUTest.cxxtest",
            "FullReductUTest.cxxtest",
            "GeneralTestCases.py",
        ],
        "counts": {
            "corpus_total": len(CORPUS),
            "runnable": len(runnable),
            "skipped": len(skipped),
            "cases": len(cases),
        },
        "cases": cases,
        "skipped": [
            {
                "id": c["id"],
                "source": c["source"],
                "input": c["combo"],
                "reason": c["reason"],
            }
            for c in skipped
        ],
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_reduct.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/reduct_effort_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "efforts": list(EFFORTS),
                "corpus_module": "tools/cpp_probe/reduct_corpus.py",
                "corpus_module_sha256": hashlib.sha256(
                    REDUCT_CORPUS_PATH.read_bytes()
                ).hexdigest(),
                "corpus_total": len(CORPUS),
                "runnable": len(runnable),
                "skipped": len(skipped),
                "sources": fixture["sources"],
            },
        ),
    )
    print(
        f"wrote {FIXTURE_PATH}: {len(cases)} cases "
        f"({len(runnable)} runnable trees x {len(EFFORTS)} efforts), "
        f"{len(skipped)} skipped (not-yet-runnable)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
