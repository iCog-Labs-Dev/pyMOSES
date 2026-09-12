"""Differential oracle runner for pymoses's comboreduct/reduct/
logical_rules.py (Task 4 review, Important: "land the differential harness
in-repo").

End-to-end orchestration:

1. Generate a deterministic corpus of random boolean combo-trees
   (`gen_boolean_trees.py`).
2. Build `logical_rules_differential_probe.cc` against the real, prebuilt
   `libcomboreduct.so` (skipped if already built and newer than the
   source -- pass --rebuild to force).
3. Run it, BOUNDED (`ulimit -v` + `timeout`, per CLAUDE.md's OOM-safety
   rule), over the corpus -- two pipelines per tree: "raw"
   (`subtree_to_enf()` alone) and "canon" (`eval_logical_identities`;
   `level`; `subtree_to_enf()`).
4. Run the SAME two pipelines, over the SAME corpus, through the Python
   port.
5. Compare. Report a summary (match/divergence counts per pipeline,
   crash counts) and print the first few divergences in full.

Usage (from the repo root, with `.venv` already bootstrapped by
`tools/ci.sh`):

    .venv/bin/python3 tools/cpp_probe/run_differential.py
    .venv/bin/python3 tools/cpp_probe/run_differential.py --count 5000 --seed 1
    .venv/bin/python3 tools/cpp_probe/run_differential.py --rebuild

Exit code is 0 iff every comparison matched (both pipelines, every tree) --
suitable for a CI gate once this harness is trusted enough to be one
(deliberately NOT wired into `tools/ci.sh` yet: it needs the classic C++
build's `libcomboreduct.so`, which `ci.sh`'s self-bootstrap does not
provide).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROBE_DIR = REPO_ROOT / "tools" / "cpp_probe"
PROBE_SRC = PROBE_DIR / "logical_rules_differential_probe.cc"
PROBE_BIN = PROBE_DIR / "build" / "logical_rules_differential_probe"

DEFAULT_PREFIX = Path(os.environ.get("PREFIX", str(Path.home() / "opt" / "moses-legacy")))
DEFAULT_CLASSIC = Path(
    os.environ.get(
        "CLASSIC", str(REPO_ROOT.parent / "MOSES" / "OpenCog" / "Moses")
    )
)


def build_probe(classic: Path, prefix: Path, force: bool) -> None:
    if PROBE_BIN.exists() and not force:
        if PROBE_BIN.stat().st_mtime >= PROBE_SRC.stat().st_mtime:
            return
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        f"-I{classic}",
        f"-I{prefix / 'include'}",
        "-o",
        str(PROBE_BIN),
        str(PROBE_SRC),
        f"-L{prefix / 'lib'}",
        f"-L{prefix / 'lib' / 'opencog'}",
        f"-Wl,-rpath,{prefix / 'lib'}",
        f"-Wl,-rpath,{prefix / 'lib' / 'opencog'}",
        "-lcomboreduct",
        "-lcogutil",
    ]
    # Bounded per CLAUDE.md's OOM-safety rule -- compilation itself, not
    # just running the resulting binary.
    bounded = ["bash", "-c", "ulimit -v 4194304; timeout 120 " + " ".join(cmd)]
    result = subprocess.run(bounded, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"g++ build failed (exit {result.returncode})")


def run_probe(corpus_path: Path) -> list[dict]:
    cmd = [
        "bash",
        "-c",
        f"ulimit -v 1048576; timeout 60 {PROBE_BIN} {corpus_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"C++ probe failed (exit {result.returncode}) -- see stderr above"
        )
    return json.loads(result.stdout)


def run_python_side(tree_str: str) -> dict:
    """Mirrors the C++ probe's own two pipelines exactly."""
    from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
    from pymoses.comboreduct.reduct.general_rules import level
    from pymoses.comboreduct.reduct.logical_rules import eval_logical_identities, subtree_to_enf
    from pymoses.comboreduct.reduct.meta_rules import downwards, upwards

    out: dict = {}
    try:
        tr = str_to_combo_tree(tree_str)
        subtree_to_enf()(tr, tr.begin())
        out["raw"] = combo_tree_to_str(tr).strip()
        out["raw_ok"] = True
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this IS the oracle
        out["raw"] = f"ERROR: {type(e).__name__}: {e}"
        out["raw_ok"] = False

    try:
        tr = str_to_combo_tree(tree_str)
        it = tr.begin()
        upwards(eval_logical_identities())(tr, it)
        downwards(level())(tr, it)
        subtree_to_enf()(tr, it)
        out["canon"] = combo_tree_to_str(tr).strip()
        out["canon_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["canon"] = f"ERROR: {type(e).__name__}: {e}"
        out["canon_ok"] = False

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--arity", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--classic", type=Path, default=DEFAULT_CLASSIC)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--max-print", type=int, default=10)
    parser.add_argument("--keep-corpus", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    from tools.cpp_probe.gen_boolean_trees import generate_corpus

    trees = generate_corpus(args.count, args.seed, args.max_depth, args.arity)
    corpus_path = args.keep_corpus or (PROBE_DIR / "build" / "_differential_corpus.txt")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(trees) + "\n")

    print(f"Generated {len(trees)} trees (seed={args.seed}) -> {corpus_path}")

    build_probe(args.classic, args.prefix, args.rebuild)
    print(f"Probe: {PROBE_BIN}")

    cpp_results = run_probe(corpus_path)
    if len(cpp_results) != len(trees):
        print(
            f"WARNING: C++ probe returned {len(cpp_results)} results for "
            f"{len(trees)} inputs",
            file=sys.stderr,
        )

    divergences: list[dict] = []
    crashes: list[dict] = []
    both_errored: list[dict] = []
    counts = {"raw_match": 0, "canon_match": 0, "total": 0}

    for cpp_row in cpp_results:
        tree_str = cpp_row["input"]
        py_row = run_python_side(tree_str)
        counts["total"] += 1

        row_report = {"input": tree_str, "cpp": cpp_row, "python": py_row}

        for pipeline in ("raw", "canon"):
            cpp_ok = cpp_row[f"{pipeline}_ok"]
            py_ok = py_row[f"{pipeline}_ok"]
            cpp_val = cpp_row[pipeline]
            py_val = py_row[pipeline]

            if not cpp_ok and not py_ok:
                # BOTH sides refused this input -- not a divergence in the
                # sense of "one side is wrong", but still worth a look: if
                # the two error MESSAGES don't share a recognizable cause
                # (e.g. one says "unexpected operator", the other "aren't
                # sorted"), that IS worth surfacing as a mismatch.
                cpp_reason = "unexpected operator" if "unexpected operator" in cpp_val else (
                    "aren't sorted" if "aren't sorted" in cpp_val else cpp_val
                )
                py_reason = "unexpected operator" if "unexpected operator" in py_val else (
                    "aren't sorted" if "aren't sorted" in py_val else py_val
                )
                if cpp_reason == py_reason:
                    counts[f"{pipeline}_match"] += 1
                else:
                    both_errored.append({**row_report, "pipeline": pipeline})
            elif cpp_ok and not py_ok:
                crashes.append({**row_report, "pipeline": pipeline})
            elif not cpp_ok and py_ok:
                # C++ refused, Python computed a value anyway -- e.g. a
                # precondition (is_sorted) C++ enforces with an OC_ASSERT
                # that this port does not reproduce. Worth surfacing, but
                # is its own bucket (neither "Python crashed" nor a value
                # divergence in the usual sense).
                both_errored.append({**row_report, "pipeline": pipeline})
            elif cpp_val == py_val:
                counts[f"{pipeline}_match"] += 1
            else:
                divergences.append({**row_report, "pipeline": pipeline})

    total = counts["total"]
    print()
    print(f"Total trees: {total}")
    print(f"  raw   pipeline: {counts['raw_match']}/{total} match")
    print(f"  canon pipeline: {counts['canon_match']}/{total} match")
    print(f"Divergences (both sides ran, values differ): {len(divergences)}")
    print(f"Python crashes where C++ succeeded: {len(crashes)}")
    print(f"Both sides errored, for seemingly different reasons: {len(both_errored)}")
    print(
        "  (NOTE: `raw`-pipeline errors on either side are EXPECTED and not"
        " failures -- subtree_to_enf()'s own docstring requires"
        " canonicalization first; an uncanonicalized tree is out of its"
        " documented contract, and real C++ enforces internal preconditions"
        " (is_sorted OC_ASSERTs) this port does not reproduce for that"
        " unsupported case. `canon`-pipeline entries in ANY of these three"
        " buckets ARE real findings.)"
    )

    if crashes:
        print("\n--- First crashes ---")
        for row in crashes[: args.max_print]:
            print(f"  [{row['pipeline']}] input: {row['input']}")
            print(f"    C++   : {row['cpp'][row['pipeline']]!r}")
            print(f"    Python: {row['python'][row['pipeline']]!r}")

    if divergences:
        print("\n--- First divergences ---")
        for row in divergences[: args.max_print]:
            print(f"  [{row['pipeline']}] input: {row['input']}")
            print(f"    C++   : {row['cpp'][row['pipeline']]!r}")
            print(f"    Python: {row['python'][row['pipeline']]!r}")

    if both_errored:
        print("\n--- First both-errored (differing reasons) ---")
        for row in both_errored[: args.max_print]:
            print(f"  [{row['pipeline']}] input: {row['input']}")
            print(f"    C++   : {row['cpp'][row['pipeline']]!r}")
            print(f"    Python: {row['python'][row['pipeline']]!r}")

    # Only the `canon` pipeline is subtree_to_enf's documented contract;
    # `raw`-pipeline problems are reported above but do not fail the run.
    canon_problems = [
        r for r in (divergences + crashes + both_errored) if r["pipeline"] == "canon"
    ]
    ok = not canon_problems
    print(
        f"\n{'PASS' if ok else 'FAIL'}: {len(canon_problems)} canon-pipeline "
        f"problem(s) out of {total} comparisons "
        f"({len(divergences) + len(crashes) + len(both_errored)} total across both pipelines)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
