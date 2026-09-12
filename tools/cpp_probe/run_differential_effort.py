"""Task 6 (S7 cycle) differential harness against the FULL effort
pipelines -- item 4 of the task brief: "Run the differential harness
against the effort pipelines and report the divergence count."

Task 4's own `run_differential.py`/`logical_rules_differential_probe.cc`
exercise `subtree_to_enf` alone (a "raw" pipeline with no pre-
canonicalization, and a "canon" pipeline that only runs `eval_logical_
identities`; `level`; `subtree_to_enf`) -- explicitly NOT the full
`logical_reduce(effort, tr, ignore_ops)` pipelines `logical_reduction.py`
assembles (`p_extra_simple`/`p_simple`/`p_medium`/`p_complexe`). This
script is that same idea -- randomized differential testing beyond the
79-tree hand-picked corpus in tests/fixtures/reduct.json -- retargeted at
those full pipelines, at efforts 1, 2 and 3, reusing:

* `gen_boolean_trees.generate_corpus` (Task 4) for the randomized,
  seeded, reproducible tree corpus.
* `tools/capture_reduct.py`'s own `build_probe`/`run_probe`/`PROBE_BIN`
  (Task 6 corpus infrastructure) to build and run
  `reduct_effort_probe.cc` -- the SAME probe tests/fixtures/reduct.json
  itself was captured with, already proven correct against that fixture
  by `test_reduct_oracle.py`'s byte-identical layer -- rather than
  writing a third copy of the same "build+run a bounded C++ probe" logic.

Usage (from the repo root, with `.venv` already bootstrapped by
`tools/ci.sh`, and the classic C++ build's `libcomboreduct.so`/
`libcogutil.so` available at $PREFIX -- see capture_reduct.py's own
module docstring):

    .venv/bin/python3 tools/cpp_probe/run_differential_effort.py
    .venv/bin/python3 tools/cpp_probe/run_differential_effort.py --count 5000 --seed 1

Exit code is 0 iff every comparison matched (all three efforts, every
tree) -- same convention as `run_differential.py`. Deliberately NOT wired
into `tools/ci.sh` (same reason as `run_differential.py`'s own docstring
gives: this needs the classic C++ build, which `ci.sh`'s self-bootstrap
does not provide) and NOT a pytest test (same reason
`tests/fixtures/reduct.json`'s own capture script,
`tools/capture_reduct.py`, is not one either) -- a developer tool, run
manually and its result recorded in task-6-report.md.

Every C++ invocation here is resource-bounded (`ulimit -v` + `timeout`)
per CLAUDE.md's OOM-safety rule, via `capture_reduct.py`'s own
`build_probe`/`run_probe`, which already do this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "comboreduct" / "reduct"))

from enf_checkers import (  # noqa: E402
    root_is_bare_junctor_token,
    truth_table_preserved,
    tt_unsound_by_construction,
)

EFFORTS = (1, 2, 3)

# The random generator intentionally creates only 2-3 child junctors.
# Keep these measured edge cases in every run so the harness exercises
# the exact childless-junctor behavior that motivated its three-phase C++
# probe. They cover harmless root folding, genuine divergence, masking,
# and reduced root-token output.
PATHOLOGICAL_TREES = (
    "and()",
    "or()",
    "and(or())",
    "and(or() or())",
    "or(and(or()) $1)",
    "and($1 or(and(or()) $1))",
    "or(and() and())",
    "or(and(or($1 and(!$1))) !$2)",
)


def run_python_side(tree_str: str, effort: int) -> dict:
    """Mirrors `reduct_effort_probe.cc`'s own per-tree, per-effort
    computation: parse, `logical_reduce(effort, tr, ignore_ops=set())`,
    print. `ignore_ops=set()` matches the probe's own `vertex_set
    ignore_ops;` (always empty -- every tree this corpus generates is a
    pure and/or/not/$n boolean tree, no predicates to ignore).

    Also reports `root_token`: whether the REDUCED tree's root is a bare
    childless `and()`/`or()` -- see `_CPP_BUILTIN_TO_BOOL_ASSERT` below
    for why this matters: upstream `test_reduct.cc` cannot report a reduced
    value for this shape because its post-reduction truth-table call throws.
    This project's three-phase probe still reports `ok` and `reduced`, then
    records `tt_computed: false` separately. The throw comes from
    `vertex.h:708-713`'s `OC_ASSERT`, matching this project's own
    `enf_checkers.ROOT_TOKEN` design."""
    from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
    from pymoses.comboreduct.reduct.logical_reduction import logical_reduce

    try:
        original = str_to_combo_tree(tree_str)
        tr = str_to_combo_tree(tree_str)
        logical_reduce(effort, tr, ignore_ops=set())
        row = {
            "ok": True,
            "reduced": combo_tree_to_str(tr),
            "original_root_token": root_is_bare_junctor_token(original),
            "root_token": root_is_bare_junctor_token(tr),
            "input_shape_flagged": tt_unsound_by_construction(original),
        }
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this IS the oracle
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    try:
        preserved = truth_table_preserved(original, tr)
        row["tt_computed"] = preserved is not None
        if preserved is not None:
            row["tt_preserved"] = preserved
    except Exception as e:  # noqa: BLE001 -- diagnostic phase must not mask reduction success
        row["tt_computed"] = False
        row["tt_error"] = f"{type(e).__name__}: {e}"
    return row


# Exact OC_ASSERT message from builtin_to_bool (vertex.h:708-713). Unlike
# the old substring, this does not also match vertex_to_bool's assertion.
_CPP_BUILTIN_TO_BOOL_ASSERT = (
    "builtin should be 'id::logical_true' or 'id::logical_false'."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--arity", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-print", type=int, default=10)
    parser.add_argument("--keep-corpus", type=Path, default=None)
    args = parser.parse_args()

    from tools.capture_reduct import PROBE_BIN, build_probe, run_probe
    from tools.cpp_probe.gen_boolean_trees import generate_corpus

    random_trees = generate_corpus(args.count, args.seed, args.max_depth, args.arity)
    trees = list(dict.fromkeys([*random_trees, *PATHOLOGICAL_TREES]))
    corpus_path = args.keep_corpus or (
        REPO_ROOT / "tools" / "cpp_probe" / "build" / "_differential_effort_corpus.txt"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(trees) + "\n")
    print(
        f"Generated {len(random_trees)} random trees and included "
        f"{len(PATHOLOGICAL_TREES)} targeted trees (seed={args.seed}); "
        f"{len(trees)} unique total -> {corpus_path}"
    )

    build_probe(force=args.rebuild)
    print(f"Probe: {PROBE_BIN}")

    divergences: list[dict] = []
    crashes: list[dict] = []
    cpp_only_errors: list[dict] = []
    both_errored_differently: list[dict] = []
    semantic_instrumentation_mismatches: list[dict] = []
    cpp_nonpreserved_flagged = 0
    cpp_nonpreserved_unflagged = 0
    root_token_agreements = 0
    counts = {e: 0 for e in EFFORTS}

    for effort in EFFORTS:
        cpp_results = run_probe(corpus_path, effort)
        if len(cpp_results) != len(trees):
            raise RuntimeError(
                f"effort {effort}: C++ probe returned {len(cpp_results)} "
                f"results for {len(trees)} inputs"
            )
        for tree_str, cpp_row in zip(trees, cpp_results, strict=True):
            if cpp_row["input"] != tree_str:
                raise RuntimeError(
                    f"effort {effort}: probe result/corpus order mismatch: "
                    f"{cpp_row['input']!r} != {tree_str!r}"
                )
            py_row = run_python_side(tree_str, effort)
            cpp_ok, py_ok = cpp_row["ok"], py_row["ok"]

            if cpp_ok and py_ok:
                if cpp_row["reduced"] == py_row["reduced"]:
                    counts[effort] += 1
                else:
                    divergences.append(
                        {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                    )
                cpp_tt_computed = cpp_row.get("tt_computed", True)
                py_tt_computed = py_row.get("tt_computed", True)
                tt_matches = cpp_tt_computed == py_tt_computed
                if cpp_tt_computed and py_tt_computed:
                    tt_matches = cpp_row["tt_preserved"] == py_row["tt_preserved"]
                    if not cpp_row["tt_preserved"]:
                        if py_row["input_shape_flagged"]:
                            cpp_nonpreserved_flagged += 1
                        else:
                            cpp_nonpreserved_unflagged += 1
                elif not cpp_tt_computed and not py_tt_computed:
                    cpp_expected_root_token = _CPP_BUILTIN_TO_BOOL_ASSERT in cpp_row.get(
                        "tt_error", ""
                    )
                    py_expected_root_token = (
                        "tt_error" not in py_row
                        and (py_row["original_root_token"] or py_row["root_token"])
                    )
                    tt_matches = cpp_expected_root_token and py_expected_root_token
                if not tt_matches:
                    semantic_instrumentation_mismatches.append(
                        {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                    )
            elif cpp_ok and not py_ok:
                crashes.append(
                    {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                )
            elif not cpp_ok and py_ok:
                # The C++ probe reports `ok` for logical_reduce alone;
                # post-hoc truth-table failures no longer reach this
                # branch. A C++ reduction failure with Python success is
                # therefore always a real mismatch.
                cpp_only_errors.append(
                    {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                )
            else:
                # Both refused. `run_differential.py` (Task 4)'s own
                # precedent: only a real mismatch if the reasons look
                # unrelated -- normalize known signatures before deciding.
                # Found empirically (not merely anticipated): at effort 3,
                # `reduce_remove_subtree_equal_tt` calls `complete_truth_
                # table` as PART OF its own real algorithm (amputate a
                # subtree, check whether the truth table is unchanged) --
                # unlike `reduct_effort_probe.cc`'s post-hoc-only call,
                # this one IS inside `logical_reduce` itself. If the
                # amputated candidate's root ends up a bare childless
                # junctor, BOTH C++ and this port's own `logical_rules.py`
                # counterpart hit the exact same `builtin_to_bool` OC_
                # ASSERT from within the real algorithm -- a shared,
                # faithfully-reproduced crash, not a divergence.
                cpp_reason = cpp_row.get("error", "")
                py_reason = py_row.get("error", "")
                if _CPP_BUILTIN_TO_BOOL_ASSERT in cpp_reason and (
                    _CPP_BUILTIN_TO_BOOL_ASSERT in py_reason
                ):
                    root_token_agreements += 1
                else:
                    both_errored_differently.append(
                        {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                    )

    total = len(trees)
    print()
    for effort in EFFORTS:
        print(f"effort {effort}: {counts[effort]}/{total} byte-identical")
    print(f"Divergences (both sides ran, values differ): {len(divergences)}")
    print(f"Python crashes where C++ succeeded: {len(crashes)}")
    print(f"C++ refused where Python succeeded (genuine): {len(cpp_only_errors)}")
    print(f"Both sides errored, for seemingly different reasons: {len(both_errored_differently)}")
    print(
        "Truth-table diagnostic mismatches between C++ and Python: "
        f"{len(semantic_instrumentation_mismatches)}"
    )
    print(
        "C++ truth-table non-preservation: "
        f"{cpp_nonpreserved_flagged} input-shape flagged, "
        f"{cpp_nonpreserved_unflagged} input-shape unflagged"
    )
    print(
        f"Shared builtin-to-bool failures inside logical_reduce "
        f"(not counted as divergences): {root_token_agreements}"
    )

    for label, bucket in (
        ("divergences", divergences),
        ("crashes", crashes),
        ("cpp_only_errors", cpp_only_errors),
        ("both_errored_differently", both_errored_differently),
        ("semantic_instrumentation_mismatches", semantic_instrumentation_mismatches),
    ):
        if not bucket:
            continue
        print(f"\n--- First {label} ---")
        for row in bucket[: args.max_print]:
            print(f"  [effort {row['effort']}] input: {row['input']}")
            print(f"    C++   : {row['cpp']}")
            print(f"    Python: {row['python']}")

    total_problems = len(divergences) + len(crashes) + len(cpp_only_errors)
    total_divergence_count = (
        total_problems
        + len(both_errored_differently)
        + len(semantic_instrumentation_mismatches)
    )
    print(f"\nTOTAL DIVERGENCE COUNT: {total_divergence_count}")
    return 0 if total_divergence_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
