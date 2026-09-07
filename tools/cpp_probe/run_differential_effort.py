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

EFFORTS = (1, 2, 3)


def run_python_side(tree_str: str, effort: int) -> dict:
    """Mirrors `reduct_effort_probe.cc`'s own per-tree, per-effort
    computation: parse, `logical_reduce(effort, tr, ignore_ops=set())`,
    print. `ignore_ops=set()` matches the probe's own `vertex_set
    ignore_ops;` (always empty -- every tree this corpus generates is a
    pure and/or/not/$n boolean tree, no predicates to ignore).

    Also reports `root_token`: whether the REDUCED tree's root is a bare
    childless `and()`/`or()` -- see `_CPP_BUILTIN_TO_BOOL_ASSERT` below
    for why this matters: this is the one shape `reduct_effort_probe.cc`
    (and upstream's own `test_reduct.cc`) cannot report a `reduced` value
    for at all, because their own POST-reduction `complete_truth_table`
    call -- instrumentation the probe adds on top of `logical_reduce`
    itself, not part of `logical_reduce`'s own contract -- aborts on
    exactly this input (`vertex.h:708-713`'s `OC_ASSERT`, matching this
    project's own `enf_checkers.ROOT_TOKEN` design)."""
    from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
    from pymetta.comboreduct.reduct.logical_reduction import logical_reduce

    sys.path.insert(0, str(REPO_ROOT / "tests" / "comboreduct" / "reduct"))
    from enf_checkers import root_is_bare_junctor_token

    try:
        tr = str_to_combo_tree(tree_str)
        logical_reduce(effort, tr, ignore_ops=set())
        return {
            "ok": True,
            "reduced": combo_tree_to_str(tr),
            "root_token": root_is_bare_junctor_token(tr),
        }
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this IS the oracle
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# The EXACT OC_ASSERT message `builtin_to_bool` (vertex.h:708-713) raises
# when `complete_truth_table` (called by `reduct_effort_probe.cc` itself,
# AFTER `logical_reduce` has already run and returned, purely to compute
# the probe's own `tt_preserved` diagnostic) evaluates a tree whose root
# is a bare, childless `and()`/`or()`. A match here means `logical_reduce`
# itself already completed successfully in C++ -- the probe's `ok: false`
# is entirely about ITS OWN post-hoc instrumentation, not about whether
# reduction succeeded or what it produced.
_CPP_BUILTIN_TO_BOOL_ASSERT = "should be 'id::logical_true' or 'id::logical_false'"


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

    trees = generate_corpus(args.count, args.seed, args.max_depth, args.arity)
    corpus_path = args.keep_corpus or (
        REPO_ROOT / "tools" / "cpp_probe" / "build" / "_differential_effort_corpus.txt"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(trees) + "\n")
    print(f"Generated {len(trees)} trees (seed={args.seed}) -> {corpus_path}")

    build_probe(force=args.rebuild)
    print(f"Probe: {PROBE_BIN}")

    divergences: list[dict] = []
    crashes: list[dict] = []
    cpp_only_errors: list[dict] = []
    both_errored_differently: list[dict] = []
    root_token_agreements = 0
    counts = {e: 0 for e in EFFORTS}

    for effort in EFFORTS:
        cpp_results = run_probe(corpus_path, effort)
        if len(cpp_results) != len(trees):
            print(
                f"WARNING: effort {effort}: C++ probe returned {len(cpp_results)} "
                f"results for {len(trees)} inputs",
                file=sys.stderr,
            )
        for tree_str, cpp_row in zip(trees, cpp_results, strict=False):
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
            elif cpp_ok and not py_ok:
                crashes.append(
                    {"effort": effort, "input": tree_str, "cpp": cpp_row, "python": py_row}
                )
            elif not cpp_ok and py_ok:
                # KNOWN, non-bug shape (found by this very harness, see
                # the module docstring's item 4 note in task-6-report.md):
                # `reduct_effort_probe.cc` computes `complete_truth_table`
                # on the tree AFTER `logical_reduce` has already run,
                # purely for its own `tt_preserved` diagnostic -- that
                # crashes (`vertex.h:708-713`'s OC_ASSERT) whenever the
                # REDUCED tree's root is a bare childless `and()`/`or()`,
                # even though `logical_reduce` itself already completed
                # and returned normally. If Python's OWN reduced tree is
                # ALSO that exact shape, both sides agree on what
                # `logical_reduce` produced -- the C++ "error" is the
                # probe's own instrumentation, not a `logical_reduce`
                # divergence -- so this is counted as agreement, not a
                # C++-refused/Python-succeeded mismatch. Anything else
                # (Python succeeded with a NON-root-token result while
                # C++'s `logical_reduce` -- inferred from that specific
                # error signature -- also hit this shape) is a genuine,
                # surprising finding and stays in `cpp_only_errors`.
                if (
                    _CPP_BUILTIN_TO_BOOL_ASSERT in cpp_row.get("error", "")
                    and py_row.get("root_token") is True
                ):
                    root_token_agreements += 1
                else:
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
        f"Root-token agreements (both sides' logical_reduce actually agreed; "
        f"C++'s own post-hoc complete_truth_table diagnostic is what crashed, "
        f"not logical_reduce -- not counted as a divergence): {root_token_agreements}"
    )

    for label, bucket in (
        ("divergences", divergences),
        ("crashes", crashes),
        ("cpp_only_errors", cpp_only_errors),
        ("both_errored_differently", both_errored_differently),
    ):
        if not bucket:
            continue
        print(f"\n--- First {label} ---")
        for row in bucket[: args.max_print]:
            print(f"  [effort {row['effort']}] input: {row['input']}")
            print(f"    C++   : {row['cpp']}")
            print(f"    Python: {row['python']}")

    total_problems = len(divergences) + len(crashes) + len(cpp_only_errors)
    total_divergence_count = total_problems + len(both_errored_differently)
    print(f"\nTOTAL DIVERGENCE COUNT: {total_divergence_count}")
    return 0 if total_divergence_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
