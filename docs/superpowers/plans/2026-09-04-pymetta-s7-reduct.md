# pyMoses S7 — Reduct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port classic MOSES's boolean reduction engine so that pyMoses produces byte-identical reduced combo trees to the C++ `test_reduct` binary at reduction effort levels 1, 2 and 3 across the full corpus.

**Architecture:** Mirror the C++ file-for-file under `pymoses/comboreduct/reduct/`, plus the two dependencies the coverage inventory identified (`interpreter/eval.py`, `combo/assumption.py`). The rule *combinators* (`meta_rules`) are ported before any individual rule, because they control traversal order and fixpoint iteration — porting rules first invites an engine that is "morally equivalent" but converges differently.

**Tech Stack:** Python 3.14, stdlib only. pytest 9.1. C++ oracle: `$CLASSIC/build/tests/comboreduct/reduct/test_reduct` (already exists).

## Global Constraints

Identical to the S0–S6 cycle. Restated because they bind every task:

- **Runtime dependencies: Python standard library only.** Dev deps `pytest` + `ruff`.
- **Python 3.14.**
- **C++ identifiers preserved verbatim, including class names.** ruff naming rules N801/N802/N803/N806 are disabled repo-wide for this.
- **Every ported symbol carries a breadcrumb** as its first docstring line: `C++: <path>:<line> <symbol>`. `moses/` paths resolve against `$CLASSIC`, `opencog/` against `$COGUTIL`. Symbols the matcher cannot handle (operators, destructors, templated names, inherited members) cite the **enclosing class** at the member's line and name the real member in the docstring body.
- **Every deliberate deviation carries `# DIVERGENCE(C++ <path>:<line>): <reason and observable impact>`.** "It was easier" is not a reason.
- **Port faithfully; document upstream bugs, never silently fix them.**
- **`$CLASSIC`** = `/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses`; **`$COGUTIL`** = `/home/yab/.../OpenCog/CogUtil`.
- **Every commit authored `Yab <yabuman46@gmail.com>`. No trailers of any kind.**
- **Never claim a step passes without running it and reading the real output.**
- **MEMORY: this machine has been OOM-killed twice. Bound every C++ binary run: `( ulimit -v 1048576; timeout 20 <cmd> )`.** Never unbounded.
- `bash tools/ci.sh` self-bootstraps and must exit 0 before any commit.

## Required reading before starting

- `docs/CARRY-FORWARD.md` — §1 is Task 0's scope; §3 lists traps in classic and its tooling; §4 lists settled questions that must not be relitigated.
- `docs/MODULE-COVERAGE.md` — the S7 rows, with the reason each file is in scope.

## File Structure

| File | Responsibility |
|---|---|
| `pymoses/comboreduct/interpreter/eval.py` | `eval_binding` / `eval_throws_tree` — the general (non-boolean) evaluator `general_rules` calls |
| `pymoses/comboreduct/combo/assumption.py` | Assumption set used by the meta and logical rules |
| `pymoses/comboreduct/reduct/reduct.py` | `rule` base class and the reduction entry points |
| `pymoses/comboreduct/reduct/meta_rules.py` | `downwards`, `upwards`, `sequential`, `iterative` — traversal and fixpoint combinators |
| `pymoses/comboreduct/reduct/general_rules.py` | Type-independent rules |
| `pymoses/comboreduct/reduct/flat_normal_form.py` | Flat clause-set representation the logical rules operate on |
| `pymoses/comboreduct/reduct/logical_rules.py` | Boolean rules, including `subtree_to_enf::reduce_to_enf` |
| `pymoses/comboreduct/reduct/logical_reduction.py` | Assembles the rule set per effort level |
| `pymoses/comboreduct/reduct/clean_reduction.py` | `clean_reduce` — removes `null_vertex` sentinels |

---

### Task 0: Pre-work — the three carry-forward items S7/S8 depend on

**Files:** Modify `pymoses/comboreduct/table/table.py`, `table_io.py`, `pymoses/util/tree.py`; tests alongside.

These are from `docs/CARRY-FORWARD.md` §1. They are done first because later tasks build on them.

- [x] **Step 1: `Table.__init__` must copy, not alias**

`table.py:574` stores `otable`/`itable` by reference; `table.cc:485` copies both. The reason this was parked is that `_load_table_old` (`table_io.py:614-619`) depends on the aliasing to observe `delete_column`.

**That trap dissolves**: C++'s `istreamTable_OLD` never calls that constructor — it fills `tab.itable` **in place** (`table_io.cc:860`). Read that code, mirror it in `_load_table_old` (construct an empty `Table`, then assign `table.itable`), and then make the constructor copy.

Write the failing test first: construct a `Table` from an `ITable`, mutate the original `ITable`, assert the `Table` is unchanged. Run it, watch it fail, then fix.

- [x] **Step 2: decide and implement `erase()` / `replace()` handle semantics**

`tree.py:659` and `replace()` leave the discarded node's own `parent`/`next_sibling` live, pointing into the tree, where C++ dangles. S8's `knob_mapper` stores one iterator per knob, so ported knob code would silently reattach where C++ crashes.

Null the discarded node's links so a stale handle is obviously dead rather than quietly wrong, and add a test asserting a handle to an erased node raises rather than navigating. Record a `DIVERGENCE` — C++ dangles, we fail fast; that is a deliberate, safer difference, not an accident.

- [x] **Step 3: add a subtree-rooted post-order walk**

`reduct::upwards` (`meta_rules.cc:111`) uses `post_order_iterator at=it, end=it; ++end; at.descend_all();` — a walk rooted at a subtree, which `iter_post()` does not offer. The private `_postorder_step` makes this cheap. Task 2 needs it.

- [x] **Step 4: full CI, then commit**

Run `bash tools/ci.sh`; expect 0. Commit as `fix(foundation): pre-work for S7 — Table copy semantics, dead handles, subtree post-order`.

---

### Task 1: `interpreter/eval.py` and `combo/assumption.py`

**Files:** Create `pymoses/comboreduct/interpreter/eval.py`, `pymoses/comboreduct/combo/assumption.py`; tests alongside.

**Interfaces produced:** `eval_binding(bmap, tr)`, `eval_throws_binding(bmap, tr)`, `eval_throws_tree(tr)` mirroring `$CLASSIC/moses/comboreduct/interpreter/eval.h`; an assumption-set type mirroring `combo/assumption.h`.

**Why this task exists:** the coverage inventory found that `general_rules.cc` calls `eval.cc`, and every reduct cxxtest requires it. The original S7 scope omitted both. `assumption.{cc,h}` is used by `meta_rules.cc`, `general_rules.cc` and `logical_rules.cc`.

- [x] **Step 1:** Read `$CLASSIC/moses/comboreduct/interpreter/eval.{h,cc}` and `combo/assumption.{h,cc}` in full. Note what Task 4 already ported (`boolean_interpreter`) and do **not** duplicate it — `eval` is the general evaluator, a different entry point.
- [x] **Step 2:** Write failing tests covering: argument binding, a tree with no arguments, an unbound argument (must raise as C++ throws), and the `assumption` set operations.
- [x] **Step 3:** Run them; confirm they fail for the right reason.
- [x] **Step 4:** Implement both modules.
- [x] **Step 5:** Run tests; then `bash tools/ci.sh` to 0.
- [x] **Step 6:** Commit as `port(comboreduct): eval and assumption — S7 prerequisites`.

---

### Task 2: `reduct.py` and `meta_rules.py` — the combinators, before any rule

**Files:** Create `pymoses/comboreduct/reduct/reduct.py`, `meta_rules.py`; tests alongside.

**Interfaces produced:** `rule` base class with the C++ call signature; combinators `downwards`, `upwards`, `sequential`, `iterative` from `meta_rules.h:177`/`meta_rules.cc:156`.

**This ordering is deliberate and must not be inverted.** The combinators decide traversal direction, where each rule is applied, and when the fixpoint terminates. A rule set ported before its combinators can pass rule-level tests and still converge to a different normal form than C++ — the exact failure this cycle exists to prevent.

`upwards` needs the subtree-rooted post-order walk added in Task 0 Step 3.

- [x] **Step 1:** Read `meta_rules.{h,cc}` and `reduct.{h,cc}` in full, and `$CLASSIC/doc/` if it says anything about rule application order.
- [x] **Step 2:** Write failing tests that assert *traversal order* directly — e.g. a probe rule that records the node sequence it visits, asserted against the order C++ produces. Order is the property that matters here, not the end result.
- [x] **Step 3:** Run; confirm failure. **Step 4:** Implement. **Step 5:** Run; `ci.sh` to 0.
- [x] **Step 6:** Commit as `port(reduct): rule base and the meta-rule combinators`.

---

### Task 3: `general_rules.py` and `flat_normal_form.py`

**Files:** Create both under `pymoses/comboreduct/reduct/`; tests alongside.

`flat_normal_form` is the clause-set representation the logical rules operate on; `general_rules` are the type-independent rules. **`argument`'s ordering is by absolute value** (`docs/CARRY-FORWARD.md` §4.4) and `flat_normal_form`'s clause ordering depends on it — do not substitute a natural sort.

- [x] **Step 1:** Read `general_rules.{h,cc}` and `flat_normal_form.{h,cc}` in full.
- [x] **Step 2:** Write failing tests, including one asserting clause ordering for a set containing both `$1` and `!$1`.
- [x] **Step 3:** Run; confirm failure. **Step 4:** Implement. **Step 5:** Run; `ci.sh` to 0.
- [x] **Step 6:** Commit as `port(reduct): general rules and flat normal form`.

---

### Task 4: `logical_rules.py`

**Files:** Create `pymoses/comboreduct/reduct/logical_rules.py`; tests alongside.

The largest and highest-risk file in the slice (860 lines), containing `subtree_to_enf::reduce_to_enf` — a single ~250-line function implementing 0-subsumption, 1-subsumption, complement subtraction and iterative constraint promotion, with a three-valued `Delete`/`Disconnect`/`Keep` result.

**Do not decompose it into separate named sub-rules.** The `metta-moses` port split this function into 12 files and it caused two documented bugs (issues #502/#503) by losing track of the *order* and *scope* of operations. Port it as one function, in C++ order.

`MOSES/elegant-normal-forms-python`'s `Utilities/ReduceToElegance.py` is a working, tested transliteration of this same state machine and is a legitimate reading aid — but its data structure is unusable here, so do not import from it.

- [x] **Step 1:** Read `logical_rules.{h,cc}` in full, paying attention to the order of operations inside `reduce_to_enf`.
- [x] **Step 2:** Write failing tests from the C++ cxxtest `$CLASSIC/tests/comboreduct/reduct/BooleanReductUTest.cxxtest` and `BooleanFullUTest.cxxtest`, preserving test names.
- [x] **Step 3:** Run; confirm failure. **Step 4:** Implement. **Step 5:** Run; `ci.sh` to 0.
- [x] **Step 6:** Commit as `port(reduct): logical rules including reduce_to_enf`.

---

### Task 5: `logical_reduction.py` and `clean_reduction.py`

**Files:** Create both under `pymoses/comboreduct/reduct/`; tests alongside.

`logical_reduction` assembles the rule set per effort level (1/2/3) — this is what `problem-params.cc:2385-2389` calls to build `bool_reduct`/`bool_reduct_rep`. `clean_reduce` removes `null_vertex` sentinels and is called from `logical_rules.cc:848` and, later, S8's `representation.cc:264`.

- [x] **Step 1:** Read `logical_reduction.cc` and `clean_reduction.cc` in full.
- [x] **Step 2:** Write failing tests asserting that each effort level assembles the rule set C++ assembles, in the same order.
- [x] **Step 3:** Run; confirm failure. **Step 4:** Implement. **Step 5:** Run; `ci.sh` to 0.
- [x] **Step 6:** Commit as `port(reduct): logical_reduction effort levels and clean_reduce`.

---

### Task 6: The oracle — differential testing against `test_reduct`

**Files:** Create `tools/capture_reduct.py`, `tests/fixtures/reduct.json`, `tests/comboreduct/reduct/test_reduct_oracle.py`; lift assets from `elegant-normal-forms-python`.

This is the slice's exit criterion.

- [x] **Step 1: Build the oracle binary.** `$CLASSIC/tests/comboreduct/reduct/test_reduct.cc` already exists; build it via `cd $CLASSIC/build && make test_reduct`. It reduces one tree and prints before/after size plus a truth-table equivalence check. Run it **bounded**.
- [x] **Step 2: Assemble the corpus.** Three sources: the 29 trees in `$CLASSIC/testTrees.sh`; every tree literal in `BooleanReductUTest`, `BooleanFullUTest` and `FullReductUTest`; and the **44 curated expressions** in `MOSES/elegant-normal-forms-python/Tests/GeneralTestCases.py` (which include the Holman p.19 and Moshe Looks examples). Capture C++ output at effort levels **1, 2 and 3** for every runnable tree; retain S14-dependent literals as explicit skipped cases.
- [x] **Step 3: Lift the ENF verification assets.** Port `Tests/TestHelpers.py`'s truth-table generation and comparison, and `Tests/EnfRuleCheckers.py`'s seven ENF axiom checkers, rewritten against pyMoses's `combo_tree`. These compare semantic outcomes to C++ and verify the structural ENF result without assuming classic always preserves truth tables.
- [x] **Step 4: Assert exact equality.** Python's reduced tree must match C++'s byte-for-byte at each effort level. Add a fixture-shape guard asserting the corpus is non-empty and holds the expected number of cases, per the vacuous-test lesson in `docs/CARRY-FORWARD.md` §6.
- [x] **Step 5:** `ci.sh` to 0. **Step 6:** Commit as `test(reduct): differential oracle against test_reduct at effort 1/2/3`.

---

## Cycle exit criterion

All verified by running the command and reading the output:

1. `bash tools/ci.sh` exits 0.
2. For every tree in the corpus, at effort levels 1, 2 and 3, pyMoses's reduced tree is **byte-identical** to `test_reduct`'s.
3. Python's truth-table outcome matches C++'s measured
   `tt_computed`/`tt_preserved` result for every runnable row, and effort-3
   output satisfies all seven ENF axioms. Classic itself is semantically lossy;
   the local childless-junctor input-shape indicator has both false positives
   (parent masking) and false negatives (earlier rules can create the shape).
   See `docs/CARRY-FORWARD.md` §3. The live differential harness therefore
   compares both implementations' semantic diagnostics directly and retains
   targeted examples of both classifier limitations.
4. `docs/ALIGNMENT.md` shows every S7 file's symbols as ported or explicitly diverged.
5. `docs/MODULE-COVERAGE.md` updated: the 16 S7 rows move from `PLANNED: S7` to `PORTED` or `PARTIAL` with a named gap.

## Self-review notes

**Coverage.** Task 0 ↔ CARRY-FORWARD §1. Task 1 ↔ the `eval.cc` ordering hazard plus `assumption.*`. Tasks 2–5 ↔ the nine reduct files. Task 6 ↔ the exit criterion.

**Known gap, deliberate:** `branch_rules`, `contin_rules`, `mixed_rules`, `full_reduction` are S14, not S7 — the boolean path does not need them. `fold_reduction`/`fold_rules` and `ann_reduction`/`ann_rules` are explicitly EXCLUDED under the boolean/continuous/action-only domain scope, not deferred work.
