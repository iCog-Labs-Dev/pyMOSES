# pyMeTTa S14: continuous and mixed domains

**Date:** 2026-09-08
**Status:** Proposed
**Depends on:** S13 (CLI), and on six prerequisites earlier slices deferred here

## The scoping fact that matters most

The backlog lists S14 as **9 files / 4,057 lines** of `reduct/`. That number is
misleading, and planning to it would guarantee a mid-slice surprise.

**Every earlier slice deferred its continuous half to S14, and none of those
deferrals appear in S14's file list.** Measured from the coverage rows:

| Deferred work | Where it lives | Named in S14's file list? |
|---|---|---|
| `contin_interpreter` / `mixed_interpreter` | `interpreter.py` | no |
| `contin_canonize` / `build_contin` | `build_knobs.py` | no |
| `contin_knob` | `knobs.py` | no |
| continuous knob entries | `knob_mapper.py` | no |
| `flip_LR`, `twiddle_contin_bit`, `generate_contin_neighbor` | `neighborhood_sampling.py` | no |
| `contin_bscore` | `bscores.py` | no |

So S14 is not "port 9 reduct files". It is **thread continuous support through
the interpreter, knob building, sampling and scoring — then port 9 reduct
files.** Sequence it that way or the reduct work will block on missing
machinery.

## A hard prerequisite, verified still open

`pymetta/comboreduct/reduct/logical_rules.py:1288` calls
`tr.equal_subtree(sib1, sib2)` **bare**, while `equal_subtree_vertex_safe`
exists and is used in `meta_rules.py` and `assumption.py`.

This is CARRY-FORWARD §2.5. It is latent only because reduct currently sees
Boolean trees. **The moment contin and builtin vertices co-occur it is live**,
because `contin_t = float` and `builtin` is an `IntEnum`, so
`1.0 == builtin.logical_true` is `True` in Python where C++'s variant
alternatives never compare equal.

**Fix it in Task 0, in the same change that enables contin reduction — not
after.** Both prior Criticals in this project were this exact type confusion.

## Task 0: type-safety prerequisite

Fix `logical_rules.py:1288` and sweep for any other bare `==`/`equal_subtree`
on a vertex that becomes reachable once contin values exist. Pin each with a
test that mixes a `contin_t` and a `builtin` — `1.0` against
`logical_true` is the canonical pair.

**Exit:** no reachable bare vertex comparison remains; each guard proven by a
mixed-type test.

## Task 1: continuous vertex, interpreter and table support

`contin_interpreter` / `mixed_interpreter` (`interpreter.cc`), and whatever
`vertex.py` / `iostream_combo.py` / `table_io.py` need to carry and print
`contin_t` end to end.

**Oracle:** `eval-table` and the `it` problem on `trainingRegression.dat`
(8,000-row contin regression, at the classic repo root). **Trap, already
recorded:** `eval-table`'s positional `$N` arguments are broken — they always
resolve to a zero column regardless of `-L/--labels`. Substitute header names
before invoking it. This nearly poisoned an earlier oracle.

**Exit:** contin trees evaluate and round-trip identically to classic.

## Task 2: continuous representation

`contin_canonize` / `build_contin` (`build_knobs.cc`), `contin_knob`
(`knobs.cc`), the continuous entries in `knob_mapper`, and `contin_spec`'s
already-ported field layout put to real use.

**Hazard, from classic's own CLAUDE.md:** knob building is the runtime hot spot
for wide datasets — `rec_canonize` + `append_linear_combination` add a
multiplier knob for *every* argument, making non-linear contin fitting
O(arity^degree) knobs. `--linear-regression=1` exists to cap that. Keep test
exemplars small and bounded.

**Exit:** the knobs built for a contin exemplar match C++ in count, order and
spec.

## Task 3: continuous sampling and scoring

`flip_LR`, `twiddle_contin_bit`, `generate_contin_neighbor`
(`neighborhood_sampling.cc:32-85`) and `contin_bscore` (`bscores.cc:181`).

`twiddle_contin_bit` and `generate_contin_neighbor` **draw from the RNG**
(`.cc:42,46,72`, the latter through `lazy_random_selector`). Draw *order* is
part of the bit-identity claim S9 established and S12 extended — verify
element-for-element, not set-wise.

**Exit:** for a fixed seed, contin neighbourhoods match C++ element for element.

## Task 4: the reduct rule sets

Now the nine files: `contin_rules.{h,cc}` (1,216), `mixed_rules.{h,cc}` (2,059),
`branch_rules.{h,cc}` (264), `contin_reduction.cc`, `mixed_reduction.cc`,
`full_reduction.cc`. `mixed_rules.cc` (1,765) and `contin_rules.cc` (1,025) are
69% of it.

**Oracle:** `test_reduct` at contin and mixed effort levels — the same binary
S7 used, which is already built.

**`move_ontop` is now fully cleared** — all 11 `contin_rules.cc` call sites were
traced (2026-09-08) and none can hit the self-cycle, because every target is a
freshly appended last child whose `next_sibling` is null. Keep the DO NOT FIX
marker; the bug is real, merely unreachable from these callers.

**Exit:** `test_reduct` agrees at contin and mixed levels across the corpus.

## Task 5: Review and closeout

Ruff, `git diff --check`, `bash tools/ci.sh`, and update `ALIGNMENT.md`,
`MODULE-COVERAGE.md` and carry-forward with measured results. Closeout is part
of the slice.

## Hazards carried in

- **`score_t` narrowing** decides exemplar selection, and contin scores exercise
  far more of the float range than Boolean ones. The S11 review found four
  unprotected sites with measured counterexamples.
- **Fixture rot via the host toolchain** (CARRY-FORWARD §5.1): S12's fixtures
  depend on libstdc++'s sort permutations. Any new contin fixture that sorts
  inherits that.
- **"Correct but nothing tests it"** — fourteen occurrences, most recently a
  self-verifying test that re-derived the code in its own body. Prove guards by
  mutation, and check which paths a corpus actually exercises.
- **Unsigned subtraction** — S9's Critical 1. Check every subtraction; reproduce
  wraps with an explicit mask, never clamp.

## What S14 does NOT do

No feature selection, diversity, domination, ensembles, subsampling, or the
non-hill-climbing optimizers (all P2). No enum, ANN, list/fold, object, message
or procedure-call domains — those are excluded from the port entirely, not
deferred.
