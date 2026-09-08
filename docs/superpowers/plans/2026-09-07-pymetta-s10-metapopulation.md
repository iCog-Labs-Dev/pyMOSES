# pyMeTTa S10: Metapopulation and Deme Expansion Plan

**Date:** 2026-09-07
**Status:** Proposed
**Depends on:** S9 (optimization) — hill-climbing, `instance_set`, `neighborhood_sampling`

## Scope

The outer search's memory and its deme factory:

```
metapopulation (pool of scored trees) <-> merging <-> deme_expander -> representation -> optimizer
```

After S10 the port holds a population of scored candidates, merges deme results
into it, and expands a chosen exemplar into a fresh deme. S11 then adds the loop
that drives this.

**Files (8 / 2,620 lines):** `deme/deme_expander.{h,cc}`, `deme/deme_params.h`,
`metapopulation/metapopulation.{h,cc}`, `metapopulation/merging.cc`,
`metapopulation/metapop_params.{h,cc}`.

## The oracle — CORRECTED 2026-09-08: it does NOT cover this slice

`metapopulationUTest` is registered and **does** build and pass (2 tests). My
first version of this plan stopped there and said "use it". That was wrong, and
wrong in the exact way this document warns about elsewhere: **I verified the
oracle exists, not that it covers the scope.**

Its two tests are `test_remove_dominated` and `test_merge_nondominated_rec`
(`tests/moses/metapopulationUTest.cxxtest:74,119`), which exercise
`remove_dominated` and `get_nondominated_disjoint` — both in **`domination.cc`**,
which is P2 and explicitly **not** in this slice's file list. They cover **none**
of Task 2's or Task 3's scope.

**Consequence:** this slice has *no* ready-made oracle. Every task must build its
own bounded C++ probe under `tools/cpp_probe/` (Task 2 did: `metapop_core_probe.cc`).
Do not plan around `metapopulationUTest`.

The general lesson, now also in `docs/CARRY-FORWARD.md`: "the oracle builds and
passes" and "the oracle tests what I am porting" are different claims, and only
the second one licenses using it.

## Measured scope decisions

Every P2 subsystem these files touch is behind a **parameter gate that defaults
off**. Verified in source:

| Subsystem | Gate | Default |
|---|---|---|
| boosting / ensemble | `_params.do_boosting` | `false` (`metapop_params.h:170`) |
| domination | `_params.discard_dominated` | `false` (`metapop_params.h:211`) |
| diversity | `diversity_enabled()` | off |
| subsampling | `subsample_deme_filter_parameters` | off |

Two details worth carrying in, both checked against source rather than inferred:

- **`ensemble` is constructed unconditionally** (`metapopulation.cc:41`,
  `_ensemble(sc, pa.ensemble_params)`) but **read only under boosting** —
  `best_composite_score()` (`metapopulation.cc:207`) returns `_best_cscore`
  outright when `not _params.do_boosting`. So S10 needs an `ensemble` object
  that can be *held*, and which **raises** if actually driven. Ensembles proper
  stay P2.
- **`domination` is NOT on the default path.** `merging.cc:163-175` reads
  `if (not discard_dominated) { insert ALL candidates } else {
  merge_nondominated(...) }` — the `merge_nondominated` call is in the **else**
  arm. With the default `discard_dominated = false` the simple insert-all path
  runs and `domination.cc` is unreachable.

**Design rule for this slice:** port the *gates* faithfully and **raise** inside
the gated bodies — the same discipline S8 used for non-Boolean domains. The
default path is then completely ported, and no P2 subsystem is silently
half-implemented.

## Task 0: Container plumbing `types.py` deliberately skipped

`pymetta/moses/moses/types.py`'s module docstring names exactly what S0–S6 left
out as "metapopulation container plumbing", and it is exactly what S10 needs:

- `scored_combo_tree_set`, `scored_combo_tree_tset`, `scored_combo_tree_ptr_set`
- the functors `sct_score_greater`, `sct_tree_greater`, `scored_combo_tree_hash`,
  `scored_combo_tree_equal`
- `bscore_base::operator()(const scored_combo_tree_set&)` in `scoring_base.py`

Also: `demeID_t` currently lives in `instance_set.py` because S9 Task 0 could not
touch `types.py` (a concurrent agent owned it). Its canonical C++ home is
`types.h:190`. Move it and leave one definition.

**Exit:** set ordering and comparator behaviour match C++ on pinned rows —
`sct_score_greater`'s tie-breaks are load-bearing for exemplar selection.

## Task 1: `metapop_params` and a minimal `ensemble`

Port the parameter struct verbatim, including defaults (they decide which paths
are live). Add the smallest `ensemble` that satisfies the member and
`get_ensemble()`, raising if driven.

**Exit:** defaults byte-match C++; the boosting path raises rather than
silently no-ops.

## Task 2: `metapopulation` core

`init`, `best_candidates`, `best_tree`, `best_composite_score`, `get_trees`,
`update_best_candidates`, `rescore`, and the `_scored_trees` pool.

`score_t` is `typedef float`; exemplar selection depends on
`composite_score::operator<`'s tie-break-by-complexity branch. Narrow exactly as
the 14 existing traced sites do (CARRY-FORWARD §4.2).

**Exit:** the constructor, `init`'s default-path scoring and insertion,
`_scored_trees` ordering and `update_best_candidates`' tie-break match a
purpose-built bounded probe (`tools/cpp_probe/metapop_core_probe.cc`).
`metapopulationUTest` does NOT cover this scope — see the oracle section.

## Task 3: `merging.cc` — the largest file (679 lines)

`merge_candidates` and `merge_demes`. Port the default insert-all path fully;
gate and raise on domination, diversity, boosting and subsampling.

**Exit:** merged pool contents and ordering match C++ for the default
configuration, measured.

## Task 4: `deme_expander`

Single-deme path only. **No feature selection** — `deme/feature_selector.{h,cc}`
(710 lines) is P2 (`--enable-fs`).

Note `deme_expander.h:44,83` takes and stores `optimizer_base& _optimize`; it does
**not** construct the optimizer. The concrete optimizer is injected by the CLI
layer (S13), so S10 wires S9's hill-climbing directly at the test level.

**Exit:** for a fixed seed, the deme produced from a given exemplar matches C++.

## Task 5: Review and closeout

Focused tests, fixture-shape tests, a bounded live differential, Ruff,
`git diff --check`, `bash tools/ci.sh` (which now includes
`tools/coverage_totals.py --check`). Update `docs/ALIGNMENT.md`,
`docs/MODULE-COVERAGE.md` and carry-forward notes with measured results.

## Hazards carried in

- **`optimizationUTest` core-dumps** upstream (`distance_based_scorer` null-derefs
  through `get_rep()` in debug logging). Any oracle driving the optimizer needs a
  throwaway `representation`. Do not "fix" it.
- **`score_t` narrowing** is load-bearing from S9 onward.
- **"Correct but nothing tests it"** — nine occurrences, and S9 Task 2 added a
  tenth in spirit: every `hillclimbing.json` fixture had zero bit fields, so an
  `information_theoretic_bits` regression could not have shown up. Prove guards
  fire by mutation; a passing corpus is not coverage.

## What S10 does NOT do

No feature selection, diversity, domination, ensembles/boosting or subsampling.
No CLI. No contin/action domains. The driving loop is S11.
