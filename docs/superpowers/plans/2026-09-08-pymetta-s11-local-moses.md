# pyMoses S11: `local_moses` — first end-to-end run

**Date:** 2026-09-08
**Status:** Proposed
**Depends on:** S10 (metapopulation + deme_expander)

## Scope

The outer loop that drives everything built so far:

```
local_moses -> expand_deme -> deme_expander -> representation -> hill-climbing
            -> merge back into metapopulation -> repeat until a termination condition
```

**This is the milestone.** After S11 the port learns Boolean programs end to end
and its trajectory can be diffed step-for-step against classic configured the
same way.

**Files (5 / 881 lines):** `moses/moses/moses_main.{h,cc}` (391 + 117),
`moses/moses/local_moses.{h,cc}` (46 + 225), `moses/moses/moses_params.h` (102).

## The loop, read from source (`local_moses.cc`)

```cpp
auto not_max_evals = [&]() { return stats.n_evals < pa.max_evals; };
auto not_max_gens  = [&]() { return pa.max_gens != stats.n_expansions; };
auto not_max_score = [&]() { return mp.best_score() < pa.max_score; };
auto not_max_time  = [&]() { return stats.elapsed_secs < pa.max_time; };

while (not_max_evals() and not_max_gens()
       and not_max_score() and not_max_time()
       and !premature_termination)
{
    premature_termination =
        expand_deme(mp, dex, pa.max_evals - stats.n_evals,
                    pa.max_time - stats.elapsed_secs, stats);
    ...
}
```

Note `not_max_gens` uses `!=`, not `<`. Port it verbatim; if `n_expansions` could
ever overshoot, `!=` never terminates where `<` would. Do not "fix" it.

Note also `pa.max_evals - stats.n_evals` is an unsigned subtraction, guarded by
`not_max_evals()` above it. **S9's Critical 1 was exactly this shape** — a
`size_t` underflow that Python does not reproduce (`hill-climbing.cc:541`). Check
every subtraction in this slice for the same hazard, and reproduce any wrap with
an explicit mask rather than clamping.

## The oracle, already built and mapped

`MOSESUTest` is registered and built. **It is not fully green in classic**:
`test_ps_2_parity` fails (`moses-framework.h:169`) because it uses `-aps`
(particle swarm), a P2 optimizer S11 does not port. Do not chase it.

**S11-usable cases** (default optimizer = hill-climbing, Boolean):
`test_2_parity`, `test_2_parity_with_labels`, `test_3_parity_multi_thread`,
`test_1_mux`, `test_2_mux`, `test_it`, `test_it_target_ignore`,
`test_cp_simple`, `test_cp`.

Out of scope here: `test_sa_*`/`test_ps_*`/`test_un_*` (P2 optimizers),
`test_4/5_parity_boosting` (P2), `test_kl`/`test_multi_kl`/`test_precision`/
`test_negative_predictive_value` (P2 scorers), `test_cp_contin`/`test_sr_3`
(S14), `test_output_labels`/`test_python_output` (S13).

Full map in `docs/CARRY-FORWARD.md`.

## Task 0: `moses_params` and `moses_statistics`

Port the parameter and statistics structs verbatim, defaults included. Pin every
default with a test — S9's review found *every* `hc_parameters` default
unpinned, including the one its own DIVERGENCE depended on.

**Exit:** defaults byte-match C++.

## Task 1: `expand_deme` and the `local_moses` loop

Port the loop above and `expand_deme`, verbatim including `!=`.

`print_stats_header` and the `Stats:` line are **output formatting = S13**. Port
the loop's *control flow* and the statistics it accumulates (`n_evals`,
`n_expansions`, `elapsed_secs`); do not reproduce the log text. Keep the
accumulation faithful — `stats.elapsed_secs` feeds a termination predicate, so it
is behaviour, not logging.

**Exit:** for a fixed seed, the sequence of expansions, `n_evals` after each, and
the metapopulation's best score after each match C++ step for step.

## Task 2: `moses_main` and wiring

The entry point, with hill-climbing injected directly.

`moses_main.h:37-41` `#include`s `star-anneal.h`, `univariate.h`,
`particle-swarm.h` and `ss_bscore.h` — **all P2**. Verified: `moses_main.cc` and
`local_moses.cc` contain **zero** references to any of them (grep count 0). A C++
`#include` is a compile dependency; Python has no equivalent, so port only the
reachable paths.

Optimizer selection is **dependency-injected**: `deme_expander.h:44,83` takes and
stores `optimizer_base& _optimize` and never constructs one. The concrete
optimizer is built by the CLI layer (S13), so S11 wires S9's hill-climbing
directly.

`distributed_moses.h` is included but is a *different* mechanism (`-j host:N`,
shells out to `ssh`) — out of scope. `mpi_moses.cc` compiles to nothing
(`FIND_PACKAGE(MPI)` is commented out of CMake).

**Exit:** the S11-usable `MOSESUTest` cases reproduce — same final score and, for
at least two of them, the same winning tree.

## Task 3: Review and closeout

Focused tests, fixture-shape tests, a bounded live differential, Ruff,
`git diff --check`, `bash tools/ci.sh` (includes `coverage_totals.py --check`).
Update `docs/ALIGNMENT.md`, `docs/MODULE-COVERAGE.md` and carry-forward notes
with measured results.

## Hazards carried in

- **Unsigned subtraction.** See above; S9's Critical 1 was this exact shape and
  truncated already-scored instances.
- **`optimizationUTest` core-dumps upstream** (`distance_based_scorer` null-derefs
  through `get_rep()` in debug logging). Any oracle driving the optimizer
  directly needs a throwaway `representation`.
- **`score_t` narrowing** decides exemplar selection. `MOSESUTest`'s own output
  prints `-3.57142853736877441` — a binary32 value at double width, classic
  demonstrating the effect.
- **"Correct but nothing tests it"** — ten occurrences. S9's trajectory oracle
  matched 4,500 instances yet walked a single control-flow path, because
  `has_improved` was true on every iteration. **A large matching corpus is not
  coverage.** Prove guards fire by mutation, and check which paths the corpus
  actually exercises before trusting it.

## What S11 does NOT do

No crossover (S12), no CLI or output formatting (S13), no contin/action (S14),
no feature selection, diversity, domination, ensembles or subsampling (P2), no
distributed or MPI.
