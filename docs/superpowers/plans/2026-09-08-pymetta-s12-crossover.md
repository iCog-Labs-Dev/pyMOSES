# pyMeTTa S12: Hill-climbing crossover

**Date:** 2026-09-08
**Status:** Proposed
**Depends on:** S9 (hill-climbing), S11 (end-to-end loop, for trajectory diffs)

## Why this slice exists

S9 ported hill-climbing with `crossover = False`. **Classic's default is
`true`** (`problem-params.cc:424`), so until S12 lands the port's search is
genuinely weaker than classic's default configuration — not merely differently
configured. A trajectory diff against a default-configured `moses` cannot match
before this.

**All of the C++ groundwork is already done** and committed as
`docs/CROSSOVER-INVENTORY.md`. **Read it first** — it is the product of an
exhaustive source read and contains the defaults table, the exact gate, the
budget split, two asymmetries that will bite, and an upstream bug to reproduce.
Do not re-derive it.

**Files:** no new files. This slice fills in `hill_climbing.py`'s deliberately
unported crossover surface: `cross_top_one`, `cross_top_two`, `cross_top_three`,
`crossover`, and the `last_chance` control flow.

## What the inventory already settles

- The gate (`hill-climbing.cc:147-150`) — five conjuncts, with the size test
  OR'd against `last_chance`.
- **`large_nbh` ⟺ arity ≥ 200** for a Boolean field set, verified at
  `hill-climbing.cc:504-505` (`estimate_neighborhood` returns `2 * nn_estimate`
  at distance 1). This is why crossover never fired in S9's measurements on
  `trainingBoolean.dat`. **Consequence for testing:** a small Boolean problem
  reaches crossover *only* through `last_chance`. To exercise the `large_nbh`
  path at all you need ~200 knobs, or `-L1`/`-T1` to push distance above 1.
- `last_chance` is one-shot per plateau: set at `:450`, `continue`s at `:451`
  skipping the termination checks at `:460-480`, cleared at `:457` next pass.
- Budget split: `min(budget, crossover_pop_size)` then `n/3 + n%3` to top-one
  and `n/3` to each of the others — **the remainder all goes to top-one**.

## Task 0: `cross_top_one` / `two` / `three`

Port the three verbatim, including their `partial_sort` bounds
(`sqrtf(2*num_to_make)+3`, `cbrtf(6*num_to_make)+3`) and clamps.

**Reproduce, do not fix:** `cross_top_three`'s inner loop runs `k < i`
(`hill-climbing.cc:687`) while its index formula
`n = i*(i-1)*(i-2)/6 + j*(j-1)/2 + k` assumes `k < j`, so `n` is not a bijection
for `k >= j` — indices collide or are skipped. Mark DO NOT FIX.

`std::partial_sort` is **not stable**, and its tie order decides which instances
get crossed. Determine what libstdc++ actually does for the tie cases the tests
hit, and reproduce that ordering — or, if it proves genuinely unreproducible,
document it as a DIVERGENCE with the measured consequence. Do not assume ties
are unreachable.

**Exit:** for a fixed deme and base instance, each function's output instances
match C++ element for element.

## Task 1: the gate and `last_chance`

Wire the five-conjunct gate and the `:434-458` block.

Two asymmetries from the inventory, both easy to get wrong:

1. **Crossover reads the *previous* iteration's block** — `crossover(deme,
   current_number_of_instances, number_of_new_instances, prev_start, prev_size,
   prev_center)` (`:165-166`) — whereas the sampling branch uses `center_inst`
   and `current_number_of_instances` (`:190`). Both then update
   `prev_start`/`prev_size`/`prev_center` at `:204-206`.
2. **`distance` is not incremented after a crossover iteration**:
   `} else if (!xover) distance++;` (`:367-368`).

S9's port already contains the `:434-458` block ported verbatim with crossover
disabled, and the S9 review confirmed that port sound — build on it rather than
rewriting.

**Exit:** the gate fires on exactly the iterations C++ fires on, measured over a
run, including the `last_chance` path on a small problem.

## Task 2: trajectory parity with crossover ON

Re-run S11's trajectory diff with `--hc-crossover=1` (the default) and confirm
step-for-step agreement.

This is the slice's real exit criterion: **after S12, the port matches classic's
default configuration**, not a special-cased one.

Remember S9's lesson — its oracle matched 4,500 instances while walking a single
control-flow path because `has_improved` was true on every iteration. Verify
which paths the corpus actually exercises: assert that crossover *fired*, and on
which iterations, rather than inferring it from a matching score.

## Task 3: Review and closeout

Ruff, `git diff --check`, `bash tools/ci.sh`, `docs/ALIGNMENT.md`,
`docs/MODULE-COVERAGE.md`, carry-forward notes. Remove S9's crossover-disabled
guard and its DIVERGENCE marker, and update `docs/CROSSOVER-INVENTORY.md` to say
it is now implemented.

## Hazards

- **The `last_chance` path is the only way small problems reach crossover.** A
  test suite built solely on small Boolean problems will exercise one of the two
  entry paths. Cover both.
- **Non-determinism through sort ties** — see Task 0.
- **`hc_parameters` defaults must be pinned** before relying on them; S9's review
  found every one unpinned, including `cross=True` itself.
