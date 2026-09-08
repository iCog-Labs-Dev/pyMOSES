# Crossover inventory (input for S12)

Assembled 2026-09-07 from classic's `hill-climbing.{h,cc}`, `problem-params.cc`,
`optimization.cc`. S9 ports hill-climbing with `crossover = false`; **S12 turns
it on**. Line numbers were read from source.

## Two field names that do NOT exist

- **`crossover_min_neighborhood`** — the field is **`crossover_min_neighbors`**
  (`unsigned`), CLI `--hc-crossover-min-neighbors`.
- **`fraction_of_remaining`** — the field is **`fraction_of_nn`** (`double`),
  and it is **not part of the crossover condition at all**; its only read is
  `hill-climbing.cc:531` inside `n_new_instances`.

## `hc_parameters` defaults (`hill-climbing.h:41-128`)

`widen_search` false · `single_step` false · **`crossover` TRUE** ·
`crossover_pop_size` 120 · `crossover_min_neighbors` 400 · `max_nn_evals` 20000 ·
`fraction_of_nn` 2.0 · `score_range` 5.0 · `max_allowed_instances` 10000 ·
`resize_to_fit_ram` false · `prefix_stat_deme` "" (set to "Demes" at
`problem-params.cc:2365`).

CLI default `hc_crossover = true` at `problem-params.cc:424`, wired at `:2362`.
Feature selection has its own defaults (`feature-selection.h:63-65`).

## The gate (`hill-climbing.cc:139-152`)

```cpp
bool large_nbh = total_number_of_neighbors >= xover_min_neighbors,  // 400
     xover = hc_params.crossover and (iteration > 2) and !already_xover and
             current_number_of_instances >= xover_min_deme and      // 3, hard-coded
             (large_nbh || last_chance);
```

`iteration` starts at 0 (`:90`), pre-incremented (`:107`), so it is 1 on the
first pass — **crossover cannot fire on iterations 1 or 2**.

### When `large_nbh` is true — verified at `:504-505`

`estimate_neighborhood` (`:486-518`) returns `1` at distance 0,
**`2 * nn_estimate`** at distance 1, `safe_binomial_coefficient` beyond.
`nn_estimate = information_theoretic_bits(fields)` = knob count for a Boolean
field set. So at the steady-state distance 1:

> **`large_nbh` <=> `2 * arity >= 400` <=> arity >= 200 knobs.**

This is why crossover never fired in S9's measurements: `trainingBoolean.dat`
(3 inputs) yields tens of knobs, not hundreds. On such problems crossover fires
**only** via `last_chance`. Distance > 1 happens only with `-L1`/`-T1`.

## `last_chance` — the OR that bypasses the size test

Declared `:103`; set at `:450` ONLY, inside `if (hc_params.crossover)`
(`:434-458`), when there was no improvement, not already last-chance, no
crossover from this center, and widen-search is off or exhausted. It then
`continue`s (`:451`), **skipping the termination checks at `:460-480`**, and the
gate now passes on `|| last_chance` even for a tiny neighborhood. Cleared at
`:457` next pass — one shot per plateau.

`already_xover` (`:100`, set `:168`, cleared `:288`) = "already crossed over from
THIS center"; cleared only when a new center is adopted.

## Why S9's crossover-free port is faithful

With `crossover = false` the whole `:434-458` block is skipped, so `has_improved`
keeps its `:282` value — the same expression `:456` would assign. The values
coincide; only the `continue` and one-shot flag differ. **No silent divergence
in `has_improved`.**

## Functions S12 must port

`cross_top_one` (`h:218`, `cc:593-621`) · `cross_top_two` (`h:223`, `cc:624-661`)
· `cross_top_three` (`h:228`, `cc:664-699`) · `crossover` (`h:235`, `cc:701-731`).

Budget split: `num_to_make = min(budget, crossover_pop_size)`, then
`num_to_make/3 + num_to_make%3` to top-one and `num_to_make/3` to each of the
others — default 120 gives 40/40/40, and **the remainder all goes to top-one**.

### Upstream bug to port, not fix

`cross_top_three`'s inner loop runs **`k < i`** (`:687`) while its index formula
`n = i*(i-1)*(i-2)/6 + j*(j-1)/2 + k` assumes `k < j`. So `n` is not a bijection
for `k >= j` — indices collide or are skipped. Reproduce literally; mark
DO NOT FIX.

## Two asymmetries that will bite a port

1. **Crossover reads the PREVIOUS iteration's block**: `crossover(deme,
   current_number_of_instances, number_of_new_instances, prev_start, prev_size,
   prev_center)` (`:165-166`), while sampling uses `center_inst` and
   `current_number_of_instances` (`:190`). Both update
   `prev_start`/`prev_size`/`prev_center` at `:204-206`.
2. **`distance` is not incremented after a crossover iteration**:
   `} else if (!xover) distance++;` (`:367-368`).

## Minor

- `n_new_instances` (`:520-591`) is passed `current_number_of_instances` where
  its parameter is named `current_number_of_evals` (`:116`), so `nleft` (`:541`)
  counts instances, not evals.
- The "why disabled" chain (`:172-183`) ends in an `OC_ASSERT(false, "There must
  be a bug")` at `:183` that is **unreachable** — `!large_nbh` catches the rest.
- Exhaustive-vs-random is decided later, in `sample_new_instances`
  (`neighborhood_sampling.cc:230-249`).
