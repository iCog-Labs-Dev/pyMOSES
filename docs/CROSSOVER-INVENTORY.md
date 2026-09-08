# Crossover inventory (input for S12)

**STATUS: IMPLEMENTED (S12, 2026-09-08).** Everything below is now ported in
`pymetta/moses/moses/optimization/hill_climbing.py` -- `cross_top_one`,
`cross_top_two`, `cross_top_three`, `crossover`, the five-conjunct gate and
the `last_chance` block -- and `hc_parameters(cross=True)`, classic's own
default, is the configuration the port runs. The oracle is
`tools/cpp_probe/crossover_probe.cc` -> `tests/fixtures/crossover.json`,
consumed by `tests/moses/moses/optimization/test_crossover.py`. S9's
`NotImplementedError` guard and its DIVERGENCE marker are gone.

Two claims below were CORRECTED by measurement during S12; both corrections
are marked inline. Read them, not the original claim.

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
(3 inputs) yields tens of knobs, not hundreds.

> **CORRECTION (S12, measured).** The original next sentence read "On such
> problems crossover fires **only** via `last_chance`. Distance > 1 happens
> only with `-L1`/`-T1`." Both halves are wrong.
>
> `distance++` at `:367-368` runs BEFORE the `last_chance` block at `:434`,
> so distance reaches 2 on any non-improving iteration -- no `-L1`/`-T1`
> needed -- and a `last_chance` crossover therefore always happens at
> `distance >= 2`, where `estimate_neighborhood` returns
> `safe_binomial_coefficient(nn, distance)`, not `2 * nn`. For a Boolean
> field set that is already `>= 400` at distance 2 once `nn >= 29`
> (`C(29,2) = 406`). So on most "small" problems the gate is opened by
> `large_nbh`, not `last_chance`, and classic's own log says so: the
> arity-6 case in `tests/fixtures/crossover.json` logs
> `n'bhood too large for exhaustive srch: 703>=400`.
>
> `last_chance` is the DECIDING conjunct only when
> `safe_binomial_coefficient(nn, 2) < 400`, i.e. `nn <= 28` -- about 17
> multiplicity-3 knobs, or an arity-4 exemplar. That is the
> `last_chance_only` fixture case, and it is the only one classic logs
> `last chance` for.

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
for `k >= j`. Reproduced literally, marked DO NOT FIX.

> **CORRECTION (S12, derived and confirmed against the fixture).** "indices
> collide **or are skipped**" -- nothing is ever skipped. For a fixed `i`
> the offsets produced are the union over `j` in `[1,i)` of the `i`
> consecutive values `[j(j-1)/2, j(j-1)/2 + i)`; successive runs start `j`
> apart with `j < i`, so they overlap and the union is the **contiguous**
> range `[0, i(i-1)/2]`. The intended range is `[0, i(i-1)/2)`, so each
> block covers all its own slots and then spills exactly one write into the
> next block's slot 0 (which that block later rewrites). Every slot below
> `num_to_make` is written; many are written repeatedly (up to O(sqrt(i))
> times) and **the last write wins** -- with a `(j, k)` pair the formula
> never intended, e.g. `n = 3` is written as `(i=3, j=1, k=2)` and then as
> `(i=3, j=2, k=1)`. Because `k` may equal `j` or `i`, some "three-way"
> crosses are really two-way. The early `return` at `:689` still leaves no
> hole: the covered offsets form a contiguous prefix at all times, so the
> first `n >= num_to_make` is only reached after every smaller slot has been
> written.

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


## What S12 measured that this document did not predict

* **`std::partial_sort` tie order is reachable and decisive.** Two
  `composite_score`s tie under `operator<` whenever penalized score AND
  complexity are equal -- the normal state of a Boolean deme. libstdc++'s
  order was measured over 15 samples and reproduced exactly by
  `hill_climbing.py`'s `_partial_sort` (a transliteration of
  `__heap_select` + `__sort_heap`). A stable Python sort measurably
  differs: for eight all-tied elements with `mid = 4`, libstdc++ returns
  `[1, 3, 0, 2, ...]` where a stable sort returns `[0, 1, 2, 3, ...]`.
  This also retired the same "a stable sort is at least as well-ordered"
  argument in `resize_deme`, where the sort is immediately followed by a
  truncation and therefore decides which tied instances survive.
* **`sqrtf`/`cbrtf` need C's `float` conversions reproduced.**
  `sqrtf(2 * num_to_make) + 3` converts the `size_t` to binary32 BEFORE
  the call and keeps the result in binary32. Measured counterexamples where
  binary64 gives a different truncated integer: `n = 8380417` (4097 vs
  4096) and, for `cbrtf(6n) + 3`, `n = 26244` (57 vs 56).
* **`(iteration > 2)` is redundant.** `current_number_of_instances >= 3`
  already excludes iterations 1 and 2, because iteration 1 runs at
  distance 0 and appends exactly one instance. Relaxing it to
  `iteration > 1` changes nothing measurable.
* **Crossover ON changes the end-to-end trajectory.** On S11's own
  five-generation `trainingBoolean.dat` scenario, `n_evals` per generation
  is 45/58/74/**198**/**214** with crossover at its default versus
  45/58/74/97/113 with it off.
