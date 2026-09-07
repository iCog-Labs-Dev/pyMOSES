# pyMeTTa S9: Serial Hill-Climbing Plan

**Date:** 2026-09-07
**Status:** Proposed; awaiting approval
**Depends on:** completed S7 (reduct) and S8 (boolean representation), both merged to `main`

## Scope

Close the loop from a packed instance to a *scored* instance, and search over
them:

```
representation -> instance_set -> scored_instance -> instance_scorer
              -> neighborhood_sampling -> hill-climbing (serial)
```

S8 deliberately stopped at `get_candidate`. S9 adds the representation-to-
optimizer bridge the S8 plan named, plus the first actual *search*. After S9 the
port can improve a candidate; it still cannot run `moses -i data.csv` end to end
(that is S11, after S10's metapopulation).

## Why this ordering

The alternative the S8 plan offered was continuous representation. Hill-climbing
first is the better call for one reason: **S9 is where the bit-identical RNG
claim finally gets exercised.** It was proved empirically in S0–S6 and has been
carried, unused, ever since — `neighborhood_sampling.cc` is the first ported
call site that draws from it (2 call sites; `hill-climbing.cc` itself draws
none). A latent RNG divergence discovered at S11, after metapopulation and CLI
are built on top, is far more expensive than one found here.

The continuous slice remains next after this, and CARRY-FORWARD §2.5 is a
prerequisite for it, not for S9.

## Fidelity rules

- Same `-r <seed>` must give the same instance sequence. This is the slice's
  central claim; everything else is secondary.
- `score_t` is `typedef float` (binary32). Narrowing is already applied at 14
  traced sites and demonstrably changes exemplar selection (20,656 inverted
  pairs at the default `complexity_ratio 3.5`). Every new score comparison
  narrows the same way.
- Port classic's iteration and termination conditions verbatim, including its
  off-by-one and early-exit quirks.

## Task 0: The bridge containers

Port `instance_set`, `scored_instance`, `instance_scorer`.

`instance.__hash__` is currently `None` by deliberate choice (S8). C++
specializes `std::hash<instance>` (`instance.h:49-62`) precisely so `instance`
can key an `unordered_map`. Decide explicitly: implement the same hash, or
document why the port does not need it. Do not leave `__hash__ = None`
unexamined — MODULE-COVERAGE's `instance.h` row names this as the gap S9 closes.

**Exit:** container ordering and scored-instance comparison match C++ on pinned
fixture rows.

## Task 1: `neighborhood_sampling`

Port `moses/moses/neighborhood_sampling.{h,cc}`.

This is the RNG-sensitive task. `cogutil`'s `RandGen` **is** `std::mt19937`, and
`randint()` is `raw >> 1`; the port must use `init_genrand`, not Python's
`init_by_array`. Sampling order, not just the sampled set, must match.

**Oracle:** `build/tests/moses/neighborhood_test <distance> <multiplicity>
<knob…>` dumps the instances generated for a synthetic all-disc `field_set`.
**Note: this binary is registered (`tests/moses/CMakeLists.txt:130`) but is NOT
currently built** — the spec's claim that it "exists" is stale. Build it first
(`make neighborhood_test`), and check `ldd` before trusting it: the prebuilt
binaries silently rotted once when boost 1.91 disappeared.

**Exit:** for a fixed seed, the generated instance sequence matches C++
element-for-element, at several distances and multiplicities.

## Task 2: Serial hill-climbing, no crossover

Port `optimization/hill-climbing.{h,cc}` and the `optimization.{h,cc}` scaffolding
it needs, with `crossover = false`.

**Trap, and the reason this task needs care: classic's default is
`crossover = true`** (`problem-params.cc:424`, `hc_crossover_opt`). A trajectory
diff against a default-configured `moses` will therefore NOT match a
crossover-free port, and the mismatch will look like a porting bug. Every C++
oracle run in this task must pass `--hc-crossover=0`. Crossover itself is S12;
read `doc/cross_top_two_explanation.md` and `doc/cross_top_three_explanation.md`
before attempting it.

Do not port `star-anneal`, `univariate` or `particle-swarm` (P2), and do not
port the distributed or MPI paths (`mpi_moses.cc` compiles to nothing —
`FIND_PACKAGE(MPI)` is commented out).

**Exit:** for a fixed seed and `--hc-crossover=0`, the sequence of evaluated
instances and their scores matches C++ step for step on small Boolean problems.

## Task 3: Review and closeout

Focused unit tests, fixture-shape tests, a bounded live differential, Ruff,
`git diff --check`, `bash tools/ci.sh` (now including
`tools/coverage_totals.py --check`). Update `docs/ALIGNMENT.md`,
`docs/MODULE-COVERAGE.md` and carry-forward notes with measured results.

## Known hazards carried in

- **`neighborhood_test` is not built.** See Task 1.
- **Crossover default.** See Task 2. This is the most likely source of a
  spurious "port is wrong" conclusion in this cycle.
- **`score_t` narrowing** (CARRY-FORWARD §4.2) becomes load-bearing here: this
  is the first slice that *compares* scores to make decisions.
- **The dominant failure mode of every cycle so far** is "correct but nothing
  tests it" — nine occurrences, most recently an entire untested branch in S8's
  `get_candidate_rec` that contained a live aliasing defect. Prove each guard
  fires by mutation; a passing corpus is not coverage.

## What S9 does NOT do

No metapopulation, no deme expansion, no feature selection, no CLI, no
continuous or action domains, no crossover, no ensembles or diversity. The first
end-to-end `moses -i data.csv -u y` remains S11.
