# Phase 1l findings: mux11 at equal time after the speed-ups

**Status:** complete. 20 runs, all `ok`, on 2026-10-15 from 10:36 to 11:43. Stage 1 decided
the result, so no extension ran. Protocol: `PHASE1L-PREREGISTRATION.md`. Code `a42d5c8`.
Results: `phase1l-results/` and `../gomea-results/1/phase1l-mux11-*/`.

**Answer:** no. On the sped-up code GOMEA is still **worse** than hill climbing (HC) on mux11
at equal wall-clock time. The speed-ups helped HC more than GOMEA, so the gap widened.

## Results (10 seeds, 3,600 s per run)

| | Phase 1j (old code) | Phase 1l (new code) |
|---|---|---|
| HC: solved / median time to solve | 10 / 615 s | 10 / **278 s** |
| GOMEA: solved / median time to solve (unsolved = 3,600 s) | 7 / 2,498 s | 7 / **2,010 s** |
| HC faster in | 10 of 10 pairs | 10 of 10 pairs |
| p (Holm) / verdict | 0.0039 / worse | 0.0020 / **worse** |

Phase 1j's p is Holm-corrected over its two problems; Phase 1l tested mux11 alone.

**CPU per evaluation (median):**

| Arm | Phase 1j | Phase 1l | Speed-up |
|---|---|---|---|
| HC | 85 ms | 37 ms | 2.3× |
| GOMEA | 472 ms | 385 ms | 1.2× |
| GOMEA / HC | 5.5× | 10.4× | – |

**Results are identical, as intended.** Every run that solved mux11 reached the solution at
exactly the same evaluation count as in Phase 1j: 10 HC runs and 7 GOMEA runs (e.g. HC seed
3001 at 8,394 evaluations). The 3 unsolved GOMEA runs ran more evaluations within the same
3,600 s.

## What this shows

1. **The speed-ups mainly helped what HC spends its time on**, scoring and candidate
   simplification. GOMEA's dominant cost, knob building for each new deme (about 68 % of its
   CPU), was untouched: the disc-probe memo got 0 hits and was not merged.
2. **GOMEA still needs fewer evaluations to solve** (median 4,354 for solved runs against
   7,953 for HC, unchanged since Phase 1j). Its loss at equal time is entirely its cost per
   evaluation.
3. **Tying HC at equal time needs knob-building cost to come down**, through fewer demes (Phase
   1m tests this) or smaller exemplars. GOMEA's exemplars grow about 2× faster than HC's; an
   analysis of why is under way.
