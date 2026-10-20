# Phase 1w findings: top-k linkage subsets per generation

**Status:** complete. Runs from 2026-10-19 19:06 to 2026-10-20 03:56:

* **mux11 part:** stopped early under a documented deviation, at 23:28.
* **Con:** stage 1 (30 runs), then the registered extension (splits 11–20, 20 runs).

Protocol: `PHASE1U-1AA-PREREGISTRATION.md`, including its 1w deviation section. Code `311d8eb`
(the deviation scripts `tools/gomea/phase1w_earlystop.py` and `phase1w_finish2.sh` are
tooling only). Results: `phase1w-results/`, `../gomea-results/phase1w/` and
`../gomea-results/1/phase1w-bool-*/`.

**Answer: no.** GOMEA restricted each generation to the K = 4 most strongly linked subsets
(`--gomea-fos lt-mi --gomea-fos-topk 4`, arm `TOPK-L`), so that capped demes complete several
generations and relearn their linkage. It was **worse** than plain GOMEA on both problems.

## Results

| Problem (metric) | Plain GOMEA | TOPK-L | TOPK-L lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|
| mux11 (CPU to solve; unsolved = +∞) | 1,232 s, 10/10 solved | 0/10 solved | 0 / 0 / 10 | 0.002 | **worse** |
| Con (test NMSE), stage 1 | 37.1 | 55.4 | 2 / 0 / 8 | 0.037 | extend |
| Con (test NMSE), stage 2, 20 splits | **35.2** | 55.9 | 2 / 0 / 18 | 0.0002 | **worse** |

**Secondary** (TOPK-L vs TOPK-R, random K = 4 subsets):

* **Con:** 55.4 against 70.9, 7 / 3, Holm p 0.39, **tie**.
* **mux11:** not decidable, since neither arm solved; 8 TOPK-R runs started before the stop.

**Cost:**

* **mux11:** TOPK runs used about 1.2 s of CPU per evaluation in their long-running demes,
  against about 0.25 s for GOMEA. That is roughly 5× more, though the summaries' per-run
  figures are distorted by the early stop.
* **Con:** about 1.35× (0.32 against 0.24 s per evaluation).

## The mux11 early stop (deviation)

* TOPK runs on mux11 would have run for up to 10 hours each.
* The registered metric gives each pair's sign as soon as a TOPK-L run has spent more CPU
  than its paired GOMEA run needed to solve. So the mux11 runner was stopped once that held
  for all 10 seeds (`earlystop.json`).
* At the stop, every TOPK-L run was unsolved, and each had spent more CPU than its pair's
  GOMEA run needed to solve: from 1.01× (seed 3003: 6,618 against 6,550 s) to about 10×
  (seed 3010). One (seed 3002) had already ended on its own after 1,016 evaluations, unsolved.
* Interrupted runs were recorded as `error`, i.e. unsolved (+∞), the outcome the rule
  assigns.
* GOMEA solved every seed in 540–6,550 s.

## What this shows

1. **Relearning the linkage every generation, from N = 16, does not pay.**
   * Each generation relearns a mutual-information model over 1,000+ knobs. That is
     expensive on mux11, and the 4 subsets kept are a tiny part of a deme's search space.
   * Demes either stagnate early (they stop after a few generations without improvement) or
     spend far more CPU per evaluation.
2. **On regression, the top-k restriction loses accuracy** (test NMSE 55.9 against 35.2).
   Plain GOM, which mixes every subset at least once, searches the neighbourhood much better
   at the same evaluation budget.
3. **Choosing subsets by linkage beat choosing them at random on Con by median** (55.4
   against 70.9), but not significantly. Both are far worse than using every subset.
4. **This closes backlog item 2.4.** Making demes complete more generations this way is not a
   route to better linkage use inside MOSES.

## Correction (2026-10-20 09:30): orphaned runs

* **What happened:** stopping the mux11 runner by terminating its child runs left some runs
  orphaned. Either the runner started queued runs in the seconds before it was terminated,
  or a child survived the signal. They kept running until killed at 09:25.
* **Two of them finished and wrote summaries into the 1w result directory after this
  write-up:**
  * `TOPK-L s3003`: solved at 12,750 evaluations and **24,939 s** of CPU. Its pair's GOMEA
    needed 6,550 s.
  * `TOPK-R s3006`: solved at 12,748 evaluations and 29,361 s.
* **Effect on the verdicts:** none. Every TOPK-L pair is still slower than its GOMEA pair,
  10 of 10, so the mux11 verdict stays **worse**.
* **Correction to the text above:** TOPK-L solved 1 of 10 seeds (after the cutoff), not 0.
* **Checksums:** `phase1w-results/phase1w-bool__summaries.sha256` records the summaries as
  analysed. The live directory now differs in these two files.

## Limits

* One K (4), one population size (16), LT-MI only.
* A larger K or N would cost more per generation and would not obviously help, given 2–3
  above.
