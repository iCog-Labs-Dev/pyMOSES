# Phase 1j findings: GOMEA vs hill climbing at equal wall-clock time

**Status:** complete. Stage 1 decided both problems, so no extension ran: 40 runs, all
`ok`, on 2026-10-13 from 11:50 to 13:58. Protocol: `PHASE1J-PREREGISTRATION.md`. Code
`7860f64`. Results: `phase1j-results/` and `../gomea-results/phase1j/` and
`../gomea-results/1/phase1j-mux11-*/` (relative to the repository root, outside it).

**Answer:** it depends on the problem. At equal time, tuned GOMEA stays **better** than
hill climbing (HC) on Con regression, but is **worse** on mux11.

## Setup

* **Termination:** both arms stop on time only, the way classic MOSES is run:
  `--max-time T -m 1000000000`.
* **Arms:**
  * `HC`: `-a hc`.
  * `LTB-C250-M`: GOMEA with LT-B, population 16, 250 evaluations per deme and the
    genotype memo, plus `--gomea-contin copy` on Con.
* **Problems:**
  * **Con:** 10 paired splits, T = 1,000 s.
  * **mux11:** 10 paired seeds, T = 3,600 s.
* **Load:** 10 workers on 22 cores, with nothing else from these experiments running. CPU
  time matched wall time within about 1 % in every run, so no run was starved of CPU.
* **Rule:** two looks at α = 0.025 each, with Holm correction over the problems. "Better"
  also needs a median at least 10 % lower.

## Results

| Problem | Metric | HC median | GOMEA median | GOMEA lower / tie / HC lower | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| Con | test NMSE | 54.0 | **43.8** | 8 / 0 / 2 | 0.0195 | 0.0195 | **better** |
| mux11 | seconds to solve (unsolved = 3,600) | **615** | 2,498 | 0 / 0 / 10 | 0.0020 | 0.0039 | **worse** |

**Con:**
* In 1,000 s, HC ran a median of 16,562 evaluations and GOMEA 6,074, about 2.7× fewer
  (HC's median wall time was 1,031 s against GOMEA's 1,004 s).
  GOMEA still found programs with lower test error in 8 of 10 splits.
* HC's split 10 is an outlier (test NMSE 242). The paired test uses ranks, and GOMEA is
  ahead without that split too: 7 of the other 9 (p 0.039 on those 9 alone).
* For comparison, at equal evaluations (10,000, Phase 1f part D) the medians were 37.1 for
  GOMEA and 54.9 for HC.

**mux11:**
* HC solved all 10 seeds, in 281–904 s. GOMEA solved 7, in 1,381–2,752 s. Seeds 3003, 3004
  and 3005 were unsolved at the limit.
* GOMEA again needed fewer evaluations to solve: a median of 4,354 (7 solved seeds) against
  7,953 for HC (10 seeds).
* Each GOMEA evaluation costs far more: a median of about 472 ms of CPU per evaluation,
  against about 85 ms for HC (about 5.5×).
* GOMEA also used more memory: a peak of 183–816 MiB against 43–45 MiB for HC.

## What this shows

1. **On Con, GOMEA's advantage survives its CPU cost.** It produced better programs per
   second as well as per evaluation, despite running fewer evaluations.
2. **On mux11 it does not.** GOMEA still needs fewer evaluations, as in Phase 1e, but about
   4× the time to a solution. At equal time, HC wins on every seed.
3. **Tentative practical reading:** on Con, tuned GOMEA was the better choice even with
   time as the limit; on mux11, HC was. Both rest on one problem, one time limit and 10 pairs;
   other regression datasets tied at equal evaluations (Bos, Yac in Phase 1f), so this does
   not yet generalise. Cutting GOMEA's cost per evaluation (about 5.5× HC's on mux11) is the
   obvious next step, and profiling it comes first.

## Limits

* One problem of each kind, 10 pairs each. Both verdicts were reached at stage 1.
* One time limit per problem. On mux11 a longer limit could raise GOMEA's solved count
  (7/10), but not the verdict: HC solved every seed within 904 s and was faster on all 10.
* The CPU cost reflects this Python implementation. A faster GOMEA would shift the
  equal-time comparison towards GOMEA.
