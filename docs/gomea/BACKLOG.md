# GOMEA backlog: ideas to come back to

Collected 2026-10-13 while Phases 1i and 1k run. None of these is implemented or
registered yet. Each experiment follows the usual process: pre-register, 10 seeds first and
10 more only if borderline, fact-check the findings.

## 1. Cost per evaluation (from profiling, 2026-10-13)

**Measurement:** mux11, seed 3001, 3,000 evaluations. GOMEA used 354 ms of CPU per evaluation,
hill climbing (HC) 69 ms (5.1×; Phase 1j measured 5.5×). Profiles and logs are in
`docs/gomea/profile-2026-10-13/` (per-deme timings and runner scripts; the cProfile files were not kept).

| GOMEA CPU, ms per evaluation | GOMEA | HC |
|---|---|---|
| Knob building per deme (mostly `build_knobs.disc_probe`) | 240 (68 %) | 15 |
| Scoring: truth table (~51 %) + candidate reduction (~49 %) | 89 | 39 |
| Merging into the metapopulation | 15 | 15 |
| Linkage learning | 8 | – |
| GOM mixing and bookkeeping | < 1 | – |

**Causes:**
* GOMEA builds about 3× as many demes, because of the 250-evaluation cap.
* Its exemplars grow about 2× faster: 70 nodes and 1,488 knobs by deme 11, against 30 nodes
  and 742 knobs for HC.
* Knob-building cost rises faster than exemplar size.

**Fixes, largest expected saving first:**
1. **Fewer, longer demes:** e.g. a cap of about 750, or continuing a deme rather than
   rebuilding it. About −45 % CPU per evaluation, but it changes the search, so it needs a
   test (see 2.1).
2. **Cheaper disc probing:** memoise probe results by the subtree's canonical string across
   demes, since successive exemplars share most subtrees. Up to about −68 %.
3. **Exemplar growth:** find out why GOMEA's best programs grow about 2× faster than HC's at
   equal evaluations. Size drives knob building and reduction.
4. **Bit-parallel truth-table scoring** with Python int bitmasks: about −40 ms per evaluation
   for GOMEA and −20 ms for HC.
5. **Cache reduced candidates** by the unreduced-tree key that `phenotype_key` already
   computes: up to about −40 ms per evaluation.
6. **Linkage:** compute the marginals in one pass, and reuse `calibrate`'s entropy matrix for
   the first `learn`: about −5 ms per evaluation.
7. **`to_instance`:** it runs twice per evaluation (scoring and the deme append), with
   per-gene validation. About −1 ms per evaluation.

GOM bookkeeping and telemetry are not worth optimising.

## 2. Experiments to run

1. **Derive the per-deme cap from the total budget.** Fix the total (e.g. 10,000 or 40,000)
   and set the cap to total / D, with D = 10, 20, 40 and 80 exemplar expansions. This tests
   fewer, longer demes with several full GOMEA generations against many short demes with one
   partial generation each. It has never been tested on regression, where GOMEA wins.
   Combine it with fix 1.1.
2. **Regression with a cap of about 1,000** (4–5 full generations per deme) instead of 250,
   on Con, EnH and EnC.
3. **A shared linkage model across demes:** learn the knob groups from a pool of good
   programs that persists between demes, so the model sees thousands of programs, not 16.
   Phase 1h showed LT-B needs N ≈ 1,024 to beat a random tree. Knobs differ between demes,
   so this needs a mapping, e.g. by knob position or by the subtree a knob inserts.
4. **Cheaper GOM generations:** use only the k strongest linkage groups per generation (e.g.
   8–16), or a random k, instead of all of them (about 2 per knob). With N = 16 and k = 8 a
   generation costs about 128 evaluations.
5. **Population size tied to the deme:** scale N with the knob count or the per-deme budget,
   or port GP-GOMEA's interleaved multistart scheme (doubling populations, no fixed N).
6. **Stronger knob-building reduction:** `-B 3` gave implausibly few knobs (1 on maj5-x3),
   because the fixture exemplars were simplified only at effort 2. Simplify the exemplars at
   effort 3 first, then test `-B 3` as a third knob setting next to Phase 1k's.
7. **More equal-time tests (Phase 1j follow-up):** further regression datasets (EnH, EnC) and
   Boolean problems (maj7, pa5), or a longer limit, before generalising "GOMEA wins on
   regression at equal time".
8. **Improvement rate as a standing metric:** the share of GOM trials that improve. LT-B's is
   higher than RT's even when final errors tie (Phase 1h step 1). Report it routinely,
   because it is more sensitive than final error.

## 3. Ideas outside Phase 1's scope

* **IEBL-style building blocks** (input-space entropy-based building-block learning,
  arXiv 2402.12510): extract frequent, informative subtrees from good programs and add them
  as new terminals or functions for later demes.
* **Phase 2, real-valued constants:** planned; needs explicit approval.
