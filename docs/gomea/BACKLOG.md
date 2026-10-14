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

9. **Seed the metapopulation with GP-GOMEA elites** (user's idea, 2026-10-14).
   * **Idea:** run stand-alone GP-GOMEA first, with a fixed small tree height (2 or 3, i.e. at
     most 7 or 15 nodes), and put its elite solutions into MOSES's starting metapopulation
     instead of starting from the empty program alone.
     * MOSES then builds knobs around those elites, and its embedded GOMEA still learns knob
       linkage inside each deme.
     * The metapopulation level gains node-level improvements at a fixed, small size: a
       "smart restart".
     * The empty exemplar (`and()` / `+()`) is added as a control, so MOSES can still grow its
       own kind of solution if the seeds are misleading.
   * **Why it is promising:** stand-alone GP-GOMEA beat C++ MOSES on all 10 regression
     datasets (Phase 1f part B) with small fixed-shape trees. MOSES's strength is refining a
     good exemplar. On the Boolean tasks, GP-GOMEA solved maj5/mux6 given enough evaluations
     (Phase 1b).
   * **Feasible with little code.**
     * `-e` already accepts several exemplars (`problem_params.py:598`, a multi-value option).
     * We already run the C++ GP-GOMEA binary (Phase 1f tooling) and parse its output.
     * Needed: a translator from GP-GOMEA's output to combo syntax.
   * **Design points to settle first:**
     1. **The control may never be used.** MOSES picks the exemplar to expand by score with a
        complexity penalty, so the empty program would almost never be chosen over good
        seeds. Make the control a separate arm (unseeded MOSES), or reserve a share of
        expansions for it. Mixing it in alone does not guarantee it is explored.
     2. **Fair budget:** GP-GOMEA's evaluations must count against the same total budget,
        e.g. 20 % for seeding and 80 % for MOSES. Otherwise seeding gets free work.
     3. **Translation:**
        * The function set must be one both systems share: +, −, ×, protected /, sin, exp,
          log; for Boolean, AND/OR/NOT.
        * GP-GOMEA's analytic quotient has no MOSES equivalent, so it must be excluded when
          seeding.
        * GP-GOMEA applies linear scaling (a + b·f(x)). The scaled form must be written into
          the seed, because MOSES's fitness has no linear scaling.
        * MOSES reduces each seed on entry (reduct), so seeds may change form.
     4. **How many seeds and which:** the best only, or the top-k distinct elites. More seeds
        give more diversity but spend metapopulation slots.
     5. **Where the final best comes from:** log whether each final program descends from a
        seed or from the empty exemplar, to see whether MOSES refines the seeds or abandons
        them.
   * **Experiment sketch:**
     * Arms: MOSES-HC and MOSES-GOMEA (cap 250), each unseeded and seeded (four arms), at an
       equal total budget that includes the seeding cost.
     * Problems: Con, EnH, EnC (+ mux11 or maj7).
     * Seeds: 10 + 10.
     * Metrics: test NMSE (or solved / evaluations to solve), final size, and seed lineage.
   * **Later extension:** re-run seeding when MOSES stagnates, using the current
     metapopulation's best programs as GP-GOMEA's starting population, for repeated smart
     restarts.

10. **From the two literature reviews** (2026-10-14; `reports/Building blocks for symbolic
    regression.md` and `reports/Redundancy based simplification in SR.md`, kept outside git).
    * **Check first:**
      * Is linear scaling part of pyMoses's regression fitness, or only applied when the error
        is reported? In the literature, linear scaling gave larger gains than building-block
        libraries.
      * How many evaluated candidates are exact duplicates after reduct? Our duplicate
        counters already record part of this.
      * Do final models contain redundant constants or near-constant subtrees?
    * **Most promising change:** every few generations, replace near-constant subtrees in
      exemplars with constants before building knobs.
      * Set the threshold relative to the noise level, and accept a change only if a held-out
        split shows no loss.
      * Expect any gain in accuracy rather than size: Aldeia et al. 2024 reported −18–20 %
        test MSE on Airfoil, Concrete, Energy, Boston and Yacht.
      * Use 10 + 10 seeds; Wine is the guard against a loss.
    * **IEBL-style building blocks for regression:** no published equivalent exists. If tried,
      follow GP-GOMEA's feature construction: at most 3 small terms, each refitted jointly with
      the others, no nesting, at equal size and evaluations against plain GOMEA.
    * **Feature selection:** low priority. Our regression sets have 5–13 inputs (Dow and Tower
      are the exceptions). MOSES's default `simple` scorer ignores redundancy and would likely
      hurt on parity and mux.

## 3. Ideas outside Phase 1's scope

* **IEBL-style building blocks** (Input-space Entropy-based Building-block Learning;
  Virgolin, Alderliesten, Witteveen and Bosman, GECCO 2017, the first GP-GOMEA paper).
  * **Method:** score small Boolean subtrees by the entropy of the *unused* inputs over the
    fitness cases they still get wrong (lower is better), then add the best ones as new
    terminals and new functions.
  * **Results:** it helped most on modular problems (Even Parity, 14-bit solved in under an
    hour).
  * **Availability:** it is not in the C++ GP-GOMEA we ported (`750220c`); the 2017 code is
    on Bosman's website.
  * **For MOSES:** score subtrees of good metapopulation programs the same way, and offer the
    best as new inputs or operators to later demes.
* **Phase 2, real-valued constants:** planned; needs explicit approval.
