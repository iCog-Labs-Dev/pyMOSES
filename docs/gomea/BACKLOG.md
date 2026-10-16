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
2. **Cheaper disc probing:** ~~memoise probe results across demes~~. **Tried 2026-10-15: no
   effect.** An exact memo got 0 hits in 12,716 probes on mux11, because each probe
   simplifies the *whole* exemplar with one knob turned, and that tree differs for every knob
   and deme. The patch is kept unmerged in `../patches/probe.patch`. What remains:
   * incremental simplification (risky for C++ fidelity);
   * fewer demes, i.e. the budget-derived cap (item 2.1), which cuts knob building in
     proportion.
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
      * Linear scaling: **checked 2026-10-15, not in pyMoses's fitness.** The search scores raw
        predictions, and linear scaling is applied only when the error is reported
        (`tools/gomea/sr_eval.py`). In the literature it gave larger gains than
        building-block libraries (e.g. Dow NMSE 65 → 23 for RDO), so adding it to the
        regression fitness, for HC and GOMEA alike, is a strong candidate experiment.
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

11. **Phase 1i follow-ups** (the uniform-distance start helped LT-B at about 6–8× the
    evaluations).
    * **Equal-cost test:** compare the uniform and default starts at equal evaluations, e.g.
      cap the uniform start at the default's ~20k, or give the default start ~150k (larger N
      or restarts). This shows whether the uniform start is a better use of the budget.
    * **End to end:** does the uniform start help inside full MOSES runs with capped demes?
      Its ~37k evaluations per generation may be prohibitive there. Try cheaper mixtures,
      e.g. `2:0.25,4:0.25,8:0.25,max:0.25` or uniform over distances 1..8.
    * **Combine with a shared linkage model (item 3):** diverse samples pooled across demes.

12. **Population size derived from the deme's budget** (user's design, 2026-10-15).
    * **Design:**
      * Each deme gets a cap C. The sampler draws N programs once, with N a fraction of C
        (e.g. C = 250 → N = 50) instead of the fixed 16.
      * That population is the deme's population for its whole life; GOM only modifies its
        members.
      * Each generation's linkage model is learned from the current N programs, until the
        deme's evaluations run out.
    * **Constraint (measured):** a generation costs about 12–13 evaluations per individual.
      So the starting population plus g full generations needs about
      N + 13·N·g ≤ C, i.e. N ≤ C / (1 + 13g). For example:
      * C = 250 allows N ≈ 9 for 2 generations; with N = 50 the deme still ends inside
        generation 1, with one model learned from 50 unselected programs.
      * C = 1,000 allows N ≈ 37 for 2 generations.
      * C = 2,500 allows N ≈ 93 for 2 generations.
    * **Pair with the budget-derived cap (item 2.1):** at a fixed total, fewer and longer
      demes, with N set from C, give the linkage model both more programs and later,
      fitness-shaped populations.
    * **Code needed (small):**
      * `--gomea-pop-fraction F`: N = max(2, floor(F · cap)), with the cap taken as
        min(per-deme cap, remaining budget).
      * `--gomea-deme-share D`: cap = total budget / D.
    * **Experiment sketch:**
      * Regression: Con, EnH and EnC at a fixed total of 10k evaluations.
      * mux11 at 40k.
      * Arms: HC; current GOMEA (C = 250, N = 16, LT-B); budget-derived GOMEA (e.g. C = 1,000,
        N ≈ 40) with LT-B and with RT.
      * Seeds: 10 + 10.
      * Questions: does it beat the current setting, does LT-B beat RT once N is larger and
        generations complete, and does it close the gap to HC?

13. **Curb GOMEA's exemplar growth** (analysis 2026-10-15, `exemplar-growth-2026-10-15/REPORT.md`).
    * **Cause on mux11:**
      * GOMEA runs about 4× more demes per evaluation than HC.
      * Each deme adds about 1.85× more literals per row gained: 5.5 against 3.0 per 100
        rows. A capped GOM deme behaves like first-improvement search, taking the first
        not-worse trial; HC takes the best neighbour.
      * The default complexity penalty (`-z 3.5`, 0.29 per literal) barely binds on mux11,
        which has 2,048 rows.
    * **Not the cause:** merging, exemplar selection and large group copies.
    * **No gap on maj5** (32 rows, where the penalty binds).
    * **Pilot with `-z 0.125`** (8 rows per literal), 3 seeds:
      * exemplars about half the size and knob-building CPU 2–17× lower;
      * 2 of 3 seeds solved sooner;
      * 1 seed failed, because `-z` also changes exemplar selection.
    * **Proposed experiment:** a stronger complexity penalty only inside GOMEA's acceptance
      (`_OracleEvaluator.worse`), with merge and selection unchanged. Optionally add a short
      end-of-deme pass that removes literals.
    * **Side finding to check:** knob counts sometimes collapse for large exemplars (a
      138-node exemplar got 123 knobs; a 58-node one got 248). Representations may
      occasionally be built incomplete.

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

## 4. Bugs and anomalies to investigate later

1. **Knob counts collapse for some exemplars: classic MOSES behaviour, not a port bug**
   (investigated 2026-10-16; exemplars and scripts in `knob-collapse-2026-10-16/`).
   * **Same in C++:** pyMoses and C++ MOSES build identical representations on the
     collapsing exemplars, e.g. 135, 250 and 114 fields in both.
   * **Cause:** `disc_probe` (`build_knobs.py:526`, C++ `build_knobs.cc:311`) disallows a
     setting when reducing the whole decorated exemplar lowers its complexity. That assumes
     the exemplar is already a fixpoint of the reducer, but effort-2 Boolean reduction is not
     confluent.
     * `build_logical` wraps literals in single-child flip junctors as it walks the tree.
     * Once one wrapper lets the reducer find a context simplification it missed on the plain
       tree, every later probe sees a reducible baseline, and almost every later knob is
       thrown away. Example: 117 knobs built before the trigger, 20 of 1,364 probes after it.
     * Signature: instance 0 (the exemplar itself) reduces to a lower complexity.
   * **Frequency on mux11:** in 2–4 of 11–20 GOMEA demes per run; none in the HC runs
     examined; fewer with a stronger complexity penalty, since exemplars stay smaller.
   * **Effect:**
     * a collapsed deme searches a much smaller neighbourhood but still pays the full probing
       cost;
     * per-deme knob statistics and per-deme budgets work on representations of very
       different sizes.
   * **Mitigations (would diverge from C++; untested):**
     * compare each probed setting against the *reduced* baseline;
     * or reduce the exemplar with the knob-building reducer before decorating it.
     * Either could be an opt-in option, tested for knob counts, CPU and accuracy.
   * **Related:** the earlier `-B 3` result (1 knob on maj5-x3) has the same mechanism, since
     the exemplars were not fixpoints at effort 3.
2. **Rejected GOM trials are stored in the deme and merged.**
   * **What happens:** every scored trial is appended to the deme, whether GOM accepted it
     or not. MOSES's merge then ranks them with its own complexity penalty.
   * **Impact today:** this matches HC, which also stores every evaluated neighbour, and is
     not a bug now.
   * **When it becomes one:** any change that penalises complexity *inside* GOMEA's
     acceptance (backlog item 13). The merge would then bring back bloated rejected trials
     with higher raw scores, unless it ranks them under the same penalty or keeps only
     accepted and elite individuals.
3. **(Fixed 2026-10-15, `a42d5c8`):** `twiddle_contin_bit` crashed on contin-only demes; C++
   has the same out-of-range read as undefined behaviour. Kept here for the record; disclosed
   in `APPROVALS.md`.
