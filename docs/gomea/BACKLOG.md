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
   * **Closed (server round 1, 1ae, 2026-10-27):** on Con, caps 500 and 1,000 were worse than
     250 (43.4 and 44.9 against 35.2, 15/20 each). Keep 250 for regression.
3. **A shared linkage model across demes:** learn the knob groups from a pool of good
   programs that persists between demes, so the model sees thousands of programs, not 16.
   Phase 1h showed LT-B needs N ≈ 1,024 to beat a random tree. Knobs differ between demes,
   so this needs a mapping, e.g. by knob position or by the subtree a knob inserts.
   * **Lower priority (server round 2, 2026-10-30):** learned linkage gave no benefit at 3–5
     generations per deme or at N = 64 (1al). This idea is the only untested route to a
     much larger linkage sample.
4. **Cheaper GOM generations:** use only the k strongest linkage groups per generation (e.g.
   8–16), or a random k, instead of all of them (about 2 per knob). With N = 16 and k = 8 a
   generation costs about 128 evaluations.
   * **Closed:** top-k subsets were worse (1w), and a FOS over a cap of 16 knobs was worse on
     mux11, Con and EnH (server round 2, 1ak, 2026-10-29).
5. **Population size tied to the deme:** scale N with the knob count or the per-deme budget,
   or port GP-GOMEA's interleaved multistart scheme (doubling populations, no fixed N).
6. **Stronger knob-building reduction:** `-B 3` gave implausibly few knobs (1 on maj5-x3),
   because the fixture exemplars were simplified only at effort 2. Simplify the exemplars at
   effort 3 first, then test `-B 3` as a third knob setting next to Phase 1k's.
7. **More equal-time tests (Phase 1j follow-up):** further regression datasets (EnH, EnC) and
   Boolean problems (maj7, pa5), or a longer limit, before generalising "GOMEA wins on
   regression at equal time".
   * **Answered for regression (server round 1, 1ah, 2026-10-26):** GOMEA beats capped HC at
     equal time on Air, EnC and EnH, and ties on Bos and Yac. maj7 and pa5 stay untested.
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
   * **Tooling (2026-10-16):** `tools/gomea/build_gpgomea_seed.sh` builds `gpgomea-seed`
     (GP-GOMEA that dumps every IMS elitist and the active populations, with full-precision
     constants and linear-scaling terms); `tools/gomea/gpgomea_seed.py` translates the dump
     to combo, checks every translation against pyMoses's own scorer and keeps the k best
     distinct seeds; an arm's optional `seed` block (`{"binary", "share" | "budget",
     "height", "k", ...}`) in `benchmarks.gomea.runner` and `tools/gomea/moses_sr_runner.py`
     runs it out of the arm's `-m` and passes `-e SEED ... -e and|+`;
     `tools/gomea/seed_lineage.py` records which exemplar each expansion and the final
     programs descend from. Pilot manifests: `configs/gomea/seed/`.
   * **Dropped (user, 2026-10-20; DECISIONS.md D4):** the later extension below was tested in
     Phase 1ab (a tie) and is not pursued further in any form.
   * **Later extension:** re-run seeding when MOSES stagnates, using the current
     metapopulation's best programs as GP-GOMEA's starting population, for repeated smart
     restarts.
     * **Built (2026-10-19), opt-in, untested in a phase:** `tools/gomea/reseed.py`
       (seed-block `reseed_after`/`reseed_share`/`max_reseeds`). Fallback design: each
       re-seed is a fresh GP-GOMEA run with a new seed, not seeded from the metapopulation
       (that needs GP-GOMEA C++ changes). Also `--reserve-empty-every K` (design point 1's
       "reserve a share of expansions"). See `USAGE.md`.

10. **From the two literature reviews** (2026-10-14; `reports/Building blocks for symbolic
    regression.md` and `reports/Redundancy based simplification in SR.md`, kept outside git).
    * **Check first:**
      * Linear scaling: **checked 2026-10-15, not in pyMoses's fitness.** The search scores raw
        predictions, and linear scaling is applied only when the error is reported
        (`tools/gomea/sr_eval.py`). In the literature it gave larger gains than
        building-block libraries (e.g. Dow NMSE 65 → 23 for RDO), so adding it to the
        regression fitness, for HC and GOMEA alike, is a strong candidate experiment.
        **Tested (server round 1, 1ag, 2026-10-27):** `--linear-scaling 1` was better for HC
        (38.6 against 60.4, 10/10) and GOMEA (25.9 against 37.1, 8/10) on Con. Recommended
        as the default with GOMEA for regression. Open: equal-time test (item 5.1).
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

14. **Seeded GOMEA on mux11: does seeding close the Boolean time gap?** (user, 2026-10-19)
    * **Why:**
      * In Phase 1r, seeding cut GOMEA's CPU per run in every pair, about 4× on Con (20/20)
        and 6.3× on EnH (10/10).
      * It also brought GOMEA's CPU from 4.6–5.2× capped HC's down to 1.65–1.9×. Exemplars
        stay small, and knob building scales with exemplar size.
      * mux11 is the problem where GOMEA loses at equal time because of CPU: Phase 1p,
        2,568 s against 1,311 s to solve.
    * **Caution:** one earlier pilot (seeded capped HC, mux11, seed 3001) needed about twice
      the evaluations to solve, so Boolean seeds may mislead.
    * **Sketch:**
      * Arms: capped HC and GOMEA (cap 250), each seeded and unseeded, at equal time
        (`--max-time`, as in Phase 1p).
      * Seeding: `gpgomea_seed.py --bool`, AND/OR/NOT, its time counted in the budget.
      * Metric: wall time to solve. Report CPU per evaluation and exemplar size.
      * Seeds: 10 + 10.
    * **Closed (server round 1, 1aj, 2026-10-27): no.** At 3,600 s, seeded GOMEA tied capped HC
      (351 against 293 s CPU to solve, 7/20) and lost to plain HC (77 s, 0/10).

15. **Adopt `--knob-probe-baseline reduced` as the default** (user, 2026-10-19).
    * **Evidence:** Phase 1q.
    * **Plan:** from the first phase after 1s and 1t, add it to every arm, GOMEA and HC
      alike, and say so in each pre-registration.
    * **Adoption:** make it the code default (a disclosed DIVERGENCE) once it has run in two
      phases without a loss. Track the status in `FINDINGS-LOG.md` → "Candidate defaults".

16. **Confirm PEN's CPU-to-solve gain on mux11** (after Phase 1s, 2026-10-19).
    * **Phase 1s:** a registered tie. Over 20 pooled seeds PEN's median was 37 % lower,
      13 / 7 pairs, p 0.097.
    * **If wanted:** 20 fresh seeds on the merged, faster code, same metric, single primary
      contrast.
    * **Cost:** about 40 runs, about 1 h with the knob speed-ups.

17. **Confirm the uniform start on a second regression dataset** (after Phase 1v).
    * **Phase 1v:** Con test NMSE 37.1 → 29.2, 10/10 splits.
    * **Check:** EnH (and EnC), 10 splits, GOMEA vs GOMEA-UNI, test NMSE.
    * **Also try:** UNI with seeding (Phase 1r), and UNI with the acceptance penalty to curb
      the larger programs.

18. **Caps 500 and 1,000** (after Phase 1x; cap 1,000 added by the user, 2026-10-20).
    * **Phase 1x:**
      * mux11 at cap 500: CPU to solve 1,000 against 1,464 s, 6/4 pairs, Holm p 0.252, just
        outside the extension band.
      * The trend: longer demes mean fewer knob rebuilds and cheaper evaluations (0.29 → 0.12 s
        per evaluation from 250 to 500), but more evaluations to solve.
      * On Con, 500 was worse by median (41.8 against 37.1).
    * **Follow-up:** caps 250, 500 and 1,000 on mux11 and Con, 10 + 10 seeds.
      * Primary: 500 vs 250 and 1,000 vs 250.
      * Report generations per deme too: does 1,000 start to give second linkage models (see
        item 19)?
    * **Bound:** Phase 1m's cap 4,000 (with N up to 200) was much worse on mux11, so the useful
      range is likely somewhere between 500 and a few thousand.
    * **Cost:** about 2–3 h.
    * **Closed (server round 1, 1ae, 2026-10-27):** mux11 cap 1,000 better (278 against 627 s,
      19/20), cap 500 tie; Con both worse. Cap 1,000 gave a second generation to 12.6 % of
      mux11 demes and 44.7 % of Con demes. Keep 250 as the default.

19. **Fewer, longer demes for promising exemplars, so linkage is relearned** (user,
    2026-10-20).
    * **Why:** run telemetry shows that with cap 250 and N = 16, 97–100 % of GOMEA demes
      learn their linkage model once, from the unselected initial population (1u, 1v, 1x).
      * Repeated relearning happened only in top-k (1w, worse) and in uncapped frozen
        single-deme runs (1h). Only the latter showed learned linkage beating a random tree
        (N = 1,024).
      * So inside MOSES, GOMEA currently "learns once and mixes in groups".
    * **Idea:** give most demes the usual short budget, but give the demes of the most
      promising exemplars (e.g. the top 1–3 by penalized score, or exemplars that improved
      the run's best) a much larger budget, so that GOM completes several generations and
      relearns linkage from a selected population.
    * **Prior evidence to respect:**
      * 1m: making every deme long (cap 4,000, N up to 200) was much worse on mux11
        (5.6× the evaluations) and a tie on regression. With N = 200, demes still stopped
        inside generation 1.
      * 1x: cap 500 already gave 3 % of demes a second generation.
      * So the idea is long demes for a few, with small N (16–32), not long demes for all.
    * **Sketch:**
      * An opt-in rule for the per-deme cap, e.g. `--gomea-long-deme-cap C_long` plus a
        selection rule for which demes get it.
      * Arms: current GOMEA; LONG-TOP (top exemplars get C_long = 2,000–4,000); a
        random-selection control with the same number of long demes; LONG-TOP with RT
        linkage (does learned linkage now beat random?).
      * Problems: mux11 (CPU to solve) and Con (test NMSE), 10 + 10 seeds, the usual rule.
      * Report generations per deme and the share of long demes that complete 2 or more.
    * **Status:** to build (subagent), then pre-register.
    * **Closed (server round 1, 1af, 2026-10-27): all ties.** LONG-TOP tied GOMEA on mux11 and
      Con, and tied LONG-RAND and LONG-TOP-RT. The options stay opt-in.

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
   * **Mitigation (opt-in, diverges from C++): `--knob-probe-baseline reduced`** (USAGE.md,
     "Knob probing baseline"). A setting is disallowed only if it loses more complexity under
     reduction than the baseline (knob at 0) does; with an irreducible baseline this is
     classic's test.
     * Knob counts, classic → reduced, default perm ratio, RNG seed 1: c3001_7 135 → 1,048;
       c3002_11 250 → 1,234; c3003_11 114 → 2,704. Controls n3001_6 (832) and n3002_10
       (1,193) are unchanged, knob for knob, disallowed settings included. On the collapsing
       exemplars every knob classic builds is still built with the same disallowed settings.
     * Reducing the exemplar before decorating it does not help: all five exemplars are
       already fixpoints of the effort-2 reducer (c3001_7 stays at 135 knobs). The
       reducibility appears only after decoration.
     * Comparing reduced complexities directly (`c(R(T_setting)) < c(R(T_0))`) was also
       tried and rejected: it lets genuinely redundant settings through (n3001_6 832 → 1,160).
     * Pilot, mux11 `-k 3 -m 3000`, seeds 3001–3003, capped GOMEA (lt-b, N 16, cap 250, memo)
       and `-a hc-capped` (cap 250), classic → reduced:
       * GOMEA: collapsed demes (knobs per node < 10) 1/3/2 → 0/0/0; best raw −162/−132/−164 →
         −164/−128/−183; CPU 738/919/2,357 s → 1,318/1,207/4,507 s, almost all of it knob
         building on the now-full representations. CPU was measured on a loaded machine
         (load ≈ 16–20 on 22 cores) and repeated runs of identical work differed by up to 2×.
       * HC: no deme collapsed with either option, and outputs and per-deme knob counts are
         identical.
       * No accuracy gain shows in three seeds; the option is a correctness fix for the
         representation, not a tuning win.
   * **Related:** the earlier `-B 3` result (1 knob on maj5-x3) has the same mechanism, since
     the exemplars were not fixpoints at effort 3.
2. **`-B 3` crash: a latent effort-3 reduction bug, also in C++** (found 2026-10-20 in Phase
   1aa; diagnosed the same day).
   * **Where:** `reduce_remove_subtree_equal_tt` (`logical_rules.py:1667-1685`), run only at
     effort 3 (`logical_reduction.py:192-195`).
   * **What happens:**
     1. It cuts the children of the root junctor one at a time and keeps a cut when the
        truth table is unchanged.
     2. That can leave a childless root `and()`/`or()`; `remove_dangling_junctors` never
        removes the root.
     3. The interpreter returns `id::logical_and`/`or` for an empty junctor (as C++
        `interpreter.cc:66-76` does).
     4. `builtin_to_bool` asserts (C++ `OC_ASSERT`, `vertex.h:710`).
   * **Reproduction:** `or(and($1 $2) $1)` crashes the rule alone. A natural case under full
     `logical_reduce(3, ·)` was found by fuzzing.
   * **Not caused by `--knob-probe-baseline reduced`.** The classic probe runs the same code
     at `-B 3`. The default `-B 2` never reaches the rule, so no earlier result is affected.
   * **Phase 1aa:** exactly one real crash (HC-B3, seed 3004); the other errors were runs
     terminated when the phase was stopped.
   * **Fix:** built in commit "fix: keep the last child of the root junctor in
     reduce_remove_subtree_equal_tt". A cut that would leave an empty root junctor counts as
     "truth table changed" and the child is restored. Marked DIVERGENCE (C++ aborts here).
     Only inputs that crashed change; identity harness byte-identical. Then rerun Phase 1aa.

3. **Rejected GOM trials are stored in the deme and merged.**
   * **What happens:** every scored trial is appended to the deme, whether GOM accepted it
     or not. MOSES's merge then ranks them with its own complexity penalty.
   * **Impact today:** this matches HC, which also stores every evaluated neighbour, and is
     not a bug now.
   * **When it becomes one:** any change that penalises complexity *inside* GOMEA's
     acceptance (backlog item 13). The merge would then bring back bloated rejected trials
     with higher raw scores, unless it ranks them under the same penalty or keeps only
     accepted and elite individuals.
4. **(Fixed 2026-10-15, `a42d5c8`):** `twiddle_contin_bit` crashed on contin-only demes; C++
   has the same out-of-range read as undefined behaviour. Kept here for the record; disclosed
   in `APPROVALS.md`.

## 5. Open items after the server rounds (2026-10-30)

From `SERVER-ROUND-FINDINGS.md` and `SERVER-ROUND2-FINDINGS.md`.

1. **GOMEA-LS against HC-LS at equal time** (regression).
   * **Why:** 1ag showed linear scaling helps both, at equal evaluations. GOMEA-LS had the
     lowest error on Con (25.9 against 38.6 for HC-LS) but used about 8× the CPU (916 against
     114 s). The recommendation (GOMEA with linear scaling) rests on this comparison.
   * **Sketch:** GOMEA-LS vs HC-LS (capped and uncapped), 1,000 s, the 1ah datasets plus Con,
     10 + 10 splits. Report size, gap and evaluations (D1).
2. **Yac: why GOMEA loses.**
   * **Why:** in 1ah, GOMEA's median test NMSE on Yac was 8.0 against 4.5 (HC-C250) and 4.4
     (HC), lower than HC-C250 in only 8/20 splits. Its gap was 1.3 against 0.4. Yac is the
     only 1ah dataset that leans to HC.
   * **First step:** a descriptive look at the Yac runs (exemplar growth, overfitting, number
     of demes), before any new phase.
3. **Size control for regression.**
   * **Why:** GOMEA's programs are 3.0–3.9× capped HC's and 1.8–2.2× HC's (1ah), and 5–10×
     GP-GOMEA's with linear scaling (1ai). The user accepts a moderate accuracy loss for
     smaller programs (`DECISIONS.md`).
   * **Candidates:** the acceptance penalty with linear scaling (UNI-PEN was a D1
     candidate in 1ad); linear scaling alone already made programs smaller (1ag).
4. **Knob-cap linkage-score cost (only if the knob-cap code is kept).**
   * **What:** in `explore`, `LinkageModel.knob_strengths` (`pymoses/gomea/fos.py`) builds the
     entropy matrix over all linked search positions in every generation, not only the K
     selected knobs. On mux11 (about 1,600 positions) linkage was 50 % of the engine time and
     about 19 % of KC's total CPU (1ak).
   * **Status:** the knob cap is not recommended (1ak worse on all problems), and it is off by
     default. Fix the cost only if a later experiment uses the cap.

## 6. Open items after server round 3 (2026-11-03)

From `SERVER-ROUND3-FINDINGS.md`. Section 5 items 1 to 3 have their answers there, except a
penalty that is weaker than R 0.005.

1. **LM against mutation at equal CPU.** 1an compared them at equal evaluations only (ties).
2. **Why LM lowers CPU and size.** 1an shows both, but no run measures the cause.
3. **Print `a` and `b` with the final program.** With `--linear-scaling 1` the printed program is
   `f`, not `a + b·f`. A user needs the training data to get the predictions right. This changes
   the output only.
4. **Early rejection of bad candidates.** Score a candidate on a sample of rows, with `a` and `b`
   fitted on that sample, and stop if it is clearly bad. Program evaluation is more than 99 % of
   the scoring time (timing on Con, EnH, Air, 2026-11-03). It can reject good programs, and it is
   not tested. Check the literature first.
5. **A regression profile of CPU.** Section 1 is from mux11. Regression has no such profile.
6. **Parked by the user (2026-11-03): the constant-tuning round** ("Maybe some other time when we
   are out of any other ideas"). It compared none, LM, mutation and polish at equal CPU on all 10
   datasets, splits 11–30. Also not selected: real-valued genes in place of the step knobs, because
   100 % of the genes in the Con, EnH and Air demes of 1ao are contin knobs (weight 0 removes a term).

## 7. Open items after server rounds 4 and 5 (2026-11-06)

The user's decisions on rounds 4 and 5 are in `APPROVALS.md`. No evaluation cap is now the base for
all problems. LM-VAL and PEN125 (R 0.125) are the preferred settings, and neither is a default.
Rounds 4 and 5 used the cap of 250 in all arms.

1. **LM-VAL with PEN125, the next candidate (user, 2026-11-06).** 1av tested LM with R 0.005 and
   0.025 only: 16–27 % smaller programs than LM, NMSE −2 % to +4 %, better on size on Air only
   (EnH Holm p 0.028). 1au chose PEN125 later. No run combines the guard with a penalty.
   **Planned:** round 7 (1aw, `SERVER-ROUND7-PREREGISTRATION.md` after the renumbering).
2. **LM-VAL with no evaluation cap (user, 2026-11-06).** LM runs once per deme. Without the cap,
   regression runs built 5.5–7 demes in 1,000 s (NOCAP, 1as) against 35.5–45 with the cap, so a
   run gets far fewer LM calls. The 1at result can change. **Planned:** round 7 (1aw).
3. **LT-B against RT with no cap (user, 2026-11-06).** Partly covered by round 6 (1ay
   secondary contrasts: LT-B vs RT at N 16 and at three dynamic population sizes). Keep this item
   until 1ay reports; close it if 1ay answers it.
4. **One GOMEA run with all the good findings combined (user, 2026-11-06).** One MOSES-GOMEA
   setting that combines every setting that helped, from the local phases and the server rounds,
   tested against MOSES hill climbing and plain GOMEA. Candidates to review when it is designed:
   no per-deme cap, linear scaling (1ag), `--knob-probe-baseline reduced` (1q), LM with the
   validation guard (1at), PEN125 (1au), the population setting from 1ay, and seeding with
   GP-GOMEA elites (1r, 1u: lower error on EnH and less CPU, but 15–20 % higher error on Con).
   Settings that did not help stay out (uniform start with linear scaling, knob caps, long demes,
   top-k FOS, `-B 3`). Round 8 (confirmation) covers part of this; the difference is the local
   findings, seeding first of all. **Seeding is an arm pair in this experiment:** the combined
   setting with and without GP-GOMEA seeds. Its result (item 5) decides if the final setting uses
   seeding (user, 2026-11-06).
5. **GP-GOMEA seeding with the current settings (user, 2026-11-06).** Run stand-alone GP-GOMEA
   first and start MOSES from its elites instead of the empty program (item 2.9). It was the
   biggest lever of the local phases: 1r (lower error on EnH; GOMEA CPU 4–6× lower, programs 7–8×
   smaller), 1u (mux11 CPU to solve −41 %). But on Con GOMEA's error stayed 15–20 % higher (1y,
   1ab), and on mux11 at equal time it lost to plain HC (1aj). All seeding phases ran before linear
   scaling, LM, no cap and PEN125, so the result can differ now. Test seeded against unseeded with
   the current base, with the seeding cost in the budget. Tools: `tools/gomea/gpgomea_seed.py`,
   `tools/gomea/build_gpgomea_seed.sh`, the `seed` block of the runners. The result decides if
   item 4 uses seeding.

## 8. Future directions from the literature search (2026-11-06)

Source: the literature search of 2026-11-06 (`docs/gomea/LITERATURE-2026-11-06.md`). Start only after rounds 6 to 8 give a
stable conclusion (user). Order: tier 1 first, then tier 2. Citation counts are OpenAlex,
2026-11-06.

**Tier 1: simple experiments**
1. **A1. Node-proximity linkage** (Koch, Alderliesten, Bosman, GECCO 2026, arXiv 2602.02311): the
   FOS from the tree distance between knob positions in the exemplar, not from population data.
   Reported P ≈ 0.80 better than MI, bias-adjusted MI and RT on 5 SR sets, at N 1,024 and 10^7
   evaluations, no runtime cost. Not tested at N 16 or on Boolean. First test: against RT and UNI.
2. **A2. LM during the search** (Kommenda et al., GPEM 2020, DOI 10.1007/s10710-019-09371-3):
   tune constants on accepted improvements inside the deme, not only at deme end. Test at equal
   CPU against LM-VAL.
3. **A4. Case-based exemplar selection** (ε-lexicase with down-sampling: La Cava et al., GECCO
   2016, DOI 10.1145/2908812.2908898; Geiger et al. 2025, arXiv 2407.21632). Arms: classic
   Boltzmann selection; `--diversity-pressure` (classic, no new code, never run in our phases);
   ε-lexicase exemplar selection (a small new rule in `select_exemplar`, `metapopulation.py`,
   from the per-case behavioural scores). Down-sampling can reuse the table weights and the
   subsampling code (`--ss-*`, `--fs-subsampling-ratio`). Boosting (`--boost`) is related, but
   the regression scorer has no `get_error`, so it needs work first. The evidence is for parent
   selection in population GP, not for MOSES exemplar selection.
4. **B4a. Least squares on the top-level terms (user, 2026-11-06).** Many MOSES regression programs
   are a sum of terms. Fit one weight per top-level term with least squares in the scorer, in place
   of the single `a + b·f` of linear scaling. It reuses the representation, the scorer and the
   linear-scaling code. Basis: ITEA (de França and Aldeia, ECJ 2021, DOI 10.1162/evco_a_00285),
   which fits per-term weights by least squares; SRBench (La Cava et al. 2021) puts ITEA in the top
   10 of 21 methods on the real-world sets, "not significantly different from RandomForest or
   AdaBoost" (exact rank not in the text). Related: MRGP (Arnaldo, Krawiec, O'Reilly, GECCO 2014;
   not checked). Risk: overfitting and collinear terms; use the validation guard and, if needed,
   regularization. Also a candidate **alternative to LM if LM does not scale**: one linear solve per
   evaluation (rows × terms²) against up to 10 LM iterations (rows × constants² each).

**Low priority: only if we run out of options (user, 2026-11-06)**
- **A3. Intron-masked MI** (Koch et al. 2026, arXiv 2602.02311): give inactive knob values (under a
  disabled parent, or contin at 0) a "masked" label before the MI step. Disallowed settings need
  no mask: they are not in the knob's domain (`multiplicity`), so they never enter the population.
  Input: the inactivity check of `--gomea-skip-inactive` (check that it covers contin at 0). Signs
  of a small effect: skip-inactive "skips almost nothing" (`USAGE.md`), and 1q changed the
  disallowed settings with no accuracy change. Not compatible with the LT-B bias correction. Test
  only after A1, and only if A1 shows that a better linkage helps.

**Tier 2: harder, needs new representation or deme code**
5. **B1. GP-RV-GOMEA** (Koch et al., PPSN 2024, DOI 10.1007/978-3-031-70055-2_15): real-valued
   GOMEA on the constants of the whole population, together with the structure. Blocker: our
   contin knobs are discrete steps.
6. **B2. Semantic backpropagation with a subtree library** (Pawlak et al., TEVC 2015; with linear
   scaling, Virgolin et al., GECCO 2019): a new knob type that puts in the library subtree closest
   to the output a node should give.
7. **B4-split. LM on the inner constants of each term, then least squares for the term weights
   (user, 2026-11-06).** Basis: Aldeia and de França, GECCO 2022 Companion, DOI
   10.1145/3520304.3533987, and Aldeia's thesis (arXiv 2512.01682, chapter 3). With two inner
   parameters per IT term, `t(θ0 + θ1·∏ x^k)`, four methods were compared on Airfoil, Concrete,
   Energy cooling and heating, Wine red and Yacht. Per-term LM then least squares ("LM+OLS") had the
   lowest or tied test NMSE; LM on all parameters together was the worst (overparameterization,
   start values); least squares alone was "a compelling choice" for accuracy against cost. LM+OLS
   gave larger expressions and more run time than least squares alone. The column labels of the
   thesis tables do not match its text in our extracted copy, so use only the text claims. For us:
   compare against our current joint LM (LM-VAL), after B4a.
8. **B4b. A full IT representation as a deme type.** Knobs: one integer power per (term, feature),
   one transformation per term, one on/off per term; weights by least squares. A fixed-length,
   fully discrete genome, where the powers of a term are a natural group for linkage. Invalid terms
   are dropped after evaluation on the training rows (ITEA, sections 4.2 and 4.5); the power range
   and the maximum number of terms limit the search beforehand. Later step: TIR (de França, GECCO
   2022, arXiv 2205.06807), a rational form; its "state-of-the-art" SRBench claim is the author's
   own and not checked.
9. **B6. Modular GP-GOMEA** (2025, arXiv 2505.01262): several small template trees that call each
   other. Possible hints for multiple demes in GP-GOMEA. **The user reads and validates the paper
   first. No experiment before that.**
10. **B7. Higher-arity GP-GOMEA** (GECCO 2024, arXiv 2402.09854): semantic subtree inheritance and
    greedy child selection. Needs a tree template.

**Control**
11. **hBOA as a deme optimizer (user, 2026-11-06):** a control against GOMEA, as in the original
    MOSES (Looks 2006). Checked 2026-11-06: classic C++ MOSES (`moses/eda/local_structure.h` in
    `OpenCog/asmoses` and `OpenCog/Moses`) has only the BOA scaffolding: the dependency-tree model
    (`dtree`, `split`), sampling, and probability learning. The structure-learning step is a
    commented-out stub (`bde_local_structure_learning`, "XXX implement this"). The only structure
    policy is `univariate`, used by `-a un`. pyMOSES ports exactly this (`pymoses/moses/eda/`,
    `univariate.py`). So neither has BOA dependency learning. A BOA control needs a new
    structure-learning policy (for example greedy decision-tree splits scored by BDe or BIC, as in
    Pelikan's BOA/hBOA and Looks 2006), on top of the existing dtree, split, sampling and
    probability code. `-a un` (univariate EDA, no dependencies) can serve as a no-code control now;
    none of our phases ran it.
