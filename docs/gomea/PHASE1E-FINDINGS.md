# Phase 1e findings: can embedded GOMEA beat MOSES hill climbing?

**Status: complete.** Stage 1 ran 330 development runs and stage 2 ran 120 confirmatory runs.
25 stage 2 runs were re-run under raised limits (a disclosed deviation, below). All 450
runs used for the analysis ended `ok`.

**Protocol:** `PHASE1E-PREREGISTRATION.md`. The code is `8224f3f`, and `63023d9` for the
re-runs, which changes only the harness.

## Answer

**Yes, on one of three larger tasks, measured in evaluations.** On mux11, capped GOMEA
(`LTB-C250-M`) solved every seed in a median of **4,633 evaluations vs HC's 8,757** (both
20/20; Holm p = 0.0063). That is the first pre-registered *better* verdict for embedded
GOMEA **against hill climbing** in this project.

**Not on the others.** maj7 and pa5 are *ties*: neither method reliably solves them at
40,000 evaluations.

**Not in time.** GOMEA used about 3–4× HC's CPU time on mux11 and maj7, and it
was slower to reach the solution on every mux11 seed.

| Task | HC | LTB-C250-M | Paired W/T/L on final error (GOMEA vs HC) | Holm p (error, evals) | Verdict |
|---|---|---|---|---|---|
| mux11 | 20/20; median 8,757 evals | 20/20; median **4,633** evals | 0/20/0 | 1, **0.0063** | **better** |
| maj7 | 1/20; median error 3 | 0/20; median error 3 | 4/4/12 | 0.58, 1 | tie |
| pa5 | 0/20; median error 7 | 0/20; median error 8 | 6/5/9 | 0.37, 1 | tie |

Sources: `phase1e-results/stage2_aggregate.csv` and `stage2_verdicts.csv`.

## Stage 1: choosing the setup (development seeds 2001–2010; not confirmatory)

Ten GOMEA and hybrid setups were tried on maj5, mux6 and pa4 at 20,000 evaluations. The
pre-registered rule ranks them by summed success rate, then by summed median
evaluations-to-success.

| Rank | Arm | Summed success rate | Summed median evals |
|---|---|---|---|
| 1 | **LTB-C250-M** (lt-b, cap 250, memo) | **2.0** | 31,016 |
| 2 | UNI-S200-M (uni, stall stop 200, memo) | 1.8 | 32,548 |
| 3 | UNI-C500-M | 1.8 | 34,168.5 |
| … | the hybrids HYB-LTB, HYB-UNI | 1.6, 1.5 | 39,685.5, 38,549 |

For reference, HC's summed success rate was 2.0 (9/10, 10/10, 1/10), the same as the
winner's (8/10, 10/10, 2/10). Descriptively:

* **A smaller cap helped the linkage tree.** LTB-C250-M beat LTB-C500-M on maj5 (8/10 vs
  7/10; median 9,094 vs 14,818.5 evaluations).
* **The HC→GOMEA hybrids did not help.** They solved maj5 only 5/10 times, against HC's
  9/10. *Interpretation:* when HC stalls, a fresh deme may be worth more than continuing
  the old one with GOMEA.
* **The memo was mixed.** It helped the linkage tree on maj5 (7/10 vs 5/10) and on mux6
  (median 1,991.5 vs 2,783 evaluations), and made little difference for uni.

## Stage 2: the larger problems

**mux11.**
* GOMEA needed fewer evaluations than HC on 16 of 20 seeds; the median paired ratio is
  0.55.
* GOMEA's evaluations-to-solve range from 3,485 to 13,734; HC's from 5,749 to 18,979.
* GOMEA ran 19 demes per run (median) against HC's 6. The per-deme cap of 250 keeps it
  moving to fresh representations, as in Phase 1d.

**maj7.**
* All 20 GOMEA runs used the full 40,000 evaluations without solving it, with final
  errors of 2–6.
* HC solved seed 3006, and its other runs ended with errors of 1–7.
* On final error GOMEA won 4 seeds, tied 4 and lost 12. That is not significant.

**pa5.** Neither method solved any seed. The median error was 7 for HC and 8 for GOMEA.

### Cost in time and memory (pure Python, same machine)

| Task | HC median CPU s | GOMEA median CPU s | HC max RSS | GOMEA max RSS |
|---|---|---|---|---|
| mux11 | 828 | 2,777 | 44 MB | 805 MB |
| maj7 | 5,130 | 18,623 | 108 MB | 139 MB |
| pa5 | 2,657 | 2,748 | 92 MB | 65 MB |

On mux11 the median CPU time to the solution was 710 s for HC and 2,772 s for GOMEA.
GOMEA was slower on all 20 seeds.

The median CPU cost per scorer call:

| Task | HC | GOMEA | Ratio |
|---|---|---|---|
| mux11 | 92 ms | 525 ms | 5.7× |
| maj7 | 129 ms | 466 ms | 3.6× |
| pa5 | 66 ms | 69 ms | about 1× |

In the original maj7 GOMEA runs, whose representations had a median of 472 knobs, time
spent inside the GOMEA code was about 16 % of the runs' CPU. Of that, scoring took about
84 % and linkage learning about 9 %. The rest of the run time was spent outside the
optimizer, in MOSES's own per-deme work. GOMEA ran 161 demes per maj7 run (median)
against HC's 43, which plausibly explains much of that (*not measured directly*).

## Deviation: the re-runs

As registered, stage 2 did not give an equal-evaluation comparison:

* all 20 maj7 GOMEA runs hit the 7,200 s wall cap, after a median of about 22,000
  evaluations;
* on mux11, three GOMEA runs hit the 512 MB memory limit and one hit the wall cap;
* one mux11 HC run (seed 3016) crashed with a `RecursionError`. MOSES's recursive
  neighborhood counting nests once per knob, past Python's default limit of 1,000.

With the user's approval (`APPROVALS.md` #10), those 25 runs were re-run with the same
code, seeds and arguments. The only changes were a 36,000 s wall cap, a 2,048 MB memory
limit and a recursion limit of 100,000. These limits decide whether a run can finish, not
its trajectory. Every re-run ended `ok`:

* the maj7 GOMEA runs took 3.3–6.7 h of wall time;
* the four mux11 GOMEA re-runs solved it in 4,407–12,310 evaluations;
* the HC seed-3016 re-run solved it in 9,679.

**Sensitivity of the mux11 verdict.**
* **Dropping the four affected seeds** (3003, 3005, 3016 and 3019; seed 3016 was re-run
  for both methods): 16 seeds remain. GOMEA's median is 4,496.5
  against HC's 8,757, GOMEA needs fewer evaluations on 13 of 16, and the raw Wilcoxon
  p is 0.0017.
* **Counting the five re-run results as failures instead:** the Wilcoxon p is 0.24, so the
  verdict would be a tie. The *better* verdict therefore rests on letting runs finish,
  which the equal-evaluation design intends.

## Interpretation

Phases 1d and 1e point the same way. Inside MOSES, GOMEA pays off only when each deme is
kept short: a cap of 250–500 evaluations. Then the linkage model can exploit a
representation for a few hundred evaluations before MOSES rebuilds it. On mux11, whose
solution is made of reusable blocks, that halves the evaluations to solve. On majority
and parity at this budget, neither method gets close enough for the difference to show.

*Not tested:* whether the mux11 gain holds at other sizes or budgets, and whether a
compiled GOMEA would remove the CPU-time penalty.

## Recommendation

* When scorer calls are the cost, e.g. an expensive fitness function, use
  `-a gomea --gomea-fos lt-b --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`.
  It needed about half HC's evaluations on mux11 and tied on maj7 and pa5. In stage 1
  it needed somewhat more than HC on mux6 (median 1,921 vs 1,644, descriptive).
* When CPU time is the cost, stay with `-a hc`. On large representations the
  pure-Python GOMEA runs cost 3.6–5.7× as much CPU per evaluation (about the same on pa5).
* Give runs that use large representations more than 512 MB of memory.
* The recursion limit is a pyMoses port limitation, and belongs in the Phase 2 backlog.

Evidence: `phase1e-results/` (aggregates, verdicts, manifests and checksums) and
`../gomea-results/1/{phase1e-stage1-*,phase1e-stage2-*,phase1e-stage2-rerun-*}`.
