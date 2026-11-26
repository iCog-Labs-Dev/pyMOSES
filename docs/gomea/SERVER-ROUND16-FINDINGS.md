# Server round 16: findings (2026-11-26)

**Pre-registration:** `SERVER-ROUND16-PREREGISTRATION.md` (approved 2026-11-23). **Results:** two
bundles in the MOSES folder, next to the repo: `bundle-final-20261125T130146Z.tar.gz` (`round16`,
S0 and splits 81–85; SHA-256 `3feeb837…816fc15`) and `bundle-final-20261126T192347Z.tar.gz`
(`round16b`, all 420 runs and the final report; SHA-256 `4c5290ba…e2518f`). **Code:** `6dc84b3`
(both plans). **Decisions:** `APPROVALS.md`.

NMSE values are × 100. "Pairs" is the number of (data set, split) pairs where A is lower / where B
is lower / ties. "Ratio" is exp(median log ratio): below 1 means that A has the lower value. For an
interaction contrast, d = log(A1 / B1) − log(A2 / B2) (100,000 against 20,000): a ratio below 1
means that the RV arm loses less to BASE-TLS at 100,000. A verdict of "better", "worse" or "gains
more" needs Holm p < 0.025 and a ratio at least 10 % from 1.

**Dow is left out of every arm in the main tables** (the user, 2026-11-26: "Without dow for neither
BASE-TLS nor RV based one. I want to see a fair analysis."). All 30 Dow runs at 100,000 failed (see
"Run health"), so a report with Dow counts them as infinitely bad on both sides. The server report
with Dow is in the round16b bundle (`phase1bk/report-stage1`). With Dow, one verdict differs: TLS-RV-100K
against BASE-TLS-100K on training error is a tie (1.076, 16/44/10), not "worse"; the 10 Dow pairs
where both arms failed are ties.
The no-Dow report: `tools/gomea/phase_report.py --stage 1` with the round 16 spec less Dow, on the
360 runs of the 6 other data sets (agent analysis, 2026-11-26).

## Why this round was run

* **The user's question (2026-11-23):** "Can we test if increasing the number of evaluation for
  everyone has a better effect in the long term? [...] I do want to know how much better the result
  is going to be in the end and make sure we cover all of our bases so that we can understand this
  better."
* **The background.** Real-valued GOMEA on the constants (option C of T3.2, the RV step) failed its
  spike gate at 20,000 evaluations (median training ratio 1.234). In the last quarter of the spike
  budget, the RV arm improved a little faster than the reference. That weak sign (post hoc) was the
  reason to test a 5 times larger budget for every arm.
* **The three questions:**
  1. At 100,000 evaluations, is the RV step (option C) better than BASE-TLS?
  2. Does the ratio of the RV arm to BASE-TLS fall when the budget grows from 20,000 to 100,000?
  3. How much does BASE-TLS itself gain from 5 times the budget?
* **Arms:** BASE-TLS (the reference); TLS-RV (`--gomea-rv-step rvg --gomea-rv-budget balance`:
  after each GOM generation, an RV generation with probability p_RV); TLS-RVL (`--gomea-rv-budget
  late`: an RV generation only after a GOM generation that did not improve the deme's best). Both
  RV arms model the active genes (non-zero, not absorbed by the per-term fit). Each arm at 20,000
  and 100,000 evaluations; 7 data sets, splits 81–90.

## Summary

1. **Question 1: no.** At 100,000, TLS-RV is 9 % worse than BASE-TLS on the test error (ratio
   1.093, pairs 17/43/0, Holm p 0.00084). By the 10 % rule this is a tie, but the direction is
   against RV on 5 of 6 data sets (only Tower is lower, 0.946). Its training error is worse too
   (1.115, "worse"). TLS-RVL is level with BASE-TLS (1.003, 20/30/10, tie).
2. **Question 2: a little, not enough.** The RV gap shrinks from 1.267 at 20,000 to 1.093 at
   100,000 (interaction 0.936, pairs 40/20, Holm p 0.009). The change is under the 10 % margin, so
   the verdict is a tie. The largest changes: Energy heating 0.659 and Tower 0.753. TLS-RVL: no
   change (1.000).
3. **Question 3: 18 % better at 12 times the CPU.** BASE-TLS-100K against BASE-TLS: test 0.820
   (56/2/2, Holm p 7.8e-09, "better"), training 0.780. The programs are larger (median per-split
   size ratio 1.50). The CPU time is 12.0 times larger for 5 times the evaluations (the later demes
   are larger).
4. **Inside the 100,000 runs, the gain slows down but does not stop:** the best training error falls
   13 % from 20,000 to 50,000 evaluations and 8 % from 50,000 to 100,000 (BASE-TLS-100K, median over
   60 runs).
5. **RV gives smaller and cheaper programs:** TLS-RV-100K against BASE-TLS-100K: size 0.747
   ("better"), CPU 0.618 ("better"). TLS-RV at 20,000: size 0.816, CPU 0.791, but test 1.267
   ("worse").
6. **The pre-registered proposal** ("What the agent proposes after the round"): primary 1 and 2 are
   ties, and primary 3 and 4 are ties, so the "Otherwise" case applies: "no sign that the budget
   helps RV. Proposal: stop T3.2 (options C and A)." The user decides.

## Main contrasts (test NMSE, no Dow, 60 pairs each)

| contrast | family | ratio | pairs (A/B/ties) | Holm p | verdict |
|---|---|---|---|---|---|
| TLS-RV-100K vs BASE-TLS-100K | primary | 1.093 | 17/43/0 | 0.00084 | tie |
| TLS-RVL-100K vs BASE-TLS-100K | primary | 1.003 | 20/30/10 | 0.245 | tie |
| RV gain, 100,000 vs 20,000 (interaction) | primary | 0.936 | 40/20/0 | 0.0091 | tie |
| RVL gain, 100,000 vs 20,000 (interaction) | primary | 1.000 | 29/21/10 | 0.364 | tie |
| TLS-RV vs BASE-TLS | secondary | 1.267 | 6/54/0 | 1.0e-08 | A worse |
| TLS-RVL vs BASE-TLS | secondary | 1.000 | 12/28/20 | 0.053 | tie |
| BASE-TLS-100K vs BASE-TLS | secondary | 0.820 | 56/2/2 | 7.8e-09 | A better |
| TLS-RV-100K vs TLS-RV | secondary | 0.726 | 57/3/0 | 2.3e-09 | A better |
| TLS-RV-100K vs BASE-TLS-100K, training | secondary | 1.115 | 16/44/0 | 0.0014 | A worse |
| BASE-TLS-100K vs BASE-TLS, training | descriptive | 0.780 | 58/0/2 | 1.1e-10 | – |

The ties of TLS-RVL: a run with no RV evaluation is the same as its BASE-TLS run. On Tower,
TLS-RVL made 0 RV evaluations in all 20 runs (`rv.rv_evals` 0; the GOM generations improved, so the
"late" rule skipped the RV step), so Tower gives 10 of 10 equal pairs at both budgets.

## Per data set (test NMSE ratio, pairs A/B; descriptive)

| data set | BASE-TLS-100K vs BASE-TLS | TLS-RV vs BASE-TLS | TLS-RV-100K vs BASE-TLS-100K | RV gain (interaction) | TLS-RVL-100K vs BASE-TLS-100K |
|---|---|---|---|---|---|
| Airfoil | 0.841 (10/0) | 1.071 (0/10) | 1.074 (1/9) | 1.007 (5/5) | 1.055 (1/9) |
| Concrete | 0.847 (10/0) | 1.267 (0/10) | 1.244 (2/8) | 0.979 (6/4) | 0.978 (5/5) |
| Energy cooling | 0.635 (10/0) | 1.196 (1/9) | 1.183 (3/7) | 0.937 (7/3) | 1.187 (2/8) |
| Energy heating | 0.550 (10/0) | 1.660 (1/9) | 1.100 (2/8) | 0.659 (8/2) | 0.967 (7/3) |
| Tower | 0.859 (9/1) | 1.347 (2/8) | 0.946 (7/3) | 0.753 (7/3) | 1.000 (0/0, 10 ties) |
| Yacht | 0.986 (7/1) | 1.490 (2/8) | 1.113 (2/8) | 0.872 (7/3) | 0.996 (5/5) |

* **Yacht gains nothing from the budget** (0.986). This agrees with D1 (`BACKLOG.md` 15.4): on
  Yacht the constants of the deme bests stay at c or c ± 1, so more demes of the same search do not
  find better constants. The Yacht check D1b tests this directly (agent reading).

## Program size, CPU and gap (descriptive, no Dow)

| arm | CPU-h (60 runs) | median demes per run |
|---|---|---|
| BASE-TLS | 10.5 | 2.5 |
| TLS-RV | 7.9 | 2.0 |
| TLS-RVL | 9.5 | 2.0 |
| BASE-TLS-100K | 126.3 | 6.5 |
| TLS-RV-100K | 110.2 | 3.0 |
| TLS-RVL-100K | 113.8 | 4.5 |

Train–test gap (test NMSE − training NMSE, × 100, median per data set, `aggregate.csv`):

| data set | BASE-TLS | BASE-TLS-100K | TLS-RV-100K |
|---|---|---|---|
| Airfoil | 1.24 | 2.73 | 1.25 |
| Concrete | 3.01 | 3.81 | 1.09 |
| Energy cooling | 0.52 | 0.42 | 0.51 |
| Energy heating | 0.06 | 0.10 | 0.10 |
| Tower | 1.08 | 1.13 | 0.97 |
| Yacht | 0.02 | 0.06 | 0.08 |

* The budget makes the gap larger on Airfoil and Concrete, where the programs grow most.
  TLS-RV-100K has a smaller gap there (Airfoil 1.25, Concrete 1.09).
* The demes are medians of the per-data-set medians (`aggregate.csv`). Size and CPU
  contrasts: TLS-RV-100K vs BASE-TLS-100K size 0.747 and CPU 0.618; TLS-RVL-100K size 0.912 (tie)
  and CPU 0.870 ("better"); BASE-TLS-100K vs BASE-TLS size 1.50 and CPU 12.0 (median per-split
  ratios, `runs.csv`).
* Per data set, the budget made the programs larger on Airfoil (2.29), Concrete (1.89), Energy
  cooling (1.58) and Energy heating (1.34), and about the same on Tower (1.04) and Yacht (1.14).

## Run health

* 420 runs: 390 ok, 30 failed. **All 30 failures are Dow at 100,000** (10 per arm), with the
  error "lost sys.stderr". Each run finished one deme (34,096–64,538 evaluations) and failed when
  the second deme started (in the interim analysis: 2,322 genes, population 539; the first deme had
  118 genes). The peak resident memory (`ru_maxrss`) was 8,045–8,160 MB, at the 8,192 MB limit.
  "Out of memory" is an inference: the recorded error is "lost sys.stderr". No other run failed.
* Dow at 20,000 (30 runs) ran ok; it is left out of this summary as the user asked.
* Active time: `round16` 29.3 h for 1bk plus 1.3 h for S0; `round16b` 30.4 h; 11 runs at a time
  for the 20,000 lists, 7 for the 100,000 lists (the 8,192 MB limit).
* S0 of `round16`: pytest 10,617 passed, 1 failed, 13 skipped. The failed test is
  `tests/fixtures/test_fixture_manifests.py::test_toolchain_drift_is_clean_on_the_real_repo`: the
  fixture records name the local tool chain, and the server's differs. It is not a code fault.
  The reproducibility check passed (4 of 4).

## Interim against final

The interim summary (splits 81–85, a snapshot of 317 runs) and the final result agree: TLS-RV-100K
1.087 → 1.093; TLS-RVL-100K 1.018 → 1.003; interaction 0.933 → 0.936; BASE-TLS-100K 0.808 → 0.820.

## Observations (agent reading)

* The RV step makes simpler programs, at a cost in accuracy that the budget does not remove. With
  the user's accuracy–size rule, TLS-RV-100K (9 % worse, 25 % smaller, 38 % less CPU) is a
  trade-off, not a win.
* The budget is the strongest lever found so far, but it is expensive: 18 % lower test error for
  12 times the CPU, and the gain per evaluation falls.
* Round 17 (HC against BASE-TLS at both budgets) now has all its GOMEA partner runs.

## Limits

* One budget pair (20,000 and 100,000). 6 data sets in the main tables. The interaction contrast
  compares ratios of independent runs; it says nothing about one run's own progress.
* Dow at 100,000 has no result for any arm (memory).

## User decisions

* **T3.2 continues** (the user, 2026-11-26: "No we are not giving up on the T3.2 yet. I want us to
  keep pushing."). The pre-registered proposal to stop T3.2 is not taken.
* **The Dow memory failure is a backlog item** (the user, 2026-11-26; `BACKLOG.md` 15.6).
* **Correction (2026-11-26):** the first version of this file described TLS-RV as "the RV step on
  every contin gene" and TLS-RVL as "on the live genes only". That was wrong; the arms differ in the
  RV budget rule (`balance` against `late`), as the pre-registration says.
