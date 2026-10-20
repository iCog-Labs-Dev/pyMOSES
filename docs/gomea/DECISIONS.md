# GOMEA in pyMoses: decisions

User decisions that shape what gets built. Each entry gives the decision, who made it, the
evidence it rests on, and what follows from it.

## D1. Accuracy is not the only criterion (user, 2026-10-20)

**Decision:** a moderate rise in test error is acceptable when it buys clearly smaller
programs and much less CPU, especially when the model generalizes as well or better.

* **Why:** the user's words were "More tree nodes complexity means more brittle and
  overfitted solution." Large programs are harder to trust and more likely to overfit.
* **How results are judged from now on:**
  * Every regression comparison reports, next to test NMSE:
    * final program size;
    * the train–test gap (test − train NMSE, and test/train);
    * CPU per run.
  * A variant that loses a little test accuracy but is much smaller, cheaper and generalizes
    as well or better is preferred.
  * A variant that is no smaller and generalizes worse gets no such allowance.
* **Registered verdicts are unchanged:** the statistics still test test NMSE or CPU to solve
  as pre-registered. D1 governs what is adopted, not how a phase is scored.

## D2. Seeding for regression: adopt the best seeded variant (user, 2026-10-20)

**Decision:** on regression, accept seeding's accuracy cost on Con for its size and cost
gains. Use the best variant found: **10 % seeding share (S10)**.

**Evidence (Con; 1y with 1r; medians):**

| Arm | Test NMSE | Train NMSE | Test − train | Test / train | Size (nodes) | CPU per run |
|---|---|---|---|---|---|---|
| Unseeded GOMEA (20 splits) | 35.2 | 32.4 | 2.3 | 1.07 | 229.5 | 2,370 s |
| Seeded, 20 % share (10 splits) | 44.9 | 37.8 | 5.8 | 1.15 | 28.5 | 536 s |
| Seeded + reserve (20) | 42.4 | 38.4 | 4.9 | 1.13 | 28 | 484 s |
| **Seeded, 10 % share (20)** | **40.6** | 39.0 | **0.5** | **1.01** | **32** | 580 s |

* S10 costs about 15 % in test NMSE against unseeded GOMEA.
* In return its programs are about 7× smaller, it uses about 4× less CPU, and it has the
  smallest generalization gap of any arm.
* On EnH (1r), seeding improved test error as well.

## D3. Deme cap: an accuracy drop is acceptable only if it buys something (user, 2026-10-20)

**Decision:** the same principle applies to the per-deme cap. A small accuracy drop at a
longer cap is acceptable when it buys cost or generalization.

**Evidence so far (1x):**

| Cap | mux11 CPU to solve | Con test NMSE | Con test − train | Con size |
|---|---|---|---|---|
| 125 | 3,490 s | 48.5 | 8.1 | 250.5 |
| 250 | 1,464 s | 37.1 | 4.0 | 187 |
| 500 | 1,000 s | 41.8 | 6.0 | 197.5 |

* **mux11:** cap 500 was cheaper by median (not significant).
* **Con:** cap 500 was **not** smaller and generalized **worse** (gap 6.0 against 4.0), so on
  Con D1 does not favour it yet.
* **Status:** pending backlog item 18 (caps 250 / 500 / 1,000, with size and gap reported).
  The cap stays 250 until then.

## D4. Re-seeding is dropped (user, 2026-10-20)

**Decision:** drop re-seeding MOSES from GP-GOMEA on stagnation, in every form. That covers
the tested design (a fresh GP-GOMEA run) and the untested one (GP-GOMEA started from MOSES's
current programs). It will not be tested further or built.

* **Evidence:** Phase 1ab.
  * Re-seeding tied plain seeding on Con: 44.6 against 44.9, with 7/10 splits identical.
  * Fresh elites rarely beat MOSES's current best.
  * Size, gap and CPU stayed in seeded territory with no gain.
* **What remains:** the `reseed_*` seed-block keys stay in the code as an opt-in, unused
  option. Seeding itself stays (D2: 10 % share).

## Open for the build decision

* **Candidate defaults:** the knob-collapse fix, and the uniform start for regression (see
  `FINDINGS-LOG.md`).
* **The uniform start (1v):**
  * Test NMSE fell from 37.1 to 29.2, but programs grew from 187 to 309 nodes.
  * Its gap was about the same (4.4 against 4.0).
  * Under D1 it should be checked together with seeding or the penalty to keep size down.
