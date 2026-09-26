# Phase 1c pre-registration: how stand-alone GP-GOMEA scales with problem size

Written 2026-09-26, after the Phase 1c pilot and **before any confirmatory run**. The
commit that adds this file is the timestamp.

**Requested by the user:** "So it looks like GP-GOMEA sovles problems but requires more
budget, can we check on bigger problems that can test the scalability of using GP-GOMEA
please? I wanna see how that goes instead." (`APPROVALS.md` #7). The design and criteria
below are agent-proposed.

## Question

As Boolean problems grow, how do GP-GOMEA's success rate and evaluations-to-solve change?
How does that compare with classic C++ MOSES hill climbing on the same truth tables? Both
systems are C++, so the language confound of Phase 1b is gone for wall time.

## Systems

* **GP-GOMEA:** the same build as Phase 1b (`gpgomea-instr`, sha256 `480a0c89…a548`, the
  authors' search unchanged). Settings: IMS `5_1` starting at population **64**, LT FOS,
  AND/OR/NOT, `--prob accuracy`, `syntuniqinit 1000`.
* **Classic C++ MOSES:** `OpenCog/Moses@5a5bf75`, binary sha256 `a3c0f6e5…4444`, the stock
  `-a hc` with every default. Each run also gets `--max-time 1500` so that it stops
  gracefully and reports.

## Tasks (sizes grow within each family)

The truth tables are generated from MOSES's own target functions
(`tools/gomea/make_truth_tables.py`; checksums in `configs/gomea/phase1c/data/`).

| Family | Sizes (inputs) | Rows |
|---|---|---|
| parity | pa3, pa4, pa5, pa6, pa7 | 8 → 128 |
| majority | maj5, maj7, maj9 | 32 → 512 |
| multiplexer | mux6, mux11 | 64 → 2048 |

## Budget and stopping

* **Budget:** 500,000 full-table evaluations per run for both systems.
* **GP-GOMEA** is read at its last generation boundary within the budget. That is
  conservative by less than one generation. Every pilot run finished within 8 minutes.
* **MOSES** stops on a perfect score, on the budget, or on its **1,500 s time limit**; the
  wall cap is 1,800 s. On large representations classic MOSES slows down so much per
  evaluation that it can exhaust the time before the evaluations. In the pilot, maj7
  reached only about 24k evaluations per minute. Such runs are **censored at the
  evaluations they actually used**; the report gives their count and treats them as
  failures.

## GP-GOMEA template height per size (from the pilot; seeds 601–603)

The pilot tried heights {5, 6, 7, 8} at every size.

**Selection rule:** most successes; then the lowest median censored evaluations; then the
lowest median error.

| Size | pa3 | pa4 | pa5 | pa6 | pa7 | maj5 | maj7 | maj9 | mux6 | mux11 |
|---|---|---|---|---|---|---|---|---|---|---|
| Height | 7 | 7 | 8 | 8 | 8 | 7 | 8 | 8 | 5 | 8 |

**Tuning asymmetry, disclosed:** GP-GOMEA gets a per-size height choice; MOSES runs on
defaults. This favours GP-GOMEA.

**Secondary, descriptive only:** height 9 (1,023-node templates) on pa5, pa6, maj9 and
mux11, the sizes the pilot left unsolved or barely solved. It tests whether a larger
template keeps buying progress.

## Seeds and runs

* **GP-GOMEA:** seeds 1001–1020 (20 per size), 200 runs.
* **MOSES:** seeds 1001–1010 (10 per size, the reference; its heavy sizes cost up to 25
  CPU-minutes each), 100 runs.
* **Height 9:** seeds 1001–1010, 40 runs.

Seeds do not pair runs across the two systems.

## Measures and analysis (`tools/gomea/phase1c_report.py`, committed with this file)

* **Per size and system:**
  * success within budget, with a Wilson 95 % CI;
  * success at 20k, 100k and 500k evaluations;
  * the median of censored evaluations-to-solve (a failure counts as budget + 1);
  * **ERT**: the evaluations spent until success or stop, summed over all runs and divided
    by the number of successes; infinite if there are none;
  * median final error, CPU and wall time.
* **Scaling:** a least-squares fit of log10(ERT) against the number of inputs, within each
  family over the sizes with finite ERT. It is reported as a growth factor per extra input,
  and only when at least 2 points are finite.
* **Per-size tests:** Fisher's exact test on success and Mann–Whitney U on censored
  evaluations, both unpaired and Holm-corrected within each family.
* **What counts as "GP-GOMEA scales better"** on a family:
  * at the larger sizes it has a significantly higher success rate (Holm p < 0.05) and at
    least 10 percentage points more than MOSES; or
  * its ERT growth factor per input is smaller, with both fits defined.

  "Scales worse" is the mirror image. Anything else is reported as mixed or undetermined.
