# Server round 17: pre-registration (2026-11-26)

**Status:** approved by the user on 2026-11-26: "Approved, leave out Dow at 100K"
(`APPROVALS.md`). Not implemented, not run. The user asked for this round and for this text: "HC-LM,
and write the pre-registration now" (2026-11-26). The agent wrote the details below; the user
approved them as a whole.

**The user's question (2026-11-24):** "Do you think more evaluation can show the difference in the
BASE-TLS and HC-LM? Because on most of the dataset where HC-LM won, it's accuracy stagnates while
the BASE-TLS still shows some type of progress?" The user then asked for HC with the constant
tuning of BASE-TLS too: "Why haven't we run HC? I mean normal HC and the one with the TLS's version
of constant tuning? I think it's only fair if we do that don't you think?" (2026-11-26).

**Origin of the ideas:**

| Item | Origin |
|---|---|
| HC-LM at 20,000 and 100,000, paired with round 16 | The user (2026-11-24, `BACKLOG.md` 15.2) |
| HC-TLS (HC with the per-term least squares of BASE-TLS) | The user's question (2026-11-24, `BACKLOG.md` 15.1; again 2026-11-26) |
| "Normal HC" = HC-LM | The user (2026-11-26) |
| No deme cap for HC | The user (2026-11-26): "I don't want HC to have a deme cap that was a limitation created because of the other algorithms and HC doesn't seem to need it." |
| The contrasts, the Dow rule, the pilot gate, the estimate, the proposal to leave out Dow at 100,000 | Agent-written, not approved |

**Questions:**

1. Does BASE-TLS gain more than HC-LM when the budget grows from 20,000 to 100,000 evaluations?
2. At 100,000 evaluations, is BASE-TLS better than HC-LM?
3. With the same constant tuning (per-term least squares, no LM), is GOMEA's search better than
   HC's? At 20,000 and at 100,000.

BASE-TLS (20,000 evaluations) stays the reference. TLS-NP and TLS-LMR stay next in line.

## What we already know

* **Rounds 13 and 14 (1bh, 1bi), 20,000 evaluations:** BASE-TLS against HC-LM is a tie (0.949 and
  0.947). HC-LM wins on nonlinear data with few inputs (Yac, Con, Air). BASE-TLS wins on
  near-additive data (EnH, EnC, Dow, Tow).
* **Training curves of rounds 13 and 14 (post hoc, agent analysis, 2026-11-24):** from 15,000 to
  20,000 evaluations, HC-LM stalls on Boston, Dow, Energy and Tower (0–6 %), while BASE-TLS still
  gains (8–49 %). On Yacht and Airfoil, HC-LM keeps pace. This is the reason for question 1.
* **Round 16 (1bk), first half (splits 81–85, Dow left out, descriptive):** BASE-TLS at 100,000
  has 0.808 times the test error of BASE-TLS at 20,000 (29 of 30 pairs lower), with programs 1.37
  times as large and 12.2 times the CPU. Most of the gain comes before 50,000 evaluations.
* **Matched HC comparisons with plain scaling** (round 3, 1ao and 1ar; round 8, 1ax): GOMEA-LS
  against HC-LS and FINAL against HC-LS were ties, with GOMEA better on Air in round 3.
* **No benchmark run has used `--linear-scaling terms` with HC.** Two earlier tests ran it, but
  neither checked the scores (corrected 2026-11-26, see "Amendments").
* **Round 16, Dow at 100,000:** all 15 GOMEA runs ran out of memory (8,192 MB) in their second
  deme (2,322 genes). Round 16b runs 15 more; the agent expects them to fail in the same way.

## Rules

* As in rounds 1 to 16: on the server, one phase at a time, W from `resources.py`, state in
  `server-run17/`. Round 17 starts only after round 16b has ended, and never overlaps with another
  round. One plan if the pilot estimate is at most 20 h; else two plans that chain, split by
  splits (81–85, then 86–90), as in round 16. The round runs in one go, with no separate server
  smoke step.
* **One phase (1bl, regression):** the 7 data sets of round 16 (Air, Con, EnC, EnH, Yac, Tow, Dow;
  Dow at 20,000 only), splits 81–90 (already built on the server for round 16), a budget of 20,000 or 100,000
  evaluations for each run (`-m`), no time limit, a wall cap of 36,000 s as a guard only.
* **Pairing with round 16:** each HC run pairs with the round 16 run of the same data set, split,
  seed and budget (BASE-TLS or BASE-TLS-100K). The report reads the round 16 results from
  `../gomea-results/phase1bk/` beside its own.
* **The code must not change the round 16 arms.** If the HC-TLS check (below) needs a code change,
  the change must keep the output of BASE-TLS byte-identical (identity tests on the round 16
  configs, and a rerun of 3 round 16 BASE-TLS runs that gives the same program and score). If that
  is not possible, HC-TLS is left out of the round and the user decides.
* **Statistics:** as in rounds 12 to 16. Failed runs count as infinitely bad. Paired Wilcoxon
  signed-rank on log(A / B), two-sided, Holm in each family, α = 0.025. "Better" or "worse" also
  needs a ratio at least 10 % from 1. The interaction contrast is the one of rounds 15 and 16:
  d = log(A1 / B1) − log(A2 / B2) for each pair; "A1 gains more" if Holm p < 0.025 and
  exp(median d) ≤ 0.9; "A2 gains more" if Holm p < 0.025 and exp(median d) ≥ 1/0.9; else "tie".
* **Program size:** as in rounds 13 to 16 (the node count plus `n_terms` with the per-term fit).
* **S0:** full pytest and the reproducibility check. S0 is not a gate.

## The Dow rule (agent-written, before any HC result)

Round 16 has no GOMEA result on Dow at 100,000. A pair with a failed GOMEA run would count as an HC
win, for a memory fault that has nothing to do with the search. So:

* **Every contrast with a verdict uses the 6 data sets without Dow:** 6 × 10 = 60 pairs, the same
  pairs for each contrast and at both budgets.
* **Dow is descriptive:** its 20,000 pairs (BASE-TLS against HC-LM and HC-TLS).
* **Decided (user, 2026-11-26): HC does not run on Dow at 100,000.** These runs have no partner and
  would cost about 45 CPU-h. The round has 260 runs. The Dow view of P1, P2 and P4 with the failure
  rule is then not possible (no HC run on Dow at 100,000), so Dow is described at 20,000 only.

## Arms (260 new runs; the GOMEA partners come from round 16)

| Arm | Budget | Flags | Runs |
|---|---|---|---|
| **HC-LM** | 20,000 | `-a hc --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val` (the round 13 and 14 flags) | 70 |
| **HC-LM-100K** | 100,000 | the same | 60 (no Dow) |
| **HC-TLS** | 20,000 | `-a hc --knob-probe-baseline reduced --linear-scaling terms --const-opt none` | 70 |
| **HC-TLS-100K** | 100,000 | the same | 60 (no Dow) |
| BASE-TLS (round 16) | 20,000 | the round 16 flags (`configs/gomea/phase1bk/sr.json`) | — |
| BASE-TLS-100K (round 16) | 100,000 | the same | — |

* **No deme cap** for either HC arm (`-a hc`, not `-a hc-capped`): the user's decision of
  2026-11-26.
* HC-TLS has the constant settings of BASE-TLS (`--linear-scaling terms --const-opt none`). HC-LM
  and HC-TLS differ in three flags: HC-LM also has `--const-opt-guard val`, so the runner gives it
  the validation file and an events log (corrected 2026-11-26). Both keep HC's own default for `--hc-fraction-of-nn`
  (2.0), as HC-LM since round 11. In BASE-TLS, `--hc-fraction-of-nn 0.05` sizes the GOMEA
  population (`--gomea-pop-size dynamic`), so it is a GOMEA setting.
* The complexity penalty of BASE-TLS (`--gomea-accept-complexity-ratio 0.125`) acts only through
  GOMEA's acceptance, so HC has no equivalent (1t). The report gives the program size of every arm.
* **The HC-TLS check (before the pilot):** a unit test and a small run with `-a hc --linear-scaling
  terms` on a test data set. Pass: the run ends ok, the final model uses the per-term fit
  (`terms.n_terms` ≥ 2 in at least one run), and the score of the final program equals its
  per-term-fit score computed again outside MOSES. If it fails, HC-TLS is left out (P3, P4 and the
  HC-TLS secondaries drop) and the agent reports to the user.

## Contrasts (60 pairs each: 6 data sets × 10 splits)

* **Primary (4), test NMSE, Holm across the 4** (across 2 if HC-TLS is left out):
  1. interaction [BASE-TLS-100K, HC-LM-100K, BASE-TLS, HC-LM] (Q1; "A1 gains more" = the ratio of
     BASE-TLS to HC-LM is at least 10 % lower at 100,000 than at 20,000);
  2. BASE-TLS-100K vs HC-LM-100K (Q2);
  3. BASE-TLS vs HC-TLS (Q3, 20,000);
  4. BASE-TLS-100K vs HC-TLS-100K (Q3, 100,000).
* **Secondary (one Holm family):** BASE-TLS vs HC-LM at 20,000 (rounds 13 and 14 on fresh splits);
  HC-LM-100K vs HC-LM (the HC budget gain); HC-TLS vs HC-LM at 20,000 and at 100,000 (what the
  per-term fit does for HC); interaction [BASE-TLS-100K, HC-TLS-100K, BASE-TLS, HC-TLS]; the 4
  primary contrasts on the training NMSE.
* **Size and CPU (one Holm family):** program size and CPU for primary 2, 3 and 4 and for BASE-TLS
  vs HC-LM at 20,000.
* **Descriptive (no verdicts):** the primary contrasts per data set; Dow (see "The Dow rule"); the
  train–test gap of each arm; HC-TLS-100K vs HC-TLS; the best-so-far training error during the run
  (HC from the MOSES log at deme resolution, GOMEA from the round 16 trace), at 1, 2, 5, 10, 20, 50
  and 100 % of the budget; demes per run; LM calls and guard rejections (HC-LM); the per-term fit
  counters (HC-TLS); the runs with a test NMSE above 100.
* **Power:** with 60 pairs, the lowest two-sided p is far below 0.025 / 4.

## What the agent proposes after the round (the user decides)

* **Primary 1 "A1 gains more":** BASE-TLS profits more than HC-LM from a larger budget. Proposal:
  a round at a larger budget (300,000 or more) on the faster data sets, and a decision on the
  standard budget, with the CPU cost beside it.
* **Primary 2 "A better":** at 100,000, GOMEA beats HC-LM. With primary 1 a tie, the advantage
  existed at 20,000 already, below the 10 % band.
* **Primary 3 or 4 "A better":** with the same constant tuning, GOMEA's search beats HC's. "Tie":
  the gain of BASE-TLS comes from the per-term fit, not from GOMEA. "A worse": HC-TLS becomes a
  candidate for the reference, under the reference rule of round 14.
* **No automatic adoption.** Each verdict is reported with the size, CPU and train–test gap (the
  user's accuracy-size rule).

## Local pilot before the server (a gate, not a selection)

* On the merged code: Con and Yac, old split 1, all 4 HC arms; Tow, old split 1, HC-LM-100K and
  HC-TLS-100K (the memory and time probe). Do not read the NMSE.
* **Gate:**
  1. every run ends ok;
  2. the peak RSS of each run is at most 50 % of its limit; else that data set and budget get a
     larger limit (the agent reports the new W and time);
  3. the evaluation count of each run is recorded (a run that ends before its budget is reported);
  4. HC-TLS: the per-term fit is active (`terms.n_terms` ≥ 2) in at least one run per data set.
* If a check fails, the agent reports to the user before the server run.

## Memory (agent proposal; the pilot sets the final values)

* In round 14, HC-LM at 20,000 peaked at 250–635 MB on the 6 data sets and at 1,983 MB on Dow.
* Each run list gets a limit from the measured peaks, so that most lists run 11 at a time (the
  user's question of 2026-11-26 about the 64 GB): the 20,000 arms 2,048 MB (Dow 4,096 MB); the
  100,000 arms from the pilot (at most 50 % of the limit in use). In round 16 the 8,192 MB limit
  of the 100,000 arms held the runs to 7 at a time, while their peaks (at most 5.3 GB per run)
  suggest that about 25 GB of the 64 GB were in use (agent estimate).

## Size and time (agent estimate, before the pilot)

* HC-LM at 20,000 (round 14 server medians, the 7 data sets): 4,617 CPU-s per split, about 13.5
  CPU-h for 70 runs (with the round 14 mean / median factor 1.05).
* The CPU factor from 20,000 to 100,000 is not known for HC. In round 16, GOMEA used 12.2 times the
  CPU for 5 times the evaluations (the later demes are larger). The estimate uses a factor of 10
  (range 5–13): HC-LM-100K about 110 CPU-h on 6 data sets (range 55–145).
* HC-TLS has never run. The estimate uses the cost of HC-LM (range 0.5–1.5 times).
* **Total without Dow at 100,000: about 250 CPU-h (range 100–400).** At 11 runs at a time, about
  23 h plus the last long runs: likely two plans that chain, about 12–14 h each.
* The estimate in the chat of 2026-11-26 ("about 150 CPU-h") assumed a factor of 5. Round 16 showed
  that the factor is larger.

## Implementation (by subagents, in separate clones, after approval)

| Clone | Content |
|---|---|
| A | The HC-TLS check (test and small run). A fix only under the identity rule of "Rules". |
| B | `configs/gomea/phase1bl/` (the run lists with their memory limits, `spec.json` with the families above and the round 16 result directories), `plan17.json` (and `plan17b.json` if needed), the smoke plans, the report reading two rounds, the HC progress curves at deme resolution, tests, the SERVER-RUN section; the local chained smoke run; then the pilot |

* Then: merge, the full test suite, the RV16 review, and the push only with a fresh approval.

## Limits

* The GOMEA partners ran in another round. The machine and the code are the same, and each run
  uses one core (CPU / wall time = 1.000 in round 16), so the CPU is comparable. The number of
  runs at a time differs (7 or 11 against the new limits), which can change the CPU a little
  through shared memory bandwidth.
* Dow is out of every verdict (see "The Dow rule").
* Two budgets only (20,000 and 100,000).
* HC-TLS has no earlier result; its check can remove it from the round.

## Amendments

* **A1 (2026-11-26, facts, no design change).** The HC-TLS check passed with no code change (clone
  A; tests `tests/gomea/test_hc_tls.py`, commit `092901a`): 6 probe runs (Con and Yac, old splits
  1–3, 2,000 evaluations) ended ok; the final model used the per-term fit in all 6 (n_terms 2–10,
  no fallback); the printed scores of 60 candidates equal their per-term-fit scores computed again
  (largest relative difference about 5.6e-8). Corrections: two earlier tests had run HC with
  `--linear-scaling terms` without a score check; HC-LM and HC-TLS differ in three flags, not two.
  The runner writes no events log for HC-TLS (`--const-opt none`), so the HC progress curves and
  deme counts come from the MOSES process, as "Contrasts" already says. More exactly (RV16 F3): a
  hook in the MOSES process writes, after each deme, the two values of the "Stats:" row of the
  MOSES log (evaluations and best raw score); the runs keep `-l WARN`, so the log itself has no
  such row (`tools/gomea/hc_progress.py`).
* **A2 (approved by the user, 2026-11-26).** The descriptive item "the per-term fit counters
  (HC-TLS)" cannot be built without new output code: the counters are not written anywhere, for
  GOMEA or for HC. Proposal: report the `terms` record of the final model instead (n_terms and the
  fallback share), which the runner already writes for every arm with `--linear-scaling terms`.
  The user chose this version ("A2: terms record") over the counters.
* **A3 (2026-11-28, agent-written, waiting for the user): the pilot values.** The local pilot of
  "Local pilot before the server" ran at commit `8dda181` (the F8 fix `5fb1da1` and the pilot run
  list `configs/gomea/phase1bl-pilot/`: the arms, flags, keys, wall cap and limits of
  `configs/gomea/phase1bl/`, old split 1). One run at a time, nice 19, each in a systemd scope
  (20,000: MemoryMax 2,200 MB; 100,000: MemoryMax 4,600 MB), started only with free memory
  (2,500 MB and 5,500 MB). The agent's script `run_pilot.sh` ran the first 6 runs and stopped on
  2026-11-27 at 11:33Z, after 60 min of low memory (a stop rule of the script). The user then
  asked for a memory wait without a time limit (relayed by the main agent), so the main agent
  started a supervisor service (`pilot_supervisor.sh`). It ran the other 4 runs (Yac and Tow at
  100,000) with the same command and caps, and ended on 2026-11-28 at 08:28Z. The user chose the
  full pilot (all 6 runs at 100,000). Nobody read an NMSE for the gate.

  | Run | Status | Evals | CPU s | Wall s | Peak RSS MB | Scope peak MB | Limit MB | Peak / limit | Demes | Evals per deme (median, max) | n_terms |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Con HC-LM | ok | 20,000 | 338 | 340 | 181 | 211 | 2,048 | 8.8 % | 21 | 921, 1,556 | – |
  | Con HC-TLS | ok | 20,000 | 890 | 895 | 333 | 361 | 2,048 | 16.3 % | 20 | 1,021, 1,485 | 8 |
  | Yac HC-LM | ok | 20,000 | 244 | 245 | 146 | 173 | 2,048 | 7.1 % | 28 | 683, 1,500 | – |
  | Yac HC-TLS | ok | 20,000 | 339 | 341 | 324 | 351 | 2,048 | 15.8 % | 16 | 916, 7,218 | 8 |
  | Con HC-LM-100K | ok | 100,000 | 2,962 | 2,986 | 679 | 709 | 8,192 | 8.3 % | 96 | 997, 1,647 | – |
  | Con HC-TLS-100K | ok | 100,000 | 11,276 | 11,324 | 1,586 | 1,620 | 8,192 | 19.4 % | 83 | 1,211, 1,689 | 11 |
  | Yac HC-LM-100K | ok | 100,000 | 2,643 | 2,650 | 705 | 737 | 8,192 | 8.6 % | 132 | 746, 1,500 | – |
  | Yac HC-TLS-100K | ok | 100,000 | 4,951 | 4,964 | 935 | 964 | 8,192 | 11.4 % | 107 | 877, 7,218 | 8 |
  | Tow HC-LM-100K | ok | 100,000 | 11,123 | 11,160 | 1,983 | 2,020 | 8,192 | 24.2 % | 47 | 2,356, 2,796 | – |
  | Tow HC-TLS-100K | ok | 100,000 | 31,751 | 31,904 | 2,802 | 2,837 | 8,192 | 34.2 % | 39 | 2,707, 3,413 | 23 |

  Peak RSS is `ru_maxrss_kb` of the summary (the MOSES process, as in rounds 10 to 16); the scope
  peak (`memory.peak`) also counts the runner. Demes and evaluations per deme come from
  `hc_progress.jsonl`.

  | Gate item | Result |
  |---|---|
  | 1. Every run ends ok | Pass: 10 of 10 (exit 0, no OOM kill, no error, no `hc_progress_error`) |
  | 2. Peak RSS ≤ 50 % of the limit | Pass: at most 34.2 % (34.6 % with the scope peak; Tow HC-TLS-100K); the 20,000 runs at most 16.3 % (18 %) |
  | 3. Evaluation counts | Pass: every run used its full budget; the last line of the hook equals `n_evals` in 10 of 10 |
  | 4. HC-TLS per-term fit active | Pass: `n_terms` 8 to 23 in 5 of 5 HC-TLS runs, no fallback |

  * **CPU ratios (pilot, same data set and split):** HC-LM 100,000 / 20,000: Con 8.8, Yac 10.9
    (the estimate used 10, range 5–13). HC-TLS / HC-LM: at 20,000 Con 2.6, Yac 1.4; at 100,000
    Con 3.8, Yac 1.9, Tow 2.9 (the estimate used 0.5–1.5, so HC-TLS costs more than registered).
    Local speed: the pilot HC-LM 20,000 runs used 0.99 (Con) and 0.86 (Yac) times the round 14
    server median CPU.
  * **New memory limits (`mem_limit_mb`, RLIMIT_AS) and W on the server** (cgroup 65,536 MB,
    12 CPUs, reserve 6,553 MB; W = min(11, floor(58,983 / limit)), `resources.py`):

    | Run list | Old | New | W old → new | Reason |
    |---|---|---|---|---|
    | `sr-100K.json` (6 data sets, 100,000) | 8,192 (placeholder) | 6,144 | 7 → 9 | Highest pilot peak 2,837 MB (Tow HC-TLS-100K) = 46 % of 6,144 MB. Tow is the largest data set (2,501 training rows, 25 inputs; Air 753 and 5, EnC and EnH 384 and 8), so its peak also covers Air, EnC and EnH, which have no pilot run |
    | `sr.json` (6 data sets, 20,000) | 2,048 | 4,096 | 11 → 11 | Air, EnC, EnH and Tow have no pilot run at 20,000. Round 14 HC-LM peaked at most 635 MB on the data sets other than Dow; in the pilot, HC-TLS peaked 1.8–2.2 times HC-LM at 20,000, so about 1,400 MB is possible, above 50 % of 2,048 MB. 4,096 MB costs no W (the CPUs limit W) |
    | `sr-Dow.json` (Dow, 20,000) | 4,096 | 8,192 | 11 → 7 | No pilot run on Dow. Round 14 HC-LM peaked at 1,983 MB on Dow (48 % of 4,096 MB). With the pilot HC-TLS / HC-LM peak ratios (1.3–2.3), HC-TLS can reach about 2,600–4,600 MB. 8,192 MB keeps the central value (about 3,600 MB) at 44 %; the high value is 56 %. Dow is descriptive only |

    * Why not the smallest limit for 10 runs at a time (5,888 MB, W = 10, peak 48 %, about 4 h
      shorter): in round 16 the highest server peak at 100,000 (about 5.3 GB) was about 1.6 times
      the pilot peak of Tow (3.3 GB). With the same growth, Tow HC-TLS-100K can reach about
      4.5 GB. An address-space failure counts as infinitely bad in P2 or P4, so the agent chose
      the larger margin. The user can choose 5,888 MB.
    * Applied in the clone: the four config sets (`phase1bl`, `phase1bl-a`,
      `server-smoke17-1bl`, `server-smoke17-1bl-a`), the 1bl steps of the four plans, the
      SERVER-RUN section and the tests. The pilot run list keeps the values it ran with.
  * **Estimate (method of round 16, script `est17.py` in the agent's scratchpad):** server CPU of
    a run = the round 14 server median CPU of HC-LM at 20,000 on its data set (Air 886, Con 343,
    Dow 831, EnC 402, EnH 476, Tow 1,396, Yac 283 s) × the pilot CPU ratio of the arm to HC-LM at
    20,000 on the same data set. The local speed cancels in the ratio. Air, EnC, EnH and Dow
    use the mean of the Con and Yac ratios (low: the lower, high: the higher). Tow at 100,000 =
    the pilot CPU / the local speed (central: the mean, 0.92; low: 0.99; high: 0.86). Each arm × 10 splits × 1.05 (the round 14
    mean / median factor); the high case × 1.1 more. Wall = CPU-h / W of each run list, plus S0
    (1.3 h), plus a tail of at most one long run per plan.

    | | HC-LM | HC-TLS | HC-LM-100K | HC-TLS-100K | Total CPU-h | Wall (two plans) |
    |---|---|---|---|---|---|---|
    | Low | 13.5 | 20.0 | 95.8 | 248.8 | 378 | 22.1 h + 20.8 h = 42.9 h |
    | **Central** | **13.5** | **27.2** | **103.4** | **288.7** | **433** | **25.1 h + 23.8 h = 48.9 h** |
    | High | 14.8 | 37.9 | 122.4 | 362.4 | 538 | 30.9 h + 29.6 h = 60.5 h |

    Central wall per run list: `sr.json` 33.4 CPU-h / 11 = 3.0 h, `sr-100K.json` 392.0 / 9 =
    43.6 h, `sr-Dow.json` 7.3 / 7 = 1.0 h. The longest single runs: Tow HC-TLS-100K about 9.6 h
    (9.0–11.3 h), Air HC-TLS-100K about 6.6 h (5.0–9.0 h). The estimate before the pilot was
    about 250 CPU-h (range 100–400) and 23 h; the difference comes from HC-TLS (about 316 of
    the 433 CPU-h) and from W = 9, not 11, for the 100,000 arms.
  * **Plan split:** the estimate is more than 20 h, so the round keeps two plans that chain, split
    by splits (81–85, then 86–90), as registered. Each plan is about 25 h (21–31 h) plus its
    tail, more than the 20 h of one plan. The rule has no third plan; a split into more plans
    is a design change for the user.
  * **Agent proposal, not approved: a larger wall cap for the 100,000 arms.** The pilot Tow
    HC-TLS-100K run used 31,904 s, 89 % of the wall cap of 36,000 s. The server estimate for
    this arm is 9.0–11.3 h (about 32,000–41,000 s), so some of its 10 runs can reach the cap.
    A run at the cap ends as "timeout" and counts as infinitely bad, so P4 and the HC-TLS
    secondaries would count a time guard against HC-TLS. The pre-registration calls the cap "a
    guard only". Proposal: `wall_cap_s` 72,000 s in `sr-100K.json` of `phase1bl` and
    `phase1bl-a` (the smoke keeps 300 s). The CPU estimate does not change; a plan can end up to
    about 10 h later only if a run is longer than 36,000 s. The configs keep 36,000 s until the
    user decides.
