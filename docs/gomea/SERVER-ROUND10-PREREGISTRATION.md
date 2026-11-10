# Server round 10 (T1.2): pre-registration (2026-11-10)

**Status:** the design is approved by the user (see `APPROVALS.md`, 2026-11-10). Implemented in 9e86f08, b120bf3, e6040ab, 9f9660d, 61ce435 and 60ca594. The LM reserve for HC-LM-W (see "Code change") is not implemented yet.
Agent-written, not user-approved in detail: splits 1–10, the contrasts and the Holm families.

**Question:** roadmap task T1.2, the last task of stage 1. How does the new base compare with MOSES
hill climbing (with the same linear scaling and LM-VAL) and with plain GOMEA? This round only
reports the results. The user and the agent discuss the reference setting for stage 2 when it
ends (user, 2026-11-10). **Seeding is a separate control experiment** (user, 2026-11-10): the
main result uses only the unseeded base, so a weak seeding result cannot lower it.

## Rules

* As in rounds 1 to 9: on the server, one phase at a time, W from `resources.py`, state in
  `server-run10/`. Round 10 never overlaps with another round.
* **Budget: evaluations, no time limit (user, 2026-11-09).** Each run gets 20,000 evaluations
  (`-m 20000`), as in round 9. LM passes and GP-GOMEA seed evaluations count. CPU is measured and
  reported, not capped. A wall cap of 36,000 s per run is only a guard against a hung process.
* **Splits 1–10** on all 10 datasets (user, 2026-11-10: "No fresh splits necessary"). Earlier
  rounds used these splits for selection, so this round is not a confirmation.
* **No extension:** the primary contrasts pool 100 pairs.
* **Statistics:** see "Contrasts". Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, evaluations used, LM calls and
  guard rejections, deme count, generations per deme, population size; for BASE-S also the seed
  cost and the lineage.
* **S0:** full pytest and the reproducibility check.
* **One change under `pymoses` (user, 2026-11-10):** `--const-opt lm --const-opt-guard val` for
  `-a hc`. See "Code change".

## Arms

| Arm | Flags |
|---|---|
| **BASE** | `-a gomea --gomea-fos lt-b --gomea-pop-size dynamic --hc-fraction-of-nn 0.05 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val --gomea-accept-complexity-ratio 0.125 --gomea-deme-store accepted` (as in round 9) |
| **BASE-S** (control) | BASE, started from GP-GOMEA elites: the seed block of 1bb (`"share": 0.2`, height 3, k = 5, functions `+_-_*`) |
| **NOCAP** (plain GOMEA) | `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1` (as in round 8) |
| **HC-LM** | `-a hc --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val`, no cap (the same cap as GOMEA) |
| **HC-LM-W** | HC-LM + `--hc-widen-search 1` (`-T`): when a step does not improve, HC searches at a larger distance and does not end the deme |

* Datasets: Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac. Splits 1–10.
* **Fair comparison (user, 2026-11-10):** both GOMEA and HC get linear scaling and LM-VAL. Round 8
  compared GOMEA with LM against HC with linear scaling only.
* **What stays different:** PEN125 (`--gomea-accept-complexity-ratio`) and the dynamic population
  exist only in GOMEA; HC keeps its own complexity handling and neighbourhood sizing. NOCAP (plain
  GOMEA) has no LM on purpose: it shows what the base adds to plain GOMEA.
* **Widen search (user, 2026-11-10):** the user reports a small gain from it in an earlier
  experiment. This repo has no record of that experiment.

## Contrasts

* **Primary (3), the main experiment, pooled over 100 pairs (10 datasets × 10 splits), test
  NMSE:**
  1. BASE vs HC-LM;
  2. BASE vs HC-LM-W;
  3. BASE vs NOCAP.
  * **Test:** as in round 8. Paired Wilcoxon signed-rank on r = log(NMSE_A / NMSE_B), two-sided,
    Holm across the 3, α = 0.025. A failed run gets the largest rank against it.
  * **Verdict:** "better" if Holm p < 0.025 and exp(median r) ≤ 0.9; "worse" if Holm p < 0.025
    and exp(median r) ≥ 1/0.9; otherwise "tie".
* **Control, seeding (4), pooled, the same test, its own Holm family:**
  4. BASE-S vs BASE;
  5. BASE-S vs HC-LM;
  6. BASE-S vs HC-LM-W;
  7. BASE-S vs NOCAP.
* **HC widen search (1), pooled, the same test, its own family:** 8. HC-LM-W vs HC-LM.
  The control family never enters the Holm correction of the primary family, so the control arm
  does not change the main result.
* **Secondary, pooled (10), the same test, Holm across the 10:** program size and CPU, each for
  BASE vs HC-LM, BASE vs HC-LM-W, BASE vs NOCAP, BASE-S vs BASE and HC-LM-W vs HC-LM.
* **Per dataset, descriptive:** contrasts 1, 2, 3 and 4 on each dataset (40), paired Wilcoxon on
  test NMSE, Holm across the 40. These give no verdicts (the lowest Holm p is above 0.025). They
  show where each arm wins or loses.

## No adoption rule

The user asked for results only (2026-11-10). The verdicts above describe the data. The reference
setting for stage 2, seeding, and widen search for HC baselines are decided after the round. The
main result is contrasts 1, 2 and 3, with BASE.

## Memory

Round 9 showed that N = 16 needs much memory at 20,000 evaluations: Dow N16 runs failed at 8 GB,
Tow N16 runs at 5 GB. NOCAP uses N = 16. A failed NOCAP run counts as a BASE win, so failures
bias contrast 2.

| Manifest | `mem_limit_mb` | W on the server (64 GB, reserve 6,553 MB) |
|---|---|---|
| Dow | 16,384 | 3 |
| Tow | 8,192 | 7 |
| All other datasets | 5,120 | 11 (the CPU limit) |

* The round 9 mechanism (`manifest_per_run_mb`) gives each manifest its own W.
* If a NOCAP run still fails on memory, the report lists it, and the findings give contrast 2 with
  and without the failed pairs (the second form is descriptive).

## Code change: LM for hill climbing

* **What:** `--const-opt lm` and `--const-opt-guard val` (with `--const-opt-val`) accepted for
  `-a hc`, regression only. The rule is the same as in GOMEA: at the end of each deme, LM on the
  deme's best program, kept if the training error is lower; with the guard, kept only if the
  validation NMSE is not worse. Each LM pass over the training rows counts as one evaluation.
* **Opt-in:** without the flags, HC runs stay byte-identical. An identity test proves it (output
  and events of a fixed HC run, before and after the change).
* **Tests:** HC with LM on a small regression task: LM runs at each deme end, the evaluation count
  includes the LM passes, and the guard rejects a change that makes the validation error worse.
* **LM reserve with widen search (user, 2026-11-10):** with widen search, HC does not end a deme at
  a local optimum. A local check at 20,000 evaluations (Air, Bos, Tow, Yac, split 1) showed 1 deme
  that used all 20,000 evaluations, so the one LM call got no budget. HC-LM in the same check had
  14–28 demes and used at most 465 LM evaluations in one deme. The fix:
  * Only with `-a hc --const-opt lm --hc-widen-search 1`: the climb gets the deme budget minus a
    reserve of min(500, budget // 2) evaluations. LM then runs on the best program with the rest.
  * If LM leaves evaluations unused, MOSES starts the next deme as usual.
  * HC-LM (no widen search) and HC without `--const-opt` stay byte-identical.
* **Not included:** the other `--const-opt` modes (`mutate`, `polish`) for HC.

## Implementation (by a subagent, in a separate clone)

* `configs/gomea/phase1bd/`: manifests (one each for Dow and Tow, one for the 8 others), spec with
  pooled contrasts, budget `{"evals": 20000}`, wall cap 36,000 s.
* `tools/server/plan10.json` (S0, 1bd), a smoke plan, tests, and the SERVER-RUN section.
* The pooled test of round 8 (`set_final.py`, `phase_report.py`): reuse it for the 3 + 4 + 1 + 10
  pooled contrasts, in four Holm families.
* The LM change for HC (see "Code change"), in a separate commit with its own tests.

## Size and time

* 500 runs (5 arms × 10 datasets × 10 splits).
* CPU per run from round 9 at 20,000 evaluations: BASE and BASE-S 200–1,400 s; N = 16 arms
  300–5,500 s. HC is the fastest; LM adds some CPU to HC.
* About 12–17 hours at W = 11, plus about 1.3 hours for S0. The Dow runs at W = 3 add about 1–2
  hours.

## Limits

* One evaluation budget. Equal evaluations are not equal CPU; CPU is reported for each arm.
* Splits 1–10 are selection data: earlier rounds chose settings on them. The results do not
  confirm the base on new data.
* HC with LM is new code. Its first use is in this round, after local tests only.
* The decision to test BASE-S came from 1bb, where seeding was better on 1 of 4 datasets (the user
  chose it against the rule, 2026-11-10). It is a control arm for this reason.

## The user's answers (2026-11-10)

1. Arms: yes, with the new hill climbing option (HC with LM-VAL) and widen search.
2. Splits: no fresh splits; splits 1–10.
3. Memory: Dow 16,384 MB, Tow 8,192 MB.
4. No adoption rule; results only.
5. LM for HC: the agent reads answer 1 ("with the new hill climbing option also added") as approval.
