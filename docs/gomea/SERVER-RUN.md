# Running the server round

How to run the round pre-registered in `SERVER-ROUND-PREREGISTRATION.md` on a Jupyter pod,
unattended, and what to bring back. Everything is in `tools/server/`.

## What you need

* A Debian-based pod (any Linux with bash, git, curl, tar and coreutils works). No sudo.
* Internet access during set-up: GitHub (uv, micromamba, GP-GOMEA), the Python download
  that uv uses, conda-forge (only if the system has no complete C++ toolchain) and
  gitlab.com (Armadillo, for GP-GOMEA).
* At least 20 GB free disk next to the clone, and a home directory that survives restarts.
* Time: the round is about 300 CPU-hours. With W parallel runs it takes about 300 / W
  hours: 8 cores (W = 7) about 43 h, 16 cores (W = 15) about 20 h, 32 cores about 10 h.
  W also needs 3 GB of memory per run plus a reserve (see "Workers" below), so 16 cores
  with 32 GB give W = 9, about 33 h.

## Steps

1. Open a **Jupyter Terminal** (File > New > Terminal) and clone:

       git clone --branch experiment/moses-gomea-discrete https://github.com/iCog-Labs-Dev/pyMOSES.git pyMoses
       cd pyMoses

   Results go to `../gomea-results` (next to the clone). To put them elsewhere, set
   `export GOMEA_RESULTS=/big/disk/gomea-results` before step 2; set-up then makes
   `../gomea-results` a link to it.

2. Set up (10 to 40 minutes; safe to rerun):

       bash tools/server/setup.sh

3. Read the summary it ends with. `SETUP OK` means you can start. Per step:
   * `disk WARN`: less than 20 GB free. Free space first if you can.
   * `gpgomea FAILED`: the C++ GP-GOMEA could not be built. The round still runs, but
     phases 1ai and 1aj are skipped (logged). The reason is in
     `../gomea-results/server-run/gpgomea-build.log`.
   * any other `FAILED`: fix it (usually the network) and rerun step 2.

4. Start the round:

       bash tools/server/start.sh

   It returns at once and prints how to follow it. The round runs in the background and
   survives closing the terminal or the browser.

## Following it

* `tail -f ../gomea-results/server-run/run.log`: what is happening, with timestamps.
* `bash tools/server/status.sh`: running or not, each step's status, the current phase
  and its W, the last events, memory and load, and the latest scoreboard.
* `cat ../gomea-results/server-run/scores-latest.txt`: per phase, problem and arm, the
  finished runs and running medians (refreshed every 10 minutes).

The order is S0 (tests and a reproducibility check), 1ah, 1ai, 1af-pilot and 1af, 1aj,
1ag, 1ae. One phase at a time.

## Jupyter notes

* A notebook cell `!bash tools/server/start.sh` (run from the clone) also works, since
  the round detaches. Do not run the phases themselves in a cell.
* Idle culling, a pod restart or a crash kills the round. Nothing is lost: run
  `bash tools/server/start.sh` again. It skips finished steps and resumes the current
  phase (finished runs are kept; only the runs that were in progress start over).
* Optional: `bash tools/server/install_autostart.sh` makes the Jupyter server rerun
  `start.sh` each time it starts (it adds a marked block to
  `~/.jupyter/jupyter_server_config.py`; nothing happens once the round is DONE).
  `bash tools/server/uninstall_autostart.sh` removes it.

## Stopping

    bash tools/server/stop.sh

stops the whole round (the orchestrator and every run it started), waits up to 60 s,
then forces it, and writes a fresh `bundle-latest.tar.gz`. `start.sh` resumes later.
`RETRY_FAILED=1 bash tools/server/start.sh` also reruns steps that failed or were skipped.

## What to bring back

When the round ends, `../gomea-results/server-run/DONE` exists and `run.log` says
`DONE. Bring back ...`. Bring back

    ../gomea-results/server-run/bundle-final-<UTC time>.tar.gz

If it stopped early, bring `../gomea-results/server-run/bundle-latest.tar.gz` (rewritten
every hour and by `stop.sh`). The bundle holds the logs, `events.jsonl`, the scoreboard
history, every run summary and manifest, the reports, the per-run GOMEA event logs and the
environment; `round-summary.md` inside it is the overview. Download it from the Jupyter
file browser (right click > Download) or with `scp`.

## Workers

`python3 tools/server/resources.py` prints the decision. W = min(usable CPUs - 1,
(memory - reserve) / 3 GB), at least 1. Usable CPUs are the container's CPU quota if it has
one, else the physical cores (no hyper-threads). Memory is the container limit, else the
machine's. The reserve is max(2 GB, 10 %). `MAX_WORKERS=N bash tools/server/start.sh`
caps W (the cap is decided again at each phase start).

## Troubleshooting

* **1ai and 1aj skipped**: the GP-GOMEA build failed. Rerun set-up with
  `REBUILD_GPGOMEA=1 bash tools/server/setup.sh` (or `FORCE_CONDA=1` to skip the system
  compiler), then `RETRY_FAILED=1 bash tools/server/start.sh`, which reruns
  failed and skipped steps (done steps are never rerun). If the round is already DONE,
  add `--force`.
* **Reproducibility mismatch** (S0, `server-run/repro.json`, flagged in `run.log` and the
  summary): the pod's floating point (CPU, maths library) gives different results from
  the local machine for the same seeds. The round goes on and all its contrasts are
  still valid, because they compare runs on the same machine; only comparisons between
  local and server numbers become invalid. `server-run/gpgomea-build.json`
  (`smoke_matches_local`) is the same check for the C++ build.
* **pytest failures** (S0, `server-run/pytest.log`): flagged, not fatal. Report them.
* **"already running"**: an orchestrator is alive; use `status.sh`, or `stop.sh` first.
* **Out of disk**: stop, free space, start again.
* **A phase failed**: `phase_run.sh` was retried once; its log is
  `../gomea-results/<phase>.log`. The round continued with the next step.

## Smoke test

`bash tools/server/start.sh --plan smoke` runs a short version (pytest on
`tests/server` only, the mux11 reproducibility run, one tiny phase in
`configs/gomea/server-smoke/`, and a step that is skipped on purpose) with its own state in
`../gomea-results/server-run-smoke/`, so it never marks real steps done. The mux11 run
takes about 15 to 20 minutes, the rest about two. `status.sh --plan smoke` and
`stop.sh --plan smoke` work the same way. Before its phase it sets the `commit` of
`configs/gomea/server-smoke/bool.json` to the current HEAD (the Boolean runner's code
check), which leaves that file modified in the working tree; `git checkout` it afterwards.

## Round 2

The second round (`SERVER-ROUND2-PREREGISTRATION.md`: the knob cap, the linkage ablation
at that cap, and the generation-0 ranking) has its own plan, `tools/server/plan2.json`:
S0 (full pytest and the round-1 reproducibility check, now with the knob-cap code, which
must reproduce round 1 exactly), then 1ak-pilot, 1ak, 1al and 1am, one phase at a time.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`; safe to rerun on a pod that ran round 1).
2. Start it:

       bash tools/server/start.sh --plan round2

   (`PLAN=round2 bash tools/server/start.sh` or `--plan tools/server/plan2.json` work
   too). `status.sh --plan round2` and `stop.sh --plan round2` work the same way. A
   round-1 autostart block does not start round 2: run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round2`.
3. After 1ak-pilot, `tools/gomea/set_knob_cap.py` picks K per problem (the largest of 8, 16
   and 32 whose median generations per deme is at least 3, else 8) and writes it into
   every knob-cap arm of `configs/gomea/phase1ak`, `phase1al` and `phase1am` (this leaves
   those files modified in the working tree; that is expected). 1al and 1am do not start
   without that choice.

Outputs, all under `../gomea-results/`:

* `server-run2/`: the round's state and logs (`run.log`, `events.jsonl`, `state.json`,
  the scoreboards, `round-summary.md`, `bundle-latest.tar.gz` and at the end
  `bundle-final-<UTC time>.tar.gz`, which is what to bring back). Round 1's
  `server-run/` is not touched.
* `phase1ak-pilot/knob-cap.json`: the pilot's numbers and the chosen K.
* `phase1ak-pilot/`, `phase1ak/`, `phase1al/`, `phase1am/`: the regression runs and
  each phase's `report-stage{1,2}/`; the Boolean runs are in `1/phase1a?-*-bool-*/`.
* `phase1ak-pilot.log`, `phase1ak.log`, ...: each phase's log.

Smoke test: `bash tools/server/start.sh --plan smoke2` runs a miniature of the same plan
(pytest on `tests/server` and the round-2 tool tests, the mux11 reproducibility run, then
the pilot, `set_knob_cap.py` and the three phases in `configs/gomea/server-smoke2-*` on
mux6, Con and EnH with tiny budgets) with its state in `../gomea-results/server-run2-smoke/`.
It sets the `commit` of the smoke Boolean manifests to HEAD and writes K into the smoke
configs, so `git checkout configs/gomea/server-smoke2-*` afterwards.

## Round 3

The third round (`SERVER-ROUND3-PREREGISTRATION.md`: constant tuning at equal evaluations
and at equal CPU time, size control with linear scaling, the uniform start, and five more
datasets) has its own plan, `tools/server/plan3.json`: S0 (full pytest and the round-1
reproducibility check, which must still match: with `--const-opt none` the new code gives
byte-identical output), then 1an, 1ao, 1ap, 1aq and 1ar, one phase at a time.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`). Set-up now also builds WiR and WiW (committed) and
   downloads Dow and Tow from github.com/grantdick/gpzgd at a pinned commit (their data has
   no licence and is not committed; see `data/gomea-sr/README.md`). It checks every file
   against its SHA-256. If a download or a check fails, set-up prints `SETUP FAILED` and
   names the dataset; phase 1ar then does not start (its step is skipped with that reason).
2. Start it:

       bash tools/server/start.sh --plan round3

   `status.sh --plan round3` and `stop.sh --plan round3` work the same way. An autostart
   block of an earlier round does not start round 3: run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round3`.
3. 1an runs as the pilot of the 1ao step. After it, `tools/gomea/set_const_opt.py` picks
   the tuning arm (LM or MUT, the lower median test NMSE pooled over the three datasets'
   medians, from 1an's final results) and writes `--const-opt lm` or `mutate` into the
   GOMEA-LS-TUNE arms of `configs/gomea/phase1ao` and `phase1ar` (this leaves those files
   modified in the working tree; that is expected). 1ao and 1ar do not start without that
   choice.

Outputs, all under `../gomea-results/`:

* `server-run3/`: the round's state and logs, as in round 2; bring back
  `bundle-final-<UTC time>.tar.gz`.
* `phase1an/const-opt-choice.json`: the per-dataset and pooled medians and the choice.
* `phase1an/` to `phase1ar/`: the runs (each `summary.json` has `const_opt` for a tuning
  arm: LM calls and statuses, tuning evaluations and their share of the run, mutation
  attempts and acceptances, polish accepted or not) and each phase's `report-stage{1,2}/`.
* `phase1an.log`, ...: each phase's log.

Cost note (measured locally on three Con programs of 212-265 nodes, 45-69 constants, 516
training rows): one LM call used 140-712 evaluations and 10-71 s of CPU. So in 1an the LM
arm spends most of its 10,000 evaluations on LM; this is the pre-registered cost
accounting.

Smoke test: `bash tools/server/start.sh --plan smoke3` runs a miniature of the same plan
(pytest on `tests/server` and the round-3 tests, the mux11 reproducibility run, then 1an
and the choice, 1ao, 1ap, 1aq and 1ar in `configs/gomea/server-smoke3-*`, 2 splits, 400
evaluations or 20 s per run) with its state in `../gomea-results/server-run3-smoke/`. It
writes the choice into the smoke configs, so `git checkout configs/gomea/server-smoke3-*`
afterwards.

## Round 4

The fourth round (`SERVER-ROUND4-PREREGISTRATION.md`: knob caps with and without the 250-evaluation
deme cap) has its own plan, `tools/server/plan4.json`: S0 (full pytest and the round-1
reproducibility check), then phase 1as. There is no pilot and no value to set during the run.
The search code does not change.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`).
2. Start the round:

       bash tools/server/start.sh --plan round4

   `status.sh --plan round4` and `stop.sh --plan round4` work the same way. An autostart block
   of an earlier round does not start round 4. Run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round4`.
3. Stage 1 cannot give a primary verdict: with 10 pairs and 20 contrasts, the lowest Holm p is
   0.039. So a large effect gets "extend", and `phase_run.sh` runs the extension of that problem.
   This is expected.

Arms and files:

* `configs/gomea/phase1as/`: `spec.json` (20 primary and 8 secondary contrasts), `bool.json` (mux11,
  seeds 3001-3010), `bool-ext-mux11.json` (seeds 3011-3020), `sr.json` (Con, EnH, Air, splits 1-10)
  and `sr-ext-{Con,EnH,Air}.json` (splits 11-20).
* All manifests hold all seven arms. The extensions also need K16-C250, because the secondary
  contrasts use it. A problem that extends costs 70 more runs.
* Time budgets: regression 1,000 s, wall cap 2,400 s. mux11 3,600 s, wall cap 5,400 s.

Outputs, all under `../gomea-results/`:

* `server-run4/`: the state and logs of the round, as in round 2. Bring back
  `bundle-final-<UTC time>.tar.gz`.
* `phase1as/phase1as-sr*/` and `1/phase1as-bool*/`: the runs, and `phase1as/report-stage{1,2}/`:
  the reports.
* `phase1as.log`, `phase1as-bench.log`, `phase1as-sr.log`: the logs of the phase.

Regression summaries have no `deme_count`. The report now has these values (see "Round 5"). The method by hand is: Take the number
of demes, the generations per deme and how demes end from the `end` events in each run's
`events.jsonl`. The bundle holds these files. A deme that starts when the time is spent has 0
evaluations and ends with `time_limit`. Do not count it as a real deme.

Time check (local, one run each, `--max-time 60`, nice 19, on a loaded machine; not a result):

| Run | Wall s | Evaluations | Demes (with evaluations) | Genes per deme | Generations per deme | Deme ends |
|---|---|---|---|---|---|---|
| Con NOCAP | 61.9 | 1,616 | 2 | 20, 162 | 7, 2 | converged, time limit |
| Con K64 | 60.9 | 1,418 | 2 | 20, 128 | 9, 1 | stagnation, time limit |
| Con K8 | 60.8 | 1,491 | 3 | 20, 135, 162 | 13, 13, 3 | stagnation, stagnation, time limit |
| mux11 NOCAP | 94.1 | 3,011 | 3 | 88, 425, 952 | 11, 6, 1 | stagnation, converged, time limit |
| mux11 K64 | 79.9 | 3,785 | 3 | 88, 317, 637 | 10, 11, 2 | stagnation, stagnation, time limit |

The time limit stops a deme before the next evaluation, so a deme does not overshoot. A mux11 run
overshoots because the program spends time outside the deme. In these two mux11 runs, 20 to 34 s
were over the limit. The wall caps allow for more.

Smoke test: `bash tools/server/start.sh --plan smoke4` runs a miniature of the same plan (pytest on
`tests/server` and the knob-cap tests, the mux11 reproducibility run, then 1as in
`configs/gomea/server-smoke4-1as`: mux6, Con, EnH and Air, 2 seeds or splits, 15 s per run) with its
state in `../gomea-results/server-run4-smoke/`. With 2 pairs no problem extends, so the smoke plan does
not run the `*-ext-*` manifests. The smoke Boolean manifests name the same `commit` as the real ones
(see "Round 5"), and the smoke plan does not change them.

## Round 5 and the chain from round 4

The fifth round (`SERVER-ROUND5-PREREGISTRATION.md`: a validation guard for LM, a weaker size
penalty, LM with a size penalty) has its own plan, `tools/server/plan5.json`: S0, then phases 1at,
1au and 1av. Its state is in `../gomea-results/server-run5/`. There is no pilot and no value to set
during the run.

`plan4.json` has `"then": "round5"`, so one command runs both rounds, one after the other:

    bash tools/server/start.sh --plan round4

How the chain works:

* When round 4 ends, it writes its own `round-summary.md`, `bundle-final-<UTC time>.tar.gz` and
  `DONE`. Then it logs the `chain_start` event and starts `start.sh --plan round5` in a new session.
* Each round reports separately: `server-run4/` and `server-run5/` each have their own summary and
  final bundle. A bundle holds only the files of its round. Bring back both.
* Round 4 goes on to round 5 when it ends with a failed or skipped step. `run.log` of round 4 reports
  those steps, and so does its `round-summary.md`.
* `stop.sh` with either plan name stops the running orchestrator of the chain. A stopped round does
  not start the next round. It writes `server-run4/chain.stop`, and the next `start.sh` removes it.
* Never two at once: each orchestrator of the chain holds `server-run4/chain.lock`, whichever round
  it runs. `start.sh` refuses, and a direct `orchestrate.py` exits with 3.
* `RETRY_FAILED` and `REPRO_ONLY` apply only to the plan that the command names. They do not pass to
  the next plan.

Commands, with the state of the chain:

| Command | Round 4 not DONE | Round 4 DONE, round 5 not DONE | Both DONE |
|---|---|---|---|
| `start.sh --plan round4` | starts or resumes round 4 | starts or resumes round 5 | says where both bundles are |
| `start.sh --plan round4 --force` | as above, without the next round | reruns the end of round 4 only | reruns the end of round 4 only |
| `status.sh --plan round4` | all, and the `chain:` section | all, and the `chain:` section | all, and the `chain:` section |
| `stop.sh --plan round4` | stops round 4 | stops round 5 | nothing to stop |

The `chain:` section of `status.sh` gives the state of each plan and where its bundle is.

`start.sh --plan round5` also works when round 4 is not DONE and not running. It prints no error,
and it does not run round 4. Use it only to skip round 4.

**Autostart.** Run `uninstall_autostart.sh`, then `install_autostart.sh --plan round4`. The block
returns only when both rounds are DONE. In all other cases it runs `start.sh --plan round4`, so a pod
restart during round 4 resumes round 4, and a pod restart during round 5 resumes round 5. As in earlier
rounds, the block also resumes a round that `stop.sh` stopped. To prevent this, run
`uninstall_autostart.sh` after `stop.sh`.

**Time.** In the step records and in `round-summary.md`, `duration_s` is now the active time: the sum
over the orchestrator sessions, without the idle time between them. `elapsed_s` is the wall clock
since the step first started. A kill loses at most one poll interval (60 s) of active time.

**Report.** `aggregate.csv` now has, for each arm, `median_train_test_gap`,
`median_generations_per_deme` and `deme_end_share_<reason>`. For regression, `median_deme_count` now
has a value. All of them count only demes with evaluations (`ledger.requests > 0`), from each
run's `events.jsonl`. The note in "Round 4" about taking these values by hand no longer applies. The
arm LM-VAL also has `median_lm_guard_checked`, `median_lm_guard_rejected` and
`median_lm_guard_rejected_share` (rejected / checked).

Arms and files:

* `configs/gomea/phase1at/`: GOMEA-LS, LM, LM-VAL on WiR, Con, EnH, Air, Yac, Tow, 1,000 s, wall cap
  2,400 s.
* `configs/gomea/phase1au/`: GOMEA-LS, PEN25, PEN125, PEN625 on Con, EnH, Air, 10,000 evaluations, wall
  cap 36,000 s (as 1ap). Primary metric: `program_size`.
* `configs/gomea/phase1av/`: GOMEA-LS, LM, LM-PEN5, LM-PEN25 on Con, EnH, Air, 1,000 s, wall cap
  2,400 s. Primary metric: `program_size`.
* Each has `spec.json`, `sr.json` (splits 1-10) and `sr-ext-<dataset>.json` (splits 11-20, all arms).
* The phases have `datasets:` requirements. `setup.sh` builds Yac and WiR from `data/gomea-sr/`, and it
  downloads Tow. If a dataset is missing, the step is skipped, and the log gives the reason. So run
  `setup.sh` after `git pull`.

Smoke test: `bash tools/server/start.sh --plan smoke4` runs the miniature of round 4, then
`smoke5` by itself (`configs/gomea/server-smoke5-*`: 2 splits, 15 s or 400 evaluations per run). The
state is in `server-run4-smoke/` and `server-run5-smoke/`.

**Boolean manifests of round 4.** The guard changed `pymoses`, so the `commit` of the Boolean manifests
of 1as and of its smoke test moved to the commit with the guard. If `pymoses`, `benchmarks` or
`pyproject.toml` change again, `check_code_commit` refuses these runs until the `commit` moves too. The name of the result folder holds a hash of the manifest,
so runs under an older `commit` are not reused: `1/phase1as-bool-<new hash>/` starts empty.


## Round 6

The sixth round (`SERVER-ROUND6-PREREGISTRATION.md`: LM-VAL with no per-deme cap, and LM-VAL with
PEN125) has its own plan, `tools/server/plan6.json`: S0, then phase 1aw. Its state is in
`../gomea-results/server-run6/`. There is no pilot and no value to set during the run.

    bash tools/server/start.sh --plan round6

Round 6 starts by itself. No earlier plan has `"then": "round6"`.

Arms and files:

* `configs/gomea/phase1aw/`: NOCAP, LMVAL, LMVAL-PEN125, LMVAL-C250 on WiR, Con, EnH, Air, Yac, Tow,
  1,000 s, wall cap 2,400 s. Only LMVAL-C250 has `--gomea-max-evals-per-deme 250`. Its argv is the
  LM-VAL argv of 1at.
* `spec.json` has 12 primary contrasts with two metrics: `test_nmse` for LMVAL vs NOCAP and
  `program_size` for LMVAL-PEN125 vs LMVAL. Each contrast names its metric (the fifth element).
  Holm covers all 12.
* A dataset extends (`sr-ext-<dataset>.json`, splits 11-20, all four arms) when one of its two
  primary contrasts is "extend" at stage 1. Then both contrasts of that dataset have 20 pairs at
  stage 2. Only a contrast that was "extend" at stage 1 takes its verdict from stage 2.
* The runner gives the validation split (`--const-opt-val`) to each arm with
  `--const-opt-guard val`, so to LMVAL, LMVAL-PEN125 and LMVAL-C250.
* The phase has a `datasets:` requirement. Run `setup.sh` after `git pull`.

Smoke test: `bash tools/server/start.sh --plan smoke6` (`configs/gomea/server-smoke6-1aw`: 2 splits,
15 s per run). The state is in `server-run6-smoke/`.

## Round 7 and the chain from round 6

The seventh round (`SERVER-ROUND7-PREREGISTRATION.md`: the confirmation round on splits 21-30) has
its own plan, `tools/server/plan7.json`: S0, then the choice of FINAL, then phase 1ax. Its state is in
`../gomea-results/server-run7/`. `plan6.json` has `"then": "round7"`, so one command runs both
rounds, one after the other, without a pause:

    bash tools/server/start.sh --plan round6

The chain works as in "Round 5 and the chain from round 4": two summaries, two final bundles, one
chain lock (`server-run6/chain.lock`). Bring back both bundles.

**The choice of FINAL.** The step `choose-final` (step type `choice`) runs `tools/gomea/set_final.py`:

1. It reads the final 1aw report: `report-stage2` if a primary contrast was "extend" at stage 1,
   else `report-stage1`. Only an "extend" contrast takes its verdict from stage 2.
2. Rule 2 of 1aw passes if LMVAL-PEN125 vs LMVAL on `program_size` is "A better" on at least 4 of
   6 datasets, and the median test NMSE of LMVAL-PEN125 / LMVAL is at most 1.10 on each dataset.
3. FINAL gets the exact argv of the LMVAL-PEN125 arm of 1aw if rule 2 passes, else of the LMVAL arm.
4. It writes `server-run7/phase1ax-config/`: `sr.json` (the resolved manifest), `spec.json` (a copy)
   and `final-choice.json` (the rule, the numbers and the choice). The orchestrator logs
   `final-choice.json` as the `choice` event, and `round-summary.md` has a "Choice" section.

The choice never writes a tracked file. `configs/gomea/phase1ax/` has only `template.json` (FINAL has
`"argv": null`) and `spec.json`. The phase step has `"config_dir": "{state}/phase1ax-config"`, and the
orchestrator gives it to `phase_run.sh` as `PHASE_CONFIG_DIR`. So the tree stays clean, and no later S0
test fails because of the choice (round 3 had this problem with `set_const_opt.py`). If the choice
fails, the step is failed, and phase 1ax is skipped with the reason in the log. A second run of
`set_final.py` with another result refuses to replace `sr.json` (use `--force` only by hand).

Arms and files:

* `configs/gomea/phase1ax/`: FINAL, NOCAP, HC-LS on Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac,
  splits 21-30, 1,000 s, wall cap 2,400 s, 3,072 MB (as 1ar). 300 runs. No arm has a per-deme cap:
  HC-LS is `-a hc`, not `hc-capped`. There is no extension.
* `spec.json` has `pooled_primary`: FINAL vs HC-LS and FINAL vs NOCAP, each pooled over the 100
  (dataset, split) pairs. `phase_report.py` tests r = log(NMSE_A / NMSE_B) with a paired Wilcoxon
  signed-rank test and Holm across the 2. A failed run gets the largest rank against it. Verdict:
  "A better" if Holm p < 0.025 and exp(median r) <= 0.9, "A worse" if exp(median r) >= 1/0.9. The rows
  have problem `pooled`, `median_log_ratio` and `ratio`.
* The 20 secondary contrasts (each contrast on each dataset) have Holm across the 20 but no verdict
  (`"secondary_verdicts": false`).
* The phase has a `datasets:` requirement for all 10 datasets. `setup.sh` downloads Dow and Tow.

Smoke test: `bash tools/server/start.sh --plan smoke6` runs the miniature of round 6, then `smoke7`
(`configs/gomea/server-smoke7-1ax`: splits 21-22, 15 s per run; the choice reads the smoke 1aw
report). The state is in `server-run6-smoke/` and `server-run7-smoke/`.
