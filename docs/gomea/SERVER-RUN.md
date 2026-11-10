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

The dynamic population of round 6 changed `pymoses` again, so the `commit` of these manifests moved
to `0e8dfa2`, the code commit of round 6. The 1ay and 1az manifests name the same commit.


## Round 6 and the chain 6, 7, 8

The sixth round (`SERVER-ROUND6-PREREGISTRATION.md`: is the fixed GOMEA population of 16 a
bottleneck?) has its own plan, `tools/server/plan6.json`: S0, then phase 1ay, then the choice of
the population, then phase 1az. Its state is in `../gomea-results/server-run6/`. Round 6 runs
first. `plan6.json` has `"then": "round7"` and `plan7.json` has `"then": "round8"`, so one command
runs the three rounds, one after the other, without a pause:

    bash tools/server/start.sh --plan round6

The chain works as in "Round 5 and the chain from round 4": three summaries, three final bundles,
one chain lock (`server-run6/chain.lock`). Bring back the three bundles. Rounds 7 and 8 read the
choice of round 6 from `server-run6/`, so do not start round 7 alone before round 6 is DONE.

**New code: `--gomea-pop-size dynamic`.** N for each deme is the number of new instances that
classic hill climbing takes at distance 1: `hill_climbing.estimate_neighborhood(1, fields)`, then
`n_new_instances(1, ...)` with `--hc-fraction-of-nn F` (default 2.0), `--hc-max-nn-evals`
(default 20,000), the classic minimum of 100 and the evaluations left of the deme. The population
builder draws N programs as before. With `--gomea-pop-size 16` (or no option) the output is
byte-identical to the code before (`tests/gomea/test_pop_size_identity.py`). The telemetry has
`deme_pop_size` and `deme_budget` on `invocation_start`, and `pop_size_dynamic` on `end`, only with
`dynamic`. `aggregate.csv` has `median_pop_size`, `min_pop_size` and `max_pop_size` for each arm
(from `pop_size_effective` of each deme with evaluations).

**The choice of the population.** The step `choose-population` runs
`tools/gomea/set_population.py choose`:

1. It reads the final 1ay report, with the stage rule of `set_final.py`.
2. POP (rounds 7 and 8): a fraction qualifies if its LT-B arm is "A better" than N16-LTB on at
   least 2 of the 3 problems and "A worse" on none. Of several, the best median rank. If none
   qualifies, POP is N = 16.
3. DYN (1az and the control arms): the fraction with the best median rank, qualified or not.
4. Median rank: on each problem, the 3 dynamic LT-B arms ranked by `median_metric` of the
   report's `aggregate.csv` (equal medians: the smaller F first). Then the median of the 3 ranks.
   Equal median ranks go to the smaller F.
5. The control population is DYN, or N = 16 if POP equals DYN.
6. It writes `server-run6/phase1az-config/`: the resolved 1az manifests, `spec.json` and
   `population-choice.json` (the rules, the numbers, POP, DYN and the control). The orchestrator
   logs it as the `choice` event.

**Templates.** A phase whose argv depends on the choice has `template.json` (resolved to `sr.json`)
and `template-<name>.json` (resolved to `<name>.json`). In an argv, the token `{POP}`, `{DYN}` or
`{CONTROL}` becomes `--gomea-pop-size 16` or `--gomea-pop-size dynamic --hc-fraction-of-nn F`.
The choices write only into the state directories, never into tracked files.

Arms and files:

* `configs/gomea/phase1ay/`: N16, DYN2 (F = 2.0), DYN025 (F = 0.25) and DYN005 (F = 0.05), each with
  `--gomea-fos lt-b` (`-LTB`) and `rt` (`-RT`). mux11 (`bool.json`, seeds 3001-3010, 3,600 s, wall cap
  5,400 s), Con and EnH (`sr.json`, splits 1-10, 1,000 s, wall cap 2,400 s). Extensions:
  `bool-ext-mux11.json` (seeds 3011-3020) and `sr-ext-{Con,EnH}.json` (splits 11-20). No arm has an
  evaluation cap. 240 runs (+ up to 240).
* `spec.json` of 1ay: 9 primary contrasts (each dynamic LT-B arm against N16-LTB, on each problem)
  and 12 secondary (LT-B against RT at each population, on each problem). With 10 pairs, Holm over
  9 can give a verdict at stage 1.
* `configs/gomea/phase1az/`: templates only. NOCAP-DYN, K8-DYN, K16-DYN, K32-DYN, K64-DYN (knob flags
  as in 1as) and NOCAP-N16, all LT-B. mux11, Con, EnH, Air, as in 1as. 16 primary and 4 secondary
  contrasts. With 16 contrasts, stage 1 gives no primary verdict, so a large effect extends (as in
  round 4). 240 runs (+ up to 240).
* The Boolean manifests of 1ay and 1az name the code commit of the dynamic population. If
  `pymoses`, `benchmarks` or `pyproject.toml` change again, move that `commit` (and the one of the
  round 4 manifests) too.

**Memory.** At F = 2.0, N on mux11 can be several thousand. The smoke test runs mux11 at F = 2.0
for 60 s. The real runs have the usual limit of 3,072 MB per run.

Smoke test: `bash tools/server/start.sh --plan smoke6` runs the miniature of round 6
(`configs/gomea/server-smoke6-1ay` and `server-smoke6-1az`: 2 splits or seeds, 15 s per regression
run, 60 s per mux11 run in 1ay and 20 s in 1az), then `smoke7` and `smoke8`. The state is in
`server-run6-smoke/`, `server-run7-smoke/` and `server-run8-smoke/`. With 2 pairs nothing is
"A better", so the smoke choice gives POP = N16.

## Round 7

The seventh round (`SERVER-ROUND7-PREREGISTRATION.md`: LM-VAL with no per-deme cap, and LM-VAL with
PEN125) has its own plan, `tools/server/plan7.json`: S0, then the step `apply-population`, then
phase 1aw. Its state is in `../gomea-results/server-run7/`. Round 6 starts it (see "Round 6 and the
chain 6, 7, 8"). When round 7 ends, it starts round 8 (see "Round 8 and the chain from round 7").

**The population (round 6).** Every GOMEA arm of 1aw uses POP. The step `apply-population` runs
`tools/gomea/set_population.py apply` with `server-run6/phase1az-config/population-choice.json` and
writes the resolved manifests into `server-run7/phase1aw-config/` (with
`population-applied.json`). The phase step reads them from there (`config_dir`). If round 6 has no
choice, the step fails and 1aw is skipped.

Arms and files:

* `configs/gomea/phase1aw/`: templates only (`template.json`, `template-sr-ext-<dataset>.json`).
  NOCAP, LMVAL, LMVAL-PEN125, LMVAL-C250 and the control arm LMVAL-DYN (LMVAL with the control
  population) on WiR, Con, EnH, Air, Yac, Tow, 1,000 s, wall cap 2,400 s. Only LMVAL-C250 has
  `--gomea-max-evals-per-deme 250`. With POP = N16 its argv is the LM-VAL argv of 1at. 300 runs
  (+ up to 300).
* `spec.json` has 12 primary contrasts with two metrics: `test_nmse` for LMVAL vs NOCAP and
  `program_size` for LMVAL-PEN125 vs LMVAL. Each contrast names its metric (the fifth element).
  Holm covers all 12. 18 secondary contrasts: the 12 of the pre-registration and test NMSE,
  LMVAL-DYN vs LMVAL, on each dataset.
* A dataset extends (`sr-ext-<dataset>.json`, splits 11-20, all five arms) when one of its two
  primary contrasts is "extend" at stage 1. Then both contrasts of that dataset have 20 pairs at
  stage 2. Only a contrast that was "extend" at stage 1 takes its verdict from stage 2.
* The runner gives the validation split (`--const-opt-val`) to each arm with
  `--const-opt-guard val`, so to LMVAL, LMVAL-PEN125 and LMVAL-C250.
* The phase has a `datasets:` requirement. Run `setup.sh` after `git pull`.

Smoke test: `configs/gomea/server-smoke7-1aw` (2 splits, 15 s per run), as part of
`bash tools/server/start.sh --plan smoke6`, because it reads the smoke choice of round 6. The state
is in `server-run7-smoke/`.

## Round 8 and the chain from round 7

The eighth round (`SERVER-ROUND8-PREREGISTRATION.md`: the confirmation round on splits 21-30) has
its own plan, `tools/server/plan8.json`: S0, then the choice of FINAL, then phase 1ax. Its state is in
`../gomea-results/server-run8/`. `plan7.json` has `"then": "round8"`, so one command runs both
rounds, one after the other, without a pause:

    bash tools/server/start.sh --plan round7

The chain works as in "Round 5 and the chain from round 4": two summaries, two final bundles, one
chain lock (`server-run7/chain.lock`). Bring back both bundles.

**The choice of FINAL.** The step `choose-final` (step type `choice`) runs `tools/gomea/set_final.py`:

1. It reads the final 1aw report: `report-stage2` if a primary contrast was "extend" at stage 1,
   else `report-stage1`. Only an "extend" contrast takes its verdict from stage 2.
2. Rule 2 of 1aw passes if LMVAL-PEN125 vs LMVAL on `program_size` is "A better" on at least 4 of
   6 datasets, and the median test NMSE of LMVAL-PEN125 / LMVAL is at most 1.10 on each dataset.
3. FINAL gets the exact argv of the LMVAL-PEN125 arm of 1aw if rule 2 passes, else of the LMVAL arm.
   It reads that argv from the resolved 1aw of round 7 (`--source-dir
   server-run7/phase1aw-config`), so FINAL has POP.
4. With `--population server-run6/phase1az-config/population-choice.json` it gives NOCAP the
   population POP, and the control arm FINAL-DYN the argv of FINAL with the control population.
5. It writes `server-run8/phase1ax-config/`: `sr.json` (the resolved manifest), `spec.json` (a copy)
   and `final-choice.json` (the rule, the numbers and the choice). The orchestrator logs
   `final-choice.json` as the `choice` event, and `round-summary.md` has a "Choice" section.

The choice never writes a tracked file. `configs/gomea/phase1ax/` has only `template.json` (FINAL has
`"argv": null`) and `spec.json`. The phase step has `"config_dir": "{state}/phase1ax-config"`, and the
orchestrator gives it to `phase_run.sh` as `PHASE_CONFIG_DIR`. So the tree stays clean, and no later S0
test fails because of the choice (round 3 had this problem with `set_const_opt.py`). If the choice
fails, the step is failed, and phase 1ax is skipped with the reason in the log. A second run of
`set_final.py` with another result refuses to replace `sr.json` (use `--force` only by hand).

Arms and files:

* `configs/gomea/phase1ax/`: FINAL, NOCAP, HC-LS and FINAL-DYN on Air, Bos, Con, Dow, EnC, EnH, Tow,
  WiR, WiW, Yac, splits 21-30, 1,000 s, wall cap 2,400 s, 3,072 MB (as 1ar). 400 runs. No arm has a
  per-deme cap: HC-LS is `-a hc`, not `hc-capped`, and keeps its flags. There is no extension.
* `spec.json` has `pooled_primary`: FINAL vs HC-LS and FINAL vs NOCAP, each pooled over the 100
  (dataset, split) pairs. `phase_report.py` tests r = log(NMSE_A / NMSE_B) with a paired Wilcoxon
  signed-rank test and Holm across the 2. A failed run gets the largest rank against it. Verdict:
  "A better" if Holm p < 0.025 and exp(median r) <= 0.9, "A worse" if exp(median r) >= 1/0.9. The rows
  have problem `pooled`, `median_log_ratio` and `ratio`.
* The 20 secondary contrasts (each contrast on each dataset) have Holm across the 20 but no verdict
  (`"secondary_verdicts": false`).
* `pooled_secondary`: FINAL-DYN vs FINAL, pooled over the 100 pairs with the test of the primary
  contrasts, its own Holm family (of 1) and a verdict. It changes no rule and not the choice of
  FINAL.
* The phase has a `datasets:` requirement for all 10 datasets. `setup.sh` downloads Dow and Tow.

Smoke test: `bash tools/server/start.sh --plan smoke6` runs the miniatures of rounds 6 and 7, then
`smoke8` (`configs/gomea/server-smoke8-1ax`: splits 21-22, 15 s per run; the choice reads the smoke
1aw report and the smoke population of round 6). The state is in `server-run8-smoke/`.

## Round 9

The ninth round (`SERVER-ROUND9-PREREGISTRATION.md`: the new base on other datasets, seeding with
the new base, and the cap of 250 with LM-VAL) has its own plan, `tools/server/plan9.json`: S0, then
phases 1ba, 1bb and 1bc. Its state is in `../gomea-results/server-run9/`. Start it with:

    bash tools/server/start.sh --plan round9

The three phases are independent. There is no choice step, and no plan chains to or from round 9.
Round 9 never overlaps with another round. One summary and one final bundle. Bring back the bundle.

**Budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so `-m 20000`) and no
`--max-time`. The wall cap of 36,000 s per run is only a guard against a hung process. CPU time is
measured, not capped.

**The new base (BASE).** `-a gomea --gomea-fos lt-b --gomea-pop-size dynamic --hc-fraction-of-nn 0.05
--gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced
--linear-scaling 1 --const-opt lm --const-opt-guard val --gomea-accept-complexity-ratio 0.125
--gomea-deme-store accepted`. BASE-N16 has `--gomea-pop-size 16` in place of the two dynamic flags.
The manifests are resolved: no `{POP}` token and no template.

Arms and files:

* `configs/gomea/phase1ba/`: BASE and BASE-N16 on Air, Bos, Dow, EnC, Tow, WiR, WiW, Yac, splits
  1-10. `sr.json` has the 7 datasets without Dow. `sr-Dow.json` has Dow alone, so that its
  `mem_limit_mb` can change without a change to the other manifests. Extensions:
  `sr-ext-<dataset>.json` (splits 11-20), with `sr-ext-Dow.json` for Dow. 160 runs (+ up to 160).
* `spec.json` of 1ba: 8 primary contrasts (test NMSE, BASE vs BASE-N16, on each dataset) and 24
  secondary (program size, train-test gap and CPU, on each dataset).
* `configs/gomea/phase1bb/`: BASE and BASE-S on Con, EnH, Air, Yac, splits 1-10, extensions
  `sr-ext-<dataset>.json` (splits 11-20). BASE-S has the seed block of GOMEA-S in 1ab (`share` 0.2,
  so 4,000 of the 20,000 evaluations go to GP-GOMEA, height 3, k 5, functions `+_-_*`, the empty
  program kept). 80 runs (+ up to 80). The phase needs the GP-GOMEA build of `setup.sh`
  (requirement `gpgomea`). Without it, 1bb is skipped with the reason in the log.
* `spec.json` of 1bb: 4 primary contrasts (test NMSE, BASE-S vs BASE) and 8 secondary (program size
  and CPU; a seeded run's `cpu_s` includes GP-GOMEA's CPU).
* `configs/gomea/phase1bc/`: BASE, BASE-C250, N16 and N16-C250 (the C250 arms add
  `--gomea-max-evals-per-deme 250`). `sr.json`: Con, Tow, Yac, WiR on splits 11-20.
  `sr-BosEnC.json`: Bos and EnC on splits 1-10. This is data that 1aw did not use. There is no
  extension. 240 runs.
* `spec.json` of 1bc: 6 primary contrasts (test NMSE, BASE-C250 vs BASE) and 18 secondary in two
  Holm families. Family A (6): test NMSE, N16-C250 vs N16. Family B (12): program size and LM
  calls, BASE-C250 vs BASE. The key `"secondary_families": [["n16c250-vs-n16"]]` sets this. It has
  `"extend": false`, so a primary contrast with Holm p in [0.025, 0.25) is "tie", not "extend".

**New report metrics.** `phase_report.py` has three new contrast metrics for the secondary
contrasts: `train_test_gap` (test NMSE minus training NMSE), `cpu_s` (the run's CPU) and `lm_calls`
(`const_opt.lm_calls`). Lower is better for each, and a failed run is infinitely bad.
The spec key `secondary_families` (a list of lists of secondary contrast names) makes each list a
Holm family of its own. The other secondary contrasts form one more family. In `contrasts.csv`
the column `holm_family` shows the family. Without the key, all secondary contrasts are one
family, as before.

**Memory and workers.** The Dow memory check found that the use is real (the reduced-candidate
cache). So the memory limits (`mem_limit_mb`) are:

| Manifests | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs) |
|---|---|---|
| `phase1ba/sr-Dow.json`, `phase1ba/sr-ext-Dow.json` | 8,192 | floor((65,536 - 6,553) / 8,192) = 7 |
| every other round 9 manifest (1ba, 1bb, 1bc) | 5,120 | min(12 - 1, floor(58,983 / 5,120) = 11) = 11 |

W comes from `resources.py`, as before. Two plan keys on a phase step set its `PER_RUN_MB`:

* `per_run_mb` (5120 on the steps of 1ba, 1bb and 1bc): the `PER_RUN_MB` of the decision for the
  phase. It replaces `PER_RUN_MB` from the environment.
* `manifest_per_run_mb` (1ba only: `{"sr-Dow.json": 8192, "sr-ext-Dow.json": 8192}`): the
  orchestrator makes one more decision for each named manifest. It gives these W values to
  `phase_run.sh` in `MANIFEST_WORKERS="sr-Dow.json=7 sr-ext-Dow.json=7"`.

`phase_run.sh` runs one manifest at a time, and only the Dow manifests have Dow runs. So never
more than the Dow W of Dow runs run at the same time, and Dow runs never run beside other runs.
The 1ba report and its Holm family still hold all 8 datasets, and the Dow extension works as
before. `events.jsonl` has one `workers_decision` for the phase and one for each Dow manifest
(field `manifest`, with `per_run_mb` 8192). `phase_attempt` has `manifest_workers`. Plans 2 to 8
have neither key, so their W comes from the environment, as before. If you change a limit,
change the manifests and the plan key together (`tests/server/test_plan9.py` checks this).
`MAX_WORKERS` caps each decision.

**Size and time.** Stage 1 is 480 runs, about 16 hours at W = 11 (the 20 Dow runs at W = 7), plus S0. With all extensions, up to
720 runs.

Smoke test: `bash tools/server/start.sh --plan smoke9` runs the limited S0 and the miniatures
`configs/gomea/server-smoke9-1ba`, `-1bb` and `-1bc`. They have the same manifests, arms and flags,
2 splits each, a wall cap of 300 s and no extension manifests. The budget is `-m 300`, and
`-m 1000` in 1bb, because the first GP-GOMEA generation alone costs about 600 evaluations. The smoke
includes the Dow run and the seeded arm. The state is in `server-run9-smoke/`. The smoke plan has the
same `per_run_mb` and `manifest_per_run_mb` keys, so the smoke Dow run also gets its own W.

## Round 10

The tenth round (`SERVER-ROUND10-PREREGISTRATION.md`, roadmap task T1.2: the new base against MOSES
hill climbing with LM-VAL and against plain GOMEA) has its own plan, `tools/server/plan10.json`:
S0, then phase 1bd. Its state is in `../gomea-results/server-run10/`. Start it with:

    bash tools/server/start.sh --plan round10

There is no choice step, no extension and no adoption rule: the round only reports the results.
No plan chains to or from round 10. Round 10 never overlaps with another round. Bring back the
final bundle.

**Budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so `-m 20000`) and no
`--max-time`. LM passes, guard passes and GP-GOMEA seed evaluations count. The wall cap of 36,000 s
per run is only a guard against a hung process.

**Arms** (`configs/gomea/phase1bd/`, the same five arms in each manifest):

| Arm | Flags |
|---|---|
| BASE | the round 9 base (as in 1bb) |
| BASE-S | BASE with the seed block of 1bb (`share` 0.2, height 3, k 5, functions `+_-_*`) |
| NOCAP | `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1` (round 8) |
| HC-LM | `-a hc --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val` |
| HC-LM-W | HC-LM + `--hc-widen-search 1` |

* HC-LM and HC-LM-W use the new LM for hill climbing (`pymoses/gomea/hc_lm.py`): at the end of each
  HC deme, LM on the deme's best program, with the same code and rule as GOMEA. The runner gives
  them `--const-opt-val` (the scaled validation split) and `--gomea-log RUN_DIR/events.jsonl`, as
  for the GOMEA arms. Each HC deme writes one "end" event, so the summary has `const_opt` (LM
  calls, guard rejections) and the report has the deme count and the steps per deme.
* Datasets: all 10 (Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac), splits 1-10. 500 runs.
* The phase needs the GP-GOMEA build of `setup.sh` (requirement `gpgomea`, for BASE-S). Without it,
  1bd is skipped with the reason in the log.

**Report** (`spec.json` of 1bd; the pooled test of round 8: paired Wilcoxon on
r = log(NMSE_A / NMSE_B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (3): BASE vs HC-LM, BASE vs HC-LM-W, BASE vs NOCAP
  (`pooled_primary`).
* Seeding control, pooled, its own Holm family (4): BASE-S vs BASE, HC-LM, HC-LM-W and NOCAP.
* Widen search, pooled, its own Holm family (1): HC-LM-W vs HC-LM.
* Secondary, pooled, one Holm family (10): program size and CPU for BASE vs HC-LM, BASE vs HC-LM-W,
  BASE vs NOCAP, BASE-S vs BASE and HC-LM-W vs HC-LM.
* Per dataset, descriptive (40): contrasts 1 to 4 on each dataset, Holm across the 40, no verdict
  (`secondary` with `"secondary_verdicts": false`).

The control, widen and size/CPU contrasts are `pooled_secondary` rows. The new spec key
`pooled_secondary_families` (`[[the 4 control names], ["hclmw-vs-hclm"]]`) makes each inner list a
Holm family of its own; the other pooled secondary contrasts form one more family. In
`contrasts.csv` the column `holm_family` shows the family of each pooled secondary row
(`pooled_secondary/<names>` or `pooled_secondary`). Without the key nothing changes for other
phases. The verdicts describe the data only.

**Memory and workers.** Round 9 showed that N = 16 needs much memory at 20,000 evaluations. Each of
Dow and Tow has a manifest of its own:

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bd/sr-Dow.json` | 16,384 | floor(58,983 / 16,384) = 3 |
| `phase1bd/sr-Tow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bd/sr.json` (the 8 other datasets) | 5,120 | min(11, 11) = 11 |

The plan step has `per_run_mb` 5120 and `manifest_per_run_mb`
`{"sr-Dow.json": 16384, "sr-Tow.json": 8192}` (the round 9 mechanism). `events.jsonl` has one
`workers_decision` for the phase and one for each of the two manifests. `phase_run.sh` runs one
manifest at a time (`sr.json`, then `sr-Dow.json`, then `sr-Tow.json`). If a NOCAP run still fails
on memory, the report counts it as a BASE win; the findings then give contrast 3 with and without
the failed pairs.

**Size and time.** 500 runs, about 12-17 hours at W = 11, plus about 1.3 hours for S0. The 50 Dow
runs at W = 3 add about 1-2 hours.

Smoke test: `bash tools/server/start.sh --plan smoke10` runs the limited S0 (with the new tests)
and `configs/gomea/server-smoke10-1bd`: the same three manifests, arms and flags, splits 1 and 2,
`-m 1000` (the first GP-GOMEA generation alone costs about 600 evaluations), a wall cap of 300 s
and no extension. 100 runs. The state is in `server-run10-smoke/`. The smoke plan has the same
`per_run_mb` and `manifest_per_run_mb` keys, so the smoke Dow and Tow runs also get their own W.
