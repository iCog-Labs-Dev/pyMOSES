# Phase 1f pre-registration: the GP-GOMEA paper's own benchmarks

Written 2026-09-28. Each part is registered here **before any of its runs**, and the commit
that adds or extends a part is its timestamp. Part A is registered first. Parts B–D are
added before they run, once the MOSES regression setup and the constant genes
(`--gomea-contin copy`) are ready.

**Requested by the user** (`APPROVALS.md` #11). Use the benchmark problems of the
GP-GOMEA paper, not the MOSES demo problems, to check four things:

1. whether pyMoses's embedded GOMEA behaves consistently with the literature, i.e. no
   regression from our design;
2. whether the authors' C++ GP-GOMEA reproduces the published results at small scale;
3. whether stand-alone GP-GOMEA beats classic C++ MOSES;
4. whether pyMoses GOMEA beats pyMoses hill climbing.

The user also asked to use the machine's full resources, to stay as close to the
literature as possible, and to use 10 seeds instead of 20 where that is justifiable. The
criteria below are agent-proposed.

## The literature

The paper is Virgolin, Alderliesten, Witteveen and Bosman (2021), *Improving Model-based
Genetic Programming for Symbolic Regression of Small Expressions*, Evolutionary
Computation 29(2); it is [P1] in the handoff, and the port's source. The references to
"Table" below are to that paper. It uses only symbolic regression on 10 real-world
datasets.

**Settings (Table 1):**
* functions {+, −, ×, ÷AQ}, where the analytic quotient is x1/√(1+x2²);
* linearly-scaled MSE fitness;
* trees of height 4 (31 nodes);
* a 50/25/25 train/validation/test split, and 30 repetitions;
* reported as the median NMSE = 100·MSE/var(y).

**Datasets.** We use the same 10 datasets, with counts verified against Table 2
(`../gomea-results/phase1f/datasets/PROVENANCE.md`).
* **Splits:** split *s* (1–30) is `random.Random("<abbr>-s<s>")` shuffling the row
  indices, with train n−2q, validation q and test q rows (q = n // 4).
* **Tool:** `tools/gomea/sr_datasets.py`.
* **Pairing:** every configuration uses the same split *s* for repetition *s*, so runs pair
  by split, as in the paper.

## Part A: replicate Table 3 with the authors' C++ GP-GOMEA (confirmatory)

Table 3 is the paper's core linkage experiment. Our port follows it: the LT–MIb̃ linkage
tree with bias correction, which is our `lt-b`.

**Code.**
* GP-GOMEA@750220c, built as in Phase 1b (`gpgomea`, SHA-256 `bfbfbf47…d1d4`).
* **LT–MI** needs a one-line patch, because the bias correction is hard-wired on. The
  plain-MI branch is `if (false)`. It is built by `tools/gomea/build_gpgomea_ltmi.sh`
  (`gpgomea-ltmi`, SHA-256 `7e3314b6…e0e4`, diff in `ltmi.diff`).

**Configurations.**
* **Common flags:** `--prob symbreg --linearscaling --functions +_-_*_aq --gomea
  --popsize 2000 --initmaxtreeheight 4 --generations 20`, with no ERC. The FOS is the LT
  by default, or `--gomfos RT`.
* **Data:** trained on `train` (50 %) and scored on `test` (25 %). Table 3 does not use
  the validation set.
* **Arms:** `LT-MIb` (stock binary), `LT-MI` (patched binary) and `RT`.
* **Seeds:** the GP-GOMEA seed is the split number.
* **Manifest:** `configs/gomea/phase1f/partA-table3.json`, run with
  `tools/gomea/gpgomea_sr_runner.py`.

**Repetitions: all 30**, as in the paper, on all 10 datasets (900 runs). A calibration
estimated about 16 CPU-hours for the stock arms. Reducing to 10 is unnecessary here.

**Known deviations from the paper.**
* The splits and seeds are ours; the paper does not state them.
* The hardware differs, but the budget is generations, so it does not matter.
* The var(y) used for NMSE is the population variance of the scored set's targets; the
  paper does not state which.

**Criteria (pre-registered).**
1. **Levels.** For each configuration, dataset and metric (train and test NMSE), take the
   ratio of our median to the published Table 3 median. A configuration's levels
   **replicate** on a metric if the ratio lies within [0.8, 1.25] on at least 8 of the 10
   datasets.
2. **Pattern.** Per dataset, run a paired Wilcoxon signed-rank test (30 pairs) of LT–MIb̃
   vs LT–MI and of LT–MIb̃ vs RT, on train and on test NMSE. Significance is p < 0.05/2
   (Bonferroni over the two comparisons, as the paper does per metric).

   The paper reports that LT–MIb̃ is significantly better than LT–MI on 8/10 datasets
   (train) and 7/10 (test), and better than RT on 10/10 (train) and 9/10 (test). The
   pattern **replicates** for a comparison and metric if:
   * our count of datasets where LT–MIb̃ is significantly better is within ±2 of the
     paper's; and
   * no dataset shows LT–MIb̃ significantly *worse*.
3. **Verdict.** "Table 3 replicated" if every level and pattern criterion holds.
   "Partly replicated" lists which hold.

## Part B: stand-alone GP-GOMEA vs classic C++ MOSES (confirmatory)

Registered 2026-09-28, before any part B run.

**Design.** The paper benchmarks GP-GOMEA against other algorithms in its Section 8 (Tables
7–9) under an **equal wall-time budget of 1,000 s per run**. The paper's reason: GP-GOMEA
does many evaluations per generation, and linkage learning costs time without evaluations.
Part B uses the same design. It also matches the paper on data use: training on the
training data, choosing the returned model on validation data, and scoring on the test
set.

* **GP-GOMEA** (`GPG-IMS4`): the Table 8 setting with IMS g = 4 (ℓ = 31).
  * **Flags:** `--linearscaling --functions +_-_*_aq --erc --gomea --initmaxtreeheight 4
    --ims 4_1 --popsize 500 --time 1000 --generations -1`. Only the stock bin-const
    constant handling exists.
  * **Base population:** the paper does not give the IMS base population; 500 is the
    authors' Python default.
  * **Data:** trained on `trainval` (75 %), with `--validation` holding out 25 % of all
    rows. GP-GOMEA draws those rows at random from the seed, so they are not our `val`
    rows. It returns the archived solution with the best validation error.
  * **Manifest:** `configs/gomea/phase1f/partB-gpgomea.json`.
* **Classic C++ MOSES** (`MOSES-CPP`, 5a5bf75): `tools/gomea/moses_sr_runner.py` with
  `--max-time 1000`.
  * **Setup:** the training target is rescaled and `-v` is set to 60·n_train, with
    `-n sin -n exp -n log`, which leaves {+, ×, ÷}. This is the setup that let MOSES run
    to budget on every table in the setup sweep; without it MOSES stalls after a few
    hundred evaluations.
  * **Data:** trained on `train` (50 %). Of its 10 printed candidates, the one with the
    best NMSE on our `val` split is chosen.
  * **Manifest:** `configs/gomea/phase1f/partB-moses.json`.
* **Differences we cannot remove.**
  * MOSES has no analytic quotient: its division invalidates a candidate that divides by
    zero.
  * MOSES has no linear scaling in training. The rescaling and the train-fitted scaling
    at scoring stand in for it.
  * MOSES has no size cap. GP-GOMEA's trees have at most 31 nodes.
  * Constants are MOSES contin knobs rather than ERCs.
* **Splits:** 1–10, paired, on all 10 datasets. 200 runs of 1,000 s on 16 workers is
  about 3.5 h.
* **Load:** part B runs with nothing else heavy on the machine, since its budget is wall
  time. Workers are not pinned to cores; the machine mixes performance and efficiency
  cores, which adds noise equally to both arms.

**Tests.**
* Per dataset: a paired Wilcoxon signed-rank test on test NMSE (10 pairs), with Holm
  correction across the 10 datasets.
* GP-GOMEA is **better** on a dataset if Holm p < 0.05 **and** its median test NMSE is at
  least 10 % lower. **Worse** is the mirror image; otherwise the dataset is a tie.
* **Overall:** "GP-GOMEA beats classic MOSES" if it is better on a majority of datasets and
  worse on none.
* **Descriptive:**
  * train NMSE and program size;
  * our GP-GOMEA test medians against the paper's Table 8 IMS g = 4 medians, a secondary
    replication check at 10 runs on faster hardware.

## Parts C and D: pyMoses GOMEA vs the literature, and vs pyMoses hill climbing

Registered 2026-09-28, before any C or D run.

**Code.** `bcccf8f` adds the approved `--gomea-contin copy`. Each MOSES constant (a
contin knob) is one gene that optimal mixing copies whole and never mutates, as the paper
treats its ERC constants. Linkage statistics bin constants as the paper's bin-const does
(γ = 100), the setting it uses from Section 6 on. A review found no issues, and default
behaviour is byte-identical.

**Runs** (`configs/gomea/phase1f/partCD-pymoses.json`, `tools/gomea/moses_sr_runner.py`).
All arms use pyMoses with the same MOSES regression setup as part B: rescaled target,
`-v 60·n_train` and `-n sin -n exp -n log`. Each is trained on `train` and returns the
printed candidate with the best `val` NMSE.

* **Budget:** **10,000 evaluations** per run. The MOSES seed is the split number.
* **Splits:** 1–10, on all 10 datasets. 400 runs, estimated at about 220 CPU-hours, or
  about 11 h on 18 workers. They start after part B, and the budget is evaluations, so
  machine load does not matter.
* **Arms:**
  * `HC`: `-a hc`, stock hill climbing.
  * `LTB`: `-a gomea --gomea-contin copy --gomea-pop-size 16 --gomea-skip-inactive 0
    --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1 --gomea-fos lt-b`. This is the
    Phase 1e winner, with the paper's bias-corrected linkage tree.
  * `LTMI` and `RT`: the same, with `--gomea-fos lt-mi` or `rt`.

**Why this budget.** pyMoses is 20–35× slower than C++ MOSES, and its GOMEA is 3–10×
slower per evaluation than its HC on these tables. A timing pilot of 1,000 evaluations on
Tow, Dow and Air took 42–58 s (HC), 114–287 s (RT) and 191–546 s (LTB). The paper's
GP-GOMEA spends about 10⁶ evaluations per run, which is far out of reach, so levels
cannot be matched. Part C therefore tests *relative* behaviour.

### Part C: is pyMoses GOMEA consistent with the literature? (regression check)

The paper's port-relevant claim is the Table 3 pattern: the bias-corrected linkage tree
beats plain MI and the random tree. Part A measures that pattern for the authors' C++ code
on our splits. Part C asks whether the same *direction* shows inside pyMoses.

* **Tests.** Per dataset, a paired Wilcoxon test (10 pairs) of LTB vs LTMI and of LTB vs
  RT, on train and on test NMSE, significant at p < 0.05/2 (Bonferroni, as in part A).
* **Verdict "consistent, no regression visible"** if both hold:
  1. LTB is **never significantly worse** than LTMI or RT on any dataset or metric; and
  2. on **train** NMSE, LTB's median is lower than LTMI's and RT's on at least 6 of 10
     datasets each. Train NMSE is the fitness the linkage models optimise.
* **Verdict "inconsistent"** if either fails. The report then says which, and whether
  part A also failed on that dataset. A pattern the C++ code does not show on our splits
  is not evidence against the port.
* **Descriptive:** pyMoses test and train medians next to part A (C++, about 10⁶
  evaluations) and Table 3. The component-level check is the existing golden parity
  suite, which ports GP-GOMEA's linkage learning and GOM against the C++ code
  (`tests/gomea/test_golden_parity.py`).

### Part D: does pyMoses GOMEA beat pyMoses hill climbing? (confirmatory)

* **Contrast:** LTB vs HC, paired by split.
* **Tests.** Per dataset, a Wilcoxon signed-rank test on **test NMSE** (10 pairs), with
  Holm correction across the 10 datasets.
* **Verdict per dataset.** **Better** if Holm p < 0.05 **and** LTB's median test NMSE is
  at least 10 % lower. **Worse** is the mirror image; otherwise a **tie**.
* **Overall:** "pyMoses GOMEA beats hill climbing on the paper's benchmarks" if it is
  better on a majority of datasets and worse on none.
* **Descriptive:** train NMSE, the other two GOMEA arms vs HC, CPU time and program size.

**Why 10 splits suffice for parts B–D.** With 10 pairs, the exact two-sided Wilcoxon p can be as
small as 0.002, below the strictest Holm threshold over 10 datasets (0.005). So a
consistent effect is detectable. An inconsistent one reads as a tie, not as a false win.

## Deviation: part B restart (recorded 2026-09-29, before any part B analysis)

The first part B launch crashed. Its MOSES runner died after 9 of its 100 runs, while
scoring `Bos` split 1.

**The cause.** MOSES's regression programs can contain `impulse(...)`. On `Bos`, the
column CHAS holds only the tokens `0`/`1`, so MOSES types it as a boolean input. The
scoring tool used pyMoses's pure contin interpreter, which rejects `impulse`, and passed
CHAS as the number 1.0 rather than as a logical value.

**The fix.** `tools/gomea/sr_eval.py` now uses the mixed interpreter, as MOSES does, and
passes columns MOSES types as boolean as logical values, applying MOSES's own
`infer_type_from_token2` to the training file. It reproduces MOSES's printed training
scores exactly on the crashing run and on a run without booleans. The runner now scores an
unscorable candidate as infinite error instead of crashing, and
`tools/gomea/watch_runs.py` raises an alert on any failure.

**What was kept and what is re-run.**
* The 9 finished MOSES runs are kept (all `Air`, which has no boolean column).
* The 13 GP-GOMEA runs that finished before the crash are kept, since both arms were
  running together.
* Three GP-GOMEA runs that finished just after the crash are re-run, along with every
  unfinished run.
* Both arms are restarted together, so they again share the machine. The protocol is
  otherwise unchanged.

Parts C and D use the same scorer and had not started.

## Deviation: a pyMoses reduct bug found during parts C and D (recorded 2026-09-29)

Run `Air__LTMI__s7` crashed with a `TypeError` in `reduce_factorize_fraction`
(`pymoses/comboreduct/reduct/contin_rules.py`), a port bug independent of GOMEA.
* **The crash:** two C++ one-argument `append_child` calls had been ported without their
  placeholder value.
* **The wrong node:** the non-childless branch moved `key_parent` where C++ moves
  `kp_parent`.

The fix matches the C++ build exactly on six trees built to reach both sub-branches, for
the rule alone and for the full `contin_reduce` at efforts 1–3. A regression test pins the
C++ output. The branch always raised before the fix, so no completed run executed it, and
completed runs are unaffected.

Runs started after the fix use the fixed code. The crashed run is re-run with the fix
after the batch finishes. Any other run that crashed the same way before the fix is
handled likewise and listed in the findings.

## Deviation: parts C and D rescoped after an out-of-memory crash (recorded 2026-09-29, before any C/D analysis)

At 22:50 the machine ran out of memory while 18 runs were in flight. The kernel killed
a pyMoses GOMEA run on Dow at about 2 GB, and the runner's pool shut down. At that point
121 runs had finished `ok`, which covers Air, Bos and Con except two runs.

Two further failures were already known:
* five Air LTMI runs had crashed on the reduct bug before its fix;
* two Dow HC runs had hit Python's recursion limit before the runner's recursion wrapper
  (`8b23deb`).

The run time had also been badly underestimated. With 18 concurrent Python processes on
22 hardware threads, a GOMEA run on Air took 1.5–2 h instead of about 15 min. At a
memory-safe concurrency, completing the registered design would take 1.5–2 more days.

**User decision:** "Drop LTMI, keep 10k". The rest proceeds as follows.
* **Budget:** 10,000 evaluations per run, unchanged.
* **Arms on the 7 remaining datasets** (Dow, EnC, EnH, Tow, WiR, WiW, Yac): HC, LTB and
  RT only. LTMI is not run there.
* **Re-runs:** the 7 failed runs above and the unfinished ones.
* **Resources:** 11 workers, a 3,072 MB address-space cap per run, and a split-by-split
  schedule so the memory-heavy runs of one dataset do not all overlap. A run that hits the
  cap is re-run with more room and fewer workers, and listed in the findings.

**Effect on the tests.**
* **Part D** (LTB vs HC) is unchanged, on all 10 datasets.
* **Part C, LTB vs RT:** unchanged, on all 10 datasets.
* **Part C, LTB vs LTMI:** rests on Air, Bos and Con only. Criterion 2 for LTMI becomes
  "LTB's train median is lower on at least 2 of those 3 datasets", the same two-thirds
  proportion as 6 of 10 would be; criterion 1 is unchanged.
