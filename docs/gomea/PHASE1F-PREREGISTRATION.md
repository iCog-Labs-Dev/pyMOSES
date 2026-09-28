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

## Parts C–D

To be registered here before their runs. The planned outline, which is not yet binding:
* **C:** pyMoses GOMEA with lt-b, lt-mi and rt against the Table 3 pattern and levels.
* **D:** pyMoses GOMEA vs pyMoses hill climbing, with 10 paired splits.

**Why 10 splits suffice for parts B–D.** With 10 pairs, the exact two-sided Wilcoxon p can be as
small as 0.002, below the strictest Holm threshold over 10 datasets (0.005). So a
consistent effect is detectable. An inconsistent one reads as a tie, not as a false win.
