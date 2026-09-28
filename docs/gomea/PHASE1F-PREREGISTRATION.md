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

## Parts B–D

To be registered here before their runs. The planned outline, which is not yet binding:

* **B:** stand-alone GP-GOMEA vs classic C++ MOSES on the 10 datasets, with 10 paired
  splits, test NMSE, Holm correction across datasets.
* **C:** pyMoses GOMEA with lt-b, lt-mi and rt against the Table 3 pattern and levels.
* **D:** pyMoses GOMEA vs pyMoses hill climbing, with 10 paired splits.

**Why 10 splits suffice for B–D.** With 10 pairs, the exact two-sided Wilcoxon p can be as
small as 0.002, below the strictest Holm threshold over 10 datasets (0.005). So a
consistent effect is detectable. An inconsistent one reads as a tie, not as a false win.
