# A. Literature: what GP-GOMEA authors measured about LT vs RT vs UNI

Date: 2026-10-30. Written in ASD-STE100 style.
Local copies of the papers and extracted text: `papers/` in this folder.
Labels: **[M]** = the paper measured it (table or figure). **[C]** = the authors claim it, without a direct measurement. **[I]** = my inference.

Terms:
- LT = linkage tree, learned from mutual information (MI) by UPGMA.
- LT-MI = LT with plain MI. LT-MIb = LT with the initialization-bias-corrected MI (ECJ 2021). Our "LT-B" is this one.
- RT = random tree (same hierarchical shape as LT, random similarity values).
- UNI = univariate FOS (one locus per subset).
- IMS = interleaved multistart scheme (population doubles for each new run).
- NMSE = 100 x MSE / var(y). Lower is better.

## 0. Short answer

- The belief "linkage learning is one of the biggest sources of gains in GP-GOMEA" is **not supported** in its strong form.
- What the papers measure:
  1. UNI is much worse than any tree-shaped FOS (RT or LT) [M, Koch 2026]. So **multi-node mixing with a hierarchical FOS plus GOM** gives a large gain.
  2. **Learned** linkage (LT) over **random** linkage (RT) gives a **small, conditional** gain. It shows only with large populations (about 1,000 to 2,000 or more, "thousands"), many evaluations (10^5 to 10^7), a small template, and a small symbol alphabet.
  3. With plain MI, LT is about equal to RT (Boolean problems in 2017; SR in 2021). With small populations (100 to 500) LT is equal to RT or worse (feature construction 2020; GOMGE 2018).
  4. The newest paper by the same group (Koch, Alderliesten, Bosman 2026) finds that the existing MI measures beat RT only by P(A>B) = 0.57 to 0.59. A **fixed, non-learned** template-distance FOS beats all learned measures. The authors say "the random baseline is not clearly outperformed for all settings".
- Our MOSES result (LT-B never beats RT or UNI at pop 16 to 64, 250 to 1,000 evaluations per deme) is **consistent** with the literature. Our setting is 1 to 3 orders of magnitude below every setting where the authors measured an LT advantage.

## 1. Paper by paper

### 1.1 Virgolin, Alderliesten, Witteveen, Bosman. GECCO 2017. "Scalable GP by GOM and IEBL"
Source: https://ir.cwi.nl/pub/26549/GECC1117.pdf (DOI 10.1145/3071178.3071287). Results: Sec. 5 and Fig. 3.

Settings [M]:
- FOS compared: U, LT (plain MI, no bias correction), RT.
- Problems: Order, Trap-3, Trap-4, KU (Koza-style positional tunable problem) = "artificial" problems with locus-based structure. Boolean: Comparator, Even Parity, Majority, Multiplexer (3, 6, 11 bits). Function set AND, OR, NAND, NOR.
- Population and height: IMS. The first run starts with population 2. Each new run doubles the population. Tree height starts at ceil(log_r(n inputs)) and grows by 1 every 2 runs. Run is terminated if a larger run has better mean fitness, or the population converges.
- Budget: run until the optimum is found. Limits 24 h, 500 GB. 30 runs. Only successful runs are reported.
- Measure: evaluations, time and solution size to reach the optimum (Fig. 3, log-log scalability plots).
- No statistical tests. No tables. Only plots with mean, min and max.

Results:
- [M+C] Artificial problems: "GP-GOMEA_LT is generally the best performing algorithm". On Order no marked difference between FOS. On Trap-4 and KU, LT scales better. On Trap-3, SGP_param is slightly better than LT.
- [M+C] Boolean problems: "the difference between the linkage models LT and RT ... is far less pronounced, suggesting that the LT is not capable of modeling key linkage information". Authors give the reason: high redundancy of the representation; "locus-based dependence is not the most important source of problem structure".
- [I] Fig. 3: the U, LT and RT curves overlap on all four Boolean problems.
- Caveat: Trap and KU are built so that fixed loci carry the structure. They are not SR.

### 1.2 Virgolin, Alderliesten, Witteveen, Bosman. Evolutionary Computation 29(2):211-237, 2021. "Improving model-based GP for SR of small expressions"
Source: https://arxiv.org/abs/1904.02050 (v4). DOI 10.1162/evco_a_00278. Also Chapter 3 of the thesis.

Why they introduced the bias correction (Sec. 5, Fig. 3, Fig. 4) [M+C]:
- GP initialization (Half-and-Half) does not give a uniform symbol distribution per locus (for example, functions never appear at leaves). So plain MI already shows "linkage" at generation 0, before any selection. Fig. 3 shows this with a population of 10^6, h = 2.
- Their hypothesis: if the MI at generation 0 is made equal to the identity matrix, "truly salient patterns will have a bigger chance of emerging".
- Method: scale each entropy by a coefficient from generation 0: beta_i = 1/H_1(i), beta_ij = 2/H_1(i,j). Then MIb(i,j) = beta_i H(i) + beta_j H(j) - beta_ij H(i,j).
- Fig. 4: with population 10, the corrected MI at generation 2 ranges from -0.787 to 1.032 (noisy). With population 10^6 it ranges from -0.018 to 0.989. So the correction itself is very noisy at small populations [M].

Experiment LT-MIb vs LT-MI vs RT (Sec. 5.3, Table 3) settings [M]:
- Population 2,000, fixed (no IMS). Height h = 4 (31 nodes). 20 generations. Up to 2l-2 = 60 evaluations per solution per generation, so up to about 2.4 million evaluations per run. Authors: "a compromise between having enough samples for linkage to be learned" and literature values.
- Functions {+, -, x, AQ}. Terminals: features only, no ERC. Linear scaling on. 10 UCI-type datasets. 50/25/25 split. 30 runs. Wilcoxon signed-rank with Bonferroni; Friedman test first.

Table 3, median NMSE of 30 runs (train | test) [M]:

| Data | LT-MIb train | LT-MI train | RT train | LT-MIb test | LT-MI test | RT test |
|---|---|---|---|---|---|---|
| Air | 29.9 | 31.2 | 32.7 | 31.8 | 34.8 | 34.0 |
| Bos | 15.4 | 15.4 | 17.5 | 24.0 | 23.0 | 22.5 |
| Con | 17.5 | 18.5 | 19.0 | 18.7 | 19.6 | 20.1 |
| Dow | 20.9 | 20.3 | 24.0 | 22.6 | 21.1 | 26.0 |
| EnC | 8.42 | 9.68 | 9.09 | 9.18 | 10.7 | 10.3 |
| EnH | 6.24 | 6.44 | 6.40 | 6.50 | 7.10 | 6.70 |
| Tow | 12.5 | 12.5 | 13.1 | 13.0 | 12.8 | 13.2 |
| WiR | 60.3 | 60.9 | 61.2 | 62.5 | 63.0 | 63.1 |
| WiW | 68.1 | 68.4 | 68.7 | 69.1 | 69.7 | 69.5 |
| Yac | 0.34 | 0.37 | 0.36 | 0.58 | 0.62 | 0.62 |

(Numbers read from the PDF text. The significance markers did not extract cleanly; I use the authors' text for significance.)

- [M, text] LT-MIb is significantly better than LT-MI on 8/10 datasets (train) and 7/10 (test). LT-MIb is better than RT on 10/10 (train) and 9/10 (test). The exception is Bos test, where RT has the lower median (22.5 vs 24.0).
- [M, text] "The LT-MI is comparable with the RT for these problems." So **plain-MI linkage learning did not beat random linkage on SR** at population 2,000.
- [I] Effect size of LT-MIb over RT: train NMSE is 1% to 13% lower (relative). Median about 6%. Largest: Dow -13%, Bos -12%, Air -9%. Smallest: WiW -0.9%, WiR -1.5%. Absolute differences are small (for example 68.1 vs 68.7).
- [M] Fig. 5 (Yac): LT-MIb reaches lower error faster. Authors say other datasets are similar (not shown).
- [M] Table 4 (Yac, 100 runs x 5 splits): share of best solutions that have a duplicate across runs: LT 39% vs RT 12% (pop 2,000); LT 46% vs RT 17% (pop 4,000). LT makes runs converge to the same model more often.

ERC experiment (Sec. 6.1, Tables 5 and 6) [M]: same setup plus an ERC terminal. Strategies all-const, no-const, bin-const (gamma = 100 bins).
- Train median NMSE, bin-const LT vs RT: Air 27.5 vs 31.4; Bos 15.0 vs 17.6; Con 17.0 vs 18.5; Dow 20.7 vs 24.5; EnC 5.76 vs 6.44; EnH 2.80 vs 4.10; Tow 12.3 vs 13.2; WiR 60.2 vs 61.2; WiW 68.0 vs 68.5; Yac 0.34 vs 0.38.
- [M, text] "In terms of training error, the RT is always outperformed by the use of the LT, no matter the strategy." Test: "less comparisons are significant".
- [M] all-const (every constant is its own symbol) is worse than bin-const on 6/10 datasets. [C] Too many symbols "hide the true salient patterns". [M] RT is not always the fastest; LT converges faster.

Benchmark (Sec. 8, Tables 7 to 9): no LT-vs-RT comparison. GP-GOMEA (LT) vs GP-Trad with a free tree shape and the same node limit: GP-GOMEA wins 87, loses 65, draws 88 of 240 comparisons [M]. Authors: "GP-GOMEA was not found to be markedly better than GP-Trad_l" [C]. Against GP-Trad with the same height template, GP-GOMEA is "almost always significantly better" [M].

Authors' own limitation [C]: "we did not study how linkage learning behaves in GP for SR in depth ... it would be interesting to assess when linkage learning is beneficial, and when it is superfluous or harmful."

### 1.3 Virgolin, Alderliesten, Bosman. Swarm and Evolutionary Computation 2020. "On explaining ML models by evolving crucial and compact features"
Source: https://arxiv.org/abs/1907.02260 (DOI 10.1016/j.swevo.2019.100640). Also thesis Chapter 5.

- Main experiments use population 100 and 10,000 evaluations. [M, preliminary, not shown] "we found the adoption of the LT to not be superior to the adoption of the RT under these circumstances". So the main study uses RT.
- [C] "the LT requires sufficiently large population sizes to be accurate and beneficial (e.g., several thousand solutions in GP for symbolic regression)".
- Fig. 10 [M], NB classifier, h = 4, 30 runs, x-axis = population / evaluation budget:
  - 500 / 50K: LT = RT on both datasets (no significant difference).
  - 2,500 / 250K: LT significantly better on Image Segmentation (19 features); train F1 about 0.777 vs 0.768. On Madelon (500 features), RT is significantly better on test.
  - Largest point (label reads "125K/1250K"; the 1:100 ratio of the other points suggests 12.5K population): LT significantly better on Image Segmentation (train F1 about 0.792 vs 0.783). On Madelon LT = RT (train) and LT worse (test).
- Thesis conclusion (p. 213) [C]: with population 100 and 10,000 evaluations, LT "was not helpful, or even harmful, compared to using random variation". Recommendation: use LT only "if the population size can be of the order of thousands or more ... Otherwise, the RT should be preferred."

### 1.4 Virgolin PhD thesis, TU Delft 2020
Source: https://ir.cwi.nl/pub/29655/29655.pdf (DOI 10.4233/uuid:03641b5f-f8f6-4ff9-be7f-11948f6d3cc7). Chapters 2, 3, 5 reprint 1.1, 1.2, 1.3.

Conditions stated in the concluding chapter (Sec. 9.2.1) [C]:
- "the number of samples (in GP, the population size) needs to be large with respect to the number of random variables (the maximum number of instructions allowed) and their sample space (the number of possible instructions) for linkage learning to be advantageous."
- GP-GOMEA "works particularly well when programs ... are small, as linkage learning needs to deal with a limited number of possibilities (genotype symbols)."
- Sec. 9.2.2 [C]: the key enabler of small, good programs is that GOM changes many unconnected nodes at once; a FOS such as the LT is "a necessary factor, to guide GOM to attempt a limited number of variation steps". The LT "is not necessarily the optimal one".
- On GOMGE (below): LT "was not performing particularly better than ... a pre-specified linkage model, nor than using random noise as linkage", possibly because of "a relatively small population size (500)".

### 1.5 Koch, Alderliesten, Bosman. arXiv 2602.02311, Feb 2026. "Introns and Templates Matter: Rethinking Linkage in GP-GOMEA"
Source: https://arxiv.org/abs/2602.02311. Same research group (CWI). Most direct re-test of LT vs RT vs UNI.

Settings (Table 2) [M]:
- Experiment 1: IMS, start population 64, 10 generations before the next (doubled) run starts. Budget 10^7 evaluations. Template height 5 (31 nodes) and 7 (127 nodes). Linear scaling on and off. Functions {+, -, x, /, sin}. ERC with 25 bins. R^2 objective. 5 SR datasets: Airfoil (5 features), Bike Sharing (12), Concrete (8), Dow Chemical (57), Tower (25). 30 runs per setting (5 folds x 6 seeds). Shared seeds.
- Experiment 2: fixed population 1,024, 20 generations, height 5, LS on.
- Measures compared: Random (RT), MI, MI_adjusted (= the 2021 bias correction = our LT-B), MI_masked (new: introns get one "masked" symbol), Node (new: fixed similarity from distance in the template, no learning, LT rebuilt each generation with random tie breaks), Node (static), #CS, Univariate (UNI).

Results [M]:
- Fig. 10, P(row measure gets higher training R^2 than column measure), stratified bootstrap 95% CI:
  - Random > Univariate: 0.92 [0.90, 0.94].
  - MI_adjusted > Random: 0.57 [0.54, 0.60].
  - MI > Random: 0.59 [0.56, 0.62].
  - MI > MI_adjusted: 0.51 [0.48, 0.54] (no difference).
  - MI_masked > Random: 0.73 [0.71, 0.76].
  - Node > Random: 0.85 [0.82, 0.87]. Node > MI_masked: 0.66.
  - "there is a roughly 15% chance of Random outperforming Node after the full budget."
- Fig. 9 (Friedman + Nemenyi, all settings): mean ranks about Univariate 7.6, Random 5.2, MI and MI_adjusted about 4.7 (not different from each other), then Node (static), #CS, MI_masked, Node (best, about 2.4).
- [C] "considering only the already existing methods, the random baseline is not clearly outperformed for all settings, suggesting that previously proposed linkage measures are suboptimal for GP."
- [M+C] The bias adjustment gives an early advantage (Fig. 4), but "continued mitigation can be harmful"; plain MI often ends better. At height 5 with LS (the 2021 setting) the 2021 result (MIb > MI) holds; elsewhere it does not.
- [M+C] Larger templates widen the gaps between measures. Linear scaling shrinks them.
- [M+C] Appendix E: with a larger operator set (10 operators), "the mutual information based measures tend to perform worse compared to the random measure".
- [M] Fig. 7 / Appendix D: all MI variants, after some generations, learn a structure that resembles template distance (parent-child closeness). [C] "the solution representation is more important than variable values when it comes to the dependency strengths."
- [C] Appendix A: UNI "clearly performs worse than all other methods indicating that linkage matters for GP". Note: "linkage" here means "multi-node subsets", not "learned linkage".

### 1.6 Medvet, Bartoli, De Lorenzo, Tarlao. PPSN 2018. "GOMGE: GOM on Grammatical Evolution" (independent group)
Source: https://medvet.inginf.units.it/publications/2018-c-mbdt-gomge/ (DOI 10.1007/978-3-319-99253-2_18).

- Settings [M]: population 500, 60 s per run, 30 runs. GE, WHGE, SGE representations. Problems: Parity-7, Nguyen-7, KLandscapes-5, Text. FOS: U, N (natural, fixed), RT, LT.
- Table 1 [M]: with WHGE, RT and LT are equal or nearly equal (for example Parity-7: 0 and 0; KLand-5: 0.29 vs 0.25; Nguyen-7: 0.18 vs 0.21). Both beat U and the baseline GE.
- [C] "the facts that LT and RT lead to the same good performance ... suggest that the improvement is related to the reiterated application of the GOM operator, rather than to the possibility of learning the linkage."

### 1.7 Bosman and Thierens, discrete LT-GOMEA / LTGA (context only)
- The LT advantage in discrete GOMEA is shown on problems with strong locus-tied structure (deceptive traps, NK, MAXCUT) and with IMS population sizing. [C, from abstracts] Thierens and Bosman GECCO 2012 "Predetermined versus learned linkage models": on intricate problems the learned LT (rebuilt each generation) was more efficient than fixed, hand-designed models. A TU Delft report (black-box GOMEA on MAX-SAT) says LTGA beats Random-Tree and Univariate GOMEA there.
- [I] These problems have positional, low-redundancy encodings. GP genotypes are highly redundant (introns, many equivalent programs). The 2017 paper itself gives this as the reason that LT did not beat RT on Boolean GP.
- I did not read the full texts of these discrete papers. Only abstracts.

### 1.8 GP-GOMEA code (github.com/marcovirgolin/GP-GOMEA, commit 7a9c690, Sep 2025)
- Default FOS = LT (`fos_type = FOSLinkageTree`). Options: LT, RT, Univariate.
- The MI bias correction is always on ("we always apply the correction", `src/GOMEA/GOMEAFOS.cpp` line 429). Constants are binned to 100 values.
- Python wrapper defaults: `gomfos='LT'`, `ims='5_1'`, `popsize=500`. C++ IMS subgeneration default 10.
- README makes no claim about the size of the linkage-learning contribution. It cites SRBench for overall performance.

## 2. Answers to the key questions

Q1. Is LT's advantage large, small, problem dependent or absent?
- Boolean / redundant GP problems: absent or very small (2017) [M+C].
- Artificial locus-structured GP problems (Trap-4, KU): present, grows with problem size (2017) [M, plots only].
- SR, plain MI: absent ("comparable with RT", 2021) [M+C]. Small edge in 2026 (P = 0.59 vs RT) [M].
- SR, bias-corrected MI: small but consistent at pop 2,000, 20 generations (2021): 1% to 13% lower train NMSE than RT, significant on 10/10 train, 9/10 test [M]. In 2026 with 10^7 evaluations: P = 0.57 vs RT, and not better than plain MI [M].
- Feature construction: none at pop 100 to 500; small at 2,500 to 12,500 with 19 features; none or negative with 500 features [M].
- What is large: UNI vs any tree FOS (P = 0.92 for RT over UNI) [M], and a template-informed FOS vs RT (P = 0.85) [M].

Q2. Why the bias correction? Plain MI sees non-uniform initial symbol distributions as linkage at generation 0 [M, Fig. 3]. They reported plain-MI LT as "comparable with the RT" [M]. The 2026 paper shows the correction helps early but can hurt later, and finds the intron noise (not the initialization bias as such) to be the cause of the MI bias [M+C].

Q3. Conditions the authors give [C unless marked]:
- Population "of the order of thousands or more" (2020), or IMS with exponential sizing. The 2021 tests used 2,000. The 2026 IMS reached large populations under 10^7 evaluations. Tests at 100 and 500 found no LT gain [M].
- Population large relative to the number of loci and to the alphabet size (thesis 9.2.1). Few features help (Image Segmentation 19 features yes; Madelon 500 no [M]). Many distinct constants hurt; binning helps [M].
- Enough generations / evaluations: 20 generations x 60 subsets x 2,000 = about 2.4M evaluations (2021); 10^7 (2026). Gains show more at larger budgets (Fig. 10 of 2020) [M].
- A fixed template (homologous positions) is needed for MI to mean anything. Smaller templates make MI easier; larger templates widen the gaps between measures (2026) [M].
- Introns add noise to MI; mask them (2026) [M].
- Diversity: the bias correction needs a large initial population to be accurate (Fig. 4, 2021) [M]. Linear scaling reduces the differences between FOS [M, 2026].

Q4. Independent replications or critiques:
- GOMGE (Medvet et al. 2018, independent group): LT = RT; gain comes from GOM, not from learning linkage [M+C].
- Koch et al. 2026 (same group, later): existing LT measures barely beat RT; a non-learned template FOS is best [M+C].
- I found no other independent ablation of LT vs RT in GP-GOMEA. SRBench (La Cava et al. 2021) uses GP-GOMEA with LT but does not ablate the FOS.

## 3. Comparison with our MOSES setting [I]

| Setting | Population | Evaluations per run | LT vs RT result |
|---|---|---|---|
| GOMGE 2018 | 500 | 60 s | LT = RT |
| Feature constr. 2020 | 100 | 10,000 | LT not better, or worse |
| Feature constr. 2020 | 500 | 50,000 | LT = RT |
| Feature constr. 2020 | 2,500 | 250,000 | LT > RT (19 features only) |
| ECJ 2021 | 2,000 | about 2.4M | LT-MIb > RT (small); LT-MI = RT |
| Koch 2026 | IMS from 64 | 10^7 | MI(adj) > RT, P = 0.57 to 0.59 |
| pyMoses demes | 16 to 64 | 250 to 1,000 | LT-B never > RT/UNI |
| pyMoses frozen maj5 | 1,024 | (see our notes) | LT > RT once |

- Our deme budgets are 50 to 10,000 times smaller than the budgets where an LT advantage was measured. Our populations are 8 to 100 times smaller.
- MI from 16 to 64 samples over an alphabet of several symbols per locus is dominated by small-sample bias. The 2021 Fig. 4 shows the corrected MI is very noisy even at population 10.
- One difference to note: in GP-GOMEA, UNI is clearly the worst FOS (P = 0.92 for RT over UNI). In our MOSES runs, UNI was about equal to RT. This suggests that our knob space has weaker multi-locus coupling than a GP tree template, or that our budgets are too small to show it. This is an open item to check, not a finding.
- Our single LT win (frozen maj5, pop 1,024) is in the regime the authors describe (population of about a thousand, small alphabet, Boolean).

## 4. Sources
- GECCO 2017: https://ir.cwi.nl/pub/26549/GECC1117.pdf
- ECJ 2021: https://arxiv.org/abs/1904.02050
- Feature construction 2020: https://arxiv.org/abs/1907.02260
- Thesis 2020: https://ir.cwi.nl/pub/29655/29655.pdf
- Koch et al. 2026: https://arxiv.org/abs/2602.02311
- GOMGE 2018: https://medvet.inginf.units.it/publications/2018-c-mbdt-gomge/
- Code: https://github.com/marcovirgolin/GP-GOMEA
- Also read, no LT/RT ablation: Harrison et al. 2025 (arXiv 2507.03777), Schlender et al. 2024 (arXiv 2402.09854), Modular GP-GOMEA 2025 (arXiv 2505.01262; populations 1,024 to 32,768).

## 5. Not verified
- Exact significance markers per cell in ECJ 2021 Tables 3 and 5. The PDF text extraction mixed them up. I used the authors' summary sentences.
- Fig. 9 ranks in Koch 2026 are read from the plot (approximate).
- Fig. 10 x-axis label "125K/1250K" in the 2020 paper (12.5K is my reading from the ratio).
- Full texts of the discrete LTGA / LT-GOMEA papers (only abstracts were read).
