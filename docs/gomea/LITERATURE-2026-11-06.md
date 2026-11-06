# Alternatives to LT-B for pyMOSES: literature search result

Date: 2026-11-06. Plan: `/tmp/claude-1000/-home-yab-Documents-Projects-iCog-HyperonRelated-MOSES-pyMoses/a5e3b8b9-02f7-4461-90c5-5b2e86043773/scratchpad/litsearch/PLAN.md`.

## How to read this

- "OA" means the OpenAlex `cited_by_count` on 2026-11-06. The Semantic Scholar API was rate-limited (HTTP 429), so I used OpenAlex. Google Scholar counts are usually higher.
- "(nv)" marks a number that I did not verify from the paper text. I took it from an abstract, a summary or my memory.
- "Measured" means that the paper reports the number. "Claim" means that the abstract states it without a number that I saw.
- Domain tags: [Bool], [Cont] or [Both].

## Main finding

- Only one method in the literature beats LT-B directly in GP-GOMEA at no extra cost: template "node proximity" linkage (Koch, Alderliesten, Bosman, GECCO 2026; arXiv 2602.02311). This is the same paper that you cite for P(LT-B better than RT) = 0.57.
- Most "better linkage" work (DSMGA-II, empirical linkage, P3, CGOMEA) has evidence only on GA benchmarks such as traps and MAX-SAT. It has no GP or real-data evidence.
- For SR, the strongest evidence is for constant handling during the search (GP-RV-GOMEA, in-loop nonlinear least squares), not for better linkage.

## Category A: Fits pyMOSES with modest changes

1. [Both] Node-proximity (template) linkage, Koch et al. 2026 — builds the FOS from tree distance between node positions (1 − distance to the closest common ancestor, normalised), not from population statistics; in MOSES, use the knob positions in the exemplar tree — measured: 5 UCI/SR sets (Airfoil, Bike, Concrete, Dow, Tower), N 1,024 and IMS from 64, 10^7 evaluations; P(Node better than MI, bias-adjusted MI and random) ≈ 0.80; it reaches the final accuracy of the old methods at about 1/3 of the budget; zero runtime cost; does not need a large N; not tested on Boolean or at N 16 — 1 OA (2026; GECCO 2026, peer-reviewed) — arXiv 2602.02311, DOI 10.1145/3795095.3805139
2. [Cont] Constant optimisation inside the search, not only at the end (Kommenda et al. 2020; supported by Koch et al. 2024) — runs Levenberg–Marquardt on candidates during evolution; Koch 2024 shows that constant optimisation during evolution beats tuning after GP — measured in Koch 2024: tuning only after GP has a "small" effect; Kommenda 2020: GP + NLS "among the best" on a large SR suite (claim; suite size and gain (nv)); cost: LM per candidate, which Kommenda says is a large overhead (nv) — 125 OA (GPEM 2020) — DOI 10.1007/s10710-019-09371-3
3. [Both] Intron-masked mutual information, Koch et al. 2026 — computes MI as now, but gives inactive (intron) values a separate "masked" label, so only active knobs give a signal; in MOSES, mask knobs under a disabled parent or with contin value 0 — measured: same 5 sets and setup as entry 1; P(masked MI better than existing MI variants) ≈ 0.66; same cost as MI; cannot combine with the bias correction; still needs population statistics, so N 16 can still hurt it — 1 OA (2026) — arXiv 2602.02311
4. [Both] ε-lexicase selection, with down-sampling (La Cava et al. 2016; Geiger et al. 2025) — selects parents case by case, not by aggregate error; in MOSES, use it to select the exemplar from the metapopulation — measured (La Cava): better test fit than tournament and age-fitness Pareto on real and synthetic sets, "negligible" overhead (claim; dataset count (nv)); Geiger: ε-lexicase + down-sampling best for a fixed evaluation budget on many SR problems (claim) — 150 OA (GECCO 2016); 3 OA (ACM TELO 2025) — DOI 10.1145/2908812.2908898, arXiv 2407.21632. Caveat: the evidence is for population GP; the MOSES deme optimizer has no parent selection, and exemplar selection is a different use.

## Category B: Meets the criteria, but the blocker is integration into MOSES

1. [Cont] GP-RV-GOMEA, Koch et al. PPSN 2024 — optimises all constants of the population with real-valued GOMEA (Gaussian model, intron-aware updates) at the same time as the structure — measured on real SR sets with linear scaling and restarts: P(RV + intron-aware better than ERCs) = 0.77; P(better than ERCs + coefficient mutation) = 0.92 (Bayesian hierarchical test, 0.01 R² margin); it makes expressions larger — 3 OA (2024) — DOI 10.1007/978-3-031-70055-2_15. Blocker: MOSES constants are discrete 63-step contin knobs; this needs a shared real-valued gene pool for each deme.
2. [Both] Random desired operator with semantic backpropagation (Pawlak, Wieloch, Krawiec 2015), with linear scaling inside (Virgolin et al. 2019) — inverts the program to find the wanted output at a node, then inserts the best subtree from a precomputed library — measured (Pawlak): beats standard and other semantic operators on SR and Boolean benchmarks (claim; gain (nv)); Virgolin 2019: RDO + LS "best among all tested GP" on real datasets (claim, size (nv)); cost: library search per operation — 97 OA (IEEE TEVC 2015); 56 OA (GECCO 2019) — DOI 10.1109/TEVC.2014.2321259, DOI 10.1145/3321707.3321758. Blocker: needs a new knob builder that makes "replace subtree from library" knobs.
3. [Cont] eggp, e-graph duplicate filtering (de França, Kronberger GECCO 2025) — stores all visited expressions in an e-graph and blocks variation that makes an equivalent expression again — measured: simple GP becomes competitive with PySR and Operon on an SRBench subset and real sets, for small expressions, with no extra cost (claim; numbers (nv)) — 7 OA (2025) — DOI 10.1145/3712256.3726383. Blocker: MOSES varies knobs, not subtrees; MOSES reduct already removes many equivalent forms.
4. [Cont] Interaction–Transformation (ITEA), de França and Aldeia 2021 — models are linear sums of transformed feature-interaction terms; it fits coefficients by least squares — measured: competitive on SR benchmarks and SRBench (claim; rank (nv)) — 66 OA (ECJ 2021) — DOI 10.1162/evco_a_00285. Blocker: a different representation; it could be a new regression deme type.
5. [Cont] Operon, Burlacu et al. 2020 — a fast C++ GP with LM constant tuning in the loop and linear scaling — measured: among the most accurate methods in SRBench (La Cava et al. 2021) (claim from SRBench; numbers (nv)) — 123 OA (GECCO 2020 companion) — DOI 10.1145/3377929.3398099. Blocker: it is a full system; you already use its main ideas (LS, LM).
6. [Cont] Modular GP-GOMEA (2025) — evolves several small template trees that call each other as subexpressions — measured: "generally outperforms" single-template GP-GOMEA (claim; numbers (nv)) — 4 OA (arXiv 2025) — arXiv 2505.01262. Blocker: depends on the fixed GP-GOMEA template; MOSES has no template.
7. [Cont] Semantic subtree inheritance and greedy child selection for GP-GOMEA (GECCO 2024) — extra variation steps that use the semantic context of a subtree and the unused parts of the template — measured: "generally positive" impact, most with large, high-arity operator sets (claim) — 7 OA (2024) — arXiv 2402.09854, DOI 10.1145/3638529.3654118. Blocker: template-specific.

## Category C: Checked and rejected

| Approach | Cites (OA) | Failed criterion |
|---|---|---|
| DSMGA-II (Hsu, Yu, GECCO 2015) | 76 | 3: GA benchmarks only (traps, MAX-SAT, spin glass). New GOMEA beats it (Dushatskiy et al. 2023). DOI 10.1145/2739480.2754737 |
| DSMGA-II two-edge model (GECCO 2017) | 22 | 3: GA benchmarks only. DOI 10.1145/3071178.3071236 |
| Empirical linkage learning, DLED (Przewozniczek et al., TEVC 2020; GECCO 2021) | 39; 20 | 2: pairwise perturbation costs extra evaluations, about O(n²) for 180–1,600 knobs against a 10k–40k budget. 3: no GP results. DOI 10.1109/TEVC.2020.2985497 |
| Linkage learning in local search (Przewozniczek et al. 2023–2024) | 12–17 | 3: GA/combinatorial benchmarks only. DOI 10.1145/3583131.3590349 |
| P3, parameter-less population pyramid (Goldman, Punch 2014/2015) | 111; 32 | 3: GA benchmarks only. DOI 10.1145/2576768.2598350 |
| Parameterless GOMEA and CGOMEA (Dushatskiy et al., ECJ 2023) | 22 | 3: nine black-box GA benchmarks only. DOI 10.1162/evco_a_00338 |
| Filtered linkage hierarchies, LTNGA (Bosman, Thierens 2013) | 39 | 3: GA benchmarks and MAXCUT only. DOI 10.1145/2463372.2463420 |
| Linkage neighbours and forced improvements (Bosman, Thierens 2012) | 37 | 3: GA benchmarks only. Note: forced improvements are part of standard GOMEA; check that pyMOSES has them. DOI 10.1145/2330163.2330247 |
| Fitness-based and conditional linkage for RV-GOMEA (TEVC 2020; GECCO 2020) | 6; 13 | 3: benchmark functions only. 4: low citations. DOI 10.1109/TEVC.2020.3039698 |
| Bias-adjusted MI vs plain MI (Koch et al. 2026 result) | 1 | Not a new method. P(bias-adjusted better than plain MI) ≈ 0.51. This confirms your LT-B finding. |
| Input-space entropy linkage for Boolean GP-GOMEA (Virgolin et al. GECCO 2017) | 47 | It is the LT family. No evidence that it beats RT. DOI 10.1145/3071178.3071287 |
| Coefficient mutation in GP-GOMEA (GECCO 2022) | 12 | Superseded. P(GP-RV-GOMEA better) = 0.83–0.92 (Koch 2024). DOI 10.1145/3520304.3534036 |
| Correlation fitness + one-time linear scaling (Haut et al. GECCO 2023) | 9 | No gain over linear scaling, which you already use. It is only cheaper. DOI 10.1145/3583131.3595918 |
| Geometric semantic GP (Moraglio et al. 2012) | 375 | Your size goal: programs grow very fast. DOI 10.1007/978-3-642-32937-1_3 |
| BOA/hBOA for programs, MOSES with hBOA (Looks et al. 2005; Looks 2007) | 38; 7 | 2: costly model build. MOSES later used hill climbing by default (nv). DOI 10.1145/1068009.1068134 |
| PIPE, ECGP, POLE (classic EDA-GP) | 244; 112; under 10 | 3: toy benchmarks only. The review by Kim et al. (GPEM 2014, 38 OA) finds limited practical gains (nv). DOI 10.1007/s10710-013-9205-x |
| Denoising autoencoder GP (Wittenberg, Rothlauf 2022–2023) | 3–8 | 2: trains a neural model each generation. 4: low citations. DOI 10.1007/s10710-023-09462-2 |
| FFX (McConaghy 2011) | 177 | 3: in SRBench, its models are large and less accurate than the top GP methods (nv). DOI 10.1007/978-1-4614-1770-5_13 |
| Inefficiency of GP for SR (Kronberger et al. PPSN 2024) | 13 | 3: an analysis of revisited expressions, not a method with a gain. MOSES reduct already makes canonical forms. arXiv 2404.17292 |
| GOMEA in Cartesian GP (2022) | 3 | 3 and 4: no real-data gain; low citations. DOI 10.1007/978-3-031-14721-0_2 |

## Suggested next steps (for you to decide)

1. Test node-proximity linkage built from knob positions in the exemplar tree, against RT and univariate, at N 16. It costs nothing and needs no statistics.
2. Test intron-masked MI only if step 1 shows that tree-based linkage helps.
3. Test LM on accepted improvements inside the deme, not only at deme end, at the same CPU budget.

## Sources

- [Koch et al. 2026, arXiv 2602.02311](https://arxiv.org/abs/2602.02311); [gpbib entry](https://gpbib.cs.ucl.ac.uk/gp-html/Koch_2026_GECCO.html)
- [Koch et al. PPSN 2024, CWI PDF](https://ir.cwi.nl/pub/34425/paper_115.pdf)
- [Modular GP-GOMEA, arXiv 2505.01262](https://arxiv.org/abs/2505.01262); [Higher-arity GP-GOMEA, arXiv 2402.09854](https://arxiv.org/abs/2402.09854)
- [Geiger et al., arXiv 2407.21632](https://arxiv.org/abs/2407.21632); [Kronberger et al., arXiv 2404.17292](https://arxiv.org/abs/2404.17292)
- OpenAlex API (`api.openalex.org/works`) for all citation counts and most abstracts.
