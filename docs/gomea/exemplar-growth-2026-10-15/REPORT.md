# Why GOMEA exemplars grow faster than HC exemplars (pyMoses)

Setup: GOMEA = `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`; HC = `-a hc`; `-m 6000`. Data comes from a monkeypatch wrapper (`wrap.py`). It logs the selected exemplar, the knob count, each deme's best and running-best steps, the merged candidates, and every GOM accept/reject. No repository file was touched. Runs: `runs/*.jsonl`; tables: `analyse.py`, `analyse2.py`, `matched.py`. GOMEA seed 3001 reproduces the profiled run exactly (deme 11: 70 nodes, 1,488 knobs). I stopped GOMEA mux seed 3003 at 3,000 evals because it was too slow (138 nodes, 4,772 s of knob building).

## 1. Measurements

**mux11 at about 3,000 evals.** Exemplar size is nodes/complexity (cx = literal count). "best raw" is the best deme result so far.

| run | demes | exemplar | knobs | best raw | knob-build CPU (s) |
|---|---|---|---|---|---|
| HC 3001/3002/3003 | 3/1/2 | 10/6, 1/0, 17/10 | 255, 88, 428 | -444, -608, -320 | 5, 0, 15 |
| GOMEA 3001/3002/3003 | 12/12/12 | 70/43, 58/35, 138/86 | 1488, 248*, 123* | -162, -132, -164 | 1016, 1336, 4772 |

**mux11 at 6,000 evals (or when solved).** HC: exemplars 39/22, 48/28, 38/22, with best raw -148, -184, -128; none solved. GOMEA: 3001 solved at 5,250 evals (exemplar 71/43), 3002 solved at 4,419 (56/34), 3003 stopped at 3,000.

**Exemplar complexity the first time the exemplar's raw score reaches a threshold** (mux11, seeds 3001/3002/3003):

| raw >= | HC | GOMEA |
|---|---|---|
| -512 | 17, 19, 17 | 18, 34, 47 |
| -384 | 22, 28, 17 | 29, 25, 59 |
| -256 | 28, 22 | 36, 32, 73 |

**maj5:** no difference. Both arms level off at cx 9-15. At raw >= -2 the exemplar cx is 9/9/12 for HC and 11/12/12 for GOMEA. Final knob counts are below 200 for both.

Full per-expansion tables (exemplar nodes/cx/raw/penalized, knobs, deme best raw/cx/nodes, merged count and cx, metapopulation size): `python3 analyse.py <run>`.

## 2. Mechanism

Exemplar growth comes from two factors multiplied together. The first is more expansions per evaluation: 12 GOMEA demes against 1-3 HC demes by 3,000 evals, because HC demes use 500-2,000 evals each. The second is about 1.85x more literals added per row gained in each deme.

**(a) Acceptance: the acceptance rule is not the cause; the order in which moves are tried is.** `composite_score.__lt__` (types.py:454) is lexicographic: penalized score first, then complexity. On mux11 one row is worth 3.5 literals, so any trial that gains raw score is accepted almost regardless of its size. HC uses the same comparator. The difference is that HC scores the whole distance-1 neighbourhood and moves to the best neighbour (hill_climbing.py:1232, 1338, 1368). GOM takes the first non-worse trial in a shuffled FOS order (gom.py:163-195). Every GOMEA deme ran exactly one generation and stopped on `inner_budget`, because the 250-eval cap is smaller than the dimension (88-2,198 knobs). In practice it behaves as first-improvement local search.
- Running-best steps, GOMEA vs HC: median gain 4 rows vs 32. 188 of 278 GOMEA steps gained 8 rows or fewer, against 8 of 61 for HC.
- GOMEA's small steps cost 130 literals for 568 rows (23 literals per 100 rows). Its larger steps cost 7.7 literals per 100 rows; HC's cost 2.8.
- Per deme, GOMEA adds 5.52 literals per 100 rows gained; HC adds 2.99.
- The share of accepted moves that add literals is not higher for GOMEA: 268 of 434 accepted improving GOM moves increase complexity (62%), against 48 of 61 HC steps (79%).

**(b) Merge: GOMEA does not flood the metapopulation.** On mux, GOMEA merges 6.5 candidates per expansion and HC merges 5.2. On maj, GOMEA merges 24 and HC merges 95. Trimming uses `useful_score_range` = 6 x 0.3 = 1.8 penalized (merging.py:320/356).

**(c) Selection: it does not favour larger programs.** Selection is Boltzmann on penalized score with inv_temp = 100/6 ~ 16.7 (metapopulation.py:1066), and each exemplar is visited at most once (revisit = 0, :1051). One extra literal at equal raw lowers the selection weight about 116-fold (e^-4.8). On mux the next exemplar was the previous deme's best in 27/27 GOMEA and 10/10 HC expansions, and on average only 1.0-1.3 candidates were within 0.06 penalized of the best. Growth therefore equals the deme-best growth.

**(d) Group copying: not the driver.** Accepted GOM changes are small: 334 of 434 improving moves and 868 of 1,108 neutral moves change exactly 1 knob, and only 6 improving moves change 9 or more. Large copies are rejected: rejected trials change 9.9 knobs and add 2.45 literals on average.

**(e) Complexity penalty: the same for both arms.** It is `-z 3.5` by default (problem_params.py:683), giving a coefficient of 1/3.5 = 0.286 per literal (scoring_base.py:423). It is set once for both arms. On mux11 (2,048 rows, raw = -errors) this penalty almost never binds. On maj5 (32 rows) it does bind, which is why maj shows no growth gap.

## 3. Recommendations

1. **Make complexity cost more for GOMEA (tested: `-z 0.125`, so a literal must buy 8 rows).** Results on mux11:
   - At 3,000 evals, exemplar cx is 23/25/16 instead of 43/35/86.
   - Literals per 100 rows drop to 2.35, better than HC.
   - Knob-build CPU drops to 272/597/275 s from 1016/1336/4772 s.
   - Best raw at 3,000 evals is -144/-128/-320 against -162/-132/-164 for default GOMEA.
   - At 6,000 evals: 3001 solved at 4,000 evals (766 s of knob building, against 5,250 evals and 3,749 s), 3003 solved at 5,175 evals (1,005 s), and 3002 failed (best -64, against solved at 4,419).

   Risk: a global `-z` also flattens selection, and seed 3002 fell back to a -400 exemplar. `-z 0.5` was weaker: 4.31 literals per 100 rows, cx 27/37.
2. **Penalize complexity inside GOM demes only (preferred, untested).** Make `_OracleEvaluator.worse` (optimizer.py:268) compare `raw - lambda*cx` with lambda of about 8 rows per literal on mux (or set it relative to the score range), and leave merge and selection on the default penalty. This should give recommendation 1's growth curb without its selection side effect. One required detail: the deme stores rejected trials too, so the merge must rank them under the same lambda, or keep only accepted and elite individuals. Otherwise bloated rejected trials with higher raw scores come back in.
3. **Add a short simplification sweep before each GOMEA deme returns (untested).** Spend 10-20% of the cap on single-knob changes to the elite that remove literals. In GOMEA's zero-gain steps, 52 simplifications removed 103 literals, so this works when it is tried. HC gets the same effect from scanning its whole neighbourhood.
4. **Test a larger per-deme cap or best-of-k acceptance.** Fewer, longer demes mean fewer knob builds (HC: 4-5 per 6,000 evals, GOMEA: 12 per 3,000). Accepting the best of k trials raises the gain per accepted step.
5. **Not recommended.**
   - Strict acceptance: the bloating steps are already strict improvements.
   - Limiting merges: the exemplar is always the previous deme's best.
   - Capping knob-set size: knobs scale at about 20 per node, so a cap treats the symptom.

*Side observation, not investigated:* knob counts sometimes collapse for large exemplars (138 nodes -> 123 knobs, 58 -> 248), which suggests occasional truncated representations.
