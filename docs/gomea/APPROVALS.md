# Approval log

The handoff contract (`00_START_HERE.md`, in the untracked handoff pack
`MOSES_GOMEA_LOCAL_AGENT_HANDOFF/moses_gomea_plan/`, not committed) requires every increment to be approved by the
user, and says the agent must not write its own approvals. This log records the user's own
words verbatim, with the scope that I (the agent) took them to cover. Anything the user did
not state is labelled "agent-proposed" and is not an approval.

| # | When (UTC date) | Increment(s) | Exact user instruction | Base commit | Scope as understood | Compute cap |
|---|---|---|---|---|---|---|
| 1 | 2026-09-24 | P1.0 → P1.5 (all of Phase 1), plus a written plan for Phase 2 (planning only) | "So I have this document exported with prompts, context and plan on how to implement a GP GOMEA extension for the current pyMOSES version. Please use your superpower skill to come up with a task breakdown and plan for the whole thing including phase 2 and continue your execution until you are done with phase 1 entirely and give me the correct deliverables?" | `0410881acee053a449d0b03d1b9781411a9cf9d7` | Write the task breakdown and plan for Phases 1 and 2. Execute Phase 1 end to end without stopping at the per-increment gates, which this instruction explicitly overrides. Create `experiment/moses-gomea-discrete`. **Do not start Phase 2 implementation.** Do not merge, push or rebase. | **Not stated by the user.** Agent-proposed: ≤ 20 concurrent single-threaded processes, ≤ 6 GiB aggregate RAM, ≈ 6 h total wall time for pilot plus confirmation; results outside the source tree at `../gomea-results/phase1/` |
| 2 | 2026-09-24 (mid-turn) | execution style | "use mutli agents whenever you feel it's appropriate" | same | Subagents may implement or review individual plan tasks, and experiment seeds may run as parallel processes. The agents work in this checkout on this branch; there is no `git worktree`, per the handoff's git policy. | same as #1 |
| 3 | 2026-09-24 | P1.5 confirmation compute envelope | Answered the agent's question "How much compute should the confirmation run use?" with: "Trimmed, ~2.5–3 h (Recommended)" (the option read: 30 seeds × 6 arms on maj5 + mux6 at 20k evals, plus pa4 at 20 seeds, plus the cheap frozen-representation replay comparison) | branch head at the time | the confirmation matrix as described in the option | ≈ 2.5–3 h wall on ≤ 20 workers (user-selected) |
| 4 | 2026-09-26 | P1.5 execution venue | "Okay so I tired running it on the server but I couldn't can you do the bentchmarking yourself please?" | `4ae644a` (manifests pin code `c6301cb`) | Run the committed, pre-registered manifests locally, unchanged, with the committed driver (`tools/gomea/run_phase1.sh`) | Not restated; the agent chose 14 workers (about one per physical core) and a 512 MB per-run memory limit. The earlier 2.5–3 h envelope (#3) was based on an estimate the agent later corrected to about 4 h |
| 5 | 2026-09-26 | Phase 1b (new experiment) | "What if we do another experiment? An experiment where we check if the normal GP-GOMEA without being integrated into MOSES beats the normal hillcimbing integrated into MOSES on similar problems, what if we check that? Can you do that and do similar analysis." | `51cb422` | Build and run the authors' GP-GOMEA stand-alone on the Phase 1 Boolean tasks; compare with the Phase 1 MOSES HC runs using a pre-registered analysis | not stated; the C++ runs take minutes |
| 6 | 2026-09-26 | Armadillo download | Chose "Allow the download (Recommended)" when the agent asked how to build GP-GOMEA (the download of Armadillo 14.0.2 source had been blocked by the permission classifier) | same | Fetch Armadillo 14.0.2 source from its official GitLab, pinned and checksummed, into `gomea-results/`, use it header-only; no system install; the user's GP-GOMEA checkout stays untouched | — |
| 7 | 2026-09-26 | Phase 1c (scalability experiment) | "So it looks like GP-GOMEA sovles problems but requires more budget, can we check on bigger problems that can test the scalability of using GP-GOMEA please? I wanna see how that goes instead." | `4aa3efe` | Measure how stand-alone GP-GOMEA's success and evaluations-to-solve scale over parity 3–7, majority 5–9 and multiplexer 6–11 at a large budget, with classic C++ MOSES hill climbing as a same-language reference (agent's design choice) | not stated; agent-estimated about 2–3 h wall on 14 workers |
| 8 | 2026-09-27 | Phase 1d (why MOSES HC beat embedded GOMEA) | "The next experiment should be trying to figure out why this is the case." then, after the agent proposed a two-stage ablation (hypotheses A–E), "yes go ahead with the full design" | `ce698be` | Implement opt-in options for the ablation (uniform initialization, strict acceptance, per-deme genotype memo) plus a harness-side duplicate counter; pre-register; stage 1 on frozen representations; stage 2 end-to-end on maj5, mux6, pa4 with the supported factors | agent-estimated 2–3 h for stage 1 and 3–4 h for stage 2 |
| 9 | 2026-09-27 | Phase 1e (make embedded GOMEA beat HC; then larger problems) | "What else do you think we do for gomea to beat hill climbing? Shall we test it with more problems instead?" then, after the agent proposed a two-stage plan, "Go ahead then what are you waiting for" | `9472b66` | Add a stall-based deme stop and an HC→GOMEA hybrid (opt-in); stage 1 on dev seeds over maj5, mux6 and pa4 to choose the best setup by a pre-registered rule; stage 2 pits it against HC on maj7, pa5 and mux11 | agent-estimated about 3 h for stage 1 and 4–5 h for stage 2 |
| 10 | 2026-09-28 | Phase 1e stage 2 re-runs | Chose "Complete it (~10 h) (Recommended)" when the agent reported that 25 stage 2 runs were time-, memory- or recursion-limited, then "Go ahead with the reruns" | `63023d9` | Re-run only the 25 affected runs with the same code, seeds and budget under a 36,000 s wall cap, a 2,048 MB memory limit and a recursion limit of 100,000; disclose as a deviation in `PHASE1E-PREREGISTRATION.md` | about 10 h (agent estimate, accepted) |
| 11 | 2026-09-28 | Phase 1f (the GP-GOMEA paper's own benchmarks) | "Now instead of using the demo problems MOSES was tested on, we need to test the demo problems GP-GOMEA was tested on for the research paper..." (four parts: pyMoses GOMEA vs the literature; replicate GP-GOMEA C++ at small scale; GP-GOMEA vs classic C++ MOSES; pyMoses GOMEA vs pyMoses HC; "reduce the number of seeds from 20 to 10 or lower if that's justifiable"; "maximum resources"). Then chose "Paper-style constants (Recommended)" and "Allow downloads (Recommended)" | `28a118a` | Download the 10 regression datasets of Virgolin et al. 2021 (checksummed, outside the repo). Add an opt-in `--gomea-contin copy` so GOMEA can run on representations with constants, treating each constant as a gene that mixing copies but never mutates, as the paper does; this is the only Phase 2-adjacent code approved. No real-valued model | not stated; agent to estimate in the pre-registration |
| 12 | 2026-09-29 | Phase 1f parts C and D rescope after an out-of-memory crash | Chose "Drop LTMI, keep 10k" | recorded in `332d3fe` | Keep 10,000 evaluations per run; run HC, LTB and RT only on the 7 remaining datasets (no LTMI there); re-run the failed and unfinished runs; LTB vs LTMI rests on Air, Bos and Con. Recorded as a deviation in `PHASE1F-PREREGISTRATION.md` | agent-proposed: 11 workers, a 3,072 MB address-space cap per run |
| 13 | 2026-09-30 | Phase 1f parts C and D, after memory pressure | Chose "Restart with 7 workers" | recorded in `fee65a2` | Restart the runner with 7 workers; the design is unchanged | 7 workers |
| 14 | 2026-09-30 | Phase 1f parts C and D early stop | Chose "Report now on what's done" | recorded in `fee65a2` | Stop the runner (at 181 of 330 runs) and analyse what finished: Air, Bos and Con confirmatory (Holm over 3), the rest descriptive | no further compute |
| 15 | 2026-09-30 | Phase 1f parts C and D extension | Chose "Finish 4 medium datasets" | registered in `b56c148`, before these runs | Complete Dow, EnC, EnH and Yac to 10 splits with HC, LTB and RT, same code and arguments; Tow, WiR and WiW stay descriptive | agent-proposed: pass 1 on 5 workers with a 3,072 MB cap, pass 2 (Dow LTB and RT) on 2 workers with a 5,120 MB cap |
| 16 | 2026-09-30 | Dow LTB in the extension | Chose "Drop Dow LTB, report it" | recorded in `8451af3` | Do not run the remaining Dow LTB runs; report Dow as "LTB does not fit in 5 GB on this machine"; part D confirmatory on the other 6 datasets. Dow RT was later stopped at 6 of 10 splits after the user asked: "Why are we running the random-tree GOMEA for Dow if the Bias-corrected GOMEA is dropped because of memory? There's no point" (`8fdf163`) | no further Dow LTB compute |
| 17 | 2026-10-11 | Phase 1g (short demes or linkage?) | "let's just check whether the performance gain we observed is actually from the smaller deme or from the usage of a linkage tree in the first place with as small of a benchmark as possible on few of selected demo probelms to make the run finish with in the next hour" | `1d6e3e0` | Add an opt-in `-a hc-capped` (hill climbing with GOMEA's per-deme cap) and run a small exploratory comparison (`PHASE1G-PREREGISTRATION.md`) | about an hour |
| 18 | 2026-10-11 | Phase 1h (bigger linkage sample) | "I want you to somehow increase the number of candidates the linkage learning gets so we can see it in action ... If this just requires you to increase the global budget and the deme cap then do it. Repeat the process until it is conclusive if either the thing is true or not is correct." | `7860f64` | Frozen maj5, LT-B/LT-MI/RT at growing population sizes with a larger budget and no deme cap; escalate per `PHASE1H-PREREGISTRATION.md` until a conclusive rule fires | user: "repeat until conclusive"; agent-estimated about 1 h, then about 5 h, for the first two steps |

**Not user-approved.** Two Phase 1f code changes outside GOMEA were bug fixes the agent
made on its own, disclosed as deviations in `PHASE1F-PREREGISTRATION.md`: the
`reduce_factorize_fraction` port-bug fix (`484e302`) and the regression runner's raised
recursion limit (`8b23deb`). No row above approves them.

**Also not user-approved (2026-10-15):** the `twiddle_contin_bit` fix. Hill climbing on a
contin-only deme crashed with `IndexError` when the last contin field's last pseudo-bit was
twiddled. C++ makes the same out-of-range read (`neighborhood_sampling.cc:46-47`,
`field_set.h:412-416`) without a bounds check, which is undefined behaviour. The port now
treats that "next" position as invalid, as the C++ header documents
(`neighborhood_sampling.h:82-83`), marked as a DIVERGENCE. Only calls that used to crash
change; no completed earlier run could have been affected. The user asked for the speed-up
work to proceed autonomously; this fix was made along the way and is disclosed here.

**Also not user-approved as code (2026-10-16):** four opt-in options, built by the agent while
working through the backlog, which the user asked it to do autonomously ("please do not stop
before finishing our backlog"). All four are off by default. With them off, outputs are
byte-identical to the previous code (identity harness, all 8 cases).

* `--knob-probe-baseline reduced` (knob-collapse fix): a DIVERGENCE from C++ `disc_probe`
  (`build_knobs.cc:311-341`), opt-in only.
* `--gomea-fos-topk K` / `--gomea-fos-topk-mode`.
* `--simplify-exemplar-numeric TOL`.
* GP-GOMEA elite seeding: `tools/gomea/gpgomea_seed.py` and the seed block in the runners.

Each is used only in a pre-registered phase that names it.

**Also agent-built (2026-10-19):** `--gomea-accept-complexity-ratio R` now also applies to
`-a hc-capped` (commit `ebd77c0`), for Phase 1t.

* **What changed:** a deme-local penalty in hill climbing's ranking of new instances. It is
  an extension hook, not a C++ divergence, and is off by default.
* **Checks:** byte-identical on all 8 identity-harness cases.
* **The user's request:** an experiment testing the penalty on hill climbing ("Run both of
  them as different experiments"). The code itself was not separately approved.

**Also not user-approved as code (2026-10-19):** two opt-in seeding options, built by the
agent from the Phase 1r follow-ups (backlog item 9) at the coordinator's request. Both are
off by default. With them off, outputs are byte-identical to the previous code (identity
harness, all 8 cases).

* `--reserve-empty-every K`: lineage tracking in the metapopulation and every K-th
  expansion restricted to the empty program's lineage (`pymoses/moses/metapopulation/
  lineage.py`); seed-block key `reserve_empty_every`.
* Re-seeding on stagnation: `tools/gomea/reseed.py`, the port-only
  `local_moses.expansion_hooks` list and `metapopulation.inject_candidates`; seed-block
  keys `reseed_after`, `reseed_share`, `max_reseeds`. A fresh GP-GOMEA run per re-seed
  (not seeded from the metapopulation).

Neither is used in a pre-registered phase yet.

**Also not user-approved as code (2026-10-20):** the `-B 3` crash fix (BACKLOG section 4,
item 2). At effort 3, `reduce_remove_subtree_equal_tt` could cut the last child of the root
`and`/`or` and then assert while evaluating the empty junctor; C++ has the same latent bug
(`logical_rules.cc:847-849`, `OC_ASSERT` at `vertex.h:710`). The port now treats that cut as
truth-table-changing and keeps the child, marked as a DIVERGENCE. Only inputs that used to
crash change; the default `-B 2` never reaches the rule, and the identity harness is
byte-identical on all 8 cases.

**Also not user-approved as code (2026-10-21):** `--linear-scaling {0,1}` (BACKLOG item 10,
Phase 1ag in `PHASE1AC-1AG-PREREGISTRATION.md`), built by a subagent at the coordinator's
request on branch `feat/linear-scaling`. With `1`, the regression scorer (`contin_bscore`,
shared by `hc`, `hc-capped`, `gomea` and the hybrid) scores each program `f` on `a + b·f`,
the least-squares fit on the training rows (`b = cov(y, f) / var(f)`, `a = mean(y) − b·mean(f)`;
`b = 0` for a constant `f`), the formula `tools/gomea/sr_eval.py` uses when it reports NMSE.
It is an extension with no C++ counterpart, off by default. With it off, outputs are
byte-identical to the previous code (the identity harness's `con-hc` and `con-gomea`, the
same two with `--knob-probe-baseline reduced`, and the Phase 1v/1r regression arm argvs on
Con split 1 at 1,500 evaluations). Phase 1ag is pre-registered, but the code itself was not
separately approved.

**Also not user-approved as code (2026-10-21):** `--knob-probe-baseline reduced` is now the
code default (BACKLOG item 15), a disclosed DIVERGENCE from C++ `disc_probe`
(`build_knobs.cc:311-341`), marked in `problem_params.py`, `deme_params.py` and
`build_knobs.py`. The user asked for this as a plan in BACKLOG item 15 (2026-10-19: "make it
the code default (a disclosed DIVERGENCE) once it has run in two phases without a loss"),
which this log does not count as approval of the code change. The default changed for the
`moses` command line (`add_gomea_options`, `parse_options`) and `deme_parameters`; the ported
`build_knobs` and `representation` constructors still default to `classic`.
`--knob-probe-baseline classic` gives the old behaviour. Every arm since Phase 1u passes
`--knob-probe-baseline reduced` explicitly, so their outputs do not change (checked
byte-identical before and after on `mux11-gomea`, `maj5-hc`, `con-hc`, `con-gomea` and the
Phase 1v/1r regression arm argvs, each with the flag; `classic` given explicitly reproduces
the old default byte for byte).

Deviations from the handoff's gate procedure that follow from instruction #1:

* The P1.0 → P1.5 stops are replaced by the review packets committed under `docs/gomea/`
  (one section per increment in `PHASE1-GATES.md`). They record what each gate would have
  asked for, so the user can review after the fact.
* The confirmatory manifest's practical-effect criterion and statistical protocol are
  **agent-proposed and pre-registered before confirmatory runs** (`PREREGISTRATION.md`).
  They are not user-approved thresholds.
