# Approval log

The handoff contract (`00_START_HERE.md`) requires every increment to be approved by the
user, and says the agent must not write its own approvals. This log records the user's own
words verbatim, with the scope that I (the agent) took them to cover. Anything the user did
not state is labelled "agent-proposed" and is not an approval.

| # | When (UTC date) | Increment(s) | Exact user instruction | Base commit | Scope as understood | Compute cap |
|---|---|---|---|---|---|---|
| 1 | 2026-09-24 | P1.0 → P1.5 (all of Phase 1), plus a written plan for Phase 2 (planning only) | "So I have this document exported with prompts, context and plan on how to implement a GP GOMEA extension for the current pyMOSES version. Please use your superpower skill to come up with a task breakdown and plan for the whole thing including phase 2 and continue your execution until you are done with phase 1 entirely and give me the correct deliverables?" | `0410881acee053a449d0b03d1b9781411a9cf9d7` | Write the task breakdown and plan for Phases 1 and 2. Execute Phase 1 end to end without stopping at the per-increment gates, which this instruction explicitly overrides. Create `experiment/moses-gomea-discrete`. **Do not start Phase 2 implementation.** Do not merge, push or rebase. | **Not stated by the user.** Agent-proposed: ≤ 20 concurrent single-threaded processes, ≤ 6 GiB aggregate RAM, ≈ 6 h total wall time for pilot plus confirmation; results outside the source tree at `../gomea-results/phase1/` |
| 2 | 2026-09-24 (mid-turn) | execution style | "use mutli agents whenever you feel it's appropriate" | same | Subagents may implement or review individual plan tasks, and experiment seeds may run as parallel processes. The agents work in this checkout on this branch; there is no `git worktree`, per the handoff's git policy. | same as #1 |

Deviations from the handoff's gate procedure that follow from instruction #1:

* The P1.0 → P1.5 stops are replaced by the review packets committed under `docs/gomea/`
  (one section per increment in `PHASE1-GATES.md`). They record what each gate would have
  asked for, so the user can review after the fact.
* The confirmatory manifest's practical-effect criterion and statistical protocol are
  **agent-proposed and pre-registered before confirmatory runs** (`PREREGISTRATION.md`).
  They are not user-approved thresholds.
