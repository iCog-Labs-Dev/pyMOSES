"""Re-seed MOSES with fresh GP-GOMEA elites when it stagnates (backlog item 9, later
extension).

`install(spec, ...)` adds a hook to pyMoses's `local_moses.expansion_hooks` (pyMoses only;
nothing else changes). After every expansion the hook compares the metapopulation's best
score with the best seen so far. When it has not strictly improved for `reseed_after`
consecutive expansions, and fewer than `max_reseeds` re-seeds have run, it:

1. gives GP-GOMEA `floor(reseed_share * (max_evals - n_evals))` evaluations, i.e. a share
   of what is left of MOSES's `-m`, and runs `gpgomea_seed.make_seeds` with a NEW random
   seed (`seed + 1000 * r` for the r-th re-seed), the same height, k, population size and
   functions as the initial seeding, verifying each translation with the running MOSES's
   own scorer;
2. adds GP-GOMEA's evaluations (its own `Evaluations:` count, which can exceed the share
   by up to one generation) to MOSES's `stats.n_evals`, so they count against `-m` and the
   run's total budget stays fixed;
3. reduces the new elites as MOSES reduces exemplars and merges them into the
   metapopulation (`metapopulation.inject_candidates`; with `seed_lineage` installed they
   get the lineage `reseed<r>`), then resets the stagnation counter.

GP-GOMEA checks its evaluation limit only between generations, so a small share is
overspent: on Con, a share of 128 cost 684 evaluations. A re-seed is therefore skipped (and
recorded) when the share is below `reseed_min_budget` evaluations, so that a re-seed
is never given less than the initial seeding was (`moses_sr_runner.py` sets that default
to the initial seeding's budget; without a runner it is GP-GOMEA's population size).
Skipped re-seeds do not count towards `max_reseeds`.
Any overspend still comes out of MOSES's `-m`; it can push the run past its total budget
only if it exceeds what MOSES has left, which `over_budget` in the event records. GP-GOMEA's initial population is NOT seeded from
the metapopulation (GP-GOMEA would need C++ changes to read programs into its fixed tree
template): each re-seed is a fresh GP-GOMEA run with a new seed. The scorer calls that
verify and merge the elites are not counted, as in the initial seeding.

`Reseeder.write(path)` stores the events as JSON (`n_reseeds`, `reseed_evals`, `events`).
"""

from __future__ import annotations

import json
from pathlib import Path

_HOOK = None


class Reseeder:
    def __init__(
        self,
        spec: dict,
        binary: str,
        workdir: str | Path,
        seed: int,
        kind: str = "regression",
        train_csv: str | Path | None = None,
        bool_flags: list[str] | None = None,
    ):
        self.after = int(spec["reseed_after"])
        if self.after <= 0:
            raise ValueError(f"reseed_after must be > 0, got {self.after}")
        self.share = float(spec.get("reseed_share", 0.2))
        if not 0 < self.share < 1:
            raise ValueError(f"reseed_share must be in (0, 1), got {self.share}")
        self.max_reseeds = int(spec.get("max_reseeds", 1))
        self.spec = spec
        self.binary = binary
        self.workdir = Path(workdir)
        self.seed = int(seed)
        self.kind = kind
        self.train_csv = train_csv
        self.bool_flags = bool_flags
        import gpgomea_seed as gs

        self.min_budget = int(
            spec.get("reseed_min_budget", spec.get("popsize") or gs.DEFAULT_POPSIZE[kind])
        )
        self.best = None
        self.stall = 0
        self.events: list[dict] = []

    @property
    def n_reseeds(self) -> int:
        return sum(1 for e in self.events if e.get("status") == "ok")

    def __call__(self, mp, dex, pa, stats) -> None:
        best = mp.best_score()
        if self.best is None or best > self.best:
            self.best, self.stall = best, 0
            return
        self.stall += 1
        if self.stall < self.after or self.n_reseeds >= self.max_reseeds:
            return
        self.stall = 0
        self.reseed(mp, dex, pa, stats)
        self.best = max(self.best, mp.best_score())

    def reseed(self, mp, dex, pa, stats) -> dict:
        import gpgomea_seed as gs

        from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
        from pymoses.comboreduct.combo.vertex import combo_tree

        r = self.n_reseeds + 1
        left = pa.max_evals - stats.n_evals
        budget = int(self.share * left)
        ev = {
            "reseed": r,
            "at_expansion": stats.n_expansions,
            "n_evals_before": stats.n_evals,
            "evals_left": left,
            "budget": budget,
            "best_before": float(mp.best_score()),
        }
        if budget < self.min_budget:
            ev.update(status="skipped", reason=f"budget {budget} < {self.min_budget}")
            self.events.append(ev)
            return ev
        spec = self.spec
        rec = gs.make_seeds(
            self.kind,
            self.binary,
            budget,
            int(spec.get("height", 3)),
            self.seed + 1000 * r,
            int(spec.get("k", 5)),
            self.workdir / f"reseed{r}",
            bool_flags=self.bool_flags,
            train_csv=self.train_csv,
            popsize=spec.get("popsize"),
            functions=spec.get("functions"),
            extra=spec.get("gpgomea_argv"),
            tol=float(spec.get("tol", 1e-5)),
            scorer=(mp.get_cscorer(), dex.simplify_candidate),
        )
        stats.n_evals += rec["evaluations_used"]
        trees = []
        for s in rec["seeds"]:
            t = combo_tree(str_to_combo_tree(s["combo"]).begin())
            dex.simplify_candidate(t)
            trees.append(t)
        label = f"reseed{r}"
        try:  # the runner's lineage tracker, if installed
            import seed_lineage

            if seed_lineage._TRACKER is not None:
                for t in trees:
                    seed_lineage._TRACKER.lineage.setdefault(seed_lineage.Tracker.key(t), label)
        except ImportError:
            pass
        mp.inject_candidates(trees, label)
        ev.update(
            status="ok",
            gpgomea_seed=self.seed + 1000 * r,
            evaluations_used=rec["evaluations_used"],
            gpgomea_cpu_s=rec["gpgomea"]["cpu_s"],
            n_injected=len(trees),
            seeds=[s["combo"] for s in rec["seeds"]],
            seed_raw=[s["pymoses_raw"] for s in rec["seeds"]],
            n_evals_after=stats.n_evals,
            over_budget=max(0, stats.n_evals - pa.max_evals),
            best_after=float(mp.best_score()),
        )
        (self.workdir / f"reseed{r}" / "seeds.json").write_text(json.dumps(rec, indent=2))
        self.events.append(ev)
        return ev

    def report(self) -> dict:
        ok = [e for e in self.events if e.get("status") == "ok"]
        return {
            "reseed_after": self.after,
            "reseed_share": self.share,
            "max_reseeds": self.max_reseeds,
            "n_reseeds": len(ok),
            "reseed_evals": sum(e["evaluations_used"] for e in ok),
            "reseed_cpu_s": sum(e["gpgomea_cpu_s"] for e in ok),
            "events": self.events,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report(), indent=2, sort_keys=True) + "\n")


def install(spec: dict, binary: str, workdir, seed: int, **kw) -> Reseeder:
    """Add a `Reseeder` to `local_moses.expansion_hooks` (replacing one installed before)."""
    global _HOOK
    from pymoses.moses.moses import local_moses

    uninstall()
    _HOOK = Reseeder(spec, binary, workdir, seed, **kw)
    local_moses.expansion_hooks.append(_HOOK)
    return _HOOK


def uninstall() -> None:
    global _HOOK
    from pymoses.moses.moses import local_moses

    if _HOOK is not None and _HOOK in local_moses.expansion_hooks:
        local_moses.expansion_hooks.remove(_HOOK)
    _HOOK = None
