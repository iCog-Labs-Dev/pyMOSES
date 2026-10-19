"""Which initial exemplar does each metapopulation program descend from? (backlog item 9)

`install(labels)` wraps three `metapopulation` methods of the running process (pyMoses
only; nothing else changes):

* `init`: the k-th exemplar MOSES starts from (the k-th `-e`, after reduction) gets
  `labels[k]`, e.g. `seed1 .. seedK, empty`;
* `select_exemplar`: records the lineage of the exemplar each expansion starts from
  (`expansions` counts expansions per lineage);
* `merge_candidates`: each program merged into the metapopulation for the first time
  inherits the lineage of the exemplar of the deme it came from.

A program is keyed by `tree_to_stream_str`, the form MOSES prints, so a printed
candidate can be looked up with `key_of(program_string)`. A program first reached from
one lineage keeps that lineage even if another lineage reaches it later.

`Tracker.report()` gives the lineage of the metapopulation's best programs (`best`,
`best_lineages`), of every program still in the metapopulation (`pool`, counted in
`metapop_by_lineage`) and the expansions per lineage; `write(path)` stores it as JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_TRACKER = None
_ORIGINALS: dict = {}


def key_of(program: str) -> str:
    """The lineage key of a printed program (parse, then print as MOSES does)."""
    from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str

    return tree_to_stream_str(str_to_combo_tree(program)).strip()


class Tracker:
    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.lineage: dict[str, str] = {}
        self.initial: dict[str, str] = {}
        self.current: str | None = None
        self.expansions: Counter = Counter()
        self.merged: Counter = Counter()
        self.metapop = None

    @staticmethod
    def key(tree) -> str:
        from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str

        return tree_to_stream_str(tree).strip()

    def report(self) -> dict:
        out: dict = {
            "labels": self.labels,
            "initial": self.initial,
            "expansions_by_lineage": dict(self.expansions),
            "programs_merged_by_lineage": dict(self.merged),
        }
        mp = self.metapop
        if mp is None:
            return out
        try:
            best = [self.key(c.get_tree()) for c in mp.best_candidates()]
        except Exception:  # never let the report break a run
            best = []
        out["best"] = [{"program": b, "lineage": self.lineage.get(b)} for b in best]
        out["best_lineages"] = sorted({self.lineage.get(b) or "unknown" for b in best})
        counts: Counter = Counter()
        pool: dict[str, str] = {}
        try:
            for sct in mp.get_trees():
                k = self.key(sct.get_tree())
                pool[k] = self.lineage.get(k) or "unknown"
                counts[pool[k]] += 1
        except Exception:
            pass
        out["metapop_by_lineage"] = dict(counts)
        out["pool"] = pool
        core = getattr(mp, "_lineage", None)  # `--reserve-empty-every` (pyMoses core)
        if core is not None:
            out["reserve_empty"] = core.report()
        return out

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report(), indent=2, sort_keys=True) + "\n")


def install(labels: list[str]) -> Tracker:
    """Wrap the metapopulation methods (once per process) and return the tracker."""
    global _TRACKER
    from pymoses.moses.metapopulation.metapopulation import metapopulation

    tracker = Tracker(labels)
    if _TRACKER is not None:  # already wrapped: just point the wrappers at a new tracker
        _TRACKER.__dict__.update(tracker.__dict__)
        return _TRACKER
    _TRACKER = tracker
    orig_init = metapopulation.init
    orig_select = metapopulation.select_exemplar
    orig_merge = metapopulation.merge_candidates
    _ORIGINALS.update(init=orig_init, select_exemplar=orig_select, merge_candidates=orig_merge)

    def init(self, exemplars):
        t = _TRACKER
        t.metapop = self
        for i, ex in enumerate(exemplars):
            label = t.labels[i] if i < len(t.labels) else f"exemplar{i + 1}"
            k = t.key(ex)
            t.lineage.setdefault(k, label)
            t.initial[label] = k
        return orig_init(self, exemplars)

    def select_exemplar(self):
        t = _TRACKER
        res = orig_select(self)
        if res is not None:
            t.current = t.lineage.get(t.key(res.get_tree()), "unknown")
            t.expansions[t.current] += 1
        return res

    def merge_candidates(self, candidates):
        t = _TRACKER
        for c in candidates:
            k = t.key(c.get_tree())
            if k not in t.lineage:
                t.lineage[k] = t.current or "unknown"
                t.merged[t.lineage[k]] += 1
        return orig_merge(self, candidates)

    metapopulation.init = init
    metapopulation.select_exemplar = select_exemplar
    metapopulation.merge_candidates = merge_candidates
    return tracker


def uninstall() -> None:
    """Restore the original metapopulation methods (for tests)."""
    global _TRACKER
    if _TRACKER is None:
        return
    from pymoses.moses.metapopulation.metapopulation import metapopulation

    for name, fn in _ORIGINALS.items():
        setattr(metapopulation, name, fn)
    _ORIGINALS.clear()
    _TRACKER = None


def labels_for(n_seeds: int, with_empty: bool = True) -> list[str]:
    return [f"seed{i + 1}" for i in range(n_seeds)] + (["empty"] if with_empty else [])
