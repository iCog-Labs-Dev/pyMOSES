"""NOT A PORT (pyMoses extension, no C++ counterpart): exemplar lineage and
reserved expansions for the empty program (`--reserve-empty-every K`).

When MOSES starts from several exemplars (`-e SEED ... -e +`), it picks the
exemplar to expand by score, so a good seed's descendants crowd out the empty
program's: Phase 1r found the empty program's lineage was never expanded once
seeds were present. With `--reserve-empty-every K` (K > 0) the metapopulation
keeps an `exemplar_lineage`, and every K-th expansion (expansions K, 2K, ...)
picks its exemplar only among the programs that descend from an initial
exemplar that is the empty program (`+` or `and`), using MOSES's normal
Boltzmann selection restricted to that subset. If no program of the subset is
selectable (none left, or all visited too often), that expansion makes the
normal choice instead.

Lineage: the k-th initial exemplar (after reduction) gets the label
`exemplar<k>`. Each program merged into the metapopulation for the first time
inherits the lineage of the exemplar of the expansion that produced it; a
program keeps the first lineage that reached it. Programs are keyed by
`tree_to_stream_str`, the form MOSES prints (`tools/gomea/seed_lineage.py`
uses the same key). Programs injected with `metapopulation.inject_candidates`
get the label passed there.

With K = 0 (the default) no `exemplar_lineage` exists and nothing changes.
"""

from __future__ import annotations

from collections import Counter

from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str

# The empty programs MOSES starts from by default (`type_to_exemplar`):
# `+` for a contin output, `and` for a Boolean one.
EMPTY_PROGRAMS = frozenset({"+", "and"})


def lineage_key(tree) -> str:
    """The lineage key of a combo tree: the program as MOSES prints it."""
    return tree_to_stream_str(tree).strip()


class exemplar_lineage:
    """NOT A PORT: which initial exemplar each metapopulation program descends
    from, and which expansions are reserved for the empty program's lineage."""

    def __init__(self, reserve_empty_every: int) -> None:
        if reserve_empty_every < 0:
            raise ValueError(f"reserve_empty_every must be >= 0, got {reserve_empty_every}")
        self.reserve_empty_every = reserve_empty_every
        self.lineage: dict[str, str] = {}
        self.initial: dict[str, str] = {}
        self.empty_labels: set[str] = set()
        self.current: str | None = None
        self.n_expansions = 0
        self.expansions: Counter = Counter()
        self.reserved_used = 0
        self.reserved_fallbacks = 0

    def add_initial(self, exemplars: list, empty: list[int] | None = None) -> None:
        """Label the initial exemplars `exemplar1 ..`, and note the empty ones:
        the indices in `empty` (the exemplars that were the empty program
        before reduction; `moses_main` passes them, since reduction turns `+`
        into `0` and `and` into `true`), or, when `empty` is None, the
        exemplars that are literally `+` or `and`."""
        for i, ex in enumerate(exemplars):
            label = f"exemplar{i + 1}"
            k = lineage_key(ex)
            self.lineage.setdefault(k, label)
            self.initial[label] = k
            if (k in EMPTY_PROGRAMS) if empty is None else (i in empty):
                self.empty_labels.add(label)

    def label(self, trees: list, label: str) -> None:
        """Give `label` to every tree not yet in the lineage map."""
        for t in trees:
            self.lineage.setdefault(lineage_key(t), label)

    def reserved_turn(self) -> bool:
        """Is the coming expansion (1-based: n_expansions + 1) a reserved one?"""
        k = self.reserve_empty_every
        return k > 0 and bool(self.empty_labels) and (self.n_expansions + 1) % k == 0

    def in_empty_lineage(self, sct) -> bool:
        return self.lineage.get(lineage_key(sct.get_tree())) in self.empty_labels

    def note_selected(self, sct, reserved: bool, fell_back: bool) -> None:
        """Record the exemplar chosen for the coming expansion."""
        if reserved:
            self.reserved_used += 1
        if fell_back:
            self.reserved_fallbacks += 1
        if sct is None:
            return
        self.current = self.lineage.get(lineage_key(sct.get_tree()), "unknown")

    def note_merged(self, candidates) -> None:
        """New programs inherit the current exemplar's lineage."""
        for c in candidates:
            self.lineage.setdefault(lineage_key(c.get_tree()), self.current or "unknown")

    def expansion_done(self) -> None:
        """One expansion has finished (called at the top of `merge_demes`)."""
        self.n_expansions += 1
        self.expansions[self.current or "unknown"] += 1

    def report(self) -> dict:
        return {
            "reserve_empty_every": self.reserve_empty_every,
            "initial": dict(self.initial),
            "empty_labels": sorted(self.empty_labels),
            "n_expansions": self.n_expansions,
            "expansions_by_lineage": dict(self.expansions),
            "empty_lineage_expansions": sum(
                n for lab, n in self.expansions.items() if lab in self.empty_labels
            ),
            "reserved_turns": self.reserved_used,
            "reserved_fallbacks": self.reserved_fallbacks,
        }
