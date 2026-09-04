"""C++: moses/moses/scoring/bscores.h and bscores.cc

`ctruth_table_bscore` -- boolean classification scorer over a compressed
truth table. Ports exactly `bscores.h:360-383` (the class declaration) and
its out-of-line members at `bscores.cc:276-424`, per task-6-brief.md's
Reference line. Every other `*_bscore` class in bscores.h (`logical_bscore`,
`discretize_contin_bscore`, `enum_table_bscore`/`enum_filter_bscore`/
`enum_graded_bscore`, `contin_bscore`, and the precision/multi-objective
scorers in the neighboring `precision_bscore.h`) is out of scope for this
task -- boolean scoring via `ctruth_table_bscore` is the entire ported
surface (see the plan's "Known gap, deliberate" note and this task's own
Context section on `CTable.get_output_type()`).

Regarding the sign/weighting convention this scorer relies on:
`ctruth_table_bscore::operator()`'s doc comment at the top of bscores.h
(reproduced where `__call__` below cites it) explains why each bscore
element is the NEGATIVE count of a row's WRONG answers, and why
`best_possible_bscore`/`worst_possible_bscore`/`min_improv` all read off
of `fmin`/`fabs` over the same `{logical_true: count, logical_false:
count}` pair per compressed row.
"""

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, negate_vertex
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, type_node
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t, very_best_score
from pymetta.moses.scoring.scoring_base import bscore_base


class ctruth_table_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:360 ctruth_table_bscore

    Extends `bscore_base` directly, per task-6-brief.md's interface line
    (`class ctruth_table_bscore(bscore_base)`) -- NOT `bscore_ctable_base`
    (bscores.h's own base, scoring_base.h:236), which is out of scope (see
    scoring_base.py's module docstring). Every `_wrk_ctable` reference in
    the C++ member functions below reads, in this port, as `self._ctable`
    -- there is no separate `_orig_ctable`/`_wrk_ctable` distinction to
    maintain without `ignore_cols`/`ignore_rows`.
    """

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/bscores.h:362 ctruth_table_bscore

        `_size = _wrk_ctable.size()` (bscores.h:365) is `len(ctable)`
        (compressed-row count, i.e. `dict.__len__`, not
        `uncompressed_size()`); `reset_weights()` (bscores.h:366) is not
        called -- it is boosting-only infrastructure this port does not
        implement (scoring_base.py's module docstring).

        # DIVERGENCE(C++ moses/moses/scoring/bscores.h:362-368): classic's
        # constructor never calls `get_output_type()`, it simply assumes a
        # boolean-output CTable, silently misbehaving otherwise
        # (`counter.get(id::logical_true)`/`get(id::logical_false)` against
        # a non-boolean counter just always returns the "not present"
        # default -- i.e. classic would silently return an all-zero
        # bscore rather than raising). This port raises `ValueError`
        # instead, added here as a deliberate, disclosed strengthening --
        # and, per task-6-brief.md's Context section, to be the FIRST real
        # consumer of `CTable.get_signature()`/`get_output_type()`
        # (table.py -- no earlier task's tests ever called a compressed
        # table's `.get_output_type()`), so as to answer its question:
        # does the "signature landmine" (a weighted `Table.compressed()`'s
        # `CTable` carrying an un-trimmed signature next to trimmed
        # `ilabels`, table.py's `Table.compressed` docstring) actually
        # fire here? It does not: `get_output_type()` reads only
        # `self._signature.otype`, wholly unaffected by any
        # `itypes`/`ilabels` LENGTH mismatch, and this class never calls
        # `CTable.filtered()` (the only method with a length cross-check
        # -- and even there, this port's own `filtered()` has no such
        # cross-check to begin with; see table.py). Verified directly
        # against tests/fixtures/table_data/weighted.csv (loaded with a
        # weighting feature, so the mismatch is actually present) in
        # tests/moses/scoring/test_bscores.py.
        """
        super().__init__()
        self._ctable = ctable
        self._size = len(ctable)
        output_type = ctable.get_output_type()
        if output_type != type_node.boolean_type:
            raise ValueError(
                f"ctruth_table_bscore requires a boolean-output CTable, "
                f"got output type {output_type.name!r}"
            )
        self._best_possible_score: behavioral_score = []
        self.set_best_possible_bscore()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:276 ctruth_table_bscore::operator()

        For each compressed row, evaluate `tr` over that row's inputs,
        NEGATE the predicted answer, and look up HOW MANY actual
        observations at that row hold the negated (i.e. WRONG) answer --
        that count, itself negated again (so a perfect row scores 0.0 and
        a row with wrong answers scores negative), is the row's bscore
        element. "By convention, we expect score to be negative (so that
        higher scores==better scores)" (types.py's composite_score
        docstring, quoting types.h verbatim) is exactly this convention.

        `interpreter_visitor(tr)` is constructed once per call (matching
        C++'s `interpreter_visitor iv(tr); auto interpret_tr =
        boost::apply_visitor(iv);`, done once before the per-row loop,
        not once per row) and applied to each row's input tuple, coerced
        to `list` since `interpreter_visitor.__call__` requires a `list`
        (interpreter.py) where a CTable row key is a `tuple`.
        """
        iv = interpreter_visitor(tr)
        bs: behavioral_score = []
        for row, counter in self._ctable.items():
            sc = _to_score_t(counter.get(negate_vertex(iv(list(row))), 0.0))
            bs.append(-sc)

        if self._return_weighted_score:  # if boosting
            # "Report the score only relative to the best-possible
            # score." (bscores.cc:290's own comment.) Unreachable in
            # this port -- see scoring_base.py's module docstring -- kept
            # for fidelity to bscores.cc:289-292.
            bs = [a - b for a, b in zip(bs, self._best_possible_score, strict=True)]

        return bs

    def set_best_possible_bscore(self) -> None:
        """C++: moses/moses/scoring/bscores.cc:350 ctruth_table_bscore::set_best_possible_bscore

        "CTable is a compressed table; different rows may have identical
        inputs, differing only in output. Clearly, in such a case, both
        outputs cannot be simultaneously satisfied, but we can try to
        satisfy the one of which there is more (the max). Thus, we take
        the fmin of the two possibilities as the number of wrong answers
        we are doomed to get. fmin is used, because the rows may be
        weighted by fractional values." (bscores.cc:356-368's own comment,
        reproduced verbatim.) Python's builtin `min` over two floats is
        `fmin`-equivalent for this port's corpus (neither operand is ever
        NaN).
        """
        self._best_possible_score = [
            -_to_score_t(
                min(counter.get(builtin.logical_true, 0.0), counter.get(builtin.logical_false, 0.0))
            )
            for counter in self._ctable.values()
        ]

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:377 ctruth_table_bscore::best_possible_bscore

        "The returned best score will always be zero [when boosting],
        because the actual best score is subtracted; this is required to
        get boosting to work." (bscores.cc:379-382's own comment.) The
        `_return_weighted_score` branch is unreachable in this port (see
        `__call__`'s docstring) but kept for fidelity.

        Returns a COPY of `self._best_possible_score`, matching C++'s
        by-value `behavioral_score` return (bscores.h:375 declares this
        `behavioral_score best_possible_bscore() const`, not a reference)
        -- returning the live list itself would let a caller's in-place
        mutation of the result (e.g. the very `bs -= _best_possible_score`
        pattern `__call__` uses when boosting) corrupt this scorer's own
        cached state for every future call.
        """
        if self._return_weighted_score:
            return [0.0] * self._size
        self.set_best_possible_bscore()
        return list(self._best_possible_score)

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:389 ctruth_table_bscore::worst_possible_bscore

        "The most that the score can improve is to flip true to false, or
        v.v. The worst score is to get the majority wrong. This works
        correctly even for weighted tables, where the counts could be
        arbitrary float-point values." (bscores.cc:395-398's own comment.)
        """
        result = []
        for counter in self._ctable.values():
            t = counter.get(builtin.logical_true, 0.0)
            f = counter.get(builtin.logical_false, 0.0)
            result.append(-_to_score_t(abs(t - f)))
        return result

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:406 ctruth_table_bscore::min_improv

        "A return value of 0.5 would be correct only if all rows had a
        weight of exactly 1.0. Otherwise, we look for the row with the
        smallest (positive) weight, and return that." (bscores.cc:408-410's
        own comment.) `min_weight` is seeded with `very_best_score`
        (types.py) -- this port's bit-exact stand-in for C++'s `FLT_MAX`
        sentinel (bscores.cc:413), so the degenerate case of a CTable
        where every row is split exactly 50/50 between `logical_true` and
        `logical_false` returns the same `0.5 * FLT_MAX` classic does,
        not `math.inf`.
        """
        min_weight = very_best_score
        for counter in self._ctable.values():
            t = counter.get(builtin.logical_true, 0.0)
            f = counter.get(builtin.logical_false, 0.0)
            w = _to_score_t(abs(t - f))
            if w != 0.0 and w < min_weight:
                min_weight = w
        # `0.5 * min_weight` is computed in double precision in C++ too
        # (0.5 is a double literal; float*double promotes to double) and
        # narrowed only at the return, since `min_improv`'s C++ return
        # type is itself `score_t`.
        return _to_score_t(0.5 * min_weight)
