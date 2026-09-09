"""C++: moses/moses/moses/partial.h

"Implements the 'leave well-enough alone' algorithm." (the class's own
comment.) Incrementally builds a `cond(pred1 concl1 pred2 concl2 ...
default)` classifier by repeatedly running MOSES on a progressively
SMALLER table: whenever a candidate's leading predicate correctly
classifies EVERY row it fires on (perfect accuracy, zero false positives),
that `(predicate, conclusion)` pair is peeled off into a growing `_leader`
prefix, every row it accounted for is deleted from the working table, and
MOSES restarts on what remains -- "leave well enough alone" once a clause
is proven perfect, rather than let further search potentially perturb it.

PARTIALLY PORTED -- read this before using `partial_solver`.
`partial_solver.solve()` is blocked end to end by TWO separate,
pre-existing gaps, both entirely outside this task's owned tree (this
task owns only `pymetta/moses/moses/partial.py`,
`pymetta/moses/eda/`, `pymetta/comboreduct/combo/similarity.py`, and
`pymetta/moses/scoring/ss_bscore.py`):

1. `solve()`'s very first two lines construct `BScore(_ctable)`
   (`BScore = enum_effective_bscore`) and, later, `record_prefix`/
   `final_cleanup` also (re)construct an enum-family bscore. NEITHER
   `enum_table_bscore` (bscores.h:409), `enum_graded_bscore` (bscores.h:
   490, `: public enum_table_bscore` -- NOT unrelated to
   `enum_table_bscore` despite living in a different-looking part of the
   header; verified by compiling a minimal probe against the real
   `bscores.h`, not merely read: `enum_table_bscore* p = new
   enum_effective_bscore(ctable);` compiles cleanly), nor
   `enum_effective_bscore` (bscores.h:516, `: public enum_graded_bscore`)
   exists anywhere in this project's `bscores.py` -- an entire
   "enum_type"-output scoring subsystem is unported. `grep -rn
   "enum_table_bscore\\|enum_effective_bscore\\|enum_graded_bscore"
   pymetta/moses/scoring/bscores.py` finds nothing.

   Incidentally: `final_cleanup`'s own comment claims it switches to "the
   non-graded (flat) scorer" (partial.cc:184-186) but its CODE
   constructs `new BScore(_ctable)` -- the SAME graded `enum_effective_
   bscore`, not `StraightBScore`/`enum_table_bscore` as the comment
   describes. This compiles (see the inheritance chain above) and is
   therefore not a type error, but the comment does not match the code
   it describes -- a genuine classic quality issue, noted here for the
   record, NOT "fixed": this port would need to reproduce class hierarchy
   AND recreation-target choice faithfully once the enum bscore family
   lands, whichever the code actually does.

2. Independently, `_meta_params.merge_callback = check_candidates`
   (`solve()`'s own second statement) would immediately hit
   `metapopulation.merge_demes`'s own gate the moment a merge actually
   ran a callback: `pymetta/moses/metapopulation/merging.py`
   (`merge_demes`, outside this task's owned tree) raises
   `NotImplementedError` unconditionally whenever
   `metapop_parameters.merge_callback is not None`
   ("metapopulation::merge_demes's merge_callback hook (merging.cc:
   338-339) is not ported"). `partial_solver` is therefore blocked
   TWICE over, by two unrelated, pre-existing gaps in two different
   files this task does not own -- reported as follow-ups, not fixed
   here.

`solve()` and the bscore-(re)construction tails of `record_prefix`/
`final_cleanup` are ported as faithful STRUCTURE that raises
`NotImplementedError` (via `_raise_enum_bscore_missing`) at the exact
point classic constructs the missing type, rather than silently doing
nothing -- this port's established "port the gate, raise inside it"
convention (see e.g. `ss_bscore.py`, `scoring_base.py`'s
`bscore_ctable_base.ignore_rows`/`ignore_rows_at_times`).

## What IS ported and tested, needing neither missing dependency

Every method below that does not itself construct an enum bscore is
genuine, real logic operating on plain `combo_tree`/`CTable` structures,
and is fully exercised in `test_partial.py` without needing `solve()` to
ever run:

* `effective` -- does a predicate classify every row it fires on
  correctly? (Returns `(good_count, fail_count)` rather than C++'s two
  reference out-parameters -- this port's usual adaptation for that
  calling convention, observably identical.)
* `trim_table` -- remove every row a predicate fires on from a `CTable`.
  Returns `(deleted, total)`.
* `eval_candidate` -- picks out a `cond` tree's first "effective"
  predicate/conclusion pair and records it if it beats the best found so
  far. classic's own FIRST lines here (`behavioral_score bs = _bscore->
  operator()(cand); ... total_score += sc; logger().debug() << "...
  candidate=" << total_score ...`, partial.cc:242-248, flagged by its own
  "XXX is this correct?" comment) compute a value with NO reader besides
  that `logger().debug()` call -- this port's established "`logger().
  debug()` is pure logging, dropped outright" convention (hill_climbing.py,
  optimization.py) applies verbatim, and since `total_score` has no other
  use, dropping it also means `eval_candidate` never touches `_bscore` at
  all, so it needs neither missing dependency.
* `eval_candidates` -- evaluates every candidate in a batch, then calls
  `record_prefix` if any improved on the best found. `record_prefix`
  itself DOES end with a bscore-recreation call (see above) -- but that
  call is the LAST thing it does, after all its real, observable mutation
  (table trimming, `_leader`/`_prefix_count` update) is already complete;
  `test_partial.py` calls `record_prefix` directly, catches the resulting
  `NotImplementedError`, and asserts the mutations already happened.
* `refresh`/`final_cleanup` -- copy candidates out of a `metapopulation`
  into the next round's exemplar list, prefixing `_leader`'s own recorded
  clauses onto each one first (`final_cleanup` only). Same "raises only
  at the very end, after all real mutation" shape as `record_prefix`.
* `__call__`/`check_candidates` -- pure orchestration, no scorer at all.

`metapopulation`/`scored_combo_tree_set` iteration: classic's `metapop-
ulation` publicly inherits a `boost::ptr_set`-like ordered container and
is iterated directly (`for (auto &item : cands)`); this port's
`metapopulation` (metapopulation.py, outside this task's owned tree) does
not implement `__iter__` itself -- its own `get_trees()` returns the
`scored_combo_tree_ptr_set` that IS the thing classic iterates, so every
`for item in cands` below where `cands: metapopulation` becomes `for item
in cands.get_trees()`. `scored_combo_tree_set` (the type `eval_candidates`/
`check_candidates` actually receive) DOES support `__iter__` directly, so
no adaptation is needed there.

`_reduct` (classic: `const rule&`, called as `_reduct(cand)`, a single
`combo_tree`-mutating functor used for BOTH `si_ca`/`si_kb` in the
`metapop_moses_results` call) is a `Simplifier`
(`pymetta.moses.deme.deme_expander.Simplifier`,
`Callable[[combo_tree], None]`) here, matching that module's own
established type for exactly this role.

`ps_params` is accepted and stored by the constructor, matching
classic's own signature (partial.h:49, partial.cc:54), and is not
threaded any further from HERE -- but that is now purely because
`solve()`'s own `metapop_moses_results` call sites (partial.cc:106,122)
are unreachable in this port: every path into them raises on the
un-ported enum bscores first (see `_ENUM_BSCORE_MISSING` below).
CORRECTED: this paragraph used to say the reason was that
`metapop_moses_results` "has no `ps_params` parameter in its own PORTED
signature at all". That was true when written and is false now --
`moses_main.py` threads `ps_params` in C++'s own position, so whenever
`solve()` becomes reachable the two calls it makes must pass
`self._ps_params` between `self._hc_params` and `self._deme_params`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_enum_type, vertex_equal
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.deme.deme_expander import Simplifier, deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.moses_main import Printer
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.optimization.particle_swarm import ps_parameters
from pymetta.moses.moses.types import scored_combo_tree_set
from pymetta.moses.scoring.scoring_base import _copy_ctable

__all__ = ["partial_solver"]

_ENUM_BSCORE_MISSING = (
    "partial_solver: enum_effective_bscore/enum_table_bscore/"
    "enum_graded_bscore (moses/moses/scoring/bscores.h:409,490,516) are "
    "not ported anywhere in this project's bscores.py -- the whole "
    "'enum_type'-output scoring subsystem partial_solver depends on is a "
    "pre-existing gap outside this task's owned tree. See partial.py's "
    "own module docstring for the full account, including a second, "
    "independent blocker in metapopulation/merging.py's merge_callback gate."
)


def _raise_enum_bscore_missing() -> None:
    """Not a ported C++ symbol -- the single raise point every BScore/
    StraightBScore (re)construction site below funnels through, so the
    message is written once and every call site's docstring can point at
    it instead of repeating the explanation.
    """
    raise NotImplementedError(_ENUM_BSCORE_MISSING)


def _copy_combo_tree(t: combo_tree) -> combo_tree:
    """Not a ported C++ symbol -- stands in for a real `combo_tree` VALUE
    copy (`combo_tree cand = item.get_tree();`, `_exemplars.push_back(...)`
    -- both copy-construct in C++). `tree.__init__`'s `isinstance(root_data,
    tree_node)` branch (tree.py) already deep-copies a subtree rooted at a
    given node; this is just that, entered via the tree's own root.
    """
    return combo_tree(t.begin())


class partial_solver:
    """C++: moses/moses/moses/partial.h:39 partial_solver

    See this module's own docstring for what is and is not reachable, and
    why. `_bscore`/`_cscore`/`_straight_bscore`/`_straight_cscore` start
    `None` (matching C++'s `NULL` initializers, partial.cc:59-62) and stay
    `None` forever in this port -- nothing here can construct them.
    """

    def __init__(
        self,
        ctable: CTable,
        exemplars: Sequence[combo_tree],
        reduct: Simplifier,
        opt_params: optim_parameters,
        hc_params: hc_parameters,
        ps_params: ps_parameters,
        deme_params: deme_parameters,
        filter_params: subsample_deme_filter_parameters,
        meta_params: metapop_parameters,
        moses_params: moses_parameters,
        printer: Printer,
    ) -> None:
        """C++: moses/moses/moses/partial.h:41 partial_solver::partial_solver

        `_ctable(ctable)` is a real VALUE COPY in C++ (a by-value member,
        copy-constructed from the `const CTable&` parameter) -- reproduced
        via `_copy_ctable` (scoring_base.py), not a bare assignment, which
        would alias `self._ctable` with `self._orig_ctable`/the caller's
        own table and let `trim_table` corrupt it (docs/CARRY-FORWARD.md
        Sec 1.2's own aliasing concern, applied here). `_orig_ctable`
        (`const CTable&`, a genuine alias in C++ too) is the CALLER's
        table, unmutated by this class -- stored for structural fidelity;
        classic itself never reads it again after the constructor (`grep
        -n "_orig_ctable" partial.cc` finds only this one assignment).

        `_leader(id::cond)` (partial.h's own member-initializer list) is a
        single-node tree whose root is the `cond` builtin, no children --
        the growing prefix of proven-perfect `(predicate, conclusion)`
        pairs gets appended as `_leader`'s children over time
        (`record_prefix`).

        `_exemplars(exemplars)` is likewise a real value copy:
        `std::vector<combo_tree>`'s copy constructor copy-constructs EACH
        element via `combo_tree`'s own copy constructor, not merely the
        vector's own storage -- `list(exemplars)` alone would copy the
        Python list but ALIAS every tree inside it with the caller's own,
        letting `final_cleanup`'s `self._reduct(cand)` (an in-place
        mutation) or `record_prefix`'s prefix insertion corrupt the
        caller's originals. `_copy_combo_tree` below is used at every
        site this class populates `_exemplars` (here and in `refresh`).
        """
        self._ctable = _copy_ctable(ctable)
        self._orig_ctable = ctable
        self._table_type_signature: TypeSignature = ctable.get_signature()
        self._exemplars: list[combo_tree] = [_copy_combo_tree(t) for t in exemplars]
        self._leader = combo_tree(builtin.cond)
        self._prefix_count = 0
        self._reduct = reduct
        self._opt_params = opt_params
        self._hc_params = hc_params
        self._ps_params = ps_params
        self._deme_params = deme_params
        self._filter_params = filter_params
        self._meta_params = meta_params
        self._moses_params = moses_params
        self._printer = printer

        self._bscore = None
        self._cscore = None
        self._straight_bscore = None
        self._straight_cscore = None

        self._num_evals = 0
        self._num_gens = 0
        self._done = False

        self._most_good = 0
        self._best_predicate = None

    def solve(self) -> None:
        """C++: moses/moses/moses/partial.h:44 / partial.cc:74 partial_solver::solve

        BLOCKED: see this module's own docstring (reason 1 -- the enum
        bscore family this method's own first line needs does not exist
        in this port). Raises immediately, before any of the loop
        bookkeeping below could run, matching where classic's OWN first
        two statements sit (partial.cc:78-79).
        """
        _raise_enum_bscore_missing()

    def __call__(
        self, metapop: metapopulation, dex: deme_expander, stats: moses_statistics
    ) -> None:
        """C++: moses/moses/moses/partial.h:58 partial_solver::operator()

        The metapop "printer" callback (the class's own comment) --
        `dex` is accepted (matching classic's own signature and the
        `Printer` type alias, moses_main.py) but never read, exactly as
        classic's own body never reads its `deme_expander&` parameter
        either.

        "If _most_good is zero, then we are here because the
        metapopulation came to a natural end... Otherwise, restart the
        search where we left off." (the header's own comment, condensed.)
        """
        del dex
        self._num_evals = stats.n_evals

        if self._most_good == 0:
            self._done = True

        if self._moses_params.max_evals <= self._num_evals:
            self._done = True

        if self._done:
            self.final_cleanup(metapop)
        else:
            self.refresh(metapop)

    @staticmethod
    def check_candidates(cands: scored_combo_tree_set, ud: partial_solver) -> bool:
        """C++: moses/moses/moses/partial.h:83 partial_solver::check_candidates

        `void *ud` is `self` here directly (Python needs no `void*`/cast
        dance to pass an object through a callback's user-data slot) --
        matching this port's usual convention for a C `void*` callback
        parameter (contrast a case that genuinely needs type erasure,
        which this does not: the ONE caller, `solve()`, always passes
        `this`).
        """
        return ud.eval_candidates(cands)

    def eval_candidates(self, cands: scored_combo_tree_set) -> bool:
        """C++: moses/moses/moses/partial.h:88 / partial.cc:136 partial_solver::eval_candidates

        "Evaluate each of the candidate solutions, see if any of the
        leading predicates are 'good enough'. If we find one that is,
        then trim the scoring tables, and return (so as to run moses on
        the smaller problem)." (the header's own comment.)
        """
        self._most_good = 0
        for item in cands:
            self.eval_candidate(item.get_tree())

        if self._most_good == 0:
            return False

        self.record_prefix()
        return True

    def eval_candidate(self, cand: combo_tree) -> None:
        """C++: moses/moses/moses/partial.h:89 / partial.cc:225 partial_solver::eval_candidate

        See this module's own docstring: classic's own first block here
        (`_bscore->operator()(cand)` + a sum used only by a dropped
        `logger().debug()` call) is omitted outright, not merely gated --
        it needs `_bscore`, which never exists in this port, but its
        result is otherwise unobservable, so omitting it changes nothing
        this port can measure.

        "We don't want constants; what else we got?" (partial.cc:249's own
        comment) -- a bare `is_enum_type` root means `cand` is just the
        catch-all default conclusion with no more predicates to peel off.

        # DIVERGENCE(C++ moses/moses/moses/partial.cc:252): `OC_ASSERT(*it
        # == id::cond, ...)` becomes `AssertionError`, this port's
        # standing convention for a direct `OC_ASSERT(false, ...)`.

        "Yank out the first effective predicate... A predicate is
        effective if it makes at least one correct identification."
        (partial.cc:254-257's own comment.) Walks `cand`'s root's children
        two at a time (predicate, conclusion, predicate, conclusion, ...,
        default), skipping ineffective (zero good AND zero fail) leading
        predicates, until either a real verdict is reached or the
        remaining conclusion is itself a bare constant (meaning every
        predicate tried so far was ineffective).
        """
        root = cand.begin()
        if is_enum_type(root.data):
            return

        if not vertex_equal(root.data, builtin.cond):
            raise AssertionError("Error: unexpected candidate!")

        siblings = list(cand.children(root))
        idx = 0
        fail_count = 0
        good_count = 0
        predicate = siblings[idx]
        while True:
            if is_enum_type(siblings[idx].data):
                # All previous predicates were ineffective.
                return

            predicate = siblings[idx]
            idx += 2

            fail_count = 0
            good_count = 0
            good_count, fail_count = self.effective(predicate, good_count, fail_count)

            if good_count > 0 or fail_count > 0:
                break

        if fail_count:
            return

        if good_count <= self._most_good:
            return

        self._most_good = good_count
        self._best_predicate = predicate

    def effective(self, pred, good_count: int, fail_count: int) -> tuple[int, int]:
        """C++: moses/moses/moses/partial.h:92 / partial.cc:190 partial_solver::effective

        "Compute the effectiveness of the predicate. That is, return how
        many answers it got right, and how many got flat-out wrong." (the
        header's own comment.) `good_count`/`fail_count` are ACCUMULATED
        into (matching C++'s reference parameters, which classic's own
        caller resets to 0 immediately before each call -- see
        `eval_candidate` above): returns `(good_count + delta, fail_count
        + delta)`, not a fresh pair, so a caller relying on accumulation
        across calls (none in this port, but matching the real C++
        signature) gets the same behaviour.

        `conq = next(predicate)` / `consequent = *conq` (partial.cc:196-197)
        is `predicate.next_sibling` here -- the sibling immediately after
        the predicate, i.e. the conclusion this predicate implies when it
        fires.
        """
        conq = pred.next_sibling
        consequent = conq.data

        total_count = 0
        iv = interpreter_visitor(pred)
        for row, counter in self._ctable.items():
            pr = iv(list(row))
            total_count += sum(counter.values())
            if vertex_equal(pr, builtin.logical_true):
                num_right = counter.get(consequent, 0)
                num_total = sum(counter.values())
                if num_right != num_total:
                    fail_count += num_total - num_right
                else:
                    good_count += num_right
        return good_count, fail_count

    def trim_table(self, taby: CTable, predicate, deleted: int, total: int) -> tuple[int, int]:
        """C++: moses/moses/moses/partial.h:94 / partial.cc:213 partial_solver::trim_table

        "Remove all rows from the table that satisfy the predicate." (the
        header's own comment.) Returns `(deleted, total)`, accumulated
        into exactly like `effective` above -- classic's own caller
        (`record_prefix`) also resets both to 0 immediately before this
        call.

        Mutates `taby` IN PLACE (a real side effect, matching C++'s
        `taby.erase(cit++)`), iterating over a snapshot of its keys so
        deleting from the dict while iterating is safe -- `CTable.erase`
        (table.py) is not assumed to exist; a plain `del taby[row]` is
        used instead, since `taby` here is always a real `CTable`
        (`dict` subclass).
        """
        iv = interpreter_visitor(predicate)
        for row in list(taby.keys()):
            counter = taby[row]
            pr = iv(list(row))
            tc = sum(counter.values())
            total += tc
            if vertex_equal(pr, builtin.logical_true):
                deleted += tc
                del taby[row]
        return deleted, total

    def refresh(self, cands: metapopulation) -> None:
        """C++: moses/moses/moses/partial.h:97 / partial.cc:227 partial_solver::refresh

        "Refresh the exemplars list. Basically, just copy the entire
        metapopulation from the previous run." (the header's own comment.)
        See this module's own docstring for why `cands.get_trees()`
        stands in for classic's direct `for (auto &item : cands)`.

        `_exemplars.push_back(item.get_tree())` copy-constructs in C++
        (`std::vector<combo_tree>::push_back` always stores by value) --
        `_copy_combo_tree` reproduces that; a bare `.append(item.get_tree
        ())` would alias the metapopulation's own stored tree (see
        `__init__`'s own docstring for the same concern).
        """
        for item in cands.get_trees():
            self._exemplars.append(_copy_combo_tree(item.get_tree()))

    def final_cleanup(self, cands: metapopulation) -> None:
        """C++: moses/moses/moses/partial.h:99 / partial.cc:157 partial_solver::final_cleanup

        "We've run out of time. So assemble the best possible exemplars
        out of the pieces we've accumulated, and feed those back into the
        main algo as exemplars." (the header's own comment, condensed.)

        For each candidate, PREPENDS every `(predicate, conclusion)` pair
        already recorded in `_leader` onto the front of the candidate's
        own children, in `_leader`'s own order -- see this module's own
        docstring for how `cand.insert_subtree(anchor, lit)` reproduces
        classic's `cit = cand.insert_subtree(cit, lit); ++cit;` loop
        (`insert_subtree`'s new node always lands as `anchor`'s new
        PREVIOUS sibling and `anchor` itself never moves, so repeated
        inserts before the same anchor land in `_leader`'s own order,
        immediately before whatever `cand` already had -- verified
        directly in test_partial.py, not merely asserted).

        `anchor is None` (cand's root had NO children to begin with) has
        no direct C++ equivalent to fall back on (cogutil's own head/feet
        sentinels make "insert before the empty end" well-defined there;
        this port's node-based `tree` has no such sentinel to insert
        before) -- `append_child` is used instead for that case, which is
        observably identical to "insert into an empty children list" for
        every accessor this port's `tree` exposes.

        BLOCKED for the trailing two lines only (BScore/behave_cscore
        recreation, partial.cc:188-189) -- see this module's own
        docstring. Everything above that raise already ran and mutated
        `self._exemplars` by the time it fires.
        """
        self._exemplars = []
        leader_root = self._leader.begin()
        leader_children = list(self._leader.children(leader_root))
        for item in cands.get_trees():
            cand = _copy_combo_tree(item.get_tree())
            cand_root = cand.begin()
            cand_children = list(cand.children(cand_root))
            anchor = cand_children[0] if cand_children else None
            for lit in leader_children:
                if anchor is None:
                    cand.append_child(cand_root, lit)
                else:
                    cand.insert_subtree(anchor, lit)
            self._reduct(cand)
            self._exemplars.append(cand)

        _raise_enum_bscore_missing()

    def record_prefix(self) -> None:
        """C++: moses/moses/moses/partial.h:101 / partial.cc:266 partial_solver::record_prefix

        "The first predicate is correctly identifying all cases; we can
        save it, and remove all table elements that it correctly
        identified." (partial.cc:262-264's own comment, sitting just
        above this function.)

        Appends the best predicate and its conclusion as `_leader`'s two
        newest children, in that order -- `self._leader.append_child(...)`
        stands in for `_leader.insert_subtree(ldr.end(), prd)` (see this
        module's own docstring: `ldr.end()` IS the append position, and
        `append_child` already deep-copies a subtree source exactly like
        `insert_subtree` does, per `tree.py`'s own merged-overload
        convention).

        BLOCKED for the trailing bscore-recreation lines only
        (partial.cc:283-286) -- see this module's own docstring.
        Everything above that raise (table trimming, `_leader`/
        `_prefix_count` update) already ran and mutated real state by the
        time it fires.
        """
        deleted, total_rows = self.trim_table(self._ctable, self._best_predicate, 0, 0)
        del deleted, total_rows  # only ever used by a dropped logger().info() call

        self._prefix_count += 1
        leader_root = self._leader.begin()
        predicate = self._best_predicate
        self._leader.append_child(leader_root, predicate)
        self._leader.append_child(leader_root, predicate.next_sibling)

        _raise_enum_bscore_missing()
