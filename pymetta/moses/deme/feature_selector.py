"""C++: moses/moses/deme/feature_selector.{h,cc}

``feature_selector`` -- the in-loop ``--enable-fs`` entry point. Given an
exemplar ``combo_tree``, restricts (and optionally re-shapes) the scoring
``CTable`` around that exemplar's own predictions, then runs
``pymetta.feature_selection.main.feature_selection.select_feature_sets``
(all five of ``simple``, ``inc``, ``smd``, ``random`` and ``hc`` are ported
there) to pick a feature subset "good
when combined with the exemplar" (the class's own C++ doc comment,
feature_selector.h:180-181).

Scope narrowing vs. the 233+477-line C++ original:

* ``log_stats_top_feature_sets`` (feature_selector.cc:250-293) is ported
  down to its early return: the "Number of demes selected" record, then
  ``if (1 == top_fs.size()) return;``. Everything past that return is a
  Boost accumulator over pairwise ``mi``/``jaccard_index``, fed straight
  to two ``logger().info()`` calls and observed by nothing -- and it is
  UNREACHABLE in any run that survives, because ``top_fs`` larger than one
  needs ``--fs-demes > 1``, which trips
  ``deme_expander::log_selected_feature_sets``' own ``OC_ASSERT``
  (deme_expander.cc:63) a few statements later. See that method's port.
* ``remove_useless_features`` (feature_selector.h:211,
  feature_selector.cc:295-309) is declared and defined but has ZERO call
  sites anywhere in classic (confirmed by grep across the whole ``moses/``
  tree) -- genuinely dead code in C++ itself, not merely out of this
  port's scope. NOT ported, matching this project's "omit dead code,
  don't add a placeholder" discipline (``deme_expander.py``'s treatment of
  ``fs_to_names``/``prune_xmplr`` is the same call, for a different
  reason -- those two DO have callers, but only inside a gated branch).
* ``rank_feature_sets``/``mi`` (feature_selector.cc:343/:434) ARE ported
  now, together with the ranked branch of ``select_top_feature_sets``
  (cc:238-244) they are the only callers of -- ``--fs-diversity-pressure
  > 0`` used to raise here. They need ``composite_score``
  (``pymetta.moses.moses.types``), ``jaccard_index``
  (``opencog/util/jaccard_index.h``) and ``powerset``
  (``opencog/util/algorithm.h``); the last two are transliterated in this
  module (they are five-line free functions with no other call site in
  this port's scope, the same call ``get_argument_abs_idx_from_zero_set``
  below already makes).
* ``--fs-subsampling-ratio < 1.0`` is ported in BOTH its branches
  (cc:206-218). The by-ROW one calls ``subsampleCTable``, which
  ``pymetta.feature_selection.ctable_subsample`` holds -- out of
  ``pymetta.comboreduct.table`` for the placement reason that module's
  own docstring gives. The by-TIME one
  (``--fs-subsampling-by-time``) is transliterated inline, over
  ``CTable.get_timestamps()``/``remove_rows_at_times()`` and the same
  ``rand_element_erase`` shape ``ss_bscore._partition_by_time`` uses.
  Note the two branches read ``subsampling_ratio`` at DIFFERENT
  precisions -- see ``build_fs_ctable``.
"""

from __future__ import annotations

import copy
import math

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_argument, vertex
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, Table, get_indices
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.feature_selection.ctable_subsample import subsampleCTable
from pymetta.feature_selection.main.feature_selection import select_feature_sets
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymetta.feature_selection.scorers.mutual_info import mutual_information_between_sets
from pymetta.moses.moses.types import _to_score_t, composite_score
from pymetta.util.logger import logger
from pymetta.util.rng import randGen

__all__ = [
    "EXEMPLAR_FEATURE_NAME",
    "csc_feature_set_pop",
    "feature_selector",
    "feature_selector_parameters",
    "get_argument_abs_idx_from_zero_set",
]

# C++: moses/moses/deme/feature_selector.h:170 csc_feature_set_pop
#
# `typedef std::multimap<composite_score, feature_set,
# std::greater<composite_score>> csc_feature_set_pop;` -- "used for
# diversity ranking". Modelled, exactly as `feature_set_pop` is
# (feature_selection_params.py's own type docstring), as a list whose
# order IS the multimap's: `rank_feature_sets` appends each element as it
# is chosen, and each chosen element's key is <= the previous one's, so
# appending and inserting into the multimap produce the same sequence --
# including for equal keys, which C++11 places at the END of their
# equivalent range.
csc_feature_set_pop = list[tuple[composite_score, feature_set]]

# C++: moses/moses/deme/feature_selector.cc:43 EXEMPLAR_FEATURE_NAME
EXEMPLAR_FEATURE_NAME = "__exemplar_feature__"


def get_argument_abs_idx_from_zero_set(tr: combo_tree) -> set[int]:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1124
    get_argument_abs_idx_from_zero_set

    Ported HERE, not in ``pymetta.comboreduct``, for the same reason
    ``mutual_info.py`` ports the ``CTable`` entropy code here: it is a
    tiny (5-line) free function this file's own scope needs, and editing
    ``pymetta/comboreduct/`` is out of bounds for this task. "Get the set
    of features of the exemplar" (feature_selector.cc:69's own comment at
    its one real call site).
    """
    result: set[int] = set()
    for node in tr.iter_leaf():
        if is_argument(node.data):
            result.add(node.data.abs_idx_from_zero())
    return result


class feature_selector_parameters:
    """C++: moses/moses/deme/feature_selector.h:41 feature_selector_parameters

    Every field mirrors the C++ struct's own name and default. Two
    primitive (``bool``) fields are OMITTED from C++'s own constructor
    initializer list (feature_selector.h:56-68) -- genuinely
    INDETERMINATE on a fresh ``feature_selector_parameters()`` in C++,
    the same "port cannot leave a field uninitialized" situation
    ``deme_params.py`` documents for ``deme_parameters.perm_ratio``.
    Pinned here to the value every real CLI caller
    (``problem-params.cc``, the only caller that ever constructs a real
    ``feature_selector``) actually sets via
    ``po::value<bool>(...)->default_value(...)`` before ``notify()``
    returns -- DIVERGENCE from the bare struct ctor, not from observed
    real-world behaviour:

    * ``prune_xmplr``: uninitialized in the struct; CLI default ``False``
      (``--fs-prune-exemplar``, problem-params.cc:1663-1665,
      ``->default_value(0)``).
    * ``diversity_jaccard``: uninitialized in the struct; CLI default
      ``True`` (``--fs-diversity-jaccard``, problem-params.cc:1809-1811,
      ``->default_value(true)``).

    classic's own comment at the one call site that assigns these structs
    from scratch (problem-params.cc:2189-2192) reads: "Please don't
    systematically use their constructors (if you can avoid it), as it
    is prone to error, whenever the constructor changes it may silently
    set the wrong things" -- i.e. classic's own authors flag exactly this
    hazard, independently of this port.
    """

    def __init__(
        self,
        fs_params: feature_selection_parameters | None = None,
        increase_target_size: bool = True,
        ignore_xmplr_features: bool = True,
        restrict_incorrect: bool = True,
        restrict_true: bool = False,
        prune_xmplr: bool = False,
        init_xmplr_features: bool = False,
        xmplr_as_feature: bool = False,
        subsampling_ratio: float = 1.0,
        subsampling_by_time: bool = False,
        n_demes: int = 1,
        diversity_pressure: float = 0.0,
        diversity_cap: int = 0,
        diversity_interaction: int = 0,
        diversity_jaccard: bool = True,
        enforce_features: dict[str, float] | None = None,
    ) -> None:
        self.fs_params = fs_params if fs_params is not None else feature_selection_parameters()
        self.increase_target_size = increase_target_size
        self.ignore_xmplr_features = ignore_xmplr_features
        self.ignored_features: feature_set = feature_set()
        self.restrict_incorrect = restrict_incorrect
        self.restrict_true = restrict_true
        self.prune_xmplr = prune_xmplr
        self.init_xmplr_features = init_xmplr_features
        self.xmplr_as_feature = xmplr_as_feature
        self.subsampling_ratio = subsampling_ratio
        self.subsampling_by_time = subsampling_by_time
        self.n_demes = n_demes
        self.diversity_pressure = diversity_pressure
        self.diversity_cap = diversity_cap
        self.diversity_interaction = diversity_interaction
        self.diversity_jaccard = diversity_jaccard
        # C++: feature_selector.h:166 is `std::map<std::string,float>`, so
        # the map NARROWS every weight to binary32 on the way in -- and the
        # value's only reader, `biased_randbool(float b)`
        # ($COGUTIL/opencog/util/random.h:127), compares it against a
        # `float` `randfloat()`. Narrowing here, at the store, is where C++
        # narrows; problem-params.cc:2248's own `float weight = stof(...)`
        # is the same narrowing on the CLI path and is already reproduced
        # in `problem_params._parse_enforce_features`, so this is
        # idempotent there and load-bearing only for a direct Python
        # caller, which would otherwise compare a double against a float32.
        self.enforce_features: dict[str, float] = (
            {name: _to_score_t(weight) for name, weight in enforce_features.items()}
            if enforce_features is not None
            else {}
        )


def _timed_counter_mode(counter) -> vertex:
    """C++: opencog/util/Counter.h:111 Counter::mode (via
    moses/comboreduct/table/table.cc:623 TimedCounter::mode, which just
    calls ``untimedCounter().mode()`` -- this port's ``CTable`` counters
    are already untimed, see ``pymetta.comboreduct.table.table.CTable``'s
    own module docstring, so there is no separate "untime" step here).

    DO NOT FIX -- ``Counter::mode()``'s C++ body is a real bug, ported
    verbatim:

    .. code-block:: c++

        T mode() const {
            T key = super::begin()->first;   // map iterates ascending by key
            CT cnt = super::begin()->second;
            for (const auto& v : *this) {
                if (cnt < v.second)
                    key = v.first;
            }                                 // `cnt` is NEVER reassigned
            return key;
        }

    ``cnt`` is captured once (the first key's count, in ascending-key
    order) and never updated inside the loop -- so the result is NOT the
    true mode. It is the LAST entry (in ascending-key iteration order)
    whose count strictly exceeds the FIRST entry's count; if no entry
    exceeds the first entry's count, the first entry's own key is
    returned. On the two fixture tables this port ships
    (``trainingBoolean.dat``/``trainingBooleanXor.dat``, both with all-
    unique input rows) every compressed ``CTable`` row's counter has
    exactly one entry, so this quirk is never actually exercised by this
    port's own oracle tests -- it is reproduced for fidelity against any
    OTHER dataset with genuinely conflicting rows (duplicate inputs,
    different outputs), not validated against a live classic run.
    """
    items = sorted(counter.items(), key=lambda kv: kv[0])
    key = items[0][0]
    cnt = items[0][1]
    for k, v in items:
        if cnt < v:
            key = k
    return key


def jaccard_index(s1, s2) -> float:
    """C++: opencog/util/jaccard_index.h:35 jaccard_index

    ``(float)set_intersection(s1, s2).size() / (float)set_union(s1,
    s2).size()`` -- both operands are C ``float``s and so is the quotient,
    so the result is narrowed to binary32 (``_to_score_t``, the same
    binary32 rounding; this value is a plain C ``float``, not a
    ``score_t``, but the conversion is identical). Transliterated here
    rather than in ``pymetta.util``: it is a five-line templated free
    function whose ONLY call site anywhere under ``moses/`` is
    ``feature_selector::rank_feature_sets`` (feature_selector.cc:386).

    # DIVERGENCE(C++ opencog/util/jaccard_index.h:37): two EMPTY sets give
    # ``0.0f / 0.0f`` -- IEEE-754 NaN in C++, where Python's ``/`` raises
    # ZeroDivisionError. Returned as ``nan`` so the C++ value survives.
    # Reachable: ``--fs-target-size=0`` makes ``simple_select_feature_sets``
    # return the empty feature set (simple.cc:44-49).
    """
    left, right = set(s1), set(s2)
    union_size = len(left | right)
    if union_size == 0:
        return math.nan
    return _to_score_t(len(left & right) / union_size)


def powerset(s, n: int, exact: bool = False) -> list[frozenset[int]]:
    """C++: opencog/util/algorithm.h:282 powerset

    "Return the power set ps of s such that all elements of ps are subsets
    of size n or below ... @param exact if true then do not include
    subsets of size below n" (algorithm.h:271-278's own doc comment).

    C++ returns a ``std::set<Set>``, so its ITERATION order is ascending
    lexicographic over each subset's own ascending elements -- and that
    order decides the summation order in ``feature_selector.mi``, where
    ``double`` addition is not associative. Reproduced by sorting on
    ``tuple(sorted(subset))`` on the way out; the recursion itself uses a
    ``set``, exactly as C++ does, so duplicates collapse the same way.

    ``exact=True`` with ``n`` GREATER than ``len(s)`` returns an EMPTY
    list: the recursion's innermost loop finds no element outside a
    full-size subset, and ``exact`` suppresses the smaller ones. That is
    C++'s behaviour too, and ``feature_selector.mi`` divides by the
    product of the two powerset sizes -- see that method for what happens
    then.
    """
    res: set[frozenset[int]] = set()
    if n > 0:
        ps = powerset(s, n - 1, exact)
        for ss in ps:
            # `for (const auto& el : s)` -- `s` is a `std::set<arity_t>`,
            # ascending. Irrelevant to the RESULT (a set), kept anyway.
            for el in sorted(s):
                if el not in ss:
                    res.add(ss | {el})
        if not exact:
            res.update(ps)
    else:
        res.add(frozenset())
    return sorted(res, key=lambda fs: tuple(sorted(fs)))


def _merge_timed_counter(dst, src) -> None:
    """C++: opencog/util/Counter.h:80 Counter

    Cites the enclosing class; the real member is
    ``Counter::operator+=(const Counter&)`` (Counter.h:80-85), which
    ``build_fs_ctable``'s ``fs_ctable[inputs] += vct.second``
    (feature_selector.cc:202) invokes. ``TimedCounter`` is a
    ``Counter<TimedValue, count_t>``, so C++ merges per
    ``(value, timestamp)`` OBSERVATION, not per value.

    This port's ``CTable`` cell exposes those observations through
    ``timed_items()`` and accumulates them through ``add()``
    (``pymetta.comboreduct.table.table.TimedCounter``); a cell shaped like
    a plain vertex-keyed ``Counter`` -- which is what this port's cells
    were before that class existed -- falls back to a per-value merge,
    which is the same thing with the timestamp half already collapsed.
    """
    if hasattr(src, "timed_items") and hasattr(dst, "add"):
        for tv, count in src.timed_items():
            dst.add(tv.value, tv.timestamp, count)
        return
    for value, count in src.items():
        dst[value] = dst.get(value, 0.0) + count


class feature_selector:
    """C++: moses/moses/deme/feature_selector.h:178 feature_selector"""

    def __init__(
        self,
        ctable_or_table: CTable | Table,
        festor_params: feature_selector_parameters | None = None,
    ) -> None:
        """C++: moses/moses/deme/feature_selector.cc:49
        feature_selector::feature_selector(const CTable&, ...) and
        moses/moses/deme/feature_selector.cc:53
        feature_selector::feature_selector(const Table&, ...)

        Python has no overload resolution; both C++ constructors collapse
        onto this one, dispatching on the argument's own type -- a
        ``Table`` is compressed (``table.compressed()``, matching the C++
        overload's own member-initializer ``_ctable(table.compressed())``
        exactly) before being stored.
        """
        # `feature_selector_parameters params;` (feature_selector.h:196) is
        # a BY-VALUE member, so `params(festor_params)` COPIES -- including
        # its own by-value `feature_selection_parameters fs_params`
        # (h:55), `ignored_features` and `enforce_features`. Storing the
        # caller's object instead would let `preprocess_params`' four
        # in-place mutations (`fs_params.target_size +=`,
        # `initial_features.extend`, `ignored_features = ...`) reach back
        # into whatever built it -- for the real CLI path, straight into
        # `problem_params.festor_params`, and through its reference-alias
        # `problem_params.fs_params` too. `deme_expander.
        # create_representations` deep-copies again before calling
        # (deme_expander.cc:182's own "Copy, as any change in the
        # parameters will not be remembered"), which is what kept this
        # inert; the copy belongs at both, as it is in C++.
        self.params = copy.deepcopy(
            festor_params if festor_params is not None else feature_selector_parameters()
        )
        self._ctable: CTable = (
            ctable_or_table.compressed() if isinstance(ctable_or_table, Table) else ctable_or_table
        )

    # ------------------------------------------------------------ helpers

    def preprocess_params(self, xmplr: combo_tree) -> None:
        """C++: moses/moses/deme/feature_selector.cc:61 feature_selector::preprocess_params

        "Do some sanity checking and mangling on the parameters." Mutates
        ``self.params`` in place -- matching C++'s own comment that this
        is safe because the enclosing struct is passed BY COPY at the one
        real call site (``deme_expander::create_representations``, which
        copies ``festor = *_params.fstor`` before calling it -- see that
        method's own port for where this copy happens in Python).
        """
        xmplr_features = get_argument_abs_idx_from_zero_set(xmplr)
        ilabels = self._ctable.get_input_labels()
        # Ascending order: `xmplr_features` is a `std::set<arity_t>` in
        # C++ (`for (arity_t i : xmplr_features)`, feature_selector.cc:74).
        xmplr_feature_names = [ilabels[i] for i in sorted(xmplr_features)]

        if self.params.init_xmplr_features:
            self.params.fs_params.initial_features.extend(xmplr_feature_names)
            self.params.increase_target_size = True

        if self.params.increase_target_size:
            self.params.fs_params.target_size += len(xmplr_features)

        self.params.ignored_features = (
            set(xmplr_features) if self.params.ignore_xmplr_features else set()
        )

        if self.params.xmplr_as_feature:
            self.params.fs_params.initial_features.append(EXEMPLAR_FEATURE_NAME)
            self.params.fs_params.target_size += 1

    def build_fs_ctable(self, xmplr: combo_tree) -> CTable:
        """C++: moses/moses/deme/feature_selector.cc:123 feature_selector::build_fs_ctable

        Builds a (possibly different, possibly smaller) ``CTable`` used
        for feature selection: filters rows by ``restrict_true``/
        ``restrict_incorrect`` against the exemplar's own predictions,
        blanks (rather than drops) ignored-feature columns
        (``init_at_visitor``, table.h:135-143 -- ``seq[_pos] = vt()``, the
        DEFAULT-constructed element -- ``builtin()`` value-initializes to
        ``0`` == ``id::null_vertex``, vertex.h:69-71, NOT
        ``logical_false``), and optionally appends the exemplar's own
        prediction as a synthetic extra feature (``xmplr_as_feature``).

        Scope: boolean output only (this port's own tables never carry
        anything else, per ``pymetta.moses.representation.representation``'s
        Boolean-only scope) -- a non-boolean ``cto`` raises.

        ``subsampling_ratio < 1.0`` (cc:206-218) runs in both its
        branches. The typing differs between them and it matters:
        ``subsampling_ratio`` is a ``double`` (feature_selector.h:120) and
        the by-TIME branch keeps it in ``double``, while the by-ROW branch
        passes it to ``subsampleCTable(float ratio, ...)``, which narrows
        it to binary32 first.

        MEASURED, and the reason the by-time branch is ported rather than
        gated: on a table with no ``--timestamp-feature``,
        ``get_timestamps()`` is empty, so ``rm_size`` is 0 and classic
        does NOTHING and exits 0. `moses -i bool_a5_maj.csv -u y -m 300
        -r 1 --enable-fs=1 --fs-subsampling-ratio=0.5
        --fs-subsampling-by-time=1` returns the same best candidate as the
        plain `--enable-fs=1` run (`-8 or(and($x2 $x3) $x1)`), where the
        by-ROW run returns a different one (`-6 or(and($x2 $x4) and($x3
        $x5))`). A raise here would therefore have diverged from classic
        on a reachable CLI combination.
        """
        sig = self._ctable.get_signature()
        # C++: `auto cto = get_type_node(get_signature_output(sig));`
        # (feature_selector.cc:125). NOT a gate: classic branches on this
        # only inside `xmplr_as_feature` (cc:190-196), where the third
        # arm is its own `OC_ASSERT(false, "Not implemented")`. This port
        # used to reject every non-boolean `cto` right here, which made
        # `-Hit` on a contin table and `-Hcluster` refuse a command line
        # classic runs -- both go through `REGRESSION(table, ...)`
        # (table-problems.cc:519,522,557), i.e. the `Table` constructor
        # overload, and both were measured failing against classic's own
        # binary before the gate moved down to cc:196 where it belongs.
        cto = sig.otype if sig is not None else type_node.unknown_type

        olabel = self._ctable.get_labels()[0]
        ilabels = list(self._ctable.get_input_labels())
        itypes = list(sig.itypes) if sig is not None else []
        if self.params.xmplr_as_feature:
            logger().debug("Append exemplar feature")  # C++: feature_selector.cc:134
            ilabels.append(EXEMPLAR_FEATURE_NAME)
            itypes.append(cto)

        new_sig = TypeSignature(itypes, cto) if sig is not None else None
        fs_ctable = CTable(olabel, ilabels, new_sig)

        # C++: `interpreter_visitor iv(xmplr); auto ai =
        # boost::apply_visitor(iv);` (feature_selector.cc:139-140),
        # constructed ONCE, before the per-row loop. `interpreter_visitor`
        # -- not `boolean_interpreter`: it picks `mixed_interpreter` when
        # the exemplar holds a contin expression or `greater_than_zero`,
        # and when the row itself is contin-valued (table.h:304-341). The
        # port pinned `boolean_interpreter` here until 2026-09-10, which
        # was wrong for exactly the two problems that reach this with a
        # non-boolean table.
        iv = interpreter_visitor(xmplr)
        ignored_sorted = sorted(self.params.ignored_features)
        if self.params.ignored_features:
            # C++: feature_selector.cc:144-145 -- `ostream_container(...,
            # params.ignored_features)` with the default " " delimiter, so
            # a space-separated list after the prefix.
            logger().debug("Ignore features: " + " ".join(str(i) for i in ignored_sorted))

        for row, counter in self._ctable.items():
            predicted_out: vertex = iv(list(row))

            consider_row = True
            if self.params.restrict_true:
                consider_row = predicted_out == builtin.logical_true
            if consider_row and self.params.restrict_incorrect:
                actual_out = _timed_counter_mode(counter)
                consider_row = predicted_out != actual_out

            if consider_row:
                inputs = list(row)
                if self.params.ignore_xmplr_features:
                    for i in ignored_sorted:
                        inputs[i] = builtin.null_vertex  # init_at_visitor's `vt()`
                if self.params.xmplr_as_feature:
                    # C++: feature_selector.cc:190-196. `get_builtin`/
                    # `get_contin` are `boost::get<>` on the vertex
                    # variant -- the identity in this port, where a
                    # `vertex` for which `is_builtin`/`is_contin` holds
                    # simply IS that value -- but the third arm is a real
                    # `OC_ASSERT(false, "Not implemented")` and is
                    # reproduced as one. This is the ONLY place in
                    # `build_fs_ctable` where the output type matters.
                    if cto is not type_node.boolean_type and cto is not type_node.contin_type:
                        raise NotImplementedError(
                            "feature_selector.build_fs_ctable: "
                            "--fs-seed=xmplr on a table whose output type is "
                            f"{cto!r} -- classic's own OC_ASSERT(false, "
                            '"Not implemented") (feature_selector.cc:196)'
                        )
                    inputs.append(predicted_out)

                key = tuple(inputs)
                # `fs_ctable[inputs] += vct.second;` (feature_selector.cc:202)
                dst = fs_ctable.setdefault(key, type(counter)())
                _merge_timed_counter(dst, counter)

        # subsample (feature_selector.cc:205-218's own comment). Runs
        # BEFORE the size log below, as in C++ -- the logged size is the
        # POST-subsampling one.
        if self.params.subsampling_ratio < 1.0:
            if self.params.subsampling_by_time:
                # `std::set<TTable::value_type> timestamps =
                # fs_ctable.get_timestamps(), rm_timestamps;`
                #
                # NOTE the typing asymmetry with the by-row branch below:
                # `params.subsampling_ratio` is a `double`
                # (feature_selector.h:120) and STAYS a double here, where
                # `subsampleCTable(float ratio, ...)` narrows it to
                # binary32 at the call. Same field, two precisions,
                # depending on the branch.
                timestamps = fs_ctable.get_timestamps()
                rm_timestamps: set = set()
                timestamps_size = len(timestamps)
                rm_size = int((1.0 - self.params.subsampling_ratio) * timestamps_size)
                # `rand_element_erase(timestamps)` (opencog/util/random.h:92)
                # is `std::next(c.begin(), rng.randint(c.size()))` then
                # erase -- a `std::set` iterates ASCENDING, so this draws a
                # uniformly random SORTED-POSITION index each round. Exactly
                # the transliteration ss_bscore._partition_by_time already
                # carries, over the same container type; `get_timestamps()`
                # never returns the not-a-date sentinel, so a plain
                # `sorted()` is `timestamp_less`'s order here.
                remaining = sorted(timestamps)
                for _ in range(rm_size):
                    rm_timestamps.add(remaining.pop(randGen().randint_n(len(remaining))))
                fs_ctable.remove_rows_at_times(rm_timestamps)
            else:
                subsampleCTable(self.params.subsampling_ratio, fs_ctable)

        # C++: feature_selector.cc:220-221. `%u` on a `size_t` is classic's
        # own format/type mismatch and prints the same digits.
        logger().debug("CTable size for feature selection = %u", len(fs_ctable))

        return fs_ctable

    def select_top_feature_sets(self, fss: feature_set_pop) -> feature_set_pop:
        """C++: moses/moses/deme/feature_selector.cc:213
        feature_selector::select_top_feature_sets

        Takes the first ``min(n_demes, len(fss))`` entries of ``fss``
        (already sorted descending by score -- see
        ``feature_selection_params.py``'s own ``feature_set_pop`` type
        docstring for why a plain slice reproduces a
        ``std::multimap``'s ``begin()``-relative range here).

        ``diversity_pressure > 0.0`` (struct default ``0.0``, off) selects
        a DIFFERENT branch: rank by diversity-penalized score first
        (``rank_feature_sets``), then take the first ``res_size`` of THAT
        order, re-keyed on each one's ``get_penalized_score()``.

        The C++ result is a fresh ``feature_set_pop``, i.e. a
        ``std::multimap<double, feature_set, std::greater<double>>`` built
        by inserting the ranked entries in ranked order -- so equal
        penalized scores keep ranked order (C++11 inserts an equivalent
        key at the END of its equivalent range). Reproduced by a STABLE
        descending sort, which is a no-op on this input (the ranked order
        is already non-increasing in penalized score, since
        ``composite_score``'s own ``operator<`` compares penalized score
        first) but is written out because that no-op is a PROPERTY of the
        input, not of the code.

        ``res_size`` is computed BEFORE the branch in C++ and is
        ``min(n_demes, fss.size())`` either way -- note it counts the
        INPUT's size, so it is never larger than ``ranked_fss``.
        """
        res_size = min(self.params.n_demes, len(fss))
        if self.params.diversity_pressure > 0.0:
            ranked_fss = self.rank_feature_sets(fss)
            res: feature_set_pop = [
                (cs.get_penalized_score(), fs) for cs, fs in ranked_fss[:res_size]
            ]
            res.sort(key=lambda entry: entry[0], reverse=True)
            return res
        return fss[:res_size]

    def rank_feature_sets(self, fs_pop: feature_set_pop) -> csc_feature_set_pop:
        """C++: moses/moses/deme/feature_selector.cc:343
        feature_selector::rank_feature_sets

        "Rank feature sets, penalized by diversity" (feature_selector.h:213).
        Greedy: repeatedly take the best remaining ``composite_score``, move
        it to the result, then re-penalize every survivor against the one
        just taken.

        Aggregation is ``max`` over the pairwise penalties, which is why
        only the LAST inserted feature set has to be compared against --
        C++'s own comment at cc:352-357 says exactly this.

        Five things decide the output and are each reproduced verbatim:

        * ``composite_score(fs.first, fs.second.size())`` -- the multimap
          key (a ``double``) narrows to ``score_t`` (binary32) inside
          ``composite_score``, and the complexity is the feature set's own
          SIZE. Measured: a raw score of ``0.9`` comes back out as
          ``0.89999997615814209``.
        * ``float sim`` -- ``jaccard_index`` already returns a C ``float``;
          ``mi()`` returns a ``double`` and narrows HERE.
        * ``float last_dp = params.diversity_pressure * sim`` -- a
          ``double`` product (``sim`` promotes) narrowed to ``float``.
        * ``std::max_element`` returns the FIRST maximum, so ties keep the
          earlier position. Written as an explicit loop with a strict
          ``<``, not ``max(...)``, so that is unmistakable.
        * ``res.insert(*mit)`` into a ``std::multimap<composite_score,
          feature_set, std::greater<composite_score>>``. Appending
          reproduces it -- see ``csc_feature_set_pop``'s own note above.

        ``composite_score``'s ``operator<`` breaks a tie on equal penalized
        scores by preferring the SMALLER complexity (``lef == rig and
        self.complexity > other.complexity`` makes the bigger-complexity
        score the lesser), so among equally-penalized feature sets the
        smaller one is ranked first. That is C++'s behaviour, not a choice
        made here.

        The C++ body wraps the re-penalization in ``OMP_ALGO::for_each``
        and the pick in ``OMP_ALGO::max_element``. ``OMP_ALGO`` is
        ``__gnu_parallel`` only when ``OC_OMP`` is defined, which needs
        ``HAVE_PARALLEL_STL``; that macro is added by CogUtil's OWN
        ``ADD_DEFINITIONS`` (CogUtil/CMakeLists.txt:157), a
        directory-scoped command, and MOSES's build never defines it
        (``grep -rn HAVE_PARALLEL_STL`` over ``$CLASSIC`` returns
        nothing). So both are ``std::`` here, and the oracle probe -- a
        plain ``g++`` command with no ``-DHAVE_PARALLEL_STL`` -- matches
        the real library.
        """
        # `csc_feature_set_pop res;` plus `last_fs_cit = res.end()`
        res: csc_feature_set_pop = []
        last_fs_cit: tuple[composite_score, feature_set] | None = None

        csc_fs_seq: list[tuple[composite_score, feature_set]] = [
            (composite_score(score, len(fs)), fs) for score, fs in fs_pop
        ]

        while csc_fs_seq:
            if last_fs_cit is not None:
                for csc_fs in csc_fs_seq:
                    # feature set similarity measure
                    sim = (
                        jaccard_index(last_fs_cit[1], csc_fs[1])
                        if self.params.diversity_jaccard
                        else _to_score_t(self.mi(last_fs_cit[1], csc_fs[1]))
                    )
                    # sim_to_penalty(sim): `params.diversity_pressure * sim`,
                    # a double product assigned to a `float`.
                    last_dp = _to_score_t(self.params.diversity_pressure * sim)
                    # aggregate the results (here max)
                    agg_dp = max(csc_fs[0].get_uniformity_penalty(), last_dp)
                    # compute and update the uniformity penalty
                    csc_fs[0].set_uniformity_penalty(agg_dp)

            # insert the best candidate in res and delete it from csc_fs_pop.
            # `std::max_element` with `csc_fs_lt` keeps the FIRST maximum.
            mit = 0
            for idx in range(1, len(csc_fs_seq)):
                if csc_fs_seq[mit][0] < csc_fs_seq[idx][0]:
                    mit = idx
            # `res.insert(*mit)` copies the pair, then `erase(mit)` destroys
            # the source -- so nothing ever mutates the inserted element
            # afterwards and sharing the object here is safe.
            last_fs_cit = csc_fs_seq.pop(mit)
            res.append(last_fs_cit)

        return res

    def mi(self, fs_l: feature_set, fs_r: feature_set) -> float:
        """C++: moses/moses/deme/feature_selector.cc:434 feature_selector::mi

        "Compute (or approximate) the mutual information between 2 feature
        sets. If params.diversity_interaction is negative then it returns
        the mutual information between fs_l and fs_r. Otherwise it returns
        the average of mutual informations between all subsets of fs_l and
        fs_r of size params.diversity_interaction + 1"
        (feature_selector.h:224-232's own doc comment).

        Computed against ``self._ctable`` -- the WHOLE table, not the
        exemplar-filtered ``fs_ctable`` ``build_fs_ctable`` returns.

        The summation runs over ``std::set<Set>``'s ascending lexicographic
        order (see ``powerset``); ``double`` addition is not associative,
        so that order is part of the answer.

        # DIVERGENCE(C++ moses/moses/deme/feature_selector.cc:452): the
        # final `mi / (pfs_l.size() * pfs_r.size())` is IEEE-754 division
        # in C++ and gives nan/+-inf when the denominator is 0, where
        # Python raises ZeroDivisionError. Reproduced explicitly. It is
        # reachable: `powerset(fs, n, exact=true)` returns the EMPTY set
        # whenever `n > fs.size()`, i.e. whenever
        # `diversity_interaction + 1` exceeds a feature set's own size --
        # `--fs-diversity-interaction=3` against any 2-feature set, say.
        """
        if self.params.diversity_interaction < 0:
            return mutual_information_between_sets(self._ctable, fs_l, fs_r)

        # Compute the average mutual informations between subsets of
        # size diversity_interaction + 1 of fs_l and fs_r
        sss = self.params.diversity_interaction + 1
        mi = 0.0
        # determine powersets of fs_l and fs_r, each containing
        # feature sets of size diversity_interaction + 1
        pfs_l = powerset(fs_l, sss, True)
        pfs_r = powerset(fs_r, sss, True)
        for sub_fs_l in pfs_l:
            for sub_fs_r in pfs_r:
                mi += mutual_information_between_sets(self._ctable, sub_fs_l, sub_fs_r)
        denominator = len(pfs_l) * len(pfs_r)
        if denominator == 0:
            if mi == 0.0:
                return math.nan
            return math.copysign(math.inf, mi)
        return mi / denominator

    def sample_enforced_features(self) -> feature_set:
        """C++: moses/moses/deme/feature_selector.cc:455
        feature_selector::sample_enforced_features

        "Sample a set of features randomly choosen given enforce_features,
        a map from feature to probability of being enforced"
        (feature_selector.h:194-196's own doc comment). Draws
        ``biased_randbool(p)`` == ``p > randGen().randfloat()``
        (opencog/util/random.h:127) for every ``(name, probability)`` pair
        in ``self.params.enforce_features`` -- iteration order over a
        Python ``dict`` is insertion order, not C++'s ``std::map``
        alphabetical order (``enforce_features`` is a
        ``std::map<std::string,float>``); this only changes which RNG
        draw lands on which key when the map has more than one entry, not
        the SET of draws made.

        With the default (empty) ``enforce_features`` the loop runs zero
        iterations and this returns an empty set without ever calling
        ``randGen().randfloat()`` -- so the RNG stream is untouched, which
        is exactly what makes the default path's draw count observable.

        BOTH SIDES OF THE COMPARISON ARE C ``float``s. ``randfloat()``
        returns binary32 (mt19937ar.cc:50) and ``biased_randbool``'s
        parameter is ``float b``; the weight reaches here already narrowed,
        because ``feature_selector_parameters`` stores it the way C++'s
        ``std::map<std::string,float>`` does (see that constructor).
        Comparing an un-narrowed double against ``randfloat()`` would flip
        the decision for any weight whose double and float32 values
        straddle the draw.
        """
        sampled_features: list[str] = []
        for name, probability in self.params.enforce_features.items():
            if probability > randGen().randfloat():  # biased_randbool(probability)
                sampled_features.append(name)

        if logger().is_debug_enabled():
            # C++: feature_selector.cc:464-468 -- the sampled NAMES, not the
            # indices, space-separated after the prefix.
            logger().debug("Enforce features: " + " ".join(sampled_features))

        ilabels = self._ctable.get_input_labels()
        return set(get_indices(sampled_features, ilabels))

    def __call__(self, xmplr: combo_tree) -> feature_set_pop:
        """C++: moses/moses/deme/feature_selector.cc:316 feature_selector::operator()

        "Return feature set population that is good when combined with
        the exemplar tr" (feature_selector.h:180-181's own doc comment).
        """
        self.preprocess_params(xmplr)
        fs_ctable = self.build_fs_ctable(xmplr)
        sf_pop = select_feature_sets(fs_ctable, self.params.fs_params)

        if 0 < self.params.diversity_cap < len(sf_pop):
            sf_pop = sf_pop[: self.params.diversity_cap]

        top_sfs = self.select_top_feature_sets(sf_pop)
        self.log_stats_top_feature_sets(top_sfs)  # C++: feature_selector.cc:338
        return top_sfs

    def log_stats_top_feature_sets(self, top_fs: feature_set_pop) -> None:
        """C++: moses/moses/deme/feature_selector.cc:250
        feature_selector::log_stats_top_feature_sets

        Ported as far as C++'s own early return (cc:255-256, "There aren't
        any stats, if there's just one feature set! (which is the case
        when diversity is turned off)").

        # DIVERGENCE(C++ moses/moses/deme/feature_selector.cc:258-292):
        # the Boost accumulator block past that return -- count/mean/
        # variance/min/max over the pairwise `mi`/`jaccard_index` of the
        # selected feature sets, streamed into two `logger().info()` calls
        # (cc:280,287) -- is not ported. It is observed by nothing, and it
        # is unreachable in any run that survives: `top_fs` bigger than
        # one requires `--fs-demes > 1`, and the very next thing
        # `create_representations` does is call
        # `deme_expander.log_selected_feature_sets`, whose
        # `OC_ASSERT(sfps == _demeIDs.size())` (deme_expander.cc:63)
        # aborts classic for exactly that case (measured: exit 134). So
        # the records it would emit only ever precede a crash.
        """
        logger().info(f"Number of demes selected: {len(top_fs)}")
        if len(top_fs) == 1:
            return
