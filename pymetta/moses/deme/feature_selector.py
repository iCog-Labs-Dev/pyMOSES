"""C++: moses/moses/deme/feature_selector.{h,cc}

``feature_selector`` -- the in-loop ``--enable-fs`` entry point. Given an
exemplar ``combo_tree``, restricts (and optionally re-shapes) the scoring
``CTable`` around that exemplar's own predictions, then runs
``pymetta.feature_selection.main.feature_selection.select_feature_sets``
(only ``algorithm="simple"`` is ported there) to pick a feature subset "good
when combined with the exemplar" (the class's own C++ doc comment,
feature_selector.h:180-181).

Scope narrowing vs. the 233+477-line C++ original:

* ``log_stats_top_feature_sets`` (feature_selector.cc:256-291) is
  logging-only (accumulator stats fed straight to ``logger().info()``,
  nothing else observes them) -- NOT ported, matching the "no logging
  surface in this port" convention ``deme_expander.py``'s own module
  docstring already states for every other logger()-only routine in this
  project.
* ``remove_useless_features`` (feature_selector.h:211,
  feature_selector.cc:295-309) is declared and defined but has ZERO call
  sites anywhere in classic (confirmed by grep across the whole ``moses/``
  tree) -- genuinely dead code in C++ itself, not merely out of this
  port's scope. NOT ported, matching this project's "omit dead code,
  don't add a placeholder" discipline (``deme_expander.py``'s treatment of
  ``fs_to_names``/``prune_xmplr`` is the same call, for a different
  reason -- those two DO have callers, but only inside a gated branch).
* ``rank_feature_sets``/``mi`` (feature_selector.cc:314-419) are reachable
  only via ``select_top_feature_sets`` when
  ``feature_selector_parameters.diversity_pressure > 0.0`` (struct default
  ``0.0`` -- off). ``rank_feature_sets`` builds a ``composite_score`` per
  feature set (``moses/moses/moses/types.h``) to combine a raw score with
  a computed "uniformity penalty" -- ``composite_score`` lives in
  ``pymetta.moses.moses``, explicitly OUT OF SCOPE for this task
  (CLAUDE.md: "Do not touch ... pymetta/moses/moses/"). Both methods
  raise ``NotImplementedError`` citing this; the default
  (``diversity_pressure <= 0.0``) path through ``select_top_feature_sets``
  is fully ported and never reaches either.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_argument, vertex
from pymetta.comboreduct.interpreter.interpreter import boolean_interpreter
from pymetta.comboreduct.table.table import CTable, Table, get_indices
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.feature_selection.main.feature_selection import select_feature_sets
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymetta.util.rng import randGen

__all__ = [
    "EXEMPLAR_FEATURE_NAME",
    "feature_selector",
    "feature_selector_parameters",
    "get_argument_abs_idx_from_zero_set",
]

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
        self.enforce_features: dict[str, float] = (
            dict(enforce_features) if enforce_features is not None else {}
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
        self.params = festor_params if festor_params is not None else feature_selector_parameters()
        self._ctable: CTable = (
            ctable_or_table.compressed()
            if isinstance(ctable_or_table, Table)
            else ctable_or_table
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
        ``subsampling_ratio < 1.0`` needs ``subsampleCTable``/
        ``get_timestamps``/``remove_rows_at_times``, none ported (out of
        scope: touching ``pymetta.comboreduct`` is off limits, and this
        port's ``CTable`` carries no timestamps at all -- see
        ``pymetta.comboreduct.table.table``'s own module docstring) --
        raises rather than silently skipping.
        """
        sig = self._ctable.get_signature()
        cto = sig.otype if sig is not None else type_node.unknown_type
        if cto is not type_node.boolean_type:
            raise NotImplementedError(
                f"feature_selector.build_fs_ctable: output type {cto!r} is not "
                "boolean_type -- this port's interpreter/representation stack "
                "is boolean-only (see build_fs_ctable's own docstring)."
            )

        olabel = self._ctable.get_labels()[0]
        ilabels = list(self._ctable.get_input_labels())
        itypes = list(sig.itypes) if sig is not None else []
        if self.params.xmplr_as_feature:
            ilabels.append(EXEMPLAR_FEATURE_NAME)
            itypes.append(cto)

        new_sig = TypeSignature(itypes, cto) if sig is not None else None
        fs_ctable = CTable(olabel, ilabels, new_sig)

        interp = boolean_interpreter
        ignored_sorted = sorted(self.params.ignored_features)

        for row, counter in self._ctable.items():
            predicted_out: vertex = interp(list(row))(xmplr)

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
                    inputs.append(predicted_out)  # get_builtin(predicted_out): identity here

                key = tuple(inputs)
                dst = fs_ctable.setdefault(key, type(counter)())
                for value, count in counter.items():
                    dst[value] = dst.get(value, 0.0) + count

        if self.params.subsampling_ratio < 1.0:
            what = "by-time" if self.params.subsampling_by_time else "by-row"
            raise NotImplementedError(
                f"feature_selector.build_fs_ctable: subsampling_ratio < 1.0 ({what}) "
                "needs subsampleCTable/timestamp support, neither ported -- "
                "pymetta.comboreduct is out of this task's scope, and this "
                "port's CTable carries no timestamps at all. Leave "
                "subsampling_ratio at its default (1.0)."
            )

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
        a DIFFERENT branch -- rank by penalized (diversity-adjusted)
        score first -- gated: see this module's own docstring for why
        (needs ``composite_score``, ``pymetta.moses.moses``, out of
        scope).
        """
        if self.params.diversity_pressure > 0.0:
            raise NotImplementedError(
                "feature_selector.select_top_feature_sets: diversity_pressure "
                "> 0.0 selects rank_feature_sets(), which needs composite_score "
                "(pymetta.moses.moses.types) -- moses/moses/moses/ is out of "
                "this task's scope. Leave diversity_pressure at its default (0.0)."
            )
        res_size = min(self.params.n_demes, len(fss))
        return fss[:res_size]

    def sample_enforced_features(self) -> feature_set:
        """C++: moses/moses/deme/feature_selector.cc:445
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
        ``randGen().randfloat()`` -- which is NOT itself ported yet
        (``pymetta.util.rng.MT19937RandGen.randfloat`` raises
        ``NotImplementedError`` by its own design, needing C float32
        truncation -- see that method's own docstring). A non-empty
        ``enforce_features`` will surface THAT ``NotImplementedError``
        here, faithfully -- not a gate this file adds, but a real,
        already-documented gap one call away.
        """
        sampled_features: list[str] = []
        for name, probability in self.params.enforce_features.items():
            if probability > randGen().randfloat():  # biased_randbool(probability)
                sampled_features.append(name)

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
        # log_stats_top_feature_sets(top_sfs): NOT ported (logging-only,
        # see module docstring).
        return top_sfs
