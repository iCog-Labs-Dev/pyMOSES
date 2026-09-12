"""C++: moses/moses/eda/local_structure.{h,cc}

Bayesian Optimization Algorithm (BOA) machinery: given a population of
scored instances, builds a "dependency tree" (`dtree`) per raw field that
records how often each of that field's possible values occurs, then samples
fresh instances from those per-field distributions. `univariate` (below) is
the degenerate case that never builds cross-field dependencies at all --
the only StructureLearningPolicy `univariate_optimization` (`univariate.py`)
uses.

`dtree_node` (C++ `typedef vector<int> dtree_node`) is ported as a bare
`list[int]`, matching the project's convention of not wrapping a plain
`typedef` in a class (e.g. `behavioral_score = list[score_t]`, `types.py`).
`dtree` (C++ `typedef tree<dtree_node> dtree`) reuses `pymoses.util.tree`
directly -- that module is a generic `tree<T>` (its own docstring: "Iterators
collapse to plain `tree_node` objects... Python object identity replaces
C++'s pointer-identity-keyed `obj_ptr_cmp<It>`"), already exercised for
`combo_tree = tree[vertex]`; nothing here needs a second implementation.

Term-algebra fields (`rec_split_term`, the `_fields.term()` half of the
constructor's `if` guard) are OMITTED OUTRIGHT, not merely guarded: term
fields do not exist anywhere in this port (`field_set.py`'s own scope
exclusion, and the same decision `hill_climbing.py`'s module docstring and
`neighborhood_sampling.py`'s module docstring both already make) -- there is
no `term_spec` class to iterate, so C++'s `for (const field_set::term_spec&
o : _fields.term())` loop always has an empty range here by construction.
The continuous half (`rec_split_contin`) IS ported: `field_set.py` carries
full `contin_spec`/`get_raw`/`contin_to_raw_idx` support (S14, landing
concurrently with this task).

`local_structure_model::operator()` (sampling) calls
`opencog::roulette_select` at every leaf -- reused directly from
`pymoses.util.selection.roulette_select` (already ported for
`metapopulation.select_exemplar`; see that module's own docstring for why
it specializes the four-argument `(items, sum, rng)` form, which is exactly
the shape `dtree` leaf-sampling also needs), consuming draws from the same
seeded `pymoses.util.rng.randGen()` singleton the rest of this port shares.
It instantiates that template at `ScoreT = int`, NOT `score_t` -- see
`_sample`'s own docstring.

THIS MODULE READS TWO INDEPENDENT RANDOM STREAMS, and getting either one
wrong changes the search:

* `randGen()` (mt19937, seeded by `-r`), read by `roulette_select`'s
  `rng.randdouble()` once per non-empty leaf inside `_sample`, in
  `_ordering` order.
* C's `rand()` (glibc TYPE_3, UNSEEDED and NOT controlled by `-r`), read
  by `randomized_topological_sort`'s `std::random_shuffle` -- once per
  `local_structure_model` construction, i.e. once per generation of
  `optimize()`. `pymoses.util.digraph` reproduces that stream draw for
  draw; its module docstring carries the whole derivation and the
  residual divergence.

The second stream decides `_ordering`, and `_ordering` decides which raw
field consumes which `randGen()` draw in `operator()`. So an error in
EITHER stream shifts the other's effect -- which is why the port matched
classic below `-m1000` (where `max_gens_total` collapses to 0, no model is
ever built, and neither stream is read) and diverged above it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.instance import disc_t, instance
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.util.digraph import digraph, randomized_topological_sort
from pymoses.util.selection import roulette_select
from pymoses.util.tree import tree, tree_node

__all__ = [
    "dtree_node",
    "local_structure_model",
    "local_structure_probs_learning",
    "univariate",
]

dtree_node = list[disc_t]  # C++: moses/moses/eda/local_structure.h:68 dtree_node
dtree = tree  # C++: moses/moses/eda/local_structure.h:73 dtree -- tree<dtree_node>


class local_structure_model:
    """C++: moses/moses/eda/local_structure.h:75 local_structure_model

    C++ is `public nullary_function<instance>, public vector<dtree>` -- one
    `dtree` per RAW field (`super(fs.raw_size())`). `dtrees()` below exposes
    that `vector<dtree>` base's public surface (what
    `local_structure_probs_learning.__call__` iterates, matching C++'s own
    `dst.begin()`/`dst.end()` over the base-class part of `dst`) rather than
    inheriting from `list` outright -- nothing else in this port needs
    `local_structure_model` to duck-type as a plain list.
    """

    def __init__(self, fields: field_set, instances: Sequence[scored_instance]) -> None:
        """C++: moses/moses/eda/local_structure.h:177
        local_structure_model::local_structure_model

        `instances` stands in for the `It from, It to` iterator pair --
        matching this port's usual "iterator pair -> Python sequence"
        convention (`pymoses/util/selection.py`'s module docstring). Every
        element converts implicitly to `const instance&` in C++ via
        `tagged_item::operator const Item&()`
        (`$COGUTIL/opencog/util/functional.h:82`) -- `.first` is read
        explicitly here instead, once, up front.
        """
        self._fields = field_set(fields)  # C++: _fields(fs) -- copy, not alias
        n = self._fields.raw_size()
        self._dtrees: list[tree] = [tree() for _ in range(n)]
        self._instance_length = self._fields.packed_width()
        self._initial_deps = digraph(n)

        insts = [si.first for si in instances]

        dtr_idx = 0
        # DIVERGENCE(C++ moses/moses/eda/local_structure.h:202): C++ guards
        # this whole block on `!_fields.contin().empty() ||
        # !_fields.term().empty()`. Term fields do not exist in this port
        # (see module docstring), so the `term()` disjunct -- and the
        # `for (const field_set::term_spec& o : _fields.term())` loop that
        # would follow it -- is omitted outright rather than guarded; it
        # is always an empty range here by construction.
        if self._fields.contin():
            for c in self._fields.contin():
                idx_base = dtr_idx
                self._make_dtree(dtr_idx, 3)  # contin arity is 3 = Stop/Left/Right
                dtr_idx += 1
                for i in range(1, c.depth):
                    self._make_dtree(dtr_idx, 3)
                    self._initial_deps.insert(idx_base + i - 1, idx_base + i)
                    # Only split on parent (the C++ comment's "Alternatively"
                    # branch -- the gggparent/.../parent recursive-split
                    # alternative just above it in local_structure.h:243-244
                    # is commented out in classic itself).
                    self._rec_split_contin(
                        insts, idx_base + i - 1, idx_base + i, self._dtrees[dtr_idx].begin()
                    )
                    dtr_idx += 1

        for d in self._fields.disc_and_bit():
            self._make_dtree(dtr_idx, d.multy)
            dtr_idx += 1

        self._ordering: list[int] = randomized_topological_sort(self._initial_deps, [])

    def dtrees(self) -> list[tree]:
        """Not itself a ported C++ symbol -- exposes the `vector<dtree>`
        PUBLIC base class `local_structure_model` inherits from in C++
        (see class docstring). `local_structure_probs_learning.__call__`
        is the one real caller.
        """
        return self._dtrees

    def __call__(self) -> instance:
        """C++: moses/moses/eda/local_structure.cc:128 local_structure_model::operator()

        "sample from a model" (the C++ comment).
        """
        n = len(self._dtrees)
        tmp: list[disc_t] = [0] * n
        for idx in self._ordering:
            tmp[idx] = self._sample(self._dtrees[idx], self._dtrees[idx].begin(), tmp)
        res = instance(self._instance_length)
        self._fields.pack(tmp, res)
        return res

    # ---- construction-time structure building ------------------------------

    def _make_dtree(self, dtr_idx: int, arity: int) -> None:
        """C++: moses/moses/eda/local_structure.cc:112 local_structure_model::make_dtree

        "plus one: the first arity will hold counts, the last will hold the
        grand-total." (the C++ comment.)
        """
        self._dtrees[dtr_idx].set_head([0] * (arity + 1))

    def _split(self, src_idx: int, tgt_idx: int, tgt: tree_node) -> None:
        """C++: moses/moses/eda/local_structure.cc:119 local_structure_model::split"""
        dtr = self._dtrees[tgt_idx]
        # DIVERGENCE(C++ moses/moses/eda/local_structure.cc:122): OC_ASSERT
        # becomes AssertionError.
        if not dtr.is_childless(tgt):
            raise AssertionError("dtree node should be childless (split)")
        copies = [list(tgt.data) for _ in range(len(tgt.data) - 1)]
        dtr.append_children(tgt, copies)
        tgt.data = [src_idx]

    def _is_uniform_on(self, insts: Sequence[instance], idx: int) -> bool:
        """C++: moses/moses/eda/local_structure.cc:40 local_structure_model::is_uniform_on

        True if raw field `idx` has the same value across every instance in
        `insts` -- vacuously true for 0 or 1 instances, matching
        `std::adjacent_find` over an empty or single-element range.
        """
        if len(insts) <= 1:
            return True
        first = self._fields.get_raw(insts[0], idx)
        return all(self._fields.get_raw(i, idx) == first for i in insts[1:])

    def _rec_split_contin(
        self, insts: Sequence[instance], src_idx: int, idx: int, node: tree_node
    ) -> None:
        """C++: moses/moses/eda/local_structure.cc:92
        local_structure_model::rec_split_contin
        """
        if self._is_uniform_on(insts, idx):
            return

        self._split(src_idx, idx, node)

        # "important - needs to match contin_spec::Left/Right" (C++ comment).
        if src_idx < idx - 1:
            dtr = self._dtrees[idx]
            groups: list[list[instance]] = [[], [], []]
            for i in insts:
                groups[self._fields.get_raw(i, src_idx)].append(i)
            self._rec_split_contin(groups[1], src_idx + 1, idx, dtr.child(node, 1))
            self._rec_split_contin(groups[2], src_idx + 1, idx, dtr.child(node, 2))

    # ---- sampling ------------------------------------------------------------

    def _sample(self, dtr_tree: tree, node: tree_node, margs: list[disc_t]) -> disc_t:
        """C++: moses/moses/eda/local_structure.cc:140 local_structure_model::sample

        `dst` is an out-parameter (`disc_t&`) in C++; returned here instead
        (see class docstring's general convention).

        `roulette_select(dtr->begin(), --(dtr->end()), dtr->back())`
        (local_structure.cc:146) instantiates the template at
        **`ScoreT = int`**, not `score_t`: `dtree_node` is `typedef
        vector<int>` (local_structure.h:68), so `dtr->back()` is an `int`
        and template argument deduction picks `int`. That makes
        `sum = ScoreT(double(sum) * rng.randdouble())` a TRUNCATION toward
        zero and the whole subtraction loop exact integer arithmetic --
        hence the explicit `score_type=int`. It is not a detail: over the
        6,214-point oracle grid in `tests/fixtures/univariate_rng.json`,
        the `int` and `score_t` instantiations of the real C++ template
        disagree at 1,409 points (see `pymoses/util/selection.py`'s module
        docstring).

        `distance(dtr->begin(), roulette_select(...))` becomes the returned
        index directly, per `roulette_select`'s own index-returning
        convention.
        """
        if dtr_tree.is_childless(node):  # a leaf
            if node.data[-1] > 0:
                return roulette_select(node.data[:-1], node.data[-1], score_type=int)
            # "if no data, do uniform selection    WHY IS THIS COMMENTED
            # OUT???" (local_structure.cc:149's own comment) -- classic
            # returns 0 unconditionally here, DO NOT "FIX" to a uniform
            # draw: that would both diverge from classic and consume an
            # extra randGen() draw classic's own trace never makes.
            return 0
        # DIVERGENCE(C++ moses/moses/eda/local_structure.cc:151-154):
        # OC_ASSERT becomes AssertionError (both checks below).
        if len(node.data) != 1:
            raise AssertionError("dtree node size should be equals to 1.")
        src_idx = node.data[0]
        if not dtr_tree.number_of_children(node) > margs[src_idx]:
            raise AssertionError("dtree node children number grearter than margs.")
        child = dtr_tree.child(node, margs[src_idx])
        return self._sample(dtr_tree, child, margs)


class univariate:
    """C++: moses/moses/eda/local_structure.h:125 univariate

    "May be used as a StructureLearningPolicy. Simplest case: does no
    structure learning at all. This is suitable for use when all
    fields/variables in a population are known to be statistically
    independent of one-another." (the header's own comment.)
    """

    model_type = local_structure_model

    def __call__(
        self,
        fields: field_set,
        instances: Sequence[scored_instance],
        model: local_structure_model,
    ) -> None:
        """C++: moses/moses/eda/local_structure.h:130 operator()

        A no-op, verbatim.
        """
        return


class local_structure_probs_learning:
    """C++: moses/moses/eda/local_structure.h:146 local_structure_probs_learning

    "Update the model given a field_set and a range of 'good' instances."
    (the header's own comment for `operator()`.)
    """

    model_type = local_structure_model

    def __call__(
        self,
        fields: field_set,
        instances: Sequence[scored_instance],
        dst: local_structure_model,
    ) -> None:
        """C++: moses/moses/eda/local_structure.h:281
        local_structure_probs_learning::operator()

        A binary `for_each` over `dst`'s dtrees (`vector<dtree>::begin()`,
        `local_structure_model.dtrees()` here) and a counting iterator --
        i.e. `rec_learn(fs, from, to, i, dst.dtrees()[i].begin())` for every
        raw field index `i`, each call seeing the FULL `instances` range.
        """
        insts = [si.first for si in instances]
        for idx, dtr_tree in enumerate(dst.dtrees()):
            self._rec_learn(fields, insts, idx, dtr_tree, dtr_tree.begin())

    def _rec_learn(
        self,
        fields: field_set,
        insts: Sequence[instance],
        idx: int,
        dtr_tree: tree,
        node: tree_node,
    ) -> None:
        """C++: moses/moses/eda/local_structure.h:298
        local_structure_probs_learning::rec_learn

        "For each node in the dependency tree, accumulate statistics from
        the instances. Basically, just count how often various values
        occur ..." (the header's own comment.) The C++ `n_way_partition`
        call reorders `[from, to)` in place by raw value; this port groups
        into fresh per-value lists instead -- the counts `rec_learn`
        actually accumulates are commutative (a running `++` per element,
        then one `sum()`), so which physical order the elements of one
        group are visited in is unobservable, unlike the tie-order-
        sensitive sorts this project's own `_sort`/`_partial_sort` exist to
        reproduce exactly (docs/CARRY-FORWARD.md 5b.1, 5.2) -- there is no
        tie order here at all to preserve.
        """
        if dtr_tree.is_childless(node):
            for inst in insts:
                v = fields.get_raw(inst, idx)
                # DIVERGENCE(C++ moses/moses/eda/local_structure.h:309):
                # OC_ASSERT becomes AssertionError.
                if not v < len(node.data) - 1:
                    raise AssertionError(
                        "assertion failed: fs.get_raw(*from, idx) < dtr->size() - 1"
                    )
                node.data[v] += 1
            node.data[-1] = sum(node.data[:-1])
        else:
            raw_arity = dtr_tree.number_of_children(node)
            src_idx = node.data[0]
            groups: list[list[instance]] = [[] for _ in range(raw_arity)]
            for inst in insts:
                groups[fields.get_raw(inst, src_idx)].append(inst)
            for i in range(raw_arity):
                self._rec_learn(fields, groups[i], idx, dtr_tree, dtr_tree.child(node, i))
