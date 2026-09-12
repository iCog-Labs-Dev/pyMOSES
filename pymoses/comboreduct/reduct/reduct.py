"""C++: moses/comboreduct/reduct/reduct.{h,cc}

Scoped to exactly what Task 2 (S7) needs and produces: the `rule` base
class -- every reduction rule and every meta-rule combinator
(`meta_rules.py`, this same task) is a `rule`. Everything else reduct.h
declares (`logical_reduction`, `contin_reduction`, `fold_reduction`,
`mixed_reduction`, `full_reduction`, `action_reduction`,
`perception_reduction`, `clean_reduction`, and the `xxx_reduce` inline
convenience wrappers around them) depends on rule SETS Tasks 3-5 build;
porting their entry points here now would just be dead code with nothing
to call. `docs/superpowers/plans/2026-09-04-pymoses-s7-reduct.md`'s own
file-structure table lists `logical_reduction.py`/`clean_reduction.py` as
separate files for exactly this reason -- this file grows again once
those exist.

Three C++ symbols from reduct.h are DELIBERATELY NOT PORTED, each with a
zero-call-site verification, not a guess:

* `crule<T>` (reduct.h:63-68) -- collapsed into `rule.clone()`'s own
  generic default implementation. See that method's docstring for the
  full reasoning (a `# DIVERGENCE` marker lives there, not here, since it
  is about ONE method's behaviour, not a missing class).
* `reduct::new_clone` (reduct.h:61, reduct.cc:30) -- "needed by boost
  ptr_vector... not otherwise used anywhere" per its own C++ comment.
  `sequential.rules` (meta_rules.py) is a plain Python `list`, not a
  `boost::ptr_vector`, so the ownership-registration hook this function
  exists to satisfy has no Python equivalent to satisfy. Zero call sites
  outside that boost machinery in $CLASSIC (`grep -rn new_clone
  moses/comboreduct/reduct/` -- only its own declaration/definition).
* `replace_without_changing_it` (reduct.h:252-264) -- used by
  `ignore_size_increase::operator()` (meta_rules.cc:74) ONLY, plus one
  more call site in `contin_rules.cc` (S14, out of scope). Not needed
  because `ignore_size_increase` itself is not ported -- see
  meta_rules.py's own module docstring for why (verified: `grep -rln
  "when(\\|ignore_size_increase(" logical_reduction.cc general_rules.cc
  logical_rules.cc clean_reduction.cc` -- zero matches; both are used
  only by contin_reduction.cc, S14).
"""

import copy
from abc import ABC, abstractmethod

from pymoses.comboreduct.combo.vertex import combo_tree, vertex_equal
from pymoses.util.tree import tree_node

# No direct C++ symbol. Sentinel for `rule.__call__`'s omitted-`it` default
# -- see that method's own docstring (FIX WAVE, Task 2 review, Important 3)
# for why plain `None` cannot serve as the default: `tree.next_sibling()`
# and friends legitimately return `None` for "no such node", so an explicit
# `it=None` from a real caller must be a hard error, not silently
# indistinguishable from "no `it` given at all".
_NO_IT = object()


class rule(ABC):
    """C++: moses/comboreduct/reduct/reduct.h:35 rule

    The base class every reduction rule and meta-rule combinator extends.
    Deliberately does NOT extend `crule<T>` -- see this class's own
    `clone()` docstring, and reduct.py's module docstring, for why
    `crule<T>` itself is not ported as a separate class.
    """

    def __init__(self, name: str) -> None:
        """C++: moses/comboreduct/reduct/reduct.h:39 rule::rule"""
        self.name = name

    @abstractmethod
    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/reduct.h:41 rule::operator()
        (combo_tree&, combo_tree::iterator) -- the pure virtual every
        concrete rule overrides with its actual reduction logic.

        Named `apply`, not `__call__`: C++ has a SECOND, CONCRETE,
        non-virtual overload at this same name (`operator()(combo_tree&)`,
        reduct.h:44) that no subclass may ever override -- it is defined
        ONCE, in this base class, and dispatches to whichever subclass's
        `apply` applies via ordinary virtual dispatch. `__call__` below is
        that second overload, and is the one whose SIGNATURE is "the C++
        call signature" a caller actually uses.
        """

    def __call__(self, tr: combo_tree, it: tree_node | None = _NO_IT) -> None:
        """C++: moses/comboreduct/reduct/reduct.h:44 rule::operator()
        (combo_tree&) [concrete] -- the pure virtual half of the same
        overloaded C++ call operator (reduct.h:41) is reached here via
        `apply`.

        Covers both C++ overloads in one Python method, the same
        convention this port uses throughout for an overloaded C++ call
        operator (e.g. interpreter.py's `boolean_interpreter.__call__`):
        `it` OMITTED selects the 1-arg overload -- if `tr` is not empty,
        recurse with `it = tr.begin()` (reduct.h:46-47, `(*this)(tr,
        tr.begin())`, itself a call THROUGH this same method, landing on
        the "real `it` given" branch below on the way back in). An empty
        `tr` with no `it` given is a silent no-op, matching C++'s `if
        (!tr.empty())` guard exactly.

        FIX WAVE (Task 2 review, Important 3): the default was plain
        `None` in an earlier version of this method, which made an
        EXPLICIT `r(tr, None)` indistinguishable from `r(tr)` -- silently
        taking the 1-arg, tr.begin()-recursing path instead of raising.
        C++ cannot conflate these: `operator()(combo_tree&)` and
        `operator()(combo_tree&, combo_tree::iterator)` are two distinct,
        statically-resolved overloads, so passing a null-equivalent
        iterator explicitly would go through the 2-arg one and either
        crash or misbehave -- never silently redirect to the root. This
        matters because a real caller CAN legitimately end up with a
        `None` `it` without meaning "omit it": `tree.next_sibling(x)`
        returns `None` for "no such node" (end of a sibling list), so a
        rule written as `r(tr, tr.next_sibling(x))` at the end of a list
        would, under the old default, silently reduce from `tr.begin()`
        instead of failing -- a wrong reduction with no error, exactly
        the class of bug this port's `# DIVERGENCE` / "fail loudly rather
        than silently corrupt" convention exists to avoid (see e.g.
        tree.py's own erase()/replace() poisoning). `_NO_IT` (a private
        sentinel `object()`, reduct.py module level) now stands in for
        "omitted"; an explicitly passed `None` is rejected HERE, directly
        and immediately, rather than left to whatever a given subclass's
        `apply` happens to do with it -- `sequential` with zero wrapped
        rules, for instance, would loop zero times and return silently
        with no error at all if `None` were simply forwarded, which would
        have reintroduced exactly the "no error" failure mode this fix is
        for.
        """
        if it is _NO_IT:
            if not tr.empty():
                self(tr, tr.begin())
            return
        if it is None:
            raise ValueError(
                "rule.__call__: it is None -- omit it entirely for the "
                "1-arg overload (reduct.h:44, applies at tr.begin()), or "
                "pass a real node. A None it usually means the caller's "
                "own node lookup (e.g. tree.next_sibling()) came up "
                "empty and should be checked before calling the rule, "
                "not passed through."
            )
        self.apply(tr, it)

    def get_name(self) -> str:
        """C++: moses/comboreduct/reduct/reduct.h:50 rule::get_name"""
        return self.name

    def clone(self) -> rule:
        """C++: moses/comboreduct/reduct/reduct.h:42 rule::clone

        Pure virtual; the CRTP override every concrete rule/meta-rule
        actually inherits in real C++ is reduct.h:67 crule<T>::clone
        (`rule* clone() const { return new T(*((T*) this)); }`).

        # DIVERGENCE(C++ moses/comboreduct/reduct/reduct.h:63-68 crule<T>): not
        # ported as its own class -- see reduct.py's module docstring for the
        # zero-call-site check on the other two reduct.h omissions; this one is
        # a design collapse, not a zero-call-site absence, so it gets its own
        # explanation here. `crule<T>`'s entire C++ purpose is generating a
        # type-correct `clone()` override per subclass via CRTP, so that a
        # `shared_ptr<const rule>`-owning container (each meta-rule's own
        # wrapped-rule member; `sequential.rules`, a `boost::ptr_vector<rule>`)
        # gets an INDEPENDENT COPY on construction, protecting against the
        # ORIGINAL rule object's own lifetime or later mutation. Neither hazard
        # exists in Python: objects are garbage collected (no dangling-owner
        # concern), and no rule anywhere in `meta_rules.cc` ever mutates its
        # own or another rule's fields after construction -- every
        # `operator()` body reads `r`/`rules` only to CALL through them
        # (`(*r)(tr, it)`, `r(tr, it)`), never assigns through them. A single
        # generic `clone()` here, shared by every subclass via inheritance, is
        # therefore behaviourally equivalent to CRTP-generated per-subclass
        # clones: it returns a NEW object (`copy.copy`, never `self` --
        # `clone() is not self` holds, matching "a new instance" being the one
        # thing C++'s `new T(*this)` guarantees that a caller can actually
        # observe). Only the object IDENTITY of a clone's own NESTED
        # `.r`/`.rules` differs from a hypothetical recursive re-clone (shared
        # by reference here, `copy.copy` being shallow, rather than
        # independently re-cloned) -- unobservable, for the same
        # never-mutated-through-a-reference reason.
        """
        return copy.copy(self)


def replace_without_changing_it(tr: combo_tree, dst: tree_node, src: tree_node) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:252 replace_without_changing_it

    S14 Task 4 addition: `meta_rules.py`'s `ignore_size_increase` and
    `contin_rules.py`'s `reduce_distribute` are this helper's only two real
    call sites in $CLASSIC (reduct.py's own module docstring already noted
    both, deferring the port to "S14, out of scope" -- that deferral ends
    here). C++'s own comment: replace `dst`'s subtree with a copy of `src`'s,
    WITHOUT changing which node `dst` itself refers to -- needed because the
    reduct engine assumes a rule never changes the iterator it was handed.

    `*dst = *src;` copies `src`'s own vertex value onto `dst` in place
    (`copy.copy`, matching every other subtree-copy entry point's use of
    `_copy_value` for the identical `argument`-aliasing reason -- see
    `tree.py`'s own module docstring). Then: if `src` is childless, `dst`'s
    children are simply erased (`tr.erase_children(dst)`); otherwise C++
    calls the general sibling-RANGE `replace(dst.begin(), dst.end(),
    src.begin(), src.end())` overload, which this port's `tree.replace`
    does not carry (scoped, like `tree.sort`/`tree.partition`, to the one
    shape every real caller in this port needs -- see those methods' own
    docstrings for the same scoping convention). Reproduced here directly:
    erase `dst`'s current children, then deep-copy each of `src`'s children
    onto `dst` in order via `tr.append_child` (already a subtree-copying
    primitive) -- net effect identical to the general range-replace for
    this "replace ALL of dst's children with copies of ALL of src's"
    shape, which is the only shape either real call site ever needs (C++'s
    own comment: "it is assumed (for now) that dst and src have the same
    number of children" -- a note about its two callers' inputs, not a
    precondition this function enforces either in C++ or here).
    """
    dst.data = copy.copy(src.data)
    tr.erase_children(dst)
    for child in list(tr.children(src)):
        tr.append_child(dst, child)


def ignores(ignore_ops, op) -> bool:
    """No direct C++ symbol: reproduces `ignore_ops.find(id::xxx) ==
    ignore_ops.end()` -- NEGATED, i.e. this returns True exactly when
    `id::xxx` IS a member of `ignore_ops` (callers write `not ignores(...)`
    for the C++ `== end()` "not found" sense; `contin_reduction.py`/
    `mixed_reduction.py`/`full_reduction.py`'s `when(...)` gates all read
    this way). `vertex_set::find` is a type-safe `boost::variant` lookup in
    C++; a bare Python `op in ignore_ops` would instead go through
    `set.__contains__`'s hash/`__eq__` machinery, which -- exactly this
    port's recurring "type before value" hazard -- can true-positive across
    alternatives a `boost::variant` never would (`hash(11.0) == hash(11)`,
    so a stray `contin_t` value numerically equal to some `builtin`'s
    ordinal could otherwise appear to "ignore" an unrelated operator).
    `op` here is always a known `builtin` constant (`id::sin`, `id::div`,
    ...); the scan is over `ignore_ops`'s actual members, whatever their
    real type turns out to be.
    """
    return any(vertex_equal(v, op) for v in ignore_ops)
