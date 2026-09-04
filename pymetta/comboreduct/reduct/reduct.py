"""C++: moses/comboreduct/reduct/reduct.{h,cc}

Scoped to exactly what Task 2 (S7) needs and produces: the `rule` base
class -- every reduction rule and every meta-rule combinator
(`meta_rules.py`, this same task) is a `rule`. Everything else reduct.h
declares (`logical_reduction`, `contin_reduction`, `fold_reduction`,
`mixed_reduction`, `full_reduction`, `action_reduction`,
`perception_reduction`, `clean_reduction`, and the `xxx_reduce` inline
convenience wrappers around them) depends on rule SETS Tasks 3-5 build;
porting their entry points here now would just be dead code with nothing
to call. `docs/superpowers/plans/2026-09-04-pymetta-s7-reduct.md`'s own
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

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.util.tree import tree_node


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

    def __call__(self, tr: combo_tree, it: tree_node | None = None) -> None:
        """C++: moses/comboreduct/reduct/reduct.h:44 rule::operator()
        (combo_tree&) [concrete] -- the pure virtual half of the same
        overloaded C++ call operator (reduct.h:41) is reached here via
        `apply`.

        Covers both C++ overloads in one Python method, the same
        convention this port uses throughout for an overloaded C++ call
        operator (e.g. interpreter.py's `boolean_interpreter.__call__`):
        `it is None` selects the 1-arg overload -- if `tr` is not empty,
        recurse with `it = tr.begin()` (reduct.h:46-47, `(*this)(tr,
        tr.begin())`, itself a call THROUGH this same method, landing on
        the `it is not None` branch below on the way back in). An empty
        `tr` with no `it` given is a silent no-op, matching C++'s `if
        (!tr.empty())` guard exactly.
        """
        if it is None:
            if not tr.empty():
                self(tr, tr.begin())
            return
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
