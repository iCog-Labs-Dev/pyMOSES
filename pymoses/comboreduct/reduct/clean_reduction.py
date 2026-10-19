"""C++: moses/comboreduct/reduct/clean_reduction.cc

`clean_reduce` -- removes `null_vertex` sentinels a reduction pass left
behind (e.g. `logical_rules.py`'s `reduce_remove_subtree_equal_tt`, which
wraps a candidate-for-removal subtree in `id::null_vertex` before testing
whether removing it changes the tree's truth table). Small file: one rule
composition (`clean_reduction`, a cached singleton `sequential` of two
already-ported rules) plus the `clean_reduce` convenience wrappers
`reduct.h:199-207` declares alongside it.

Both wrapped rules are untyped (`downwards(remove_null_vertices())`,
`upwards(remove_dangling_junctors())` -- clean_reduction.cc:51-52 passes
no `type_node` to either `downwards` or `upwards`), so this file needs
none of the `type_checker` slice `meta_rules.py`'s typed `downwards`
branch needed -- unlike `logical_reduction.py`.

`remove_null_vertices` (`general_rules.py`, Task 3) and
`remove_dangling_junctors` (`logical_rules.py`, Task 4) were both already
fully ported and tested by earlier tasks in this cycle; this file only
composes them, exactly as `clean_reduction.cc:51-52` does.
"""

from __future__ import annotations

from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.general_rules import remove_null_vertices
from pymoses.comboreduct.reduct.logical_rules import remove_dangling_junctors
from pymoses.comboreduct.reduct.meta_rules import downwards, sequential, upwards
from pymoses.comboreduct.reduct.reduct import _NO_IT, rule
from pymoses.util.tree import tree_node

_singleton: rule | None = None


def clean_reduction() -> rule:
    """C++: moses/comboreduct/reduct/clean_reduction.cc:32 clean_reduction

    C++ caches a single, lazily-constructed `static sequential` behind a
    double-checked-locking `static rule* pr` (clean_reduction.cc:34-46) --
    the same pattern, and the same reason, `logical_reduction::do_init`
    uses for `p_extra_simple`/`p_simple` (see `logical_reduction.py`'s own
    `_do_init` docstring). This port has no concurrent construction to
    race (module-level Python execution is single-threaded and each
    module body runs at most once), so the module-level `_singleton`
    global below reproduces only the OBSERVABLE half of that pattern --
    "the same rule object is reused across calls" -- not the locking
    machinery itself, which nothing here can ever race.
    """
    global _singleton
    if _singleton is None:
        _singleton = sequential(
            downwards(remove_null_vertices()),
            upwards(remove_dangling_junctors()),
        )
    return _singleton


def clean_reduce(tr: combo_tree, it: tree_node = _NO_IT) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:199 clean_reduce

    Two C++ overloads collapsed into one Python signature (this port's
    usual convention for an overloaded C++ free function): `clean_reduce
    (tr, it)` (reduct.h:199-202) and `clean_reduce(tr)` (reduct.h:204-207,
    `it` omitted -> `clean_reduction()(tr)`, the 1-arg `rule.__call__`
    overload, itself `tr.begin()` if `tr` is non-empty). `it`'s default is
    `reduct.py`'s own `_NO_IT` sentinel (not a bare `None`) so that an
    explicit `clean_reduce(tr, None)` still raises rather than silently
    behaving like the 1-arg overload -- see `rule.__call__`'s own
    docstring for why that distinction is load-bearing.
    """
    clean_reduction()(tr, it)


_dangling_singleton: rule | None = None


class _HasCond(Exception):
    """A kept `cond` node: `remove_null_vertices` treats its children as pairs."""


def _prune_null(node: tree_node) -> bool:
    data = node.data
    if isinstance(data, builtin):
        if data == builtin.null_vertex:
            return True
        if data == builtin.cond:
            raise _HasCond
    return False


def clean_reduced_copy(it: tree_node) -> combo_tree:
    """No C++ counterpart: `combo_tree tmp(it); clean_reduce(tmp);` in one
    step, as `build_knobs::disc_probe` does for every probed knob setting.

    `clean_reduction`'s first pass, `downwards(remove_null_vertices())`,
    erases each `null_vertex`-rooted child of each node it visits, so the
    erased subtrees are never visited; outside `cond`, the tree it leaves
    is exactly the copy with those subtrees left out. This builds that
    copy directly (most of a decorated exemplar is disallowed knob
    settings under `null_vertex`), then runs the second pass,
    `upwards(remove_dangling_junctors())`, on it; the first pass would be a
    no-op there. A tree that keeps a `cond` node, whose children
    `remove_null_vertices` handles in pairs, takes the literal path.
    """
    global _dangling_singleton
    if not (isinstance(it.data, builtin) and it.data == builtin.cond):
        try:
            tmp = combo_tree.pruned_copy(it, _prune_null)
        except _HasCond:
            pass
        else:
            if _dangling_singleton is None:
                _dangling_singleton = upwards(remove_dangling_junctors())
            _dangling_singleton(tmp)
            return tmp
    tmp = combo_tree(it)
    clean_reduce(tmp)
    return tmp
