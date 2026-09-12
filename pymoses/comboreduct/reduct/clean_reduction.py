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

from pymoses.comboreduct.combo.vertex import combo_tree
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
