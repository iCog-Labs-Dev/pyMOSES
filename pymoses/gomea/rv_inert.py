"""NOT A PORT, T3.2 (server round 16): the structural inert rule of ``--gomea-rv-step rvg``.

The rule comes from the T3.2 spikes (``docs/gomea/T3.2-FEASIBILITY.md``; spike commits
3b8a4d3 and be83e08, module ``rv_step.py``). ``rv_core.RVCore`` uses it to find the active
constant genes of an individual. Only the parts that the RV core uses are here. The probe and
the ``uni`` and ``block`` RV steps of the first spike are left out.

A constant gene of a program (``--gomea-contin copy``) is one of:

* ``zero``: its value is 0 (the term is not there);
* ``dead``: a zero factor above it makes its value have no effect;
* ``absorbed``: the linear scaling of the scorer (``--linear-scaling terms``: the weight of its
  top-level term and the intercept; else ``a + b f``) absorbs any change of its value;
* ``live``: a change of its value can change the training error.

``dead`` and ``absorbed`` are the structurally inert genes. The rule is conservative: an
operator that is not ``+``, ``*`` or ``/`` makes a gene below it ``live``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pymoses.comboreduct.combo.vertex import builtin, combo_tree, get_contin, is_argument, is_contin
from pymoses.moses.scoring.bscores import top_level_terms

PLAIN = "plain"
TERMS = "terms"


def _is(v, b) -> bool:
    return isinstance(v, builtin) and v is b


def _children(node) -> list:
    out = []
    c = node.first_child
    while c is not None:
        out.append(c)
        c = c.next_sibling
    return out


def _const(node, memo: dict) -> float | None:
    """The value of the subtree at ``node`` when it does not depend on the inputs,
    else ``None``. A product with a zero factor is 0. Conservative: an operator that
    is not ``+``, ``*`` or ``/`` gives ``None`` unless the subtree is a number."""
    k = id(node)
    if k in memo:
        return memo[k]
    d = node.data
    v: float | None
    if is_contin(d):
        v = float(get_contin(d))
    elif is_argument(d):
        v = None
    elif _is(d, builtin.plus):
        ks = [_const(c, memo) for c in _children(node)]
        v = math.fsum(ks) if all(x is not None for x in ks) else None
    elif _is(d, builtin.times):
        ks = [_const(c, memo) for c in _children(node)]
        if any(x == 0.0 for x in ks if x is not None):
            v = 0.0
        elif all(x is not None for x in ks):
            v = math.prod(ks)
        else:
            v = None
    elif _is(d, builtin.div):
        kids = _children(node)
        num = _const(kids[0], memo)
        den = _const(kids[1], memo) if len(kids) > 1 else None
        if num == 0.0 and den != 0.0:
            v = 0.0
        elif num is not None and den is not None and den != 0.0:
            v = num / den
        else:
            v = None
    else:
        v = None
    memo[k] = v
    return v


def _chain_nodes(root, memo: dict) -> set[int]:
    """The ids of the "chain" nodes of the unreduced tree: the nodes that the reducer
    flattens into the root sum. A ``+`` is a chain node. A ``*`` (or ``/``) whose
    other children (or denominator) are constant, with one non-constant child that is
    a chain node, is a chain node too (the reducer distributes a constant over a sum of
    ``*(c x)`` terms). Only nodes reached from the root through chain nodes count. The
    children of chain nodes that are not chain nodes are the top-level terms."""
    out: set[int] = set()

    def chainable(node) -> bool:
        d = node.data
        if _is(d, builtin.plus):
            return True
        if _is(d, builtin.times):
            var = [c for c in _children(node) if _const(c, memo) is None]
            return len(var) == 1 and chainable(var[0])
        if _is(d, builtin.div):
            kids = _children(node)
            if len(kids) != 2:
                return False
            den = _const(kids[1], memo)
            return den is not None and den != 0.0 and chainable(kids[0])
        return False

    def visit(node) -> None:
        if not chainable(node):
            return
        out.add(id(node))
        for c in _children(node):
            visit(c)

    if root is not None:
        visit(root)
    return out


def fit_mode(reduced: combo_tree) -> str:
    """The fit that the scorer uses for the reduced program (``--linear-scaling terms``):
    ``terms`` (per-term least squares) when the root is a ``+`` with at least two
    children and no numeric child, else ``plain`` (a + b f). The rule does not predict
    the fallbacks for values that are not finite."""
    nodes = top_level_terms(reduced)
    if len(nodes) < 2 or any(is_contin(n.data) for n in nodes):
        return PLAIN
    return TERMS


def _walk(leaf, chain: set[int], mode: str, memo: dict) -> str:
    """``"dead"``, ``"absorbed"`` or ``"live"`` for the constant ``leaf``.

    The state is whether a change of the value changes the current node by a pure
    scale (S), by a pure shift (T), or by an affine map (A) of its old value. At the
    leaf all three hold. With the ``terms`` fit the walk stops at a top-level term (a
    child of a chain node): a term that changes by an affine map is absorbed by its
    weight and the intercept. With the ``plain`` fit the walk goes to the root."""
    # First: a zero factor anywhere above the leaf makes the gene dead.
    node = leaf
    while node.parent is not None:
        par = node.parent
        d = par.data
        if _is(d, builtin.times):
            if any(_const(c, memo) == 0.0 for c in _children(par) if c is not node):
                return "dead"
        elif _is(d, builtin.div):
            kids = _children(par)
            if node is not kids[0] and _const(kids[0], memo) == 0.0:
                return "dead"
        node = par
    S = T = A = True
    node = leaf
    if mode == TERMS and node.parent is not None and id(node.parent) in chain:
        return "absorbed"  # an additive constant, a scale of terms, or a denominator scale
    while node.parent is not None:
        if mode == TERMS and id(node.parent) in chain:
            break
        par = node.parent
        d = par.data
        others = [c for c in _children(par) if c is not node]
        if _is(d, builtin.plus):
            ks = [_const(c, memo) for c in others]
            if all(x is not None for x in ks):
                S = S and math.fsum(ks) == 0.0
            else:
                S = False
                A = T
        elif _is(d, builtin.times):
            ks = [_const(c, memo) for c in others]
            if any(x == 0.0 for x in ks if x is not None):
                return "dead"
            if not all(x is not None for x in ks):
                T = False
                A = S
        elif _is(d, builtin.div):
            kids = _children(par)
            if node is kids[0]:
                den = _const(kids[1], memo)
                if den is None:
                    T = False
                    A = S
            else:
                num = _const(kids[0], memo)
                if num == 0.0:
                    return "dead"
                T = False
                A = S
        else:
            return "live"
        if not A:
            return "live"
        node = par
    return "absorbed" if A else "live"


def analyse(schema, g: Sequence[int], reduce: bool = True, mode: str | None = None) -> dict:
    """The structural analysis of genotype ``g``: ``{"mode": fit mode, "genes": {gene:
    "zero" | "dead" | "absorbed" | "live"}, "reduced": the reduced program}`` for each
    contin gene of the unreduced program. ``dead`` and ``absorbed`` are the structurally
    inert genes.

    NOT A PORT, backlog 15.4 (D1, ``best_genes.gene_classes``): ``mode`` (``PLAIN`` or
    ``TERMS``), when given, is the fit mode in place of ``fit_mode`` of the reduced program
    (``PLAIN`` for ``--linear-scaling 1``). ``None`` (the default) keeps the rule of the RV
    core."""
    if mode not in (None, PLAIN, TERMS):
        raise ValueError(f"mode must be None, {PLAIN!r} or {TERMS!r}, got {mode!r}")
    tree, genes = schema.candidate_genes(g)
    reduced = schema.candidate(g, reduce)
    if mode is None:
        mode = fit_mode(reduced)
    memo: dict = {}
    chain = _chain_nodes(tree.begin(), memo) if mode == TERMS else set()
    leaves = [n for n in tree if is_contin(n.data)]
    out: dict[int, str] = {}
    for leaf, p in zip(leaves, genes, strict=True):
        if p is None:
            continue
        if float(get_contin(leaf.data)) == 0.0:
            out[p] = "zero"
        else:
            out[p] = _walk(leaf, chain, mode, memo)
    return {"mode": mode, "genes": out, "reduced": reduced}
