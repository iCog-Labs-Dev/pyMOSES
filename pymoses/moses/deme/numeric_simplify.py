"""Numerical simplification of regression exemplars (port-only, opt-in).

NOT A PORT: classic MOSES has no counterpart. This module is a pyMoses
extension enabled by ``--simplify-exemplar-numeric TOL`` (default ``0``,
off). With the option off nothing here is constructed or called, so a run is
unchanged.

What it does. When MOSES has selected an exemplar and is about to build
knobs around it (``deme_expander.create_demes``, right before
``create_representations``), every non-leaf subtree of the exemplar is
evaluated on the training inputs. A subtree whose outputs are all finite
contin values and whose range ``max - min`` is below ``TOL * scale`` is a
*near-constant* subtree and is a candidate for replacement by a single
constant, the mean of its outputs (the range-to-constant rule of Kinzett et
al. 2009). It acts on exemplars, so it applies equally to every optimizer
(``hc``, ``hc-capped``, ``gomea``, ...).

Scale. ``scale`` is the population standard deviation of the training
target. The tolerance is therefore relative, as the literature requires (a
fixed absolute tolerance failed on Yacht in Aldeia et al. 2024), and in the
units the error is measured in: NMSE divides the squared error by the
target's variance, so a run whose training NMSE floor is ``e`` has residuals
of about ``sqrt(e) * scale``, and ``TOL = sqrt(e)`` puts the threshold at
that noise level (Kinzett et al. 2010 found the best threshold between one
and five times the noise level). The target's spread is fixed for the run
and does not depend on the exemplar, so the same TOL means the same thing
at every expansion. A subtree's own output can be in other units than the
target (it may be scaled further up the tree); the guard below is what
protects against a replacement that matters more than its range suggests.
A constant target (``scale == 0``) disables the pass.

Guard. There is no validation split inside MOSES (the regression runner
keeps validation and test data outside), so a replacement is checked on
the training data, with the problem's own scorer: the same composite score
(raw score with MOSES's complexity penalty) MOSES ranks candidates by.
Candidates are tried one at a time, in pre-order (largest subtree first);
each trial tree gets the candidate replaced by its constant and is reduced
with the candidate reducer (the one MOSES applies to every candidate). The
trial is kept only if its composite score is not worse than the current
tree's (``not (trial < current)``), so the final exemplar is never worse
than the original on the training score. When a candidate is kept its
descendants are gone; when it is rejected, its descendants are still tried.
If nothing is kept the original exemplar is returned unchanged (the same
object).

These extra scorer calls are not charged to ``-m`` and do not touch the
scorer's cache (``get_cscore_nocache``): they are bookkeeping of the
exemplar, not search evaluations.

Telemetry. Each call logs one INFO line, ``Numeric exemplar
simplification: ...``, with this expansion's counts (candidates, accepted,
rejected, nodes before/after) and the run's running totals, so experiments
can report how often it fired and how many nodes it removed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pymoses.comboreduct.combo.vertex import combo_tree, is_contin
from pymoses.comboreduct.interpreter.interpreter import interpreter_visitor
from pymoses.util.logger import logger

__all__ = ["exemplar_numeric_simplifier", "subtree_outputs"]


def subtree_outputs(node, rows: Sequence[Sequence]) -> list[float] | None:
    """The outputs of the subtree rooted at ``node`` on every row, or
    ``None`` if any output is not a finite contin value or evaluation
    raises (such a subtree is never a candidate)."""
    sub = combo_tree(node)
    try:
        iv = interpreter_visitor(sub)
        out = [iv(list(r)) for r in rows]
    except Exception:  # noqa: BLE001 -- an unevaluable subtree is just skipped
        return None
    for v in out:
        if not is_contin(v) or not math.isfinite(v):
            return None
    return out


class exemplar_numeric_simplifier:
    """NOT A PORT (see module docstring): replace near-constant subtrees of
    an exemplar by constants, guarded by the training score.

    ``rows`` are the training input rows, ``targets`` the training target
    values (used only for the scale), ``tol`` the relative tolerance (> 0).
    Counters accumulate over the run: ``calls`` (exemplars examined),
    ``fired`` (exemplars changed), ``candidates``, ``accepted``,
    ``rejected``, ``nodes_removed``.
    """

    def __init__(self, rows: Sequence[Sequence], targets: Sequence[float], tol: float) -> None:
        if not (tol > 0 and math.isfinite(tol)):
            raise ValueError(f"tol must be a finite number > 0, got {tol}")
        self.rows = [list(r) for r in rows]
        ys = [float(y) for y in targets]
        mean = sum(ys) / len(ys) if ys else 0.0
        self.scale = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys)) if ys else 0.0
        self.tol = tol
        self.threshold = tol * self.scale
        self.calls = 0
        self.fired = 0
        self.candidates = 0
        self.accepted = 0
        self.rejected = 0
        self.nodes_removed = 0

    def near_constant(self, node) -> float | None:
        """The mean of the subtree's outputs if it is near-constant (range
        strictly below the threshold), else ``None``. Leaves are never
        near-constant candidates (a constant is already one; an argument is
        an input)."""
        if node.first_child is None:
            return None
        out = self.subtree_outputs(node)
        if out is None or not out:
            return None
        if max(out) - min(out) >= self.threshold:
            return None
        return math.fsum(out) / len(out)

    def subtree_outputs(self, node) -> list[float] | None:
        return subtree_outputs(node, self.rows)

    def __call__(
        self,
        exemplar: combo_tree,
        score: Callable[[combo_tree], object],
        reduce: Callable[[combo_tree], None],
    ) -> combo_tree:
        """Return the simplified exemplar, or ``exemplar`` itself (the same
        object, untouched) when no replacement passed the guard.

        ``score`` is the problem's composite scorer (a tree -> a
        ``composite_score``); ``reduce`` the candidate reducer, applied in
        place."""
        self.calls += 1
        n_before = exemplar.size()
        cand = acc = rej = 0
        best = exemplar
        if self.threshold > 0 and not exemplar.empty():
            cur = combo_tree(exemplar.begin())
            cur_score = score(exemplar)
            node = cur.begin()
            while node is not None:
                mean = self.near_constant(node)
                if mean is None:
                    node = cur.next_pre_order(node)
                    continue
                cand += 1
                # Trial: a copy of `cur` with this node replaced, reduced.
                path = _path_to(node)
                trial = combo_tree(cur.begin())
                t_node = _node_at(trial, path)
                trial.erase_children(t_node)
                t_node.data = float(mean)
                reduce(trial)
                t_score = score(trial)
                if t_score < cur_score:
                    rej += 1
                    node = cur.next_pre_order(node)
                    continue
                acc += 1
                # Keep it: apply the same replacement to `cur` (unreduced, so
                # the remaining node handles stay valid) and skip its subtree.
                nxt = cur.subtree_pre_order_end(node)
                cur.erase_children(node)
                node.data = float(mean)
                best, cur_score = trial, t_score
                node = nxt
        n_after = best.size()
        self.candidates += cand
        self.accepted += acc
        self.rejected += rej
        if best is not exemplar:
            self.fired += 1
            self.nodes_removed += n_before - n_after
        if logger().is_info_enabled():
            logger().info(
                "Numeric exemplar simplification: "
                f"candidates={cand} accepted={acc} rejected={rej} "
                f"nodes={n_before}->{n_after} threshold={self.threshold:.6g} | "
                f"totals: calls={self.calls} fired={self.fired} "
                f"candidates={self.candidates} accepted={self.accepted} "
                f"rejected={self.rejected} nodes_removed={self.nodes_removed}"
            )
        return best


def _path_to(node) -> list[int]:
    """Child indices from the root down to ``node``."""
    path = []
    while node.parent is not None:
        i = 0
        sib = node.parent.first_child
        while sib is not node:
            sib = sib.next_sibling
            i += 1
        path.append(i)
        node = node.parent
    path.reverse()
    return path


def _node_at(tr: combo_tree, path: list[int]):
    node = tr.begin()
    for i in path:
        node = tr.child(node, i)
    return node
