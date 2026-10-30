"""Constant tuning for regression programs (``--const-opt``, server round 3).

NOT A PORT (pyMoses extension; no C++ counterpart). See
``docs/gomea/SERVER-ROUND3-PREREGISTRATION.md`` for the design and
``docs/gomea/USAGE.md`` for the flags. Three modes use this module:

* ``lm``: at the end of each GOMEA deme, Levenberg-Marquardt (LM) on the
  numeric constants of the deme's best program (``levenberg_marquardt``);
* ``mutate``: after each GOM step, one Gaussian change to each constant of
  the individual (the optimizer does it; ``gaussian`` and
  ``mutation_sigma`` are here);
* ``polish``: at the end of the run, LM once on the final best program
  (``polish_metapopulation``), kept only if the validation NMSE is not worse.

LM works on the constants of the program TREE (every ``contin_t`` leaf of the
tree that the scorer sees), not on the contin-knob codes of a genotype. Why: a
contin knob can only hold the ``2 ** (depth + 1) - 1`` values of its step path,
so a tuned value usually has no code; and the reduced program can hold
constants that no knob gives (reduction merges constants). The tuned tree goes
back to MOSES as a deme entry that carries its own tree
(``scored_tree_instance``; ``merging.deme_to_trees`` uses that tree).

Cost: every pass over the training rows (one residual vector, one
finite-difference column, one trial step, one final score) calls ``charge()``
once before it runs. The caller makes ``charge`` count one evaluation against
the run's budget and raise ``BudgetExhausted`` when none is left.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pymoses.comboreduct.combo.vertex import combo_tree, get_contin, is_contin
from pymoses.comboreduct.interpreter.interpreter import interpreter_visitor
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.moses.scoring.bscores import contin_bscore, linear_scaling

CONST_OPT_MODES = ("none", "lm", "mutate", "polish")

# LM settings (pre-registration, "The constant-tuning code").
LM_MAX_ITERATIONS = 10
LM_REL_TOL = 1e-6
# Implementation settings the pre-registration does not give (docs/gomea/USAGE.md):
# the first Marquardt damping, the most trial steps (each with 10x the damping)
# per iteration, and the forward-difference step h = FD_STEP * max(|c|, 1).
LM_LAMBDA0 = 1e-3
LM_MAX_TRIALS = 10
FD_STEP = 1.4901161193847656e-08  # sqrt(2 ** -52)
# A Jacobian column whose norm is at most INERT_TOL times the larger of the
# largest column norm and the norm of the centred target is "inert": the
# constant does not change the residuals beyond
# finite-difference noise (for example an additive or multiplicative constant
# that linear scaling absorbs), and LM holds it fixed.
INERT_TOL = 1e-7
# A pivot below PIVOT_TOL times the largest diagonal entry is "singular".
PIVOT_TOL = 1e-12

# ``mutate``: sigma = max(SIGMA_REL * |c|, SIGMA_MIN).
SIGMA_REL = 0.1
SIGMA_MIN = 0.1

# ``polish``: the share of -m held back for the final LM call.
POLISH_RESERVE_SHARE = 0.05


class scored_tree_instance(scored_instance):
    """NOT A PORT: a deme entry that carries its own program ``tree``.

    ``first`` is a placeholder instance (the genotype with its tuned constants
    at code 0); the merge (``merging.deme_to_trees`` and
    ``keep_top_unique_candidates``) uses ``tree`` instead of decoding it."""

    __slots__ = ("tree",)

    def __init__(self, i, s, tree: combo_tree) -> None:
        super().__init__(i, s)
        self.tree = combo_tree(tree.begin())


def find_contin_bscore(cscorer) -> contin_bscore:
    """The ``contin_bscore`` behind a ``behave_cscore``; ``ValueError`` if
    there is none (constant tuning is for regression only)."""
    b = cscorer.get_bscorer()
    if not isinstance(b, contin_bscore):
        raise ValueError(
            "--const-opt needs a regression problem scored by contin_bscore, "
            f"not {type(b).__name__}"
        )
    return b


class RegressionRows:
    """The rows a ``contin_bscore`` scores (inputs and targets), and the
    residuals of a program on them: ``(a + b f(x)) - y`` with the
    least-squares ``(a, b)`` when the scorer uses linear scaling, else
    ``f(x) - y``. ``scaling`` gives the ``(a, b)`` to use instead of a fit
    (``polish`` scores validation rows with the training fit)."""

    def __init__(self, inputs: Sequence[Sequence], targets: Sequence[float], scaled: bool):
        self.inputs = [list(r) for r in inputs]
        self.targets = [float(y) for y in targets]
        self.scaled = scaled
        mean = math.fsum(self.targets) / len(self.targets) if self.targets else 0.0
        # the norm of the centred target, the scale of an inert column
        self.spread = math.sqrt(math.fsum((y - mean) ** 2 for y in self.targets))

    @classmethod
    def of_bscore(cls, b: contin_bscore) -> RegressionRows:
        return cls(b._cti, [get_contin(v) for v in b._target], bool(b._linear_scaling))

    def predictions(self, tr: combo_tree) -> list[float]:
        iv = interpreter_visitor(tr)
        return [float(get_contin(iv(list(row)))) for row in self.inputs]

    def residuals(self, tr: combo_tree, scaling=None) -> tuple[list[float], tuple[float, float]]:
        """``(residuals, (a, b))``; non-finite residuals when the program
        cannot be evaluated on a row."""
        try:
            p = self.predictions(tr)
        except Exception:  # any evaluation error (overflow, domain) makes it non-finite
            return [math.inf] * len(self.targets), (0.0, 1.0)
        if scaling is None:
            scaling = linear_scaling(self.targets, p) if self.scaled else (0.0, 1.0)
        a, b = scaling
        return [a + b * x - y for x, y in zip(p, self.targets, strict=True)], (a, b)


def mse(r: Sequence[float]) -> float:
    s = math.fsum(x * x for x in r) / len(r)
    return s if math.isfinite(s) else math.inf


def constant_nodes(tr: combo_tree) -> list:
    """The ``contin_t`` leaves of ``tr`` in pre-order."""
    return [n for n in tr if is_contin(n.data)]


def solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """``a x = b`` by Gaussian elimination with partial pivoting; ``None``
    when a pivot is below ``PIVOT_TOL`` times the largest diagonal entry
    (numerically singular) or the result is not finite."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    scale = max((abs(a[i][i]) for i in range(n)), default=0.0)
    if not (scale > 0 and math.isfinite(scale)):
        return None
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) <= PIVOT_TOL * scale:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (m[c][n] - math.fsum(m[c][k] * x[k] for k in range(c + 1, n))) / m[c][c]
    return x if all(math.isfinite(v) for v in x) else None


@dataclass
class LMResult:
    """One LM call. ``status``: ``improved`` (the training MSE went down),
    ``not_improved``, ``singular`` (the damped normal equations could not be
    solved, or no constant changes the residuals), ``failed`` (the program
    or a Jacobian column is not finite), ``no_constants``, or ``budget`` (the
    budget ran out before the first pass). The flag ``budget`` is set
    whenever the budget ran out during the call (the best point found so far
    is kept). ``tree`` is the tuned copy (``None`` unless improved)."""

    status: str
    n_constants: int
    mse0: float
    mse: float
    evals: int = 0
    iterations: int = 0
    n_inert: int = 0
    budget: bool = False
    tree: combo_tree | None = None
    values: list[float] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "status": self.status,
            "n_constants": self.n_constants,
            "n_inert": self.n_inert,
            "mse0": self.mse0,
            "mse": self.mse,
            "evals": self.evals,
            "iterations": self.iterations,
            "budget": self.budget,
        }


def levenberg_marquardt(
    tr: combo_tree,
    rows: RegressionRows,
    charge: Callable[[], None],
    max_iterations: int = LM_MAX_ITERATIONS,
    rel_tol: float = LM_REL_TOL,
) -> LMResult:
    """LM on every constant of ``tr`` (a copy; ``tr`` is not changed),
    minimising the training MSE of ``rows`` (after linear scaling when the
    rows use it, refitted at every pass).

    Per iteration: a forward-difference Jacobian (one pass per constant), then
    trial steps from ``(J'J + lambda diag(J'J)) d = -J'r`` (Marquardt damping),
    one pass each, until a step lowers the MSE (lambda / 10) or
    ``LM_MAX_TRIALS`` steps fail (lambda x 10 each; LM then stops). A
    constant whose Jacobian column is zero (it does not change the
    residuals beyond finite-difference noise, ``INERT_TOL``; for example an
    additive constant that linear scaling absorbs) is held fixed. Stops after
    ``max_iterations`` iterations or when an
    accepted step lowers the MSE by less than ``rel_tol`` relative to the
    MSE before it."""
    work = combo_tree(tr.begin())
    nodes = constant_nodes(work)
    k = len(nodes)
    theta = [float(get_contin(n.data)) for n in nodes]
    res = LMResult("budget", k, math.inf, math.inf)

    def passes(values: list[float]) -> list[float]:
        charge()
        res.evals += 1
        for n, v in zip(nodes, values, strict=True):
            n.data = float(v)
        return rows.residuals(work)[0]

    best = list(theta)
    try:
        r = passes(theta)
        res.mse0 = res.mse = cur = mse(r)
        if k == 0:
            res.status = "no_constants"
            return res
        if not math.isfinite(cur):
            res.status = "failed"
            return res
        res.status = "not_improved"
        lam = LM_LAMBDA0
        for _ in range(max_iterations):
            res.iterations += 1
            cols: list[list[float]] = []
            for j in range(k):
                h = FD_STEP * max(abs(theta[j]), 1.0)
                tj = list(theta)
                tj[j] += h
                rj = passes(tj)
                col = [(x - y) / h for x, y in zip(rj, r, strict=True)]
                if not all(math.isfinite(v) for v in col):
                    res.status = "failed" if res.status != "improved" else res.status
                    raise _Stop
                cols.append(col)
            norms = [math.sqrt(math.fsum(x * x for x in c)) for c in cols]
            ref = max(max(norms), rows.spread)
            active = [j for j in range(k) if norms[j] > INERT_TOL * ref]
            res.n_inert = k - len(active)
            if not active:
                if res.status != "improved":
                    res.status = "singular"
                break
            jtj = [
                [math.fsum(x * y for x, y in zip(cols[i], cols[j], strict=True)) for j in active]
                for i in active
            ]
            g = [math.fsum(x * y for x, y in zip(cols[i], r, strict=True)) for i in active]
            accepted = False
            for _t in range(LM_MAX_TRIALS):
                m = [row[:] for row in jtj]
                for d in range(len(active)):
                    m[d][d] += lam * jtj[d][d]
                step = solve(m, [-x for x in g])
                if step is None:
                    if res.status != "improved":
                        res.status = "singular"
                    raise _Stop
                trial = list(theta)
                for d, j in enumerate(active):
                    trial[j] += step[d]
                rt = passes(trial)
                mt = mse(rt)
                if mt < cur:
                    rel = (cur - mt) / cur
                    theta, r, cur = trial, rt, mt
                    best = list(theta)
                    res.status = "improved"
                    res.mse = cur
                    lam = max(lam / 10.0, 1e-12)
                    accepted = True
                    break
                lam *= 10.0
            if not accepted or rel < rel_tol or cur == 0.0:
                break
    except BudgetExhausted:
        res.budget = True
    except _Stop:
        pass
    if res.status == "improved":
        for n, v in zip(nodes, best, strict=True):
            n.data = float(v)
        res.tree = work
        res.values = best
    return res


class _Stop(Exception):
    pass


def mutation_sigma(c: float) -> float:
    """``max(0.1 |c|, 0.1)`` (pre-registration, ``mutate``)."""
    return max(SIGMA_REL * abs(c), SIGMA_MIN)


def gaussian(rng) -> float:
    """A standard normal draw (Box-Muller, two ``randdouble`` draws)."""
    u1 = 1.0 - rng.randdouble()
    u2 = rng.randdouble()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def validation_nmse(tr: combo_tree, train: RegressionRows, val: RegressionRows) -> float:
    """NMSE of ``tr`` on the validation rows with the linear scaling fitted
    on the training rows (as ``tools/gomea/sr_eval.py`` scores a report),
    divided by the variance of the validation target."""
    try:
        p = train.predictions(tr)
    except Exception:  # any evaluation error makes it non-finite
        return math.inf
    a, b = linear_scaling(train.targets, p)
    r, _ = val.residuals(tr, (a, b))
    n = len(val.targets)
    mean = math.fsum(val.targets) / n
    var = math.fsum((y - mean) ** 2 for y in val.targets) / n
    e = mse(r)
    return e / var if var > 0 else e


def load_rows(path: str, labels: Sequence[str], scaled: bool) -> RegressionRows:
    """Validation rows from a table file with the training columns
    (``labels``, in that order) and target ``y``."""
    from pymoses.comboreduct.table.table_io import loadTable

    t = loadTable(path, "y")
    if list(t.itable.get_labels()) != list(labels):
        raise ValueError(
            f"--const-opt-val {path}: input columns {t.itable.get_labels()} differ from "
            f"the training columns {list(labels)}"
        )
    return RegressionRows(t.itable, [get_contin(v) for v in t.otable], scaled)


def polish_metapopulation(metapop, cscorer, budget: int, val_path: str) -> dict:
    """``--const-opt polish``: LM once on the final best program (the first
    of ``metapop.get_trees()``, the first one MOSES prints) with at most
    ``budget`` evaluations (the final score included). The tuned program
    replaces it in the metapopulation only if LM lowered the training MSE
    and the validation NMSE is not worse. Returns the record for the
    telemetry (``evals`` is what to add to the run's evaluation count)."""
    from pymoses.moses.moses.types import scored_combo_tree, scored_combo_tree_set

    b = find_contin_bscore(cscorer)
    train = RegressionRows.of_bscore(b)
    val = load_rows(val_path, b._cti.get_labels(), train.scaled)
    trees = list(metapop.get_trees())
    rec: dict = {"accepted": False, "budget_available": budget}
    if not trees:
        rec.update(status="empty", evals=0)
        return rec
    old = trees[0]
    used = [0]
    limit = [budget - 1]  # one evaluation stays for scoring the tuned program

    def charge() -> None:
        if used[0] >= limit[0]:
            raise BudgetExhausted("global_budget")
        used[0] += 1

    lm = levenberg_marquardt(old.get_tree(), train, charge)
    limit[0] = budget
    rec.update(lm.record())
    if lm.tree is not None:
        v_old = validation_nmse(old.get_tree(), train, val)
        v_new = validation_nmse(lm.tree, train, val)
        rec.update(val_nmse_old=v_old, val_nmse_new=v_new)
        if v_new <= v_old:
            try:
                charge()
                score = cscorer.get_cscore(lm.tree)
            except BudgetExhausted:
                rec["budget"] = True
            else:
                new = scored_combo_tree(lm.tree, old.get_demeID(), copy.copy(score))
                metapop._scored_trees.erase(old)
                metapop._scored_trees.insert(new)
                cands = scored_combo_tree_set()
                cands.insert(new)
                metapop.update_best_candidates(cands)
                rec["accepted"] = True
    rec["evals"] = used[0]
    return rec
