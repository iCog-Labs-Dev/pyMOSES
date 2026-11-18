"""Constant tuning for regression programs (``--const-opt``, server round 3).

NOT A PORT (pyMoses extension; no C++ counterpart). See
``docs/gomea/SERVER-ROUND3-PREREGISTRATION.md`` for the design and
``docs/gomea/USAGE.md`` for the flags. Server round 5 adds the validation guard of
``lm`` (``guard_accepts``, ``docs/gomea/SERVER-ROUND5-PREREGISTRATION.md``). Server round 13
(T3.1) adds LM with ``--linear-scaling terms`` (``RegressionRows`` with ``terms``,
``docs/gomea/T3.1-FEASIBILITY.md``) and the pole check of ``lm`` (``pole_check``,
``--const-opt-pole-check box``, BASE-PS, ``docs/gomea/SERVER-ROUND13-PREREGISTRATION.md``).
Three modes use this module:

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
from pymoses.moses.scoring.bscores import (
    contin_bscore,
    fit_program_terms,
    linear_scaling,
    term_columns,
    top_level_terms,
)

# ``solve`` and ``PIVOT_TOL`` moved to ``least_squares`` in server round 12 (unchanged), so
# that the scorer can use the solver too. They stay importable from here.
from pymoses.moses.scoring.least_squares import PIVOT_TOL, predict_terms, solve  # noqa: F401
from pymoses.util.rng import MT19937RandGen

CONST_OPT_MODES = ("none", "lm", "mutate", "polish")
CONST_OPT_GUARDS = ("none", "val")
# NOT A PORT, server round 12 (``--const-opt-during``): ``none`` (off) or ``elite`` (LM on
# each offspring that becomes a new deme best, during the search).
CONST_OPT_DURING_MODES = ("none", "elite")
# NOT A PORT, server round 13 (BASE-PS, ``--const-opt-pole-check``): ``none`` (off) or
# ``box`` (the pole check after LM, ``pole_check``). The number of probe points, the margin
# of the probe box (a share of the range of each input) and the factor of the target range
# (pre-registration, "BASE-PS").
POLE_CHECK_MODES = ("none", "box")
POLE_PROBES = 256
POLE_MARGIN = 0.1
POLE_FACTOR = 10.0

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
# A pivot below PIVOT_TOL times the largest diagonal entry is "singular" (``PIVOT_TOL``,
# 1e-12, is defined in ``least_squares``, with ``solve``).

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
    (``polish`` scores validation rows with the training fit).

    NOT A PORT, server round 13 (T3.1, ``--linear-scaling terms`` with ``--const-opt lm``):
    with ``terms``, ``residuals`` uses the scorer's per-term rule
    (``bscores.fit_program_terms``): ``w_0 + sum_i w_i t_i - y``, with the weights fitted on
    these rows at each call. A program with one term, or whose per-term fit falls back, gets
    the plain ``(a + b f(x)) - y`` (``scaled`` is then true), as the scorer does."""

    def __init__(
        self,
        inputs: Sequence[Sequence],
        targets: Sequence[float],
        scaled: bool,
        terms: bool = False,
    ):
        self.inputs = [list(r) for r in inputs]
        self.targets = [float(y) for y in targets]
        self.scaled = scaled
        self.terms = terms
        mean = math.fsum(self.targets) / len(self.targets) if self.targets else 0.0
        # the norm of the centred target, the scale of an inert column
        self.spread = math.sqrt(math.fsum((y - mean) ** 2 for y in self.targets))

    @classmethod
    def of_bscore(cls, b: contin_bscore) -> RegressionRows:
        targets = [get_contin(v) for v in b._target]
        if getattr(b, "_linear_scaling_terms", False):
            # Server round 13 (T3.1): the scorer's per-term rule. Its plain path is the
            # a + b*f fit (``contin_bscore._scaled_bscore``).
            return cls(b._cti, targets, True, terms=True)
        return cls(b._cti, targets, bool(b._linear_scaling))

    def predictions(self, tr: combo_tree) -> list[float]:
        iv = interpreter_visitor(tr)
        return [float(get_contin(iv(list(row)))) for row in self.inputs]

    def term_fit(self, tr: combo_tree):
        """NOT A PORT, server round 13: ``(number of terms, fit)`` of ``tr`` on these rows,
        by the rule of the scorer (``bscores.fit_program_terms``)."""
        return fit_program_terms(tr, self.inputs, self.targets, interpreter_visitor(tr))

    def residuals(self, tr: combo_tree, scaling=None) -> tuple[list[float], tuple[float, ...]]:
        """``(residuals, (a, b))``; non-finite residuals when the program
        cannot be evaluated on a row. With ``terms`` and no ``scaling``, the
        per-term fit when it does not fall back: ``(residuals, (w_0, .., w_k))``."""
        if self.terms and scaling is None:
            _k, fit = self.term_fit(tr)
            if fit is not None and fit.fitted is not None:
                return (
                    [f - y for f, y in zip(fit.fitted, self.targets, strict=True)],
                    fit.weights,
                )
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


def structure_key(tr: combo_tree) -> tuple:
    """NOT A PORT, server round 12 (``--const-opt-top-k``): a key of the structure of ``tr``.
    Two trees have the same key iff they are the same tree when every constant (``contin_t``
    leaf) is replaced by one placeholder. Each other vertex keeps its type and value, as in
    ``behave_cscore._cache_key``."""
    root = tr.begin()
    if root is None:
        return ()

    def walk(node) -> tuple:
        children = []
        child = node.first_child
        while child is not None:
            children.append(walk(child))
            child = child.next_sibling
        data = node.data
        return (type(data), None if is_contin(data) else data, tuple(children))

    return walk(root)


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
    tune: Sequence[bool] | None = None,
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
    MSE before it.

    NOT A PORT, server round 12 (``--const-opt-during elite``): ``tune``, when given, has one
    flag for each constant of ``tr`` (``constant_nodes`` order). Only the flagged constants
    are LM parameters; the others keep their values. ``n_constants`` and ``values`` then
    count only the flagged constants. ``None`` (the default) tunes every constant, as
    before."""
    work = combo_tree(tr.begin())
    nodes = constant_nodes(work)
    if tune is not None:
        nodes = [n for n, t in zip(nodes, tune, strict=True) if t]
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
    divided by the variance of the validation target.

    NOT A PORT, server round 13 (T3.1): when ``train`` uses ``terms``, the per-term weights
    fitted on the training rows are applied to the terms on the validation rows (as
    ``sr_eval.evaluate(..., terms=True)`` does). A program with one term, or whose per-term
    fit falls back, gets the plain ``(a, b)`` above."""
    if train.terms:
        _k, fit = train.term_fit(tr)
        if fit is not None and fit.weights is not None:
            return _terms_validation_nmse(tr, fit.weights, val)
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


def _terms_validation_nmse(tr: combo_tree, weights: Sequence[float], val: RegressionRows) -> float:
    """NOT A PORT, server round 13 (T3.1): ``validation_nmse`` with the per-term ``weights``
    (fitted on the training rows). A term that cannot be evaluated on a validation row
    makes the NMSE infinite."""
    try:
        cols = term_columns(top_level_terms(tr), val.inputs, interpreter_visitor(tr))
    except Exception:  # any evaluation error makes it non-finite
        return math.inf
    p = predict_terms(weights, cols)
    r = [x - y for x, y in zip(p, val.targets, strict=True)]
    n = len(val.targets)
    mean = math.fsum(val.targets) / n
    var = math.fsum((y - mean) ** 2 for y in val.targets) / n
    e = mse(r)
    return e / var if var > 0 else e


def guard_accepts(
    old: combo_tree,
    new: combo_tree,
    train: RegressionRows,
    val: RegressionRows,
    charge: Callable[[], None],
) -> dict:
    """NOT A PORT (``--const-opt-guard val``, server round 5): is the validation
    NMSE of the tuned program ``new`` not worse than that of ``old``? Each of
    the two validation passes calls ``charge()`` once before it runs, and
    ``BudgetExhausted`` from ``charge`` goes to the caller. The linear scaling
    comes from the training rows (``validation_nmse``). Its training pass is not
    charged: LM already paid for that pass at the same constants. Returns
    ``accepted``, ``val_nmse_old`` and ``val_nmse_new``."""
    charge()
    v_old = validation_nmse(old, train, val)
    charge()
    v_new = validation_nmse(new, train, val)
    return {"accepted": v_new <= v_old, "val_nmse_old": v_old, "val_nmse_new": v_new}


def pole_probes(
    rows: RegressionRows, seed: int, n: int = POLE_PROBES, margin: float = POLE_MARGIN
) -> list[list[float]]:
    """NOT A PORT, server round 13 (BASE-PS): ``n`` probe points, uniform in the box
    ``[min_j - margin r_j, max_j + margin r_j]`` of each input ``j`` of ``rows``
    (``r_j = max_j - min_j``). The draws come from a new ``MT19937RandGen(seed)``, point by
    point and input by input (``lo + (hi - lo) * randdouble()``). Only that generator draws,
    so no other random stream changes. ``ValueError`` when an input is not numeric."""
    k = len(rows.inputs[0]) if rows.inputs else 0
    box = []
    for j in range(k):
        col = [row[j] for row in rows.inputs]
        if not all(is_contin(x) for x in col):
            raise ValueError("--const-opt-pole-check box needs numeric inputs")
        lo, hi = min(col), max(col)
        r = hi - lo
        box.append((lo - margin * r, hi + margin * r))
    rng = MT19937RandGen(seed)
    return [[lo + (hi - lo) * rng.randdouble() for lo, hi in box] for _ in range(n)]


def probe_spread(
    tr: combo_tree, rows: RegressionRows, probes: Sequence[Sequence[float]]
) -> tuple[float, bool]:
    """NOT A PORT, server round 13 (BASE-PS): ``(S, finite)`` of the program ``tr``. ``S`` is
    the largest ``|a + b f(x) - mean(y)|`` over the probe points ``x``, with ``(a, b)`` the
    linear scaling of ``tr`` fitted on the training ``rows`` (``linear_scaling``) and ``y``
    the training target. A prediction that is not finite (or a probe where ``tr`` cannot be
    evaluated) counts as ``+inf``. ``finite`` tells whether every prediction is finite. The
    training predictions are not a charged pass: LM already paid for that pass."""
    try:
        p = rows.predictions(tr)
    except Exception:  # any evaluation error makes it non-finite
        return math.inf, False
    a, b = linear_scaling(rows.targets, p)
    mean = math.fsum(rows.targets) / len(rows.targets)
    iv = interpreter_visitor(tr)
    s, finite = 0.0, True
    for x in probes:
        try:
            v = a + b * float(get_contin(iv(list(x))))
        except Exception:  # any evaluation error (a pole, overflow, domain)
            v = math.nan
        if not math.isfinite(v):
            s, finite = math.inf, False
            continue
        s = max(s, abs(v - mean))
    return s, finite


def pole_check(
    old: combo_tree,
    new: combo_tree,
    rows: RegressionRows,
    probes: Sequence[Sequence[float]],
    charge: Callable[[], None],
    factor: float = POLE_FACTOR,
) -> dict:
    """NOT A PORT, server round 13 (``--const-opt-pole-check box``, BASE-PS): does the tuned
    program ``new`` keep away from poles on the probe points? Each of the two probe passes
    (``old``, then ``new``) calls ``charge()`` once before it runs, and ``BudgetExhausted``
    from ``charge`` goes to the caller.

    With ``S`` from ``probe_spread`` and ``R`` = max - min of the training target, the tuned
    program is rejected if (a) a probe prediction of ``new`` is not finite and all of those
    of ``old`` are finite, or (b) ``S(new) > factor * R`` and ``S(new) > S(old)``. A
    prediction that is not finite counts as ``+inf`` in ``S``; so when both programs have
    one, (a) does not apply and (b) gives ``inf > inf``, false: the check accepts. Returns
    ``accepted``, ``pole_s_old`` and ``pole_s_new``."""
    charge()
    s_old, finite_old = probe_spread(old, rows, probes)
    charge()
    s_new, finite_new = probe_spread(new, rows, probes)
    r = max(rows.targets) - min(rows.targets)
    reject = (not finite_new and finite_old) or (s_new > factor * r and s_new > s_old)
    return {"accepted": not reject, "pole_s_old": s_old, "pole_s_new": s_new}


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
