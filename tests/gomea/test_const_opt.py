"""Constant tuning (``--const-opt``, server round 3): LM, coefficient mutation,
polish, and their evaluation counts."""

from __future__ import annotations

import io
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import get_contin
from pymoses.gomea.const_opt import (
    RegressionRows,
    constant_nodes,
    levenberg_marquardt,
    mse,
    mutation_sigma,
    polish_metapopulation,
    solve,
    validation_nmse,
)
from pymoses.gomea.gom import Individual
from pymoses.gomea.optimizer import gomea_optimization, gomea_parameters
from pymoses.gomea.oracle import BudgetExhausted, EvaluationLedger
from pymoses.moses.moses.types import composite_score
from pymoses.moses.scoring.bscores import linear_scaling

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"


def _sine_rows(n=40, a=2.5, w=1.3, scaled=False):
    xs = [-3 + 6 * i / (n - 1) for i in range(n)]
    return RegressionRows([[x] for x in xs], [a * math.sin(w * x) for x in xs], scaled)


class _Counter:
    def __init__(self, limit=None):
        self.n = 0
        self.limit = limit

    def __call__(self):
        if self.limit is not None and self.n >= self.limit:
            raise BudgetExhausted("global_budget")
        self.n += 1


def _values(tr):
    return [get_contin(n.data) for n in constant_nodes(tr)]


# ---- LM


def test_lm_fits_a_known_function():
    tr = str_to_combo_tree("*(2.0 sin(*(1.2 $1)))")
    charge = _Counter()
    res = levenberg_marquardt(tr, _sine_rows(), charge)
    assert res.status == "improved"
    assert res.mse < 1e-12 < res.mse0
    a, w = res.values
    assert a == pytest.approx(2.5, abs=1e-6) and w == pytest.approx(1.3, abs=1e-6)
    assert _values(res.tree) == res.values
    assert _values(tr) == [2.0, 1.2]  # the input tree is not changed
    # every pass over the rows was charged once
    assert res.evals == charge.n


def test_lm_with_linear_scaling_tunes_only_the_inner_constant():
    # with linear scaling the outer factor is absorbed by b: its column is inert
    tr = str_to_combo_tree("*(2.0 sin(*(1.2 $1)))")
    res = levenberg_marquardt(tr, _sine_rows(scaled=True), _Counter())
    assert res.status == "improved" and res.n_inert == 1
    assert res.values[0] == 2.0 and res.values[1] == pytest.approx(1.3, abs=1e-6)


def test_lm_evaluation_count_per_iteration():
    # one pass to start, then per iteration k Jacobian passes plus the trial steps
    tr = str_to_combo_tree("*(2.0 sin(*(1.2 $1)))")
    res = levenberg_marquardt(tr, _sine_rows(), _Counter())
    assert res.evals >= 1 + res.iterations * (2 + 1)
    assert res.iterations <= 10


def test_lm_all_inert_constants_are_singular_and_keep_the_program():
    # linear scaling absorbs the additive constant: no constant changes the residuals
    tr = str_to_combo_tree("+(3.0 sin($1))")
    res = levenberg_marquardt(tr, _sine_rows(scaled=True), _Counter())
    assert res.status == "singular" and res.tree is None


def test_solve_reports_a_singular_matrix():
    assert solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]) is None
    assert solve([[0.0]], [1.0]) is None
    assert solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 2.0]) == [1.0, 0.5]


def test_lm_does_not_return_a_program_unless_the_error_is_lower():
    tr = str_to_combo_tree("*(2.5 sin(*(1.3 $1)))")  # already exact
    res = levenberg_marquardt(tr, _sine_rows(), _Counter())
    assert res.status == "not_improved" and res.tree is None
    assert res.mse == res.mse0 == 0.0


def test_lm_failed_on_a_non_finite_program():
    tr = str_to_combo_tree("*(2.0 exp(*(1000.0 $1)))")
    res = levenberg_marquardt(tr, _sine_rows(), _Counter())
    assert res.status == "failed" and res.tree is None


def test_lm_no_constants():
    res = levenberg_marquardt(str_to_combo_tree("sin($1)"), _sine_rows(), _Counter())
    assert res.status == "no_constants" and res.evals == 1


def test_lm_stops_at_the_budget_and_keeps_the_best_point():
    tr = str_to_combo_tree("*(2.0 sin(*(1.2 $1)))")
    charge = _Counter(limit=6)
    res = levenberg_marquardt(tr, _sine_rows(), charge)
    assert res.budget and res.evals == 6 == charge.n
    if res.status == "improved":
        assert res.mse < res.mse0
    budget0 = levenberg_marquardt(tr, _sine_rows(), _Counter(limit=0))
    assert budget0.status == "budget" and budget0.evals == 0


def test_oracle_charge_tuning_counts_against_the_run_budget():
    class _O:
        pass

    from pymoses.gomea.oracle import EvaluationOracle

    o = _O()
    o.ledger = EvaluationLedger()
    o.max_evals, o.max_time, o.start = 3, 1e9, 0.0
    o.clock = lambda: 0.0
    for _ in range(3):
        EvaluationOracle.charge_tuning(o)
    assert o.ledger.requests == 3 and o.ledger.tuning_requests == 3
    with pytest.raises(BudgetExhausted):
        EvaluationOracle.charge_tuning(o)
    assert o.ledger.as_dict()["tuning_requests"] == 3
    assert "tuning_requests" not in EvaluationLedger().as_dict()


# ---- mutate


def test_mutation_sigma():
    assert mutation_sigma(5.0) == 0.5
    assert mutation_sigma(-5.0) == 0.5
    assert mutation_sigma(0.2) == 0.1


class _FakeSchema:
    def __init__(self):
        self.values = {0: 1.0, 1: 2.0, 2: 3.0}
        self.free: list[float] = []

    def active_contin(self, g):
        return [0, 1, 2]

    def contin_value(self, p, code):
        return self.free[code - 100] if code >= 100 else self.values[p]

    def register_contin_value(self, p, v):
        self.free.append(v)
        return 99 + len(self.free)


class _FakeEv:
    def __init__(self, scores):
        self.scores = list(scores)

    def evaluate(self, g):
        return self.scores.pop(0)

    def worse(self, a, b):
        return a < b

    def phenotype_key(self, g):
        return None


class _FakeOracle:
    def __init__(self):
        self.ledger = EvaluationLedger()


def test_mutate_keeps_a_change_unless_the_score_is_worse():
    opt = gomea_optimization(gp=gomea_parameters(contin="copy", const_opt="mutate"))
    from pymoses.util.rng import MT19937RandGen

    opt._rng_mut = MT19937RandGen(7)
    ev = _FakeEv([5.0, 4.0, 5.0])  # better, worse, equal (vs 1.0, 5.0, 5.0)
    schema = _FakeSchema()
    mut = opt._mutator(schema, _FakeOracle(), ev)
    out = mut(Individual((0, 0, 0), 1.0, None))
    assert out.score == 5.0
    assert out.genotype == (100, 0, 102)  # gene 0 kept, gene 1 rejected, gene 2 kept (equal)
    assert opt._co["mutate_attempts"] == 3 and opt._co["mutate_accepted"] == 2
    # each new value is c + N(0, sigma), sigma = max(0.1 |c|, 0.1)
    assert all(
        abs(v - c) < 10 * mutation_sigma(c) for v, c in zip(schema.free, (1, 2, 3), strict=True)
    )


def test_mutate_off_gives_no_hook():
    opt = gomea_optimization(gp=gomea_parameters(contin="copy"))
    assert opt._mutator(_FakeSchema(), _FakeOracle(), _FakeEv([])) is None


def test_parameters_validation():
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="mutate")  # needs contin copy
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="polish")  # needs a validation table
    with pytest.raises(ValueError):
        gomea_parameters(const_opt="bfgs")
    assert "const_opt" not in gomea_parameters().as_dict()


# ---- polish


def _table_file(path, xs, ys):
    path.write_text("x1,y\n" + "".join(f"{x!r},{y!r}\n" for x, y in zip(xs, ys, strict=True)))
    return str(path)


class _FakeMetapop:
    def __init__(self, sct):
        self.trees = [sct]
        outer = self

        class _Set:
            def erase(self, s):
                outer.trees.remove(s)

            def insert(self, s):
                outer.trees.insert(0, s)

        self._scored_trees = _Set()
        self.best = None

    def get_trees(self):
        return list(self.trees)

    def update_best_candidates(self, cands):
        self.best = list(cands)


class _FakeCscorer:
    def __init__(self, b):
        self.b = b
        self.calls = 0

    def get_bscorer(self):
        return self.b

    def get_cscore(self, tr):
        self.calls += 1
        return composite_score(-sum(x * x for x in self.b(tr)) * -1.0, 3, 0.0, 0.0)


def _polish_case(tmp_path, val_ys):
    from pymoses.comboreduct.table.table_io import loadTable
    from pymoses.moses.moses.types import scored_combo_tree
    from pymoses.moses.scoring.bscores import contin_bscore

    xs = [-3 + 6 * i / 39 for i in range(40)]
    train = _table_file(tmp_path / "train.csv", xs, [2.5 * math.sin(1.3 * x) for x in xs])
    b = contin_bscore(loadTable(train, "y"))
    b.set_linear_scaling(True)
    vx = [-2.9 + 5.8 * i / 29 for i in range(30)]
    val = _table_file(tmp_path / "val.csv", vx, val_ys(vx))
    old = scored_combo_tree(
        str_to_combo_tree("*(2.0 sin(*(1.2 $1)))"), 1, composite_score(-1.0, 3, 0.0, 0.0)
    )
    mp = _FakeMetapop(old)
    cs = _FakeCscorer(b)
    rec = polish_metapopulation(mp, cs, 500, val)
    return mp, old, rec, cs


def test_polish_keeps_the_tuned_program_when_validation_is_not_worse(tmp_path):
    mp, old, rec, cs = _polish_case(tmp_path, lambda vx: [2.5 * math.sin(1.3 * x) for x in vx])
    assert rec["status"] == "improved" and rec["accepted"]
    assert rec["val_nmse_new"] <= rec["val_nmse_old"]
    assert mp.trees[0] is not old and len(mp.trees) == 1
    assert rec["evals"] <= 500 and cs.calls == 1


def test_polish_rejects_the_tuned_program_when_validation_is_worse(tmp_path):
    # the validation target is exactly the old program after its training fit:
    # its validation NMSE is 0, so any tuned program is worse there
    xs = [-3 + 6 * i / 39 for i in range(40)]
    a, b = linear_scaling(
        [2.5 * math.sin(1.3 * x) for x in xs], [2.0 * math.sin(1.2 * x) for x in xs]
    )
    mp, old, rec, cs = _polish_case(
        tmp_path, lambda vx: [a + b * 2.0 * math.sin(1.2 * x) for x in vx]
    )
    assert rec["status"] == "improved" and not rec["accepted"]
    assert rec["val_nmse_old"] == pytest.approx(0.0, abs=1e-20)
    assert mp.trees == [old] and cs.calls == 0


def test_validation_nmse_uses_the_training_fit():
    train = _sine_rows(scaled=True)
    tr = str_to_combo_tree("sin(*(1.3 $1))")
    assert validation_nmse(tr, train, _sine_rows(n=25)) == pytest.approx(0.0, abs=1e-20)
    assert mse([1.0, -1.0]) == 1.0


# ---- end to end (in process)


def _run(argv, tmp_path):
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


_ARGV = [
    "-i", str(DATA / "train.csv"), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "3", "-V1", "-l", "WARN", "-m", "700", "-a", "gomea",
    "--gomea-pop-size", "16", "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "1",
]  # fmt: skip


@pytest.mark.parametrize("mode", ["lm", "mutate"])
def test_tuning_evaluations_are_in_the_run_count(mode, tmp_path):
    out, events = _run(_ARGV + ["--const-opt", mode], tmp_path)
    ends = [e for e in events if e["type"] == "end"]
    n_evals = int(out.rsplit("n_evals:", 1)[1])
    assert n_evals <= 700
    assert n_evals == sum(e["ledger"]["requests"] for e in ends)
    tuned = sum(e["const_opt"]["tuning_evals"] for e in ends)
    assert tuned > 0
    if mode == "lm":
        assert tuned == sum(e["ledger"].get("tuning_requests", 0) for e in ends)
        for e in ends:
            lm = e["const_opt"]["lm"]
            if lm is not None and "written_back" in lm:
                assert lm["status"] == "improved"
            if lm is not None and lm["status"] != "improved":
                assert "written_back" not in lm
    else:
        assert sum(e["const_opt"]["mutate_attempts"] for e in ends) == tuned


def test_polish_end_to_end(tmp_path):
    val = DATA / "train.csv"
    out, events = _run(_ARGV + ["--const-opt", "polish", "--const-opt-val", str(val)], tmp_path)
    pol = [e for e in events if e["type"] == "polish"]
    assert len(pol) == 1
    n_evals = int(out.rsplit("n_evals:", 1)[1])
    ends = [e for e in events if e["type"] == "end"]
    assert n_evals == sum(e["ledger"]["requests"] for e in ends) + pol[0]["evals"] <= 700
