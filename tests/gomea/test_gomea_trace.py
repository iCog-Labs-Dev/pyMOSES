"""NOT A PORT, server round 16 (the RV16 review, finding 1 and nit 8): the option
``--gomea-trace 1`` of the GOMEA log.

* ``EvaluationOracle.trace``: one ``[requests, raw]`` pair for each charged evaluation whose
  finite raw score is strictly higher than every raw score before it; a memo hit adds nothing;
  off, ``trace`` is ``None``.
* End to end (the flags of the T3.2 spike test, ``-m 1500``; RV off, ``balance`` and
  ``late``): the "end" trace of each deme equals the trace computed from an independent record
  of every charged evaluation of the deme's oracle. The RV evaluations go through the same
  oracle, so the trace sees them. The runner's ``anytime_records`` on the real log gives the
  exact best-so-far value at each checkpoint.
* The digest of the initial genotypes (``init_digest``): the "init" event of the first deme of
  an RV run has the digest of the off run, and of the genotypes that ``build_population`` gave.
* Refusals: the CLI (``-a gomea`` only, not with ``--const-opt lm``, the values 0 and 1),
  ``gomea_parameters`` and the hybrid.

The byte identity of ``--gomea-trace 0`` against 2d407b2, and of ``--gomea-trace 1`` without its
added fields, is in ``test_rv_step_identity.py``.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.gomea import optimizer as gomea_optimizer
from pymoses.gomea.oracle import BudgetExhausted, EvaluationOracle
from pymoses.gomea.rv_core import RVCore
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.moses.types import composite_score, very_worst_score
from pymoses.moses.representation.instance_set import instance_set

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

TRACE = ["--gomea-trace", "1"]


# ---- the oracle


class _Scores:
    """An iscorer that gives the raw scores in call order (charged calls only)."""

    def __init__(self, raws):
        self.raws = list(raws)

    def __call__(self, inst):
        return composite_score(self.raws.pop(0), 0, 0.0, 0.0)


def _genotypes(schema, n):
    """The first ``n`` genotypes of the schema (all distinct)."""
    out = list(itertools.islice(itertools.product(*(range(k) for k in schema.domains)), n))
    assert len(out) == n, "not enough genotypes"
    return out


def test_the_oracle_trace(make_boolean_rep):
    rep = make_boolean_rep("and(or($1 $2) or(!$1 $3))", 3)
    schema = FrozenSchema(rep)
    a, b, c, d, e, f, g = _genotypes(schema, 7)
    raws = [-5.0, -7.0, -3.0, math.nan, -3.0, very_worst_score, -1.0]
    oracle = EvaluationOracle(schema, _Scores(raws), instance_set(schema.fields), 1000, 3600.0,
                              genotype_memo=True, trace=True)
    for x in (a, b, a, c, d, e, f, b, g):  # the second a and b are memo hits (not charged)
        oracle.evaluate(x)
    assert oracle.ledger.requests == 7 and oracle.ledger.memo_hits == 2
    # a new best at 1 and 3 (the memo hit is not counted); NaN, an equal score and the worst
    # score are not; -1 at 7
    assert oracle.trace == [[1, -5.0], [3, -3.0], [7, -1.0]]
    # the worst score (-FLT_MAX, an invalid program) is finite: it is a best when it is the
    # first score; NaN and infinity never are
    worst = EvaluationOracle(schema, _Scores([math.nan, -math.inf, very_worst_score, -9.0]),
                             instance_set(schema.fields), 10, 3600.0, trace=True)
    for x in (a, b, c, d):
        worst.evaluate(x)
    assert worst.trace == [[3, very_worst_score], [4, -9.0]]
    # a stop by the budget charges nothing and adds nothing
    small = EvaluationOracle(schema, _Scores([-2.0]), instance_set(schema.fields), 1, 3600.0,
                             trace=True)
    small.evaluate(a)
    with pytest.raises(BudgetExhausted):
        small.evaluate(b)
    assert small.trace == [[1, -2.0]]
    # off: no trace
    off = EvaluationOracle(schema, _Scores([-2.0]), instance_set(schema.fields), 10, 3600.0)
    off.evaluate(a)
    assert off.trace is None


def test_init_digest():
    want = hashlib.sha256(b"[[0,1,2],[3,4,5]]").hexdigest()
    assert gomea_optimizer.init_digest([(0, 1, 2), (3, 4, 5)]) == want
    assert gomea_optimizer.init_digest([(3, 4, 5), (0, 1, 2)]) != want  # the order counts
    assert gomea_optimizer.init_digest([]) == hashlib.sha256(b"[]").hexdigest()


# ---- end to end


def _argv(seed: str = "1") -> list[str]:
    """The flags of the T3.2 spike test (test_rv_core.py), with the trace."""
    return [
        "-i", str(DATA / "train.csv"), "-u", "y", "-r", seed, "-n", "sin", "-n", "exp",
        "-n", "log", "-v", "2400", "-c", "3", "-V1", "-l", "WARN", "-m", "1500", "-a", "gomea",
        "--gomea-pop-size", "24", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "terms",
        "--gomea-deme-store", "accepted", "--gomea-accept-complexity-ratio", "0.125",
    ] + TRACE  # fmt: skip


RV = {"off": [], "balance": ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "balance"],
      "late": ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "late"]}


def _run(argv, tmp_path):
    from pymoses.moses.main.moses_exec import moses_exec

    tmp_path.mkdir(parents=True, exist_ok=True)
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
    return buf.getvalue(), log


def _run_recorded(argv, tmp_path, monkeypatch):
    """``_run``, plus an independent record of each oracle (one per deme, in order): the
    (requests, raw score, from RV) of each charged evaluation, the memo hits, and the initial
    genotypes of each ``build_population`` call."""
    oracles: list[list] = []
    memo = [0]
    in_rv = [False]
    inits = []
    real_init, real_eval = EvaluationOracle.__init__, EvaluationOracle.evaluate
    real_rv, real_build = RVCore._evaluate, gomea_optimizer.build_population

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        self._record = []
        oracles.append(self._record)

    def evaluate(self, g):
        r0 = self.ledger.requests
        score = real_eval(self, g)
        if self.ledger.requests > r0:
            self._record.append((self.ledger.requests, score.get_score(), in_rv[0]))
        else:
            memo[0] += 1
        return score

    def rv_evaluate(self, g):
        in_rv[0] = True
        try:
            return real_rv(self, g)
        finally:
            in_rv[0] = False

    def build(*a, **kw):
        res = real_build(*a, **kw)
        inits.append(res)
        return res

    monkeypatch.setattr(EvaluationOracle, "__init__", init)
    monkeypatch.setattr(EvaluationOracle, "evaluate", evaluate)
    monkeypatch.setattr(RVCore, "_evaluate", rv_evaluate)
    monkeypatch.setattr(gomea_optimizer, "build_population", build)
    out, log = _run(argv, tmp_path)
    monkeypatch.undo()
    events = [json.loads(x) for x in log.read_text().splitlines()]
    return out, log, events, oracles, memo[0], inits


def _expected_trace(record) -> list[list]:
    out, best = [], None
    for req, raw, _ in record:
        if math.isfinite(raw) and (best is None or raw > best):
            best = raw
            out.append([req, raw])
    return out


@pytest.mark.parametrize("rv", list(RV))
def test_the_trace_sees_every_charged_evaluation(rv, tmp_path, monkeypatch):
    out, log, events, oracles, memo, _ = _run_recorded(_argv() + RV[rv], tmp_path, monkeypatch)
    ends = [e for e in events if e["type"] == "end"]
    assert len(ends) == len(oracles) >= 2
    for e, rec in zip(ends, oracles, strict=True):
        assert e["ledger"]["requests"] == len(rec)
        assert e["trace"] == _expected_trace(rec)
        assert e["trace"][0][0] == 1  # the first evaluation of the deme (a finite score)
    assert memo > 0  # memo hits happened and are not in the trace
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert all(s["params"]["trace"] is True for s in starts)
    if rv != "off":
        # the RV evaluations went through the same oracle (the record above has them)
        from_rv = [x for rec in oracles for x in rec if x[2]]
        assert from_rv and sum(e["rv"]["rv_evals"] for e in ends) == len(from_rv)
    # the runner: the anytime values are the exact best so far of all charged evaluations
    n_evals = int(out.rsplit("n_evals:", 1)[1])
    recs, check = runner.anytime_records(log, 1500, 1, 1.0, n_evals=n_evals)
    assert check["anytime_source"] == "trace" and check["log_evals_match_n_evals"] is True
    assert check["trace_well_formed"] is True and check["trace_dominates_log"] is True
    assert check["trace_first_evals"] == 1
    every, offset = [], 0
    for e, rec in zip(ends, oracles, strict=True):
        every += [(offset + req, raw) for req, raw, _ in rec if math.isfinite(raw)]
        offset += e["ledger"]["requests"]
    for r in recs:
        best = max(raw for at, raw in every if at <= r["evals"])
        assert r["best_train_nmse"] == pytest.approx(-100.0 * best, rel=1e-15)
    assert check["trace_best_raw"] == max(raw for _, raw in every)


def test_the_first_deme_of_an_rv_run_has_the_digest_of_the_off_run(tmp_path, monkeypatch):
    """Nit 8: the pilot can compare the initial genotypes of an RV run with its BASE-TLS run."""
    digests = {}
    for rv in ("off", "balance"):
        _, _, events, _, _, inits = _run_recorded(_argv() + RV[rv], tmp_path / rv, monkeypatch)
        got = [e["genotypes_sha256"] for e in events if e["type"] == "init"]
        assert got == [gomea_optimizer.init_digest(i.genotypes) for i in inits]
        digests[rv] = got
    assert digests["balance"][0] == digests["off"][0]


# ---- refusals


def _cli_error(argv, tmp_path):
    with pytest.raises(ValueError) as exc:
        _run(argv, tmp_path)
    return str(exc.value)


@pytest.mark.parametrize("algo", ["hc", "hc-capped", "hc-gomea"])
def test_the_cli_refuses_the_trace_without_gomea(algo, tmp_path):
    argv = _argv()
    argv[argv.index("-a") + 1] = algo
    assert _cli_error(argv, tmp_path) == f"--gomea-trace 1 needs -a gomea, got -a {algo}"


def test_the_cli_refuses_the_trace_with_lm_and_a_wrong_value(tmp_path):
    argv = [x for x in _argv() if x != "--gomea-accept-complexity-ratio" and x != "0.125"]
    assert "does not support --const-opt lm" in _cli_error(argv + ["--const-opt", "lm"],
                                                           tmp_path)
    bad = _argv()
    bad[-1] = "2"
    assert _cli_error(bad, tmp_path) == "--gomea-trace must be 0 or 1, got 2"


def test_gomea_parameters_and_the_hybrid():
    from pymoses.gomea.hybrid import hc_gomea_optimization

    gp = gomea_optimizer.gomea_parameters
    assert "trace" not in gp().as_dict()
    assert gp(contin="copy", trace=True).as_dict()["trace"] is True
    with pytest.raises(ValueError, match="trace must be True or False"):
        gp(trace=1)
    with pytest.raises(ValueError, match="trace does not support const_opt 'lm'"):
        gp(trace=True, const_opt="lm")
    gp(contin="copy", trace=True, const_opt="mutate")  # mutation goes through the oracle
    with pytest.raises(ValueError, match="trace needs -a gomea, not the hc-gomea hybrid"):
        hc_gomea_optimization(None, None, gp(trace=True))
