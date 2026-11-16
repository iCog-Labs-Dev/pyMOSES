"""Server round 12, T2.3: LM on new deme bests during the GOMEA search (``-a gomea
--const-opt lm --const-opt-during elite``), as ``docs/gomea/SERVER-ROUND12-PREREGISTRATION.md``
("T2.3") registers it: when a GOM offspring becomes a new deme best (a strict improvement
under the MOSES score), LM with the validation guard tunes the constants of its unreduced
program, the tuned constants go back into its genotype as free contin codes, and the tuned
offspring replaces it if it is not worse under the deme's comparator. These passes obey the
deme cap and the stall rule. The identity of the arms without the option is
``test_const_opt_topk_during_identity.py``."""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.gomea import engine as engine_mod
from pymoses.gomea.optimizer import gomea_optimization, gomea_parameters
from pymoses.gomea.oracle import BudgetExhausted, EvaluationOracle
from pymoses.moses.scoring.behave_cscore import _cache_key

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
TRAIN = DATA / "train.csv"

# The round 12 BASE flags (the round 11 BASE arm, configs/gomea/phase1be/), without the
# validation table (each test adds it).
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
]  # fmt: skip
_ELITE = ["--const-opt-during", "elite"]
# BASE without the deme-local complexity ratio: GOM then compares with the MOSES score.
_I = _BASE.index("--gomea-accept-complexity-ratio")
_NO_RATIO = _BASE[:_I] + _BASE[_I + 2 :]


def _reg(m: str, train: Path = TRAIN) -> list[str]:
    return ["-i", str(train), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


def _lme(m: str, ratio: bool = True) -> list[str]:
    """BASE-LME (or BASE-LME without the complexity ratio) on the identity-test table; the
    validation table is the training table."""
    return _reg(m) + (_BASE if ratio else _NO_RATIO) + ["--const-opt-val", str(TRAIN)] + _ELITE


def _run(argv, tmp_path):
    """(stdout, events of --gomea-log) of one in-process run."""
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    log.unlink(missing_ok=True)
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


def _run_error(argv, tmp_path, capsys):
    """The error text of a run that the command line must refuse."""
    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        try:
            rc = moses_exec(["moses", *argv])
        except (ValueError, SystemExit) as exc:
            text = str(exc)
        else:
            assert rc != 0
            text = ""
    finally:
        os.chdir(old)
    cap = capsys.readouterr()
    return text + cap.out + cap.err


def _ends(events):
    return [e for e in events if e["type"] == "end"]


def _n_evals(out):
    return int(out.rsplit("n_evals:", 1)[1])


# ---- the flag


_VAL = ["--const-opt-val", str(TRAIN)]


@pytest.mark.parametrize("extra, message", [
    (_BASE + _VAL + ["--const-opt-during", "all"], "--const-opt-during all is not recognized"),
    (["-a", "gomea", "--gomea-contin", "copy", "--linear-scaling", "1"] + _ELITE,
     "--const-opt-during elite needs --const-opt lm"),
    (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "mutate"] + _ELITE,
     "--const-opt-during elite needs --const-opt lm"),
    (["-a", "hc", "--linear-scaling", "1", "--const-opt", "lm"] + _ELITE,
     "--const-opt-during elite needs -a gomea"),
    (["-a", "hc-gomea", "--gomea-contin", "copy", "--const-opt", "lm"] + _ELITE,
     "--const-opt-during elite needs -a gomea"),
    (["-a", "gomea", "--linear-scaling", "1", "--const-opt", "lm"] + _ELITE,
     "--const-opt-during elite needs --gomea-contin copy"),
    (_BASE + _VAL + _ELITE + ["--const-opt-top-k", "5"], "exclude each other"),
], ids=["bad-value", "no-const-opt", "mutate", "hc", "hc-gomea", "no-contin-copy", "top-k"])
def test_the_flag_is_refused(extra, message, tmp_path, capsys):
    assert message in _run_error(_reg("300") + extra, tmp_path, capsys)


def test_the_parameters_refuse_a_bad_value():
    with pytest.raises(ValueError, match="const_opt_during must be one of"):
        gomea_parameters(const_opt="lm", contin="copy", const_opt_during="all")
    with pytest.raises(ValueError, match="needs const_opt 'lm'"):
        gomea_parameters(contin="copy", const_opt_during="elite")
    with pytest.raises(ValueError, match="needs contin 'copy'"):
        gomea_parameters(const_opt="lm", const_opt_during="elite")
    with pytest.raises(ValueError, match="exclude each other"):
        gomea_parameters(const_opt="lm", contin="copy", const_opt_during="elite",
                         const_opt_top_k=5)


def test_the_hybrid_and_hc_lm_refuse_the_option():
    from pymoses.gomea.hc_lm import hc_lm_optimization
    from pymoses.gomea.hybrid import hc_gomea_optimization

    gp = gomea_parameters(const_opt="lm", contin="copy", const_opt_during="elite")
    with pytest.raises(ValueError, match="const_opt_during needs -a gomea"):
        hc_gomea_optimization(gp=gp)
    with pytest.raises(ValueError, match="const_opt_during needs -a gomea"):
        hc_lm_optimization(gp=gp)


def test_off_builds_no_hook_and_leaves_the_telemetry_alone():
    off = gomea_optimization(gp=gomea_parameters(const_opt="lm", contin="copy"))
    assert off._elite_tuner(None, None, None) is None
    assert "const_opt_during" not in off.gomea_params.as_dict()
    on = gomea_parameters(const_opt="lm", contin="copy", const_opt_during="elite")
    assert on.as_dict()["const_opt_during"] == "elite"


# ---- the charge function


def test_the_search_charge_obeys_every_rule_of_the_oracle():
    """``charge_search_tuning`` stops at the deme cap, the stall rule, the run's budget and
    the time limit; ``charge_tuning`` (LM at the end of a deme) ignores the cap and the stall
    rule."""
    t = [0.0]
    o = EvaluationOracle(None, None, [], 5, 10.0, inner_cap=3, clock=lambda: t[0])
    for _ in range(3):
        o.charge_search_tuning()
    with pytest.raises(BudgetExhausted) as exc:
        o.charge_search_tuning()
    assert exc.value.reason == "inner_budget"
    o.charge_tuning()
    o.charge_tuning()
    assert (o.ledger.requests, o.ledger.tuning_requests) == (5, 5)
    with pytest.raises(BudgetExhausted) as exc:
        o.charge_search_tuning()
    assert exc.value.reason == "global_budget"

    s = EvaluationOracle(None, None, [], 100, 10.0, stall_evals=2, clock=lambda: t[0])
    s.charge_search_tuning()
    s.charge_search_tuning()
    with pytest.raises(BudgetExhausted) as exc:
        s.charge_search_tuning()
    assert exc.value.reason == "stall_evals"
    s.charge_tuning()  # the stall rule does not apply after the search

    c = EvaluationOracle(None, None, [], 100, 10.0, clock=lambda: t[0])
    c.charge_search_tuning()
    t[0] = 11.0
    with pytest.raises(BudgetExhausted) as exc:
        c.charge_search_tuning()
    assert exc.value.reason == "time_limit"


# ---- the map from constants to genes


@pytest.fixture
def deme_spy(monkeypatch):
    """Record, at the end of each deme (before LM), the schema, the deme and the best
    genotype; the wrapped method is called unchanged."""
    demes: list[dict] = []

    def wrap(orig):
        def tune(self, deme, oracle, schema):
            demes.append({"schema": schema, "best": oracle.best_genotype(),
                          "entries": [(si.first, getattr(si, "tree", None)) for si in deme]})
            return orig(self, deme, oracle, schema)

        return tune

    # _tune_best without the option, _tune_deme_best with it
    for name in ("_tune_best", "_tune_deme_best"):
        monkeypatch.setattr(gomea_optimization, name, wrap(getattr(gomea_optimization, name)))
    return demes


def test_each_constant_of_the_unreduced_program_is_one_contin_gene(deme_spy, tmp_path):
    """For every program of the first deme: the tree of ``candidate_genes`` is the unreduced
    program, each of its constants is the value of one contin gene, no gene gives two
    constants, and the genes are the active contin genes. Also with a free value."""
    from pymoses.gomea.const_opt import constant_nodes

    _run(_reg("1500") + _BASE + _VAL, tmp_path)
    d = deme_spy[0]
    schema = d["schema"]
    genotypes = {schema.to_genotype(inst) for inst, tree in d["entries"] if tree is None}
    assert len(genotypes) > 50
    for g in genotypes:
        tree, genes = schema.candidate_genes(g)
        assert _cache_key(tree) == _cache_key(schema.candidate(g, False))
        assert None not in genes and len(set(genes)) == len(genes)
        assert sorted(genes) == schema.active_contin(g)
        values = [float(n.data) for n in constant_nodes(tree)]
        assert values == [schema.contin_value(p, g[p]) for p in genes]
    g = next(iter(genotypes))
    _, genes = schema.candidate_genes(g)
    p = genes[0]
    g2 = g[:p] + (schema.register_contin_value(p, 1.2345678),) + g[p + 1 :]
    tree2, genes2 = schema.candidate_genes(g2)
    assert genes2 == genes
    assert float(constant_nodes(tree2)[0].data) == 1.2345678
    assert _cache_key(tree2) == _cache_key(schema.candidate(g2, False))


# ---- LM during the search


@pytest.fixture
def hook_spy(monkeypatch):
    """Record each call of the ``tune_elite`` hook: the offspring, whether it was a new
    deme best (the MOSES best strictly improved since the previous call and the offspring is
    that best; the scores are compared here, not the request counts the optimizer uses),
    whether LM ran, and what the hook returned. Also check, after each generation, that a
    tuned offspring is in the new population."""
    calls: list[dict] = []
    orig = gomea_optimization._elite_tuner

    def _elite_tuner(self, schema, oracle, ev):
        hook = orig(self, schema, oracle, ev)
        if hook is None:
            return None
        state = {"best": oracle.best_so_far()}

        def spy(ind):
            cur = oracle.best_so_far()
            new_best = state["best"] < cur and oracle.best_genotype() == ind.genotype
            n0 = self._co["lm_elite_calls"]
            rec = {"in": ind, "new_best": new_best, "schema": schema}
            calls.append(rec)
            try:
                out = hook(ind)
            finally:
                rec["lm"] = self._co["lm_elite_calls"] > n0
                state["best"] = oracle.best_so_far()
            rec["out"] = out
            rec["record"] = self._co["lm_elite"][-1] if rec["lm"] else None
            return out

        return spy

    perform = engine_mod.perform_generation

    def perform_generation(eng, P, model, nis, generation):
        first = len(calls)
        new_pop, record, stop = perform(eng, P, model, nis, generation)
        pop = {ind.genotype for ind in new_pop}
        for c in calls[first:]:
            if "out" in c and c["out"] is not c["in"]:
                assert c["out"].genotype in pop
        return new_pop, record, stop

    monkeypatch.setattr(gomea_optimization, "_elite_tuner", _elite_tuner)
    monkeypatch.setattr(engine_mod, "perform_generation", perform_generation)
    return calls


@pytest.mark.parametrize("ratio", [True, False], ids=["ratio", "no-ratio"])
def test_lm_runs_only_on_new_deme_bests(ratio, hook_spy, tmp_path):
    out, events = _run(_lme("3000", ratio), tmp_path)
    assert _n_evals(out) <= 3000
    ran = [c for c in hook_spy if c["lm"]]
    assert ran and len(ran) < len(hook_spy)
    for c in hook_spy:
        assert c["lm"] == c["new_best"]
    ends = _ends(events)
    assert len(ran) == sum(e["const_opt"]["lm_elite_calls"] for e in ends)


def test_the_population_gets_the_tuned_constants(hook_spy, deme_spy, tmp_path):
    """A kept LM result is the offspring with free contin codes at the tuned genes (the
    other genes unchanged), its unreduced program holds the tuned constants, its score is not
    worse under the deme's comparator (here the MOSES score: no complexity ratio), it is in
    the next population (checked by the spy), and its deme entry, which carries its program,
    is kept. Only the constants that are not 0 are LM parameters: each constant with value 0
    (an inactive coefficient that knob building inserts) keeps the value 0 and its code."""
    from pymoses.gomea.const_opt import constant_nodes

    _, events = _run(_lme("3000", ratio=False), tmp_path)
    # a call that the budget stopped returned nothing (GomInterrupted): it has no "out"
    kept = [c for c in hook_spy if c["lm"] and c.get("out", c["in"]) is not c["in"]]
    assert kept, "no kept LM result"
    for c in kept:
        schema, old, new = c["schema"], c["in"], c["out"]
        assert c["record"]["outcome"] == "kept" and c["record"]["written_back"]
        _, genes = schema.candidate_genes(old.genotype)
        tree, genes_new = schema.candidate_genes(new.genotype)
        assert genes_new == genes
        free = schema.free_values(new.genotype)
        changed = [p for p in range(len(old.genotype)) if old.genotype[p] != new.genotype[p]]
        # each tuned gene now holds a free code (an older free code can be inherited)
        assert changed and set(changed) <= set(genes) and set(changed) <= set(free)
        values = [float(n.data) for n in constant_nodes(tree)]
        assert [values[genes.index(p)] for p in changed] == [free[p] for p in changed]
        assert not new.score < old.score
        old_tree, _ = schema.candidate_genes(old.genotype)
        old_values = [float(n.data) for n in constant_nodes(old_tree)]
        zero = [p for p, v in zip(genes, old_values, strict=True) if v == 0.0]
        active = [p for p, v in zip(genes, old_values, strict=True) if v != 0.0]
        for p in zero:
            assert new.genotype[p] == old.genotype[p]
            assert values[genes.index(p)] == 0.0
        assert set(changed) <= set(active)
        assert c["record"]["n_constants"] == len(active)
        assert c["record"]["n_constants_unreduced"] == len(genes)
    assert any(c["record"]["n_constants_unreduced"] > c["record"]["n_constants"] for c in kept), (
        "no kept LM result with an inactive zero coefficient")
    assert sum(e["const_opt"]["lm_elite_kept"] for e in _ends(events)) == len(kept)
    trees = sum(1 for d in deme_spy for _, tree in d["entries"] if tree is not None)
    assert trees >= 1


def test_lm_with_a_mask_tunes_only_the_flagged_constants():
    """``levenberg_marquardt(..., tune=...)``: a constant that is not flagged keeps its value
    (here the 0 of an inactive term); the flagged one is tuned. Without the mask both are
    tuned."""
    from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
    from pymoses.gomea.const_opt import RegressionRows, constant_nodes, levenberg_marquardt

    x = [[0.1 * i, (i * 7) % 5 + 0.5] for i in range(30)]
    rows = RegressionRows(x, [2.5 * a + 3.0 * b for a, b in x], False)
    tree = str_to_combo_tree("+(*(0 $2) *(1 $1))")
    masked = levenberg_marquardt(tree, rows, lambda: None, tune=[False, True])
    assert masked.status == "improved" and masked.n_constants == 1
    assert [float(n.data) for n in constant_nodes(masked.tree)][0] == 0.0
    assert len(masked.values) == 1 and masked.values[0] != 1.0
    full = levenberg_marquardt(tree, rows, lambda: None)
    assert full.n_constants == 2
    assert [float(n.data) for n in constant_nodes(full.tree)][0] != 0.0


class _EndOracle:
    """The part of ``EvaluationOracle`` that LM at the end of a deme uses."""

    def __init__(self, iscorer, best):
        from types import SimpleNamespace

        self.iscorer = iscorer
        self._best = best
        self.ledger = SimpleNamespace(requests=0, tuning_requests=0)

    def best_genotype(self):
        return self._best

    def charge_tuning(self):
        self.ledger.requests += 1
        self.ledger.tuning_requests += 1


def test_the_deme_end_lm_of_a_free_code_best_replaces_only_its_own_entries(tmp_path,
                                                                           monkeypatch):
    """With the option on, the deme's best program can have free contin codes. Its deme
    entries carry their program, with the placeholder instance (code 0 at the free genes).
    LM at the end of the deme (``_tune_deme_best``) replaces only the entry of that program:
    the plain entry of the placeholder instance (another program) and the entry of another
    free genotype stay. ``_tune_best`` (every other arm) removes the plain entry and keeps
    the stale one, which is why the option has its own path."""
    from pymoses.gomea.const_opt import constant_nodes, scored_tree_instance
    from pymoses.gomea.optimizer import _new_const_opt_counts
    from pymoses.moses.representation.instance_set import instance_set
    from pymoses.moses.representation.scored_instance import scored_instance

    got: dict = {}
    orig = gomea_optimization._tune_best

    def spy(self, deme, oracle, schema):
        got.setdefault("iscorer", oracle.iscorer)
        got.setdefault("schema", schema)
        got.setdefault("best", oracle.best_genotype())
        return orig(self, deme, oracle, schema)

    monkeypatch.setattr(gomea_optimization, "_tune_best", spy)
    _run(_reg("1500") + _BASE + _VAL, tmp_path)
    monkeypatch.undo()
    schema, iscorer, g = got["schema"], got["iscorer"], got["best"]
    cscorer = iscorer._cscorer

    def entry(geno, tree_entry):
        tree = schema.candidate(geno, True)
        score = cscorer.get_cscore(tree)
        inst = schema.to_instance(geno)
        return scored_tree_instance(inst, score, tree) if tree_entry else scored_instance(
            inst, score)

    tree, genes = schema.candidate_genes(g)
    written = 0
    for p, node in zip(genes, constant_nodes(tree), strict=True):
        v = float(node.data)
        if v == 0.0:
            continue
        g_free = g[:p] + (schema.register_contin_value(p, v + 0.75),) + g[p + 1 :]
        g_other = g[:p] + (schema.register_contin_value(p, v - 0.5),) + g[p + 1 :]
        g_zero = g[:p] + (0,) + g[p + 1 :]  # the plain genotype of the placeholder
        inst = schema.to_instance(g_free)
        assert inst == schema.to_instance(g_other) == schema.to_instance(g_zero)
        key = _cache_key(schema.candidate(g_free, True))

        def deme_of(g_zero=g_zero, g_free=g_free, g_other=g_other):
            d = instance_set(schema.fields)
            d.append(entry(g_zero, False))
            d.append(entry(g_free, True))
            d.append(entry(g_other, True))
            return d

        def tuned(path, g_free=g_free, deme_of=deme_of):
            opt = gomea_optimization(gp=gomea_parameters(
                const_opt="lm", contin="copy", const_opt_guard="val",
                const_opt_val=str(TRAIN), const_opt_during="elite"))
            opt._co = _new_const_opt_counts()
            d = deme_of()
            getattr(opt, path)(d, _EndOracle(iscorer, g_free), schema)
            return d, opt._co["lm"]

        new, rec = tuned("_tune_deme_best")
        plain = [si for si in new if si.first == inst and getattr(si, "tree", None) is None]
        stale = [si for si in new if getattr(si, "tree", None) is not None
                 and _cache_key(si.tree) == key]
        other = [si for si in new if getattr(si, "tree", None) is not None
                 and _cache_key(si.tree) == _cache_key(schema.candidate(g_other, True))]
        assert len(plain) == 1 and len(other) == 1
        if not rec.get("written_back"):
            assert len(new) == 3 and len(stale) == 1
            continue
        written += 1
        assert not stale and len(new) == 3  # the stale entry out, the tuned one in
        old, rec_old = tuned("_tune_best")
        assert rec_old["written_back"]
        assert not [si for si in old if getattr(si, "tree", None) is None]  # plain entry lost
        assert [si for si in old if getattr(si, "tree", None) is not None
                and _cache_key(si.tree) == key]  # stale entry kept
    assert written, "no constant of the best program could be tuned"


def test_the_deme_cap_is_respected(tmp_path):
    """With a deme cap of 300, the search with its LM passes stops at 300 evaluations; only
    LM at the end of the deme (the run's budget, as in BASE) can pass it."""
    argv = _lme("3000", ratio=False) + ["--gomea-max-evals-per-deme", "300"]
    out, events = _run(argv, tmp_path)
    ends = _ends(events)
    assert _n_evals(out) <= 3000
    assert any(e["termination_reason"] == "inner_budget" for e in ends)
    assert sum(e["const_opt"]["lm_elite_calls"] for e in ends) > 0
    for e in ends:
        co = e["const_opt"]
        end_lm = co["lm"]["evals"] if co["lm"] else 0
        assert e["ledger"]["requests"] - end_lm <= 300


@pytest.mark.parametrize("m, ratio", [("300", True), ("600", True), ("1500", True),
                                      ("3000", True), ("3000", False)])
def test_the_evaluations_never_pass_m(m, ratio, tmp_path):
    out, events = _run(_lme(m, ratio), tmp_path)
    ends = _ends(events)
    n = _n_evals(out)
    assert n <= int(m)
    assert n == sum(e["ledger"]["requests"] for e in ends)
    for e in ends:
        co = e["const_opt"]
        assert co["lm_elite_calls"] == len(co["lm_elite"])
        assert co["lm_elite_evals"] == sum(r["evals"] for r in co["lm_elite"])
        assert co["tuning_evals"] >= co["lm_elite_evals"]
        # the deme-end LM call is the one ``lm`` records; the elite calls are the others
        assert co["lm_calls"] - co["lm_elite_calls"] in (0, 1)


def test_the_option_on_the_round_12_test_table_runs(tmp_path):
    """BASE-LME on the identity-test table (the arm's flags, two demes at -m 1500)."""
    out, events = _run(_lme("1500"), tmp_path)
    assert _n_evals(out) <= 1500
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert all(e["params"]["const_opt_during"] == "elite" for e in starts)
    for e in _ends(events):
        assert {"lm_elite", "lm_elite_calls", "lm_elite_not_knob"} <= set(e["const_opt"])
        assert e["const_opt"]["lm_elite_not_knob"] == 0


# ---- the runner


def _runner():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "gomea" / "moses_sr_runner.py"
    spec = importlib.util.spec_from_file_location("moses_sr_runner_r12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_runner_sums_the_counts_of_lm_during_the_search(tmp_path):
    ev = tmp_path / "events.jsonl"
    base = {"mode": "lm", "lm_calls": 3, "tuning_evals": 50, "lm_elite_calls": 2,
            "lm_elite_evals": 40, "lm_elite_kept": 1, "lm_elite_worse": 1,
            "lm_elite_guard_rejected": 0, "lm_elite_not_knob": 0, "lm_elite": []}
    ev.write_text("\n".join(json.dumps({"type": "end", "const_opt": base}) for _ in range(2))
                  + "\n")
    s = _runner().const_opt_summary(ev, 200)
    assert s["lm_calls"] == 6 and s["tuning_evals"] == 100 and s["tuning_share"] == 0.5
    assert s["lm_elite_calls"] == 4 and s["lm_elite_evals"] == 80 and s["lm_elite_kept"] == 2


def test_the_runner_summary_without_the_option_has_no_elite_keys(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text(json.dumps({"type": "end", "const_opt": {"mode": "lm", "lm_calls": 1}}) + "\n")
    assert not [k for k in _runner().const_opt_summary(ev, 100) if "elite" in k]
