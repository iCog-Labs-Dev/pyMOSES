"""Server round 12, T1.6: LM on the top K programs of each GOMEA deme (``-a gomea --const-opt
lm --const-opt-top-k K``), as ``docs/gomea/SERVER-ROUND12-PREREGISTRATION.md`` ("T1.6")
registers it: at the end of each deme, LM with the validation guard on up to K programs with
distinct structures, the first one being the program that BASE tunes. The identity of the
arms without the option is ``test_const_opt_topk_during_identity.py``."""

from __future__ import annotations

import csv
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.gomea.const_opt import structure_key
from pymoses.gomea.optimizer import gomea_optimization

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
TRAIN = DATA / "train.csv"

# The round 12 BASE flags (the round 11 BASE arm, configs/gomea/phase1be/), with the
# training table as the validation table of the guard.
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
    "--const-opt-val", str(TRAIN),
]  # fmt: skip
TOPK_KEYS = ("lm_topk", "lm_topk_picked", "lm_topk_best_in_deme")


def _reg(m: str) -> list[str]:
    return ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


def _k(k: int) -> list[str]:
    return ["--const-opt-top-k", str(k)]


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


def _without_topk(events):
    """The events with the option's own telemetry left out."""
    out = []
    for e in events:
        e = json.loads(json.dumps(e))
        e.get("params", {}).pop("const_opt_top_k", None)
        for k in TOPK_KEYS:
            (e.get("const_opt") or {}).pop(k, None)
        for k in ("linkage_s", "variation_s", "timings", "max_time"):
            e.pop(k, None)
        out.append(e)
    return out


# ---- the flag


@pytest.mark.parametrize("extra, message", [
    (_BASE + _k(-1), "--const-opt-top-k must be >= 0"),
    (["-a", "gomea", "--gomea-contin", "copy", "--linear-scaling", "1"] + _k(5),
     "--const-opt-top-k needs --const-opt lm"),
    (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "mutate"] + _k(5),
     "--const-opt-top-k needs --const-opt lm"),
    (["-a", "hc", "--linear-scaling", "1", "--const-opt", "lm"] + _k(5),
     "--const-opt-top-k needs -a gomea"),
    (["-a", "hc-gomea", "--gomea-contin", "copy", "--const-opt", "lm"] + _k(5),
     "--const-opt-top-k needs -a gomea"),
], ids=["negative", "no-const-opt", "mutate", "hc", "hc-gomea"])
def test_the_flag_is_refused(extra, message, tmp_path, capsys):
    assert message in _run_error(_reg("300") + extra, tmp_path, capsys)


def test_the_flag_needs_an_integer(tmp_path, capsys):
    assert "invalid int value" in _run_error(_reg("300") + _BASE + ["--const-opt-top-k", "2.5"],
                                             tmp_path, capsys)


@pytest.mark.parametrize("k", [-1, 2.0, "5", True])
def test_the_parameters_refuse_a_bad_k(k):
    from pymoses.gomea.optimizer import gomea_parameters

    with pytest.raises(ValueError, match="const_opt_top_k must be an integer >= 0"):
        gomea_parameters(const_opt="lm", const_opt_top_k=k)


def test_the_parameters_need_lm():
    from pymoses.gomea.optimizer import gomea_parameters

    with pytest.raises(ValueError, match="const_opt_top_k needs const_opt 'lm'"):
        gomea_parameters(const_opt_top_k=5)
    with pytest.raises(ValueError, match="const_opt_top_k needs const_opt 'lm'"):
        gomea_parameters(const_opt="mutate", contin="copy", const_opt_top_k=5)


def test_the_hybrid_and_hc_lm_refuse_top_k():
    from pymoses.gomea.hc_lm import hc_lm_optimization
    from pymoses.gomea.hybrid import hc_gomea_optimization
    from pymoses.gomea.optimizer import gomea_parameters

    gp = gomea_parameters(const_opt="lm", const_opt_top_k=5)
    with pytest.raises(ValueError, match="const_opt_top_k needs -a gomea"):
        hc_gomea_optimization(gp=gp)
    with pytest.raises(ValueError, match="const_opt_top_k needs -a gomea"):
        hc_lm_optimization(gp=gp)


def test_k_is_left_out_of_the_telemetry_while_0():
    from pymoses.gomea.optimizer import gomea_parameters

    assert "const_opt_top_k" not in gomea_parameters(const_opt="lm").as_dict()
    assert gomea_parameters(const_opt="lm", const_opt_top_k=5).as_dict()["const_opt_top_k"] == 5


# ---- the structure key


def test_the_structure_key_ignores_only_the_constants():
    def key(text):
        return structure_key(str_to_combo_tree(text))

    assert key("+(*(2.5 $1) -1)") == key("+(*(0.1 $1) 7)")
    assert key("+(*(2.5 $1) -1)") != key("+(*(2.5 $2) -1)")
    assert key("+(*(2.5 $1) -1)") != key("+(*(2.5 $1) $2)")
    assert key("+(*(2.5 $1) -1)") != key("*(+(2.5 $1) -1)")
    assert key("+(1 $1)") != key("+(true $1)")  # a constant 1.0 is not the Boolean true


# ---- which programs are tuned


@pytest.fixture
def spy(monkeypatch):
    """Record, for each deme, the deme before LM, the best genotype and the tuned
    genotypes; the wrapped methods are called unchanged."""
    demes: list[dict] = []
    tune_top_k = gomea_optimization._tune_top_k
    tune_program = gomea_optimization._tune_program

    def _tune_top_k(self, deme, oracle, schema):
        demes.append({"entries": [(si.first, si.second) for si in deme],
                      "best": oracle.best_genotype(), "schema": schema, "tuned": []})
        return tune_top_k(self, deme, oracle, schema)

    def _tune_program(self, deme, oracle, schema, g):
        demes[-1]["tuned"].append(g)
        return tune_program(self, deme, oracle, schema, g)

    monkeypatch.setattr(gomea_optimization, "_tune_top_k", _tune_top_k)
    monkeypatch.setattr(gomea_optimization, "_tune_program", _tune_program)
    return demes


def test_k_distinct_structures_are_tuned_and_the_first_is_the_best(spy, tmp_path):
    """At -m 1500 the first deme ends by stagnation with evaluations left: LM tunes 5
    programs with distinct structures, the first one is the deme's best, and no structure
    that was not picked has an entry strictly better than a picked one."""
    _, events = _run(_reg("1500") + _BASE + _k(5), tmp_path)
    first = _ends(events)[0]
    assert first["termination_reason"] == "stagnation"
    d = spy[0]
    schema = d["schema"]

    def key(g):
        return structure_key(schema.candidate(g, True))

    tuned = d["tuned"]
    assert len(tuned) == 5 == first["const_opt"]["lm_calls"]
    assert len(first["const_opt"]["lm_topk"]) == 5 == first["const_opt"]["lm_topk_picked"]
    assert tuned[0] == d["best"]
    keys = [key(g) for g in tuned]
    assert len(set(keys)) == 5
    # the best entry of each picked structure (after the first pick), and every other entry
    best_of = {}
    for inst, score in d["entries"]:
        k = key(schema.to_genotype(inst))
        if k not in best_of or best_of[k] < score:
            best_of[k] = score
    weakest = min((best_of[k] for k in keys[1:]), key=lambda s: (s.get_penalized_score(),
                                                                -s.get_complexity()))
    for k, score in best_of.items():
        if k not in keys:
            assert not weakest < score


def test_the_first_call_equals_the_base_call(tmp_path):
    """The first deme of BASE-K5 is BASE's first deme up to LM: the first top-K call is
    BASE's LM call, and ``const_opt.lm`` is its record."""
    _, base = _run(_reg("1500") + _BASE, tmp_path)
    _, k5 = _run(_reg("1500") + _BASE + _k(5), tmp_path)
    b, k = _ends(base)[0]["const_opt"], _ends(k5)[0]["const_opt"]
    assert k["lm_topk"][0] == b["lm"] == k["lm"]
    assert k["lm_calls"] == 5 and b["lm_calls"] == 1
    upto = [e["type"] for e in base].index("end")
    assert _without_topk(base[:upto]) == _without_topk(k5[:upto])


def test_k1_tunes_the_same_program_as_base(tmp_path):
    """K = 1 is BASE: the same output, and the same events except the option's own
    telemetry."""
    out_b, base = _run(_reg("1500") + _BASE, tmp_path)
    out_1, k1 = _run(_reg("1500") + _BASE + _k(1), tmp_path)
    assert out_1 == out_b
    assert _without_topk(k1) == _without_topk(base)
    for e in _ends(k1):
        assert e["const_opt"]["lm_topk"] == [e["const_opt"]["lm"]]


@pytest.mark.parametrize("m", ["300", "600", "1500", "3000"])
def test_the_evaluations_never_pass_m(m, tmp_path):
    out, events = _run(_reg(m) + _BASE + _k(5), tmp_path)
    ends = _ends(events)
    n = _n_evals(out)
    assert n <= int(m)
    assert n == sum(e["ledger"]["requests"] for e in ends)
    for e in ends:
        co = e["const_opt"]
        assert co["lm_calls"] == len(co["lm_topk"]) <= co["lm_topk_picked"] <= 5
        assert co["tuning_evals"] == sum(r["evals"] for r in co["lm_topk"])
        # a call that ran out of the budget is the last one
        assert not any(r["budget"] for r in co["lm_topk"][:-1])


def test_the_best_missing_from_the_kept_deme_is_still_the_first_pick(monkeypatch, tmp_path):
    """With the complexity ratio, GOM can reject the best program under the MOSES score, so
    the kept deme can miss it. Simulated here: its entries are removed before the pick. It
    is still the first pick (the program BASE tunes), and the other picks are the first
    entries of the sorted deme with new structures."""
    seen: list[dict] = []
    orig = gomea_optimization._top_k_genotypes

    def _top_k_genotypes(self, deme, oracle, schema):
        best = oracle.best_genotype()
        inst = schema.to_instance(best)
        deme[:] = [si for si in deme if si.first != inst]
        picks, in_deme = orig(self, deme, oracle, schema)
        seen.append({"best": best, "picks": picks, "in_deme": in_deme, "schema": schema,
                     "entries": [(si.first, si.second) for si in deme]})
        return picks, in_deme

    monkeypatch.setattr(gomea_optimization, "_top_k_genotypes", _top_k_genotypes)
    _, events = _run(_reg("1500") + _BASE + _k(3), tmp_path)
    first = seen[0]
    assert first["in_deme"] is False
    assert _ends(events)[0]["const_opt"]["lm_topk_best_in_deme"] is False
    assert first["picks"][0] == first["best"]
    schema = first["schema"]
    entries = sorted(enumerate(first["entries"]), key=lambda p: (p[1][1], -p[0]),
                     reverse=True)
    keys = [structure_key(schema.candidate(first["best"], True))]
    expected = [first["best"]]
    for _, (inst, _score) in entries:
        g = schema.to_genotype(inst)
        k = structure_key(schema.candidate(g, True))
        if k not in keys and len(expected) < 3:
            keys.append(k)
            expected.append(g)
    assert first["picks"] == expected


def _table(path: Path, c: float = 1.37) -> Path:
    """40 rows, y = x1 (x2 + c): linear scaling cannot absorb c, so LM can lower the
    training error (as in ``test_hc_lm.py``)."""
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x1", "x2", "y"])
        for i in range(40):
            x1, x2 = -3 + 6 * i / 39, ((i * 7) % 11) / 5.0
            wr.writerow([repr(x1), repr(x2), repr(x1 * (x2 + c))])
    return path


def test_each_accepted_program_is_written_back(monkeypatch, tmp_path):
    """On y = x1 (x2 + 1.37) at -m 3000, LM improves all 5 programs of the second deme and
    the guard accepts them. Each tuned program replaces the plain entries of its instance in
    the deme (Lamarckian, as for the best one)."""
    after: list[dict] = []
    orig = gomea_optimization._tune_top_k

    def _tune_top_k(self, deme, oracle, schema):
        picks, _ = self._top_k_genotypes(deme, oracle, schema)
        orig(self, deme, oracle, schema)
        after.append({"picks": picks, "schema": schema, "co": dict(self._co),
                      "deme": [(si.first, getattr(si, "tree", None)) for si in deme]})

    monkeypatch.setattr(gomea_optimization, "_tune_top_k", _tune_top_k)
    train = _table(tmp_path / "train.csv")
    argv = ["-i", str(train), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", "3000",
            *_BASE[:-1], str(train), *_k(5)]
    out, events = _run(argv, tmp_path)
    assert _n_evals(out) <= 3000
    written = [d for d in after if sum(r.get("written_back", False)
                                       for r in d["co"]["lm_topk"]) >= 2]
    assert written, "no deme with two or more programs written back"
    d = written[0]
    schema = d["schema"]
    for g, rec in zip(d["picks"], d["co"]["lm_topk"], strict=False):
        inst = schema.to_instance(g)
        plain = [t for i, t in d["deme"] if i == inst and t is None]
        tuned = [t for i, t in d["deme"] if i == inst and t is not None]
        if rec.get("written_back"):
            assert rec["status"] == "improved" and rec["guard"] == "accepted"
            assert not plain and len(tuned) == 1
        else:
            assert not tuned


# ---- with the LM reserve (user decision 2026-11-16: the arm BASE-K5 is BASE +
# --const-opt-reserve 500 + --const-opt-top-k 5)


def _reserve(r: int) -> list[str]:
    return ["--const-opt-reserve", str(r)]


def _search(end):
    """The evaluations of the GOMEA search of a deme (LM passes left out)."""
    return end["ledger"]["requests"] - end["ledger"].get("tuning_requests", 0)


def test_the_parameters_accept_top_k_with_the_reserve():
    from pymoses.gomea.optimizer import gomea_parameters

    d = gomea_parameters(const_opt="lm", const_opt_top_k=5, const_opt_reserve=500).as_dict()
    assert d["const_opt_top_k"] == 5 and d["const_opt_reserve"] == 500


def test_with_the_reserve_the_walk_gets_the_evaluations_that_are_left(tmp_path):
    """At -m 600 the one deme of BASE ends at the budget, so LM gets no evaluations
    (``test_const_opt_reserve.py``). With R = 500 the search stops at 300 (the deme budget
    600 minus min(500, 300)), and the top-K walk uses the evaluations that are left."""
    out, events = _run(_reg("600") + _BASE + _reserve(500) + _k(5), tmp_path)
    first = _ends(events)[0]
    assert first["const_opt_reserve"] == 300
    assert _search(first) == 300
    co = first["const_opt"]
    assert len(co["lm_topk"]) >= 2 and co["lm_topk"][0]["status"] != "budget"
    assert co["lm_calls"] == len(co["lm_topk"])
    assert co["tuning_evals"] == sum(r["evals"] for r in co["lm_topk"]) > 0
    assert _n_evals(out) <= 600


def test_k1_with_the_reserve_is_base_r(tmp_path):
    """K = 1 with the reserve is BASE-R: the same output, and the same events except the
    option's own telemetry."""
    out_r, base_r = _run(_reg("1500") + _BASE + _reserve(500), tmp_path)
    out_1, k1 = _run(_reg("1500") + _BASE + _reserve(500) + _k(1), tmp_path)
    assert out_1 == out_r
    assert _without_topk(k1) == _without_topk(base_r)


@pytest.mark.parametrize("m", ["600", "1500", "3000"])
def test_with_the_reserve_the_evaluations_never_pass_m(m, tmp_path):
    out, events = _run(_reg(m) + _BASE + _reserve(500) + _k(5), tmp_path)
    ends = _ends(events)
    starts = [e for e in events if e["type"] == "invocation_start"]
    n = _n_evals(out)
    assert n <= int(m)
    assert n == sum(e["ledger"]["requests"] for e in ends)
    for start, end in zip(starts, ends, strict=True):
        budget = start["max_evals"]  # no deme cap: the deme budget is what is left of -m
        assert end["const_opt_reserve"] == min(500, budget // 2)
        assert _search(end) <= budget - end["const_opt_reserve"]
        assert end["ledger"]["requests"] <= budget
        co = end["const_opt"]
        assert co["tuning_evals"] == sum(r["evals"] for r in co["lm_topk"])
        # a call that ran out of the budget is the last one
        assert not any(r["budget"] for r in co["lm_topk"][:-1])
