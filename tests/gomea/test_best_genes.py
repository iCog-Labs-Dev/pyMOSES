"""NOT A PORT, backlog 15.4 (diagnostic D1): the option ``--gomea-log-best-genes`` and the field
``best_genes`` of the "end" events (``pymoses/gomea/best_genes.py``).

* The grid helpers: the step path of a code, the walk of ``contin_stepper``, the 63 values of a
  depth-5 knob (range +-31 x step size), the edge rule and the record of a free code.
* In process (the round 16 BASE-TLS flags on ``data/const_opt_none/train.csv``): the field is
  built from the oracle's best genotype without a change of state (the global generator, the
  value memo of the schema, the reduction cache of the representation), and its content is
  that of the schema: the programs, the values, the classes of ``rv_inert.analyse``. The
  logged reduced program has the score ``best.raw`` of the event (the scorer's rule), so the
  field describes the individual that ``best`` describes. With ``--linear-scaling 1`` the fit
  is plain; with ``--linear-scaling 0`` no gene is absorbed.
* Identity, against the references of ``data/rv_step_2d407b2/`` (``test_rv_step_identity.py``):
  with ``--gomea-log-best-genes 0``, stdout, the MOSES log (without the command line) and the
  GOMEA log (without the wall-clock fields) are the references; with ``1`` (``-a gomea``
  cases), stdout and the MOSES log are the references, and the GOMEA log is the reference
  after the field ``best_genes`` of each "end" event is removed. The same for the D1 arm (the
  round 16 BASE-TLS argv with ``--gomea-trace 1``).
* Identity through the runner (``moses_sr_runner.py``, the D1 arm argv of
  ``configs/gomea/d1/sr.json`` at 1,500 evaluations, with and without the option): the same
  stdout, ``candidates.json`` and summary (without the clock fields), and the same GOMEA log
  after the field is removed (and the clock fields).
* The analysis script ``tools/gomea/d1_grid_report.py`` on the runner output: the reduction
  and the leaf checks hold, ``mse0`` is the raw score of the log, and LM does not make the
  training MSE worse.
* Refusals: the CLI (``-a gomea`` only, ``--gomea-log`` needed, the values 0 and 1),
  ``gomea_parameters`` and the hybrid.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.gomea import best_genes as bgm
from pymoses.gomea import optimizer as gomea_optimizer
from pymoses.gomea.rv_inert import PLAIN, analyse
from pymoses.moses.representation.field_set import contin_spec
from pymoses.moses.scoring.bscores import contin_bscore
from pymoses.util.rng import randGen

REPO = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402
import test_rv_step_identity as ident  # noqa: E402
from round16_data import splits  # noqa: E402

DATA = TESTS / "data" / "const_opt_none"
TRAIN = DATA / "train.csv"
ON = ["--gomea-log-best-genes", "1"]
OFF = ["--gomea-log-best-genes", "0"]
CLASSES = {"zero", "dead", "absorbed", "live", "absent"}
D1 = json.loads((REPO / "configs" / "gomea" / "d1" / "sr.json").read_text())


# ---- the grid helpers


def test_step_path_and_value():
    spec = contin_spec(2.5, 1.0, 2.0, 5)
    assert bgm.step_path(0, 5) == ""
    L, R = contin_spec.Left, contin_spec.Right
    code = L | (L << 2) | (R << 4)  # L L R, then Stop
    assert bgm.step_path(code, 5) == "LLR"
    assert bgm.step_path(code | (L << 8), 5) == "LLR"  # the walk stops at the first Stop
    with pytest.raises(ValueError):
        bgm.step_path(3, 5)
    # -1 (the step becomes 2), -2 (the step becomes 4), then a turn: the step 4 / (2 x 2) = 1
    assert bgm.path_value(spec, "LLR") == 2.5 - 1 - 2 + 1
    assert bgm.path_value(spec, "LLLLL") == 2.5 - 31
    assert bgm.path_value(spec, "RRRRR") == 2.5 + 31
    assert bgm.path_value(spec, "") == 2.5


def test_the_grid_of_a_depth_5_knob():
    spec = contin_spec(0.0, 1.0, 2.0, 5)
    paths = bgm.all_paths(5)
    assert len(paths) == 63 == len(set(paths))
    grid = bgm.grid_values(spec)
    assert len(grid) == 63 == len(set(grid))  # each path gives its own value
    assert grid[0] == -31.0 and grid[-1] == 31.0
    assert min(b - a for a, b in zip(grid, grid[1:], strict=False)) == 0.0625
    # the values are those of field_set.get_contin (FrozenSchema.contin_value's walk)
    assert bgm.path_value(contin_spec(-3.0, 0.5, 2.0, 5), "RRRRR") == -3.0 + 31 * 0.5
    assert bgm.is_edge("LLLLL", 5) and bgm.is_edge("RRRRR", 5)
    assert not bgm.is_edge("LLLL", 5) and not bgm.is_edge("LLLLR", 5)
    assert not bgm.is_edge("", 0)


# ---- in process


def _argv(m: str = "1500", ls: str = "terms") -> list[str]:
    """The round 16 BASE-TLS argv (``configs/gomea/phase1bk/sr.json``) on the test data set."""
    argv = ident._reg(m) + ident._BASE_TLS + ["--gomea-trace", "1"]
    argv[argv.index("--linear-scaling") + 1] = ls
    return argv


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
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


def _rng_state():
    r = randGen()
    return (list(r._mt), r._mti)


def _run_captured(argv, tmp_path, monkeypatch):
    """``_run``, with each call of ``best_genes_record`` recorded as (schema, genotype, linear
    scaling, reduce, record), and the state of the run checked before and after the call."""
    calls = []
    real = gomea_optimizer.best_genes_record

    def wrapped(schema, g, ls, reduce):
        before = (_rng_state(), dict(schema._contin_values),
                  list((schema.rep._reduced_cache or {}).keys()))  # fmt: skip
        rec = real(schema, g, ls, reduce)
        after = (_rng_state(), dict(schema._contin_values),
                 list((schema.rep._reduced_cache or {}).keys()))  # fmt: skip
        assert after == before, "best_genes_record changed the state of the run"
        calls.append((schema, tuple(g), ls, reduce, rec))
        return rec

    monkeypatch.setattr(gomea_optimizer, "best_genes_record", wrapped)
    out, events = _run(argv, tmp_path)
    monkeypatch.undo()
    return out, events, calls


def _terms_score(program: str, ls: str) -> float:
    """The raw score (the sum of the behavioural score) of ``program`` on the test data set,
    by the rule of the run's scorer."""
    b = contin_bscore(loadTable(str(TRAIN), "y"))
    if ls == "terms":
        b.set_linear_scaling_terms(True)
    elif ls == "plain":
        b.set_linear_scaling(True)
    return math.fsum(b(str_to_combo_tree(program)))


def _check_record(schema, g, rec):
    """The content of one field against the schema (see the module docstring)."""
    tree, leaf_genes = schema.candidate_genes(g)
    assert rec["program"] == tree_to_stream_str(tree, 17)
    assert rec["leaf_genes"] == leaf_genes
    parsed = str_to_combo_tree(rec["program"])
    assert tree_to_stream_str(parsed, 17) == rec["program"]  # the text gives the tree back
    assert [g_["gene"] for g_ in rec["genes"]] == list(schema.contin_positions)
    for r in rec["genes"]:
        p = r["gene"]
        assert r["code"] == g[p]
        assert r["value"] == schema.contin_value(p, g[p])
        spec = schema._contin_specs[p]
        assert (r["mean"], r["step_size"], r["expansion"], r["depth"]) == (
            spec.mean, spec.step_size, spec.expansion, spec.depth)  # fmt: skip
        assert r["free"] is False and r["length"] == len(r["path"])
        assert r["edge"] == (r["path"] in ("L" * r["depth"], "R" * r["depth"]))
        assert r["class"] in CLASSES
        assert (r["class"] == "absent") == (p not in leaf_genes)
        if r["class"] != "absent":
            assert (r["class"] == "zero") == (r["value"] == 0.0)


def test_the_field_in_process(tmp_path, monkeypatch):
    out, events, calls = _run_captured(_argv("3000") + ON, tmp_path, monkeypatch)
    ends = [e for e in events if e["type"] == "end"]
    assert ends and len(calls) == sum(e["ledger"]["requests"] > 0 for e in ends)
    with_field = [e for e in ends if e["best_genes"] is not None]
    assert len(with_field) == len(calls)
    classes = set()
    for e, (schema, g, ls, reduce, rec) in zip(with_field, calls, strict=True):
        assert json.loads(json.dumps(rec)) == e["best_genes"]
        assert ls == "terms" and reduce is True
        assert rec["linear_scaling"] == "terms"
        _check_record(schema, g, rec)
        assert rec["program_reduced"] == tree_to_stream_str(schema.candidate(g, True), 17)
        a = analyse(schema, g, True)
        assert rec["fit"] == a["mode"]
        assert {r["gene"]: r["class"] for r in rec["genes"] if r["class"] != "absent"} == a["genes"]
        classes |= {r["class"] for r in rec["genes"]}
        # the logged reduced program has the event's raw score: the field describes `best`
        want = e["best"]["raw"]
        assert math.isclose(_terms_score(rec["program_reduced"], "terms"), want, rel_tol=1e-6)
    assert {"zero", "absorbed", "live"} <= classes


@pytest.mark.parametrize("ls, name", [("1", "plain"), ("0", "none")])
def test_the_classes_follow_the_linear_scaling(ls, name, tmp_path, monkeypatch):
    _out, events, calls = _run_captured(_argv("1500", ls) + ON, tmp_path, monkeypatch)
    assert calls
    for e, (schema, g, got_ls, reduce, rec) in zip(
        [e for e in events if e["type"] == "end" and e["best_genes"] is not None],
        calls,
        strict=True,
    ):
        assert got_ls == name and rec["linear_scaling"] == name and rec["fit"] == PLAIN
        _check_record(schema, g, rec)
        want = analyse(schema, g, reduce, mode=PLAIN)["genes"]
        got = {r["gene"]: r["class"] for r in rec["genes"] if r["class"] != "absent"}
        if name == "none":
            assert "absorbed" not in got.values()
            want = {p: ("live" if c == "absorbed" else c) for p, c in want.items()}
        assert got == want
        assert math.isclose(_terms_score(rec["program_reduced"], name), e["best"]["raw"],
                            rel_tol=1e-6)  # fmt: skip


def test_a_free_code_record():
    """A free code (``--const-opt mutate``) has its registered value and no path."""
    from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
    from pymoses.gomea.schema import FrozenSchema
    from pymoses.moses.representation.representation import representation

    rep = representation(lambda _t: None, lambda _t: None, str_to_combo_tree("$1"),
                         gen_signature(type_node.contin_type, 1), linear_contin=True)  # fmt: skip
    schema = FrozenSchema(rep, contin="copy")
    p = schema.contin_positions[0]
    code = schema.register_contin_value(p, 0.123456789)
    r = bgm.gene_record(schema, p, code, "live")
    assert r["free"] is True and r["path"] is None and r["length"] is None
    assert r["edge"] is False and r["value"] == 0.123456789


# ---- identity against the references of test_rv_step_identity.py


def strip_best_genes(events: str) -> str:
    """The canonical GOMEA log without ``best_genes``. Each "end" event must have the field,
    ``None`` exactly when the deme has no evaluation."""
    lines = []
    for line in events.splitlines():
        ev = json.loads(line)
        if ev["type"] == "end":
            bg = ev.pop("best_genes")
            assert (bg is None) == (ev["ledger"]["requests"] == 0)
            assert bg is None or ("error" not in bg and "program" in bg)
        lines.append(json.dumps(ev, sort_keys=True))
    return "\n".join(lines) + "\n" if lines else ""


@pytest.mark.parametrize("case", sorted(ident.CASES))
def test_the_option_off_is_byte_identical(case, tmp_path):
    stdout, log, events = ident.run_case(ident.CASES[case] + OFF, tmp_path)
    assert stdout == (ident.REF / f"{case}.out").read_text()
    assert log.replace(" " + " ".join(OFF), "") == (ident.REF / f"{case}.log").read_text()
    assert events == (ident.REF / f"{case}.events.jsonl").read_text()


@pytest.mark.parametrize("case", sorted(c for c in ident.CASES if c != "hc-gomea"))
def test_the_option_on_adds_only_its_field(case, tmp_path):
    stdout, log, events = ident.run_case(ident.CASES[case] + ON, tmp_path)
    assert stdout == (ident.REF / f"{case}.out").read_text()
    assert log.replace(" " + " ".join(ON), "") == (ident.REF / f"{case}.log").read_text()
    assert strip_best_genes(events) == (ident.REF / f"{case}.events.jsonl").read_text()
    assert '"best_genes": {' in events


def test_the_d1_arm_adds_only_its_field(tmp_path):
    """The round 16 BASE-TLS argv with ``--gomea-trace 1`` (the D1 arm), with and without the
    option."""
    argv = ident.CASES["base-tls-m3000-r2"] + ident.TRACE
    stdout, log, events = ident.run_case(argv + ON, tmp_path)
    ref = "base-tls-m3000-r2"
    assert stdout == (ident.REF / f"{ref}.out").read_text()
    flags = " " + " ".join(ident.TRACE + ON)
    assert log.replace(flags, "") == (ident.REF / f"{ref}.log").read_text()
    assert ident.strip_trace(strip_best_genes(events)) == (
        ident.REF / f"{ref}.events.jsonl").read_text()  # fmt: skip


# ---- identity through the runner, and the analysis script


def _job(work: Path, argv: list[str]) -> dict:
    return {
        "run_id": "T__D1__s1", "task": "T", "arm": "D1", "seed": 1,
        "binary": str(REPO / ".venv" / "bin" / "moses"), "argv": argv,
        "budget": {"evals": 1500}, "seed_spec": None,
        "split_dir": str(splits(work / "splits" / "s1")), "wall_cap_s": 900,
        "mem_limit_mb": None, "run_dir": str(work / "run"),
        "clip_test_nmse": True, "plain_nmse": True, "anytime": True,
    }  # fmt: skip


CLOCK = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")


def _canonical_events(text: str) -> str:
    return ident.canonical_events(text).replace(str(Path.cwd()), "CWD")


def _summary(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    return {k: v for k, v in s.items() if k not in CLOCK}


@pytest.mark.skipif(not (REPO / ".venv" / "bin" / "moses").exists(), reason="no .venv/bin/moses")
def test_the_runner_and_the_report(tmp_path):
    argv_on = D1["arms"]["BASE-TLS-D1"]["argv"]
    assert argv_on[-2:] == ON
    argv_off = argv_on[:-2]
    runs = {}
    for name, argv in (("off", argv_off), ("on", argv_on)):
        work = tmp_path / name
        s = runner._run_one(_job(work, argv))
        assert s["status"] == "ok", s.get("error")
        runs[name] = work / "run"
    off, on = runs["off"], runs["on"]
    for f in ("stdout.txt", "candidates.json"):
        assert (on / f).read_bytes() == (off / f).read_bytes(), f
    assert _summary(on) == _summary(off)
    ev_on = (on / "events.jsonl").read_text().replace(str(on), "RUN")
    ev_off = (off / "events.jsonl").read_text().replace(str(off), "RUN")
    assert strip_best_genes(ident.canonical_events(ev_on)) == ident.canonical_events(ev_off)
    # the analysis script (python -I) on the run with the option; it writes only into --out
    before = sorted((p.name, p.stat().st_mtime_ns) for p in on.iterdir())
    out = tmp_path / "report"
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "tools" / "gomea" / "d1_grid_report.py"), str(on),
         "--out", str(out)],
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    assert sorted((p.name, p.stat().st_mtime_ns) for p in on.iterdir()) == before
    demes = list(csv.DictReader((out / "demes.csv").open()))
    ends = [json.loads(x) for x in (on / "events.jsonl").read_text().splitlines()]
    ends = [e for e in ends if e["type"] == "end" and e.get("best_genes")]
    assert len(demes) == len(ends) > 0
    for d in demes:
        assert d["reduce_match"] == "True" and d["leaves_match"] == "True"
        assert float(d["raw_rel_diff"]) < 1e-6
        assert float(d["mse_lm_genes"]) <= float(d["mse0"])
        assert float(d["mse_lm_program"]) <= float(d["mse0_program"])
        assert math.isclose(float(d["mse0"]), float(d["mse0_program"]), rel_tol=1e-9)
    genes = list(csv.DictReader((out / "genes.csv").open()))
    assert len(genes) == sum(len(e["best_genes"]["genes"]) for e in ends)
    live = [g for g in genes if g["class"] == "live"]
    assert all(g["lm_value"] != "" for g in live)
    for name in ("summary.csv", "distributions.csv", "summary.txt"):
        assert (out / name).stat().st_size > 0


def test_the_report_grid_position():
    import d1_grid_report as rep

    grid = bgm.grid_values(contin_spec(0.0, 1.0, 2.0, 5))
    p = rep.grid_position(0.95, grid, 1.0)
    assert p["in_range"] and p["nearest"] == 0.9375 and p["spacing"] == 0.0625
    assert math.isclose(p["dist_rel"], 0.0125 / 0.0625)
    q = rep.grid_position(33.0, grid, 1.0)
    assert not q["in_range"] and q["outside_steps"] == 2.0 and q["dist_rel"] is None
    e = rep.grid_position(-31.0, grid, 1.0)
    assert e["in_range"] and e["dist"] == 0.0 and e["dist_rel"] == 0.0


# ---- refusals


def _cli_error(argv, tmp_path):
    with pytest.raises(ValueError) as exc:
        _run(argv, tmp_path)
    return str(exc.value)


@pytest.mark.parametrize("algo", ["hc", "hc-capped", "hc-gomea"])
def test_the_cli_refuses_the_option_without_gomea(algo, tmp_path):
    argv = ident._reg("1500") + ident._BASE_TLS + ON  # no trace: it refuses these too
    argv[argv.index("-a") + 1] = algo
    msg = f"--gomea-log-best-genes 1 needs -a gomea, got -a {algo}"
    assert _cli_error(argv, tmp_path) == msg


def test_the_cli_refuses_the_option_without_a_log_and_a_wrong_value(tmp_path):
    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError) as exc:
            moses_exec(["moses", *_argv(), *ON])
    finally:
        os.chdir(old)
    assert str(exc.value) == "--gomea-log-best-genes 1 needs --gomea-log FILE"
    assert _cli_error(_argv() + ["--gomea-log-best-genes", "2"], tmp_path) == (
        "--gomea-log-best-genes must be 0 or 1, got 2")  # fmt: skip


def test_gomea_parameters_and_the_hybrid():
    from pymoses.gomea.hybrid import hc_gomea_optimization

    gp = gomea_optimizer.gomea_parameters
    assert "log_best_genes" not in gp().as_dict()
    assert "log_best_genes" not in gp(log_best_genes=True).as_dict()  # always left out
    with pytest.raises(ValueError, match="log_best_genes must be True or False"):
        gp(log_best_genes=1)
    with pytest.raises(ValueError, match="log_best_genes needs -a gomea, not the hc-gomea"):
        hc_gomea_optimization(None, None, gp(log_best_genes=True))
