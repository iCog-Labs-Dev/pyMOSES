"""Server round 15 (RV16 nit 1): the verdict lines of the round summary (orchestrate.py, the
table "Verdicts (contrasts.csv)") and of the live scoreboard (tools/gomea/scores_now.py).

* A row of an interaction contrast (values in the columns a2 and b2) shows its contrast name and
  its four arms. In a file with the column a2, a row with a value in holm_family shows it.
* A contrasts.csv without the column a2 (each round before 15) gives the same text as the code
  at 1abde69 (git), byte for byte, and the same errors.
"""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "server"))
import orchestrate  # noqa: E402

OLD = "1abde69"  # the commit before this change
SCORES = REPO / "tools" / "gomea" / "scores_now.py"

# The columns in the order that phase_report.py writes them for the 1bj spec (made-up values).
ROUND15 = """\
contrast,family,problem,a,a2,a_lower,b,b2,b_lower,holm_family,median_a,median_a2,median_b,\
median_b2,median_log_ratio,metric,p,p_holm,pairs,ratio,ties,verdict
uni-gain-it-vs-t,primary,pooled,IT-UNI,T-UNI,38,IT-RT,T-RT,62,P1,12.5,12.0,12.4,11.9,0.004,\
interaction:train_nmse,0.2,0.8,100,1.004,0,tie
ltb-gain-it-vs-t,primary,pooled,IT-LTB,T-LTB,90,IT-RT,T-RT,10,P1,7.5,11.4,12.4,11.9,-0.46,\
interaction:train_nmse,1e-12,4e-12,100,0.63,0,A1 gains more
it-uni-vs-rt,primary,pooled,IT-UNI,,40,IT-RT,,60,P2,12.5,,12.4,,0.003,log_ratio:train_nmse,\
0.3,0.9,100,1.003,0,tie
t-ltb-vs-rt,secondary,pooled,T-LTB,,70,T-RT,,30,S1,11.4,,11.9,,-0.05,log_ratio:train_nmse,\
0.001,0.005,100,0.95,0,tie
uni-gain-it-vs-t,descriptive,Air,IT-UNI,T-UNI,4,IT-RT,T-RT,6,D,12.1,11.8,12.0,11.7,0.002,\
interaction:train_nmse,0.6,1.0,10,1.002,0,descriptive
"""

# Files of the rounds before 15, in the shapes of the real reports (made-up values).
OLD_STYLE = {
    # rounds 8-14: per-problem and pooled rows, Holm families, a descriptive row
    "p8": ("""\
contrast,family,problem,a,a_lower,b,b_lower,holm_family,median_a,median_b,median_log_ratio,\
metric,p,p_holm,pairs,ratio,ties,verdict
main,primary,Air,A,3,B,7,,1.5,1.2,,,0.125,0.25,10,,0,tie
s-a,secondary,Con,A,8,C,2,secondary/s-a+s-b,1.1,1.6,,,0.01,0.02,10,,0,A better
pool,secondary,pooled,A,60,B,40,pooled_secondary/pool,1.3,1.2,0.01,log_ratio:test_nmse,0.04,\
0.08,100,1.01,0,tie
""", """\
contrast,family,problem,a,a_lower,b,b_lower,holm_family,median_a,median_b,p,p_holm,pairs,ties,\
verdict
main,primary,Air,A,9,B,1,,1.5,1.9,0.002,0.004,10,0,A better
d,descriptive,Air,A,5,D,5,,1.5,1.5,0.9,,10,0,
"""),
    # phase 1aa: per-problem rows with a family, no holm_family
    "p1aa": ("""\
contrast,family,problem,a,a_lower,b,b_lower,median_a,median_b,metric,p,p_holm,pairs,ties,verdict
b3-hc,primary,mux11,HC-B3,0,HC,10,inf,423.7,cpu_to_target,0.001953125,0.00390625,10,0,A worse
b3-gomea,secondary,Air,GOMEA-B3,3,GOMEA,7,1.5,1.2,test_nmse,0.125,0.25,10,0,tie
""", None),
    # phase 1m: no family column
    "p1m": ("""\
contrast,problem,a,a_lower,b,b_lower,median_a,median_b,p,p_holm,pairs,ties,verdict
design,mux11,G-D10-LTB,0,G-C250-N16,10,30006.0,5344.5,0.001953125,0.015625,10,0,A worse
""", None),
    # phase 1i: no problem, a or b column
    "p1i": ("""\
contrast,mean,negative,p,p_holm,pairs,positive,verdict,zero
start,-0.35,17,0.0105,0.0211,40,5,uniform better,18
""", None),
}


def _old_module(tmp_path, rel, name):
    """The module `rel` at the commit OLD, loaded under the name `name`."""
    r = subprocess.run(["git", "show", f"{OLD}:{rel}"], cwd=REPO, capture_output=True)
    if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
        pytest.fail(f"git show {OLD}:{rel} failed: {r.stderr.decode().strip()!r}. The test "
                    f"needs the reference commit {OLD}; use a clone with the full history.")
    path = tmp_path / f"{name}.py"
    path.write_bytes(r.stdout)
    return _load(path, name)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _contrasts(res, phase, stage1, stage2=None):
    for stage, text in ((1, stage1), (2, stage2)):
        if text is not None:
            d = res / phase / f"report-stage{stage}"
            d.mkdir(parents=True)
            (d / "contrasts.csv").write_text(text)
    # scores_now.py shows the verdicts of a phase only when the phase has runs
    run = res / phase / "x" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(
        {"task": "Air", "arm": "A", "status": "ok", "test_nmse": 1.0, "train_nmse": 0.9,
         "final_program": "$1", "n_evals": 10, "cpu_s": 1.0}))


def _summary(mod, res, phases):
    """round-summary.md of a plan with one phase step for each of `phases`."""
    plan = res / "plan-verdicts.json"
    plan.write_text(json.dumps({"name": "verdicts", "state_dir": "server-run-verdicts",
                                "steps": [{"id": p, "type": "phase", "phase": p}
                                          for p in phases]}))
    mod.Round(plan).write_summary()
    return (res / "server-run-verdicts" / "round-summary.md").read_text()


def _verdict_table(summary, phase):
    lines = summary.split(f"### {phase} stage 1\n\n", 1)[1].splitlines()
    return lines[2:lines.index("")]


def _table(mod, phase):
    """The stdout of scores_now.table(phase), and the error that stopped it (or None)."""
    out, err = io.StringIO(), None
    with contextlib.redirect_stdout(out):
        try:
            mod.table(phase)
        except Exception as e:  # the old code stops on some old files
            err = repr(e)
    return out.getvalue(), err


@pytest.fixture
def old_orchestrate(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the old module adds its directory
    mod = _old_module(tmp_path, "tools/server/orchestrate.py", f"orchestrate_{OLD}")
    monkeypatch.setattr(mod, "RES", tmp_path / "res")
    monkeypatch.setattr(orchestrate, "RES", tmp_path / "res")
    return mod


@pytest.fixture
def scores(tmp_path):
    """(scores_now.py now, scores_now.py at OLD), both on the results root tmp_path/res."""
    new = _load(SCORES, "scores_now")
    old = _old_module(tmp_path, "tools/gomea/scores_now.py", f"scores_now_{OLD}")
    for mod in (new, old):
        mod.RES = tmp_path / "res"
    return new, old


# ---- orchestrate.py: the table "Verdicts (contrasts.csv)" of the round summary


def test_summary_names_the_interaction_contrast_and_its_holm_family(old_orchestrate, tmp_path):
    res = tmp_path / "res"
    _contrasts(res, "phase1bj", ROUND15)
    assert _verdict_table(_summary(orchestrate, res, ["phase1bj"]), "phase1bj") == [
        "| pooled | uni-gain-it-vs-t: (IT-UNI vs IT-RT) vs (T-UNI vs T-RT) | primary (P1) | 38 | "
        "62 | 0.8 | tie |",
        "| pooled | ltb-gain-it-vs-t: (IT-LTB vs IT-RT) vs (T-LTB vs T-RT) | primary (P1) | 90 | "
        "10 | 4e-12 | A1 gains more |",
        "| pooled | IT-UNI vs IT-RT | primary (P2) | 40 | 60 | 0.9 | tie |",
        "| pooled | T-LTB vs T-RT | secondary (S1) | 70 | 30 | 0.005 | tie |",
        "| Air | uni-gain-it-vs-t: (IT-UNI vs IT-RT) vs (T-UNI vs T-RT) | descriptive (D) | 4 | 6 "
        "| 1.0 | descriptive |"]
    # the nit: at OLD the P1 row looks like the P2 row of the same arms
    old = _verdict_table(_summary(old_orchestrate, res, ["phase1bj"]), "phase1bj")
    assert old[0] == "| pooled | IT-UNI vs IT-RT | primary | 38 | 62 | 0.8 | tie |"
    assert old[2] == "| pooled | IT-UNI vs IT-RT | primary | 40 | 60 | 0.9 | tie |"


def test_summary_of_an_old_round_is_byte_identical(old_orchestrate, tmp_path):
    res = tmp_path / "res"
    for phase, (s1, s2) in OLD_STYLE.items():
        _contrasts(res, phase, s1, s2)
    old = _summary(old_orchestrate, res, list(OLD_STYLE))
    new = _summary(orchestrate, res, list(OLD_STYLE))
    assert new == old
    assert "| Air | A vs B | primary | 3 | 7 | 0.25 | tie |" in new.splitlines()
    assert "| Con | A vs C | secondary | 8 | 2 | 0.02 | A better |" in new.splitlines()
    assert "| None | None vs None | None | None | None | 0.0211 | uniform better |" in new
    assert new.count("### ") == len(OLD_STYLE) + 1  # p8 has a stage 2 report too


# ---- tools/gomea/scores_now.py: the stage verdict lines of the live scoreboard


def test_scoreboard_names_the_interaction_contrast_and_its_holm_family(scores, tmp_path):
    new, old = scores
    _contrasts(tmp_path / "res", "phase1bj", ROUND15)
    out, err = _table(new, "phase1bj")
    assert err is None
    assert out.split("  stage 1 verdicts:\n", 1)[1].splitlines() == [
        "    pooled uni-gain-it-vs-t: (IT-UNI vs IT-RT) vs (T-UNI vs T-RT) (primary, P1): 38/62 "
        "lower, Holm p 0.8 -> tie",
        "    pooled ltb-gain-it-vs-t: (IT-LTB vs IT-RT) vs (T-LTB vs T-RT) (primary, P1): 90/10 "
        "lower, Holm p 4e-12 -> A1 gains more",
        "    pooled IT-UNI vs IT-RT (primary, P2): 40/60 lower, Holm p 0.9 -> tie",
        "    pooled T-LTB vs T-RT (secondary, S1): 70/30 lower, Holm p 0.005 -> tie",
        "    Air    uni-gain-it-vs-t: (IT-UNI vs IT-RT) vs (T-UNI vs T-RT) (descriptive, D): 4/6 "
        "lower, Holm p 1 -> descriptive"]
    old_out, _ = _table(old, "phase1bj")
    assert "    pooled IT-UNI vs IT-RT (primary): 38/62 lower, Holm p 0.8 -> tie" in old_out


@pytest.mark.parametrize("phase", list(OLD_STYLE))
def test_scoreboard_of_an_old_round_is_byte_identical(scores, tmp_path, phase):
    new, old = scores
    _contrasts(tmp_path / "res", phase, *OLD_STYLE[phase])
    assert _table(new, phase) == _table(old, phase)


def test_the_old_files_cover_the_old_lines_and_errors(scores, tmp_path):
    new, _ = scores
    for phase, (s1, s2) in OLD_STYLE.items():
        _contrasts(tmp_path / "res", phase, s1, s2)
    got = {p: _table(new, p) for p in OLD_STYLE}
    assert "    Air    A vs B (primary): 3/7 lower, Holm p 0.25 -> tie\n" in got["p8"][0]
    assert "    pooled A vs B (secondary): 60/40 lower, Holm p 0.08 -> tie\n" in got["p8"][0]
    assert got["p8"][1] == repr(ValueError("could not convert string to float: ''"))
    assert got["p1aa"][1] is None
    assert "    mux11  HC-B3 vs HC (primary): 0/10 lower, Holm p 0.00391 -> A worse\n" in (
        got["p1aa"][0])
    assert got["p1m"][1] == repr(KeyError("family"))
    assert got["p1i"][1] == repr(KeyError("problem"))
