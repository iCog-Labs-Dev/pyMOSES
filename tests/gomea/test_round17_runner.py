"""Server round 17 (``SERVER-ROUND17-PREREGISTRATION.md``): the runner additions of
``tools/gomea/moses_sr_runner.py`` for the HC arms (the manifest key ``"hc_anytime"``).

* ``hc_anytime_records``: the ``demes``, ``anytime`` and ``anytime_check`` records from the lines
  of the expansion hook (``hc_progress.py``), by hand; the check of the rule of the HC "end"
  events (``budget - max_evals + ledger.requests``) against the lines.
* ``_wants_hc_anytime`` and ``expand``: the key is opt-in, for pyMoses ``-a hc`` arms only.
* ``_run_one`` with a fake pyMoses: the boot installs the hook only with the key, and the
  summary gets the fields from the lines.
* Byte identity against 88a5e02 (the commit before this change, from git): a GOMEA arm (with
  and without the round 16 key ``"anytime"``, with the trace) and an HC-LM arm without the key
  give the same candidates, summary and command line.

Real runs: ``test_hc_progress.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402
from round16_data import (  # noqa: E402
    DIGEST,
    TWO_DEMES,
    TWO_DEMES_TRACE,
    gomea_log,
    job,
)
from round17_data import (  # noqa: E402
    EVENTS,
    HC_ENDS,
    HC_LM,
    HC_TLS,
    LINES,
    hc_job,
    hc_lines,
)

OLD = "88a5e02"  # the commit before the round 17 runner changes
VOLATILE = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")
NEW = {"demes", "anytime", "anytime_source", "anytime_check"}
N, VAR = 100, 100.0  # made-up n_train and var(y')


def _nmse(raw):
    return 100.0 * (-raw / N) / VAR


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ---- hc_anytime_records


def test_the_checkpoints_take_the_best_of_the_expansions_that_ended_before(tmp_path):
    p = _write(tmp_path, "p.jsonl", hc_lines((250, -900.0), (520, -700.0), (800, -650.0),
                                           (1000, -650.0)))
    demes, recs, chk = runner.hc_anytime_records(p, 1000, N, VAR, n_evals=1000)
    assert demes == {"count": 4, "start_evals": [0, 250, 520, 800],
                     "requests": [250, 270, 280, 200], "second_start_evals": 250,
                     "evals_total": 1000}
    assert [r["evals"] for r in recs] == [10, 20, 50, 100, 200, 500, 1000]
    assert [r["frac"] for r in recs] == list(runner.ANYTIME_FRACTIONS)
    want = [None, None, None, None, None, -900.0, -650.0]
    assert [r["best_train_nmse"] for r in recs] == [None if w is None else _nmse(w)
                                                    for w in want]
    assert [r["log_evals"] for r in recs] == [None] * 5 + [250, 1000]
    assert all(r["reached"] for r in recs)
    assert chk["anytime_source"] == runner.HC_ANYTIME_SOURCE == "hc-demes"
    assert chk["log_evals_total"] == 1000 and chk["log_evals_match_n_evals"] is True
    assert chk["expansions"] == 4 and chk["best_raw_monotone"] is True
    assert chk["log_best_raw"] == -650.0 and chk["log_best_train_nmse"] == _nmse(-650.0)
    assert "hc_end_events" not in chk


def test_a_run_that_ends_before_the_budget_and_a_cut_line(tmp_path):
    text = hc_lines((300, None), (600, -500.0)) + '{"expansion": 3, "n_ev'
    p = _write(tmp_path, "p.jsonl", text)
    demes, recs, chk = runner.hc_anytime_records(p, 1000, N, VAR, first_score=-500.0,
                                                 first_train_nmse=_nmse(-500.0) * 1.01,
                                                 n_evals=601)
    assert demes["count"] == 2 and demes["evals_total"] == 600
    assert [r["reached"] for r in recs] == [True] * 6 + [False]
    # the first line has no candidate yet (null); the last checkpoint has the final value
    assert recs[4]["best_train_nmse"] is None and recs[5]["best_train_nmse"] is None
    assert recs[6]["best_train_nmse"] == _nmse(-500.0) and recs[6]["log_evals"] == 600
    assert chk["log_evals_match_n_evals"] is False and chk["expansions"] == 2
    assert chk["log_best_is_first"] is True
    assert chk["search_rel_diff"] == pytest.approx(0.01 / 1.01)


def test_no_lines(tmp_path):
    demes, recs, chk = runner.hc_anytime_records(_write(tmp_path, "p.jsonl", ""), 1000, N, VAR)
    assert demes is None and all(r["best_train_nmse"] is None for r in recs)
    assert chk["expansions"] == 0 and chk["log_evals_total"] == 0
    assert chk["log_evals_match_n_evals"] is None


def test_the_rule_of_the_hc_end_events(tmp_path):
    p = _write(tmp_path, "p.jsonl", hc_lines((250, -900.0), (520, -700.0), (1000, -700.0)))
    ev = _write(tmp_path, "events.jsonl", "".join(json.dumps(e) + "\n" for e in HC_ENDS))
    _, _, chk = runner.hc_anytime_records(p, 1000, N, VAR, events=ev)
    assert chk["hc_end_events"] == 3 and chk["hc_end_evals_match"] is True
    # deme 1: lm.mse (written back); deme 2: mse0 (the guard rejected LM); deme 3: no value
    assert chk["hc_end_best_train_nmse"] == pytest.approx(100.0 * 7.0 / VAR)
    assert chk["hc_end_rel_diff"] == pytest.approx(abs(7.0 - _nmse(-700.0)) / _nmse(-700.0))
    # a line at another evaluation, or a missing event, breaks the match
    bad = _write(tmp_path, "q.jsonl", hc_lines((250, -900.0), (521, -700.0), (1000, -700.0)))
    assert runner.hc_anytime_records(bad, 1000, N, VAR, events=ev)[2]["hc_end_evals_match"] \
        is False
    two = _write(tmp_path, "e2.jsonl", "".join(json.dumps(e) + "\n" for e in HC_ENDS[:2]))
    assert runner.hc_anytime_records(p, 1000, N, VAR, events=two)[2]["hc_end_evals_match"] \
        is False
    # GOMEA "end" events are not HC "end" events
    gl = _write(tmp_path, "g.jsonl", gomea_log(TWO_DEMES))
    assert "hc_end_events" not in runner.hc_anytime_records(p, 1000, N, VAR, events=gl)[2]


# ---- the key


@pytest.mark.parametrize("change", [
    {"argv": HC_TLS},
    {"argv": ["--opt-algo=hc", "--linear-scaling", "terms"]},
])
def test_wants_hc_anytime(change):
    base = {"binary": "/x/.venv/bin/moses", "argv": HC_LM, "budget": {"evals": 10},
            "seed_spec": None, "hc_anytime": True}
    assert runner._wants_hc_anytime(base) and runner._wants_hc_anytime({**base, **change})


@pytest.mark.parametrize("change", [
    {"argv": ["-a", "gomea", "--linear-scaling", "terms"]},
    {"argv": ["-a", "hc-capped"]},
    {"argv": ["-a", "hc", "-a", "gomea"]},  # the last -a counts, as in argparse
    {"budget": {"time_s": 10}},
    {"seed_spec": {"binary": "x"}},
    {"binary": "/x/build/moses/moses/main/moses-exec"},  # classic MOSES
    {"hc_anytime": False},
])
def test_no_hc_checkpoints_otherwise(change):
    base = {"binary": "/x/.venv/bin/moses", "argv": HC_LM, "budget": {"evals": 10},
            "seed_spec": None, "hc_anytime": True}
    assert not runner._wants_hc_anytime({**base, **change})
    assert not runner._wants_anytime({**base, "anytime": True})  # round 16: -a gomea only


def _manifest(arms: dict, **kw) -> dict:
    return {"name": "m", "datasets_dir": "d", "datasets": ["Con"], "splits": [1, 2],
            "wall_cap_s": 10, "arms": arms, **kw}


def test_expand_carries_the_key_and_refuses_a_wrong_one(tmp_path):
    hc = {"binary": ".venv/bin/moses", "budget": {"evals": 10}, "argv": HC_LM}
    jobs = runner.expand(_manifest({"A": hc, "L": {"linear": "ols"}}, hc_anytime=True),
                         tmp_path, REPO)
    assert [j.get("hc_anytime") for j in jobs if "binary" in j] == [True, True]
    assert all("hc_anytime" not in j for j in jobs if "linear" in j)
    assert all("hc_anytime" not in j
               for j in runner.expand(_manifest({"A": hc}), tmp_path, REPO))
    assert all("hc_anytime" not in j
               for j in runner.expand(_manifest({"A": hc}, hc_anytime=False), tmp_path, REPO))
    with pytest.raises(ValueError, match="hc_anytime must be true or false"):
        runner.expand(_manifest({"A": hc}, hc_anytime="yes"), tmp_path, REPO)
    for bad in ({**hc, "argv": ["-a", "gomea"]}, {**hc, "argv": ["-a", "hc-gomea"]},
                {**hc, "argv": ["--linear-scaling", "1"]},  # no -a: MOSES's default
                {**hc, "binary": "/x/build/moses/moses/main/moses-exec"}):
        with pytest.raises(ValueError, match="hc_anytime needs a pyMoses -a hc arm"):
            runner.expand(_manifest({"A": hc, "B": bad}, hc_anytime=True), tmp_path, REPO)
    # the round 16 key still refuses an HC arm
    with pytest.raises(ValueError, match="anytime needs -a gomea"):
        runner.expand(_manifest({"A": hc}, anytime=True), tmp_path, REPO)


# ---- _run_one with a fake pyMoses


def test_run_one_with_the_key_installs_the_hook_and_writes_the_fields(tmp_path):
    s = runner._run_one(hc_job(tmp_path / "a", LINES, EVENTS, hc_anytime=True))
    assert s["status"] == "ok" and NEW <= set(s)
    assert "hc_progress.install(" + repr(str(tmp_path / "a" / "run" / "hc_progress.jsonl")) \
        in s["argv"][2]
    assert s["anytime_source"] == "hc-demes" and s["demes"]["count"] == 3
    assert s["anytime"][-1]["best_train_nmse"] == pytest.approx(
        100.0 * (10.0 / s["n_train"]) / s["anytime_check"]["var_scaled_target"])
    chk = s["anytime_check"]
    assert chk["log_evals_match_n_evals"] is True and chk["first_score"] == -10.0
    assert chk["log_best_is_first"] is True and chk["hc_end_evals_match"] is True
    disk = json.loads((tmp_path / "a" / "run" / "summary.json").read_text())
    assert disk["anytime"] == s["anytime"] and disk["demes"] == s["demes"]
    # HC-TLS: no GOMEA log, so no "end" events to compare
    s = runner._run_one(hc_job(tmp_path / "b", LINES, argv=HC_TLS, hc_anytime=True))
    assert s["anytime_source"] == "hc-demes" and "hc_end_events" not in s["anytime_check"]
    assert "--gomea-log" not in s["argv"]


def test_run_one_without_the_key_has_no_hook_and_no_fields(tmp_path):
    s = runner._run_one(hc_job(tmp_path / "a", LINES, EVENTS))
    assert not NEW & set(s) and "hc_progress" not in s["argv"][2]
    assert not (tmp_path / "a" / "run" / "hc_progress.jsonl").exists()
    assert s["const_opt"]["tuning_evals"] == 30  # the HC "end" events are still read


def test_a_line_left_by_an_earlier_attempt_is_removed(tmp_path):
    j = hc_job(tmp_path / "a", "", EVENTS, hc_anytime=True)
    rd = Path(j["run_dir"])
    rd.mkdir(parents=True)
    (rd / "hc_progress.jsonl").write_text(hc_lines((5, -1.0)))  # no summary.json: a rerun
    s = runner._run_one(j)
    assert s["anytime_check"]["expansions"] == 0 and s["demes"] is None


# ---- byte identity against 88a5e02 (git)


def _old_tool(tmp_path: Path) -> Path:
    """tools/gomea/ files of the runner at OLD, in a directory of their own. A missing commit
    fails the test (no skip)."""
    d = tmp_path / f"old-{OLD}"
    d.mkdir()
    for n in ("moses_sr_runner.py", "sr_eval.py", "sr_linear.py"):
        r = subprocess.run(["git", "show", f"{OLD}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:
            pytest.fail(f"git show {OLD}:tools/gomea/{n} failed: {r.stderr.decode().strip()!r}. "
                        f"The byte-identity tests need the reference commit {OLD}; use a clone "
                        f"with the full history.")
        (d / n).write_bytes(r.stdout)
    return d


RUN_ONE = ("import json, sys; sys.path.insert(0, sys.argv[1]); import moses_sr_runner as r; "
           "r._run_one(json.loads(sys.argv[2]))")


def _run_in(tool_dir: Path, job: dict) -> dict:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run([sys.executable, "-c", RUN_ONE, str(tool_dir), json.dumps(job)], env=env,
                   check=True, cwd=job["run_dir"].rsplit("/", 1)[0])
    return json.loads((Path(job["run_dir"]) / "summary.json").read_text())


CASES = {
    "gomea-anytime-trace": lambda w: job(w, gomea_log(TWO_DEMES, traces=TWO_DEMES_TRACE,
                                                      digest=DIGEST),
                                         anytime=True, clip_test_nmse=True, plain_nmse=True),
    "gomea-no-key": lambda w: job(w, gomea_log(TWO_DEMES), clip_test_nmse=True),
    "hc-lm-no-key": lambda w: hc_job(w, LINES, EVENTS, clip_test_nmse=True, plain_nmse=True),
    "hc-tls-no-key": lambda w: hc_job(w, LINES, argv=HC_TLS, clip_test_nmse=True,
                                       plain_nmse=True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_run_without_the_key_is_byte_identical(case, tmp_path):
    old = _old_tool(tmp_path)
    jo, jn = CASES[case](tmp_path / "o"), CASES[case](tmp_path / "n")
    so, sn = _run_in(old, jo), _run_in(TOOLS, jn)
    co, cn = (Path(j["run_dir"]) / "candidates.json" for j in (jo, jn))
    assert cn.read_bytes() == co.read_bytes()

    def text(s):
        return json.dumps({k: v for k, v in s.items() if k not in VOLATILE}, indent=2,
                          sort_keys=True)
    assert text(sn) == text(so)
    # the command line: the same, up to the work directory
    assert [x.replace(str(tmp_path / "n"), "W") for x in sn["argv"]] == [
        x.replace(str(tmp_path / "o"), "W") for x in so["argv"]]
