"""Server round 16 (``SERVER-ROUND16-PREREGISTRATION.md``): the runner additions of
``tools/gomea/moses_sr_runner.py``.

* ``rv_summary``: the RV totals from the "rv" records of the GOMEA log; ``None`` without them.
* ``deme_summary``: the number of demes and the run evaluation at which each deme starts.
* ``anytime_records``: the best-so-far training NMSE at 1, 2, 5, 10, 20, 50 and 100 % of the
  budget, in the record format of the stand-alone harness, and the check of the conversion
  from MOSES's raw score to the training NMSE.
* ``_run_one`` writes ``rv`` when the log has "rv" records, and ``demes``, ``anytime`` and
  ``anytime_check`` only with the manifest key ``"anytime": true``.
* Byte identity against 2d407b2 (the commit before this change, from git): a GOMEA arm with a
  log and without the key gives the same candidates and summary; with an RV log the summary
  only gains ``rv``.

The real log ``data/round16_rv_spike2/events.jsonl`` is the GOMEA log of the T3.2 spike 2 run
``rvg`` on Con, old split 1, 20,000 evaluations (the path of the log is replaced by its name).
It has no data rows.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.moses.scoring.bscores import contin_bscore

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import linkage_report  # noqa: E402
import moses_sr_runner as runner  # noqa: E402
import sr_eval  # noqa: E402
from round16_data import PROGRAMS, TLS, TRAIN, TWO_DEMES, gomea_log, job  # noqa: E402

OLD = "2d407b2"  # the commit before the round 16 runner changes
SPIKE = REPO / "tests" / "gomea" / "data" / "round16_rv_spike2" / "events.jsonl"
VOLATILE = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "events.jsonl"
    p.write_text(text)
    return p


# ---- rv_summary


def test_rv_summary_is_none_without_an_rv_record(tmp_path):
    assert runner.rv_summary(tmp_path / "none.jsonl", 1000) is None
    plain = [(g, e, None) for g, e, _ in TWO_DEMES]
    assert runner.rv_summary(_write(tmp_path, gomea_log(plain)), 1000) is None


def test_rv_summary_sums_the_demes(tmp_path):
    rv = runner.rv_summary(_write(tmp_path, gomea_log(TWO_DEMES)), 1000)
    assert rv["mode"] == "rvg" and rv["budget"] == "balance" and rv["demes"] == 2
    assert rv["rv_evals"] == 100 and rv["rv_share"] == 0.1
    assert rv["generations"] == 3 and rv["improving"] == 30 and rv["kept_p"] == 7
    assert rv["rejected"] == 63 and rv["ams1_evals"] == 9 and rv["ams2_evals"] == 2
    assert rv["ams1_improving"] == 2 and rv["fi_triggers"] == 1 and rv["fi_evals"] == 4
    assert rv["skip_struct"] == 8 and rv["skip_cache"] == 5 and rv["skip_dup"] == 2
    assert rv["skipped_p_rv"] == 1 and rv["fi_improved"] == 0  # a missing counter counts 0
    # AMS rounds: the RV generations (ran) with AMS evaluations
    assert rv["ams1_rounds"] == 2 and rv["ams2_rounds"] == 1
    assert rv["converged_by"] == {"structure": 1} and rv["nis_max_seen"] == 5
    assert rv["eval_s"] == 0.75 and rv["step_s"] == 1.25 and rv["analyse_s"] == 0.0
    assert set(runner.RV_COUNTERS) <= set(rv)
    assert runner.rv_summary(tmp_path / "events.jsonl", None)["rv_share"] is None


def test_rv_summary_of_the_spikegomea_log():
    rv = runner.rv_summary(SPIKE, 20000)
    assert rv["demes"] == 2 and rv["rv_evals"] == 2491 + 1118
    assert rv["rv_share"] == pytest.approx(3609 / 20000)
    assert rv["generations"] == 10 and rv["improving"] == 735 + 372
    assert rv["kept_p"] == 91 + 38 and rv["rejected"] == 1665 + 708
    assert rv["ams1_evals"] == 420 and rv["ams2_evals"] == 71
    assert rv["fi_triggers"] == 0 and rv["skip_struct"] == 2622 and rv["skip_dup"] == 233 + 259
    assert rv["converged_by"] == {"structure": 1} and rv["nis_max_seen"] == 4


def test_a_cut_last_line_is_left_out(tmp_path):
    text = gomea_log(TWO_DEMES) + '{"type": "end", "rv": {"rv_ev'
    assert runner.rv_summary(_write(tmp_path, text), 1000)["rv_evals"] == 100


# ---- deme_summary


def test_deme_summary_of_two_demes(tmp_path):
    d = runner.deme_summary(_write(tmp_path, gomea_log(TWO_DEMES)))
    assert d == {"count": 2, "start_evals": [0, 300], "requests": [300, 700],
                 "second_start_evals": 300, "evals_total": 1000}


def test_a_deme_without_evaluations_is_not_counted(tmp_path):
    demes = TWO_DEMES[:1] + [([], (0, -590.0), None)] + TWO_DEMES[1:]
    d = runner.deme_summary(_write(tmp_path, gomea_log(demes)))
    assert d["count"] == 2 and d["start_evals"] == [0, 300] and d["evals_total"] == 1000
    one = runner.deme_summary(_write(tmp_path, gomea_log(TWO_DEMES[:1])))
    assert one["count"] == 1 and one["second_start_evals"] is None
    assert runner.deme_summary(tmp_path / "none.jsonl") is None


def test_a_deme_without_an_end_event_counts_its_last_generation(tmp_path):
    text = gomea_log(TWO_DEMES[:1]) + json.dumps(
        {"type": "invocation_start"}) + "\n" + json.dumps(
        {"type": "generation", "best_raw": -100.0, "ledger": {"requests": 50}}) + "\n"
    d = runner.deme_summary(_write(tmp_path, text))
    assert d["count"] == 2 and d["start_evals"] == [0, 300] and d["evals_total"] == 350


def test_the_demes_of_the_spike_log_sum_to_the_budget():
    d = runner.deme_summary(SPIKE)
    assert d == {"count": 2, "start_evals": [0, 12789], "requests": [12789, 7211],
                 "second_start_evals": 12789, "evals_total": 20000}


# ---- the conversion and anytime_records


@pytest.mark.parametrize("prog", PROGRAMS)
def test_the_conversion_gives_the_sr_eval_training_nmse(prog, tmp_path):
    """The scorer of `--linear-scaling terms` on the scaled training table (as the runner
    writes it): 100 * (-raw / n) / var(y') is the training NMSE of sr_eval on the original
    table (float32 scores)."""
    (tmp_path / "train.csv").write_text(
        "\n".join(TRAIN.read_text().splitlines()[:41]) + "\n")
    n, _, _ = runner._scale_train(tmp_path / "train.csv", tmp_path / "train_scaled.csv")
    scorer = contin_bscore(loadTable(str(tmp_path / "train_scaled.csv"), "y"))
    scorer.set_linear_scaling_terms(True)
    raw = scorer.sum_bscore(scorer(str_to_combo_tree(prog)))
    var = runner._scaled_target_var(tmp_path / "train_scaled.csv")
    assert var == pytest.approx(100.0, rel=1e-12)
    train = sr_eval.read_table(str(tmp_path / "train.csv"))
    ref = sr_eval.evaluate(prog, train, terms=True)["train"]["nmse_scaled"]
    assert runner.raw_to_train_nmse(raw, n, var) == pytest.approx(ref, rel=1e-5)


def test_the_conversion_of_a_score_that_is_not_finite():
    assert runner.raw_to_train_nmse(None, 10, 100.0) is None
    assert runner.raw_to_train_nmse(-math.inf, 10, 100.0) is None
    assert runner.raw_to_train_nmse(-250.0, 10, 100.0) == 25.0
    assert runner.raw_to_train_nmse(-250.0, 10, None) is None


def test_anytime_records_take_the_best_raw_score_up_to_each_checkpoint(tmp_path):
    ev = _write(tmp_path, gomea_log(TWO_DEMES))
    recs, check = runner.anytime_records(ev, 1000, 10, 100.0, first_score=-500.0,
                                         first_train_nmse=50.0)
    assert [r["evals"] for r in recs] == [10, 20, 50, 100, 200, 500, 1000]
    assert [r["frac"] for r in recs] == list(runner.ANYTIME_FRACTIONS)
    assert all(r["reached"] for r in recs)
    # 10, 20: no record yet; 50 and 100: deme 1 at 30; 200: at 120; 500: the end of deme 1
    # (300, -590) and deme 2 at 400 (-700, worse); 1000: deme 2 at 1000 (-500)
    assert [r["best_train_nmse"] for r in recs] == [None, None, 80.0, 80.0, 60.0, 59.0, 50.0]
    assert [r["log_evals"] for r in recs] == [None, None, 30, 30, 120, 400, 1000]
    assert check["first_score_train_nmse"] == 50.0 and check["search_rel_diff"] == 0.0
    assert check["log_best_raw"] == -500.0 and check["log_best_is_first"] is True
    assert check["log_best_rel_diff"] == 0.0 and check["log_evals_total"] == 1000
    assert check["rule"] == "100 * (-raw / n_train) / var(y_scaled)"


def test_a_run_that_ends_before_the_budget(tmp_path):
    ev = _write(tmp_path, gomea_log(TWO_DEMES[:1]))  # 300 evaluations of a budget of 1000
    recs, check = runner.anytime_records(ev, 1000, 10, 100.0, first_score=-600.0,
                                         first_train_nmse=60.3)
    assert [r["reached"] for r in recs] == [True] * 5 + [False, False]
    assert recs[-1]["best_train_nmse"] == 59.0 and recs[-1]["log_evals"] == 300
    assert recs[-2]["best_train_nmse"] == 59.0
    assert check["log_best_is_first"] is False  # the log has a better raw score
    assert check["search_rel_diff"] == pytest.approx(0.3 / 60.3)
    assert check["log_best_rel_diff"] == pytest.approx(1.3 / 60.3)


def test_anytime_area_reads_the_records(tmp_path):
    ev = _write(tmp_path, gomea_log(TWO_DEMES))
    recs, _ = runner.anytime_records(ev, 1000, 10, 100.0)
    assert linkage_report.anytime_area({"anytime": recs}) is None  # null values at 1 and 2 %
    # a budget of 10,000 for a run of 1,000 evaluations: a value at every checkpoint
    full, _ = runner.anytime_records(ev, 10000, 10, 100.0)
    assert [r["reached"] for r in full] == [True] * 4 + [False] * 3
    assert full[0]["evals"] == 100 and full[0]["best_train_nmse"] == 80.0
    a = linkage_report.anytime_area({"anytime": full})
    assert a is not None and 50.0 < a < 80.0


def test_anytime_of_the_spikegomea_log():
    recs, check = runner.anytime_records(SPIKE, 20000, 516, 100.0,
                                         first_score=-15581.8642578125)
    assert [r["evals"] for r in recs] == [200, 400, 1000, 2000, 4000, 10000, 20000]
    assert [r["log_evals"] for r in recs] == [None, None, None, 1768, 3702, 9754, 20000]
    assert recs[3]["best_train_nmse"] == pytest.approx(20170.037109375 / 516)
    assert recs[-1]["best_train_nmse"] == pytest.approx(15581.8642578125 / 516)
    assert check["log_best_is_first"] is True and check["log_evals_total"] == 20000
    assert check["search_rel_diff"] is None  # no sr_eval value given


# ---- _run_one with a fake pyMoses


def test_run_one_writes_rv_and_with_the_key_the_checkpoints(tmp_path):
    s = runner._run_one(job(tmp_path / "a", gomea_log(TWO_DEMES), anytime=True))
    assert s["status"] == "ok" and "--gomea-log" in s["argv"]
    assert s["rv"]["rv_evals"] == 100 and s["rv"]["rv_share"] == 0.1
    assert s["demes"]["count"] == 2 and s["demes"]["second_start_evals"] == 300
    assert len(s["anytime"]) == 7 and s["anytime"][-1]["evals"] == 1000
    chk = s["anytime_check"]
    assert chk["n_train"] == s["n_train"] and chk["var_scaled_target"] == pytest.approx(100.0)
    assert chk["first_score"] == -10.0 and chk["first_train_nmse"] == s["first_train_nmse"]
    assert chk["first_score_train_nmse"] == pytest.approx(100.0 * 10.0 / s["n_train"] / 100.0)
    assert chk["search_rel_diff"] == pytest.approx(
        abs(chk["first_score_train_nmse"] - s["first_train_nmse"]) / s["first_train_nmse"])
    disk = json.loads((tmp_path / "a" / "run" / "summary.json").read_text())
    assert disk["anytime"] == s["anytime"] and disk["demes"] == s["demes"]


def test_run_one_without_the_key_writes_only_rv(tmp_path):
    s = runner._run_one(job(tmp_path / "b", gomea_log(TWO_DEMES)))
    assert "rv" in s and not {"demes", "anytime", "anytime_check"} & set(s)
    plain = [(g, e, None) for g, e, _ in TWO_DEMES]
    s = runner._run_one(job(tmp_path / "c", gomea_log(plain), anytime=True))
    assert "rv" not in s and s["demes"]["count"] == 2


@pytest.mark.parametrize("job", [
    {"argv": ["-a", "hc", "--linear-scaling", "1"]},  # not a GOMEA arm
    {"budget": {"time_s": 10}},  # no evaluation budget
    {"seed_spec": {"binary": "x"}},  # a seed block
    {"binary": "/x/build/moses/moses/main/moses-exec"},  # classic MOSES
    {"anytime": False},
])
def test_no_checkpoints_otherwise(job):
    base = {"binary": "/x/.venv/bin/moses", "argv": TLS, "budget": {"evals": 10},
            "seed_spec": None, "anytime": True}
    assert runner._wants_anytime(base)
    assert not runner._wants_anytime({**base, **job})


def test_expand_carries_the_anytime_key_and_refuses_a_wrong_one(tmp_path):
    arm = {"binary": "/x/.venv/bin/moses", "argv": TLS, "budget": {"evals": 10}}
    m = {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con"], "splits": [1],
         "wall_cap_s": 10, "arms": {"BASE-TLS": arm, "LIN": {"linear": "ols"}}}
    assert all("anytime" not in j for j in runner.expand(m, tmp_path, tmp_path))
    on = runner.expand({**m, "anytime": True}, tmp_path, tmp_path)
    assert [j.get("anytime") for j in on] == [True, None]  # not the linear job
    off = runner.expand({**m, "anytime": False}, tmp_path, tmp_path)
    assert all("anytime" not in j for j in off)
    for v in ("yes", 1, None):
        with pytest.raises(ValueError, match="anytime must be true or false"):
            runner.expand({**m, "anytime": v}, tmp_path, tmp_path)


# ---- byte identity against 2d407b2 (git)


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


@pytest.mark.parametrize("with_rv", [False, True], ids=["log", "rv-log"])
@pytest.mark.parametrize("keys", [{}, {"clip_test_nmse": True, "plain_nmse": True}],
                         ids=["no-keys", "keys"])
def test_a_gomea_arm_without_the_key_is_byte_identical(with_rv, keys, tmp_path):
    demes = TWO_DEMES if with_rv else [(g, e, None) for g, e, _ in TWO_DEMES]
    old = _old_tool(tmp_path)
    jo = job(tmp_path / "o", gomea_log(demes), **keys)
    jn = job(tmp_path / "n", gomea_log(demes), **keys)
    so, sn = _run_in(old, jo), _run_in(TOOLS, jn)
    co, cn = (Path(j["run_dir"]) / "candidates.json" for j in (jo, jn))
    assert cn.read_bytes() == co.read_bytes()
    assert set(sn) - set(so) == ({"rv"} if with_rv else set()) and set(so) <= set(sn)

    def text(s):
        return json.dumps({k: v for k, v in s.items() if k not in VOLATILE and k != "rv"},
                          indent=2, sort_keys=True)
    assert text(sn) == text(so)
