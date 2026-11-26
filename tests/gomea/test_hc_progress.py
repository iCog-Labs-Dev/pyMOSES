"""Server round 17 (``SERVER-ROUND17-PREREGISTRATION.md``, "Contrasts", descriptive): the
expansion hook of ``tools/gomea/hc_progress.py`` on real pyMoses ``-a hc`` runs.

* The hook does not change the search: with and without the hook, the MOSES stdout is the same
  (HC-TLS: ``--linear-scaling terms --const-opt none``; HC-LM: ``--linear-scaling 1 --const-opt
  lm --const-opt-guard val``).
* The hook writes the values of the "Stats:" rows of the MOSES log: with ``-l INFO``, each line
  has the ``n_evals`` and (to the 6 digits of the row) the best score of the row of its
  expansion.
* Through the runner (``moses_sr_runner._run_one``, the manifest key ``"hc_anytime"``): the
  candidates and stdout are the same with and without the key, and the summary only gains
  ``demes``, ``anytime``, ``anytime_source`` and ``anytime_check``. For HC-LM the rule of the
  HC "end" events (``budget - max_evals + ledger.requests``) gives the ``n_evals`` of the hook.

The data is ``data/const_opt_none/train.csv`` (40 rows), with the fixed regression setup of the
runner and 800 evaluations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402
from round16_data import TRAIN, splits  # noqa: E402

REG = ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
       "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", "800"]
HC_TLS = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
          "--const-opt", "none"]
HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
         "--const-opt", "lm", "--const-opt-guard", "val", "--const-opt-val", str(TRAIN)]
ARMS = {"HC-TLS": HC_TLS, "HC-LM": HC_LM}
NEW = {"demes", "anytime", "anytime_source", "anytime_check"}
VOLATILE = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")


def _boot(progress: Path | None) -> str:
    hook = "" if progress is None else (
        f"sys.path.insert(0, {str(TOOLS)!r}); import hc_progress; "
        f"hc_progress.install({str(progress)!r}); ")
    return ("import sys; sys.setrecursionlimit(100000); " + hook
            + "from pymoses.moses.main.moses_exec import moses_exec; "
            "sys.exit(moses_exec(['moses', *sys.argv[1:]]))")


def _run(argv: list[str], work: Path, hook: bool) -> tuple[str, list[dict], str]:
    """(stdout, hook lines, MOSES log) of one run in `work` (a new process)."""
    work.mkdir(parents=True, exist_ok=True)
    progress = work / "hc_progress.jsonl"
    r = subprocess.run([sys.executable, "-c", _boot(progress if hook else None), *argv,
                        "-f", "moses.log"], cwd=work, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONHASHSEED": "0"},
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = ([json.loads(x) for x in progress.read_text().splitlines()]
             if progress.exists() else [])
    return r.stdout, lines, (work / "moses.log").read_text()


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_hook_does_not_change_the_search(arm, tmp_path):
    plain, none, _ = _run(REG + ARMS[arm], tmp_path / "plain", hook=False)
    hooked, lines, _ = _run(REG + ARMS[arm], tmp_path / "hook", hook=True)
    assert none == [] and hooked == plain
    n = [x["n_evals"] for x in lines]
    assert lines and n == sorted(n) and n[-1] == int(re.search(r"^n_evals: (\d+)", plain,
                                                               re.M).group(1))
    assert [x["expansion"] for x in lines] == list(range(1, len(lines) + 1))
    raws = [x["best_raw"] for x in lines if x["best_raw"] is not None]
    assert raws and raws == sorted(raws)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_lines_are_the_stats_rows_of_the_moses_log(arm, tmp_path):
    stdout, lines, log = _run(REG + ARMS[arm] + ["-l", "INFO"], tmp_path, hook=True)
    rows = [r.split("\t") for r in re.findall(r"Stats: (\d+\t.*)$", log, re.M)]
    assert len(rows) == len(lines) >= 1
    for row, x in zip(rows, lines, strict=True):
        assert int(row[0]) == x["expansion"] and int(row[1]) == x["n_evals"]
        assert float(row[4]) == pytest.approx(x["best_raw"], rel=1e-5)  # 6 digits in the row
    # the log level does not change the search either
    assert stdout == _run(REG + ARMS[arm], tmp_path / "warn", hook=False)[0]


def _job(work: Path, arm: str, **kw) -> dict:
    """A runner job with a real pyMoses: `bin/python` is this Python, `bin/moses` only names
    the binary (the runner runs `bin/python -c BOOT`)."""
    b = work / "bin"
    b.mkdir(parents=True, exist_ok=True)
    (b / "moses").write_text("#!/bin/sh\nexit 1\n")
    if not (b / "python").exists():
        (b / "python").symlink_to(sys.executable)
    argv = [x for x in ARMS[arm] if x not in ("--const-opt-val", str(TRAIN))]
    return {"run_id": f"T__{arm}__s1", "task": "T", "arm": arm, "seed": 1,
            "binary": str(b / "moses"), "argv": argv, "budget": {"evals": 800},
            "seed_spec": None, "split_dir": str(splits(work / "splits" / "s1")),
            "wall_cap_s": 600, "mem_limit_mb": None, "run_dir": str(work / "run"), **kw}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_runner_key_adds_only_the_new_fields(arm, tmp_path):
    so = runner._run_one(_job(tmp_path / "o", arm, clip_test_nmse=True, plain_nmse=True))
    sn = runner._run_one(_job(tmp_path / "n", arm, clip_test_nmse=True, plain_nmse=True,
                              hc_anytime=True))
    assert so["status"] == sn["status"] == "ok"
    ro, rn = (Path(tmp_path / x / "run") for x in ("o", "n"))
    assert (rn / "candidates.json").read_bytes() == (ro / "candidates.json").read_bytes()
    assert (rn / "stdout.txt").read_bytes() == (ro / "stdout.txt").read_bytes()
    assert set(sn) - set(so) == NEW and set(so) <= set(sn)
    assert ({k: v for k, v in sn.items() if k not in NEW and k not in VOLATILE}
            == {k: v for k, v in so.items() if k not in VOLATILE})
    assert not (ro / "hc_progress.jsonl").exists() and (rn / "hc_progress.jsonl").exists()
    assert "hc_progress.install" in sn["argv"][2] and "hc_progress" not in so["argv"][2]
    chk = sn["anytime_check"]
    assert sn["anytime_source"] == chk["anytime_source"] == "hc-demes"
    assert chk["log_evals_match_n_evals"] is True and chk["log_evals_total"] == sn["n_evals"]
    assert chk["best_raw_monotone"] is True and sn["demes"]["count"] == chk["expansions"]
    assert sn["anytime"][-1]["best_train_nmse"] == pytest.approx(chk["log_best_train_nmse"])
    # the conversion of the raw score: the first printed program against sr_eval.py
    assert chk["search_rel_diff"] is not None and chk["search_rel_diff"] < 1e-5
    if arm == "HC-LM":
        assert chk["hc_end_events"] == chk["expansions"] and chk["hc_end_evals_match"] is True
    else:
        assert "hc_end_events" not in chk  # -a hc without --const-opt writes no GOMEA log
        assert not (rn / "events.jsonl").exists()
