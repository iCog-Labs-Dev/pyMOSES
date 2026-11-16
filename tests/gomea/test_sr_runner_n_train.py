"""tools/gomea/moses_sr_runner.py: the placeholder `{K*n_train}` in an arm's argv (server round
12, `SERVER-ROUND12-PREREGISTRATION.md`, T2.4: BASE-DIV passes `--diversity-pressure P` with
P = 6 x the training rows of the run's dataset).

The runner puts the integer K * n_train in the place of each `{K*n_train}` (`resolve_argv`,
called by `moses_argv`). The placeholder is opt-in: an argv without it gives the same command
line as before, element for element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import moses_sr_runner as sr  # noqa: E402

PY_MOSES = "/x/.venv/bin/moses"
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--linear-scaling", "1", "--const-opt", "lm",
        "--const-opt-guard", "val"]


def _job(argv, **kw):
    return {"binary": PY_MOSES, "seed": 41, "budget": {"evals": 20000}, "argv": argv, **kw}


def _fixed(tmp_path, n_train):
    """The command line before the arm's argv (as in test_sr_runner_events.py)."""
    return [PY_MOSES, "-i", str(tmp_path / "train_scaled.csv"), "-u", "y", "-r", "41",
            "-n", "sin", "-n", "exp", "-n", "log", "-v", str(60 * n_train), "-c", "10", "-V1",
            "-l", "WARN", "-f", str(tmp_path / "moses.log"), "-m", "20000"]


def test_an_argv_without_the_placeholder_does_not_change(tmp_path):
    argv = BASE + ["--const-opt-reserve", "500", "--diversity-pressure", "3000"]
    out = sr.moses_argv(_job(argv), tmp_path, 753)
    assert out == _fixed(tmp_path, 753) + argv + ["--gomea-log", str(tmp_path / "events.jsonl")]
    # the same objects, not copies: nothing is rewritten
    body = out[len(_fixed(tmp_path, 753)):-2]
    assert all(a is b for a, b in zip(body, argv, strict=True))


@pytest.mark.parametrize("n_train, want", [(753, "4518"), (154, "924"), (2501, "15006")])
def test_the_placeholder_is_6_times_the_training_rows(tmp_path, n_train, want):
    argv = BASE + ["--diversity-pressure", "{6*n_train}", "--diversity-autoscale", "0"]
    out = sr.moses_argv(_job(argv), tmp_path, n_train)
    i = out.index("--diversity-pressure")
    assert out[i + 1] == want and int(want) == 6 * n_train
    # every other element is as before
    assert out[:i + 1] == _fixed(tmp_path, n_train) + BASE + ["--diversity-pressure"]
    assert out[i + 2:] == ["--diversity-autoscale", "0", "--gomea-log",
                           str(tmp_path / "events.jsonl")]


def test_resolve_argv():
    assert sr.resolve_argv(["{6*n_train}"], 100) == ["600"]
    assert sr.resolve_argv(["{1*n_train}", "{12*n_train}"], 7) == ["7", "84"]
    assert sr.resolve_argv(["--x={6*n_train}"], 10) == ["--x=60"]
    assert sr.resolve_argv(["{2*n_train}:{3*n_train}"], 10) == ["20:30"]
    assert sr.resolve_argv([], 10) == []
    # not the placeholder: unchanged
    for tok in ("{n_train}", "{0*n_train}", "{6 * n_train}", "{6*n_test}", "6*n_train",
                "{-6*n_train}", "{06*n_train}"):
        assert sr.resolve_argv([tok], 10) == [tok], tok
    assert sr.resolve_argv([5, 2.5], 10) == [5, 2.5]


def test_expand_keeps_the_placeholder_and_the_runner_reads_the_arm_as_before(tmp_path):
    """The placeholder is replaced per run, in `moses_argv`; the job keeps the arm's argv, so
    `_const_opt`, `_const_opt_guard`, `_linear_scaling` and `events_argv` read it as before."""
    argv = BASE + ["--diversity-pressure", "{6*n_train}", "--diversity-autoscale", "0"]
    manifest = {"name": "t", "datasets_dir": "d", "datasets": ["Air"], "splits": [41],
                "wall_cap_s": 10,
                "arms": {"DIV": {"binary": PY_MOSES, "budget": {"evals": 20000}, "argv": argv}}}
    (job,) = sr.expand(manifest, tmp_path, tmp_path)
    assert job["argv"] == argv
    assert sr._const_opt(job["argv"]) == "lm" and sr._const_opt_guard(job["argv"]) == "val"
    assert sr._linear_scaling(job["argv"]) == "1"
    assert sr.events_argv(job, tmp_path) == ["--gomea-log", str(tmp_path / "events.jsonl")]
