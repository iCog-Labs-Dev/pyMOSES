"""tools/gomea/moses_sr_runner.py: the GOMEA telemetry (`--gomea-log`) it passes.

A pyMoses GOMEA arm gets `--gomea-log RUN_DIR/events.jsonl`, as the Boolean runner
(`benchmarks.gomea.manifest`) does; HC arms, classic C++ MOSES, an arm that already
passes its own `--gomea-log` and an arm with `"events": false` do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import moses_sr_runner as sr  # noqa: E402

PY_MOSES = "/x/.venv/bin/moses"
CPP_MOSES = "/x/build/moses/moses/main/moses-exec"


def _job(argv, binary=PY_MOSES, **kw):
    return {"binary": binary, "seed": 3, "budget": {"evals": 1000}, "argv": argv, **kw}


@pytest.mark.parametrize(
    "argv",
    [["-a", "gomea"], ["-a", "hc-gomea"], ["--opt-algo", "gomea"], ["--opt-algo=gomea"],
     ["-agomea"]],
)  # fmt: skip
def test_gomea_arms_get_the_events_log(argv, tmp_path):
    out = sr.moses_argv(_job(argv), tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    # everything before it is unchanged
    assert out[:-2] == sr.moses_argv(_job(argv, events=False), tmp_path, 100)


def test_the_rest_of_the_command_line_is_unchanged(tmp_path):
    argv = ["-a", "gomea", "--gomea-max-evals-per-deme", "250"]
    out = sr.moses_argv(_job(argv), tmp_path, 100)
    assert out == [
        PY_MOSES, "-i", str(tmp_path / "train_scaled.csv"), "-u", "y", "-r", "3",
        "-n", "sin", "-n", "exp", "-n", "log", "-v", "6000", "-c", "10", "-V1", "-l", "WARN",
        "-f", str(tmp_path / "moses.log"), "-m", "1000", *argv,
        "--gomea-log", str(tmp_path / "events.jsonl"),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "job",
    [
        _job(["-a", "hc"]),
        _job(["-a", "hc-capped", "--gomea-max-evals-per-deme", "250"]),
        _job(["--gomea-fos", "rt"]),  # no -a gomea
        _job(["-a", "gomea"], binary=CPP_MOSES),
        _job(["-a", "gomea"], events=False),
        _job(["-a", "gomea", "--gomea-log", "/elsewhere.jsonl"]),
    ],
)
def test_no_events_log_otherwise(job, tmp_path):
    assert sr.events_argv(job, tmp_path) == []
    assert sr.moses_argv(job, tmp_path, 100).count("--gomea-log") == job["argv"].count(
        "--gomea-log"
    )


def test_expand_carries_an_arm_events_switch(tmp_path):
    manifest = {
        "name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con"], "splits": [1],
        "wall_cap_s": 10,
        "arms": {
            "G": {"binary": PY_MOSES, "argv": ["-a", "gomea"], "budget": {"evals": 10}},
            "Q": {"binary": PY_MOSES, "argv": ["-a", "gomea"], "budget": {"evals": 10},
                  "events": False},
        },
    }  # fmt: skip
    jobs = {j["arm"]: j for j in sr.expand(manifest, tmp_path, tmp_path)}
    assert "events" not in jobs["G"] and jobs["Q"]["events"] is False
