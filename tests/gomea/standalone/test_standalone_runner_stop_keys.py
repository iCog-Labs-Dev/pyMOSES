"""NOT A PORT: the IT linkage study (BACKLOG 13.9, server round 15). The stop keys of the runner
arm "standalone": `max_generations` (`--max-generations`) and `stall_generations`
(`--stall-generations`).

* ``expand`` carries them and refuses a value that is not a positive integer.
* ``standalone_argv`` passes them to the module, and the real module CLI reads them.
* A runner job with `stall_generations` on a table where the search stops evaluating ends with
  the end reason "stalled".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from standalone_data import write_split

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

SPEC = {"genome": "template", "fos": "lt-mi", "pop": 24, "max_evals": 20000}


def _manifest(tmp_path, arms):
    return {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con"], "splits": [3],
            "wall_cap_s": 300, "mem_limit_mb": 4096, "arms": arms}


def test_expand_carries_the_stop_keys_and_the_module_reads_them(tmp_path):
    from pymoses.gomea.standalone.run import parse_args

    spec = {**SPEC, "max_generations": 300, "stall_generations": 10}
    [job] = runner.expand(_manifest(tmp_path, {"A": {"standalone": spec}}), tmp_path, tmp_path)
    assert job["standalone"] == spec
    argv = runner.standalone_argv(job, tmp_path / "run")
    assert " ".join(argv).endswith("--max-generations 300 --stall-generations 10")
    a = parse_args(argv[3:])
    assert (a.max_generations, a.stall_generations) == (300, 10)
    plain = runner.standalone_argv({**job, "standalone": dict(SPEC)}, tmp_path / "run")
    assert not any(t in plain for t in ("--max-generations", "--stall-generations"))
    assert parse_args(plain[3:]).stall_generations is None


@pytest.mark.parametrize("key", ["max_generations", "stall_generations"])
@pytest.mark.parametrize("value", [0, -1, "10", True, 2.5])
def test_expand_refuses_a_stop_key_that_is_not_a_positive_integer(key, value, tmp_path):
    arm = {"standalone": {**SPEC, key: value}}
    with pytest.raises(ValueError, match=f"standalone {key} must be a positive integer"):
        runner.expand(_manifest(tmp_path, {"A": arm}), tmp_path, tmp_path)


def test_a_runner_job_with_the_stall_key_ends_stalled(tmp_path):
    """The test table with template LT, N 24 and seed 3 (the split number) stops evaluating
    after about 12 generations (test_standalone_stall.py)."""
    split = write_split(tmp_path / "Con" / "splits" / "s3")
    arm = {"standalone": {**SPEC, "stall_generations": 3, "max_generations": 300}}
    [job] = runner.expand(_manifest(tmp_path, {"T-LT": arm}), tmp_path / "out", tmp_path)
    assert job["split_dir"] == str(split)
    s = runner._run_one(job)
    assert s["status"] == "ok" and s["termination_reason"] == "stalled"
    disk = json.loads((Path(job["run_dir"]) / "summary.json").read_text())
    assert disk["standalone"]["stall_generations"] == 3
    assert disk["standalone"]["max_generations"] == 300
    assert disk["standalone"]["stall_count"] == 3 and disk["crosscheck"]["passed"] is True
