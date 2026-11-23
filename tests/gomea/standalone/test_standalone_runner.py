"""The IT linkage study (BACKLOG 13.9, plan sections 6.4 and 8.2): the runner arm key
``"standalone"`` of ``tools/gomea/moses_sr_runner.py``.

* ``expand`` gives a stand-alone job (no binary, no argv, no budget) and refuses a wrong block
  or a stand-alone arm with another key.
* ``_run_one`` runs ``python -m pymoses.gomea.standalone`` and keeps its summary; a run with no
  summary (a timeout, a crash) gets a summary of its own.
* Identity: every committed regression manifest expands to the jobs of the runner at 5fdd335
  (the commit before this change, from git), byte for byte.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from standalone_data import write_split

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402

OLD = "5fdd335"  # the commit before the stand-alone arm
SPEC = {"genome": "it", "fos": "lt-b", "pop": 16, "max_evals": 600}


def _manifest(tmp_path, arms, **kw):
    return {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con", "Air"],
            "splits": [1, 2], "wall_cap_s": 120, "mem_limit_mb": 4096, "arms": arms, **kw}


def test_expand_gives_a_standalone_job(tmp_path):
    moses = {"binary": "/x/.venv/bin/moses", "argv": ["-a", "hc"], "budget": {"evals": 10}}
    m = _manifest(tmp_path, {"HC": moses, "LIN": {"linear": "ols"},
                             "IT-LTB": {"standalone": SPEC},
                             "T-NP": {"standalone": {**SPEC, "genome": "template", "fos": "np",
                                                     "template_height": 3}}},
                  clip_test_nmse=True)
    jobs = runner.expand(m, tmp_path / "o", tmp_path)
    assert len(jobs) == 16
    st = [j for j in jobs if "standalone" in j]
    assert len(st) == 8
    for j in st:
        assert set(j) == {"run_id", "task", "arm", "seed", "standalone", "split_dir",
                          "wall_cap_s", "mem_limit_mb", "run_dir", "clip_test_nmse"}
        assert j["run_id"] == f"{j['task']}__{j['arm']}__s{j['seed']}"
        assert j["split_dir"] == str(tmp_path / j["task"] / "splits" / f"s{j['seed']}")
        assert j["run_dir"] == str(tmp_path / "o" / "t" / "runs" / j["run_id"])
        assert j["mem_limit_mb"] == 4096 and j["wall_cap_s"] == 120
        with pytest.raises(KeyError):
            runner.moses_argv(j, tmp_path, 100)  # no MOSES command line
    t = next(j for j in st if j["arm"] == "T-NP")
    assert t["standalone"]["template_height"] == 3


@pytest.mark.parametrize("arm, why", [
    ({"standalone": {**SPEC, "genome": "tree"}}, "genome must be one of"),
    ({"standalone": {**SPEC, "fos": "lt"}}, "fos must be one of"),
    ({"standalone": {k: v for k, v in SPEC.items() if k != "pop"}}, "missing \\['pop'\\]"),
    ({"standalone": {**SPEC, "budget": 3}}, "unknown \\['budget'\\]"),
    ({"standalone": {**SPEC, "pop": 1}}, "pop >= 2"),
    ({"standalone": {**SPEC, "max_evals": 10}}, "max_evals >= pop"),
    ({"standalone": {**SPEC, "pop": "16"}}, "pop must be a positive integer"),
    ({"standalone": {**SPEC, "pop": True}}, "pop must be a positive integer"),
    ({"standalone": {**SPEC, "it_terms": 0}}, "it_terms must be a positive integer"),
    ({"standalone": {**SPEC, "it_powers": [1, 0]}}, "it_powers must be"),
    ({"standalone": {**SPEC, "it_powers": [1, 1]}}, "it_powers must be"),
    ({"standalone": {**SPEC, "it_powers": "1,2"}}, "it_powers must be"),
    ({"standalone": {**SPEC, "column_cache": 1}}, "column_cache must be true or false"),
    ({"standalone": "it"}, "standalone must be a block"),
    ({"standalone": SPEC, "argv": []}, 'only the key "standalone"'),
    ({"standalone": SPEC, "binary": ".venv/bin/moses"}, 'only the key "standalone"'),
    ({"standalone": SPEC, "budget": {"evals": 10}}, 'only the key "standalone"'),
])
def test_expand_refuses_a_wrong_standalone_arm(arm, why, tmp_path):
    with pytest.raises(ValueError, match=why):
        runner.expand(_manifest(tmp_path, {"X": arm}), tmp_path, tmp_path)


def test_the_command_line_of_a_standalone_job(tmp_path):
    job = {"run_id": "T__A__s3", "task": "T", "arm": "A", "seed": 3,
           "standalone": {**SPEC, "it_terms": 4, "it_factors": 3, "it_powers": [-1, 1, 2],
                          "column_cache": False, "it_cache_max_pairs": 50},
           "split_dir": str(tmp_path / "s3")}
    argv = runner.standalone_argv(job, tmp_path / "run")
    assert argv[:3] == [sys.executable, "-m", "pymoses.gomea.standalone"]
    s = " ".join(argv[3:])
    assert "--genome it --fos lt-b" in s and "--pop 16 --max-evals 600 --seed 3" in s
    assert f"--train {tmp_path / 's3' / 'train.csv'}" in s and "--target y" in s
    assert "--it-terms 4 --it-factors 3 --it-powers=-1,1,2 --no-column-cache" in s
    assert "--it-cache-max-pairs 50" in s and "--run-id T__A__s3" in s
    job["standalone"] = {**SPEC, "column_cache": True}
    assert "--no-column-cache" not in runner.standalone_argv(job, tmp_path)


def _job(tmp_path, split, **kw):
    return {"run_id": "T__IT__s1", "task": "T", "arm": "IT", "seed": 1,
            "standalone": dict(SPEC), "split_dir": str(split), "wall_cap_s": 300,
            "mem_limit_mb": None, "run_dir": str(tmp_path / "run"), **kw}


def test_the_runner_runs_the_module_and_keeps_its_summary(tmp_path):
    split = write_split(tmp_path / "s1")
    job = _job(tmp_path, split)
    s = runner._run_one(job)
    assert s["status"] == "ok", s.get("error")
    assert (s["run_id"], s["task"], s["arm"], s["seed"]) == ("T__IT__s1", "T", "IT", 1)
    run_dir = Path(job["run_dir"])
    for f in ("summary.json", "candidates.json", "events.jsonl", "train_scaled.csv",
              "config.json", "stderr.txt"):
        assert (run_dir / f).exists(), f
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["resolved_argv"][1:3] == ["-m", "pymoses.gomea.standalone"]
    assert s["test_nmse_clipped"] is not None and s["terms"]["n_terms"] == s["n_terms"]
    # resumable: an existing summary is kept
    (run_dir / "summary.json").write_text(json.dumps({"status": "kept"}))
    assert runner._run_one(job) == {"status": "kept"}


def test_a_timeout_gets_a_summary(tmp_path):
    split = write_split(tmp_path / "s1")
    s = runner._run_one(_job(tmp_path, split, wall_cap_s=0.01))
    assert s["status"] == "timeout" and s["train_nmse"] is None
    assert s["run_id"] == "T__IT__s1"


def test_a_failed_run_gets_status_error(tmp_path):
    split = write_split(tmp_path / "s1")
    text = (split / "train.csv").read_text().splitlines()
    rows = [",".join(r.split(",")[:-1] + ["1.0"]) for r in text[1:]]  # a constant target
    (split / "train.csv").write_text("\n".join([text[0], *rows]) + "\n")
    s = runner._run_one(_job(tmp_path, split))
    assert s["status"] == "error" and "constant" in s["error"]


# ---- identity against the runner at 5fdd335


def _old_runner(tmp_path: Path) -> Path:
    d = tmp_path / f"old-{OLD}"
    d.mkdir()
    for n in ("moses_sr_runner.py", "sr_eval.py", "sr_linear.py"):
        r = subprocess.run(["git", "show", f"{OLD}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:
            pytest.fail(f"git show {OLD}:tools/gomea/{n} failed: {r.stderr.decode().strip()!r}. "
                        f"The identity test needs the reference commit {OLD}.")
        (d / n).write_bytes(r.stdout)
    return d


EXPAND = ("import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
          "import moses_sr_runner as r; out = {}\n"
          "for f in sys.argv[3:]:\n"
          "    try:\n"
          "        m = json.loads(Path(f).read_text())\n"
          "        out[f] = r.expand(m, Path('/o'), Path(sys.argv[2]))\n"
          "    except Exception as e:\n"
          "        out[f] = 'error: ' + repr(e)\n"
          "print(json.dumps(out, sort_keys=True))")


def _has_standalone_arm(f: Path) -> bool:
    """Server round 15 (1bj) manifests have stand-alone arms; the runner at 5fdd335 cannot
    expand them (the arm is new), so the identity test leaves them out."""
    return any("standalone" in a for a in json.loads(f.read_text())["arms"].values())


def _has_anytime_key(f: Path) -> bool:
    """Server round 16 (1bk) manifests have the key "anytime", which the jobs carry; the runner
    at 5fdd335 does not know it, so the identity test leaves them out."""
    return "anytime" in json.loads(f.read_text())


def test_every_committed_manifest_expands_to_the_jobs_of_5fdd335(tmp_path):
    old = _old_runner(tmp_path)
    every = sorted((REPO / "configs" / "gomea").glob("*/sr*.json"))
    files = [str(f) for f in every if not _has_standalone_arm(f) and not _has_anytime_key(f)]
    assert len(files) > 150
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    got = [json.loads(subprocess.run([sys.executable, "-c", EXPAND, str(d), str(REPO), *files],
                                     env=env, check=True, capture_output=True, text=True).stdout)
           for d in (old, TOOLS)]
    assert got[0] == got[1]
    assert sum(isinstance(v, list) for v in got[1].values()) == len(files)
    # server round 15: the manifests with stand-alone arms expand with the current runner
    for f in every:
        if _has_standalone_arm(f):
            jobs = runner.expand(json.loads(f.read_text()), tmp_path, REPO)
            assert jobs and all("standalone" in j for j in jobs), f
    # server round 16: the manifests with the key "anytime" expand with the current runner, and
    # without the key their jobs are those of the runner at 5fdd335
    for f in every:
        if _has_anytime_key(f):
            m = json.loads(f.read_text())
            jobs = runner.expand(m, tmp_path, REPO)
            assert jobs and all(j.get("anytime") is True for j in jobs), f
            m.pop("anytime")
            assert [{k: v for k, v in j.items() if k != "anytime"} for j in jobs] == \
                runner.expand(m, tmp_path, REPO)
