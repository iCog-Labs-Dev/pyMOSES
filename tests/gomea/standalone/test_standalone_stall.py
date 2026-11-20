"""NOT A PORT: the IT linkage study (BACKLOG 13.9, server round 15). The opt-in
``--stall-generations K`` option of the stand-alone run.

* ``RegressionEvaluator.check``: with K, "stalled" when K generations in a row made no new
  evaluation (one call at the start of each generation); a new evaluation resets the count; the
  budget check comes first; without K, ``check`` never says "stalled".
* A run that stops evaluating (one phenotype, different genotypes) ends with the end reason
  ``stalled`` after K such generations, and its first generations are those of the run without
  the option.
* Byte identity: without the option, ``summary.json``, ``events.jsonl`` and ``candidates.json``
  are those of the code at d4c257f (the commit before this change, from git), apart from times.
* A K that the run does not reach gives the outputs of the run without the option, apart from
  the new keys.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pymoses.gomea.standalone import run as srun
from pymoses.gomea.standalone.evaluator import RegressionEvaluator

REPO = Path(__file__).resolve().parents[3]
OLD = "d4c257f"  # the commit before the stall rule
NEW_KEYS = ("stall_generations", "stall_count")


def _argv(split, out, genome, fos, *extra, pop=24, evals=20000, seed=3):
    return ["--genome", genome, "--fos", fos, "--train", str(split / "train.csv"),
            "--val", str(split / "val.csv"), "--test", str(split / "test.csv"),
            "--pop", str(pop), "--max-evals", str(evals), "--seed", str(seed),
            "--out", str(out), *extra]


def _strip(o, drop=()):
    """The outputs without the times (they change from run to run) and without `drop`."""
    if isinstance(o, dict):
        return {k: _strip(v, drop) for k, v in o.items()
                if k not in ("ru_maxrss_kb", *drop) and not k.endswith("_s")}
    if isinstance(o, list):
        return [_strip(v, drop) for v in o]
    return o


def _outputs(out: Path, drop=()):
    s = json.loads((out / "summary.json").read_text())
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    cand = json.loads((out / "candidates.json").read_text())
    return _strip(s, drop), _strip(events, drop), cand


def _gens(out: Path) -> list[int]:
    return [e["n_evals"] for e in (json.loads(x) for x in (out / "events.jsonl").read_text()
                                   .splitlines()) if e["type"] == "generation"]


# ---- the evaluator


def _ev(k=None, budget=100):
    return RegressionEvaluator(SimpleNamespace(kind="template"), None, budget,
                               stall_generations=k)


def test_check_says_stalled_after_k_generations_without_a_new_evaluation():
    ev = _ev(k=2)
    ev.n_evals = 10
    assert ev.check() is None  # the start of generation 1: nothing to compare
    assert ev.check() is None and ev.stall_count == 1  # generation 1 made no evaluation
    ev.n_evals = 11
    assert ev.check() is None and ev.stall_count == 0  # a new evaluation resets the count
    assert ev.check() is None and ev.stall_count == 1
    assert ev.check() == "stalled" and ev.stall_count == 2


def test_the_budget_comes_first_and_no_k_means_no_stall_rule():
    ev = _ev(k=1, budget=5)
    ev.n_evals = 5
    assert ev.check() == "global_budget"
    ev = _ev()
    ev.n_evals = 3
    assert [ev.check() for _ in range(50)] == [None] * 50 and ev.stall_count == 0
    with pytest.raises(ValueError, match="stall_generations"):
        _ev(k=0)


def test_the_cli_refuses_a_k_below_1(split_dir, tmp_path):
    with pytest.raises(SystemExit):
        srun.parse_args(_argv(split_dir, tmp_path / "o", "template", "rt",
                              "--stall-generations", "0"))
    a = srun.parse_args(_argv(split_dir, tmp_path / "o", "template", "rt"))
    assert a.stall_generations is None


# ---- runs


def test_a_run_that_stops_evaluating_ends_stalled(split_dir, tmp_path):
    """Template LT on the test table (N 24, seed 3) makes no new evaluation from about
    generation 12 on, and without a rule it converges only much later."""
    free, k3 = tmp_path / "free", tmp_path / "k3"
    srun.run(srun.parse_args(_argv(split_dir, free, "template", "lt-mi")))
    s = srun.run(srun.parse_args(_argv(split_dir, k3, "template", "lt-mi",
                                       "--stall-generations", "3")))
    sf = json.loads((free / "summary.json").read_text())
    assert s["termination_reason"] == "stalled" and sf["termination_reason"] == "converged"
    g = s["generations"]
    assert g < sf["generations"]
    evals = _gens(k3)
    assert len(evals) == g and evals[-4:] == [evals[-4]] * 4  # 3 generations without a new one
    assert len(set(evals[-5:])) == 2  # the generation before them made one
    assert s["standalone"]["stall_generations"] == 3 and s["standalone"]["stall_count"] == 3
    assert s["n_evals"] == evals[-1] < 20000
    # the first g generations are those of the run without the option
    _, ef, _ = _outputs(free)
    _, e3, _ = _outputs(k3, drop=NEW_KEYS)
    assert e3[: g + 1] == ef[: g + 1]
    start = json.loads((k3 / "events.jsonl").read_text().splitlines()[0])
    assert start["params"]["stall_generations"] == 3


def test_the_generation_limit_comes_before_the_stall_rule(split_dir, tmp_path):
    out = tmp_path / "o"
    s = srun.run(srun.parse_args(_argv(split_dir, out, "template", "lt-mi",
                                       "--stall-generations", "3", "--max-generations", "5")))
    assert s["termination_reason"] == "max_generations" and s["generations"] == 5


@pytest.mark.parametrize("genome, fos", [("template", "lt-mi"), ("it", "lt-b")])
def test_a_k_that_the_run_does_not_reach_changes_only_the_new_keys(genome, fos, split_dir,
                                                                    tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    srun.run(srun.parse_args(_argv(split_dir, a, genome, fos, evals=3000)))
    s = srun.run(srun.parse_args(_argv(split_dir, b, genome, fos, "--stall-generations",
                                       "100000", evals=3000)))
    assert s["standalone"]["stall_generations"] == 100000
    sa, ea, ca = _outputs(a)
    sb, eb, cb = _outputs(b, drop=NEW_KEYS)
    assert (sb, eb, cb) == (sa, ea, ca)


# ---- byte identity against d4c257f (git)


def _old_package(tmp_path: Path) -> Path:
    """`pymoses/` and `tools/gomea/` (the harness loads sr_eval and sr_linear from there) at
    OLD, unpacked into a directory of its own."""
    d = tmp_path / f"old-{OLD}"
    d.mkdir()
    r = subprocess.run(["git", "archive", "--format=tar", OLD, "pymoses", "tools/gomea"],
                       cwd=REPO, capture_output=True)
    if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
        pytest.fail(f"git archive {OLD} failed: {r.stderr.decode().strip()!r}. The "
                    f"byte-identity test needs the reference commit {OLD}.")
    tar = tmp_path / "old.tar"
    tar.write_bytes(r.stdout)
    with tarfile.open(tar) as tf:
        tf.extractall(d, filter="data")
    return d


@pytest.mark.parametrize("genome, fos", [("template", "lt-mi"), ("template", "np"),
                                         ("it", "lt-b"), ("it", "rt")])
def test_without_the_option_the_outputs_are_those_of_d4c257f(genome, fos, split_dir, tmp_path):
    old = _old_package(tmp_path)
    outs = {}
    for tag, path in (("old", old), ("new", REPO)):
        outs[tag] = tmp_path / tag
        argv = _argv(split_dir, outs[tag], genome, fos, evals=5000)
        subprocess.run([sys.executable, "-m", "pymoses.gomea.standalone", *argv], check=True,
                       cwd=tmp_path, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(path), "PYTHONHASHSEED": "0"})
    assert _outputs(outs["new"]) == _outputs(outs["old"])
    assert (outs["new"] / "candidates.json").read_bytes() == (
        outs["old"] / "candidates.json").read_bytes()
    s = json.loads((outs["new"] / "summary.json").read_text())
    assert not set(NEW_KEYS) & set(s["standalone"])
