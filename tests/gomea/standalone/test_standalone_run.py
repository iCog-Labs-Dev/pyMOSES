"""The IT linkage study (BACKLOG 13.9, plan sections 6 and 8.1): the stand-alone run.
Reproducibility (the same seed gives the same ``summary.json`` and ``events.jsonl``, apart from
times, for each of the 10 arms); the cached and uncached IT scorers give bit-identical runs;
the ``python -m pymoses.gomea.standalone`` entry; the scaled training target equals the
runner's ``train_scaled.csv``; a failed cross-check fails the run."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pymoses.gomea.standalone import run as srun

REPO = Path(__file__).resolve().parents[3]
TIMES = ("wall_s", "cpu_s", "ru_maxrss_kb")
ARMS = [(g, f) for g in ("template", "it") for f in ("uni", "rt", "lt-mi", "lt-b", "np")]


def _argv(split, out, genome, fos, *extra, pop=16, evals=700, seed=5):
    return ["--genome", genome, "--fos", fos, "--train", str(split / "train.csv"),
            "--val", str(split / "val.csv"), "--test", str(split / "test.csv"),
            "--pop", str(pop), "--max-evals", str(evals), "--seed", str(seed),
            "--out", str(out), *extra]


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items()
                if k not in TIMES and not k.endswith("_s")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _outputs(out: Path, drop=()):
    s = json.loads((out / "summary.json").read_text())
    for k in drop:
        s.get("standalone", {}).pop(k, None)
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    return _strip(s), _strip(events)


@pytest.mark.parametrize("genome, fos", ARMS)
def test_the_same_seed_gives_the_same_outputs(genome, fos, split_dir, tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        s = srun.run(srun.parse_args(_argv(split_dir, out, genome, fos)))
        assert s["status"] == "ok" and s["crosscheck"]["passed"]
        outs.append(_outputs(out))
    assert outs[0] == outs[1]
    s, events = outs[0]
    assert s["n_evals"] <= 700 and s["standalone"]["fos"] == fos
    assert events[0]["type"] == "start" and events[-1]["type"] == "end"
    gens = [e for e in events if e["type"] == "generation"]
    assert len(gens) == s["generations"] >= 1
    if fos in ("lt-mi", "lt-b"):
        assert len(gens[0]["mi_top_pairs"]) == 20
        assert 0.0 <= gens[0]["mi_top_aligned_share"] <= 1.0
    else:
        assert "mi_top_pairs" not in gens[0]
    if fos == "np":
        assert all(e["fos_jaccard_prev"] == 1.0 for e in gens[1:])
    if genome == "it":
        assert "slot_dispersion_max" in gens[0] and s["n_terms"] == s["terms"]["n_terms"]
    else:
        assert s["final_program_nodes"] == s["size"] == s["nodes"]


def test_a_different_seed_gives_a_different_run(split_dir, tmp_path):
    a = srun.run(srun.parse_args(_argv(split_dir, tmp_path / "a", "it", "lt-b", seed=1)))
    b = srun.run(srun.parse_args(_argv(split_dir, tmp_path / "b", "it", "lt-b", seed=2)))
    assert a["standalone"]["start"] != b["standalone"]["start"] or (
        a["final_program"] != b["final_program"])


@pytest.mark.parametrize("fos", ["lt-b", "rt", "np"])
def test_cached_and_uncached_it_runs_are_bit_identical(fos, split_dir, tmp_path):
    a = tmp_path / "cached"
    b = tmp_path / "uncached"
    srun.run(srun.parse_args(_argv(split_dir, a, "it", fos, evals=1500, pop=24)))
    srun.run(srun.parse_args(_argv(split_dir, b, "it", fos, "--no-column-cache",
                                   evals=1500, pop=24)))
    drop = ("column_cache", "scorer_stats")
    assert _outputs(a, drop) == _outputs(b, drop)
    sa = json.loads((a / "summary.json").read_text())
    sb = json.loads((b / "summary.json").read_text())
    assert sa["standalone"]["scorer_stats"]["column_hits"] > 0
    assert sb["standalone"]["scorer_stats"].get("column_hits", 0) == 0


def test_the_run_ends_at_the_budget(split_dir, tmp_path):
    s = srun.run(srun.parse_args(_argv(split_dir, tmp_path / "o", "template", "lt-b",
                                       evals=500, pop=16)))
    assert s["termination_reason"] == "global_budget" and s["n_evals"] == 500
    assert [a["frac"] for a in s["anytime"]] == list(srun.ANYTIME_FRACTIONS)
    assert all(a["reached"] for a in s["anytime"])
    best = [a["best_train_nmse"] for a in s["anytime"]]
    assert best == sorted(best, reverse=True)


def test_the_module_entry_point(split_dir, tmp_path):
    out = tmp_path / "m"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    p = subprocess.run([sys.executable, "-m", "pymoses.gomea.standalone",
                        *_argv(split_dir, out, "it", "uni", "--arm", "IT-UNI", "--task", "T",
                               "--run-id", "T__IT-UNI__s5")],
                       env=env, cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    s = json.loads((out / "summary.json").read_text())
    assert (s["status"], s["arm"], s["task"], s["run_id"]) == ("ok", "IT-UNI", "T",
                                                              "T__IT-UNI__s5")
    cands = json.loads((out / "candidates.json").read_text())
    assert cands[0]["program"] == s["final_program"]


def test_the_scaled_target_is_the_runners(split_dir, tmp_path):
    sys.path.insert(0, str(REPO / "tools" / "gomea"))
    import moses_sr_runner as runner

    dst = tmp_path / "train_scaled.csv"
    runner._scale_train(split_dir / "train.csv", dst)
    rows = list(csv.reader(dst.open()))
    want = [float(r[rows[0].index("y")]) for r in rows[1:]]
    train = srun.sr_eval().read_table(str(split_dir / "train.csv"))
    got, _, _ = srun.scaled_target(train.y)
    assert got == want


def test_a_failed_cross_check_fails_the_run(split_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(srun, "_own_it", lambda *a: {"train": 1.0, "val": 1.0, "test": 1.0,
                                                     "test_clipped": 1.0})
    out = tmp_path / "x"
    with pytest.raises(srun.CrossCheckError):
        srun.main(_argv(split_dir, out, "it", "lt-b"))
    s = json.loads((out / "summary.json").read_text())
    assert s["status"] == "error" and "cross-check failed" in s["error"]


@pytest.mark.parametrize("extra, why", [
    (["--pop", "1"], "--pop must be at least 2"),
    (["--max-evals", "3"], "--max-evals must be at least --pop"),
])
def test_bad_sizes_are_refused(extra, why, split_dir, tmp_path, capsys):
    with pytest.raises(SystemExit):
        srun.parse_args(_argv(split_dir, tmp_path, "it", "uni") + extra)
    assert why in capsys.readouterr().err
