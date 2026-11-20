"""NOT A PORT: the IT linkage study (BACKLOG 13.9). The opt-in ``--max-generations G`` option of
the stand-alone run (gate 1 of plan section 8.3 uses G = 20, as GP-GOMEA's ``--generations 20``).

* With the option, the run stops after G GOM generations after the start population, with the
  end reason ``max_generations``, and the first generations are the same as without the option.
* Without the option, the engine gets the limit ``BIG``, and no new key is in the outputs.
* A limit that the run does not reach gives the same outputs as no limit, apart from the new
  keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymoses.gomea.engine import EngineParams
from pymoses.gomea.standalone import run as srun

TIMES = ("wall_s", "cpu_s", "ru_maxrss_kb")


def _argv(split, out, genome, fos, *extra, pop=24, evals=2000, seed=3):
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


def _outputs(out: Path):
    s = json.loads((out / "summary.json").read_text())
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    return _strip(s), _strip(events)


def _capture_params(monkeypatch) -> list:
    seen = []

    class Spy(EngineParams):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

    monkeypatch.setattr(srun, "EngineParams", Spy)
    return seen


@pytest.mark.parametrize("genome, fos", [("template", "lt-b"), ("template", "rt"),
                                         ("it", "lt-b")])
def test_the_run_stops_after_g_generations(genome, fos, split_dir, tmp_path):
    G = 2
    out = tmp_path / "g"
    s = srun.run(srun.parse_args(_argv(split_dir, out, genome, fos, "--max-generations",
                                       str(G), evals=10**6)))
    assert s["status"] == "ok" and s["crosscheck"]["passed"]
    assert s["termination_reason"] == "max_generations"
    assert s["generations"] == G and s["standalone"]["max_generations"] == G
    _, events = _outputs(out)
    assert events[0]["params"]["max_generations"] == G
    gens = [e for e in events if e["type"] == "generation"]
    assert [e["generation"] for e in gens] == list(range(G))
    assert events[-1]["generations"] == G
    # the start population is not a generation: G generations of GOM come after it
    assert s["n_evals"] > s["standalone"]["start"]["init_evals"]

    # the same seed with no limit: the first G generations are the same
    free = tmp_path / "free"
    srun.run(srun.parse_args(_argv(split_dir, free, genome, fos, evals=4000)))
    _, free_events = _outputs(free)
    free_gens = [e for e in free_events if e["type"] == "generation"]
    assert len(free_gens) > G
    assert free_gens[:G] == gens


def test_without_the_option_the_limit_is_big_and_no_key_is_added(split_dir, tmp_path,
                                                                  monkeypatch):
    seen = _capture_params(monkeypatch)
    out = tmp_path / "o"
    args = srun.parse_args(_argv(split_dir, out, "template", "lt-b"))
    assert args.max_generations is None
    s = srun.run(args)
    assert [p.max_generations for p in seen] == [srun.BIG]
    assert s["termination_reason"] != "max_generations"
    assert "max_generations" not in s["standalone"]
    _, events = _outputs(out)
    assert "max_generations" not in events[0]["params"]


def test_the_option_sets_the_engine_limit(split_dir, tmp_path, monkeypatch):
    seen = _capture_params(monkeypatch)
    srun.run(srun.parse_args(_argv(split_dir, tmp_path / "o", "it", "rt",
                                   "--max-generations", "1")))
    assert [(p.max_generations, p.stall_generations) for p in seen] == [(1, srun.BIG)]


def test_a_limit_not_reached_gives_the_same_outputs_as_no_limit(split_dir, tmp_path):
    a, b = tmp_path / "none", tmp_path / "high"
    sa = srun.run(srun.parse_args(_argv(split_dir, a, "template", "lt-b")))
    high = sa["generations"] + 5
    sb = srun.run(srun.parse_args(_argv(split_dir, b, "template", "lt-b",
                                        "--max-generations", str(high))))
    assert sb["termination_reason"] == sa["termination_reason"] != "max_generations"
    oa, ob = _outputs(a), _outputs(b)
    assert ob[0]["standalone"].pop("max_generations") == high
    assert ob[1][0]["params"].pop("max_generations") == high
    assert oa == ob


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_a_limit_below_one_is_refused(bad, split_dir, tmp_path, capsys):
    with pytest.raises(SystemExit):
        srun.parse_args(_argv(split_dir, tmp_path, "it", "uni", "--max-generations", bad))
    assert "--max-generations must be at least 1" in capsys.readouterr().err
