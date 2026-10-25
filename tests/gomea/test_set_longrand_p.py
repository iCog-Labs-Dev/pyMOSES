"""Tests for tools/gomea/set_longrand_p.py on a fake results tree and a copy of 1af's configs."""

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import set_longrand_p as slp  # noqa: E402


def _configs(tmp_path: Path) -> Path:
    root = tmp_path / "configs" / "gomea"
    for p in ("phase1af", "phase1af-pilot"):
        shutil.copytree(REPO / "configs" / "gomea" / p, root / p)
    return root / "phase1af"


def _events(path: Path, counts: list[tuple[int, int]]) -> None:
    """One invocation_start/end pair per deme, with the run's counts so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n_long, n in counts:
        for t in ("invocation_start", "end"):
            lines.append(json.dumps({"type": t, "long": False, "n_demes": n,
                                     "n_long_demes": n_long}))
        lines.append(json.dumps({"type": "generation"}))
    path.write_text("\n".join(lines) + "\nnot json\n")


def _tree(tmp_path: Path, configs: Path, mux=((1, 10), (3, 20)), con=((2, 8), (2, 4))):
    res = tmp_path / "results"
    pats = slp.pilot_globs(res, configs)
    for prob, runs in (("mux11", mux), ("Con", con)):
        for i, (n_long, n) in enumerate(runs, start=1):
            seed = 3000 + i if prob == "mux11" else i
            f = Path(pats[prob].replace("s*", f"s{seed}"))
            _events(f, [(min(n_long, k), k) for k in range(1, n)] + [(n_long, n)])
    return res


def _p(configs: Path, name: str) -> str:
    argv = json.loads((configs / name).read_text())["arms"]["LONG-RAND"]["argv"]
    return argv[argv.index("--gomea-long-deme-prob") + 1]


def test_sets_median_share_per_problem(tmp_path):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs)
    assert slp.main(["--results", str(res), "--configs", str(configs)]) == 0
    # mux11: shares 0.1 and 0.15 -> 0.125; Con: 0.25 and 0.5 -> 0.375
    assert _p(configs, "bool.json") == "0.125"
    assert _p(configs, "sr.json") == "0.375"
    log = json.loads((res / "phase1af-pilot" / "longrand-p.json").read_text())
    assert log["mux11"]["p"] == 0.125 and len(log["Con"]["runs"]) == 2
    # nothing else in the manifests changes
    orig = json.loads((REPO / "configs/gomea/phase1af/sr.json").read_text())
    new = json.loads((configs / "sr.json").read_text())
    orig["arms"]["LONG-RAND"]["argv"][-1] = "0.375"
    assert new == orig


def test_pilot_paths_match_the_runners(tmp_path):
    """The globs are where the runners put the pilot's runs."""
    from benchmarks.gomea.manifest import expand_runs, load_manifest

    configs = _configs(tmp_path)
    res = tmp_path / "results"
    pats = slp.pilot_globs(res, configs)
    m = load_manifest(configs.parent / "phase1af-pilot" / "bool.json")
    m["output_root"] = str(res)
    runs = expand_runs(m)
    assert len(runs) == 2
    for r in runs:
        assert r["argv"][-1] == str(Path(r["run_dir"]) / "events.jsonl")
        assert Path(r["argv"][-1]).match(pats["mux11"])
    sys.path.insert(0, str(REPO / "tools" / "gomea"))
    import moses_sr_runner

    sr = json.loads((configs.parent / "phase1af-pilot" / "sr.json").read_text())
    jobs = moses_sr_runner.expand(sr, res / "phase1af-pilot", REPO)
    assert len(jobs) == 2
    for j in jobs:
        ev = moses_sr_runner.events_argv(j, Path(j["run_dir"]))
        assert ev[0] == "--gomea-log" and Path(ev[1]).match(pats["Con"])


def test_wrong_run_count_exits_1_and_changes_nothing(tmp_path, capsys):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs, mux=((1, 10), (3, 20), (2, 10)))
    before = {f.name: f.read_text() for f in configs.iterdir()}
    assert slp.main(["--results", str(res), "--configs", str(configs)]) == 1
    assert "mux11: expected 2 pilot runs, found 3" in capsys.readouterr().err
    assert before == {f.name: f.read_text() for f in configs.iterdir()}
    assert not (res / "phase1af-pilot" / "longrand-p.json").exists()


def test_missing_runs_or_telemetry_exit_1(tmp_path, capsys):
    configs = _configs(tmp_path)
    assert slp.main(["--results", str(tmp_path / "none"), "--configs", str(configs)]) == 1
    err = capsys.readouterr().err
    assert "mux11: expected 2 pilot runs, found 0" in err and "Con: expected 2" in err
    res = _tree(tmp_path, configs)
    f = sorted((res / "phase1af-pilot").glob("*/runs/Con__LONG-TOP__s1/events.jsonl"))[0]
    f.write_text(json.dumps({"type": "end"}) + "\n")
    assert slp.main(["--results", str(res), "--configs", str(configs)]) == 1
    assert "no n_demes/n_long_demes telemetry" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs)
    before = {f.name: f.read_text() for f in configs.iterdir()}
    assert slp.main(["--results", str(res), "--configs", str(configs), "--dry-run"]) == 0
    assert before == {f.name: f.read_text() for f in configs.iterdir()}
    assert not (res / "phase1af-pilot" / "longrand-p.json").exists()
