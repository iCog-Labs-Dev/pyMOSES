"""Tests for tools/gomea/set_knob_cap.py on a fake results tree and a copy of round 2's configs."""

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import set_knob_cap as skc  # noqa: E402

PHASES = ("phase1ak-pilot", "phase1ak", "phase1al", "phase1am")
# per problem and K, each pilot run's per-deme generations
GENS = {
    "mux11": {8: ([5, 6, 4], [5, 7]), 16: ([3, 3, 2], [4, 3]), 32: ([2, 1], [2, 3])},
    "Con": {8: ([2, 2], [1, 3]), 16: ([1, 1], [2]), 32: ([0, 1], [1])},
    "EnH": {8: ([9], [9]), 16: ([6, 5], [4]), 32: ([4, 3], [3, 5])},
}


def _configs(tmp_path: Path) -> Path:
    root = tmp_path / "configs" / "gomea"
    for p in PHASES:
        shutil.copytree(REPO / "configs" / "gomea" / p, root / p)
    return root


def _events(path: Path, gens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for g in gens:
        lines.append(json.dumps({"type": "invocation_start"}))
        lines += [json.dumps({"type": "generation", "generation": i}) for i in range(g)]
        lines.append(json.dumps({"type": "end", "generations": g}))
    path.write_text("\n".join(lines) + "\nnot json\n")


def _tree(tmp_path: Path, configs: Path, gens=GENS, skip=()):
    res = tmp_path / "results"
    runs = skc.pilot_runs(res, configs / "phase1ak-pilot")
    for prob, by_k in runs.items():
        for k, files in by_k.items():
            for f, g in zip(sorted(files), gens[prob][k], strict=True):
                if (prob, k) not in skip:
                    _events(Path(f), g)
    return res


def _ks(configs: Path) -> dict:
    """{(phase, file, arm): K or None} over the target phases."""
    out = {}
    for ph in PHASES[1:]:
        for f in sorted((configs / ph).glob("*.json")):
            m = json.loads(f.read_text())
            for arm, spec in m.get("arms", {}).items():
                out[(ph, f.name, arm)] = skc._k_of(spec["argv"])
    return out


def test_pilot_runs_cover_every_problem_and_k(tmp_path):
    configs = _configs(tmp_path)
    runs = skc.pilot_runs(tmp_path / "r", configs / "phase1ak-pilot")
    assert set(runs) == {"mux11", "Con", "EnH"}
    for by_k in runs.values():
        assert set(by_k) == {8, 16, 32} and all(len(f) == 2 for f in by_k.values())


def test_sets_k_per_problem_into_every_kc_arm(tmp_path):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs)
    before = _ks(configs)
    assert skc.main(["--results", str(res), "--configs", str(configs)]) == 0
    # mux11: medians 5, 3, 2 -> 16; Con: 2, 1, 1 -> fallback 8; EnH: 9, 5, 3.5 -> 32
    want = {"mux11": 16, "Con": 8, "EnH": 32}
    after = _ks(configs)
    assert after.keys() == before.keys()
    for (ph, name, arm), k in after.items():
        prob = "mux11" if name.startswith("bool") else name.split("-")[-1].removesuffix(".json")
        if arm == "GOMEA":
            assert k is None
        else:
            assert k == want[prob], (ph, name, arm)
    log = json.loads((res / "phase1ak-pilot" / "knob-cap.json").read_text())
    assert {p: v["K"] for p, v in log["problems"].items()} == want
    assert log["problems"]["EnH"]["per_k"]["32"]["median_generations"] == 3.5
    assert log["problems"]["mux11"]["per_k"]["8"]["n_demes"] == 5
    # nothing but K changes in a manifest
    for ph in PHASES[1:]:
        for f in (configs / ph).glob("*.json"):
            orig = json.loads((REPO / "configs" / "gomea" / ph / f.name).read_text())
            new = json.loads(f.read_text())
            for spec in list(orig.get("arms", {}).values()) + list(new.get("arms", {}).values()):
                if skc.FLAG in spec["argv"]:
                    spec["argv"][spec["argv"].index(skc.FLAG) + 1] = "K"
            assert orig == new, f


def test_missing_runs_change_nothing(tmp_path, capsys):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs, skip={("Con", 16)})
    before = _ks(configs)
    assert skc.main(["--results", str(res), "--configs", str(configs)]) == 1
    assert _ks(configs) == before
    assert not (res / "phase1ak-pilot" / "knob-cap.json").exists()
    assert "Con: K=16: missing" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path):
    configs = _configs(tmp_path)
    res = _tree(tmp_path, configs)
    before = _ks(configs)
    assert skc.main(["--results", str(res), "--configs", str(configs), "--dry-run"]) == 0
    assert _ks(configs) == before
    assert not (res / "phase1ak-pilot" / "knob-cap.json").exists()


def test_a_manifest_with_problems_of_different_k_is_an_error(tmp_path):
    configs = _configs(tmp_path)
    m = json.loads((configs / "phase1ak" / "sr-Con.json").read_text())
    m["datasets"] = ["Con", "EnH"]
    (configs / "phase1ak" / "sr-Con.json").write_text(json.dumps(m))
    found = {"Con": {"K": 8}, "EnH": {"K": 32}, "mux11": {"K": 16}}
    _, errors = skc.plan_writes(configs, ["phase1ak"], found)
    assert len(errors) == 1 and "different K" in errors[0]


def test_choose_rule():
    assert skc.choose({8: 5, 16: 3, 32: 2.5}) == 16
    assert skc.choose({8: 2.5, 16: 1, 32: 0}) == 8
    assert skc.choose({8: 9, 16: 9, 32: 3}) == 32


# ---- extension manifests of the pilot (the rule of set_const_opt.py)


def _ext_setup(tmp_path):
    """A pilot with `sr-ext-Con.json` and `bool-ext-mux11.json`, which no run has used yet."""
    configs = _configs(tmp_path)
    pilot = configs / "phase1ak-pilot"
    sr = json.loads((pilot / "sr.json").read_text())
    ext = dict(sr, name=sr["name"] + "-ext-Con", datasets=["Con"], splits=[s + 100 for s in sr["splits"]])
    (pilot / "sr-ext-Con.json").write_text(json.dumps(ext))
    b = json.loads((pilot / "bool.json").read_text())
    bext = dict(b, name=b["name"] + "-ext-mux11", seeds=[s + 100 for s in b["seeds"]])
    (pilot / "bool-ext-mux11.json").write_text(json.dumps(bext))
    res = _tree(tmp_path, configs)  # the stage-1 runs only
    return configs, res


def _verdict(res, verdicts):
    d = res / "phase1ak-pilot" / "report-stage1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verdict.json").write_text(json.dumps(verdicts))


def _spec(configs, primary):
    (configs / "phase1ak-pilot" / "spec.json").write_text(json.dumps({"primary": primary}))


def _argv(configs, res, *extra):
    return ["--results", str(res), "--configs", str(configs), *extra]


def test_extensions_without_an_extend_verdict_are_not_expected(tmp_path, capsys):
    configs, res = _ext_setup(tmp_path)
    _spec(configs, [["Con", "x", "A", "B"], ["mux11", "y", "A", "B"]])
    _verdict(res, {"Con:x": "tie", "mux11:y": "tie"})
    assert skc.main(_argv(configs, res)) == 0
    out = capsys.readouterr().out
    assert "ignore extension sr-ext-Con.json" in out
    assert "ignore extension bool-ext-mux11.json" in out
    log = json.loads((res / "phase1ak-pilot" / "knob-cap.json").read_text())
    assert log["extended"] == []


def test_an_extend_verdict_makes_the_runs_of_that_extension_necessary(tmp_path, capsys):
    configs, res = _ext_setup(tmp_path)
    _spec(configs, [["Con", "x", "A", "B"], ["mux11", "y", "A", "B"]])
    _verdict(res, {"Con:x": "extend", "mux11:y": "tie"})
    before = _ks(configs)
    assert skc.main(_argv(configs, res)) == 1
    assert "missing pilot run telemetry" in capsys.readouterr().err
    assert _ks(configs) == before
    assert not (res / "phase1ak-pilot" / "knob-cap.json").exists()


def test_the_runs_of_an_extended_problem_count(tmp_path):
    configs, res = _ext_setup(tmp_path)
    pilot = configs / "phase1ak-pilot"
    _spec(configs, [["Con", "x", "A", "B"]])
    _verdict(res, {"Con:x": "extend"})
    # Con's K = 8 by stage 1. The extension makes K = 16 qualify: many generations per deme.
    ext = skc.select_extensions(res, pilot, False)[0]
    assert [f.name for f in ext] == ["sr-ext-Con.json"]
    stage1 = skc.pilot_runs(res, pilot)
    both = skc.pilot_runs(res, pilot, ext)
    for k, files in both["Con"].items():
        for f in set(files) - set(stage1["Con"][k]):
            _events(Path(f), [20] * 30)
    assert skc.main(_argv(configs, res)) == 0
    log = json.loads((res / "phase1ak-pilot" / "knob-cap.json").read_text())
    assert log["extended"] == ["Con"]
    assert log["problems"]["Con"]["K"] == 32
    assert log["problems"]["mux11"]["K"] == 16  # no extension: stage 1 only


def test_stage1_only_reads_no_verdict(tmp_path):
    configs, res = _ext_setup(tmp_path)
    assert skc.main(_argv(configs, res, "--stage1-only")) == 0


def test_extension_manifests_need_a_readable_verdict(tmp_path, capsys):
    configs, res = _ext_setup(tmp_path)
    _spec(configs, [["Con", "x", "A", "B"]])
    assert skc.main(_argv(configs, res)) == 1
    assert "cannot tell which extensions ran" in capsys.readouterr().err
