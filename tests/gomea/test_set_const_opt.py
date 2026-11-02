"""tools/gomea/set_const_opt.py: the 1an -> 1ao/1ar TUNE choice (server round 3)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import set_const_opt  # noqa: E402

DS = ["Con", "EnH", "Air"]


def _manifest(name, datasets, splits, arms):
    return {"name": name, "datasets": datasets, "splits": splits,
            "arms": {a: {"argv": argv} for a, argv in arms.items()}}


def _write_run(res, name, ds, arm, s, v):
    d = res / "phase1an" / name / "runs" / f"{ds}__{arm}__s{s}"
    d.mkdir(parents=True)
    d.joinpath("summary.json").write_text(json.dumps(
        {"status": "ok" if v is not None else "error", "test_nmse": v}))


def _setup(tmp_path, nmse, ext=None, missing=(), idle_ext=()):
    """nmse(mode, ds, split) -> value or None (failed run).

    `ext`: a dataset that extended (manifest, runs and an "extend" verdict). `idle_ext`:
    datasets that did not extend (manifest only, verdicts "tie"). With either, the source
    phase also has spec.json and report-stage1/verdict.json, as after phase_run.sh.
    """
    cfg, res = tmp_path / "configs", tmp_path / "results"
    src = cfg / "phase1an"
    src.mkdir(parents=True)
    arms = {"GOMEA-LS": [], "GOMEA-LS-LM": ["--const-opt", "lm"],
            "GOMEA-LS-MUT": ["--const-opt", "mutate"]}
    mans = [_manifest("phase1an-sr", DS, list(range(1, 11)), arms)]
    (src / "sr.json").write_text(json.dumps(mans[0]))
    if ext:
        m = _manifest(f"phase1an-sr-ext-{ext}", [ext], list(range(11, 21)), arms)
        (src / f"sr-ext-{ext}.json").write_text(json.dumps(m))
        mans.append(m)
    for m in mans:
        for ds in m["datasets"]:
            for s in m["splits"]:
                for mode, arm in set_const_opt.ARMS.items():
                    if (ds, arm, s) in missing:
                        continue
                    _write_run(res, m["name"], ds, arm, s, nmse(mode, ds, s))
    for ds in idle_ext:
        m = _manifest(f"phase1an-sr-ext-{ds}", [ds], list(range(11, 21)), arms)
        (src / f"sr-ext-{ds}.json").write_text(json.dumps(m))
    if ext or idle_ext:
        rows = [[ds, c, a, "GOMEA-LS"] for ds in DS
                for c, a in (("lm", "GOMEA-LS-LM"), ("mut", "GOMEA-LS-MUT"))]
        (src / "spec.json").write_text(json.dumps({"phase": "1an", "primary": rows}))
        report = res / "phase1an" / "report-stage1"
        report.mkdir(parents=True)
        (report / "verdict.json").write_text(json.dumps(
            {f"{ds}:{c}": "extend" if (ds, c) == (ext, "mut") else "tie" for ds, c, *_ in rows}))
    tune = {"GOMEA-LS": [], "GOMEA-LS-TUNE": ["-a", "gomea", "--const-opt",
                                              set_const_opt.PLACEHOLDER], "HC-LS": []}
    for ph in ("phase1ao", "phase1ar"):
        (cfg / ph).mkdir()
        (cfg / ph / "sr.json").write_text(json.dumps(_manifest(f"{ph}-sr", DS, [1], tune)))
        (cfg / ph / "sr-ext-Con.json").write_text(json.dumps(_manifest(f"{ph}-x", DS, [11], tune)))
        (cfg / ph / "spec.json").write_text("{}")
    return cfg, res


def _run(cfg, res, *extra):
    return set_const_opt.main(["--results", str(res), "--configs", str(cfg), *extra])


def _tune(cfg, ph="phase1ao", f="sr.json"):
    argv = json.loads((cfg / ph / f).read_text())["arms"]["GOMEA-LS-TUNE"]["argv"]
    return argv[argv.index("--const-opt") + 1]


def test_lower_pooled_median_wins(tmp_path):
    # LM is better on Con and EnH, MUT much better on Air: the pooled median
    # (median of 3 per-dataset medians) prefers LM
    def nmse(mode, ds, s):
        base = {"Con": 30.0, "EnH": 5.0, "Air": 40.0}[ds]
        if mode == "lm":
            return base - 1.0 + 0.01 * s if ds != "Air" else base + 20
        return base + 0.01 * s if ds != "Air" else base - 5

    cfg, res = _setup(tmp_path, nmse)
    assert _run(cfg, res) == 0
    log = json.loads((res / "phase1an" / "const-opt-choice.json").read_text())
    assert log["choice"] == "lm"
    assert _tune(cfg) == "lm" and _tune(cfg, "phase1ar") == "lm"
    assert _tune(cfg, f="sr-ext-Con.json") == "lm"
    assert log["n_runs"]["lm"] == {"Con": 10, "EnH": 10, "Air": 10}


def test_failed_runs_count_as_infinite(tmp_path):
    # LM has lower values but fails on 6 of 10 EnH and 6 of 10 Con splits
    def nmse(mode, ds, s):
        if mode == "lm":
            return None if (ds != "Air" and s <= 6) else 1.0
        return 2.0

    cfg, res = _setup(tmp_path, nmse)
    assert _run(cfg, res) == 0
    assert _tune(cfg) == "mutate"


def test_extension_runs_are_included_unless_stage1_only(tmp_path):
    def nmse(mode, ds, s):
        if ds == "Con" and s > 10:
            return 100.0 if mode == "lm" else 1.0
        return {"lm": 1.0, "mutate": 1.5}[mode]

    cfg, res = _setup(tmp_path, nmse, ext="Con")
    # final results: the Con medians become lm 50.5, mut 1.25; pooled lm 1.0 < mut 1.5
    assert _run(cfg, res, "--dry-run") == 0
    assert _tune(cfg) == set_const_opt.PLACEHOLDER  # dry run changes nothing
    assert not (res / "phase1an" / "const-opt-choice.json").exists()
    assert _run(cfg, res, "--stage1-only") == 0
    log = json.loads((res / "phase1an" / "const-opt-choice.json").read_text())
    assert log["n_runs"]["lm"]["Con"] == 10 and log["choice"] == "lm"
    assert _run(cfg, res) == 0
    log = json.loads((res / "phase1an" / "const-opt-choice.json").read_text())
    assert log["n_runs"]["lm"]["Con"] == 20


def test_tie_goes_to_mutate(tmp_path):
    cfg, res = _setup(tmp_path, lambda m, d, s: 1.0)
    assert _run(cfg, res) == 0 and _tune(cfg) == "mutate"


def test_missing_run_changes_nothing(tmp_path):
    cfg, res = _setup(tmp_path, lambda m, d, s: 1.0, missing={("EnH", "GOMEA-LS-LM", 3)})
    assert _run(cfg, res) == 1
    assert _tune(cfg) == set_const_opt.PLACEHOLDER
    assert not (res / "phase1an" / "const-opt-choice.json").exists()


def _lm_wins(mode, ds, s):
    return {"lm": 1.0, "mutate": 1.5}[mode]


def _log(res):
    return json.loads((res / "phase1an" / "const-opt-choice.json").read_text())


def _nothing_written(cfg, res):
    return (_tune(cfg) == set_const_opt.PLACEHOLDER
            and not (res / "phase1an" / "const-opt-choice.json").exists())


def test_extension_manifests_without_an_extend_verdict_are_ignored(tmp_path):
    # the server failure: the manifests exist, no extension ran, no verdict is "extend"
    cfg, res = _setup(tmp_path, _lm_wins, idle_ext=DS)
    assert _run(cfg, res) == 0
    log = _log(res)
    assert log["n_runs"]["lm"] == log["n_runs"]["mutate"] == {"Con": 10, "EnH": 10, "Air": 10}
    assert log["choice"] == "lm" and _tune(cfg) == "lm" and _tune(cfg, "phase1ar") == "lm"
    assert log["extended_datasets"] == []


def test_only_the_extended_dataset_gets_its_extension(tmp_path, capsys):
    cfg, res = _setup(tmp_path, _lm_wins, ext="Con", idle_ext=["EnH", "Air"])
    assert _run(cfg, res) == 0
    log = _log(res)
    assert log["n_runs"]["lm"] == log["n_runs"]["mutate"] == {"Con": 20, "EnH": 10, "Air": 10}
    assert log["extended_datasets"] == ["Con"]
    out = capsys.readouterr().out
    assert "include extension sr-ext-Con.json" in out
    assert "ignore extension sr-ext-EnH.json" in out
    assert "ignore extension sr-ext-Air.json" in out


def test_extend_verdict_with_a_missing_extension_run_changes_nothing(tmp_path):
    cfg, res = _setup(tmp_path, _lm_wins, ext="Con", idle_ext=["EnH", "Air"],
                      missing={("Con", "GOMEA-LS-LM", 15)})
    assert _run(cfg, res) == 1
    assert _nothing_written(cfg, res)


def test_missing_stage1_run_changes_nothing_with_extension_manifests(tmp_path):
    cfg, res = _setup(tmp_path, _lm_wins, idle_ext=DS, missing={("EnH", "GOMEA-LS-LM", 3)})
    assert _run(cfg, res) == 1
    assert _nothing_written(cfg, res)


def test_ignored_extension_with_some_runs_does_not_count(tmp_path):
    cfg, res = _setup(tmp_path, _lm_wins, idle_ext=["Con"])
    for s in range(11, 21):  # MUT would win if these values counted
        _write_run(res, "phase1an-sr-ext-Con", "Con", "GOMEA-LS-LM", s, 100.0)
    assert _run(cfg, res) == 0
    log = _log(res)
    assert log["n_runs"]["lm"]["Con"] == 10 and log["choice"] == "lm"
    assert log["extended_datasets"] == []


def _verdict_path(cfg, res):
    return res / "phase1an" / "report-stage1" / "verdict.json"


def _spec_path(cfg, res):
    return cfg / "phase1an" / "spec.json"


@pytest.mark.parametrize("path", [_verdict_path, _spec_path], ids=["verdict", "spec"])
@pytest.mark.parametrize("content", [None, "{not json", "[1]"], ids=["missing", "corrupt", "shape"])
def test_extension_manifests_need_a_readable_spec_and_verdict(tmp_path, path, content):
    cfg, res = _setup(tmp_path, _lm_wins, idle_ext=DS)
    if content is None:
        path(cfg, res).unlink()
    else:
        path(cfg, res).write_text(content)
    assert _run(cfg, res) == 1
    assert _nothing_written(cfg, res)


def test_stage1_only_does_not_need_the_verdict(tmp_path):
    cfg, res = _setup(tmp_path, _lm_wins, idle_ext=DS)
    _verdict_path(cfg, res).unlink()
    assert _run(cfg, res, "--stage1-only") == 0
    log = _log(res)
    assert log["n_runs"]["lm"] == {"Con": 10, "EnH": 10, "Air": 10}
    assert log["extended_datasets"] == []


def test_real_phase1an_configs_with_all_tie_verdicts(tmp_path):
    cfg, res = tmp_path / "configs", tmp_path / "results"
    for ph in ("phase1an", "phase1ao", "phase1ar"):
        shutil.copytree(REPO / "configs" / "gomea" / ph, cfg / ph)
    assert sorted(f.name for f in (cfg / "phase1an").glob("sr-ext-*.json")) == [
        "sr-ext-Air.json", "sr-ext-Con.json", "sr-ext-EnH.json"]
    m = json.loads((cfg / "phase1an" / "sr.json").read_text())
    for ds in m["datasets"]:
        for s in m["splits"]:
            for arm in set_const_opt.ARMS.values():
                _write_run(res, m["name"], ds, arm, s, 1.0)
    spec = json.loads((cfg / "phase1an" / "spec.json").read_text())
    report = res / "phase1an" / "report-stage1"
    report.mkdir(parents=True)
    (report / "verdict.json").write_text(json.dumps(
        {f"{p}:{n}": "tie" for p, n, *_ in spec["primary"] + spec["secondary"]}))
    assert _run(cfg, res) == 0
    log = _log(res)
    assert log["n_runs"]["lm"] == {ds: 10 for ds in m["datasets"]}
    assert log["choice"] == set_const_opt.TIE and log["extended_datasets"] == []
    assert _tune(cfg) == _tune(cfg, f="sr-ext-Con.json") == set_const_opt.TIE
    assert _tune(cfg, "phase1ar") == set_const_opt.TIE
