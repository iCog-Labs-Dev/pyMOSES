"""tools/gomea/set_const_opt.py: the 1an -> 1ao/1ar TUNE choice (server round 3)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import set_const_opt  # noqa: E402

DS = ["Con", "EnH", "Air"]


def _manifest(name, datasets, splits, arms):
    return {"name": name, "datasets": datasets, "splits": splits,
            "arms": {a: {"argv": argv} for a, argv in arms.items()}}


def _setup(tmp_path, nmse, ext=None, missing=()):
    """nmse(mode, ds, split) -> value or None (failed run)."""
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
                    d = res / "phase1an" / m["name"] / "runs" / f"{ds}__{arm}__s{s}"
                    d.mkdir(parents=True)
                    v = nmse(mode, ds, s)
                    d.joinpath("summary.json").write_text(json.dumps(
                        {"status": "ok" if v is not None else "error", "test_nmse": v}))
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
