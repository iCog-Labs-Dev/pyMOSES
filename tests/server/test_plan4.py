"""Server round 4: the plans (plan4.json, plan4-smoke.json) and the configurations of 1as, as
SERVER-ROUND4-PREREGISTRATION.md registers them."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from benchmarks.gomea.manifest import check_code_commit, expand_runs, load_manifest
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank
from pymoses.gomea.knob_cap import default_explore

REPO = Path(__file__).resolve().parents[2]
PHASE = REPO / "configs" / "gomea" / "phase1as"
SMOKE = REPO / "configs" / "gomea" / "server-smoke4-1as"

COMMON = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
          "--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1",
          "--knob-probe-baseline", "reduced"]
# arm: (--gomea-knob-cap, --gomea-max-evals-per-deme)
ARMS = {"BASE": (None, "250"), "NOCAP": (None, "0"), "K8": ("8", "0"), "K16": ("16", "0"),
        "K32": ("32", "0"), "K64": ("64", "0"), "K16-C250": ("16", "250")}
DATASETS = ["Con", "EnH", "Air"]


def load(path):
    return json.loads(path.read_text())


def flags(argv):
    """The argv as {flag: value}; every round-4 flag takes one value."""
    assert len(argv) % 2 == 0, argv
    d = dict(zip(argv[::2], argv[1::2], strict=True))
    assert len(d) == len(argv) // 2, f"a repeated flag in {argv}"
    return d


def check_arm(arm, argv, regression):
    knob_cap, deme_cap = ARMS[arm]
    expected = flags(COMMON) | {"--gomea-max-evals-per-deme": deme_cap}
    if regression:
        expected |= {"--gomea-contin": "copy", "--linear-scaling": "1"}
    if knob_cap is not None:
        # the default explore count: no --gomea-knob-explore
        expected |= {"--gomea-knob-cap": knob_cap, "--gomea-knob-select": "explore",
                     "--gomea-knob-rank0": "contrast"}
    assert flags(argv) == expected, arm


@pytest.mark.parametrize("plan, phase, state_dir", [
    ("plan4.json", "phase1as", "server-run4"),
    ("plan4-smoke.json", "server-smoke4-1as", "server-run4-smoke"),
])
def test_plan_is_s0_then_the_phase(plan, phase, state_dir):
    p = load(REPO / "tools" / "server" / plan)
    assert p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert p["steps"][1]["phase"] == phase
    assert p["steps"][1].get("requires", []) == []
    d = REPO / "configs" / "gomea" / phase
    assert (d / "spec.json").exists() and (d / "bool.json").exists() and (d / "sr.json").exists()
    for rel in p["steps"][1].get("set_commit_in", []):
        assert (REPO / rel).exists(), rel


def test_s0_of_the_round_is_complete():
    s0 = load(REPO / "tools" / "server" / "plan4.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


@pytest.mark.parametrize("name, plan", [("round4", "plan4.json"), ("smoke4", "plan4-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(REPO / "tools" / "server" / plan)


@pytest.mark.parametrize("script", ["start.sh", "stop.sh", "status.sh", "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    text = (REPO / "tools" / "server" / script).read_text()
    assert "round4|smoke4" in text


@pytest.mark.parametrize("manifest", ["bool.json", "bool-ext-mux11.json"])
def test_boolean_manifests(manifest):
    m = load(PHASE / manifest)
    assert list(m["tasks"]) == ["mux11"]
    assert m["tasks"]["mux11"]["argv"] == ["-H", "mux", "-k", "3"]
    assert m["budget"] == {"argv": ["--max-time", "3600", "-m", "1000000000"], "kind": "wall_time"}
    assert m["wall_cap_s"] > 3600
    assert list(m["arms"]) == list(ARMS)
    for arm, spec in m["arms"].items():
        check_arm(arm, spec["argv"], regression=False)
    first = 3001 if manifest == "bool.json" else 3011
    assert m["seeds"] == list(range(first, first + 10))
    load_manifest(PHASE / manifest)  # the runner's own validation
    assert len(expand_runs(m)) == 70


def test_regression_stage_1():
    m = load(PHASE / "sr.json")
    assert m["datasets"] == DATASETS and m["splits"] == list(range(1, 11))
    assert list(m["arms"]) == list(ARMS)
    for arm, spec in m["arms"].items():
        check_arm(arm, spec["argv"], regression=True)
        assert spec["budget"] == {"time_s": 1000}
    assert m["wall_cap_s"] > 1000


@pytest.mark.parametrize("dataset", DATASETS)
def test_regression_extensions_hold_all_seven_arms(dataset):
    m = load(PHASE / f"sr-ext-{dataset}.json")
    assert m["datasets"] == [dataset] and m["splits"] == list(range(11, 21))
    assert m["arms"] == load(PHASE / "sr.json")["arms"]
    assert m["wall_cap_s"] == load(PHASE / "sr.json")["wall_cap_s"]


def test_boolean_extension_holds_all_seven_arms():
    assert load(PHASE / "bool-ext-mux11.json")["arms"] == load(PHASE / "bool.json")["arms"]


def test_no_constant_tuning_in_any_arm():
    for f in [*PHASE.glob("*.json"), *SMOKE.glob("*.json")]:
        assert "--const-opt" not in f.read_text(), f


def test_spec_has_the_registered_contrasts():
    spec = load(PHASE / "spec.json")
    problems = ["mux11", *DATASETS]
    assert spec["metrics"] == {"mux11": "cpu_to_target", "Con": "test_nmse", "EnH": "test_nmse",
                               "Air": "test_nmse"}
    assert len(spec["primary"]) == 20 and len(spec["secondary"]) == 8
    assert {(p, a, b) for p, _, a, b in spec["primary"]} == {
        (p, a, "BASE") for p in problems for a in ("NOCAP", "K8", "K16", "K32", "K64")}
    assert {(p, a, b) for p, _, a, b in spec["secondary"]} == {
        (p, a, b) for p in problems
        for a, b in (("K16-C250", "BASE"), ("K16", "K16-C250"))}
    used = {x for c in spec["primary"] + spec["secondary"] for x in c[2:]}
    assert used == set(ARMS)
    names = [(c[0], c[1]) for c in spec["primary"] + spec["secondary"]]
    assert len(set(names)) == len(names)


def test_the_extensions_match_the_problems():
    """phase_run.sh finds an extension by the problem's name."""
    spec = load(PHASE / "spec.json")
    for p in {c[0] for c in spec["primary"]}:
        assert (PHASE / f"bool-ext-{p}.json").exists() or (PHASE / f"sr-ext-{p}.json").exists(), p


def test_manifest_commit_is_a_full_hash():
    for name in ("bool.json", "bool-ext-mux11.json"):
        assert re.fullmatch(r"[0-9a-f]{40}", load(PHASE / name)["commit"]), name


def test_manifest_commit_names_this_code():
    git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO,
                         capture_output=True, text=True)
    if git.returncode != 0:
        pytest.skip("no git checkout")
    res = check_code_commit(REPO, load(PHASE / "bool.json")["commit"])
    assert res["is_ancestor"] and res["code_unchanged"], res


def test_budgets_give_the_registered_size():
    sr, b = load(PHASE / "sr.json"), load(PHASE / "bool.json")
    runs_sr = len(sr["arms"]) * len(sr["datasets"]) * len(sr["splits"])
    runs_b = len(b["arms"]) * len(b["seeds"])
    assert runs_sr + runs_b == 280  # 7 arms x 40 tasks
    assert runs_sr * 1000 / 3600 == pytest.approx(58.3, abs=0.1)
    assert runs_b * 3600 / 3600 == 70


def test_stage_1_has_no_primary_verdict_but_can_extend():
    """The pre-registration's power note: 10 pairs, 20 contrasts, the lowest Holm p is 0.039."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert p == pytest.approx(0.00195, abs=1e-5)
    low = holm([p] * 20)[0]
    assert low == pytest.approx(0.039, abs=1e-3)
    assert 0.025 <= low < 0.25


@pytest.mark.parametrize("k, e", [(8, 2), (16, 4), (32, 8), (64, 16)])
def test_default_explore_count(k, e):
    assert default_explore(k) == e


def test_smoke_has_the_same_arms_and_is_small():
    bm, sm = load(SMOKE / "bool.json"), load(SMOKE / "sr.json")
    assert bm["commit"] == "SET_AT_RUNTIME"
    assert load(SMOKE / "bool-ext-mux6.json")["commit"] == "SET_AT_RUNTIME"
    assert bm["arms"] == load(PHASE / "bool.json")["arms"]
    assert sm["arms"].keys() == load(PHASE / "sr.json")["arms"].keys()
    for arm, spec in sm["arms"].items():
        assert spec["argv"] == load(PHASE / "sr.json")["arms"][arm]["argv"]
        assert spec["budget"]["time_s"] <= 20
    assert len(bm["seeds"]) == 2 and len(sm["splits"]) == 2
    spec = load(SMOKE / "spec.json")
    assert len(spec["primary"]) == 20 and len(spec["secondary"]) == 8
