"""Server round 5: the plans (plan5.json, plan5-smoke.json) and the configurations of 1at, 1au and
1av, as SERVER-ROUND5-PREREGISTRATION.md registers them."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"

BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
        "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
LM = ["--const-opt", "lm"]
D6 = ["WiR", "Con", "EnH", "Air", "Yac", "Tow"]
D3 = ["Con", "EnH", "Air"]


def pen(r):
    return ["--gomea-accept-complexity-ratio", r, "--gomea-deme-store", "accepted"]


ARMS = {
    "1at": {"GOMEA-LS": [], "LM": LM, "LM-VAL": [*LM, "--const-opt-guard", "val"]},
    "1au": {"GOMEA-LS": [], "PEN25": pen("0.025"), "PEN125": pen("0.125"), "PEN625": pen("0.625")},
    "1av": {"GOMEA-LS": [], "LM": LM, "LM-PEN5": [*LM, *pen("0.005")],
            "LM-PEN25": [*LM, *pen("0.025")]},
}
DATASETS = {"1at": D6, "1au": D3, "1av": D3}
BUDGET = {"1at": {"time_s": 1000}, "1au": {"evals": 10000}, "1av": {"time_s": 1000}}
WALL = {"1at": 2400, "1au": 36000, "1av": 2400}
SIZE_BASE = {"1au": "GOMEA-LS", "1av": "LM"}
PHASES = ["1at", "1au", "1av"]


def load(path):
    return json.loads(path.read_text())


def test_base_arm_is_the_gomea_ls_arm_of_1ao():
    assert load(CONFIGS / "phase1ao" / "sr.json")["arms"]["GOMEA-LS"]["argv"] == BASE


@pytest.mark.parametrize("ph", PHASES)
def test_stage_1_manifest(ph):
    m = load(CONFIGS / f"phase{ph}" / "sr.json")
    assert m["name"] == f"phase{ph}-sr"
    assert m["datasets"] == DATASETS[ph] and m["splits"] == list(range(1, 11))
    assert m["wall_cap_s"] == WALL[ph] and m["mem_limit_mb"] == 3072
    assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
    assert list(m["arms"]) == list(ARMS[ph])
    for arm, spec in m["arms"].items():
        assert spec["argv"] == BASE + ARMS[ph][arm], arm
        assert spec["budget"] == BUDGET[ph] and spec["binary"] == ".venv/bin/moses"


@pytest.mark.parametrize("ph", PHASES)
def test_extensions_hold_all_arms_on_splits_11_to_20(ph):
    q = CONFIGS / f"phase{ph}"
    stage1 = load(q / "sr.json")
    assert {f.name for f in q.glob("sr-ext-*.json")} == {f"sr-ext-{d}.json" for d in DATASETS[ph]}
    for d in DATASETS[ph]:
        m = load(q / f"sr-ext-{d}.json")
        assert m["name"] == f"phase{ph}-sr-ext-{d}"
        assert m["datasets"] == [d] and m["splits"] == list(range(11, 21))
        assert m["arms"] == stage1["arms"] and m["wall_cap_s"] == stage1["wall_cap_s"]


@pytest.mark.parametrize("prefix", ["phase", "server-smoke5-"])
def test_spec_of_1at(prefix):
    s = load(CONFIGS / f"{prefix}1at" / "spec.json")
    assert s["metrics"] == dict.fromkeys(D6, "test_nmse")
    assert [(p, a, b) for p, _, a, b in s["primary"]] == [(d, "LM-VAL", "LM") for d in D6]
    assert [(p, a, b) for p, _, a, b in s["secondary"]] == [(d, "LM-VAL", "GOMEA-LS") for d in D6]


@pytest.mark.parametrize("prefix", ["phase", "server-smoke5-"])
@pytest.mark.parametrize("ph", ["1au", "1av"])
def test_spec_of_the_size_phases(ph, prefix):
    s = load(CONFIGS / f"{prefix}{ph}" / "spec.json")
    base = SIZE_BASE[ph]
    arms = [a for a in ARMS[ph] if a not in ("GOMEA-LS", "LM")]
    assert s["metrics"] == dict.fromkeys(D3, "program_size")
    assert len(s["primary"]) == 3 * len(arms) == len(s["secondary"])
    assert {(p, a, b) for p, _, a, b in s["primary"]} == {(d, a, base) for d in D3 for a in arms}
    assert {(p, a, b, m) for p, _, a, b, m in s["secondary"]} == {
        (d, a, base, "test_nmse") for d in D3 for a in arms}
    names = [(c[0], c[1]) for c in s["primary"] + s["secondary"]]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("ph, primary", [("1at", 6), ("1au", 9), ("1av", 6)])
def test_stage_1_can_give_a_verdict(ph, primary):
    """10 pairs: the lowest raw p is 0.00195, so Holm over the phase's family stays below 0.025."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert len(load(CONFIGS / f"phase{ph}" / "spec.json")["primary"]) == primary
    assert holm([p] * primary)[0] < 0.025


def test_registered_run_counts():
    runs = 0
    for ph in PHASES:
        m = load(CONFIGS / f"phase{ph}" / "sr.json")
        n = len(m["arms"]) * len(m["datasets"]) * len(m["splits"])
        assert n == {"1at": 180, "1au": 120, "1av": 120}[ph]
        ext = sum(len(load(f)["arms"]) * len(load(f)["splits"])
                  for f in (CONFIGS / f"phase{ph}").glob("sr-ext-*.json"))
        assert ext == n
        runs += n
    assert runs == 420


@pytest.mark.parametrize("ph", PHASES)
def test_no_boolean_manifest_so_no_commit_to_set(ph):
    assert not list((CONFIGS / f"phase{ph}").glob("bool*.json"))
    assert not list((CONFIGS / f"server-smoke5-{ph}").glob("bool*.json"))


@pytest.mark.parametrize("plan, state_dir, phases", [
    ("plan5.json", "server-run5", ["phase1at", "phase1au", "phase1av"]),
    ("plan5-smoke.json", "server-run5-smoke",
     ["server-smoke5-1at", "server-smoke5-1au", "server-smoke5-1av"]),
])
def test_plan_is_s0_then_the_three_phases(plan, state_dir, phases):
    p = load(SERVER / plan)
    assert p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase", "phase", "phase"]
    assert [s["phase"] for s in p["steps"][1:]] == phases
    assert "then" not in p
    for s, ds in zip(p["steps"][1:], [D6, D3, D3], strict=True):
        assert s["requires"] == [f"datasets:{','.join(ds)}"]
        assert not s.get("pilot") and not s.get("set_commit_in")
        assert (CONFIGS / s["phase"] / "spec.json").exists()
        assert (CONFIGS / s["phase"] / "sr.json").exists()


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan5.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


@pytest.mark.parametrize("name, plan", [("round5", "plan5.json"), ("smoke5", "plan5-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)


@pytest.mark.parametrize("script", ["start.sh", "stop.sh", "status.sh", "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round5|smoke5" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    """Tow is a download, the others come from data/gomea-sr."""
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(D6) - {"Tow"} <= set(built)
    assert '"Tow tower ' in text


@pytest.mark.parametrize("ph", PHASES)
def test_smoke_has_the_same_arms_and_is_small(ph):
    m = load(CONFIGS / f"server-smoke5-{ph}" / "sr.json")
    full = load(CONFIGS / f"phase{ph}" / "sr.json")
    assert m["name"] == f"server-smoke5-{ph}-sr"
    assert m["datasets"] == full["datasets"] and m["splits"] == [1, 2]
    assert m["wall_cap_s"] <= 300
    assert list(m["arms"]) == list(full["arms"])
    for arm, spec in m["arms"].items():
        assert spec["argv"] == full["arms"][arm]["argv"]
        b = spec["budget"]
        assert b.get("time_s", 0) <= 15 and b.get("evals", 0) <= 400
