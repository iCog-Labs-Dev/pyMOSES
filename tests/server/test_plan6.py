"""Server round 6: the plans (plan6.json, plan6-smoke.json) and the configurations of 1ay and
1az, as SERVER-ROUND6-PREREGISTRATION.md registers them."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.gomea.manifest import check_code_commit
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"

POPS = {"N16": ["--gomea-pop-size", "16"],
        "DYN2": ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "2.0"],
        "DYN025": ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.25"],
        "DYN005": ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05"]}
REG_BASE = ["--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
            "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
MUX_BASE = ["--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "0",
            "--gomea-genotype-memo", "1", "--knob-probe-baseline", "reduced"]
FOS = {"LTB": "lt-b", "RT": "rt"}
AY = {f"{p}-{f}": (FOS[f], POPS[p]) for p in POPS for f in FOS}
KNOB = {k: ["--gomea-knob-cap", str(k), "--gomea-knob-select", "explore",
            "--gomea-knob-rank0", "contrast"] for k in (8, 16, 32, 64)}
AZ = {"NOCAP-DYN": (["{DYN}"], []), **{f"K{k}-DYN": (["{DYN}"], KNOB[k]) for k in KNOB},
      "NOCAP-N16": (POPS["N16"], [])}
P3, P4 = ["mux11", "Con", "EnH"], ["mux11", "Con", "EnH", "Air"]


def load(path):
    return json.loads(path.read_text())


def ay_argv(arm, base):
    fos, pop = AY[arm]
    return ["-a", "gomea", "--gomea-fos", fos, *pop, *base]


def az_argv(arm, base):
    pop, knob = AZ[arm]
    return ["-a", "gomea", "--gomea-fos", "lt-b", *pop, *base, *knob]


# ---- 1ay


@pytest.mark.parametrize("q", ["phase1ay", "server-smoke6-1ay"])
def test_1ay_arms_and_flags(q):
    d = CONFIGS / q
    names = [d / "sr.json", d / "sr-ext-Con.json", d / "sr-ext-EnH.json"]
    for f in names:
        m = load(f)
        assert list(m["arms"]) == list(AY), f.name
        for arm, spec in m["arms"].items():
            assert spec["argv"] == ay_argv(arm, REG_BASE), (f.name, arm)
            assert spec["binary"] == ".venv/bin/moses"
            assert set(spec) == {"binary", "budget", "argv"}
    for f in (d / "bool.json", d / "bool-ext-mux11.json"):
        m = load(f)
        assert m["tasks"] == {"mux11": {"argv": ["-H", "mux", "-k", "3"], "target_raw": 0}}
        assert list(m["arms"]) == list(AY)
        for arm, spec in m["arms"].items():
            assert spec == {"argv": ay_argv(arm, MUX_BASE)}, (f.name, arm)


def test_1ay_problems_budgets_and_seeds():
    d = CONFIGS / "phase1ay"
    m = load(d / "sr.json")
    assert m["name"] == "phase1ay-sr" and m["datasets"] == ["Con", "EnH"]
    assert m["splits"] == list(range(1, 11))
    assert m["wall_cap_s"] == 2400 and m["mem_limit_mb"] == 3072
    assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
    assert {s["budget"]["time_s"] for s in m["arms"].values()} == {1000}
    for ds in ("Con", "EnH"):
        e = load(d / f"sr-ext-{ds}.json")
        assert e["name"] == f"phase1ay-sr-ext-{ds}" and e["datasets"] == [ds]
        assert e["splits"] == list(range(11, 21)) and e["arms"] == m["arms"]
    b, be = load(d / "bool.json"), load(d / "bool-ext-mux11.json")
    assert b["name"] == "phase1ay-bool" and be["name"] == "phase1ay-bool-ext-mux11"
    assert b["seeds"] == list(range(3001, 3011)) and be["seeds"] == list(range(3011, 3021))
    assert b["budget"] == be["budget"] == {"argv": ["--max-time", "3600", "-m", "1000000000"],
                                           "kind": "wall_time"}
    assert b["wall_cap_s"] == 5400 and b["arms"] == be["arms"]


def test_1ay_spec():
    s = load(CONFIGS / "phase1ay" / "spec.json")
    assert s["phase"] == "1ay"
    assert s["metrics"] == {"mux11": "cpu_to_target", "Con": "test_nmse", "EnH": "test_nmse"}
    assert [(p, a, b) for p, _, a, b in s["primary"]] == [
        (p, f"{d}-LTB", "N16-LTB") for p in P3 for d in ("DYN2", "DYN025", "DYN005")]
    assert [(p, a, b) for p, _, a, b in s["secondary"]] == [
        (p, f"{q}-LTB", f"{q}-RT") for p in P3 for q in POPS]
    names = [(c[0], c[1]) for c in s["primary"] + s["secondary"]]
    assert len(set(names)) == len(names) == 21
    assert "descriptive" not in s and "pooled_primary" not in s


def test_1ay_stage_1_can_give_a_verdict():
    """10 pairs: the lowest raw p is 0.00195; Holm over 9 gives 0.0176 < 0.025."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p] * 9)[0] < 0.025


def test_registered_run_counts():
    def n(q, *names):
        tot = 0
        for f in names:
            m = load(CONFIGS / q / f)
            tot += len(m["arms"]) * (len(m["seeds"]) if "seeds" in m
                                     else len(m["datasets"]) * len(m["splits"]))
        return tot
    assert n("phase1ay", "sr.json", "bool.json") == 240
    assert n("phase1ay", "sr-ext-Con.json", "sr-ext-EnH.json", "bool-ext-mux11.json") == 240
    assert n("phase1az", "template.json", "template-bool.json") == 240
    assert n("phase1az", "template-sr-ext-Con.json", "template-sr-ext-EnH.json",
             "template-sr-ext-Air.json", "template-bool-ext-mux11.json") == 240


def test_every_arm_parses():
    """Each argv of 1ay (and 1az with DYN resolved) is a valid moses command line."""
    from pymoses.moses.main.problem_params import _gomea_params_from_args
    from tests.gomea.test_integration import _cli_parser

    argvs = [ay_argv(a, b) for a in AY for b in (REG_BASE, MUX_BASE)]
    argvs += [[y for x in az_argv(a, REG_BASE) for y in (POPS["DYN025"] if x == "{DYN}" else [x])]
              for a in AZ]
    for argv in argvs:
        gp = _gomea_params_from_args(_cli_parser(True).parse_args(argv))
        assert gp.dynamic() == ("dynamic" in argv), argv
        if gp.dynamic():
            assert gp.hc_fraction_of_nn == float(argv[argv.index("--hc-fraction-of-nn") + 1])
        assert gp.max_evals_per_deme is None  # no evaluation cap in any arm


# ---- 1az


@pytest.mark.parametrize("q", ["phase1az", "server-smoke6-1az"])
def test_1az_is_templates_only(q):
    """The arms need DYN from 1ay: set_population.py writes the manifests into the state
    directory, so no tracked sr.json or bool.json exists (phase_run.sh refuses the tracked
    directory)."""
    d = CONFIGS / q
    assert sorted(f.name for f in d.iterdir()) == [
        "spec.json", "template-bool-ext-mux11.json", "template-bool.json",
        "template-sr-ext-Air.json", "template-sr-ext-Con.json", "template-sr-ext-EnH.json",
        "template.json"]


@pytest.mark.parametrize("q", ["phase1az", "server-smoke6-1az"])
def test_1az_arms_and_flags(q):
    d = CONFIGS / q
    for f in sorted(d.glob("template*.json")):
        m = load(f)
        assert list(m["arms"]) == list(AZ), f.name
        base = MUX_BASE if "bool" in f.name else REG_BASE
        for arm, spec in m["arms"].items():
            assert spec["argv"] == az_argv(arm, base), (f.name, arm)


def test_1az_knob_flags_are_those_of_1as():
    """K-DYN = the K arm of 1as with the population DYN and no deme-cap flag."""
    old = load(CONFIGS / "phase1as" / "sr.json")["arms"]
    new = load(CONFIGS / "phase1az" / "template.json")["arms"]
    for k in KNOB:
        a = old[f"K{k}"]["argv"]
        i = a.index("--gomea-max-evals-per-deme")
        a = a[:i] + a[i + 2:]
        j = a.index("--gomea-pop-size")
        assert a[:j] + ["{DYN}"] + a[j + 2:] == new[f"K{k}-DYN"]["argv"]


def test_1az_problems_budgets_and_seeds():
    d = CONFIGS / "phase1az"
    m = load(d / "template.json")
    assert m["name"] == "phase1az-sr" and m["datasets"] == ["Con", "EnH", "Air"]
    assert m["splits"] == list(range(1, 11)) and m["wall_cap_s"] == 2400
    assert {s["budget"]["time_s"] for s in m["arms"].values()} == {1000}
    for ds in m["datasets"]:
        e = load(d / f"template-sr-ext-{ds}.json")
        assert e["name"] == f"phase1az-sr-ext-{ds}" and e["datasets"] == [ds]
        assert e["splits"] == list(range(11, 21)) and e["arms"] == m["arms"]
    b, be = load(d / "template-bool.json"), load(d / "template-bool-ext-mux11.json")
    assert b["name"] == "phase1az-bool" and be["name"] == "phase1az-bool-ext-mux11"
    assert b["seeds"] == list(range(3001, 3011)) and be["seeds"] == list(range(3011, 3021))
    assert b["budget"]["argv"] == ["--max-time", "3600", "-m", "1000000000"]
    assert b["wall_cap_s"] == 5400


def test_1az_spec():
    s = load(CONFIGS / "phase1az" / "spec.json")
    assert s["phase"] == "1az"
    assert s["metrics"] == load(CONFIGS / "phase1as" / "spec.json")["metrics"]
    assert [(p, a, b) for p, _, a, b in s["primary"]] == [
        (p, f"K{k}-DYN", "NOCAP-DYN") for p in P4 for k in KNOB]
    assert [(p, a, b) for p, _, a, b in s["secondary"]] == [
        (p, "NOCAP-DYN", "NOCAP-N16") for p in P4]


def test_1az_stage_1_extends_like_1as():
    """16 primary contrasts and 10 pairs: the lowest Holm p is 0.031, so stage 1 gives no
    primary verdict, and a large effect extends that problem (as in round 4)."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert 0.025 <= holm([p] * 16)[0] < 0.25


@pytest.mark.parametrize("prefix", ["phase", "server-smoke6-"])
def test_specs_of_smoke_and_real_have_the_same_contrasts(prefix):
    for ph in ("1ay", "1az"):
        a = load(CONFIGS / f"phase{ph}" / "spec.json")
        b = load(CONFIGS / f"server-smoke6-{ph}" / "spec.json")
        assert b["phase"] == f"smoke6-{ph}"
        assert {k: v for k, v in a.items() if k != "phase"} == {
            k: v for k, v in b.items() if k != "phase"}


# ---- the Boolean code check


BOOL = [("phase1ay", "bool.json"), ("phase1ay", "bool-ext-mux11.json"),
        ("server-smoke6-1ay", "bool.json"), ("server-smoke6-1ay", "bool-ext-mux11.json"),
        ("phase1az", "template-bool.json"), ("phase1az", "template-bool-ext-mux11.json"),
        ("server-smoke6-1az", "template-bool.json"),
        ("server-smoke6-1az", "template-bool-ext-mux11.json")]


def test_all_boolean_manifests_name_one_full_commit():
    commits = {load(CONFIGS / q / f)["commit"] for q, f in BOOL}
    assert len(commits) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", commits.pop())


def test_the_boolean_commit_names_this_code():
    git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO,
                         capture_output=True, text=True)
    if git.returncode != 0:
        pytest.skip("no git checkout")
    res = check_code_commit(REPO, load(CONFIGS / "phase1ay" / "bool.json")["commit"])
    assert res["is_ancestor"] and res["code_unchanged"], res


def test_smoke_is_small_and_has_the_same_arms():
    for ph in ("1ay", "1az"):
        sfx = "" if ph == "1ay" else "template-"
        full_sr = load(CONFIGS / f"phase{ph}" / ("sr.json" if ph == "1ay" else "template.json"))
        sm = load(CONFIGS / f"server-smoke6-{ph}" / ("sr.json" if ph == "1ay" else "template.json"))
        assert sm["name"] == f"server-smoke6-{ph}-sr"
        assert sm["splits"] == [1, 2] and sm["datasets"] == full_sr["datasets"]
        assert sm["wall_cap_s"] <= 300
        assert {s["budget"]["time_s"] for s in sm["arms"].values()} == {15}
        assert {a: s["argv"] for a, s in sm["arms"].items()} == {
            a: s["argv"] for a, s in full_sr["arms"].items()}
        b = load(CONFIGS / f"server-smoke6-{ph}" / f"{sfx}bool.json")
        fb = load(CONFIGS / f"phase{ph}" / f"{sfx}bool.json")
        assert b["arms"] == fb["arms"] and b["tasks"] == fb["tasks"]
        assert b["seeds"] == [3001, 3002]
        assert int(b["budget"]["argv"][1]) <= 60


def test_smoke_runs_mux11_at_f_2():
    """The pre-registration's memory risk: the smoke test runs mux11 at F = 2.0."""
    b = load(CONFIGS / "server-smoke6-1ay" / "bool.json")
    assert "mux11" in b["tasks"] and b["arms"]["DYN2-LTB"]["argv"][4:8] == POPS["DYN2"]


# ---- the plans


@pytest.mark.parametrize("plan, name, state_dir, ay, az, then", [
    ("plan6.json", "server-round6-2026-11", "server-run6", "phase1ay", "phase1az", "round7"),
    ("plan6-smoke.json", "server-round6-smoke", "server-run6-smoke", "server-smoke6-1ay",
     "server-smoke6-1az", "smoke7"),
])
def test_plan_is_s0_1ay_the_choice_then_1az(plan, name, state_dir, ay, az, then):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and p["then"] == then
    assert [s["type"] for s in p["steps"]] == ["check", "phase", "choice", "phase"]
    s0, a, ch, z = p["steps"]
    sid = "phase" if plan == "plan6.json" else "smoke6-"
    assert [s["id"] for s in p["steps"]] == ["S0", f"{sid}1ay", "choose-population", f"{sid}1az"]
    assert a["phase"] == ay and a["requires"] == ["datasets:Con,EnH"]
    assert "config_dir" not in a
    cfg = f"{{state}}/{az}-config"
    assert ch["argv"] == [".venv/bin/python", "tools/gomea/set_population.py", "choose",
                          "--results", "{results}", "--source", ay, "--target", az,
                          "--out", cfg]
    assert ch["record"] == f"{cfg}/population-choice.json"
    assert z["phase"] == az and z["config_dir"] == cfg
    assert z["requires"] == ["datasets:Con,EnH,Air"]
    for s in (a, z):
        assert not s.get("pilot") and not s.get("set_commit_in")
        assert (CONFIGS / s["phase"] / "spec.json").exists()


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan6.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_round_6_tests():
    s0 = load(SERVER / "plan6-smoke.json")["steps"][0]
    for t in ("tests/server", "tests/gomea/test_pop_size_dynamic.py",
              "tests/gomea/test_set_population.py", "tests/gomea/test_phase_report.py"):
        assert t in s0["pytest"]["args"]
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


def test_no_plan_chains_to_round_6():
    """Round 6 starts by itself."""
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round6", "smoke6"), f.name


@pytest.mark.parametrize("name, plan", [("round6", "plan6.json"), ("smoke6", "plan6-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan


def test_the_chain_is_6_7_8():
    sys.path.insert(0, str(SERVER))
    import chain
    assert [p.name for p in chain.forward(SERVER / "plan6.json")] == [
        "plan6.json", "plan7.json", "plan8.json"]
    assert [p.name for p in chain.forward(SERVER / "plan6-smoke.json")] == [
        "plan6-smoke.json", "plan7-smoke.json", "plan8-smoke.json"]


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round6|smoke6" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert {"Con", "EnH", "Air"} <= set(built)
