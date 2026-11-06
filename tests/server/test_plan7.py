"""Server round 7: the plans (plan7.json, plan7-smoke.json), the configuration of 1ax, the chain
from round 6, and the orchestrator's `choice` step and `config_dir`, as
SERVER-ROUND7-PREREGISTRATION.md registers them."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"

NOCAP = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
         "--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
         "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
HC = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
D10 = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
PREFIXES = ["phase", "server-smoke7-"]


def load(path):
    return json.loads(path.read_text())


def test_nocap_is_the_nocap_arm_of_1aw():
    assert load(CONFIGS / "phase1aw" / "sr.json")["arms"]["NOCAP"]["argv"] == NOCAP


def test_template_of_1ax():
    m = load(CONFIGS / "phase1ax" / "template.json")
    assert m["name"] == "phase1ax-sr"
    assert m["datasets"] == D10 and m["splits"] == list(range(21, 31))
    ar = load(CONFIGS / "phase1ar" / "sr.json")
    assert m["wall_cap_s"] == ar["wall_cap_s"] == 2400
    assert m["mem_limit_mb"] == ar["mem_limit_mb"] == 3072
    assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
    assert list(m["arms"]) == ["FINAL", "NOCAP", "HC-LS"]
    for spec in m["arms"].values():
        assert spec["budget"] == {"time_s": 1000} and spec["binary"] == ".venv/bin/moses"
        assert set(spec) == {"binary", "budget", "argv"}
    assert m["arms"]["NOCAP"]["argv"] == NOCAP
    assert m["arms"]["HC-LS"]["argv"] == HC


@pytest.mark.parametrize("prefix", PREFIXES)
def test_final_waits_for_the_choice(prefix):
    """FINAL has no argv until set_final.py writes the resolved manifest into the state
    directory; no tracked sr.json exists, so phase_run.sh refuses the tracked directory."""
    q = CONFIGS / f"{prefix}1ax"
    assert load(q / "template.json")["arms"]["FINAL"]["argv"] is None
    assert not (q / "sr.json").exists()
    assert sorted(f.name for f in q.iterdir()) == ["spec.json", "template.json"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_no_cap_in_any_arm(prefix):
    m = load(CONFIGS / f"{prefix}1ax" / "template.json")
    for arm, spec in m["arms"].items():
        argv = spec["argv"] or []
        assert not any(t.startswith("--gomea-max-evals-per-deme") for t in argv), arm
        assert "hc-capped" not in argv


@pytest.mark.parametrize("prefix", PREFIXES)
def test_no_extension(prefix):
    q = CONFIGS / f"{prefix}1ax"
    assert not list(q.glob("*-ext-*.json"))
    assert load(q / "spec.json")["primary"] == []


def test_fresh_splits_only():
    """Rounds 1 to 6 used splits 1-20 only; 1ax uses 21-30, which exist for every dataset."""
    m = load(CONFIGS / "phase1ax" / "template.json")
    assert min(m["splits"]) == 21 and max(m["splits"]) == 30
    for f in CONFIGS.glob("phase1a[a-w]/*.json"):
        d = load(f)
        if "splits" in d:
            assert max(d["splits"]) <= 20, f


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec_of_1ax(prefix):
    s = load(CONFIGS / f"{prefix}1ax" / "spec.json")
    assert s["phase"] == ("1ax" if prefix == "phase" else "smoke7-1ax")
    assert s["metrics"] == dict.fromkeys(D10, "test_nmse")
    assert s["pooled_primary"] == [["final-vs-hc", "FINAL", "HC-LS", "test_nmse"],
                                   ["final-vs-nocap", "FINAL", "NOCAP", "test_nmse"]]
    assert s["secondary"] == (
        [[d, "final-vs-hc", "FINAL", "HC-LS", "test_nmse"] for d in D10]
        + [[d, "final-vs-nocap", "FINAL", "NOCAP", "test_nmse"] for d in D10])
    assert s["secondary_verdicts"] is False and "descriptive" not in s


def test_secondary_cannot_give_a_verdict():
    """The pre-registration: with 10 pairs and 20 contrasts the lowest Holm p is 0.039."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p] * 20)[0] == pytest.approx(0.0390625)


def test_registered_run_count():
    m = load(CONFIGS / "phase1ax" / "template.json")
    assert len(m["arms"]) * len(m["datasets"]) * len(m["splits"]) == 300


def test_smoke_has_the_same_arms_and_is_small():
    full = load(CONFIGS / "phase1ax" / "template.json")
    m = load(CONFIGS / "server-smoke7-1ax" / "template.json")
    assert m["name"] == "server-smoke7-1ax-sr"
    assert m["datasets"] == D10 and m["splits"] == [21, 22]
    assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == full["mem_limit_mb"]
    assert list(m["arms"]) == list(full["arms"])
    for arm, spec in m["arms"].items():
        assert spec["argv"] == full["arms"][arm]["argv"]
        assert spec["budget"] == {"time_s": 15}


@pytest.mark.parametrize("plan, name, state_dir, source, target, sid", [
    ("plan7.json", "server-round7-2026-11", "server-run7", "phase1aw", "phase1ax", "phase1ax"),
    ("plan7-smoke.json", "server-round7-smoke", "server-run7-smoke", "server-smoke6-1aw",
     "server-smoke7-1ax", "smoke7-1ax"),
])
def test_plan_is_s0_then_the_choice_then_1ax(plan, name, state_dir, source, target, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "choice", "phase"]
    s0, ch, ph = p["steps"]
    assert [s["id"] for s in p["steps"]] == ["S0", "choose-final", sid]
    cfg = f"{{state}}/{target}-config"
    assert ch["argv"] == [".venv/bin/python", "tools/gomea/set_final.py", "--results",
                          "{results}", "--source", source, "--target", target, "--out", cfg]
    assert ch["record"] == f"{cfg}/final-choice.json"
    assert ph["phase"] == target and ph["config_dir"] == cfg
    assert ph["requires"] == [f"datasets:{','.join(D10)}"]
    assert not ph.get("pilot") and not ph.get("set_commit_in")
    assert (CONFIGS / target / "spec.json").exists()
    assert "then" not in p


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan7.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    args = load(SERVER / "plan7-smoke.json")["steps"][0]["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_set_final.py",
              "tests/gomea/test_phase_report_pooled.py", "tests/gomea/test_phase_report.py"):
        assert t in args


@pytest.mark.parametrize("plan, then", [("plan6.json", "round7"), ("plan6-smoke.json", "smoke7")])
def test_round_6_chains_to_round_7(plan, then):
    assert load(SERVER / plan)["then"] == then


@pytest.mark.parametrize("name, plan", [("round7", "plan7.json"), ("smoke7", "plan7-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round7|smoke7" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(D10) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text


FAKE_PHASE = r"""#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
echo "$1 PHASE_CONFIG_DIR=${PHASE_CONFIG_DIR:-unset}" >> ../gomea-results/trace.log
"""
FAKE_CHOICE = r"""import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
if (Path(sys.argv[0]).parent / "fail").exists():
    sys.exit(1)
out.mkdir(parents=True, exist_ok=True)
(out / "spec.json").write_text("{}")
(out / "sr.json").write_text("{}")
(out / "final-choice.json").write_text(json.dumps({"chosen_arm": "LMVAL", "size_better": 2}))
print("-> FINAL = LMVAL")
"""


def _rig(tmp_path, fail=False):
    repo, res = tmp_path / "pyMoses", tmp_path / "gomea-results"
    (repo / "tools" / "server").mkdir(parents=True)
    res.mkdir()
    for f in SERVER.iterdir():
        if f.suffix in (".py", ".sh"):
            shutil.copy(f, repo / "tools" / "server" / f.name)
    (repo / "tools" / "gomea").mkdir()
    (repo / "tools" / "gomea" / "phase_run.sh").write_text(FAKE_PHASE)
    (repo / "tools" / "gomea" / "set_final.py").write_text(FAKE_CHOICE)
    if fail:
        (repo / "tools" / "gomea" / "fail").write_text("")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
    for ph in ("p6", "p7"):
        (repo / "configs" / "gomea" / ph).mkdir(parents=True)
        (repo / "configs" / "gomea" / ph / "spec.json").write_text("{}")
    plan = {"name": "t7", "state_dir": "server-run7", "steps": [
        {"id": "p6", "type": "phase", "phase": "p6"},
        {"id": "choose-final", "type": "choice",
         "argv": [".venv/bin/python", "tools/gomea/set_final.py", "--results", "{results}",
                  "--out", "{state}/p7-config"],
         "record": "{state}/p7-config/final-choice.json"},
        {"id": "p7", "type": "phase", "phase": "p7", "config_dir": "{state}/p7-config"}]}
    (repo / "tools" / "server" / "plan-t7.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PHASE_CONFIG_DIR="/must/not/leak")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t7.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    sd = res / "server-run7"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    return res, sd, events


def test_choice_step_and_config_dir(tmp_path):
    res, sd, events = _rig(tmp_path)
    trace = (res / "trace.log").read_text().splitlines()
    assert trace == ["p6 PHASE_CONFIG_DIR=unset", f"p7 PHASE_CONFIG_DIR={sd / 'p7-config'}"]
    ch = [e for e in events if e["event"] == "choice"]
    assert len(ch) == 1 and ch[0]["rc"] == 0
    assert ch[0]["record"] == {"chosen_arm": "LMVAL", "size_better": 2}
    assert ch[0]["argv"][3] == str(res) and ch[0]["argv"][5] == str(sd / "p7-config")
    st = load(sd / "state.json")["steps"]
    assert [st[s]["status"] for s in ("p6", "choose-final", "p7")] == ["done"] * 3
    assert "## Choice" in (sd / "round-summary.md").read_text()
    assert (sd / "DONE").exists()


def test_failed_choice_skips_the_phase(tmp_path):
    res, sd, events = _rig(tmp_path, fail=True)
    assert (res / "trace.log").read_text().splitlines() == ["p6 PHASE_CONFIG_DIR=unset"]
    st = load(sd / "state.json")["steps"]
    assert st["choose-final"]["status"] == "failed" and st["p7"]["status"] == "skipped"
    assert "did the choice step fail?" in st["p7"]["reason"]
