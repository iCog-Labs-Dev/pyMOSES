"""Server round 3: the plans (plan3.json, plan3-smoke.json), their configurations and the
orchestrator's round-3 requirements (const-opt:PHASE, datasets:A,B)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "server"))
import orchestrate  # noqa: E402

PLANS = ["plan3.json", "plan3-smoke.json"]
PLACEHOLDER = "SET-BY-set_const_opt.py"


@pytest.mark.parametrize("plan", PLANS)
def test_plan_steps_have_configs(plan):
    p = json.loads((REPO / "tools" / "server" / plan).read_text())
    phases = []
    for s in p["steps"]:
        for key in ("pilot", "phase"):
            if s.get(key):
                phases.append(s[key])
                d = REPO / "configs" / "gomea" / s[key]
                assert (d / "spec.json").exists() and (d / "sr.json").exists(), d
        for req in s.get("requires", []):
            assert req.split(":", 1)[0] in ("const-opt", "datasets", "gpgomea", "knob-cap")
    # the round's order: 1an (as 1ao's pilot), 1ao, 1ap, 1aq, 1ar
    assert [ph.rsplit("1", 1)[1] for ph in phases] == ["an", "ao", "ap", "aq", "ar"]


@pytest.mark.parametrize("prefix", ["phase", "server-smoke3-"])
def test_tune_arms_wait_for_the_choice(prefix):
    for ph in ("1ao", "1ar"):
        for f in (REPO / "configs" / "gomea" / f"{prefix}{ph}").glob("sr*.json"):
            m = json.loads(f.read_text())
            if "GOMEA-LS-TUNE" not in m["arms"]:
                assert f.name.startswith("sr-ext-"), f  # 1ar extensions: primary arms only
                continue
            argv = m["arms"]["GOMEA-LS-TUNE"]["argv"]
            assert argv[argv.index("--const-opt") + 1] == PLACEHOLDER, f
            assert m["arms"]["GOMEA-LS-TUNE"]["budget"] == m["arms"]["GOMEA-LS"]["budget"]


def test_round3_arms_as_preregistered():
    sr = json.loads((REPO / "configs/gomea/phase1an/sr.json").read_text())
    base = json.loads((REPO / "configs/gomea/phase1ag/sr.json").read_text())["arms"]
    assert sr["arms"]["GOMEA-LS"]["argv"] == base["GOMEA-LS"]["argv"]
    assert sr["arms"]["HC-LS"]["argv"] == base["HC-LS"]["argv"]
    assert sr["datasets"] == ["Con", "EnH", "Air"] and sr["splits"] == list(range(1, 11))
    assert all(a["budget"] == {"evals": 10000} for a in sr["arms"].values())
    for ph in ("phase1ao", "phase1ar"):
        m = json.loads((REPO / f"configs/gomea/{ph}/sr.json").read_text())
        assert all(a["budget"] == {"time_s": 1000} for a in m["arms"].values())
    ar = json.loads((REPO / "configs/gomea/phase1ar/sr.json").read_text())
    assert ar["datasets"] == ["Yac", "Dow", "Tow", "WiR", "WiW"]
    ap = json.loads((REPO / "configs/gomea/phase1ap/sr.json").read_text())["arms"]
    for arm, r in (("GOMEA-LS-PEN5", "0.005"), ("GOMEA-LS-PEN1", "0.001")):
        assert ap[arm]["argv"][-4:] == ["--gomea-accept-complexity-ratio", r,
                                         "--gomea-deme-store", "accepted"]
    spec = json.loads((REPO / "configs/gomea/phase1ap/spec.json").read_text())
    assert set(spec["metrics"].values()) == {"program_size"}
    assert all(c[4] == "test_nmse" for c in spec["secondary"])


@pytest.fixture
def round_in(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    return orchestrate.Round(REPO / "tools" / "server" / "plan3-smoke.json")


def test_const_opt_requirement(round_in, tmp_path):
    why = round_in.requirement("const-opt:phase1an")
    assert why and "const-opt-choice.json" in why
    (tmp_path / "phase1an").mkdir()
    (tmp_path / "phase1an" / "const-opt-choice.json").write_text(json.dumps({"choice": "lm"}))
    assert round_in.requirement("const-opt:phase1an") is None


def test_datasets_requirement_names_the_missing_ones(round_in, tmp_path):
    for a in ("Yac", "WiR"):
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
    why = round_in.requirement("datasets:Yac,Dow,Tow,WiR")
    assert why and "Dow, Tow" in why and "Yac" not in why
    assert round_in.requirement("datasets:Yac,WiR") is None
