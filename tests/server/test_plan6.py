"""Server round 6: the plans (plan6.json, plan6-smoke.json) and the configurations of 1aw, as
SERVER-ROUND6-PREREGISTRATION.md registers them."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"

# The base flags of the pre-registration: the base of 1at without the per-deme cap.
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
        "--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
CAP = ["--gomea-max-evals-per-deme", "250"]
LMVAL = ["--const-opt", "lm", "--const-opt-guard", "val"]
PEN125 = ["--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
# LMVAL-C250 is the LM-VAL arm of 1at: the cap stays at its place in the 1at argv.
C250 = [*BASE[:8], *CAP, *BASE[8:], *LMVAL]
ARMS = {
    "NOCAP": BASE,
    "LMVAL": [*BASE, *LMVAL],
    "LMVAL-PEN125": [*BASE, *LMVAL, *PEN125],
    "LMVAL-C250": C250,
}
D6 = ["WiR", "Con", "EnH", "Air", "Yac", "Tow"]
PRIMARY = ([(d, "LMVAL", "NOCAP", "test_nmse") for d in D6]
           + [(d, "LMVAL-PEN125", "LMVAL", "program_size") for d in D6])
SECONDARY = ([(d, "LMVAL", "LMVAL-C250", "test_nmse") for d in D6]
             + [(d, "LMVAL-PEN125", "LMVAL", "test_nmse") for d in D6])
PREFIXES = ["phase", "server-smoke6-"]


def load(path):
    return json.loads(path.read_text())


def manifests(prefix):
    q = CONFIGS / f"{prefix}1aw"
    return [q / "sr.json", *sorted(q.glob("sr-ext-*.json"))]


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_lmval_c250_is_the_lm_val_arm_of_1at():
    assert load(CONFIGS / "phase1at" / "sr.json")["arms"]["LM-VAL"]["argv"] == C250


def test_nocap_is_the_gomea_ls_arm_of_1at_without_the_cap():
    a = load(CONFIGS / "phase1at" / "sr.json")["arms"]["GOMEA-LS"]["argv"]
    i = a.index("--gomea-max-evals-per-deme")
    assert a[:i] + a[i + 2:] == BASE


def test_stage_1_manifest():
    m = load(CONFIGS / "phase1aw" / "sr.json")
    assert m["name"] == "phase1aw-sr"
    assert m["datasets"] == D6 and m["splits"] == list(range(1, 11))
    assert m["wall_cap_s"] == 2400 and m["mem_limit_mb"] == 3072
    assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
    assert list(m["arms"]) == list(ARMS)
    for arm, spec in m["arms"].items():
        assert spec["argv"] == ARMS[arm], arm
        assert spec["budget"] == {"time_s": 1000} and spec["binary"] == ".venv/bin/moses"
        assert set(spec) == {"binary", "budget", "argv"}, arm


def test_extensions_hold_all_arms_on_splits_11_to_20():
    q = CONFIGS / "phase1aw"
    stage1 = load(q / "sr.json")
    assert {f.name for f in q.glob("sr-ext-*.json")} == {f"sr-ext-{d}.json" for d in D6}
    for d in D6:
        m = load(q / f"sr-ext-{d}.json")
        assert m["name"] == f"phase1aw-sr-ext-{d}"
        assert m["datasets"] == [d] and m["splits"] == list(range(11, 21))
        assert m["arms"] == stage1["arms"]
        for k in ("wall_cap_s", "mem_limit_mb", "datasets_dir"):
            assert m[k] == stage1[k], k


@pytest.mark.parametrize("prefix", PREFIXES)
def test_the_cap_is_only_in_lmval_c250(prefix):
    for f in manifests(prefix):
        for arm, spec in load(f)["arms"].items():
            argv = spec["argv"]
            assert not any(t.startswith("--gomea-max-evals-per-deme=") for t in argv)
            if arm == "LMVAL-C250":
                assert argv.count("--gomea-max-evals-per-deme") == 1
                assert flag_value(argv, "--gomea-max-evals-per-deme") == "250"
            else:
                assert "--gomea-max-evals-per-deme" not in argv, (f.name, arm)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_lm_and_the_guard_are_in_the_lmval_arms_only(prefix):
    for f in manifests(prefix):
        for arm, spec in load(f)["arms"].items():
            argv = spec["argv"]
            if arm == "NOCAP":
                assert "--const-opt" not in argv and "--const-opt-guard" not in argv
            else:
                assert flag_value(argv, "--const-opt") == "lm", (f.name, arm)
                assert flag_value(argv, "--const-opt-guard") == "val", (f.name, arm)
            # the runner adds the validation file itself; a manifest never names one
            assert "--const-opt-val" not in argv


@pytest.mark.parametrize("prefix", PREFIXES)
def test_the_penalty_is_only_in_lmval_pen125(prefix):
    for f in manifests(prefix):
        for arm, spec in load(f)["arms"].items():
            argv = spec["argv"]
            if arm == "LMVAL-PEN125":
                assert flag_value(argv, "--gomea-accept-complexity-ratio") == "0.125"
                assert flag_value(argv, "--gomea-deme-store") == "accepted"
            else:
                assert "--gomea-accept-complexity-ratio" not in argv, (f.name, arm)
                assert "--gomea-deme-store" not in argv, (f.name, arm)


def test_the_runner_gives_the_validation_file_to_the_lmval_arms(tmp_path):
    """moses_sr_runner adds `--const-opt-val` for an arm with `--const-opt-guard val`."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "gomea"))
    import moses_sr_runner as r
    for arm, argv in ARMS.items():
        assert (r._const_opt_guard(argv) == "val") == (arm != "NOCAP"), arm
        assert r._const_opt(argv) == ("none" if arm == "NOCAP" else "lm"), arm


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec_has_12_primary_and_12_secondary_contrasts(prefix):
    s = load(CONFIGS / f"{prefix}1aw" / "spec.json")
    assert s["phase"] == ("1aw" if prefix == "phase" else "smoke6-1aw")
    assert s["metrics"] == dict.fromkeys(D6, "test_nmse")
    assert [(p, a, b, m) for p, _, a, b, m in s["primary"]] == PRIMARY
    assert [(p, a, b, m) for p, _, a, b, m in s["secondary"]] == SECONDARY
    assert "descriptive" not in s
    names = [(c[0], c[1]) for c in s["primary"] + s["secondary"]]
    assert len(set(names)) == len(names) == 24


def test_smoke_spec_has_the_same_contrasts():
    a = load(CONFIGS / "phase1aw" / "spec.json")
    b = load(CONFIGS / "server-smoke6-1aw" / "spec.json")
    assert {k: v for k, v in a.items() if k != "phase"} == {
        k: v for k, v in b.items() if k != "phase"}


def test_stage_1_can_give_a_verdict():
    """10 pairs: the lowest raw p is 0.00195, so Holm over the 12 primary contrasts stays
    below 0.025 (12 x 0.00195 = 0.0234)."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert len(load(CONFIGS / "phase1aw" / "spec.json")["primary"]) == 12
    assert holm([p] * 12)[0] < 0.025


def _summary(task, arm, seed, nmse, size):
    prog = "(" + " ".join(["x"] * (size - 1)) + ")" if size > 1 else "x"
    return {"task": task, "arm": arm, "seed": seed, "status": "ok", "test_nmse": nmse,
            "train_nmse": nmse, "final_program": prog}


def test_report_takes_two_metrics_in_one_holm_family(tmp_path):
    """phase_report: the NMSE and size contrasts of 1aw form one Holm family of 12, each with
    its own metric."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "gomea"))
    import phase_report
    runs = tmp_path / "res" / "runs"
    n = 0
    for d in D6:
        for k in range(1, 11):
            # LMVAL: 30 % lower NMSE than NOCAP on WiR only; PEN125: 40 % smaller on Con only
            vals = {"NOCAP": (1.0 * k, 21), "LMVAL": ((0.7 if d == "WiR" else 1.0) * k, 21),
                    "LMVAL-PEN125": ((0.7 if d == "WiR" else 1.0) * k, 13 if d == "Con" else 21),
                    "LMVAL-C250": (1.0 * k, 21)}
            for arm, (nmse, size) in vals.items():
                n += 1
                (runs / str(n)).mkdir(parents=True)
                (runs / str(n) / "summary.json").write_text(
                    json.dumps(_summary(d, arm, k, nmse, size)))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1aw" / "spec.json"), str(out),
                            str(tmp_path / "res")])
    assert rc == 0
    import csv
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    prim = [r for r in rows if r["family"] == "primary"]
    assert len(prim) == 12 and len([r for r in rows if r["family"] == "secondary"]) == 12
    assert {r["metric"] for r in prim if r["a"] == "LMVAL"} == {"test_nmse"}
    assert {r["metric"] for r in prim if r["a"] == "LMVAL-PEN125"} == {"program_size"}
    by = {(r["problem"], r["contrast"]): r for r in rows}
    s = load(CONFIGS / "phase1aw" / "spec.json")
    nm, sz = s["primary"][0][1], s["primary"][6][1]
    # each p is 0.00195 raw: Holm over 12 gives 0.0234 for the two real effects
    assert by[("WiR", nm)]["verdict"] == "A better"
    assert by[("Con", sz)]["verdict"] == "A better"
    assert float(by[("WiR", nm)]["p_holm"]) == pytest.approx(12 * 0.001953125)
    assert by[("Con", nm)]["verdict"] == "tie" and by[("WiR", sz)]["verdict"] == "tie"
    v = load(out / "verdict.json")
    assert len(v) == 24


def test_phase_run_extends_a_dataset_on_either_primary_metric(tmp_path):
    """phase_run.sh runs sr-ext-<dataset>.json (all four arms) for each dataset with an
    "extend" primary contrast, of either metric; a secondary "extend" never extends."""
    text = (REPO / "tools" / "gomea" / "phase_run.sh").read_text()
    code = text.split("EXTP=$($PY -c '", 1)[1].split("' \"$Q/spec.json\"", 1)[0]
    s = load(CONFIGS / "phase1aw" / "spec.json")
    v = {f"{p}:{n}": "tie" for p, n, *_ in s["primary"] + s["secondary"]}
    v["Con:lmval-vs-nocap"] = "extend"
    v["Yac:pen125-size"] = "extend"
    v["Tow:lmval-vs-c250"] = "extend"
    vf = tmp_path / "verdict.json"
    vf.write_text(json.dumps(v))
    r = subprocess.run(["python3", "-c", code, str(CONFIGS / "phase1aw" / "spec.json"), str(vf)],
                       capture_output=True, text=True, check=True)
    assert r.stdout.split() == ["Con", "Yac"]


def test_registered_run_counts():
    m = load(CONFIGS / "phase1aw" / "sr.json")
    n = len(m["arms"]) * len(m["datasets"]) * len(m["splits"])
    assert n == 240
    ext = sum(len(load(f)["arms"]) * len(load(f)["datasets"]) * len(load(f)["splits"])
              for f in (CONFIGS / "phase1aw").glob("sr-ext-*.json"))
    assert ext == 240


@pytest.mark.parametrize("prefix", PREFIXES)
def test_no_boolean_manifest_so_no_commit_to_set(prefix):
    assert not list((CONFIGS / f"{prefix}1aw").glob("bool*.json"))
    assert not list((CONFIGS / f"{prefix}1aw").glob("gp*.json"))


@pytest.mark.parametrize("plan, name, state_dir, phase, sid, then", [
    ("plan6.json", "server-round6-2026-11", "server-run6", "phase1aw", "phase1aw", "round7"),
    ("plan6-smoke.json", "server-round6-smoke", "server-run6-smoke", "server-smoke6-1aw",
     "smoke6-1aw", "smoke7"),
])
def test_plan_is_s0_then_1aw(plan, name, state_dir, phase, sid, then):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", sid]
    s = p["steps"][1]
    assert s["phase"] == phase
    assert s["requires"] == [f"datasets:{','.join(D6)}"]
    assert not s.get("pilot") and not s.get("set_commit_in") and not s.get("after_pilot")
    assert (CONFIGS / phase / "spec.json").exists() and (CONFIGS / phase / "sr.json").exists()
    assert p["then"] == then  # round 7, the confirmation round, follows without a pause


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan6.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_round_6_tests():
    s0 = load(SERVER / "plan6-smoke.json")["steps"][0]
    assert "tests/server" in s0["pytest"]["args"]
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


def test_no_plan_chains_to_round_6():
    """Round 6 starts by itself; a later round may follow, but no "then" is set yet."""
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round6", "smoke6"), f.name


@pytest.mark.parametrize("name, plan", [("round6", "plan6.json"), ("smoke6", "plan6-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    import sys
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round6|smoke6" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    """Tow is a download, the others come from data/gomea-sr."""
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(D6) - {"Tow"} <= set(built)
    assert '"Tow tower ' in text


def test_smoke_has_the_same_arms_and_is_small():
    q = CONFIGS / "server-smoke6-1aw"
    full = load(CONFIGS / "phase1aw" / "sr.json")
    m = load(q / "sr.json")
    assert m["name"] == "server-smoke6-1aw-sr"
    assert m["datasets"] == D6 and m["splits"] == [1, 2]
    assert m["wall_cap_s"] <= 300 and m["datasets_dir"] == full["datasets_dir"]
    assert {f.name for f in q.glob("sr-ext-*.json")} == {f"sr-ext-{d}.json" for d in D6}
    for f in manifests("server-smoke6-"):
        m = load(f)
        assert list(m["arms"]) == list(full["arms"])
        for arm, spec in m["arms"].items():
            assert spec["argv"] == full["arms"][arm]["argv"]
            assert spec["budget"] == {"time_s": 15} and spec["binary"] == ".venv/bin/moses"
    for d in D6:
        m = load(q / f"sr-ext-{d}.json")
        assert m["name"] == f"server-smoke6-1aw-sr-ext-{d}"
        assert m["datasets"] == [d] and m["splits"] == [3, 4]
