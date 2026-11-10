"""Server round 10: the plans (plan10.json, plan10-smoke.json) and the configuration of 1bd, as
SERVER-ROUND10-PREREGISTRATION.md registers them: BASE against HC-LM, HC-LM-W and NOCAP on the
10 datasets, splits 1-10, BASE-S as a separate seeding control, an evaluation budget of 20,000
with no time limit, no extension, no choice step, no adoption rule and no chain."""

import csv
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
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402

# The arms of the pre-registration, flag for flag.
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
        "--const-opt", "lm", "--const-opt-guard", "val",
        "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
NOCAP = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
         "--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
         "--knob-probe-baseline", "reduced", "--linear-scaling", "1"]
HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
         "--const-opt", "lm", "--const-opt-guard", "val"]
HC_LM_W = HC_LM + ["--hc-widen-search", "1"]
ARMS = {"BASE": BASE, "BASE-S": BASE, "NOCAP": NOCAP, "HC-LM": HC_LM, "HC-LM-W": HC_LM_W}
SEED = {"binary": "../gomea-results/gpgomea-seed-build/gpgomea-seed", "share": 0.2,
        "height": 3, "k": 5, "functions": "+_-_*"}
DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
SPLITS = list(range(1, 11))
MANIFESTS = ["sr.json", "sr-Dow.json", "sr-Tow.json"]
MB = {"sr-Dow.json": 16384, "sr-Tow.json": 8192}  # "Memory"
OTHER_MB = 5120
PREFIXES = ["phase1bd", "server-smoke10-1bd"]


def load(path):
    return json.loads(path.read_text())


def stage1(q):
    """The stage-1 manifests of phase_run.sh: sr.json, then sr-*.json without sr-ext-*."""
    return [q / "sr.json"] + sorted(f for f in q.glob("sr-*.json")
                                    if not f.name.startswith("sr-ext-"))


def runs(ms):
    out = {}
    for f in ms:
        m = load(f)
        for d in m["datasets"]:
            for s in m["splits"]:
                assert (d, s) not in out, (f, d, s)
                out[(d, s)] = list(m["arms"])
    return out


# ---- the arms


@pytest.mark.parametrize("prefix", PREFIXES)
def test_arm_flags_match_the_preregistration(prefix):
    q = CONFIGS / prefix
    assert [f.name for f in stage1(q)] == MANIFESTS
    for f in stage1(q):
        m = load(f)
        assert list(m["arms"]) == list(ARMS)
        assert {a: s["argv"] for a, s in m["arms"].items()} == ARMS, f


def test_base_and_base_s_are_the_arms_of_1bb():
    bb = load(CONFIGS / "phase1bb" / "sr.json")["arms"]
    for f in stage1(CONFIGS / "phase1bd"):
        m = load(f)["arms"]
        assert m["BASE"] == bb["BASE"] and m["BASE-S"] == bb["BASE-S"]


def test_nocap_is_the_nocap_of_round_8_with_n_16():
    t = load(CONFIGS / "phase1ax" / "template.json")["arms"]["NOCAP"]["argv"]
    i = t.index("{POP}")
    assert NOCAP == t[:i] + ["--gomea-pop-size", "16"] + t[i + 1:]


def test_hc_lm_w_is_hc_lm_with_widen_search():
    assert HC_LM_W[:-2] == HC_LM and HC_LM_W[-2:] == ["--hc-widen-search", "1"]
    # HC-LM is the round-8 HC-LS with LM-VAL; no deme cap ("the same cap as GOMEA": none)
    hcls = load(CONFIGS / "phase1ax" / "template.json")["arms"]["HC-LS"]["argv"]
    assert HC_LM == hcls + ["--const-opt", "lm", "--const-opt-guard", "val"]
    for a in (HC_LM, HC_LM_W):
        assert not any("max-evals-per-deme" in x or "deme-share" in x for x in a)


def test_the_seed_block_is_only_on_base_s():
    for prefix in PREFIXES:
        for f in stage1(CONFIGS / prefix):
            for a, s in load(f)["arms"].items():
                if a == "BASE-S":
                    assert s["seed"] == SEED and set(s) == {"binary", "budget", "argv", "seed"}
                else:
                    assert set(s) == {"binary", "budget", "argv"}, (f, a)


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1bd"):
        m = load(f)
        assert m["wall_cap_s"] == 36000
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": 20000}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


@pytest.mark.parametrize("arm", ["BASE", "BASE-S", "HC-LM", "HC-LM-W"])
def test_the_runner_gives_the_lm_val_arms_the_validation_table_and_the_log(arm, tmp_path):
    """`_run_one` adds `--const-opt-val` for `--const-opt-guard val` (any optimizer), and
    `events_argv` adds `--gomea-log` for GOMEA and for `-a hc` with `--const-opt`."""
    argv = ARMS[arm]
    assert runner._const_opt_guard(argv) == "val"
    job = {"binary": ".venv/bin/moses", "seed": 1, "budget": {"evals": 20000}, "argv": argv}
    assert runner.moses_argv(job, tmp_path, 100)[-2:] == [
        "--gomea-log", str(tmp_path / "events.jsonl")]


def test_nocap_has_no_validation_table():
    assert runner._const_opt_guard(NOCAP) == "none" and runner._const_opt(NOCAP) == "none"


# ---- datasets, splits, memory


def test_datasets_and_splits_and_no_extension():
    q = CONFIGS / "phase1bd"
    assert set(runs(stage1(q))) == {(d, s) for d in DATASETS for s in SPLITS}
    assert sorted(q.glob("*-ext-*.json")) == []
    assert load(q / "sr.json")["datasets"] == [d for d in DATASETS if d not in ("Dow", "Tow")]
    assert load(q / "sr-Dow.json")["datasets"] == ["Dow"]
    assert load(q / "sr-Tow.json")["datasets"] == ["Tow"]
    for f in stage1(q):
        assert load(f)["name"] == f"phase1bd-{f.stem}"


def test_run_count_is_500():
    assert sum(len(a) for a in runs(stage1(CONFIGS / "phase1bd")).values()) == 500


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == MB.get(f.name, OTHER_MB), f


# ---- the spec


PRIMARY = [["base-vs-hclm", "BASE", "HC-LM"], ["base-vs-hclmw", "BASE", "HC-LM-W"],
           ["base-vs-nocap", "BASE", "NOCAP"]]
CONTROL = [["seeded-vs-base", "BASE-S", "BASE"], ["seeded-vs-hclm", "BASE-S", "HC-LM"],
           ["seeded-vs-hclmw", "BASE-S", "HC-LM-W"], ["seeded-vs-nocap", "BASE-S", "NOCAP"]]
WIDEN = [["hclmw-vs-hclm", "HC-LM-W", "HC-LM"]]
SECONDARY_PAIRS = PRIMARY + CONTROL[:1] + WIDEN


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bd" if prefix == "phase1bd" else "smoke10-1bd")
    assert s["metrics"] == dict.fromkeys(DATASETS, "test_nmse")
    assert s["primary"] == []
    assert s["pooled_primary"] == [c + ["test_nmse"] for c in PRIMARY]
    want2 = ([c + ["test_nmse"] for c in CONTROL + WIDEN]
             + [[f"{n}-size", a, b, "program_size"] for n, a, b in SECONDARY_PAIRS]
             + [[f"{n}-cpu", a, b, "cpu_s"] for n, a, b in SECONDARY_PAIRS])
    assert s["pooled_secondary"] == want2 and len(want2) == 4 + 1 + 10
    assert s["pooled_secondary_families"] == [[c[0] for c in CONTROL], ["hclmw-vs-hclm"]]
    per = [[d, n, a, b, "test_nmse"] for n, a, b in PRIMARY + CONTROL[:1] for d in DATASETS]
    assert s["secondary"] == per and len(per) == 40
    assert s["secondary_verdicts"] is False and s["extend"] is False
    assert "secondary_families" not in s and "descriptive" not in s


def test_holm_families_at_100_pairs():
    """With 100 pooled pairs every family can give a verdict; the 40 per-dataset contrasts
    (10 pairs each) cannot (40 * 2/1024 > 0.025): they are descriptive."""
    p100 = wilcoxon_signed_rank([float(i + 1) for i in range(100)])[1]
    for n in (3, 4, 1, 10):
        assert holm([p100] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 40)[0] > phase_report.ALPHA


def _summary(task, arm, split, nmse, status="ok"):
    return {"arm": arm, "task": task, "seed": split, "status": status,
            "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse, "cpu_s": 10.0,
            "n_evals": 20000, "final_program": "(+ $x1 $x2)"}


def test_the_report_of_the_real_spec(tmp_path):
    """phase_report.py on the 1bd spec with 500 made-up runs: 3 + 15 pooled rows and 40
    per-dataset rows, the four Holm families, verdicts only on the pooled rows."""
    factor = {"BASE": 1.0, "BASE-S": 1.0, "NOCAP": 2.0, "HC-LM": 1.5, "HC-LM-W": 1.5}
    root = tmp_path / "r"
    for d in DATASETS:
        for k in SPLITS:
            for arm, f in factor.items():
                failed = arm == "NOCAP" and d == "Dow" and k == 1  # a memory failure
                s = _summary(d, arm, k, f * (0.1 + k / 100 + DATASETS.index(d) / 1000),
                             "error" if failed else "ok")
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1bd" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    pooled = [r for r in rows if r["problem"] == "pooled"]
    assert len(pooled) == 18 and len(rows) == 58
    assert all(r["pairs"] == "100" for r in pooled)
    fam = {r["contrast"]: (r["family"], r["holm_family"]) for r in pooled}
    assert fam["base-vs-nocap"] == ("primary", "")
    assert fam["seeded-vs-nocap"] == ("secondary", "pooled_secondary/" + "+".join(
        c[0] for c in CONTROL))
    assert fam["hclmw-vs-hclm"] == ("secondary", "pooled_secondary/hclmw-vs-hclm")
    assert fam["base-vs-hclm-cpu"] == ("secondary", "pooled_secondary")
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:base-vs-hclm"] == "A better" and v["pooled:base-vs-nocap"] == "A better"
    assert v["pooled:seeded-vs-base"] == "tie" and v["pooled:hclmw-vs-hclm"] == "tie"
    assert all(v[f"{d}:base-vs-hclm"] is None for d in DATASETS)
    assert sum(r["verdict"] == "" for r in rows) == 40
    assert not any(x == "extend" for x in v.values())


# ---- the plans


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan10.json", "server-round10-2026-11", "server-run10", "phase1bd", "phase1bd"),
    ("plan10-smoke.json", "server-round10-smoke", "server-run10-smoke", "server-smoke10-1bd",
     "smoke10-1bd"),
])
def test_plan_is_s0_then_1bd(plan, name, state_dir, prefix, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", sid]
    st = p["steps"][1]
    assert st["phase"] == prefix
    assert "config_dir" not in st and not st.get("pilot") and not st.get("set_commit_in")
    assert st["requires"] == ["gpgomea", f"datasets:{','.join(DATASETS)}"]
    assert st["per_run_mb"] == OTHER_MB and st["manifest_per_run_mb"] == MB
    assert "per_run_mb" not in p["steps"][0] and "manifest_per_run_mb" not in p["steps"][0]
    for f in stage1(CONFIGS / prefix):
        assert st["manifest_per_run_mb"].get(f.name, st["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((CONFIGS / prefix / n).exists() for n in MB)
    assert "then" not in p


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan10.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    args = load(SERVER / "plan10-smoke.json")["steps"][0]["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_phase_report_round10.py",
              "tests/gomea/test_hc_lm.py", "tests/gomea/test_hc_lm_identity.py"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t


def test_no_plan_chains_to_or_from_round_10():
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round10", "smoke10"), f
    for f in ("plan10.json", "plan10-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan", [("round10", "plan10.json"),
                                        ("smoke10", "plan10-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan
    assert [f.name for f in chain.forward(SERVER / plan)] == [plan]


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round10|smoke10" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bd", CONFIGS / "server-smoke10-1bd"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1bd", "server-smoke10-1bd")
        assert m["datasets"] == fm["datasets"] and m["splits"] == [1, 2]
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert s["argv"] == fm["arms"][a]["argv"]
            assert s.get("seed") == fm["arms"][a].get("seed")
            # GP-GOMEA's first generation alone costs about 600 evaluations
            assert s["budget"] == {"evals": 1000}


# ---- W for each manifest


def _server(tmp_path, cores=12, mem_mb=65536):
    """A fake /sys and /proc: `cores` physical cores, a cgroup memory limit of `mem_mb`."""
    sys.path.insert(0, str(SERVER))
    import resources
    s, pr = tmp_path / "sys", tmp_path / "proc"
    for c in range(cores):
        t = s / f"devices/system/cpu/cpu{c}/topology"
        t.mkdir(parents=True)
        (t / "core_id").write_text(f"{c}\n")
        (t / "physical_package_id").write_text("0\n")
    (s / "fs/cgroup").mkdir(parents=True)
    (s / "fs/cgroup/memory.max").write_text(f"{mem_mb * 2**20}\n")
    pr.mkdir()
    (pr / "meminfo").write_text(f"MemTotal: {4 * mem_mb * 1024} kB\n")
    (pr / "self").mkdir()
    (pr / "self/cgroup").write_text("0::/\n")
    return lambda mb: resources.decide(s, pr, affinity=set(range(cores)),
                                       env={"PER_RUN_MB": str(mb)})


def test_server_workers_are_3_for_dow_7_for_tow_and_11_otherwise(tmp_path):
    """The table "Memory" of the pre-registration."""
    d = _server(tmp_path)
    assert (d(16384)["workers"], d(16384)["limited_by"]) == (3, "memory")
    assert (d(8192)["workers"], d(8192)["limited_by"]) == (7, "memory")
    assert (d(OTHER_MB)["workers"], d(OTHER_MB)["limited_by"]) == (11, "cpu")


FAKE_SR = r"""import json, os, sys
from pathlib import Path
w = sys.argv[sys.argv.index("--workers") + 1]
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"manifest": Path(sys.argv[1]).name, "workers": int(w)}) + "\n")
"""
FAKE_REPORT = r"""import json, sys
from pathlib import Path
out = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
(out / "verdict.json").write_text("{}")
"""


def _w(mb):
    sys.path.insert(0, str(SERVER))
    import resources
    return resources.decide(env={"PER_RUN_MB": str(mb)})["workers"]


def test_dow_and_tow_manifests_run_with_their_own_workers(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bd configs; the runner and the
    report are fakes."""
    repo, res = tmp_path / "pyMoses", tmp_path / "gomea-results"
    (repo / "tools" / "server").mkdir(parents=True)
    res.mkdir()
    for f in SERVER.iterdir():
        if f.suffix in (".py", ".sh"):
            shutil.copy(f, repo / "tools" / "server" / f.name)
    g = repo / "tools" / "gomea"
    g.mkdir()
    shutil.copy(REPO / "tools" / "gomea" / "phase_run.sh", g / "phase_run.sh")
    (g / "moses_sr_runner.py").write_text(FAKE_SR)
    (g / "phase_report.py").write_text(FAKE_REPORT)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
    shutil.copytree(CONFIGS / "phase1bd", repo / "configs" / "gomea" / "phase1bd")
    step = dict(load(SERVER / "plan10.json")["steps"][1])
    step.pop("requires")
    plan = {"name": "t10", "state_dir": "server-run10-t", "steps": [step]}
    (repo / "tools" / "server" / "plan-t10.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="3072", MANIFEST_WORKERS="sr.json=99")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t10.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    assert [t["manifest"] for t in trace] == MANIFESTS
    for t in trace:
        assert t["workers"] == _w(MB.get(t["manifest"], OTHER_MB)), t
    sd = res / "server-run10-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [e for e in events if e["event"] == "workers_decision"]
    assert [(e.get("manifest"), e["per_run_mb"]) for e in wd] == [
        (None, OTHER_MB), ("sr-Dow.json", 16384), ("sr-Tow.json", 8192)]
    att = [e for e in events if e["event"] == "phase_attempt"]
    assert att[0]["manifest_workers"] == {n: _w(mb) for n, mb in MB.items()}
    assert load(sd / "state.json")["steps"]["phase1bd"]["status"] == "done"
