"""Server round 11: the plans (plan11.json, plan11-smoke.json) and the configuration of 1be, as
SERVER-ROUND11-PREREGISTRATION.md registers them: BASE against BASE-R (LM reserve 500),
BASE-D5, BASE-D10 and BASE-D20 (deme caps), with HC-LM as the accuracy baseline, on the 10
datasets, fresh splits 31-40, an evaluation budget of 20,000 with no time limit, no
extension, no choice step, no adoption rule and no chain."""

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
import sr_datasets  # noqa: E402

# The arms of the pre-registration, flag for flag. BASE and HC-LM are the round 10 arms.
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
        "--const-opt", "lm", "--const-opt-guard", "val",
        "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
         "--const-opt", "lm", "--const-opt-guard", "val"]
ARMS = {
    "BASE": BASE,
    "BASE-R": BASE + ["--const-opt-reserve", "500"],
    "BASE-D5": BASE + ["--gomea-deme-share", "5"],
    "BASE-D10": BASE + ["--gomea-deme-share", "10"],
    "BASE-D20": BASE + ["--gomea-deme-share", "20"],
    "HC-LM": HC_LM,
}
DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
SPLITS = list(range(31, 41))
MANIFESTS = ["sr.json", "sr-Dow.json"]
MB = {"sr-Dow.json": 8192}  # "Memory"
OTHER_MB = 5120
PREFIXES = ["phase1be", "server-smoke11-1be"]
REQUIRES = [f"datasets:{','.join(DATASETS)}:31-40"]


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
        for s in m["arms"].values():
            assert set(s) == {"binary", "budget", "argv"}  # no seed block


def test_base_and_hc_lm_are_the_arms_of_1bd():
    bd = load(CONFIGS / "phase1bd" / "sr.json")["arms"]
    for f in stage1(CONFIGS / "phase1be"):
        m = load(f)["arms"]
        assert m["BASE"] == bd["BASE"] and m["HC-LM"] == bd["HC-LM"]
    for f in stage1(CONFIGS / "phase1bd"):
        assert load(f)["arms"]["BASE"]["argv"] == BASE
        assert load(f)["arms"]["HC-LM"]["argv"] == HC_LM


@pytest.mark.parametrize("arm, cap, reserve", [
    ("BASE", None, 0), ("BASE-R", None, 500), ("BASE-D5", 4000, 0), ("BASE-D10", 2000, 0),
    ("BASE-D20", 1000, 0)])
def test_the_cli_reads_each_gomea_arm(arm, cap, reserve):
    """The flags parse as the runner passes them (-m 20000, the validation table): BASE-R has
    the reserve, the D arms the caps 4,000, 2,000 and 1,000."""
    from pymoses.moses.main.problem_params import _gomea_params_from_args
    from tests.gomea.test_integration import _cli_parser

    argv = ["-m", "20000"] + ARMS[arm] + ["--const-opt-val", "val.csv"]
    gp = _gomea_params_from_args(_cli_parser(True).parse_args(argv))
    assert gp.max_evals_per_deme == cap
    assert gp.const_opt_reserve == reserve
    assert gp.const_opt == "lm" and gp.const_opt_guard == "val" and gp.dynamic()
    assert gp.deme_share == (20000 // cap if cap else 0)


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1be"):
        m = load(f)
        assert m["wall_cap_s"] == 36000
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": 20000}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_runner_gives_each_arm_m_the_validation_table_and_the_log(arm, tmp_path):
    """`-m 20000` comes before the arm's flags (the deme share reads it); `_run_one` adds
    `--const-opt-val` for `--const-opt-guard val`, and `events_argv` adds `--gomea-log`."""
    argv = ARMS[arm]
    assert runner._const_opt_guard(argv) == "val" and runner._const_opt(argv) == "lm"
    job = {"binary": ".venv/bin/moses", "seed": 31, "budget": {"evals": 20000}, "argv": argv}
    out = runner.moses_argv(job, tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    i = out.index("-m")
    assert out[i + 1] == "20000" and out[i + 2:-2] == argv


# ---- datasets, splits, memory


def test_datasets_and_fresh_splits_and_no_extension():
    q = CONFIGS / "phase1be"
    assert set(runs(stage1(q))) == {(d, s) for d in DATASETS for s in SPLITS}
    assert sorted(q.glob("*-ext-*.json")) == []
    assert load(q / "sr.json")["datasets"] == [d for d in DATASETS if d != "Dow"]
    assert load(q / "sr-Dow.json")["datasets"] == ["Dow"]
    for f in stage1(q):
        assert load(f)["name"] == f"phase1be-{f.stem}"
        assert load(f)["splits"] == SPLITS
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_run_count_is_600():
    assert sum(len(a) for a in runs(stage1(CONFIGS / "phase1be")).values()) == 600


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == MB.get(f.name, OTHER_MB), f


# ---- the spec


PRIMARY = [["baser-vs-base", "BASE-R", "BASE"], ["based5-vs-base", "BASE-D5", "BASE"],
           ["based10-vs-base", "BASE-D10", "BASE"], ["based20-vs-base", "BASE-D20", "BASE"]]
GAP = [["base-vs-hclm", "BASE", "HC-LM"], ["baser-vs-hclm", "BASE-R", "HC-LM"],
       ["based5-vs-hclm", "BASE-D5", "HC-LM"], ["based10-vs-hclm", "BASE-D10", "HC-LM"],
       ["based20-vs-hclm", "BASE-D20", "HC-LM"]]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1be" if prefix == "phase1be" else "smoke11-1be")
    assert s["metrics"] == dict.fromkeys(DATASETS, "test_nmse")
    assert s["primary"] == []
    assert s["pooled_primary"] == [c + ["test_nmse"] for c in PRIMARY]
    want2 = ([c + ["test_nmse"] for c in GAP]
             + [[f"{n}-size", a, b, "program_size"] for n, a, b in PRIMARY]
             + [[f"{n}-cpu", a, b, "cpu_s"] for n, a, b in PRIMARY])
    assert s["pooled_secondary"] == want2 and len(want2) == 5 + 8
    assert s["pooled_secondary_families"] == [[c[0] for c in GAP]]
    per = [[d, n, a, b, "test_nmse"] for n, a, b in PRIMARY for d in DATASETS]
    assert s["secondary"] == per and len(per) == 40
    assert s["secondary_verdicts"] is False and s["extend"] is False
    assert "secondary_families" not in s and "descriptive" not in s


def test_holm_families_at_100_pairs():
    """With 100 pooled pairs every family (4, 5, 8) can give a verdict; the 40 per-dataset
    contrasts (10 pairs each) cannot (40 * 2/1024 > 0.025): they are descriptive."""
    p100 = wilcoxon_signed_rank([float(i + 1) for i in range(100)])[1]
    for n in (4, 5, 8):
        assert holm([p100] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 40)[0] > phase_report.ALPHA


def _summary(task, arm, split, nmse, status="ok"):
    return {"arm": arm, "task": task, "seed": split, "status": status,
            "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse, "cpu_s": 10.0,
            "n_evals": 20000, "final_program": "(+ $x1 $x2)",
            "const_opt": {"mode": "lm", "lm_calls": 2, "tuning_evals": 100,
                          "tuning_share": 0.005}}


def test_the_report_of_the_real_spec(tmp_path):
    """phase_report.py on the 1be spec with 600 made-up runs: 4 + 13 pooled rows and 40
    per-dataset rows, the three Holm families, verdicts only on the pooled rows. The
    aggregate has the columns of the trade-off table that it already computes per dataset and
    arm (demes in a run, the LM share of the evaluations, the program size)."""
    factor = {"BASE": 1.0, "BASE-R": 1.0, "BASE-D5": 0.7, "BASE-D10": 0.7, "BASE-D20": 0.7,
              "HC-LM": 0.5}
    root = tmp_path / "r"
    end = json.dumps({"type": "end", "generations": 4, "termination_reason": "stagnation",
                      "ledger": {"requests": 10}})
    for d in DATASETS:
        for k in SPLITS:
            for arm, f in factor.items():
                failed = arm == "BASE-D20" and d == "Dow" and k == 31  # a memory failure
                s = _summary(d, arm, k, f * (0.1 + k / 1000 + DATASETS.index(d) / 1000),
                             "error" if failed else "ok")
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))
                (rd / "events.jsonl").write_text((end + "\n") * (3 if f < 1 else 1))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1be" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    pooled = [r for r in rows if r["problem"] == "pooled"]
    assert len(pooled) == 17 and len(rows) == 57
    assert all(r["pairs"] == "100" for r in pooled)
    fam = {r["contrast"]: (r["family"], r["holm_family"]) for r in pooled}
    assert fam["based20-vs-base"] == ("primary", "")
    assert fam["based20-vs-hclm"] == ("secondary", "pooled_secondary/" + "+".join(
        c[0] for c in GAP))
    assert fam["baser-vs-base-cpu"] == ("secondary", "pooled_secondary")
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:baser-vs-base"] == "tie" and v["pooled:based5-vs-base"] == "A better"
    assert v["pooled:base-vs-hclm"] == "A worse"
    assert all(v[f"{d}:based10-vs-base"] is None for d in DATASETS)
    assert sum(r["verdict"] == "" for r in rows) == 40
    agg = list(csv.DictReader((out / "aggregate.csv").open()))
    assert len(agg) == 60
    for col in ("median_deme_count", "median_tuning_share", "median_tuning_evals",
                "median_size", "median_lm_calls"):
        assert col in agg[0], col
    d5 = next(r for r in agg if r["problem"] == "Air" and r["arm"] == "BASE-D5")
    assert float(d5["median_deme_count"]) == 3 and float(d5["median_tuning_share"]) == 0.005


# ---- the plans


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan11.json", "server-round11-2026-11", "server-run11", "phase1be", "phase1be"),
    ("plan11-smoke.json", "server-round11-smoke", "server-run11-smoke", "server-smoke11-1be",
     "smoke11-1be"),
])
def test_plan_is_s0_then_1be(plan, name, state_dir, prefix, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", sid]
    st = p["steps"][1]
    assert st["phase"] == prefix
    assert "config_dir" not in st and not st.get("pilot") and not st.get("set_commit_in")
    # no seeded arm, so no GP-GOMEA build; the datasets with the fresh splits 31-40
    assert st["requires"] == REQUIRES
    assert st["per_run_mb"] == OTHER_MB and st["manifest_per_run_mb"] == MB
    assert "per_run_mb" not in p["steps"][0] and "manifest_per_run_mb" not in p["steps"][0]
    for f in stage1(CONFIGS / prefix):
        assert st["manifest_per_run_mb"].get(f.name, st["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((CONFIGS / prefix / n).exists() for n in MB)
    assert "then" not in p


def test_the_requirement_needs_the_fresh_splits(tmp_path, monkeypatch):
    """A dataset set built before round 11 (splits 1-30) does not meet the requirement."""
    sys.path.insert(0, str(SERVER))
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan11.json")
    for a in DATASETS:
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
        for k in range(1, 31):
            (d / "splits" / f"s{k}").mkdir()
            for t in ("train", "val", "test"):
                (d / "splits" / f"s{k}" / f"{t}.csv").write_text("x1,y\n")
    why = r.requirement(REQUIRES[0])
    assert why and "31-40" in why
    for a in DATASETS:
        for k in SPLITS:
            s = tmp_path / "phase1f" / "datasets" / a / "splits" / f"s{k}"
            s.mkdir()
            for t in ("train", "val", "test"):
                (s / f"{t}.csv").write_text("x1,y\n")
    assert r.requirement(REQUIRES[0]) is None


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan11.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    args = load(SERVER / "plan11-smoke.json")["steps"][0]["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_const_opt_reserve.py",
              "tests/gomea/test_const_opt_reserve_identity.py",
              "tests/gomea/test_pop_size_dynamic.py", "tests/gomea/test_sr_datasets.py"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t


def test_no_plan_chains_to_or_from_round_11():
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round11", "smoke11"), f
    for f in ("plan11.json", "plan11-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan", [("round11", "plan11.json"),
                                        ("smoke11", "plan11-smoke.json")])
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
    assert "round11|smoke11" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan_with_the_fresh_splits():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text
    assert f'-eq {8 * len(sr_datasets.SEEDS)} ]' in text and max(sr_datasets.SEEDS) == 40


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1be", CONFIGS / "server-smoke11-1be"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1be", "server-smoke11-1be")
        assert m["datasets"] == fm["datasets"] and m["splits"] == [31, 32]
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert m["datasets_dir"] == fm["datasets_dir"]
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert s["argv"] == fm["arms"][a]["argv"]
            # -m 1000: the reserve of BASE-R (500) and the caps of the D arms apply
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


def test_server_workers_are_7_for_dow_and_11_otherwise(tmp_path):
    """The table "Memory" of the pre-registration."""
    d = _server(tmp_path)
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


def test_the_dow_manifest_runs_with_its_own_workers(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1be configs; the runner and the
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
    shutil.copytree(CONFIGS / "phase1be", repo / "configs" / "gomea" / "phase1be")
    step = dict(load(SERVER / "plan11.json")["steps"][1])
    step.pop("requires")
    plan = {"name": "t11", "state_dir": "server-run11-t", "steps": [step]}
    (repo / "tools" / "server" / "plan-t11.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="3072", MANIFEST_WORKERS="sr.json=99")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t11.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    assert [t["manifest"] for t in trace] == MANIFESTS
    for t in trace:
        assert t["workers"] == _w(MB.get(t["manifest"], OTHER_MB)), t
    sd = res / "server-run11-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [e for e in events if e["event"] == "workers_decision"]
    assert [(e.get("manifest"), e["per_run_mb"]) for e in wd] == [
        (None, OTHER_MB), ("sr-Dow.json", 8192)]
    att = [e for e in events if e["event"] == "phase_attempt"]
    assert att[0]["manifest_workers"] == {n: _w(mb) for n, mb in MB.items()}
    assert load(sd / "state.json")["steps"]["phase1be"]["status"] == "done"
