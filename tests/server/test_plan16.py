"""Server round 16: the plans (plan16.json, plan16b.json and their smoke plans) and the
configuration of 1bk, as SERVER-ROUND16-PREREGISTRATION.md registers them.

1bk (the budget round): BASE-TLS, TLS-RV and TLS-RVL at 20,000 and at 100,000 evaluations per
run (6 arms; TLS-RV and TLS-RVL are BASE-TLS with the RV option of T3.2), on 7 datasets (Air,
Con, Dow, EnC, EnH, Tow, Yac), fresh splits 81-90, no time limit: 420 MOSES runs. One phase
(regression), no Boolean phase, no extension, no choice step, no adoption rule.

The command lines of the RV arms pass the real CLI only with the code of clone K (the options
--gomea-rv-step and --gomea-rv-budget).

The RV16 review (finding 1, option a): every arm also has `--gomea-trace 1` at the end of its
argv (the exact best-so-far trace in the GOMEA log; the search does not change). The arms do
not change otherwise. The command lines pass the real CLI only with the code of clone M.

The local pilot (gate item 3): the Tow BASE-TLS-100K run peaked at 3,316 MB RSS, 64.8 % of its
5,120 MB limit. So the 100,000 arms of the 6 datasets other than Dow have a manifest of their own,
sr-100K.json, with 8,192 MB. sr.json keeps the 20,000 arms (5,120 MB); sr-Dow.json does not
change. The jobs (dataset, arm, split, argv, budget, the other fields) are the jobs of the two
manifests before the move (commit 0b48c96). Only the memory limit and the result directory of
the 180 moved jobs change: sr-100K.json has a name of its own, phase1bk-sr-100K, so its runs go
to phase1bk-sr-100K/runs/ and each manifest keeps its own manifest.json (the runner writes
<name>/manifest.json with no check). Tow comes first in sr-100K.json: the runner starts the jobs
in the order of the manifest, and the Tow 100,000 runs are the longest.

The chain (the local pilot estimate is about 39 h, more than 20 h for one plan; the rule of the
pre-registration, "Rules"): round16 (plan16.json) runs S0 and 1bk on splits 81-85
(configs/gomea/phase1bk-a: the full spec and the three manifests with splits 81-85 only; 210
jobs; an interim report), then round16b (plan16b.json) runs 1bk with the full config: the runner
keeps the 210 runs of round16 (the same manifest names, so the same result directories) and runs
the other 210; the report covers all 420 runs. round16b has no S0. The smoke plans chain in the
same way (split 81, then splits 81 and 82)."""

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

# The arms of the pre-registration, flag for flag. BASE-TLS is the BASE-TLS arm of round 14.
TLS = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
       "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
       "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
       "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
       "--const-opt", "none",
       "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
RV = ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "balance"]
RVL = ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "late"]
TRACE = ["--gomea-trace", "1"]  # the RV16 review, finding 1 (option a)
# The flags of each arm without the trace (the arms of the pre-registration).
SEARCH = {
    "BASE-TLS": TLS,
    "BASE-TLS-100K": TLS,
    "TLS-RV": TLS + RV,
    "TLS-RV-100K": TLS + RV,
    "TLS-RVL": TLS + RVL,
    "TLS-RVL-100K": TLS + RVL,
}
ARMS = {
    "BASE-TLS": (TLS + TRACE, 20000),
    "BASE-TLS-100K": (TLS + TRACE, 100000),
    "TLS-RV": (TLS + RV + TRACE, 20000),
    "TLS-RV-100K": (TLS + RV + TRACE, 100000),
    "TLS-RVL": (TLS + RVL + TRACE, 20000),
    "TLS-RVL-100K": (TLS + RVL + TRACE, 100000),
}
RV_BUDGET = {"TLS-RV": "balance", "TLS-RV-100K": "balance", "TLS-RVL": "late",
             "TLS-RVL-100K": "late"}
DATASETS = ["Air", "Con", "Dow", "EnC", "EnH", "Tow", "Yac"]
SPLITS = list(range(81, 91))
MANIFESTS = ["sr.json", "sr-100K.json", "sr-Dow.json"]
MB = {"sr-100K.json": 8192, "sr-Dow.json": 8192}  # "Memory" and the pilot (gate item 3)
OTHER_MB = 5120
K20 = ["BASE-TLS", "TLS-RV", "TLS-RVL"]
K100 = ["BASE-TLS-100K", "TLS-RV-100K", "TLS-RVL-100K"]
ARMS_OF = {"sr.json": K20, "sr-100K.json": K100, "sr-Dow.json": list(ARMS)}
NOT_DOW = [d for d in DATASETS if d != "Dow"]
DATASETS_OF = {"sr.json": NOT_DOW, "sr-100K.json": ["Tow"] + [d for d in NOT_DOW if d != "Tow"],
               "sr-Dow.json": ["Dow"]}
OLD16 = "0b48c96"  # the last commit with the 100,000 arms in sr.json
PREFIXES = ["phase1bk", "server-smoke16-1bk"]
HALF_PREFIXES = {"phase1bk": "phase1bk-a", "server-smoke16-1bk": "server-smoke16-1bk-a"}
HALF_SPLITS = {"phase1bk": [81, 82, 83, 84, 85], "server-smoke16-1bk": [81]}
REQUIRES = [f"datasets:{','.join(DATASETS)}:81-90"]
SMOKE_BUDGET = {20000: 1000, 100000: 2000}
# training rows of each dataset (the same for every split: n - 2 * (n // 4))
N_TRAIN = {d: json.loads((REPO / "data" / "gomea-sr" / d / "splits.json").read_text())["n_train"]
           for d in DATASETS}


def load(path):
    return json.loads(path.read_text())


def stage1(q):
    """The stage-1 manifests of phase_run.sh: sr.json, then sr-*.json without sr-ext-*."""
    return [q / "sr.json"] + sorted(f for f in q.glob("sr-*.json")
                                    if not f.name.startswith("sr-ext-"))


def runs(ms):
    """{(dataset, split): [arm, ...]} over the manifests `ms`; no (dataset, split, arm) twice."""
    out = {}
    for f in ms:
        m = load(f)
        for d in m["datasets"]:
            for s in m["splits"]:
                for a in m["arms"]:
                    assert a not in out.get((d, s), []), (f, d, s, a)
                    out.setdefault((d, s), []).append(a)
    return out


def cli(argv):
    """Parse `argv` (without the program name) as `moses_exec` does, with every check of
    `problem_params.parse_options` (also the cross-option ones), and return the parameters."""
    import argparse

    from pymoses.moses.main.demo_problems import register_demo_problems
    from pymoses.moses.main.moses_exec import _add_demo_options
    from pymoses.moses.main.problem import problem_manager
    from pymoses.moses.main.problem_params import problem_params
    from pymoses.moses.main.table_problems import ip_problem_params, register_table_problems

    pmr = problem_manager()
    ippp = ip_problem_params()
    tpp = register_table_problems(pmr, ippp)
    register_demo_problems(pmr)
    pp = problem_params()
    p = argparse.ArgumentParser(prog="moses", add_help=False)
    p.add_argument("-h", "--help", dest="help", action="store_true", default=False)
    pp.add_options(p)
    tpp.add_options(p)
    ippp.add_options(p)
    _add_demo_options(p)
    pp.add_gomea_options(p)
    ns = p.parse_args([str(x) for x in argv])
    pp.parse_options(ns)
    tpp.parse_options(ns)
    ippp.parse_options(ns)
    return pp, ns


def run_argv(job, run_dir, n_train):
    """The command line `moses_sr_runner._run_one` gives MOSES (without the program name):
    `moses_argv`, then `--const-opt-val` for polish or the guard, as `_run_one` adds it."""
    argv = runner.moses_argv(job, run_dir, n_train)
    if runner._const_opt(job["argv"]) == "polish" or runner._const_opt_guard(job["argv"]) == "val":
        argv += ["--const-opt-val", str(run_dir / "val_scaled.csv")]
    return argv[1:]


# ---- the arms of 1bk


@pytest.mark.parametrize("prefix", PREFIXES)
def test_arm_flags_match_the_preregistration(prefix):
    q = CONFIGS / prefix
    assert [f.name for f in stage1(q)] == MANIFESTS
    for f in stage1(q):
        m = load(f)
        assert list(m["arms"]) == ARMS_OF[f.name], f
        for a, s in m["arms"].items():
            assert set(s) == {"binary", "budget", "argv"}, (f, a)  # no seed block
            assert s["argv"] == ARMS[a][0], (f, a)


def test_base_tls_is_the_base_tls_arm_of_round_14():
    """With the trace flag at the end (the RV16 review); the search flags are those of 1bi."""
    for name, bi_name in (("sr.json", "sr.json"), ("sr-100K.json", "sr.json"),
                          ("sr-Dow.json", "sr-Dow.json")):
        bi = load(CONFIGS / "phase1bi" / bi_name)["arms"]["BASE-TLS"]
        m = load(CONFIGS / "phase1bk" / name)["arms"]
        assert bi["argv"] == TLS and bi["binary"] == ".venv/bin/moses"
        for a in ("BASE-TLS", "BASE-TLS-100K"):
            if a in m:
                assert m[a]["argv"] == bi["argv"] + TRACE, (name, a)
                assert m[a]["binary"] == bi["binary"], (name, a)
        if "BASE-TLS" in m:
            assert m["BASE-TLS"] == {**bi, "argv": bi["argv"] + TRACE}


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_arm_adds_only_the_trace_at_the_end(prefix):
    """The RV16 review (finding 1, option a): each arm is its arm of the pre-registration
    (`SEARCH`) with `--gomea-trace 1` at the end, and nothing else changes."""
    for f in stage1(CONFIGS / prefix):
        for a, s in load(f)["arms"].items():
            assert s["argv"][-2:] == TRACE and s["argv"][:-2] == SEARCH[a], (f, a)
            assert s["argv"].count("--gomea-trace") == 1, (f, a)


def test_each_rv_arm_adds_only_its_rv_flags():
    """The table "Arms": TLS-RV and TLS-RVL are BASE-TLS with --gomea-rv-step rvg and
    --gomea-rv-budget balance or late; the 100K arms have the flags of their 20K arm."""
    for a, budget in RV_BUDGET.items():
        argv = ARMS[a][0]
        assert argv[:len(TLS)] == TLS
        assert argv[len(TLS):] == ["--gomea-rv-step", "rvg", "--gomea-rv-budget", budget] + TRACE
    for a in ARMS:
        if a.endswith("-100K"):
            assert ARMS[a][0] == ARMS[a[:-5]][0]
    # the RV option needs --gomea-contin copy and --const-opt none (the BASE-TLS flags)
    assert runner._const_opt(TLS) == "none" and TLS[TLS.index("--gomea-contin") + 1] == "copy"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_manifest_asks_for_the_clipped_nmse_the_plain_fields_and_the_checkpoints(prefix):
    """The manifest keys "clip_test_nmse", "plain_nmse" and "anytime" (round 16): expand()
    gives them to every run, so every summary has test_nmse_clipped, demes and anytime."""
    for f in stage1(CONFIGS / prefix):
        m = load(f)
        assert m["clip_test_nmse"] is True and m["plain_nmse"] is True and m["anytime"] is True
        jobs = runner.expand(m, Path("/o"), REPO)
        assert jobs and all(j["clip_test_nmse"] is True and j["plain_nmse"] is True
                            and j["anytime"] is True for j in jobs)
        assert all(runner._linear_scaling(j["argv"]) == "terms" for j in jobs)
        assert all(runner._wants_anytime(j) for j in jobs)


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1bk"):
        m = load(f)
        assert m["wall_cap_s"] == 36000
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": ARMS[a][1]}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_runner_gives_each_arm_m_and_the_log_and_no_validation_table(arm, tmp_path):
    """`-m` (20,000 or 100,000) comes before the arm's flags; `events_argv` adds
    `--gomea-log`; no arm has the LM guard, so `_run_one` adds no `--const-opt-val`."""
    argv, budget = ARMS[arm]
    job = {"binary": ".venv/bin/moses", "seed": 81, "budget": {"evals": budget}, "argv": argv}
    out = runner.moses_argv(job, tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    i = out.index("-m")
    assert out[i + 1] == str(budget) and out[i + 2:-2] == argv
    full = run_argv(job, tmp_path, 100)
    assert runner._const_opt_guard(argv) == "none"
    assert not any(t.startswith(("--const-opt-val", "--const-opt-guard")) for t in full)


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_command_line_parses_with_the_real_cli(arm, tmp_path):
    """The command line of each arm on each dataset, as the runner builds it, passes every check
    of the real pymoses CLI, and gives the options of the pre-registration. The RV arms need
    the code of clone K (--gomea-rv-step, --gomea-rv-budget)."""
    argv, budget = ARMS[arm]
    for d in DATASETS:
        job = {"binary": ".venv/bin/moses", "seed": 81, "budget": {"evals": budget},
               "argv": argv}
        pp, ns = cli(run_argv(job, tmp_path, N_TRAIN[d]))
        assert ns.max_evals == budget
        assert pp.linear_scaling == "terms"
        assert ns.const_opt_val == ""
        assert pp.meta_params.exemplar_selection == "boltzmann"
        gp = pp.gomea_params
        assert ns.opt_algo == "gomea" and gp.dynamic()
        assert gp.fos == "lt-b" and gp.const_opt == "none" and gp.const_opt_guard == "none"
        assert gp.max_evals_per_deme is None
        if arm in RV_BUDGET:
            assert ns.gomea_rv_step == "rvg" and ns.gomea_rv_budget == RV_BUDGET[arm]
        else:
            assert getattr(ns, "gomea_rv_step", "off") == "off"
        # the RV16 review (finding 1): the trace on every arm, and nothing else changes
        assert ns.gomea_trace == 1 and gp.trace is True
        assert {k: v for k, v in gp.as_dict().items() if k not in ("trace", "telemetry_path")} \
            == {k: v for k, v in cli(run_argv({**job, "argv": SEARCH[arm]}, tmp_path,
                                              N_TRAIN[d]))[0].gomea_params.as_dict().items()
                if k != "telemetry_path"}


def test_no_arm_uses_the_placeholder():
    for argv, _ in ARMS.values():
        assert runner.resolve_argv(argv, 534) == argv
        assert not any("n_train" in t for t in argv)


# ---- datasets, splits, memory


def test_datasets_and_fresh_splits_and_no_extension():
    q = CONFIGS / "phase1bk"
    assert set(runs(stage1(q))) == {(d, s) for d in DATASETS for s in SPLITS}
    assert sorted(q.glob("*-ext-*.json")) == []
    assert sorted(f.name for f in q.iterdir()) == ["spec.json", "sr-100K.json", "sr-Dow.json",
                                                   "sr.json"]
    for f in stage1(q):
        assert load(f)["datasets"] == DATASETS_OF[f.name], f
        assert load(f)["name"] == f"phase1bk-{f.stem}"  # a result directory of its own
        assert load(f)["splits"] == SPLITS
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_splits_81_to_90_are_new():
    """No earlier regression manifest uses splits 81-90."""
    for f in CONFIGS.glob("*/*.json"):
        if f.parent.name in PREFIXES or f.parent.name in HALF_PREFIXES.values():
            continue
        m = load(f)
        if isinstance(m, dict) and isinstance(m.get("splits"), list):
            assert not set(m["splits"]) & set(SPLITS), f


def test_run_count_is_420():
    assert sum(len(a) for a in runs(stage1(CONFIGS / "phase1bk")).values()) == 420
    jobs = [j for f in stage1(CONFIGS / "phase1bk")
            for j in runner.expand(load(f), Path("/o"), REPO)]
    assert len(jobs) == 420 and len({j["run_id"] for j in jobs}) == 420
    assert len({j["run_dir"] for j in jobs}) == 420
    for key, n in (("arm", 70), ("seed", 42), ("task", 60)):
        counts = {}
        for j in jobs:
            counts[j[key]] = counts.get(j[key], 0) + 1
        assert set(counts.values()) == {n}, key
    assert sum(j["budget"]["evals"] for j in jobs) == 210 * 20000 + 210 * 100000


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == MB.get(f.name, OTHER_MB), f


def _old_manifest(rel: str) -> dict:
    r = subprocess.run(["git", "show", f"{OLD16}:{rel}"], cwd=REPO, capture_output=True)
    if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
        pytest.fail(f"git show {OLD16}:{rel} failed: {r.stderr.decode().strip()!r}. The test "
                    f"needs the reference commit {OLD16}; use a clone with the full history.")
    return json.loads(r.stdout)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_the_100k_arms_moved_and_no_job_changed_but_its_memory_limit(prefix):
    """The jobs of sr.json, sr-100K.json and sr-Dow.json are the jobs of sr.json and sr-Dow.json
    at 0b48c96, field for field (dataset, arm, split, binary, argv, budget, split directory,
    result directory, wall cap, the manifest keys). Only two fields of the 180 moved jobs (the
    100,000 arms of the 6 datasets other than Dow) change: the memory limit, from 5,120 to
    8,192 MB, and the result directory, now under <prefix>-sr-100K/runs/ (the name of
    sr-100K.json)."""
    q = CONFIGS / prefix
    old = {n: _old_manifest(f"configs/gomea/{prefix}/{n}") for n in ("sr.json", "sr-Dow.json")}
    assert list(old["sr.json"]["arms"]) == list(ARMS) and old["sr.json"]["mem_limit_mb"] == 5120
    # the arms are byte-identical; the other keys of the manifest do not change
    new = {f.name: load(f) for f in stage1(q)}
    assert new["sr-Dow.json"] == old["sr-Dow.json"]
    assert new["sr.json"] == {**old["sr.json"],
                              "arms": {a: old["sr.json"]["arms"][a] for a in K20}}
    name100 = f"{prefix}-sr-100K"
    assert new["sr-100K.json"] == {**old["sr.json"], "name": name100, "mem_limit_mb": 8192,
                                   "datasets": DATASETS_OF["sr-100K.json"],
                                   "arms": {a: old["sr.json"]["arms"][a] for a in K100}}
    # the expanded jobs
    before = {j["run_id"]: j for m in old.values() for j in runner.expand(m, Path("/o"), REPO)}
    after = [j for m in new.values() for j in runner.expand(m, Path("/o"), REPO)]
    assert len(after) == len({j["run_id"] for j in after}) == len(before)
    moved = {j["run_id"] for j in runner.expand(new["sr-100K.json"], Path("/o"), REPO)}
    n = len(SPLITS) if prefix == "phase1bk" else 2
    assert len(moved) == 6 * 3 * n  # 180 in phase 1bk
    for j in after:
        b = before[j["run_id"]]
        if j["run_id"] in moved:
            assert j["task"] != "Dow" and j["arm"] in K100
            assert (b["mem_limit_mb"], j["mem_limit_mb"]) == (5120, 8192)
            assert b["run_dir"] == f"/o/{prefix}-sr/runs/{j['run_id']}"
            assert j["run_dir"] == f"/o/{name100}/runs/{j['run_id']}"
            assert {k: v for k, v in j.items() if k not in ("mem_limit_mb", "run_dir")} == {
                k: v for k, v in b.items() if k not in ("mem_limit_mb", "run_dir")}
            assert set(j) == set(b)
        else:
            assert j == b


def test_the_runner_starts_the_jobs_in_the_order_of_the_manifest_so_tow_comes_first():
    """moses_sr_runner.py gives the jobs to Pool.imap_unordered (chunk size 1) in the order of
    expand(): dataset, then arm, then split. phase_run.sh does not ask for --split-major (the
    only other order). So the 30 Tow runs of sr-100K.json (the longest runs) start first."""
    src = (REPO / "tools" / "gomea" / "moses_sr_runner.py").read_text()
    assert "pool.imap_unordered(_run_one, jobs)" in src
    assert "--split-major" not in (REPO / "tools" / "gomea" / "phase_run.sh").read_text()
    jobs = runner.expand(load(CONFIGS / "phase1bk" / "sr-100K.json"), Path("/o"), REPO)
    assert [j["task"] for j in jobs[:30]] == ["Tow"] * 30
    assert {j["arm"] for j in jobs[:30]} == set(K100)
    assert [j["task"] for j in jobs[30:]] == [d for d in DATASETS_OF["sr-100K.json"][1:]
                                              for _ in range(30)]


# ---- the spec of 1bk (test_round16_report.py checks it in full)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_the_spec_covers_every_arm(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bk" if prefix == "phase1bk" else "smoke16-1bk")
    assert s["metrics"] == dict.fromkeys(DATASETS, "test_nmse")
    arms = {a for f in s["contrast_families"] for c in f["contrasts"] for a in c[1:-1]}
    assert arms == set(ARMS)
    assert {f["name"]: len(f["contrasts"]) for f in s["contrast_families"]} == {
        "P": 4, "S": 8, "SC": 8, "D": 4, "DP": 3}
    assert set(s["anytime_ratios"]) == set(ARMS) - {"BASE-TLS", "BASE-TLS-100K"}


def test_holm_families_at_70_pairs():
    """With 70 pooled pairs every family that gives verdicts (4, 8, 8) can give one; the 28
    per-dataset rows (10 pairs each) cannot (28 * 2/1024 > 0.025): they are descriptive."""
    p70 = wilcoxon_signed_rank([float(i + 1) for i in range(70)])[1]
    for n in (4, 8):
        assert holm([p70] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 28)[0] > phase_report.ALPHA


# ---- the plans


def _memory_keys_match_the_manifests(step, q):
    """per_run_mb and manifest_per_run_mb give each manifest of `q` its mem_limit_mb, so W
    follows from the limit of the runs it starts; every key names a manifest of `q`."""
    assert step["per_run_mb"] == OTHER_MB and step["manifest_per_run_mb"] == MB
    assert [f.name for f in stage1(q)] == MANIFESTS
    for f in stage1(q):
        assert step["manifest_per_run_mb"].get(f.name, step["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((q / n).exists() for n in step["manifest_per_run_mb"])


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid, then", [
    ("plan16.json", "server-round16-2026-11", "server-run16", "phase1bk", "phase1bk",
     "round16b"),
    ("plan16-smoke.json", "server-round16-smoke", "server-run16-smoke", "server-smoke16-1bk",
     "smoke16-1bk", "smoke16b"),
])
def test_the_first_plan_is_s0_then_1bk_on_the_first_splits(plan, name, state_dir, prefix, sid,
                                                           then):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and p["then"] == then
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", sid]
    assert "per_run_mb" not in p["steps"][0] and "manifest_per_run_mb" not in p["steps"][0]
    a = p["steps"][1]
    assert a["phase"] == prefix  # the results stay under the full phase
    assert a["config_dir"] == "{repo}/configs/gomea/" + HALF_PREFIXES[prefix]
    assert not a.get("pilot") and not a.get("set_commit_in")
    # no seeded arm, so no GP-GOMEA build; the 7 datasets with the fresh splits 81-90 (set-up
    # builds all 90 splits at once; so round16b cannot miss a split that round16 had)
    assert a["requires"] == REQUIRES
    _memory_keys_match_the_manifests(a, CONFIGS / HALF_PREFIXES[prefix])


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan16b.json", "server-round16b-2026-11", "server-run16b", "phase1bk", "phase1bk"),
    ("plan16b-smoke.json", "server-round16b-smoke", "server-run16b-smoke", "server-smoke16-1bk",
     "smoke16-1bk"),
])
def test_the_second_plan_is_1bk_with_the_full_config_and_no_s0(plan, name, state_dir, prefix,
                                                               sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and "then" not in p
    assert [s["type"] for s in p["steps"]] == ["phase"]  # no S0: round16 ran it
    a = p["steps"][0]
    assert a["id"] == sid and a["phase"] == prefix and "config_dir" not in a
    assert not a.get("pilot") and not a.get("set_commit_in")
    assert a["requires"] == REQUIRES
    _memory_keys_match_the_manifests(a, CONFIGS / prefix)


@pytest.mark.parametrize("full, half", list(HALF_PREFIXES.items()))
def test_the_first_half_is_the_full_config_with_the_first_splits(full, half):
    """The same spec.json and the same three manifests (the same names, so the same result
    directories), with the first splits only (81-85; the smoke: 81)."""
    q, h = CONFIGS / full, CONFIGS / half
    assert sorted(f.name for f in h.iterdir()) == sorted(f.name for f in q.iterdir())
    assert sorted(f.name for f in h.iterdir()) == ["spec.json"] + sorted(MANIFESTS)
    assert (h / "spec.json").read_bytes() == (q / "spec.json").read_bytes()
    for n in MANIFESTS:
        m, fm = load(h / n), load(q / n)
        assert m["splits"] == HALF_SPLITS[full] and set(m["splits"]) < set(fm["splits"])
        assert fm["splits"][:len(m["splits"])] == m["splits"]
        assert {k: v for k, v in m.items() if k != "splits"} == {
            k: v for k, v in fm.items() if k != "splits"}, n


@pytest.mark.parametrize("full, half, n_half", [("phase1bk", "phase1bk-a", 210),
                                                ("server-smoke16-1bk", "server-smoke16-1bk-a",
                                                 42)])
def test_the_chain_runs_each_job_once(full, half, n_half):
    """The jobs of the first plan are the jobs of the full config with the first splits, field
    for field (the same argv, budget, memory limit and result directory). So the second plan
    keeps them (the runner skips a run that has a summary.json) and runs the others."""
    a = [j for f in stage1(CONFIGS / half) for j in runner.expand(load(f), Path("/o"), REPO)]
    b = [j for f in stage1(CONFIGS / full) for j in runner.expand(load(f), Path("/o"), REPO)]
    assert len(a) == n_half and len(b) == 2 * n_half
    assert len({j["run_dir"] for j in a}) == n_half and len({j["run_dir"] for j in b}) == 2 * n_half
    first = set(HALF_SPLITS[full])
    assert {j["run_id"]: j for j in a} == {j["run_id"]: j for j in b if j["seed"] in first}
    assert {j["seed"] for j in a} == first
    # each plan: every arm on every dataset, and the 100,000 jobs with 8,192 MB
    for jobs in (a, b):
        assert {(j["task"], j["arm"]) for j in jobs} == {(d, x) for d in DATASETS for x in ARMS}
        assert {j["run_id"] for j in jobs if j["mem_limit_mb"] == 8192} == {
            j["run_id"] for j in jobs if j["task"] == "Dow" or j["arm"] in K100}


def test_the_requirement_needs_the_fresh_splits(tmp_path, monkeypatch):
    """A dataset set built before round 16 (splits 1-80) does not meet the requirement."""
    sys.path.insert(0, str(SERVER))
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan16.json")
    for a in DATASETS:
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
        for k in range(1, 81):
            (d / "splits" / f"s{k}").mkdir()
            for t in ("train", "val", "test"):
                (d / "splits" / f"s{k}" / f"{t}.csv").write_text("x1,y\n")
    why = r.requirement(REQUIRES[0])
    assert why and "81-90" in why
    for a in DATASETS:
        for k in SPLITS:
            s = tmp_path / "phase1f" / "datasets" / a / "splits" / f"s{k}"
            s.mkdir()
            for t in ("train", "val", "test"):
                (s / f"{t}.csv").write_text("x1,y\n")
    assert r.requirement(REQUIRES[0]) is None


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan16.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    s0 = load(SERVER / "plan16-smoke.json")["steps"][0]
    args = s0["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_sr_datasets.py",
              "tests/gomea/test_round16_runner.py", "tests/gomea/test_round16_report.py",
              "tests/gomea/test_rv_core.py", "tests/gomea/test_rv_step_identity.py",
              "tests/gomea/test_gomea_trace.py"):  # the RV16 review: the trace option
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


@pytest.mark.parametrize("first, second", [("plan16.json", "round16b"),
                                           ("plan16-smoke.json", "smoke16b")])
def test_the_real_plans_chain(first, second):
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.next_plan(SERVER / first) == chain.resolve(second)
    assert chain.root(chain.resolve(second)) == SERVER / first
    assert chain.lock_path(SERVER / first) == chain.lock_path(chain.resolve(second))
    assert chain.stop_marker(SERVER / first) == chain.stop_marker(chain.resolve(second))
    assert chain.members(chain.resolve(second)) == [SERVER / first, chain.resolve(second)]


def test_no_other_plan_chains_to_round_16():
    for f in SERVER.glob("plan*.json"):
        if f.name in ("plan16.json", "plan16-smoke.json"):
            continue
        assert load(f).get("then") not in ("round16", "smoke16", "round16b", "smoke16b"), f
    for f in ("plan16b.json", "plan16b-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan, chain_files", [
    ("round16", "plan16.json", ["plan16.json", "plan16b.json"]),
    ("smoke16", "plan16-smoke.json", ["plan16-smoke.json", "plan16b-smoke.json"]),
    ("round16b", "plan16b.json", ["plan16b.json"]),
    ("smoke16b", "plan16b-smoke.json", ["plan16b-smoke.json"]),
])
def test_plan_names_resolve(name, plan, chain_files):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan
    assert [f.name for f in chain.forward(SERVER / plan)] == chain_files


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round16|smoke16|round16b|smoke16b" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan_with_the_fresh_splits():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text
    assert f'-eq {8 * len(sr_datasets.SEEDS)} ]' in text and max(sr_datasets.SEEDS) == 90
    assert '-type f | wc -l)" -eq 720 ]' in text
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bk", CONFIGS / "server-smoke16-1bk"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1bk", "server-smoke16-1bk")
        assert m["datasets"] == fm["datasets"] and m["splits"] == [81, 82]
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert m["datasets_dir"] == fm["datasets_dir"]
        for k in ("clip_test_nmse", "plain_nmse", "anytime"):
            assert m[k] is fm[k] is True
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert {k: v for k, v in s.items() if k != "budget"} == {
                k: v for k, v in fm["arms"][a].items() if k != "budget"}
            assert s["budget"] == {"evals": SMOKE_BUDGET[fm["arms"][a]["budget"]["evals"]]}
    sp, fp = load(q / "spec.json"), load(full / "spec.json")
    assert {k: v for k, v in sp.items() if k != "phase"} == {
        k: v for k, v in fp.items() if k != "phase"}


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


def test_server_workers_are_7_for_dow_and_the_100k_arms_and_11_otherwise(tmp_path):
    """The section "Memory" of the pre-registration and the pilot (gate item 3), on the 12-CPU
    64 GB server: sr-100K.json and sr-Dow.json (8,192 MB) W = 7, sr.json (5,120 MB) W = 11."""
    d = _server(tmp_path)
    for name in MANIFESTS:
        w = d(MB.get(name, OTHER_MB))
        want = (7, "memory") if name in MB else (11, "cpu")
        assert (w["workers"], w["limited_by"]) == want, name


FAKE_SR = r"""import json, os, sys
from pathlib import Path
w = sys.argv[sys.argv.index("--workers") + 1]
m = json.loads(Path(sys.argv[1]).read_text())
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"manifest": Path(sys.argv[1]).name, "workers": int(w),
                         "config": Path(sys.argv[1]).parent.name, "name": m["name"],
                         "splits": m["splits"], "mem": m["mem_limit_mb"],
                         "out": sys.argv[2]}) + "\n")
"""
FAKE_REPORT = r"""import json, sys
from pathlib import Path
out = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
(out / "verdict.json").write_text("{}")
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"report": sys.argv[1:4], "dirs": sys.argv[5].split(",")}) + "\n")
"""


def _w(mb):
    sys.path.insert(0, str(SERVER))
    import resources
    return resources.decide(env={"PER_RUN_MB": str(mb)})["workers"]


def test_the_phase_runs_with_its_own_workers(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bk configs and the phase steps of
    both plans of the chain; the runner and the report are fakes. The first step runs the three
    manifests of configs/gomea/phase1bk-a (PHASE_CONFIG_DIR, splits 81-85), the second the three
    of configs/gomea/phase1bk (splits 81-90): sr.json, then sr-100K.json and sr-Dow.json with
    their own W, all into the result directories of phase1bk. The report of each step reads the
    result directories of all three manifests."""
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
    for q in ("phase1bk", "phase1bk-a"):
        shutil.copytree(CONFIGS / q, repo / "configs" / "gomea" / q)
    steps = ([dict(s) for s in load(SERVER / "plan16.json")["steps"][1:]]
             + [dict(s) for s in load(SERVER / "plan16b.json")["steps"]])
    for i, s in enumerate(steps):
        s.pop("requires")
        s["id"] = f"{s['id']}-{i}"  # two steps of one test plan need two ids
    plan = {"name": "t16", "state_dir": "server-run16-t", "steps": steps}
    (repo / "tools" / "server" / "plan-t16.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS",
                        "PYTHONPATH")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="1024", MANIFEST_WORKERS="sr.json=99")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t16.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    runs_, reports = [t for t in trace if "manifest" in t], [t for t in trace if "report" in t]
    assert [t["manifest"] for t in runs_] == MANIFESTS * 2
    assert [t["config"] for t in runs_] == ["phase1bk-a"] * 3 + ["phase1bk"] * 3
    assert [t["name"] for t in runs_] == ["phase1bk-sr", "phase1bk-sr-100K", "phase1bk-sr-Dow"] * 2
    assert [t["splits"] for t in runs_] == [list(range(81, 86))] * 3 + [SPLITS] * 3
    for t in runs_:
        assert t["workers"] == _w(MB.get(t["manifest"], OTHER_MB)), t
        assert t["mem"] == MB.get(t["manifest"], OTHER_MB), t
        assert t["out"] == "../gomea-results/phase1bk"
    # the result directory of each of the three manifests
    dirs = [f"../gomea-results/phase1bk/phase1bk-{x}" for x in ("sr", "sr-100K", "sr-Dow")]
    assert [t["dirs"] for t in reports] == [dirs, dirs]
    assert [Path(t["report"][2]).parent.name for t in reports] == ["phase1bk-a", "phase1bk"]
    sd = res / "server-run16-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [(e.get("manifest"), e["per_run_mb"]) for e in events if e["event"] == "workers_decision"]
    assert wd == [(None, OTHER_MB), ("sr-100K.json", 8192), ("sr-Dow.json", 8192)] * 2
    st = load(sd / "state.json")["steps"]
    assert [st[s["id"]]["status"] for s in steps] == ["done", "done"]
