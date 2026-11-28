"""Server round 17: the plans (plan17.json, plan17b.json and their smoke plans) and the
configuration of 1bl, as SERVER-ROUND17-PREREGISTRATION.md registers them.

1bl (HC against BASE-TLS): HC-LM and HC-TLS at 20,000 and at 100,000 evaluations per run, no
deme cap (`-a hc`), no time limit, on the 7 datasets of round 16 (Dow at 20,000 only), splits
81-90: 260 MOSES runs. The GOMEA partners (BASE-TLS and BASE-TLS-100K) are the round 16 runs of
the same dataset, split, seed and budget: the report reads them from ../gomea-results/phase1bk/
(the spec key "partners"), and each 1bl step needs their 130 summaries (the requirement kind
"summaries:", added for this round; the same failure behaviour as "datasets:": the step is
skipped with the reason in the log). One phase (regression), no Boolean phase, no extension, no
choice step, no adoption rule.

The design (arms, flags, families, memory) is in tests/gomea/round17_design.py. HC-TLS is in the
design while its check (clone A) has not failed; tools/gomea/drop_arms.py and the flag
`HC_TLS` of that module leave it out (see its docstring).

The chain (the agent estimate after the local pilot, about 49 h, range 43-61 h, is more than
20 h for one plan; the rule of the pre-registration, "Rules"; amendment A3): round17
(plan17.json) runs S0 and 1bl on splits 81-85 (configs/gomea/phase1bl-a), then round17b
(plan17b.json) runs 1bl with the full config; the runner keeps the runs of round17 and runs
the others. round17b has no S0.

The smoke plans chain in the same way, on OLD splits 1 and 2 (never 81-90). smoke17 first runs
smoke17-1bk, the GOMEA partners in miniature (the BASE-TLS arms of round 16), so the smoke needs
no round 16 results: its 1bl spec reads the partners from ../gomea-results/server-smoke17-1bk/."""

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
sys.path.insert(0, str(REPO / "tests" / "gomea"))
sys.path.insert(0, str(SERVER))
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402
import round17_design as design  # noqa: E402
import sr_datasets  # noqa: E402

ARMS = design.arms()
PREFIXES = ["phase1bl", "server-smoke17-1bl"]
HALF_PREFIXES = {"phase1bl": "phase1bl-a", "server-smoke17-1bl": "server-smoke17-1bl-a"}
SPLITS_OF = {"phase1bl": design.SPLITS, "server-smoke17-1bl": design.SMOKE_SPLITS}
HALF_SPLITS = {"phase1bl": design.HALF_SPLITS, "server-smoke17-1bl": design.SMOKE_SPLITS[:1]}
D7, D6 = ",".join(design.DS7), ",".join(design.DS6)


def partners(phase: str, span: str) -> list[str]:
    return [f"summaries:{phase}/{phase}-sr:BASE-TLS:{D6}:{span}",
            f"summaries:{phase}/{phase}-sr-100K:BASE-TLS-100K:{D6}:{span}",
            f"summaries:{phase}/{phase}-sr-Dow:BASE-TLS:Dow:{span}"]


REQUIRES = [f"datasets:{D7}:81-90"] + partners("phase1bk", "81-90")
SMOKE_REQUIRES = [f"datasets:{D7}:1-2"] + partners("server-smoke17-1bk", "1-2")
SMOKE_BUDGET = {20000: 1000, 100000: 2000}
OTHER_MB = design.MB["sr.json"]
MANIFEST_MB = {"sr-Dow.json": 8192, "sr-100K.json": 6553}  # the local pilot (A3)
N_TRAIN = {d: json.loads((REPO / "data" / "gomea-sr" / d / "splits.json").read_text())["n_train"]
           for d in design.DS7}


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
    `problem_params.parse_options`, and return the parameters (as test_plan16.cli)."""
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
    """The command line `moses_sr_runner._run_one` gives MOSES (without the program name)."""
    argv = runner.moses_argv(job, run_dir, n_train)
    if runner._const_opt(job["argv"]) == "polish" or runner._const_opt_guard(job["argv"]) == "val":
        argv += ["--const-opt-val", str(run_dir / "val_scaled.csv")]
    return argv[1:]


# ---- the arms of 1bl


@pytest.mark.parametrize("prefix", PREFIXES + list(HALF_PREFIXES.values()))
def test_arm_flags_match_the_preregistration(prefix):
    q = CONFIGS / prefix
    assert [f.name for f in stage1(q)] == design.MANIFESTS
    for f in stage1(q):
        m = load(f)
        assert list(m["arms"]) == design.arms_of(f.name), f
        for a, s in m["arms"].items():
            assert set(s) == {"binary", "budget", "argv"}, (f, a)  # no seed block
            assert s["argv"] == ARMS[a][0] and s["binary"] == ".venv/bin/moses", (f, a)


def test_hc_lm_is_the_hc_lm_arm_of_rounds_13_and_14():
    for ph in ("phase1bh", "phase1bi"):
        for n in ("sr.json", "sr-Dow.json"):
            assert load(CONFIGS / ph / n)["arms"]["HC-LM"]["argv"] == design.HC_LM_ARGV, (ph, n)


def test_hc_lm_and_hc_tls_differ_only_in_the_constant_flags():
    """The table "Arms": HC-TLS has the constant settings of BASE-TLS (--linear-scaling terms
    --const-opt none). HC-LM has three constant flags (--linear-scaling 1 --const-opt lm
    --const-opt-guard val), so the two arms differ in three flags, not in the two that the text
    under the table names; the configs follow the table. No deme cap: -a hc, not -a hc-capped."""
    lm, tls = design.HC_LM_ARGV, design.HC_TLS_ARGV
    assert lm[:4] == tls[:4] == ["-a", "hc", "--knob-probe-baseline", "reduced"]
    assert lm[4:] == ["--linear-scaling", "1", "--const-opt", "lm", "--const-opt-guard", "val"]
    assert tls[4:] == ["--linear-scaling", "terms", "--const-opt", "none"]
    base = load(CONFIGS / "phase1bk" / "sr.json")["arms"]["BASE-TLS"]["argv"]
    for k in ("--linear-scaling", "--const-opt"):
        assert tls[tls.index(k) + 1] == base[base.index(k) + 1]
    for a, (argv, budget) in ARMS.items():
        assert runner._opt_algo(argv) == "hc" and "--hc-fraction-of-nn" not in argv, a
        assert not any(t.startswith(("--gomea-max-evals-per-deme", "--max-time")) for t in argv)
        if a.endswith("-100K"):
            assert (argv, budget) == (ARMS[a[:-5]][0], 100000)


@pytest.mark.parametrize("prefix", PREFIXES + list(HALF_PREFIXES.values()))
def test_every_manifest_has_the_keys_of_round_16_and_the_hc_progress_key(prefix):
    """clip_test_nmse, plain_nmse (round 16) and hc_anytime (round 17: the HC progress curves at
    deme resolution); the round 16 key "anytime" is for -a gomea only."""
    for f in stage1(CONFIGS / prefix):
        m = load(f)
        assert m["clip_test_nmse"] is True and m["plain_nmse"] is True
        assert m["hc_anytime"] is True and "anytime" not in m
        jobs = runner.expand(m, Path("/o"), REPO)
        assert jobs and all(runner._wants_hc_anytime(j) for j in jobs)
        assert not any(runner._wants_anytime(j) for j in jobs)


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1bl"):
        m = load(f)
        assert m["wall_cap_s"] == design.WALL_CAP_S[f.name], f  # A3: 72,000 s for sr-100K.json
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": ARMS[a][1]}, (f, a)


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_command_line_parses_with_the_real_cli(arm, tmp_path):
    """The command line of each arm on each dataset, as the runner builds it, passes every check
    of the real pymoses CLI and gives the options of the pre-registration."""
    argv, budget = ARMS[arm]
    for d in design.DS7:
        job = {"binary": ".venv/bin/moses", "seed": 81, "budget": {"evals": budget},
               "argv": argv}
        pp, ns = cli(run_argv(job, tmp_path, N_TRAIN[d]))
        assert ns.max_evals == budget and ns.opt_algo == "hc"
        gp = pp.gomea_params
        if "TLS" in arm:
            assert pp.linear_scaling == "terms" and gp.const_opt == "none"
            assert ns.const_opt_val == ""
        else:
            assert str(pp.linear_scaling) == "1" and gp.const_opt == "lm"
            assert gp.const_opt_guard == "val" and ns.const_opt_val.endswith("val_scaled.csv")
        assert gp.max_evals_per_deme is None  # no deme cap
        assert pp.meta_params.exemplar_selection == "boltzmann"


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_runner_gives_hc_lm_its_log_and_hc_tls_none(arm, tmp_path):
    """events_argv: -a hc with --const-opt lm gets --gomea-log (its "end" events); -a hc with
    --const-opt none gets none (its progress comes from the expansion hook)."""
    argv, budget = ARMS[arm]
    job = {"binary": ".venv/bin/moses", "seed": 81, "budget": {"evals": budget}, "argv": argv}
    out = runner.moses_argv(job, tmp_path, 100)
    i = out.index("-m")
    assert out[i + 1] == str(budget)
    if "TLS" in arm:
        assert out[i + 2:] == argv
    else:
        assert out[i + 2:] == argv + ["--gomea-log", str(tmp_path / "events.jsonl")]


# ---- datasets, splits, memory


def test_datasets_splits_and_no_extension():
    q = CONFIGS / "phase1bl"
    want = {(d, s): [a for a in ARMS if d != "Dow" or not a.endswith("-100K")]
            for d in design.DS7 for s in design.SPLITS}
    got = runs(stage1(q))
    assert set(got) == set(want) and all(sorted(got[k]) == sorted(want[k]) for k in want)
    assert sorted(q.glob("*-ext-*.json")) == []
    assert sorted(f.name for f in q.iterdir()) == ["spec.json"] + sorted(design.MANIFESTS)
    for f in stage1(q):
        m = load(f)
        assert m["datasets"] == design.DATASETS_OF[f.name], f
        assert m["name"] == f"phase1bl-{f.stem}"  # a result directory of its own
        assert m["splits"] == design.SPLITS
    assert set(design.SPLITS) <= set(sr_datasets.SEEDS)


def test_the_splits_are_the_splits_of_round_16():
    """Each HC run pairs with the round 16 run of the same dataset, split and budget."""
    r16 = runs(stage1(CONFIGS / "phase1bk"))
    for (d, s), arms in runs(stage1(CONFIGS / "phase1bl")).items():
        for a in arms:
            assert design.PARTNER[a] in r16[(d, s)], (d, s, a)


def test_run_count():
    jobs = [j for f in stage1(CONFIGS / "phase1bl")
            for j in runner.expand(load(f), Path("/o"), REPO)]
    n = 260 if design.HC_TLS else 130
    assert len(jobs) == n and len({j["run_id"] for j in jobs}) == n
    assert len({j["run_dir"] for j in jobs}) == n
    counts = {}
    for j in jobs:
        counts[j["arm"]] = counts.get(j["arm"], 0) + 1
    assert counts == {a: 70 if b == 20000 else 60 for a, (_, b) in ARMS.items()}


@pytest.mark.parametrize("prefix", PREFIXES + list(HALF_PREFIXES.values()))
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == design.MB[f.name], f


def test_the_runner_starts_the_tow_100k_jobs_first():
    jobs = runner.expand(load(CONFIGS / "phase1bl" / "sr-100K.json"), Path("/o"), REPO)
    k = 10 * len(design.arms_of("sr-100K.json"))
    assert [j["task"] for j in jobs[:k]] == ["Tow"] * k


# ---- the spec (test_round17_report.py checks the report)


def test_the_spec_is_the_design():
    real = design.spec("1bl", design.REAL_PARTNER_DIRS)
    assert load(CONFIGS / "phase1bl" / "spec.json") == real
    assert load(CONFIGS / "phase1bl-a" / "spec.json") == real
    sm = design.spec("smoke17-1bl", design.SMOKE_PARTNER_DIRS)
    assert load(CONFIGS / "server-smoke17-1bl" / "spec.json") == sm
    assert load(CONFIGS / "server-smoke17-1bl-a" / "spec.json") == sm


def test_holm_families_at_60_pairs():
    """With 60 pooled pairs each family that gives verdicts (4 or 2, 9, 8) can give one."""
    p60 = wilcoxon_signed_rank([float(i + 1) for i in range(60)])[1]
    s = load(CONFIGS / "phase1bl" / "spec.json")
    for f in s["contrast_families"]:
        if f["role"] != "descriptive":
            assert holm([p60] * len(f["contrasts"]))[0] < phase_report.ALPHA, f["name"]


# ---- the plans


def _memory_keys_match_the_manifests(step, q):
    assert step["per_run_mb"] == OTHER_MB and step["manifest_per_run_mb"] == MANIFEST_MB
    for f in stage1(q):
        assert step["manifest_per_run_mb"].get(f.name, step["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((q / n).exists() for n in step["manifest_per_run_mb"])


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid, then, requires", [
    ("plan17.json", "server-round17-2026-11", "server-run17", "phase1bl", "phase1bl",
     "round17b", REQUIRES),
    ("plan17-smoke.json", "server-round17-smoke", "server-run17-smoke", "server-smoke17-1bl",
     "smoke17-1bl", "smoke17b", SMOKE_REQUIRES),
])
def test_the_first_plan_is_s0_then_1bl_on_the_first_splits(plan, name, state_dir, prefix, sid,
                                                           then, requires):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and p["then"] == then
    steps = p["steps"]
    assert steps[0]["id"] == "S0" and steps[0]["type"] == "check"
    if plan == "plan17-smoke.json":
        # the smoke makes its own GOMEA partners first (round 16 in miniature, BASE-TLS only)
        assert [s["id"] for s in steps] == ["S0", "smoke17-1bk", sid]
        pk = steps[1]
        assert pk["phase"] == "server-smoke17-1bk" and "config_dir" not in pk
        assert pk["requires"] == [f"datasets:{D7}:1-2"]
        assert pk["per_run_mb"] == 5120
        assert pk["manifest_per_run_mb"] == {"sr-100K.json": 8192, "sr-Dow.json": 8192}
    else:
        assert [s["id"] for s in steps] == ["S0", sid]
    a = steps[-1]
    assert a["phase"] == prefix and a["type"] == "phase"
    assert a["config_dir"] == "{repo}/configs/gomea/" + HALF_PREFIXES[prefix]
    assert not a.get("pilot") and not a.get("set_commit_in")
    assert a["requires"] == requires
    _memory_keys_match_the_manifests(a, CONFIGS / HALF_PREFIXES[prefix])
    assert ("6,553 MB, wall cap 72,000 s); the limits come from the local pilot (amendment A3"
            in p["description"] and "placeholder" not in p["description"]) or plan != "plan17.json"


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid, requires", [
    ("plan17b.json", "server-round17b-2026-11", "server-run17b", "phase1bl", "phase1bl",
     REQUIRES),
    ("plan17b-smoke.json", "server-round17b-smoke", "server-run17b-smoke", "server-smoke17-1bl",
     "smoke17-1bl", SMOKE_REQUIRES),
])
def test_the_second_plan_is_1bl_with_the_full_config_and_no_s0(plan, name, state_dir, prefix,
                                                               sid, requires):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and "then" not in p
    assert [s["type"] for s in p["steps"]] == ["phase"]
    a = p["steps"][0]
    assert a["id"] == sid and a["phase"] == prefix and "config_dir" not in a
    assert a["requires"] == requires
    _memory_keys_match_the_manifests(a, CONFIGS / prefix)


@pytest.mark.parametrize("full, half", list(HALF_PREFIXES.items()))
def test_the_first_half_is_the_full_config_with_the_first_splits(full, half):
    q, h = CONFIGS / full, CONFIGS / half
    assert sorted(f.name for f in h.iterdir()) == sorted(f.name for f in q.iterdir())
    assert (h / "spec.json").read_bytes() == (q / "spec.json").read_bytes()
    for n in design.MANIFESTS:
        m, fm = load(h / n), load(q / n)
        assert m["splits"] == HALF_SPLITS[full] and fm["splits"] == SPLITS_OF[full]
        assert {k: v for k, v in m.items() if k != "splits"} == {
            k: v for k, v in fm.items() if k != "splits"}, n


@pytest.mark.parametrize("full, half", list(HALF_PREFIXES.items()))
def test_the_chain_runs_each_job_once(full, half):
    a = [j for f in stage1(CONFIGS / half) for j in runner.expand(load(f), Path("/o"), REPO)]
    b = [j for f in stage1(CONFIGS / full) for j in runner.expand(load(f), Path("/o"), REPO)]
    first = set(HALF_SPLITS[full])
    assert {j["run_id"]: j for j in a} == {j["run_id"]: j for j in b if j["seed"] in first}
    assert len(b) == len(a) * len(SPLITS_OF[full]) // len(first)


def test_the_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bl", CONFIGS / "server-smoke17-1bl"
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1bl", "server-smoke17-1bl")
        assert m["datasets"] == fm["datasets"] and m["splits"] == design.SMOKE_SPLITS
        assert not set(m["splits"]) & set(design.SPLITS)  # never splits 81-90
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert {k: v for k, v in m.items() if k not in ("name", "splits", "wall_cap_s", "arms")} \
            == {k: v for k, v in fm.items() if k not in ("name", "splits", "wall_cap_s", "arms")}
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert s["argv"] == fm["arms"][a]["argv"]
            assert s["budget"] == {"evals": SMOKE_BUDGET[fm["arms"][a]["budget"]["evals"]]}


def test_the_smoke_partners_are_the_base_tls_arms_of_round_16():
    q, r16 = CONFIGS / "server-smoke17-1bk", CONFIGS / "phase1bk"
    assert sorted(f.name for f in q.iterdir()) == ["spec.json"] + sorted(design.MANIFESTS)
    for n, arm in (("sr.json", "BASE-TLS"), ("sr-100K.json", "BASE-TLS-100K"),
                   ("sr-Dow.json", "BASE-TLS")):
        m, fm = load(q / n), load(r16 / n)
        assert m["name"] == fm["name"].replace("phase1bk", "server-smoke17-1bk")
        assert list(m["arms"]) == [arm] and m["splits"] == design.SMOKE_SPLITS
        src = fm["arms"][arm]
        assert m["arms"][arm] == {**src, "budget": {"evals": SMOKE_BUDGET[src["budget"]["evals"]]}}
        assert {k: v for k, v in m.items() if k not in ("name", "splits", "wall_cap_s", "arms")} \
            == {k: v for k, v in fm.items() if k not in ("name", "splits", "wall_cap_s", "arms")}
    # the smoke 1bl spec reads these directories; the requirement names them too
    dirs = load(CONFIGS / "server-smoke17-1bl" / "spec.json")["partners"]["dirs"]
    assert dirs == [f"../gomea-results/server-smoke17-1bk/{load(q / n)['name']}"
                    for n in design.MANIFESTS]


def test_the_partner_requirements_name_the_partner_directories_of_the_spec():
    """Each "summaries:" requirement names one directory of the spec's "partners" (relative to
    ../gomea-results), the partner arm and the datasets and splits of the runs whose partners it
    has: 60 + 60 + 10 = 130 summaries for the real round."""
    for plan, prefix in (("plan17.json", "phase1bl"), ("plan17b.json", "phase1bl"),
                         ("plan17-smoke.json", "server-smoke17-1bl"),
                         ("plan17b-smoke.json", "server-smoke17-1bl")):
        step = load(SERVER / plan)["steps"][-1]
        dirs = load(CONFIGS / prefix / "spec.json")["partners"]["dirs"]
        reqs = [r for r in step["requires"] if r.startswith("summaries:")]
        assert ["../gomea-results/" + r.split(":")[1] for r in reqs] == dirs, plan
        n = 0
        for r in reqs:
            _, _, arm, names, span = r.split(":")
            lo, hi = (int(x) for x in span.split("-"))
            n += len(names.split(",")) * (hi - lo + 1)
            assert arm in set(design.PARTNER.values())
        assert n == (130 if plan.startswith("plan17.") or plan.startswith("plan17b.") else 26)


def _write_summary(res: Path, rel: str, task: str, arm: str, seed: int, **kw) -> Path:
    d = res / rel / "runs" / f"{task}__{arm}__s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "summary.json"
    f.write_text(json.dumps({"task": task, "arm": arm, "seed": seed, "status": "ok", **kw}))
    return f


def test_the_partner_requirement(tmp_path, monkeypatch):
    """None when the 130 partner summaries exist (a failed run counts as present); else the
    reason, with the count and the first missing runs. A cut summary or a summary of another
    run does not count."""
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan17.json")
    reqs = REQUIRES[1:]
    for q in reqs:
        assert "missing" in r.requirement(q) and "of 60" in r.requirement(q) or "Dow" in q
    files = []
    for q in reqs:
        _, rel, arm, names, span = q.split(":")
        for a in names.split(","):
            for k in range(81, 91):
                files.append(_write_summary(tmp_path, rel, a, arm, k,
                                            status="error" if a == "Dow" else "ok"))
    assert len(files) == 130
    assert all(r.requirement(q) is None for q in reqs)
    files[3].write_text('{"task": "Air"')  # a half-written summary
    why = r.requirement(reqs[0])
    assert why == (f"1 of 60 BASE-TLS summaries missing under {tmp_path}/phase1bk/phase1bk-sr "
                   "(the partner runs of the report; first: Air__BASE-TLS__s84)")
    files[3].write_text(json.dumps({"task": "Air", "arm": "BASE-TLS", "seed": 85}))  # another
    assert "Air__BASE-TLS__s84" in r.requirement(reqs[0])
    assert r.requirement("summaries:phase1bk:BASE-TLS:Air") .startswith("bad requirement")
    assert r.requirement("summaries:phase1bk/x:BASE-TLS:Air:a-b").startswith("bad requirement")


def test_a_failed_partner_requirement_skips_the_step_with_the_reason(tmp_path, monkeypatch):
    """The same failure behaviour as "datasets:": the step is skipped, the reason is in the
    state, the events and run.log, and the round goes on."""
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan17b.json")
    step = {**load(SERVER / "plan17b.json")["steps"][0],
            "requires": REQUIRES[1:]}
    st = r.step_state(step["id"])
    r.do_phase(step, st)
    assert st["status"] == "skipped" and "BASE-TLS summaries missing" in st["reason"]
    ev = [json.loads(x) for x in (r.sd / "events.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "step_skip" and ev[-1]["reason"] == st["reason"]
    assert "SKIP phase1bl:" in (r.sd / "run.log").read_text()


def test_the_bundle_holds_the_hc_progress_lines(tmp_path, monkeypatch):
    """The lines of the expansion hook (one per deme) go into the bundle beside the summary,
    so the curve at deme resolution comes back from the server too; moses.log does not."""
    import tarfile

    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan17b.json")
    run = tmp_path / "phase1bl" / "phase1bl-sr" / "runs" / "Con__HC-TLS__s81"
    run.mkdir(parents=True)
    for f in ("summary.json", "hc_progress.jsonl", "moses.log", "train_scaled.csv",
              "candidates.json"):
        (run / f).write_text("{}\n")
    with tarfile.open(orchestrate.make_bundle(r)) as tf:
        names = {n.rsplit("/", 1)[1] for n in tf.getnames() if "/runs/" in n}
    assert names == {"summary.json", "hc_progress.jsonl"}


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan17.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    s0 = load(SERVER / "plan17-smoke.json")["steps"][0]
    args = s0["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_round17_runner.py",
              "tests/gomea/test_hc_progress.py", "tests/gomea/test_round17_report.py",
              "tests/gomea/test_round16_report.py", "tests/gomea/test_hc_lm_identity.py"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


@pytest.mark.parametrize("first, second", [("plan17.json", "round17b"),
                                           ("plan17-smoke.json", "smoke17b")])
def test_the_plans_chain(first, second):
    import chain
    assert chain.next_plan(SERVER / first) == chain.resolve(second)
    assert chain.root(chain.resolve(second)) == SERVER / first
    assert chain.lock_path(SERVER / first) == chain.lock_path(chain.resolve(second))
    assert chain.members(chain.resolve(second)) == [SERVER / first, chain.resolve(second)]


def test_no_other_plan_chains_to_round_17():
    for f in SERVER.glob("plan*.json"):
        if f.name in ("plan17.json", "plan17-smoke.json"):
            continue
        assert load(f).get("then") not in ("round17", "smoke17", "round17b", "smoke17b"), f
    for f in ("plan17b.json", "plan17b-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan, chain_files", [
    ("round17", "plan17.json", ["plan17.json", "plan17b.json"]),
    ("smoke17", "plan17-smoke.json", ["plan17-smoke.json", "plan17b-smoke.json"]),
    ("round17b", "plan17b.json", ["plan17b.json"]),
    ("smoke17b", "plan17b-smoke.json", ["plan17b-smoke.json"]),
])
def test_plan_names_resolve(name, plan, chain_files):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan
    assert [f.name for f in chain.forward(SERVER / plan)] == chain_files


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round16|smoke16|round16b|smoke16b|round17|smoke17|round17b|smoke17b|FILE" in (
        SERVER / script).read_text()


# ---- W for each manifest


def test_server_workers(tmp_path):
    """On the 12-CPU 64 GB server (reserve 6,553 MB): sr.json (4,096 MB) W = 11 (the CPUs),
    sr-100K.json (6,553 MB, the largest limit for W = 9) W = 9 and sr-Dow.json (8,192 MB) W = 7
    (the memory; amendment A3)."""
    import resources
    s, pr = tmp_path / "sys", tmp_path / "proc"
    for c in range(12):
        t = s / f"devices/system/cpu/cpu{c}/topology"
        t.mkdir(parents=True)
        (t / "core_id").write_text(f"{c}\n")
        (t / "physical_package_id").write_text("0\n")
    (s / "fs/cgroup").mkdir(parents=True)
    (s / "fs/cgroup/memory.max").write_text(f"{65536 * 2**20}\n")
    pr.mkdir()
    (pr / "meminfo").write_text(f"MemTotal: {4 * 65536 * 1024} kB\n")
    (pr / "self").mkdir()
    (pr / "self/cgroup").write_text("0::/\n")
    for name, want in (("sr.json", (11, "cpu")), ("sr-Dow.json", (7, "memory")),
                       ("sr-100K.json", (9, "memory"))):
        w = resources.decide(s, pr, affinity=set(range(12)),
                             env={"PER_RUN_MB": str(design.MB[name])})
        assert (w["workers"], w["limited_by"]) == want, name
    # A3 (R1 of the RV16 review, round 4): 6,553 MB is the largest limit with W = 9
    more = resources.decide(s, pr, affinity=set(range(12)),
                            env={"PER_RUN_MB": str(design.MB["sr-100K.json"] + 1)})
    assert more["workers"] == 8


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
spec = json.loads(Path(sys.argv[3]).read_text())
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"report": sys.argv[1:4], "dirs": sys.argv[5].split(","),
                         "partners": spec["partners"]["dirs"]}) + "\n")
"""


def test_the_phase_runs_with_its_own_workers_and_the_partner_requirement(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bl configs and the phase steps of
    both plans (the runner and the report are fakes). Without the partner summaries the first
    step is skipped; with them both steps run the three manifests with their own W into the
    result directories of phase1bl, and the report gets its own three directories and (in the
    spec) the three directories of round 16, which resolve from the repository root."""
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
    for q in ("phase1bl", "phase1bl-a"):
        shutil.copytree(CONFIGS / q, repo / "configs" / "gomea" / q)
    steps = ([dict(s) for s in load(SERVER / "plan17.json")["steps"][1:]]
             + [dict(s) for s in load(SERVER / "plan17b.json")["steps"]])
    for i, s in enumerate(steps):
        s["requires"] = [x for x in s["requires"] if x.startswith("summaries:")]
        s["id"] = f"{s['id']}-{i}"
    plan = {"name": "t17", "state_dir": "server-run17-t", "steps": steps}
    (repo / "tools" / "server" / "plan-t17.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS",
                        "PYTHONPATH")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="1024")

    def orchestrate():
        subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                        "tools/server/plan-t17.json"], cwd=repo, env=env, capture_output=True,
                       text=True, timeout=120, check=True)

    orchestrate()
    sd = res / "server-run17-t"
    st = load(sd / "state.json")["steps"]
    assert [st[s["id"]]["status"] for s in steps] == ["skipped", "skipped"]
    assert "130" not in st[steps[0]["id"]]["reason"]  # the reason of the first requirement
    assert "60 BASE-TLS summaries missing" in st[steps[0]["id"]]["reason"]
    assert not (res / "trace.jsonl").exists()
    # the 130 partner summaries of round 16
    for q in REQUIRES[1:]:
        _, rel, arm, names, span = q.split(":")
        for a in names.split(","):
            for k in range(81, 91):
                _write_summary(res, rel, a, arm, k)
    shutil.rmtree(sd)
    orchestrate()
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    runs_, reports = [t for t in trace if "manifest" in t], [t for t in trace if "report" in t]
    assert [t["manifest"] for t in runs_] == design.MANIFESTS * 2
    assert [t["config"] for t in runs_] == ["phase1bl-a"] * 3 + ["phase1bl"] * 3
    assert [t["splits"] for t in runs_] == [design.HALF_SPLITS] * 3 + [design.SPLITS] * 3
    for t in runs_:
        assert t["mem"] == design.MB[t["manifest"]] and t["out"] == "../gomea-results/phase1bl"
    dirs = [f"../gomea-results/phase1bl/phase1bl-{x}" for x in ("sr", "sr-100K", "sr-Dow")]
    assert [t["dirs"] for t in reports] == [dirs, dirs]
    for t in reports:
        assert t["partners"] == design.REAL_PARTNER_DIRS
        assert all((repo / d).resolve() == (res / d.split("/", 2)[2]).resolve()
                   and (repo / d).is_dir() for d in t["partners"])
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [(e.get("manifest"), e["per_run_mb"]) for e in events if e["event"] == "workers_decision"]
    assert wd == [(None, 4096), ("sr-Dow.json", 8192), ("sr-100K.json", 6553)] * 2
    st = load(sd / "state.json")["steps"]
    assert [st[s["id"]]["status"] for s in steps] == ["done", "done"]
