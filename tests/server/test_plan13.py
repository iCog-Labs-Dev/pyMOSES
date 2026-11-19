"""Server round 13: the plans (plan13.json, plan13-smoke.json) and the configuration of 1bh, as
SERVER-ROUND13-PREREGISTRATION.md registers them.

1bh: BASE-TLS against BASE (the confirmation), TLS-NP, TLS-G and TLS-LM against BASE-TLS, and
BASE-LEX and BASE-PS against BASE, with HC-LM as the accuracy baseline, on the 10 datasets,
fresh splits 51-60, 20,000 evaluations with no time limit. One phase (regression), no Boolean
phase, no extension, no choice step, no adoption rule and no chain."""

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
import run_table  # noqa: E402
import sr_datasets  # noqa: E402

# The arms of the pre-registration, flag for flag. BASE, HC-LM and BASE-TLS are the arms of 1bf.
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
        "--const-opt", "lm", "--const-opt-guard", "val",
        "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
         "--const-opt", "lm", "--const-opt-guard", "val"]
TLS = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
       "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
       "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
       "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
       "--const-opt", "none",
       "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
TLS_LM = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
          "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
          "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
          "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
          "--const-opt", "lm", "--const-opt-guard", "val",
          "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
ARMS = {
    "BASE": BASE,
    "HC-LM": HC_LM,
    "BASE-TLS": TLS,
    "TLS-NP": [("np" if x == "lt-b" else x) for x in TLS],
    "TLS-G": TLS,
    "TLS-LM": TLS_LM,
    "BASE-LEX": BASE + ["--exemplar-selection", "lexicase"],
    "BASE-PS": BASE + ["--const-opt-pole-check", "box"],
}
GUARD = {"TLS-G": "val"}  # the arm key "terms_guard"
NO_VAL = {"BASE-TLS", "TLS-NP", "TLS-G"}  # no --const-opt-guard, so no --const-opt-val
TERMS = {"BASE-TLS", "TLS-NP", "TLS-G", "TLS-LM"}
DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
SPLITS = list(range(51, 61))
MANIFESTS = ["sr.json", "sr-Dow.json"]
MB = {"sr-Dow.json": 8192}  # "Memory"
OTHER_MB = 5120
PREFIXES = ["phase1bh", "server-smoke13-1bh"]
REQUIRES = [f"datasets:{','.join(DATASETS)}:51-60"]
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
    out = {}
    for f in ms:
        m = load(f)
        for d in m["datasets"]:
            for s in m["splits"]:
                assert (d, s) not in out, (f, d, s)
                out[(d, s)] = list(m["arms"])
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


# ---- the arms of 1bh


@pytest.mark.parametrize("prefix", PREFIXES)
def test_arm_flags_match_the_preregistration(prefix):
    q = CONFIGS / prefix
    assert [f.name for f in stage1(q)] == MANIFESTS
    for f in stage1(q):
        m = load(f)
        assert list(m["arms"]) == list(ARMS)
        assert {a: s["argv"] for a, s in m["arms"].items()} == ARMS, f
        for a, s in m["arms"].items():
            if a in GUARD:
                assert set(s) == {"binary", "budget", "argv", "terms_guard"}
                assert s["terms_guard"] == GUARD[a]
            else:
                assert set(s) == {"binary", "budget", "argv"}, (f, a)  # no seed block


def test_base_hc_lm_and_base_tls_are_the_arms_of_1bf():
    bf = load(CONFIGS / "phase1bf" / "sr.json")["arms"]
    for f in stage1(CONFIGS / "phase1bh"):
        m = load(f)["arms"]
        for a in ("BASE", "HC-LM", "BASE-TLS"):
            assert m[a] == bf[a], (f, a)
        assert m["TLS-G"]["argv"] == bf["BASE-TLS"]["argv"]


def test_each_new_arm_changes_only_its_own_flags():
    """The table "Arms": TLS-NP has np in place of lt-b; TLS-G has the argv of BASE-TLS; TLS-LM
    has --const-opt lm --const-opt-guard val in place of --const-opt none; BASE-LEX and BASE-PS
    add one flag at the end of BASE."""
    def pairs(argv):
        return [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]

    tls = pairs(TLS)
    assert pairs(ARMS["TLS-NP"]) == [(k, "np" if k == "--gomea-fos" else v) for k, v in tls]
    assert ARMS["TLS-G"] == TLS
    i = tls.index(("--const-opt", "none"))
    assert pairs(ARMS["TLS-LM"]) == tls[:i] + [("--const-opt", "lm"),
                                               ("--const-opt-guard", "val")] + tls[i + 1:]
    assert pairs(ARMS["BASE-LEX"]) == pairs(BASE) + [("--exemplar-selection", "lexicase")]
    assert pairs(ARMS["BASE-PS"]) == pairs(BASE) + [("--const-opt-pole-check", "box")]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_manifest_asks_for_the_clipped_test_nmse(prefix):
    """The manifest key "clip_test_nmse": true ("Descriptive for every arm"); expand() gives
    it to every run, so every summary of the phase has test_nmse_clipped."""
    for f in stage1(CONFIGS / prefix):
        m = load(f)
        assert m["clip_test_nmse"] is True
        jobs = runner.expand(m, Path("/o"), REPO)
        assert jobs and all(j["clip_test_nmse"] is True for j in jobs)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_manifest_asks_for_the_plain_scaling_fields(prefix):
    """Backlog 12.2: the plain-scaling fields of the TLS arms need the manifest key
    "plain_nmse": true. The round 13 manifests have it, so a rerun writes them as round 13
    did; expand() gives it to every run."""
    for f in stage1(CONFIGS / prefix):
        m = load(f)
        assert m["plain_nmse"] is True
        jobs = runner.expand(m, Path("/o"), REPO)
        assert jobs and all(j["plain_nmse"] is True for j in jobs)
        assert any(runner._linear_scaling(j["argv"]) == "terms" for j in jobs)


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1bh"):
        m = load(f)
        assert m["wall_cap_s"] == 36000
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": 20000}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_runner_gives_each_arm_m_the_log_and_the_validation_table_of_the_guard(arm, tmp_path):
    """`-m 20000` comes before the arm's flags; `events_argv` adds `--gomea-log`; `_run_one`
    adds `--const-opt-val` for the LM guard only: so for TLS-LM, and not for BASE-TLS, TLS-NP
    and TLS-G."""
    job = {"binary": ".venv/bin/moses", "seed": 51, "budget": {"evals": 20000},
           "argv": ARMS[arm]}
    out = runner.moses_argv(job, tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    i = out.index("-m")
    assert out[i + 1] == "20000" and out[i + 2:-2] == ARMS[arm]
    full = run_argv(job, tmp_path, 100)
    assert runner._linear_scaling(ARMS[arm]) == ("terms" if arm in TERMS else "1")
    if arm in NO_VAL:
        assert runner._const_opt(ARMS[arm]) == "none"
        assert runner._const_opt_guard(ARMS[arm]) == "none"
        assert not any(t.startswith(("--const-opt-val", "--const-opt-guard")) for t in full)
    else:
        assert runner._const_opt(ARMS[arm]) == "lm"
        assert runner._const_opt_guard(ARMS[arm]) == "val"
        assert full[-2:] == ["--const-opt-val", str(tmp_path / "val_scaled.csv")]


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_command_line_parses_with_the_real_cli(arm, tmp_path):
    """The command line of each arm on each dataset, as the runner builds it, passes every
    check of the real pymoses CLI, and gives the options of the pre-registration."""
    for d in DATASETS:
        job = {"binary": ".venv/bin/moses", "seed": 51, "budget": {"evals": 20000},
               "argv": ARMS[arm]}
        pp, ns = cli(run_argv(job, tmp_path, N_TRAIN[d]))
        assert ns.max_evals == 20000
        assert pp.linear_scaling == ("terms" if arm in TERMS else 1)
        assert ns.const_opt_val == ("" if arm in NO_VAL else str(tmp_path / "val_scaled.csv"))
        assert pp.meta_params.exemplar_selection == (
            "lexicase" if arm == "BASE-LEX" else "boltzmann")
        if arm == "HC-LM":
            assert ns.opt_algo == "hc" and ns.const_opt == "lm"
            continue
        gp = pp.gomea_params
        assert ns.opt_algo == "gomea" and gp.dynamic()
        assert gp.fos == ("np" if arm == "TLS-NP" else "lt-b")
        assert gp.const_opt == ("none" if arm in NO_VAL else "lm")
        assert gp.const_opt_guard == ("none" if arm in NO_VAL else "val")
        assert gp.const_opt_pole_check == ("box" if arm == "BASE-PS" else "none")
        assert gp.const_opt_reserve == 0 and gp.const_opt_top_k == 0
        assert gp.const_opt_during == "none"
        assert gp.max_evals_per_deme is None


def test_the_cli_refuses_the_validation_table_for_the_tls_arms_without_lm():
    """Why the runner must not give BASE-TLS, TLS-NP and TLS-G `--const-opt-val`."""
    for arm in sorted(NO_VAL):
        with pytest.raises(ValueError, match="--const-opt-val"):
            cli(["-i", "t.csv", "-u", "y", "-m", "20000", *ARMS[arm], "--const-opt-val", "v.csv"])


def test_tls_g_has_the_guard_key_and_the_search_of_base_tls(tmp_path):
    """expand() carries "terms_guard" for TLS-G only; the MOSES command line of TLS-G is that of
    BASE-TLS, so the search is the same."""
    jobs = runner.expand(load(CONFIGS / "phase1bh" / "sr.json"), Path("/o"), REPO)
    for j in jobs:
        assert j.get("terms_guard") == GUARD.get(j["arm"]), j["run_id"]
    by = {(j["task"], j["seed"], j["arm"]): j for j in jobs}
    for d in DATASETS:
        if d == "Dow":
            continue
        for k in SPLITS:
            g, t = by[(d, k, "TLS-G")], by[(d, k, "BASE-TLS")]
            assert runner.moses_argv(g, tmp_path, N_TRAIN[d]) == runner.moses_argv(
                t, tmp_path, N_TRAIN[d])
    bad = load(CONFIGS / "phase1bh" / "sr.json")
    bad["arms"]["BASE"]["terms_guard"] = "val"
    with pytest.raises(ValueError, match="--linear-scaling terms"):
        runner.expand(bad, Path("/o"), REPO)


def test_no_arm_uses_the_placeholder():
    for argv in ARMS.values():
        assert runner.resolve_argv(argv, 534) == argv
        assert not any("n_train" in t for t in argv)


# ---- datasets, splits, memory


def test_datasets_and_fresh_splits_and_no_extension():
    q = CONFIGS / "phase1bh"
    assert set(runs(stage1(q))) == {(d, s) for d in DATASETS for s in SPLITS}
    assert sorted(q.glob("*-ext-*.json")) == []
    assert sorted(f.name for f in q.iterdir()) == ["spec.json", "sr-Dow.json", "sr.json"]
    assert load(q / "sr.json")["datasets"] == [d for d in DATASETS if d != "Dow"]
    assert load(q / "sr-Dow.json")["datasets"] == ["Dow"]
    for f in stage1(q):
        assert load(f)["name"] == f"phase1bh-{f.stem}"
        assert load(f)["splits"] == SPLITS
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_splits_51_to_60_are_new():
    """No earlier regression manifest uses splits 51-60."""
    for f in CONFIGS.glob("*/*.json"):
        if f.parent.name in PREFIXES:
            continue
        m = load(f)
        if isinstance(m, dict) and isinstance(m.get("splits"), list):
            assert not set(m["splits"]) & set(SPLITS), f


def test_run_count_is_800():
    assert sum(len(a) for a in runs(stage1(CONFIGS / "phase1bh")).values()) == 800
    jobs = [j for f in stage1(CONFIGS / "phase1bh")
            for j in runner.expand(load(f), Path("/o"), REPO)]
    assert len(jobs) == 800 and len({j["run_id"] for j in jobs}) == 800


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == MB.get(f.name, OTHER_MB), f


# ---- the spec of 1bh


PRIMARY = [["basetls-vs-base", "BASE-TLS", "BASE"], ["tlsnp-vs-basetls", "TLS-NP", "BASE-TLS"],
           ["tlsg-vs-basetls", "TLS-G", "BASE-TLS"], ["tlslm-vs-basetls", "TLS-LM", "BASE-TLS"],
           ["baselex-vs-base", "BASE-LEX", "BASE"], ["baseps-vs-base", "BASE-PS", "BASE"]]
GAP = [["base-vs-hclm", "BASE", "HC-LM"], ["basetls-vs-hclm", "BASE-TLS", "HC-LM"],
       ["tlsnp-vs-hclm", "TLS-NP", "HC-LM"], ["tlsg-vs-hclm", "TLS-G", "HC-LM"],
       ["tlslm-vs-hclm", "TLS-LM", "HC-LM"], ["baselex-vs-hclm", "BASE-LEX", "HC-LM"],
       ["baseps-vs-hclm", "BASE-PS", "HC-LM"]]
DESCRIPTIVE = ([["tlsnp-vs-base", "TLS-NP", "BASE", "test_nmse"],
                ["tlsg-vs-base", "TLS-G", "BASE", "test_nmse"],
                ["tlslm-vs-base", "TLS-LM", "BASE", "test_nmse"]]
               + [[f"{n}-clipped", a, b, "test_nmse_clipped"] for n, a, b in PRIMARY])


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bh" if prefix == "phase1bh" else "smoke13-1bh")
    assert s["metrics"] == dict.fromkeys(DATASETS, "test_nmse")
    assert s["primary"] == []
    # primary (6), one Holm family
    assert s["pooled_primary"] == [c + ["test_nmse"] for c in PRIMARY]
    want2 = ([c + ["test_nmse"] for c in GAP]
             + [[f"{n}-size", a, b, "program_size"] for n, a, b in PRIMARY]
             + [[f"{n}-cpu", a, b, "cpu_s"] for n, a, b in PRIMARY]
             + DESCRIPTIVE)
    assert s["pooled_secondary"] == want2 and len(want2) == 7 + 12 + 9
    # the gap to HC-LM (7) and the descriptive contrasts (9) are families of their own; size
    # and CPU (12) are the remaining family
    assert s["pooled_secondary_families"] == [[c[0] for c in GAP], [c[0] for c in DESCRIPTIVE]]
    per = [[d, n, a, b, "test_nmse"] for n, a, b in PRIMARY for d in DATASETS]
    assert s["secondary"] == per and len(per) == 60
    assert s["secondary_verdicts"] is False and s["extend"] is False
    assert "secondary_families" not in s and "descriptive" not in s
    # round 13: the size of a TLS arm with its term weights, the per-run table and the list
    # above 100, and the check that TLS-G and BASE-TLS printed the same candidates
    assert s["size_terms"] is True and s["run_table"] is True
    assert s["same_search"] == [["TLS-G", "BASE-TLS"]]
    assert "test_nmse_clipped" in phase_report.METRICS


def test_holm_families_at_100_pairs():
    """With 100 pooled pairs every family (6, 7, 12, 9) can give a verdict; the 60 per-dataset
    contrasts (10 pairs each) cannot (60 * 2/1024 > 0.025): they are descriptive."""
    p100 = wilcoxon_signed_rank([float(i + 1) for i in range(100)])[1]
    for n in (6, 7, 12, 9):
        assert holm([p100] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 60)[0] > phase_report.ALPHA


def _summary(task, arm, split, nmse, status="ok"):
    s = {"arm": arm, "task": task, "seed": split, "status": status,
         "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse / 2,
         "val_nmse": nmse, "test_nmse_clipped": nmse if status == "ok" else None,
         "cpu_s": 10.0, "n_evals": 20000, "final_program": "+(*($1 $2) $3)",
         "const_opt": {"mode": "lm", "lm_calls": 2, "tuning_evals": 100, "tuning_share": 0.005}}
    if arm in TERMS:
        s["terms"] = {"n_terms": 2, "fallback": False, "reason": None, "weights": [0, 1, 1]}
        if arm == "TLS-G":
            s["terms"]["guard"] = "terms" if split % 2 else "plain"
    return s


def test_the_report_of_the_real_spec(tmp_path):
    """phase_report.py on the 1bh spec with 800 made-up runs: 6 + 28 pooled rows and 60
    per-dataset rows, the four Holm families, verdicts only on the pooled rows, runs.csv with
    800 rows, the list above 100 and the same-search check."""
    factor = {"BASE": 1.0, "HC-LM": 0.5, "BASE-TLS": 0.85, "TLS-NP": 1.0, "TLS-G": 0.85,
              "TLS-LM": 0.85, "BASE-LEX": 1.0, "BASE-PS": 1.0}
    root = tmp_path / "r"
    end = json.dumps({"type": "end", "generations": 4, "termination_reason": "stagnation",
                      "ledger": {"requests": 10}, "dim": 50})
    for d in DATASETS:
        for k in SPLITS:
            for arm, f in factor.items():
                failed = arm == "TLS-LM" and d == "Dow" and k == 51  # a memory failure
                nmse = 150.0 if (arm == "BASE-PS" and d == "Bos" and k == 55) else (
                    f * (0.1 + k / 1000 + DATASETS.index(d) / 1000))
                s = _summary(d, arm, k, nmse, "error" if failed else "ok")
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))
                (rd / "events.jsonl").write_text((end + "\n") * (3 if f < 1 else 1))
                progs = ["$1", "$2"] if not (arm == "TLS-G" and d == "Air" and k == 51) else ["$3"]
                (rd / "candidates.json").write_text(json.dumps([{"program": p} for p in progs]))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1bh" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    pooled = [r for r in rows if r["problem"] == "pooled"]
    assert len(pooled) == 34 and len(rows) == 94
    assert all(r["pairs"] == "100" for r in pooled)
    fam = {r["contrast"]: (r["family"], r["holm_family"]) for r in pooled}
    assert fam["tlsg-vs-basetls"] == ("primary", "")
    assert fam["baseps-vs-hclm"] == ("secondary", "pooled_secondary/" + "+".join(
        c[0] for c in GAP))
    assert fam["basetls-vs-base-cpu"] == ("secondary", "pooled_secondary")
    assert fam["tlslm-vs-base"] == ("secondary", "pooled_secondary/" + "+".join(
        c[0] for c in DESCRIPTIVE))
    assert sum(r["holm_family"] == "pooled_secondary" for r in pooled) == 12
    assert sum(r["metric"] == "log_ratio:test_nmse_clipped" for r in pooled) == 6
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:basetls-vs-base"] == "A better" and v["pooled:tlsnp-vs-basetls"] == "A worse"
    assert v["pooled:tlsg-vs-basetls"] == "tie"
    assert v["pooled:base-vs-hclm"] == "A worse"
    assert all(v[f"{d}:baselex-vs-base"] is None for d in DATASETS)
    assert sum(r["verdict"] == "" for r in rows) == 60
    # the size of a TLS arm counts its 2 term weights; TLS-G only with guard "terms"
    # ("+(*($1 $2) $3)": 2 "(" + 5 atoms = 7 nodes)
    size = {r["contrast"]: r for r in pooled if r["metric"] == "log_ratio:program_size"}
    assert float(size["basetls-vs-base-size"]["median_a"]) == 9.0
    assert float(size["basetls-vs-base-size"]["median_b"]) == 7.0
    assert float(size["tlsg-vs-basetls-size"]["median_a"]) == 8.0  # half 9, half 7
    agg = list(csv.DictReader((out / "aggregate.csv").open()))
    assert len(agg) == 80
    for col in ("median_deme_count", "median_tuning_share", "median_size", "median_nodes",
                "median_lm_calls"):
        assert col in agg[0], col
    table = list(csv.DictReader((out / "runs.csv").open()))
    assert len(table) == 800 and list(table[0]) == list(run_table.RUN_FIELDS)
    above = list(csv.DictReader((out / "nmse_above_100.csv").open()))
    assert [(r["dataset"], r["arm"], r["split"]) for r in above] == [("Bos", "BASE-PS", "55")]
    ss = json.loads((out / "same_search.json").read_text())
    assert ss == [{"a": "TLS-G", "b": "BASE-TLS", "pairs": 100, "same": 99, "differ": 1,
                   "missing": 0}]


# ---- the plans


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan13.json", "server-round13-2026-11", "server-run13", "phase1bh", "phase1bh"),
    ("plan13-smoke.json", "server-round13-smoke", "server-run13-smoke", "server-smoke13-1bh",
     "smoke13-1bh"),
])
def test_plan_is_s0_then_1bh(plan, name, state_dir, prefix, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", sid]
    assert "per_run_mb" not in p["steps"][0] and "manifest_per_run_mb" not in p["steps"][0]
    a = p["steps"][1]
    assert a["phase"] == prefix
    assert "config_dir" not in a and not a.get("pilot") and not a.get("set_commit_in")
    # no seeded arm, so no GP-GOMEA build; the datasets with the fresh splits 51-60
    assert a["requires"] == REQUIRES
    assert a["per_run_mb"] == OTHER_MB and a["manifest_per_run_mb"] == MB
    for f in stage1(CONFIGS / prefix):
        assert a["manifest_per_run_mb"].get(f.name, a["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((CONFIGS / prefix / n).exists() for n in MB)
    assert "then" not in p


def test_the_requirement_needs_the_fresh_splits(tmp_path, monkeypatch):
    """A dataset set built before round 13 (splits 1-50) does not meet the requirement."""
    sys.path.insert(0, str(SERVER))
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan13.json")
    for a in DATASETS:
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
        for k in range(1, 51):
            (d / "splits" / f"s{k}").mkdir()
            for t in ("train", "val", "test"):
                (d / "splits" / f"s{k}" / f"{t}.csv").write_text("x1,y\n")
    why = r.requirement(REQUIRES[0])
    assert why and "51-60" in why
    for a in DATASETS:
        for k in SPLITS:
            s = tmp_path / "phase1f" / "datasets" / a / "splits" / f"s{k}"
            s.mkdir()
            for t in ("train", "val", "test"):
                (s / f"{t}.csv").write_text("x1,y\n")
    assert r.requirement(REQUIRES[0]) is None


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan13.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    s0 = load(SERVER / "plan13-smoke.json")["steps"][0]
    args = s0["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_sr_datasets.py",
              "tests/gomea/test_round13_runner_report.py",
              "tests/gomea/test_round13_lm_identity.py", "tests/gomea/test_const_opt_terms.py",
              "tests/gomea/test_const_opt_pole.py",
              "tests/gomea/test_exemplar_selection_identity.py",
              "tests/moses/metapopulation/test_lexicase.py"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


def test_no_plan_chains_to_or_from_round_13():
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round13", "smoke13"), f
    for f in ("plan13.json", "plan13-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan", [("round13", "plan13.json"),
                                        ("smoke13", "plan13-smoke.json")])
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
    assert "round13|smoke13" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan_with_the_fresh_splits():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text
    # round 14 adds splits 61-70 (SEEDS 1-70); round 13 needs that set-up builds 51-60
    assert f'-eq {8 * len(sr_datasets.SEEDS)} ]' in text and max(sr_datasets.SEEDS) == 70
    assert '-type f | wc -l)" -eq 560 ]' in text
    assert set(SPLITS) <= set(sr_datasets.SEEDS)
    assert set(range(61, 71)) <= set(sr_datasets.SEEDS)


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bh", CONFIGS / "server-smoke13-1bh"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1bh", "server-smoke13-1bh")
        assert m["datasets"] == fm["datasets"] and m["splits"] == [51, 52]
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert m["datasets_dir"] == fm["datasets_dir"]
        assert m["clip_test_nmse"] is fm["clip_test_nmse"] is True
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert {k: v for k, v in s.items() if k != "budget"} == {
                k: v for k, v in fm["arms"][a].items() if k != "budget"}
            assert s["budget"] == {"evals": 1000}
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


def test_server_workers_are_7_for_dow_and_11_otherwise(tmp_path):
    """The section "Memory" of the pre-registration, on the 12-CPU 64 GB server."""
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


def test_the_phase_runs_with_its_own_workers(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bh configs; the runner and the
    report are fakes. sr.json, then sr-Dow.json with its own W."""
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
    shutil.copytree(CONFIGS / "phase1bh", repo / "configs" / "gomea" / "phase1bh")
    steps = [dict(s) for s in load(SERVER / "plan13.json")["steps"][1:]]
    steps[0].pop("requires")
    plan = {"name": "t13", "state_dir": "server-run13-t", "steps": steps}
    (repo / "tools" / "server" / "plan-t13.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS",
                        "PYTHONPATH")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="1024", MANIFEST_WORKERS="sr.json=99")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t13.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    assert [t["manifest"] for t in trace] == MANIFESTS
    for t in trace:
        assert t["workers"] == _w(MB.get(t["manifest"], OTHER_MB)), t
    sd = res / "server-run13-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [e for e in events if e["event"] == "workers_decision"]
    assert [(e.get("manifest"), e["per_run_mb"]) for e in wd] == [
        (None, OTHER_MB), ("sr-Dow.json", 8192)]
    att = [e for e in events if e["event"] == "phase_attempt"]
    assert att[0]["manifest_workers"] == {n: _w(mb) for n, mb in MB.items()}
    st = load(sd / "state.json")["steps"]
    assert st["phase1bh"]["status"] == "done"
