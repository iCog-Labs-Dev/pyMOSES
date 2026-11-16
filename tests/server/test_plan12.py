"""Server round 12: the plans (plan12.json, plan12-smoke.json) and the configurations of 1bf and
1bg, as SERVER-ROUND12-PREREGISTRATION.md registers them.

1bf: BASE against BASE-R (T1.5), BASE-NP (T2.1), BASE-TLS (T2.2), BASE-LME (T2.3) and BASE-DIV
(T2.4), and BASE-K5 (T1.6) against BASE-R, with HC-LM as the accuracy baseline, on the 10
datasets, fresh splits 41-50, 20,000 evaluations with no time limit. 1bg: node-proximity
linkage (NP) against RT, UNI and LTB on mux11, fresh seeds 3101-3110, 3,600 s per run. No
extension, no choice step, no adoption rule and no chain."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.gomea.manifest import check_code_commit, expand_runs, load_manifest
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402
import sr_datasets  # noqa: E402

# The arms of the pre-registration, flag for flag. BASE and HC-LM are the arms of 1be.
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
ARMS = {
    "BASE": BASE,
    "HC-LM": HC_LM,
    "BASE-R": BASE + ["--const-opt-reserve", "500"],
    "BASE-K5": BASE + ["--const-opt-reserve", "500", "--const-opt-top-k", "5"],
    "BASE-NP": [("np" if x == "lt-b" else x) for x in BASE],
    "BASE-TLS": TLS,
    "BASE-LME": BASE + ["--const-opt-during", "elite"],
    "BASE-DIV": BASE + ["--diversity-pressure", "{6*n_train}", "--diversity-autoscale", "0"],
}
DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
SPLITS = list(range(41, 51))
MANIFESTS = ["sr.json", "sr-Dow.json"]
MB = {"sr-Dow.json": 8192}  # "Memory"
OTHER_MB = 5120
MUX_MB = 3072
PREFIXES = ["phase1bf", "server-smoke12-1bf"]
BOOL_PREFIXES = ["phase1bg", "server-smoke12-1bg"]
REQUIRES = [f"datasets:{','.join(DATASETS)}:41-50"]
# training rows of each dataset (the same for every split: n - 2 * (n // 4))
N_TRAIN = {d: json.loads((REPO / "data" / "gomea-sr" / d / "splits.json").read_text())["n_train"]
           for d in DATASETS}
# 1bg: the Boolean setting of round 6 with each FOS
BOOL_BASE = ["-a", "gomea", "--gomea-fos", "{FOS}", "--gomea-pop-size", "dynamic",
             "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
             "--gomea-max-evals-per-deme", "0", "--gomea-genotype-memo", "1",
             "--knob-probe-baseline", "reduced"]
FOS = {"NP": "np", "LTB": "lt-b", "RT": "rt", "UNI": "uni"}
BOOL_ARMS = {a: [f if x == "{FOS}" else x for x in BOOL_BASE] for a, f in FOS.items()}


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


# ---- the arms of 1bf


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


def test_base_and_hc_lm_are_the_arms_of_1be():
    be = load(CONFIGS / "phase1be" / "sr.json")["arms"]
    for f in stage1(CONFIGS / "phase1bf"):
        m = load(f)["arms"]
        assert m["BASE"] == be["BASE"] and m["HC-LM"] == be["HC-LM"]
        assert m["BASE-R"] == be["BASE-R"]


def test_each_new_arm_changes_only_its_own_flags():
    """The table "Phase A arms": BASE-NP has np in place of lt-b; BASE-TLS has terms and
    --const-opt none, and no guard; the others add flags at the end of BASE."""
    def pairs(argv):
        return [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]

    base = pairs(BASE)
    assert pairs(ARMS["BASE-NP"]) == [(k, "np" if k == "--gomea-fos" else v) for k, v in base]
    want = [(k, {"--linear-scaling": "terms", "--const-opt": "none"}.get(k, v))
            for k, v in base if k != "--const-opt-guard"]
    assert pairs(ARMS["BASE-TLS"]) == want
    for arm, extra in [("BASE-R", [("--const-opt-reserve", "500")]),
                       ("BASE-K5", [("--const-opt-reserve", "500"), ("--const-opt-top-k", "5")]),
                       ("BASE-LME", [("--const-opt-during", "elite")]),
                       ("BASE-DIV", [("--diversity-pressure", "{6*n_train}"),
                                     ("--diversity-autoscale", "0")])]:
        assert pairs(ARMS[arm]) == base + extra, arm


def test_budget_is_evaluations_with_no_time_limit():
    for f in stage1(CONFIGS / "phase1bf"):
        m = load(f)
        assert m["wall_cap_s"] == 36000
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": 20000}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_runner_gives_each_arm_m_and_the_log(arm, tmp_path):
    """`-m 20000` comes before the arm's flags; `events_argv` adds `--gomea-log` (HC-LM too:
    -a hc with LM); `_run_one` adds `--const-opt-val` for the guard only, so not for BASE-TLS."""
    job = {"binary": ".venv/bin/moses", "seed": 41, "budget": {"evals": 20000},
           "argv": ARMS[arm]}
    out = runner.moses_argv(job, tmp_path, 100)
    assert out[-2:] == ["--gomea-log", str(tmp_path / "events.jsonl")]
    i = out.index("-m")
    assert out[i + 1] == "20000" and out[i + 2:-2] == runner.resolve_argv(ARMS[arm], 100)
    full = run_argv(job, tmp_path, 100)
    if arm == "BASE-TLS":
        assert runner._const_opt(ARMS[arm]) == "none"
        assert runner._const_opt_guard(ARMS[arm]) == "none"
        assert runner._linear_scaling(ARMS[arm]) == "terms"
        assert not any(t.startswith("--const-opt-val") for t in full)
        assert not any(t.startswith("--const-opt-guard") for t in full)
    else:
        assert runner._const_opt(ARMS[arm]) == "lm"
        assert runner._const_opt_guard(ARMS[arm]) == "val"
        assert full[-2:] == ["--const-opt-val", str(tmp_path / "val_scaled.csv")]


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_command_line_parses_with_the_real_cli(arm, tmp_path):
    """The command line of each arm on each dataset, as the runner builds it, passes every
    check of the real pymoses CLI, and gives the options of the pre-registration."""
    for d in DATASETS:
        job = {"binary": ".venv/bin/moses", "seed": 41, "budget": {"evals": 20000},
               "argv": ARMS[arm]}
        pp, ns = cli(run_argv(job, tmp_path, N_TRAIN[d]))
        assert ns.max_evals == 20000
        assert pp.linear_scaling == ("terms" if arm == "BASE-TLS" else 1)
        assert ns.diversity_pressure == (6.0 * N_TRAIN[d] if arm == "BASE-DIV" else 0.0)
        assert ns.diversity_autoscale is False
        assert ns.const_opt_val == ("" if arm == "BASE-TLS" else str(tmp_path / "val_scaled.csv"))
        if arm == "HC-LM":
            assert ns.opt_algo == "hc" and ns.const_opt == "lm"
            continue
        gp = pp.gomea_params
        assert ns.opt_algo == "gomea" and gp.dynamic()
        assert gp.fos == ("np" if arm == "BASE-NP" else "lt-b")
        assert gp.const_opt == ("none" if arm == "BASE-TLS" else "lm")
        assert gp.const_opt_guard == ("none" if arm == "BASE-TLS" else "val")
        assert gp.const_opt_reserve == (500 if arm in ("BASE-R", "BASE-K5") else 0)
        assert gp.const_opt_top_k == (5 if arm == "BASE-K5" else 0)
        assert gp.const_opt_during == ("elite" if arm == "BASE-LME" else "none")
        assert gp.max_evals_per_deme is None


def test_the_cli_refuses_the_validation_table_for_base_tls():
    """Why the runner must not give BASE-TLS `--const-opt-val`."""
    with pytest.raises(ValueError, match="--const-opt-val"):
        cli(["-i", "t.csv", "-u", "y", "-m", "20000", *TLS, "--const-opt-val", "v.csv"])


@pytest.mark.parametrize("d", DATASETS)
def test_base_div_gets_6_times_the_training_rows_of_its_dataset(d, tmp_path):
    """P = 6 x training rows (T2.4): one third of the selection window 0.3 x -v."""
    job = {"binary": ".venv/bin/moses", "seed": 41, "budget": {"evals": 20000},
           "argv": ARMS["BASE-DIV"]}
    out = runner.moses_argv(job, tmp_path, N_TRAIN[d])
    p = int(out[out.index("--diversity-pressure") + 1])
    v = int(out[out.index("-v") + 1])
    assert p == 6 * N_TRAIN[d] and 3 * p == 0.3 * v
    assert N_TRAIN[d] == sr_datasets.DATASETS[d][4] - 2 * (sr_datasets.DATASETS[d][4] // 4)


def test_only_base_div_uses_the_placeholder():
    for arm, argv in ARMS.items():
        resolved = runner.resolve_argv(argv, 534)
        if arm == "BASE-DIV":
            assert resolved == BASE + ["--diversity-pressure", "3204", "--diversity-autoscale", "0"]
        else:
            assert resolved == argv and all(a is b for a, b in zip(resolved, argv, strict=True))
            assert not any("n_train" in t for t in argv)


def test_expand_keeps_the_placeholder_until_the_run():
    jobs = runner.expand(load(CONFIGS / "phase1bf" / "sr.json"), Path("/o"), REPO)
    div = [j for j in jobs if j["arm"] == "BASE-DIV"]
    assert len(div) == 90 and all(j["argv"] == ARMS["BASE-DIV"] for j in div)


# ---- datasets, splits, memory


def test_datasets_and_fresh_splits_and_no_extension():
    q = CONFIGS / "phase1bf"
    assert set(runs(stage1(q))) == {(d, s) for d in DATASETS for s in SPLITS}
    assert sorted(q.glob("*-ext-*.json")) == []
    assert load(q / "sr.json")["datasets"] == [d for d in DATASETS if d != "Dow"]
    assert load(q / "sr-Dow.json")["datasets"] == ["Dow"]
    for f in stage1(q):
        assert load(f)["name"] == f"phase1bf-{f.stem}"
        assert load(f)["splits"] == SPLITS
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_splits_41_to_50_are_new():
    """No earlier regression manifest uses splits 41-50."""
    for f in CONFIGS.glob("*/*.json"):
        if f.parent.name in PREFIXES:
            continue
        m = load(f)
        if isinstance(m, dict) and isinstance(m.get("splits"), list):
            assert not set(m["splits"]) & set(SPLITS), f


def test_run_count_is_800():
    assert sum(len(a) for a in runs(stage1(CONFIGS / "phase1bf")).values()) == 800


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limits(prefix):
    for f in stage1(CONFIGS / prefix):
        assert load(f)["mem_limit_mb"] == MB.get(f.name, OTHER_MB), f


# ---- the spec of 1bf


PRIMARY = [["baser-vs-base", "BASE-R", "BASE"], ["basek5-vs-baser", "BASE-K5", "BASE-R"],
           ["basenp-vs-base", "BASE-NP", "BASE"], ["basetls-vs-base", "BASE-TLS", "BASE"],
           ["baselme-vs-base", "BASE-LME", "BASE"], ["basediv-vs-base", "BASE-DIV", "BASE"]]
GAP = [["base-vs-hclm", "BASE", "HC-LM"], ["baser-vs-hclm", "BASE-R", "HC-LM"],
       ["basek5-vs-hclm", "BASE-K5", "HC-LM"], ["basenp-vs-hclm", "BASE-NP", "HC-LM"],
       ["basetls-vs-hclm", "BASE-TLS", "HC-LM"], ["baselme-vs-hclm", "BASE-LME", "HC-LM"],
       ["basediv-vs-hclm", "BASE-DIV", "HC-LM"]]
DESCRIPTIVE = ["basek5-vs-base", "BASE-K5", "BASE"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bf" if prefix == "phase1bf" else "smoke12-1bf")
    assert s["metrics"] == dict.fromkeys(DATASETS, "test_nmse")
    assert s["primary"] == []
    # primary (6): BASE-K5 against BASE-R (change of 2026-11-16), the others against BASE
    assert s["pooled_primary"] == [c + ["test_nmse"] for c in PRIMARY]
    want2 = ([c + ["test_nmse"] for c in GAP]
             + [[f"{n}-size", a, b, "program_size"] for n, a, b in PRIMARY]
             + [[f"{n}-cpu", a, b, "cpu_s"] for n, a, b in PRIMARY]
             + [DESCRIPTIVE + ["test_nmse"]])
    assert s["pooled_secondary"] == want2 and len(want2) == 7 + 12 + 1
    # the gap to HC-LM (7) and BASE-K5 against BASE (1) are families of their own; size and
    # CPU (12) are the remaining family
    assert s["pooled_secondary_families"] == [[c[0] for c in GAP], [DESCRIPTIVE[0]]]
    per = [[d, n, a, b, "test_nmse"] for n, a, b in PRIMARY for d in DATASETS]
    assert s["secondary"] == per and len(per) == 60
    assert s["secondary_verdicts"] is False and s["extend"] is False
    assert "secondary_families" not in s and "descriptive" not in s


def test_holm_families_at_100_pairs():
    """With 100 pooled pairs every family (6, 7, 12, 1) can give a verdict; the 60 per-dataset
    contrasts (10 pairs each) cannot (60 * 2/1024 > 0.025): they are descriptive."""
    p100 = wilcoxon_signed_rank([float(i + 1) for i in range(100)])[1]
    for n in (6, 7, 12, 1):
        assert holm([p100] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 60)[0] > phase_report.ALPHA


def _summary(task, arm, split, nmse, status="ok"):
    return {"arm": arm, "task": task, "seed": split, "status": status,
            "test_nmse": nmse if status == "ok" else None, "train_nmse": nmse, "cpu_s": 10.0,
            "n_evals": 20000, "final_program": "(+ $x1 $x2)",
            "const_opt": {"mode": "lm", "lm_calls": 2, "tuning_evals": 100,
                          "tuning_share": 0.005}}


def test_the_report_of_the_real_spec(tmp_path):
    """phase_report.py on the 1bf spec with 800 made-up runs: 6 + 20 pooled rows and 60
    per-dataset rows, the four Holm families, verdicts only on the pooled rows."""
    factor = {"BASE": 1.0, "HC-LM": 0.5, "BASE-R": 1.0, "BASE-K5": 0.7, "BASE-NP": 1.0,
              "BASE-TLS": 1.0, "BASE-LME": 0.7, "BASE-DIV": 1.0}
    root = tmp_path / "r"
    end = json.dumps({"type": "end", "generations": 4, "termination_reason": "stagnation",
                      "ledger": {"requests": 10}})
    for d in DATASETS:
        for k in SPLITS:
            for arm, f in factor.items():
                failed = arm == "BASE-TLS" and d == "Dow" and k == 41  # a memory failure
                s = _summary(d, arm, k, f * (0.1 + k / 1000 + DATASETS.index(d) / 1000),
                             "error" if failed else "ok")
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))
                (rd / "events.jsonl").write_text((end + "\n") * (3 if f < 1 else 1))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1bf" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    pooled = [r for r in rows if r["problem"] == "pooled"]
    assert len(pooled) == 26 and len(rows) == 86
    assert all(r["pairs"] == "100" for r in pooled)
    fam = {r["contrast"]: (r["family"], r["holm_family"]) for r in pooled}
    assert fam["basek5-vs-baser"] == ("primary", "")
    assert fam["basediv-vs-hclm"] == ("secondary", "pooled_secondary/" + "+".join(
        c[0] for c in GAP))
    assert fam["basetls-vs-base-cpu"] == ("secondary", "pooled_secondary")
    assert fam["basek5-vs-base"] == ("secondary", "pooled_secondary/basek5-vs-base")
    k5 = next(r for r in pooled if r["contrast"] == "basek5-vs-base")
    assert float(k5["p_holm"]) == float(k5["p"])  # a family of one
    assert sum(r["holm_family"] == "pooled_secondary" for r in pooled) == 12
    v = json.loads((out / "verdict.json").read_text())
    assert v["pooled:baser-vs-base"] == "tie" and v["pooled:basek5-vs-baser"] == "A better"
    assert v["pooled:basetls-vs-base"] == "tie"  # one failed run in 100 pairs
    assert v["pooled:base-vs-hclm"] == "A worse"
    assert all(v[f"{d}:baselme-vs-base"] is None for d in DATASETS)
    assert sum(r["verdict"] == "" for r in rows) == 60
    agg = list(csv.DictReader((out / "aggregate.csv").open()))
    assert len(agg) == 80
    for col in ("median_deme_count", "median_tuning_share", "median_tuning_evals",
                "median_size", "median_lm_calls"):
        assert col in agg[0], col


# ---- 1bg (mux11)


@pytest.mark.parametrize("prefix", BOOL_PREFIXES)
def test_the_mux11_manifest(prefix):
    q = CONFIGS / prefix
    assert sorted(f.name for f in q.iterdir()) == ["bool.json", "spec.json"]  # no extension
    m = load(q / "bool.json")
    load_manifest(q / "bool.json")  # the runner's own validation
    assert m["name"] == f"{prefix}-bool"
    assert {a: s["argv"] for a, s in m["arms"].items()} == BOOL_ARMS
    assert list(m["arms"]) == ["NP", "LTB", "RT", "UNI"]
    assert m["tasks"] == {"mux11": {"argv": ["-H", "mux", "-k", "3"], "target_raw": 0}}
    assert m["common_argv"] == ["-l", "WARN"] and m["output_root"] == "../gomea-results"
    assert m["budget"]["kind"] == "wall_time"
    if prefix == "phase1bg":
        assert m["seeds"] == list(range(3101, 3111))
        assert m["budget"]["argv"] == ["--max-time", "3600", "-m", "1000000000"]
        assert m["wall_cap_s"] == 5400
        assert len(expand_runs(m)) == 40
    else:
        assert m["seeds"] == [3101, 3102]
        assert m["budget"]["argv"] == ["--max-time", "120", "-m", "1000000000"]
        assert m["wall_cap_s"] <= 600
        assert len(expand_runs(m)) == 8


def test_the_mux11_arms_are_the_boolean_setting_of_round_6():
    """The DYN005 arms of 1ay (dynamic population, F 0.05, no cap) with each FOS."""
    ay = load(CONFIGS / "phase1ay" / "bool.json")["arms"]
    assert BOOL_ARMS["LTB"] == ay["DYN005-LTB"]["argv"]
    assert BOOL_ARMS["RT"] == ay["DYN005-RT"]["argv"]


def test_the_mux11_seeds_are_new():
    """Seeds 3101-3110 are in no earlier manifest."""
    for f in CONFIGS.glob("*/*.json"):
        if f.parent.name in BOOL_PREFIXES:
            continue
        m = load(f)
        if isinstance(m, dict) and isinstance(m.get("seeds"), list):
            assert not set(m["seeds"]) & set(range(3101, 3111)), f


def test_every_mux11_command_line_parses_with_the_real_cli():
    for run in expand_runs(load(CONFIGS / "phase1bg" / "bool.json"), base="/o"):
        pp, ns = cli(run["argv"] + ["-f", "/o/moses.log"])
        gp = pp.gomea_params
        assert ns.problem == "mux" and ns.max_time == 3600 and ns.rand_seed == run["seed"]
        assert gp.fos == FOS[run["arm"]] and gp.dynamic() and gp.max_evals_per_deme is None
        assert gp.hc_fraction_of_nn == 0.05
        assert run["argv"][-2] == "--gomea-log"


@pytest.mark.parametrize("prefix", BOOL_PREFIXES)
def test_the_mux11_commit_names_this_code(prefix):
    """The runner's code check (F8): the manifest names the newest code commit, as the
    Boolean manifests of rounds 4 and 6 do (no set_commit_in)."""
    c = load(CONFIGS / prefix / "bool.json")["commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", c)
    assert c == load(CONFIGS / "phase1ay" / "bool.json")["commit"]
    git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO,
                         capture_output=True, text=True)
    if git.returncode != 0:
        pytest.skip("no git checkout")
    res = check_code_commit(REPO, c)
    assert res["is_ancestor"] and res["code_unchanged"], res


@pytest.mark.parametrize("prefix", BOOL_PREFIXES)
def test_the_mux11_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bg" if prefix == "phase1bg" else "smoke12-1bg")
    assert s["metrics"] == {"mux11": "cpu_to_target"}
    assert s["primary"] == [["mux11", "np-vs-rt", "NP", "RT"], ["mux11", "np-vs-uni", "NP", "UNI"],
                            ["mux11", "np-vs-ltb", "NP", "LTB"]]
    assert s["secondary"] == [] and s["extend"] is False
    assert not {"pooled_primary", "pooled_secondary", "descriptive"} & set(s)


def test_the_report_of_the_mux11_spec(tmp_path):
    """3 primary rows in one Holm family; an unsolved run is infinite; no "extend"."""
    root = tmp_path / "r"
    cpu = {"NP": 10.0, "LTB": 12.0, "RT": 30.0, "UNI": 11.0}
    for arm, c in cpu.items():
        for k in range(3101, 3111):
            solved = not (arm == "RT" and k == 3101)
            s = {"task": "mux11", "arm": arm, "seed": k, "status": "ok", "success": solved,
                 "cpu_to_target": c + (k - 3100) / 10 if solved else None,
                 "evals_to_target": 1000, "cpu_s": c, "n_calls": 1000}
            rd = root / "runs" / f"mux11__{arm}__s{k}"
            rd.mkdir(parents=True)
            (rd / "summary.json").write_text(json.dumps(s))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1bg" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    assert [(r["contrast"], r["family"]) for r in rows] == [
        ("np-vs-rt", "primary"), ("np-vs-uni", "primary"), ("np-vs-ltb", "primary")]
    assert [float(r["p_holm"]) for r in rows] == holm([float(r["p"]) for r in rows])
    v = json.loads((out / "verdict.json").read_text())
    assert v["mux11:np-vs-rt"] == "A better" and v["mux11:np-vs-uni"] == "tie"
    assert "extend" not in v.values()


# ---- the plans


@pytest.mark.parametrize("plan, name, state_dir, prefixes, sids", [
    ("plan12.json", "server-round12-2026-11", "server-run12", ["phase1bf", "phase1bg"],
     ["phase1bf", "phase1bg"]),
    ("plan12-smoke.json", "server-round12-smoke", "server-run12-smoke",
     ["server-smoke12-1bf", "server-smoke12-1bg"], ["smoke12-1bf", "smoke12-1bg"]),
])
def test_plan_is_s0_then_1bf_then_1bg(plan, name, state_dir, prefixes, sids):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0", *sids]
    assert "per_run_mb" not in p["steps"][0] and "manifest_per_run_mb" not in p["steps"][0]
    a, b = p["steps"][1], p["steps"][2]
    assert [a["phase"], b["phase"]] == prefixes
    for st in (a, b):
        assert "config_dir" not in st and not st.get("pilot") and not st.get("set_commit_in")
    # 1bf: no seeded arm, so no GP-GOMEA build; the datasets with the fresh splits 41-50
    assert a["requires"] == REQUIRES
    assert a["per_run_mb"] == OTHER_MB and a["manifest_per_run_mb"] == MB
    for f in stage1(CONFIGS / prefixes[0]):
        assert a["manifest_per_run_mb"].get(f.name, a["per_run_mb"]) == load(f)[
            "mem_limit_mb"], f
    assert all((CONFIGS / prefixes[0] / n).exists() for n in MB)
    # 1bg: mux11 needs no dataset; 3,072 MB, the limit phase_run.sh gives a Boolean run
    assert "requires" not in b and b["per_run_mb"] == MUX_MB and "manifest_per_run_mb" not in b
    assert f"--mem-limit-mb {MUX_MB}" in (REPO / "tools" / "gomea" / "phase_run.sh").read_text()
    assert "then" not in p


def test_the_requirement_needs_the_fresh_splits(tmp_path, monkeypatch):
    """A dataset set built before round 12 (splits 1-40) does not meet the requirement."""
    sys.path.insert(0, str(SERVER))
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan12.json")
    for a in DATASETS:
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
        for k in range(1, 41):
            (d / "splits" / f"s{k}").mkdir()
            for t in ("train", "val", "test"):
                (d / "splits" / f"s{k}" / f"{t}.csv").write_text("x1,y\n")
    why = r.requirement(REQUIRES[0])
    assert why and "41-50" in why
    for a in DATASETS:
        for k in SPLITS:
            s = tmp_path / "phase1f" / "datasets" / a / "splits" / f"s{k}"
            s.mkdir()
            for t in ("train", "val", "test"):
                (s / f"{t}.csv").write_text("x1,y\n")
    assert r.requirement(REQUIRES[0]) is None


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan12.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    s0 = load(SERVER / "plan12-smoke.json")["steps"][0]
    args = s0["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_sr_datasets.py",
              "tests/gomea/test_sr_runner_n_train.py",
              "tests/gomea/test_const_opt_reserve.py",
              "tests/gomea/test_const_opt_topk.py", "tests/gomea/test_const_opt_during.py",
              "tests/gomea/test_const_opt_topk_during_identity.py",
              "tests/gomea/test_fos_np.py", "tests/gomea/test_fos_np_identity.py",
              "tests/gomea/test_linear_scaling_terms_identity.py",
              "tests/gomea/test_sr_eval_terms.py",
              "tests/moses/scoring/test_linear_scaling_terms.py",
              "tests/moses/scoring/test_least_squares.py"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


def test_no_plan_chains_to_or_from_round_12():
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round12", "smoke12"), f
    for f in ("plan12.json", "plan12-smoke.json"):
        assert "then" not in load(SERVER / f)


@pytest.mark.parametrize("name, plan", [("round12", "plan12.json"),
                                        ("smoke12", "plan12-smoke.json")])
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
    assert "round12|smoke12" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan_with_the_fresh_splits():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text
    assert f'-eq {8 * len(sr_datasets.SEEDS)} ]' in text and max(sr_datasets.SEEDS) == 50
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bf", CONFIGS / "server-smoke12-1bf"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace("phase1bf", "server-smoke12-1bf")
        assert m["datasets"] == fm["datasets"] and m["splits"] == [41, 42]
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert m["datasets_dir"] == fm["datasets_dir"]
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert s["argv"] == fm["arms"][a]["argv"]
            # -m 1000: the reserve of BASE-R and BASE-K5 (500) applies
            assert s["budget"] == {"evals": 1000}
    bf, bs = load(CONFIGS / "phase1bg" / "bool.json"), load(q.with_name("server-smoke12-1bg")
                                                             / "bool.json")
    assert bs["arms"] == bf["arms"] and bs["tasks"] == bf["tasks"]


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
    assert (d(MUX_MB)["workers"], d(MUX_MB)["limited_by"]) == (11, "cpu")


FAKE_SR = r"""import json, os, sys
from pathlib import Path
w = sys.argv[sys.argv.index("--workers") + 1]
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"manifest": Path(sys.argv[1]).name, "workers": int(w)}) + "\n")
"""
FAKE_BOOL = r"""import json, sys
from pathlib import Path
a = sys.argv[1:]
if "--dry-run" in a:
    print("result directory: /nowhere/bool")
    sys.exit(0)
with open(Path.cwd().parent / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"manifest": Path(a[0]).name, "phase": Path(a[0]).parent.name,
                         "workers": int(a[a.index("--workers") + 1]),
                         "mem_limit_mb": int(a[a.index("--mem-limit-mb") + 1])}) + "\n")
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


def test_the_phases_run_with_their_own_workers(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bf and 1bg configs; the runners
    and the report are fakes. 1bf: sr.json, then sr-Dow.json with its own W; then 1bg with the
    W of 3,072 MB and phase_run.sh's limit of 3,072 MB for each Boolean run."""
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
    b = repo / "benchmarks" / "gomea"
    b.mkdir(parents=True)
    (repo / "benchmarks" / "__init__.py").write_text("")
    (b / "__init__.py").write_text("")
    (b / "runner.py").write_text(FAKE_BOOL)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
    for q in ("phase1bf", "phase1bg"):
        shutil.copytree(CONFIGS / q, repo / "configs" / "gomea" / q)
    steps = [dict(s) for s in load(SERVER / "plan12.json")["steps"][1:]]
    steps[0].pop("requires")
    plan = {"name": "t12", "state_dir": "server-run12-t", "steps": steps}
    (repo / "tools" / "server" / "plan-t12.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS",
                        "PYTHONPATH")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="1024", MANIFEST_WORKERS="sr.json=99")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t12.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    assert [t["manifest"] for t in trace] == MANIFESTS + ["bool.json"]
    for t in trace[:2]:
        assert t["workers"] == _w(MB.get(t["manifest"], OTHER_MB)), t
    assert trace[2] == {"manifest": "bool.json", "phase": "phase1bg", "workers": _w(MUX_MB),
                        "mem_limit_mb": MUX_MB}
    sd = res / "server-run12-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [e for e in events if e["event"] == "workers_decision"]
    assert [(e.get("manifest"), e["per_run_mb"]) for e in wd] == [
        (None, OTHER_MB), ("sr-Dow.json", 8192), (None, MUX_MB)]
    att = [e for e in events if e["event"] == "phase_attempt"]
    assert att[0]["manifest_workers"] == {n: _w(mb) for n, mb in MB.items()}
    st = load(sd / "state.json")["steps"]
    assert st["phase1bf"]["status"] == "done" and st["phase1bg"]["status"] == "done"
