"""Server round 15 (NOT A PORT: the IT linkage study, BACKLOG 13.9): the plans (plan15.json,
plan15-smoke.json) and the configuration of 1bj, as SERVER-ROUND15-PREREGISTRATION.md and
IT-LINKAGE-STUDY-PLAN.md register them.

1bj: 10 arms = 2 genomes (template, IT) x 5 linkage models (UNI, RT, LT, LT-B, NP), each run a
stand-alone GOMEA population (the runner arm key "standalone"; no MOSES run), N = 1,000 and
E = 500,000 evaluations, on the 10 datasets with the fresh splits 71-80: 1,000 jobs. A run also
ends when 10 generations in a row made no new evaluation ("stalled") or after 300 generations
(the stop rule that the pilot showed to be necessary). One phase,
no Boolean phase, no extension, no choice step, no adoption rule.

The chain (the pilot showed more than 20 h for one plan): round15 (plan15.json) runs S0 and 1bj on
the first half of the datasets, Air, Con, Dow, EnH and Tow (the D2 fallback set;
configs/gomea/phase1bj-a, 500 jobs), then round15b (plan15b.json) runs 1bj with the full manifest:
the runner keeps the 500 runs of round15 (the same manifest name, so the same result directory)
and runs the other 500, and the report covers all 1,000 runs.

The report: P1 (4 interaction contrasts) and P2 (5 contrasts on IT) are primary Holm families
of their own; S1 (5 contrasts on the template) and S2 (2 contrasts on test NMSE) are secondary
families of their own; S3 (P1 and P2 on test NMSE and on the anytime area) and the per-dataset
P1 and P2 are descriptive. The evaluations to reach the median final training NMSE of RT on the
same genome and dataset (evals_to_reference.csv)."""

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

FOS = {"UNI": "uni", "RT": "rt", "LT": "lt-mi", "LTB": "lt-b", "NP": "np"}
ARMS = {f"{g}-{m}": (genome, fos) for g, genome in (("T", "template"), ("IT", "it"))
        for m, fos in FOS.items()}
ARM_NAMES = ["T-UNI", "T-RT", "T-LT", "T-LTB", "T-NP", "IT-UNI", "IT-RT", "IT-LT", "IT-LTB",
             "IT-NP"]
N, E = 1000, 500000  # D1: the start values; the pilot kept them
STOP = {"max_generations": 300, "stall_generations": 10}  # the stop rule after the pilot
SMOKE_E = 3000
SIZES = {"template": {"template_height": 4},  # D3: 31 positions
         "it": {"it_terms": 8, "it_factors": 2, "it_powers": [-2, -1, 1, 2]}}  # 32 positions
DATASETS = ["Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac"]
SPLITS = list(range(71, 81))
MEM_MB = 2048  # "Memory": the pilot peak is far below 50 % of it (gate 4)
WALL_CAP_S = 36000  # a guard against a hung process only
PREFIXES = ["phase1bj", "server-smoke15-1bj"]
HALF_A = ["Air", "Con", "Dow", "EnH", "Tow"]  # the first plan of the chain: the D2 fallback set
HALF_PREFIXES = {"phase1bj": "phase1bj-a", "server-smoke15-1bj": "server-smoke15-1bj-a"}
REQUIRES = [f"datasets:{','.join(DATASETS)}:71-80"]
REQUIRES_A = [f"datasets:{','.join(HALF_A)}:71-80"]


def load(path):
    return json.loads(path.read_text())


def stage1(q):
    """The stage-1 manifests of phase_run.sh: sr.json, then sr-*.json without sr-ext-*."""
    return [q / "sr.json"] + sorted(f for f in q.glob("sr-*.json")
                                    if not f.name.startswith("sr-ext-"))


def block(arm, e=E):
    genome, fos = ARMS[arm]
    return {"genome": genome, "fos": fos, "pop": N, "max_evals": e, **STOP, **SIZES[genome]}


# ---- the arms of 1bj


@pytest.mark.parametrize("prefix", PREFIXES)
def test_arms_match_the_preregistration(prefix):
    q = CONFIGS / prefix
    assert [f.name for f in stage1(q)] == ["sr.json"]
    m = load(q / "sr.json")
    assert list(m["arms"]) == ARM_NAMES
    e = E if prefix == "phase1bj" else SMOKE_E
    for a, s in m["arms"].items():
        assert s == {"standalone": block(a, e)}, a  # no binary, argv, budget or seed block


def test_the_arm_names_are_genome_and_linkage_model():
    """T- is the template, IT- the IT genome; LT is lt-mi and LTB is lt-b (plan 3.1)."""
    for a, (genome, fos) in ARMS.items():
        assert a.startswith("T-" if genome == "template" else "IT-")
    assert set(FOS.values()) == set(runner.STANDALONE_FOS)


def test_expand_gives_stand_alone_jobs_with_no_moses_command_line(tmp_path):
    jobs = runner.expand(load(CONFIGS / "phase1bj" / "sr.json"), Path("/o"), REPO)
    for j in jobs:
        assert j["standalone"] == block(j["arm"])
        assert not {"binary", "argv", "budget", "seed_spec", "linear"} & set(j)
        assert j["wall_cap_s"] == WALL_CAP_S and j["mem_limit_mb"] == MEM_MB
        assert j["split_dir"].endswith(f"phase1f/datasets/{j['task']}/splits/s{j['seed']}")
        with pytest.raises(KeyError):
            runner.moses_argv(j, tmp_path, 100)


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_every_command_line_parses_with_the_real_module_cli(arm, tmp_path):
    """The runner's command line of each arm passes the real CLI of
    pymoses.gomea.standalone, and gives the settings of the pre-registration."""
    from pymoses.gomea.standalone.run import parse_args

    job = {"standalone": block(arm), "split_dir": str(tmp_path / "s71"), "seed": 71,
           "arm": arm, "task": "Con", "run_id": f"Con__{arm}__s71"}
    argv = runner.standalone_argv(job, tmp_path / "run")
    assert argv[1:3] == ["-m", "pymoses.gomea.standalone"]
    a = parse_args(argv[3:])
    genome, fos = ARMS[arm]
    assert (a.genome, a.fos, a.pop, a.max_evals, a.seed) == (genome, fos, N, E, 71)
    assert (a.template_height, a.it_terms, a.it_factors, a.it_powers) == (4, 8, 2, (-2, -1, 1, 2))
    assert (a.max_generations, a.stall_generations) == (300, 10)  # the stop rule
    assert not a.no_column_cache  # the IT column cache is on (plan 5.5, section 9)
    assert a.train.endswith("s71/train.csv") and a.val.endswith("s71/val.csv")
    assert a.test.endswith("s71/test.csv") and a.target == "y"
    assert a.arm == arm and a.task == "Con"


# ---- datasets, splits, memory


def test_datasets_and_fresh_splits_and_no_extension():
    q = CONFIGS / "phase1bj"
    assert sorted(f.name for f in q.iterdir()) == ["spec.json", "sr.json"]
    m = load(q / "sr.json")
    assert m["name"] == "phase1bj-sr"
    assert m["datasets"] == DATASETS and m["splits"] == SPLITS
    assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
    assert m["wall_cap_s"] == WALL_CAP_S
    assert "clip_test_nmse" not in m and "plain_nmse" not in m  # the harness writes its own
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_splits_71_to_80_are_new():
    """No earlier regression manifest uses splits 71-80."""
    for f in CONFIGS.glob("*/*.json"):
        if f.parent.name in PREFIXES or f.parent.name in HALF_PREFIXES.values():
            continue
        m = load(f)
        if isinstance(m, dict) and isinstance(m.get("splits"), list):
            assert not set(m["splits"]) & set(SPLITS), f


def test_run_count_is_1000():
    jobs = runner.expand(load(CONFIGS / "phase1bj" / "sr.json"), Path("/o"), REPO)
    assert len(jobs) == 1000 and len({j["run_id"] for j in jobs}) == 1000
    assert all("standalone" in j for j in jobs)
    for key in ("arm", "seed", "task"):
        counts = {}
        for j in jobs:
            counts[j[key]] = counts.get(j[key], 0) + 1
        assert set(counts.values()) == {100}, key


@pytest.mark.parametrize("prefix", PREFIXES)
def test_memory_limit(prefix):
    assert load(CONFIGS / prefix / "sr.json")["mem_limit_mb"] == MEM_MB


# ---- the spec of 1bj

P1 = [[f"{m.lower()}-gain-it-vs-t", f"IT-{m}", "IT-RT", f"T-{m}", "T-RT", "train_nmse"]
      for m in ("UNI", "LT", "LTB", "NP")]
P2 = ([[f"it-{m.lower()}-vs-rt", f"IT-{m}", "IT-RT", "train_nmse"]
       for m in ("UNI", "LT", "LTB", "NP")] + [["it-np-vs-ltb", "IT-NP", "IT-LTB", "train_nmse"]])
S1 = ([[f"t-{m.lower()}-vs-rt", f"T-{m}", "T-RT", "train_nmse"]
       for m in ("UNI", "LT", "LTB", "NP")] + [["t-np-vs-ltb", "T-NP", "T-LTB", "train_nmse"]])
S2 = [["ltb-it-vs-t", "IT-LTB", "T-LTB", "test_nmse"], ["np-it-vs-t", "IT-NP", "T-NP", "test_nmse"]]
S3 = ([[c[0] + "-test"] + c[1:-1] + ["test_nmse"] for c in P1 + P2]
      + [[c[0] + "-anytime"] + c[1:-1] + ["anytime_area"] for c in P1 + P2])


@pytest.mark.parametrize("prefix", PREFIXES)
def test_spec(prefix):
    s = load(CONFIGS / prefix / "spec.json")
    assert s["phase"] == ("1bj" if prefix == "phase1bj" else "smoke15-1bj")
    assert s["metrics"] == dict.fromkeys(DATASETS, "train_nmse")  # D6: the primary metric
    assert s["primary"] == [] and s["extend"] is False
    for k in ("secondary", "descriptive", "pooled_primary", "pooled_secondary",
              "pooled_secondary_families", "pooled_descriptive", "secondary_families",
              "same_search"):
        assert k not in s, k
    fams = {f["name"]: f for f in s["contrast_families"]}
    assert list(fams) == ["P1", "P2", "S1", "S2", "S3", "D"]
    assert [fams[n]["role"] for n in fams] == ["primary", "primary", "secondary", "secondary",
                                               "descriptive", "descriptive"]
    assert fams["P1"]["contrasts"] == P1 and fams["P2"]["contrasts"] == P2
    assert fams["S1"]["contrasts"] == S1 and fams["S2"]["contrasts"] == S2
    assert fams["S3"]["contrasts"] == S3 and len(S3) == 18
    assert fams["D"]["contrasts"] == P1 + P2 and fams["D"]["per_dataset"] is True
    assert all("per_dataset" not in fams[n] for n in ("P1", "P2", "S1", "S2", "S3"))
    assert s["evals_to_reference"] == {a: ("T-RT" if a.startswith("T-") else "IT-RT")
                                       for a in ARM_NAMES}
    assert s["size_terms"] is True and s["run_table"] is True
    every = {a for f in fams.values() for c in f["contrasts"] for a in c[1:-1]}
    assert every == set(ARM_NAMES)
    assert phase_report.linkage_report.check_spec(s, phase_report.METRICS) is None


def test_p1_is_the_interaction_of_the_plan():
    """P1: for f in UNI, LT, LT-B, NP, d = log(IT-f / IT-RT) - log(T-f / T-RT) on training NMSE
    (plan 7.2; d < 0: f gains more over RT on IT than on the template)."""
    for c, m in zip(P1, ("UNI", "LT", "LTB", "NP"), strict=True):
        assert c[1:] == [f"IT-{m}", "IT-RT", f"T-{m}", "T-RT", "train_nmse"]


def test_holm_families_at_100_pairs():
    """With 100 pooled pairs each family with verdicts (4, 5, 5, 2) can give one; the 90
    per-dataset rows (10 pairs each) cannot (90 * 2/1024 > 0.025): they are descriptive."""
    p100 = wilcoxon_signed_rank([float(i + 1) for i in range(100)])[1]
    for n in (4, 5, 2):
        assert holm([p100] * n)[0] < phase_report.ALPHA
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert holm([p10] * 90)[0] > phase_report.ALPHA


def _anytime(train):
    return [{"evals": round(f * E), "frac": f, "best_train_nmse": train * (1 + (1 - f)),
             "reached": True} for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)]


def _summary(task, arm, split, train, status="ok"):
    ok = status == "ok"
    s = {"arm": arm, "task": task, "seed": split, "status": status,
         "train_nmse": train if ok else None, "val_nmse": 1.1 * train if ok else None,
         "test_nmse": 1.3 * train if ok else None,
         "test_nmse_clipped": 1.2 * train if ok else None, "cpu_s": 100.0,
         "n_evals": E if ok else None, "generations": 20, "cache_hits": 1000,
         "termination_reason": "global_budget", "anytime": _anytime(train),
         "final_program": "+(*($1 $2) $3)" if arm.startswith("IT-") else "+($1 *($2 $3))"}
    if arm.startswith("IT-"):
        s["terms"] = {"n_terms": 2, "fallback": False, "reason": None, "weights": [0, 1, 1]}
    else:
        s["final_program_nodes"] = 5
    return s


def test_the_report_of_the_real_spec(tmp_path):
    """phase_report.py on the 1bj spec with 1,000 made-up runs: 4 + 5 + 5 + 2 + 18 pooled rows
    and 90 per-dataset rows, the Holm families, verdicts for P1, P2, S1 and S2 only, the
    evaluations-to-RT table (100 + 10 rows) and runs.csv (1,000 rows)."""
    factor = {"T-UNI": 1.0, "T-RT": 1.0, "T-LT": 1.0, "T-LTB": 0.95, "T-NP": 1.0,
              "IT-UNI": 1.0, "IT-RT": 1.0, "IT-LT": 0.8, "IT-LTB": 0.6, "IT-NP": 1.0}
    root = tmp_path / "r"
    for d in DATASETS:
        for k in SPLITS:
            for arm, f in factor.items():
                failed = arm == "IT-NP" and d == "Tow" and k == 71
                train = f * (5.0 + k / 10 + DATASETS.index(d)) * (1 + (k * 13 % 7) / 1000)
                s = _summary(d, arm, k, train, "error" if failed else "ok")
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))
                end = {"type": "end", "n_evals": E, "generations": 20,
                       "trace": [[1, 3 * train], [E // 4, 1.5 * train], [E // 2, train]]}
                (rd / "events.jsonl").write_text(json.dumps(end) + "\n")
                (rd / "candidates.json").write_text(json.dumps([{"program": "$1"}]))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(CONFIGS / "phase1bj" / "spec.json"), str(out),
                            str(root)])
    assert rc == 0
    rows = list(csv.DictReader((out / "contrasts.csv").open()))
    pooled = [r for r in rows if r["problem"] == "pooled"]
    assert len(pooled) == 4 + 5 + 5 + 2 + 18 and len(rows) == len(pooled) + 90
    assert all(r["pairs"] == "100" for r in pooled)
    assert all(r["pairs"] == "10" for r in rows if r["problem"] != "pooled")
    fam = {}
    for r in rows:
        fam.setdefault(r["holm_family"], []).append(r)
    assert {k: len(v) for k, v in fam.items()} == {"P1": 4, "P2": 5, "S1": 5, "S2": 2,
                                                   "S3": 18, "D": 90}
    for name, rs in fam.items():
        assert [float(r["p_holm"]) for r in rs] == pytest.approx(
            holm([float(r["p"]) for r in rs]), rel=1e-12), name
    by = {r["contrast"]: r for r in pooled}
    assert by["ltb-gain-it-vs-t"]["verdict"] == "A1 gains more"
    assert by["ltb-gain-it-vs-t"]["metric"] == "interaction:train_nmse"
    assert by["uni-gain-it-vs-t"]["verdict"] == "tie"
    assert by["it-ltb-vs-rt"]["verdict"] == "A better"
    assert by["it-np-vs-ltb"]["verdict"] == "A worse"
    assert by["t-ltb-vs-rt"]["verdict"] == "tie"  # 5 % is inside the 10 % band
    assert all(by[c[0]]["verdict"] == "descriptive" for c in S3)
    assert all(r["verdict"] == "descriptive" for r in fam["D"])
    v = json.loads((out / "verdict.json").read_text())
    assert set(v) == {f"pooled:{c[0]}" for c in P1 + P2 + S1 + S2}
    ev = list(csv.DictReader((out / "evals_to_reference.csv").open()))
    assert len(ev) == 110
    tow = {r["arm"]: r for r in ev if r["problem"] == "Tow"}
    assert tow["IT-NP"]["failed"] == "1" and tow["IT-NP"]["reference"] == "IT-RT"
    assert tow["T-LT"]["reference"] == "T-RT"
    table = list(csv.DictReader((out / "runs.csv").open()))
    assert len(table) == 1000 and list(table[0]) == list(run_table.RUN_FIELDS)
    sizes = {r["arm"]: r["size"] for r in table if r["dataset"] == "Con"}
    assert sizes["IT-LTB"] == "9" and sizes["T-LTB"] == "5"  # IT: 7 nodes + 2 terms


# ---- the plans


@pytest.mark.parametrize("full, half", list(HALF_PREFIXES.items()))
def test_the_first_half_is_the_fallback_set_with_the_full_manifest_name(full, half):
    """The first plan's manifest is the full one with the datasets of the D2 fallback set and the
    same name, so its runs go to the result directory of the full phase; its spec is the full
    spec on those datasets (an interim report)."""
    m, fm = load(CONFIGS / half / "sr.json"), load(CONFIGS / full / "sr.json")
    assert sorted(f.name for f in (CONFIGS / half).iterdir()) == ["spec.json", "sr.json"]
    assert m["datasets"] == HALF_A and m["name"] == fm["name"]
    assert {k: v for k, v in m.items() if k != "datasets"} == {
        k: v for k, v in fm.items() if k != "datasets"}
    s, fs = load(CONFIGS / half / "spec.json"), load(CONFIGS / full / "spec.json")
    assert s["phase"] == fs["phase"] + "-a"
    assert s["metrics"] == {d: "train_nmse" for d in HALF_A}
    assert {k: v for k, v in s.items() if k not in ("phase", "metrics")} == {
        k: v for k, v in fs.items() if k not in ("phase", "metrics")}


def test_the_chain_runs_each_job_once():
    """round15 expands to 500 jobs, round15b to 1,000; the 500 of round15 have the same run
    directories in round15b, so the runner keeps them: 1,000 distinct jobs in the chain."""
    a = runner.expand(load(CONFIGS / "phase1bj-a" / "sr.json"), Path("/o"), REPO)
    b = runner.expand(load(CONFIGS / "phase1bj" / "sr.json"), Path("/o"), REPO)
    assert len(a) == 500 and len(b) == 1000
    assert {j["run_dir"] for j in a} <= {j["run_dir"] for j in b}
    assert {j["task"] for j in a} == set(HALF_A)
    assert len({j["run_dir"] for j in a} | {j["run_dir"] for j in b}) == 1000
    assert {j["run_id"]: j for j in b if j["task"] in HALF_A} == {j["run_id"]: j for j in a}


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid, then", [
    ("plan15.json", "server-round15-2026-11", "server-run15", "phase1bj", "phase1bj-a",
     "round15b"),
    ("plan15-smoke.json", "server-round15-smoke", "server-run15-smoke", "server-smoke15-1bj",
     "smoke15-1bj-a", "smoke15b"),
])
def test_the_first_plan_is_s0_then_1bj_on_the_first_half(plan, name, state_dir, prefix, sid,
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
    assert a["requires"] == REQUIRES_A  # no GP-GOMEA build; the datasets with splits 71-80
    assert a["per_run_mb"] == MEM_MB == load(CONFIGS / HALF_PREFIXES[prefix] / "sr.json")[
        "mem_limit_mb"]
    assert "manifest_per_run_mb" not in a


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan15b.json", "server-round15b-2026-11", "server-run15b", "phase1bj", "phase1bj"),
    ("plan15b-smoke.json", "server-round15b-smoke", "server-run15b-smoke", "server-smoke15-1bj",
     "smoke15-1bj"),
])
def test_the_second_plan_is_1bj_with_the_full_manifest(plan, name, state_dir, prefix, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir and "then" not in p
    assert [s["type"] for s in p["steps"]] == ["phase"]  # no S0: round15 ran it
    a = p["steps"][0]
    assert a["id"] == sid and a["phase"] == prefix and "config_dir" not in a
    assert a["requires"] == REQUIRES
    assert a["per_run_mb"] == MEM_MB == load(CONFIGS / prefix / "sr.json")["mem_limit_mb"]


def test_the_requirement_needs_the_fresh_splits(tmp_path, monkeypatch):
    """A dataset set built before round 15 (splits 1-70) does not meet the requirement."""
    sys.path.insert(0, str(SERVER))
    import orchestrate

    monkeypatch.setattr(orchestrate, "RES", tmp_path)
    r = orchestrate.Round(SERVER / "plan15b.json")
    for a in DATASETS:
        d = tmp_path / "phase1f" / "datasets" / a
        (d / "splits").mkdir(parents=True)
        (d / "full.csv").write_text("x1,y\n")
        (d / "splits" / "splits.json").write_text("{}")
        for k in range(1, 71):
            (d / "splits" / f"s{k}").mkdir()
            for t in ("train", "val", "test"):
                (d / "splits" / f"s{k}" / f"{t}.csv").write_text("x1,y\n")
    for req in (REQUIRES[0], REQUIRES_A[0]):
        why = r.requirement(req)
        assert why and "71-80" in why
    for a in DATASETS:
        for k in SPLITS:
            s = tmp_path / "phase1f" / "datasets" / a / "splits" / f"s{k}"
            s.mkdir()
            for t in ("train", "val", "test"):
                (s / f"{t}.csv").write_text("x1,y\n")
    assert r.requirement(REQUIRES[0]) is None and r.requirement(REQUIRES_A[0]) is None


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan15.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    s0 = load(SERVER / "plan15-smoke.json")["steps"][0]
    args = s0["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_sr_datasets.py",
              "tests/gomea/test_round15_report.py", "tests/gomea/standalone"):
        assert t in args
    for t in args:
        assert (REPO / t).exists(), t
    assert s0["repro"] == {"only": ["1ae-mux11-GOMEA-s3001"]}


@pytest.mark.parametrize("first, second", [("plan15.json", "round15b"),
                                           ("plan15-smoke.json", "smoke15b")])
def test_the_real_plans_chain(first, second):
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.next_plan(SERVER / first) == chain.resolve(second)
    assert chain.root(chain.resolve(second)) == SERVER / first
    assert chain.lock_path(SERVER / first) == chain.lock_path(chain.resolve(second))
    assert chain.members(chain.resolve(second)) == [SERVER / first, chain.resolve(second)]


def test_no_other_plan_chains_to_round_15():
    for f in SERVER.glob("plan*.json"):
        if f.name in ("plan15.json", "plan15-smoke.json"):
            continue
        assert load(f).get("then") not in ("round15", "smoke15", "round15b", "smoke15b"), f


@pytest.mark.parametrize("name, plan, chain_files", [
    ("round15", "plan15.json", ["plan15.json", "plan15b.json"]),
    ("smoke15", "plan15-smoke.json", ["plan15-smoke.json", "plan15b-smoke.json"]),
    ("round15b", "plan15b.json", ["plan15b.json"]),
    ("smoke15b", "plan15b-smoke.json", ["plan15b-smoke.json"]),
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
    assert "round15|smoke15|round15b|smoke15b" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan_with_the_fresh_splits():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    assert set(DATASETS) - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text
    # round 16 adds splits 81-90 (SEEDS 1-90); round 15 needs that set-up builds 71-80
    assert f'-eq {8 * len(sr_datasets.SEEDS)} ]' in text and max(sr_datasets.SEEDS) == 90
    assert '-type f | wc -l)" -eq 720 ]' in text
    assert set(SPLITS) <= set(sr_datasets.SEEDS)


def test_smoke_has_the_same_arms_and_is_small():
    full, q = CONFIGS / "phase1bj", CONFIGS / "server-smoke15-1bj"
    assert sorted(q.glob("*-ext-*.json")) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    m, fm = load(q / "sr.json"), load(full / "sr.json")
    assert m["name"] == fm["name"].replace("phase1bj", "server-smoke15-1bj")
    assert m["datasets"] == fm["datasets"] and m["splits"] == [71, 72]
    assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
    assert m["datasets_dir"] == fm["datasets_dir"]
    assert list(m["arms"]) == list(fm["arms"])
    for a, s in m["arms"].items():
        want = dict(fm["arms"][a]["standalone"], max_evals=SMOKE_E)
        assert s == {"standalone": want}, a
    assert SMOKE_E >= N  # the runner needs max_evals >= pop
    sp, fp = load(q / "spec.json"), load(full / "spec.json")
    assert {k: v for k, v in sp.items() if k != "phase"} == {
        k: v for k, v in fp.items() if k != "phase"}


# ---- W on the server


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


def test_server_workers_are_11_and_limited_by_the_cpus(tmp_path):
    """The section "Memory": on the 12-CPU 64 GB server, W = 11 (the CPUs), not the memory."""
    d = _server(tmp_path)
    assert (d(MEM_MB)["workers"], d(MEM_MB)["limited_by"]) == (11, "cpu")


FAKE_SR = r"""import json, os, sys
from pathlib import Path
w = sys.argv[sys.argv.index("--workers") + 1]
m = json.loads(Path(sys.argv[1]).read_text())
with open(Path(__file__).resolve().parents[3] / "gomea-results" / "trace.jsonl", "a") as fh:
    fh.write(json.dumps({"manifest": Path(sys.argv[1]).name, "workers": int(w),
                         "datasets": m["datasets"], "name": m["name"],
                         "out": sys.argv[2]}) + "\n")
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


def test_the_phase_runs_with_the_workers_of_its_memory_limit(tmp_path):
    """The real orchestrate.py and phase_run.sh on the real 1bj configs and the phase steps of
    both plans of the chain; the runner and the report are fakes. The first step runs the
    manifest of configs/gomea/phase1bj-a (PHASE_CONFIG_DIR), the second the full one, both with
    the W of per_run_mb and both into the result directory of phase1bj."""
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
    for q in ("phase1bj", "phase1bj-a"):
        shutil.copytree(CONFIGS / q, repo / "configs" / "gomea" / q)
    steps = ([dict(s) for s in load(SERVER / "plan15.json")["steps"][1:]]
             + [dict(s) for s in load(SERVER / "plan15b.json")["steps"]])
    for s in steps:
        s.pop("requires")
    plan = {"name": "t15", "state_dir": "server-run15-t", "steps": steps}
    (repo / "tools" / "server" / "plan-t15.json").write_text(json.dumps(plan))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "MAX_WORKERS",
                        "PHASE_CONFIG_DIR", "PER_RUN_MB", "RESERVE_MB", "MANIFEST_WORKERS",
                        "PYTHONPATH")}
    env.update(ORCH_NO_CHAIN="1", ORCH_POLL_S="1", ORCH_SCORES_S="100000",
               ORCH_BUNDLE_S="100000", PER_RUN_MB="1024")
    subprocess.run([sys.executable, "tools/server/orchestrate.py", "--plan",
                    "tools/server/plan-t15.json"], cwd=repo, env=env, capture_output=True,
                   text=True, timeout=120, check=True)
    trace = [json.loads(x) for x in (res / "trace.jsonl").read_text().splitlines()]
    assert [t["datasets"] for t in trace] == [HALF_A, DATASETS]
    for t in trace:
        assert t["manifest"] == "sr.json" and t["workers"] == _w(MEM_MB)
        assert t["name"] == "phase1bj-sr" and t["out"] == "../gomea-results/phase1bj"
    sd = res / "server-run15-t"
    events = [json.loads(x) for x in (sd / "events.jsonl").read_text().splitlines()]
    wd = [e for e in events if e["event"] == "workers_decision"]
    assert [(e.get("manifest"), e["per_run_mb"]) for e in wd] == [(None, MEM_MB)] * 2
    st = load(sd / "state.json")["steps"]
    assert st["phase1bj-a"]["status"] == "done" and st["phase1bj"]["status"] == "done"
