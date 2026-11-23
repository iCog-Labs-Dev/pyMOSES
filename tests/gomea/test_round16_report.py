"""Server round 16 (``SERVER-ROUND16-PREREGISTRATION.md``, "Contrasts"): the phase 1bk spec and
the report additions of ``phase_report.py`` and ``budget_report.py``.

* the spec has the contrasts of the pre-registration: primary 4 (2 interaction contrasts),
  secondary 8, size and CPU 8, descriptive (the 4 primary contrasts per dataset; primary 1 and
  2 on the clipped test NMSE; BASE-TLS-100K vs BASE-TLS on the training NMSE);
* the report computes every metric of the spec (test_nmse, train_nmse, test_nmse_clipped,
  program_size, cpu_s) from summaries of ``moses_sr_runner.py`` (a real summary of the runner
  with a fake pyMoses is the template of the made-up runs);
* the descriptive tables of the new spec keys ``"anytime_ratios"``, ``"deme_table"`` and
  ``"rv_table"``, by hand; ``anytime_ratios`` also on summaries whose ``anytime`` comes from
  the exact trace (``--gomea-trace 1``, the RV16 review, finding 1);
* byte identity: for every spec.json of ``configs/gomea/`` at 2d407b2 (the commit before these
  changes), the report files and the stdout of the report at 2d407b2 and now are the same, on
  made-up runs that also have the round 16 summary fields;
* the three manifests (the local pilot, gate item 3: the 100,000 arms in ``sr-100K.json``) and
  the chain (round16 on splits 81-85, ``configs/gomea/phase1bk-a``): the report pairs the runs
  that the jobs of ``sr-100K.json`` and of ``sr.json`` write, when it reads the result
  directories that ``phase_run.sh`` gives it.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import budget_report as br  # noqa: E402
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402
from round16_data import DIGEST, TWO_DEMES, TWO_DEMES_TRACE, gomea_log, job  # noqa: E402

OLD = "2d407b2"  # the commit before the round 16 report changes
CFG = REPO / "configs" / "gomea"
SPEC = CFG / "phase1bk" / "spec.json"
ARMS = ["BASE-TLS", "BASE-TLS-100K", "TLS-RV", "TLS-RV-100K", "TLS-RVL", "TLS-RVL-100K"]
DATASETS = ["Air", "Con", "Dow", "EnC", "EnH", "Tow", "Yac"]
SPLITS = list(range(81, 91))
FRACS = br.ANYTIME_FRACTIONS


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open()))


# ---- the spec


def test_the_spec_has_the_contrasts_of_the_preregistration():
    s = _load(SPEC)
    assert s["phase"] == "1bk" and s["metrics"] == {d: "test_nmse" for d in DATASETS}
    assert s["primary"] == [] and s["extend"] is False
    assert s["size_terms"] is True and s["run_table"] is True
    fam = {f["name"]: f for f in s["contrast_families"]}
    assert list(fam) == ["P", "S", "SC", "D", "DP"]
    assert [fam[n]["role"] for n in fam] == ["primary", "secondary", "secondary",
                                             "descriptive", "descriptive"]
    p1 = ["rv100k-vs-basetls100k", "TLS-RV-100K", "BASE-TLS-100K"]
    p2 = ["rvl100k-vs-basetls100k", "TLS-RVL-100K", "BASE-TLS-100K"]
    p3 = ["rv-gain-100k-vs-20k", "TLS-RV-100K", "BASE-TLS-100K", "TLS-RV", "BASE-TLS"]
    p4 = ["rvl-gain-100k-vs-20k", "TLS-RVL-100K", "BASE-TLS-100K", "TLS-RVL", "BASE-TLS"]
    rv20, rvl20 = ["rv-vs-basetls", "TLS-RV", "BASE-TLS"], ["rvl-vs-basetls", "TLS-RVL",
                                                            "BASE-TLS"]

    def c(base, metric, suffix=""):
        return [base[0] + suffix, *base[1:], metric]

    primary = [c(p, "test_nmse") for p in (p1, p2, p3, p4)]
    assert fam["P"]["contrasts"] == primary
    assert sum(len(x) == 6 for x in primary) == 2  # two interaction contrasts
    assert fam["S"]["contrasts"] == [
        c(rv20, "test_nmse"), c(rvl20, "test_nmse"),
        ["basetls100k-vs-basetls", "BASE-TLS-100K", "BASE-TLS", "test_nmse"],
        ["rv100k-vs-rv", "TLS-RV-100K", "TLS-RV", "test_nmse"],
        *[c(p, "train_nmse", "-train") for p in (p1, p2, p3, p4)]]
    assert fam["SC"]["contrasts"] == [c(p, m, f"-{x}") for p in (p1, p2, rv20, rvl20)
                                      for m, x in (("program_size", "size"), ("cpu_s", "cpu"))]
    assert fam["D"]["per_dataset"] is True and fam["D"]["contrasts"] == primary
    assert fam["DP"]["contrasts"] == [
        c(p1, "test_nmse_clipped", "-clipped"), c(p2, "test_nmse_clipped", "-clipped"),
        ["basetls100k-vs-basetls-train", "BASE-TLS-100K", "BASE-TLS", "train_nmse"]]
    assert "per_dataset" not in fam["DP"]
    # the descriptive tables: each arm to BASE-TLS of the same budget
    assert s["anytime_ratios"] == {"TLS-RV": "BASE-TLS", "TLS-RVL": "BASE-TLS",
                                   "TLS-RV-100K": "BASE-TLS-100K",
                                   "TLS-RVL-100K": "BASE-TLS-100K"}
    assert s["deme_table"] is True and s["rv_table"] is True
    assert br.check_spec(s) is None
    smoke = _load(CFG / "server-smoke16-1bk" / "spec.json")
    assert smoke == {**s, "phase": "smoke16-1bk"}


# ---- made-up runs from a real runner summary


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> dict:
    """A summary of `moses_sr_runner._run_one` for a TLS arm with a fake pyMoses, an RV log and
    the manifest keys of phase 1bk."""
    work = tmp_path_factory.mktemp("template")
    s = runner._run_one(job(work, gomea_log(TWO_DEMES), anytime=True, clip_test_nmse=True,
                            plain_nmse=True))
    assert s["status"] == "ok" and {"rv", "demes", "anytime", "terms"} <= set(s)
    assert s["terms"]["n_terms"] == 2 and s["terms"]["fallback"] is False
    return s


@pytest.fixture(scope="module")
def template_trace(tmp_path_factory) -> dict:
    """`template`, but the log has the trace of `--gomea-trace 1` (the RV16 review)."""
    work = tmp_path_factory.mktemp("template_trace")
    log = gomea_log(TWO_DEMES, traces=TWO_DEMES_TRACE, digest=DIGEST)
    s = runner._run_one(job(work, log, anytime=True, clip_test_nmse=True, plain_nmse=True))
    assert s["status"] == "ok" and s["anytime_source"] == "trace"
    assert s["init_genotypes_sha256"] == DIGEST
    return s


def _make_runs(root: Path, template: dict, factor, fail=()) -> None:
    """The 420 runs of phase 1bk: the template with the metrics times `factor(arm, d, k)`."""
    for d in DATASETS:
        for k in SPLITS:
            for arm in ARMS:
                s = copy.deepcopy(template)
                f = factor(arm, d, k) * (1 + k / 100 + DATASETS.index(d) / 50)
                s.update(run_id=f"{d}__{arm}__s{k}", task=d, arm=arm, seed=k)
                for m in ("train_nmse", "val_nmse", "test_nmse", "test_nmse_clipped",
                          "first_train_nmse", "first_test_nmse"):
                    s[m] = s[m] * f
                s["cpu_s"] = 100.0 * f
                for p in s["anytime"]:
                    if p["best_train_nmse"] is not None:
                        p["best_train_nmse"] *= f
                if "RV" not in arm:
                    del s["rv"]
                if (d, arm, k) in fail:
                    s = {**s, "status": "timeout", "test_nmse": None, "train_nmse": None}
                rd = root / "runs" / s["run_id"]
                rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps(s))


def _factor(arm, d, k):
    return {"TLS-RV-100K": 0.7, "BASE-TLS-100K": 0.9, "TLS-RVL-100K": 0.8,
            "TLS-RV": 1.1, "TLS-RVL": 1.0}.get(arm, 1.0) * (1 + (k % 3) / 1000)


def test_the_report_of_the_real_spec_on_runner_summaries(template, tmp_path):
    root = tmp_path / "r"
    _make_runs(root, template, _factor, fail={("Con", "TLS-RV", 81)})
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(SPEC), str(out), str(root)]) == 0
    rows = _rows(out / "contrasts.csv")
    by = {}
    for r in rows:
        by.setdefault(r["holm_family"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {"P": 4, "S": 8, "SC": 8, "D": 28, "DP": 3}
    assert all(r["pairs"] == "70" for k in ("P", "S", "SC", "DP") for r in by[k])
    assert all(r["pairs"] == "10" for r in by["D"])
    assert {r["problem"] for r in by["D"]} == set(DATASETS)
    # every metric of the spec has a finite value for the ok runs (from runner summaries)
    for r in rows:
        assert math.isfinite(float(r["median_a"])) and math.isfinite(float(r["median_b"])), r
    metrics = {r["metric"] for r in rows}
    assert metrics == {"log_ratio:test_nmse", "interaction:test_nmse", "log_ratio:train_nmse",
                       "interaction:train_nmse", "log_ratio:program_size", "log_ratio:cpu_s",
                       "log_ratio:test_nmse_clipped"}
    # program size: the node count of the reported program plus its n_terms (size_terms)
    size = next(r for r in rows if r["contrast"] == "rv100k-vs-basetls100k-size")
    want = phase_report._nodes(template) + template["terms"]["n_terms"]
    assert float(size["median_a"]) == want and float(size["ratio"]) == 1.0
    p1 = next(r for r in by["P"] if r["contrast"] == "rv100k-vs-basetls100k")
    assert float(p1["ratio"]) == pytest.approx(0.7 / 0.9, rel=1e-3)
    assert p1["verdict"] == "A better" and p1["a_lower"] == "70"
    p3 = next(r for r in by["P"] if r["contrast"] == "rv-gain-100k-vs-20k")
    assert float(p3["ratio"]) == pytest.approx((0.7 / 0.9) / 1.1, rel=1e-3)
    assert p3["verdict"] == "A1 gains more"
    assert {r["verdict"] for r in by["D"] + by["DP"]} == {"descriptive"}
    verdict = _load(out / "verdict.json")
    assert len(verdict) == 20 and verdict["pooled:rv100k-vs-basetls100k"] == "A better"
    # the tables of the spec
    assert len(_rows(out / "anytime_ratios.csv")) == 4 * 8 * 7
    assert len(_rows(out / "demes.csv")) == 6 * 8
    assert len(_rows(out / "rv.csv")) == 4 * 8
    assert (out / "runs.csv").exists() and (out / "nmse_above_100.csv").exists()
    agg = _rows(out / "aggregate.csv")
    assert all(r["median_train_test_gap"] != "" for r in agg)


def test_the_anytime_ratios_by_hand(template, tmp_path):
    root = tmp_path / "r"
    _make_runs(root, template, _factor, fail={("Con", "TLS-RV", 81)})
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(SPEC), str(out), str(root)]) == 0
    rows = {(r["problem"], r["arm"], float(r["frac"])): r
            for r in _rows(out / "anytime_ratios.csv")}
    # the template log: values from 5 % of 1,000 evaluations on; none at 1 and 2 %
    for frac in (0.01, 0.02):
        early = rows[("pooled", "TLS-RV-100K", frac)]
        assert early["pairs"] == "0" and early["missing"] == "70"
        assert early["median_ratio"] == ""
    assert rows[("pooled", "TLS-RV-100K", 0.05)]["pairs"] == "70"
    r = rows[("pooled", "TLS-RV-100K", 1.0)]
    assert r["reference"] == "BASE-TLS-100K" and r["pairs"] == "70"
    assert float(r["evals"]) == 1000
    assert float(r["median_ratio"]) == pytest.approx(0.7 / 0.9, rel=1e-3)
    assert r["arm_lower"] == "70" and r["reference_lower"] == "0"
    con = rows[("Con", "TLS-RV", 0.5)]
    assert con["pairs"] == "9" and con["missing"] == "1"  # the failed run
    assert float(con["median_ratio"]) == pytest.approx(1.1, rel=1e-3)


def test_the_anytime_ratios_on_trace_summaries(template_trace, tmp_path):
    """The RV16 review (finding 1): `budget_report.py` does not change; with summaries whose
    `anytime` comes from the trace, every checkpoint has a value, so the 1 and 2 % rows have
    pairs (with the generation records they had none, see the test above)."""
    root = tmp_path / "r"
    _make_runs(root, template_trace, _factor, fail={("Con", "TLS-RV", 81)})
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(SPEC), str(out), str(root)]) == 0
    rows = {(r["problem"], r["arm"], float(r["frac"])): r
            for r in _rows(out / "anytime_ratios.csv")}
    assert len(rows) == 4 * 8 * 7
    # the template: the trace raw scores of TWO_DEMES_TRACE at the checkpoints of 1,000
    raws = [-850.0, -850.0, -800.0, -600.0, -600.0, -580.0, -500.0]
    n, var = template_trace["n_train"], template_trace["anytime_check"]["var_scaled_target"]
    assert [p["best_train_nmse"] for p in template_trace["anytime"]] == pytest.approx(
        [100.0 * (-r / n) / var for r in raws], rel=1e-12)
    for frac, evals in zip(FRACS, (10, 20, 50, 100, 200, 500, 1000), strict=True):
        r = rows[("pooled", "TLS-RV-100K", frac)]
        assert r["pairs"] == "70" and r["missing"] == "0", frac
        assert float(r["evals"]) == evals
        assert float(r["median_ratio"]) == pytest.approx(0.7 / 0.9, rel=1e-3)
        assert r["arm_lower"] == "70" and r["reference_lower"] == "0"
        con = rows[("Con", "TLS-RV", frac)]
        assert con["pairs"] == "9" and con["missing"] == "1"  # the failed run
        assert float(con["median_ratio"]) == pytest.approx(1.1, rel=1e-3)
    r = rows[("pooled", "TLS-RV-100K", 0.01)]
    assert float(r["median_arm"]) / float(r["median_reference"]) == pytest.approx(0.7 / 0.9,
                                                                                   rel=1e-3)


def test_the_deme_and_rv_tables_by_hand(template, tmp_path):
    root = tmp_path / "r"
    _make_runs(root, template, _factor)
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(SPEC), str(out), str(root)]) == 0
    demes = {(r["problem"], r["arm"]): r for r in _rows(out / "demes.csv")}
    d = demes[("pooled", "BASE-TLS")]
    assert d["runs"] == "70" and d["with_record"] == "70" and float(d["median_demes"]) == 2
    assert d["one_deme_runs"] == "0" and float(d["median_second_start_evals"]) == 300
    assert float(d["median_second_start_share"]) == pytest.approx(0.3)
    rv = {(r["problem"], r["arm"]): r for r in _rows(out / "rv.csv")}
    assert {a for _, a in rv} == {"TLS-RV", "TLS-RV-100K", "TLS-RVL", "TLS-RVL-100K"}
    r = rv[("Con", "TLS-RV")]
    assert r["runs"] == "10" and r["with_rv"] == "10" and float(r["median_rv_evals"]) == 100
    assert float(r["median_rv_share"]) == 0.1 and float(r["max_rv_share"]) == 0.1
    assert r["rv_evals"] == "1000" and r["improving"] == "300" and r["kept_p"] == "70"
    assert r["ams1_rounds"] == "20" and r["fi_triggers"] == "10"
    assert float(r["improving_share"]) == 0.3 and float(r["kept_p_share"]) == 0.07
    assert rv[("pooled", "TLS-RV")]["rv_evals"] == "7000"


# ---- the three manifests of 1bk and the chain


MANIFESTS = ["sr.json", "sr-100K.json", "sr-Dow.json"]  # the order of phase_run.sh


def _place_runs(made: Path, results: Path, config: Path) -> list[str]:
    """Copy the made-up summary of each job of the three manifests of `config` to the result
    directory of the job (moses_sr_runner.expand with the output root RESULTS/phase1bk, as
    phase_run.sh runs the runner). Return the directories that phase_run.sh gives the report:
    RESULTS/phase1bk/<manifest name> for each of the three manifests, in its order."""
    out_root = results / "phase1bk"
    dirs = []
    for n in MANIFESTS:
        m = _load(config / n)
        dirs.append(str(out_root / m["name"]))
        for j in runner.expand(m, out_root, REPO):
            rd = Path(j["run_dir"])
            rd.mkdir(parents=True)
            shutil.copyfile(made / "runs" / j["run_id"] / "summary.json", rd / "summary.json")
    return dirs


@pytest.mark.parametrize("config, n_splits", [("phase1bk", 10), ("phase1bk-a", 5)])
def test_the_report_pairs_the_runs_of_the_three_manifests(template_trace, tmp_path, config,
                                                          n_splits):
    """The runs of the 100,000 arms come from the jobs of sr-100K.json (and sr-Dow.json), the
    runs of the 20,000 arms from the jobs of sr.json (and sr-Dow.json). The report of the full
    config (round16b) and of the first half (round16, splits 81-85) pairs them by dataset and
    split: 70 (35) pairs in each pooled contrast, also in the interaction contrasts
    [X-100K, BASE-TLS-100K, X, BASE-TLS], and 10 (5) for each dataset."""
    made = tmp_path / "made"
    _make_runs(made, template_trace, _factor)
    dirs = _place_runs(made, tmp_path / "res", CFG / config)
    # three result directories of their own; the 100,000 runs of the 6 datasets other than Dow
    # are in phase1bk-sr-100K only
    assert [Path(d).name for d in dirs] == ["phase1bk-sr", "phase1bk-sr-100K", "phase1bk-sr-Dow"]
    per_dir = {Path(d).name: {Path(f).parent.name.split("__")[1]
                              for f in Path(d).glob("runs/*/summary.json")} for d in dirs}
    assert per_dir == {"phase1bk-sr": {"BASE-TLS", "TLS-RV", "TLS-RVL"},
                       "phase1bk-sr-100K": {"BASE-TLS-100K", "TLS-RV-100K", "TLS-RVL-100K"},
                       "phase1bk-sr-Dow": set(ARMS)}
    assert len(list((tmp_path / "res").glob("phase1bk/*/runs/*/summary.json"))) == 42 * n_splits
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(CFG / config / "spec.json"), str(out),
                              ",".join(dirs)]) == 0
    by = {}
    for r in _rows(out / "contrasts.csv"):
        by.setdefault(r["holm_family"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {"P": 4, "S": 8, "SC": 8, "D": 28, "DP": 3}
    assert all(r["pairs"] == str(7 * n_splits) for k in ("P", "S", "SC", "DP") for r in by[k])
    assert all(r["pairs"] == str(n_splits) for r in by["D"])
    assert {r["problem"] for r in by["D"]} == set(DATASETS)
    splits = SPLITS[:n_splits]
    for name, x in (("rv-gain-100k-vs-20k", "TLS-RV"), ("rvl-gain-100k-vs-20k", "TLS-RVL")):
        r = next(r for r in by["P"] if r["contrast"] == name)
        assert (r["a"], r["b"], r["a2"], r["b2"]) == (f"{x}-100K", "BASE-TLS-100K", x,
                                                      "BASE-TLS")
        assert r["metric"] == "interaction:test_nmse"
        want = statistics.median(
            math.log(_factor(f"{x}-100K", d, k) / _factor("BASE-TLS-100K", d, k))
            - math.log(_factor(x, d, k) / _factor("BASE-TLS", d, k))
            for d in DATASETS for k in splits)
        assert float(r["median_log_ratio"]) == pytest.approx(want, abs=1e-9)
        assert r["a_lower"] == str(7 * n_splits) and r["verdict"] == "A1 gains more"
    assert len(_rows(out / "anytime_ratios.csv")) == 4 * 8 * 7
    assert len(_rows(out / "runs.csv")) == 42 * n_splits


def test_without_the_directory_of_sr_100k_only_the_dow_pairs_are_left(template_trace,
                                                                      tmp_path):
    """A check of the check above: when the report does not read the result directory of
    sr-100K.json, each primary contrast has only the 10 Dow pairs. So the other 60 pairs come
    from the runs of sr-100K.json."""
    made = tmp_path / "made"
    _make_runs(made, template_trace, _factor)
    dirs = _place_runs(made, tmp_path / "res", CFG / "phase1bk")
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(SPEC), str(out),
                              ",".join(d for d in dirs if not d.endswith("-sr-100K"))]) == 0
    p = [r for r in _rows(out / "contrasts.csv") if r["holm_family"] == "P"]
    assert len(p) == 4 and all(r["pairs"] == "10" for r in p)


@pytest.mark.parametrize("key,value", [
    ("anytime_ratios", []), ("anytime_ratios", {}), ("anytime_ratios", {"A": 1}),
    ("deme_table", "yes"), ("rv_table", 1),
])
def test_a_bad_key_is_refused(key, value, tmp_path, capsys):
    spec = {**_load(SPEC), key: value}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    rd = tmp_path / "r" / "runs" / "Con__BASE-TLS__s81"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(json.dumps(
        {"task": "Con", "arm": "BASE-TLS", "seed": 81, "status": "ok", "test_nmse": 1.0}))
    assert phase_report.main(["--stage", "1", str(p), str(tmp_path / "o"),
                              str(tmp_path / "r")]) == 2
    assert key in capsys.readouterr().err


# ---- byte identity against 2d407b2 (git)


def _old_tools(tmp_path: Path) -> Path:
    d = tmp_path / f"old-{OLD}"
    d.mkdir(exist_ok=True)
    for n in ("phase_report.py", "run_table.py", "linkage_report.py"):
        r = subprocess.run(["git", "show", f"{OLD}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
            pytest.fail(f"git show {OLD}:tools/gomea/{n} failed: {r.stderr.decode().strip()!r}. "
                        f"The byte-identity test needs the reference commit {OLD}; use a clone "
                        f"with the full history.")
        (d / n).write_bytes(r.stdout)
    return d


def _reference_specs() -> list[str]:
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", OLD, "configs/gomea"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"git ls-tree {OLD} failed: {r.stderr.strip()!r}")
    return sorted(p for p in r.stdout.split() if p.endswith("/spec.json"))


def _arms_of(spec: dict) -> list[str]:
    arms = []
    for k in ("primary", "secondary", "descriptive"):
        for c in spec.get(k, []):
            arms += [c[2], c[3]]
    for k in ("pooled_primary", "pooled_secondary"):
        for c in spec.get(k, []):
            arms += [c[1], c[2]]
    for f in spec.get("contrast_families", []):
        for c in f["contrasts"]:
            arms += c[1:-1]
    for a, b in spec.get("same_search", []):
        arms += [a, b]
    return list(dict.fromkeys(arms))


BOOL_METRICS = {"cpu_to_target", "evals_to_target"}


def _fixture_runs(root: Path, spec: dict, template: dict) -> None:
    """Made-up runs for one spec (6 seeds): Boolean summaries for a problem with a Boolean
    metric, else runner summaries with the round 16 fields (`rv`, `demes`, `anytime`), failed
    runs and an "end" record with a trace."""
    problems = sorted(set(spec["metrics"]) | {c[0] for k in ("primary", "secondary",
                                                             "descriptive")
                                              for c in spec.get(k, [])})
    arms = _arms_of(spec) or ["A", "B"]
    for pi, p in enumerate(problems):
        boolean = spec["metrics"].get(p) in BOOL_METRICS
        for ai, arm in enumerate(arms):
            for k in range(1, 7):
                rd = root / "runs" / f"{p}__{arm}__s{k}"
                rd.mkdir(parents=True)
                x = (1.0 + 0.15 * ai) * (1.0 + k / 37.0) * (1.0 + pi / 11.0)
                failed = (ai + k + pi) % 13 == 0
                status = "error" if failed else "ok"
                if boolean:
                    success = not failed and (ai * 3 + k) % 7 != 0
                    s = {"arm": arm, "task": p, "seed": k, "status": status, "success": success,
                         "cpu_to_target": 2.0 * x if success else None,
                         "evals_to_target": int(1000 * x) if success else None,
                         "cpu_s": 3.0 * x, "n_calls": 5000, "deme_count": 2 + ai % 3,
                         "best_cpx": 5 + k % 4, "best_raw": 0 if success else -2}
                else:
                    s = copy.deepcopy(template)
                    s.update(arm=arm, task=p, seed=k, status=status, cpu_s=4.0 * x,
                             test_nmse=None if failed else 10.0 * x,
                             train_nmse=None if failed else 8.0 * x,
                             val_nmse=None if failed else 9.0 * x,
                             test_nmse_clipped=None if failed else 9.5 * x)
                    if pi == 0 and k == 1 and not failed:
                        s["test_nmse"] = 150.0  # a run above 100
                (rd / "summary.json").write_text(json.dumps(s))
                end = {"type": "end", "generations": 3 + k % 2, "termination_reason": "stall",
                       "ledger": {"requests": 10}, "dim": 30 + ai, "pop_size_effective": 64}
                st_end = {"type": "end", "n_evals": 20000, "trace": [[1, 30.0], [900, 8.0 * x]]}
                (rd / "events.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in (end, end, st_end)) + "\n")
                (rd / "candidates.json").write_text(json.dumps(
                    [{"program": "$1"}, {"program": f"+($1 {ai % 2})"}]))


DRIVER = r"""import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
import phase_report
for spec, out, root, log in json.load(open(sys.argv[2])):
    for stage in ("1", "2"):
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            rc = phase_report.main(["--stage", stage, spec, out + "-stage" + stage, root])
        with open(log + "-stage" + stage, "w") as fh:
            json.dump({"rc": rc, "stdout": o.getvalue(), "stderr": e.getvalue()}, fh)
"""


def test_every_old_spec_gives_a_byte_identical_report(template, tmp_path):
    """Every spec.json of configs/gomea/ at 2d407b2 (the rounds before 16, the round 15 keys
    included): the files and the stdout of the report at 2d407b2 and now are the same, at
    stage 1 and stage 2, on runs with the round 16 summary fields."""
    specs = _reference_specs()
    assert len(specs) >= 74 and "configs/gomea/phase1bj/spec.json" in specs
    old = _old_tools(tmp_path)
    jobs = {"old": [], "new": []}
    for rel in specs:
        spec = REPO / rel
        name = spec.parent.name
        sj = _load(spec)
        assert not {"anytime_ratios", "deme_table", "rv_table"} & set(sj)
        root = tmp_path / "runs" / name
        _fixture_runs(root, sj, template)
        for tag in jobs:
            jobs[tag].append([str(spec), str(tmp_path / tag / name), str(root),
                              str(tmp_path / tag / f"{name}.log")])
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for tag, tool in (("old", old), ("new", TOOLS)):
        (tmp_path / tag).mkdir()
        jf = tmp_path / f"{tag}-jobs.json"
        jf.write_text(json.dumps(jobs[tag]))
        subprocess.run([sys.executable, "-c", DRIVER, str(tool), str(jf)], env=env, check=True,
                       cwd=tmp_path)
    compared = 0
    for (_, o_out, _, o_log), (_, n_out, _, n_log) in zip(jobs["old"], jobs["new"], strict=True):
        for stage in ("1", "2"):
            lo = json.loads(Path(o_log + "-stage" + stage).read_text())
            ln = json.loads(Path(n_log + "-stage" + stage).read_text())
            assert ln == lo, o_out
            od, nd = Path(o_out + "-stage" + stage), Path(n_out + "-stage" + stage)
            files = sorted(p.name for p in od.iterdir()) if od.exists() else []
            assert (sorted(p.name for p in nd.iterdir()) if nd.exists() else []) == files
            for f in files:
                assert (nd / f).read_bytes() == (od / f).read_bytes(), (o_out, f)
                compared += 1
            if stage == "1":
                assert lo["rc"] == 0, (o_out, lo["stderr"])
    assert compared >= 3 * 2 * len(specs)
