"""Server round 17 (``SERVER-ROUND17-PREREGISTRATION.md``, "Pairing with round 16", "The Dow
rule", "Contrasts"): the report of phase 1bl, which reads two rounds.

* The spec keys: ``"partners"`` (``partner_report.py``: the report loads the round 16 partners of
  its own runs from the round 16 result directories), ``"pooled_problems"`` and a family's
  ``"problems"`` (``linkage_report.py``: every family with verdicts uses the 6 datasets without
  Dow; Dow is a descriptive family of its own).
* Made-up runs in two sets of directories: the HC runs of 1bl and the round 16 runs (BASE-TLS,
  BASE-TLS-100K, and arms that 1bl does not pair: TLS-RV, the failed BASE-TLS-100K on Dow).
  The families, the pairs, a contrast and an interaction by hand, the Dow rule, the first half
  (splits 81-85), the anytime and deme tables.
* Missing partners: the report lists them (partners.csv, partners.json, a warning) and does not
  stop; with no partner directory at all it still writes the report.
* The real spec from the repository root: its relative partner directories are those of the
  server (``../gomea-results/phase1bk/...``).
* ``drop_arms.py`` gives the design without HC-TLS (``round17_design.spec(..., hc_tls=False)``).
* Byte identity: for every spec.json of ``configs/gomea/`` at 88a5e02 (the commit before these
  changes), the report files and the stdout of the report at 88a5e02 and now are the same.
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
CFG = REPO / "configs" / "gomea"
sys.path.insert(0, str(TOOLS))
import budget_report as br  # noqa: E402
import drop_arms  # noqa: E402
import linkage_report  # noqa: E402
import moses_sr_runner as runner  # noqa: E402
import partner_report  # noqa: E402
import phase_report  # noqa: E402
import round17_design as design  # noqa: E402
from round16_data import DIGEST, TWO_DEMES, TWO_DEMES_TRACE, gomea_log, job  # noqa: E402
from round17_data import EVENTS, HC_LM, HC_TLS, LINES, hc_job  # noqa: E402

OLD = "88a5e02"  # the commit before the round 17 report changes
SPEC = CFG / "phase1bl" / "spec.json"
OWN = {"sr": design.DS6, "sr-100K": design.DS6, "sr-Dow": ["Dow"]}
R16_ARMS = {"sr": ["BASE-TLS", "TLS-RV"], "sr-100K": ["BASE-TLS-100K", "TLS-RV-100K"],
            "sr-Dow": ["BASE-TLS", "BASE-TLS-100K", "TLS-RV"]}
FACTOR = {"BASE-TLS": 1.0, "BASE-TLS-100K": 0.8, "HC-LM": 1.05, "HC-LM-100K": 1.0,
          "HC-TLS": 1.2, "HC-TLS-100K": 1.1, "TLS-RV": 0.5, "TLS-RV-100K": 0.5}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open()))


# ---- templates: a real runner summary of each kind (fake pyMoses)


@pytest.fixture(scope="module")
def templates(tmp_path_factory) -> dict:
    w = tmp_path_factory.mktemp("templates")
    keys = {"clip_test_nmse": True, "plain_nmse": True}
    plain = [(gens, end, None) for gens, end, _ in TWO_DEMES]  # no RV record
    g = runner._run_one(job(w / "g", gomea_log(plain, traces=TWO_DEMES_TRACE, digest=DIGEST),
                            anytime=True, **keys))
    lm = runner._run_one(hc_job(w / "lm", LINES, EVENTS, argv=HC_LM, hc_anytime=True, **keys))
    tls = runner._run_one(hc_job(w / "tls", LINES, argv=HC_TLS, hc_anytime=True, **keys))
    assert g["anytime_source"] == "trace" and "rv" not in g
    assert lm["anytime_source"] == tls["anytime_source"] == "hc-demes"
    assert "const_opt" in lm and "terms" in tls
    return {"gomea": g, "HC-LM": lm, "HC-TLS": tls}


def _value(arm: str, d: str, k: int) -> float:
    """The made-up metric scale of a run: the arm's factor, a dataset effect and a split
    effect. Dow gets extreme values, so a family that took Dow would show it."""
    f = FACTOR[arm] * (1 + k / 100 + design.DS7.index(d) / 50)
    if d == "Dow":
        f *= 1000.0 if arm.startswith("HC") else 1.0
    return f


def _summary(templates: dict, arm: str, d: str, k: int) -> dict:
    t = templates["HC-LM" if arm.startswith("HC-LM") else
                  "HC-TLS" if arm.startswith("HC-TLS") else "gomea"]
    s = copy.deepcopy(t)
    f = _value(arm, d, k)
    s.update(run_id=f"{d}__{arm}__s{k}", task=d, arm=arm, seed=k)
    for m in ("train_nmse", "val_nmse", "test_nmse", "test_nmse_clipped", "first_train_nmse",
              "first_test_nmse"):
        s[m] = s[m] * f
    s["cpu_s"] = 100.0 * f
    for p in s["anytime"]:
        if p["best_train_nmse"] is not None:
            p["best_train_nmse"] *= f
    return s


def _make(root: Path, templates: dict, splits=design.SPLITS, hc_tls=True) -> tuple[list, list]:
    """The own result directories of 1bl and the round 16 directories under `root`."""
    own, r16 = [], []
    for name, dsets in OWN.items():
        od = root / "phase1bl" / f"phase1bl-{name}"
        own.append(str(od))
        big = name == "sr-100K"
        for d in dsets:
            for a in design.arms(hc_tls):
                if a.endswith("-100K") != big:
                    continue
                for k in splits:
                    _put(od, _summary(templates, a, d, k))
    for name, arms in R16_ARMS.items():
        rd = root / "phase1bk" / f"phase1bk-{name}"
        r16.append(str(rd))
        for d in (design.DS6 if name != "sr-Dow" else ["Dow"]):
            for a in arms:
                for k in design.SPLITS:  # round 16 has every split
                    s = _summary(templates, a, d, k)
                    if d == "Dow" and a == "BASE-TLS-100K":  # out of memory in round 16
                        s = {**s, "status": "error", "test_nmse": None, "train_nmse": None}
                    _put(rd, s)
    return own, r16


def _put(d: Path, s: dict) -> Path:
    rd = d / "runs" / s["run_id"]
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "summary.json").write_text(json.dumps(s))
    return rd / "summary.json"


def _spec(tmp_path: Path, r16: list, base: dict | None = None) -> Path:
    s = copy.deepcopy(base or _load(SPEC))
    s["partners"]["dirs"] = r16
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    return p


def _report(spec: Path, out: Path, own: list, capsys=None) -> int:
    return phase_report.main(["--stage", "1", str(spec), str(out), ",".join(own)])


def _by_family(out: Path) -> dict:
    by: dict = {}
    for r in _rows(out / "contrasts.csv"):
        by.setdefault(r["holm_family"], []).append(r)
    return by


# ---- the spec


def test_the_spec_keys_pass_their_checks():
    s = _load(SPEC)
    assert s == design.spec("1bl", design.REAL_PARTNER_DIRS)
    assert linkage_report.check_spec(s, phase_report.METRICS) is None
    assert br.check_spec(s) is None and partner_report.check_spec(s) is None
    assert s["pooled_problems"] == design.DS6 and "Dow" in s["metrics"]
    fam = {f["name"]: f for f in s["contrast_families"]}
    assert fam["DD"]["problems"] == ["Dow"] and fam["DD"]["role"] == "descriptive"
    assert all("problems" not in f for n, f in fam.items() if n != "DD")


# ---- the report on two rounds


def test_the_report_reads_two_rounds(templates, tmp_path, capsys):
    own, r16 = _make(tmp_path / "res", templates)
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    by = _by_family(out)
    n_tls = 2 if design.HC_TLS else 0
    assert {k: len(v) for k, v in by.items()} == {
        "P": 2 + n_tls, "S": 4 + 5 * design.HC_TLS, "SC": 4 + 2 * n_tls, "D": 6 * (2 + n_tls),
        "DD": 1 + design.HC_TLS, **({"DP": 1} if design.HC_TLS else {})}
    # the Dow rule: every pooled family has the 60 pairs of the 6 datasets without Dow
    for k in ("P", "S", "SC", "DP"):
        assert all(r["problem"] == "pooled" and r["pairs"] == "60" for r in by.get(k, [])), k
    assert {r["problem"] for r in by["D"]} == set(design.DS6)
    assert all(r["pairs"] == "10" for r in by["D"] + by["DD"])
    assert {r["problem"] for r in by["DD"]} == {"Dow"}
    assert {r["verdict"] for r in by["D"] + by["DD"]} == {"descriptive"}
    # only the partners of the own runs are loaded (no TLS-RV, no BASE-TLS-100K on Dow)
    agg = {(r["problem"], r["arm"]): r for r in _rows(out / "aggregate.csv")}
    assert {a for _, a in agg} == set(design.arms()) | {"BASE-TLS", "BASE-TLS-100K"}
    assert ("Dow", "BASE-TLS-100K") not in agg and agg[("Dow", "BASE-TLS")]["runs"] == "10"
    assert all(agg[(d, "BASE-TLS-100K")]["runs"] == "10" for d in design.DS6)
    p = _load(out / "partners.json")
    n = 130 if design.HC_TLS else 130  # the same partner runs for one or two HC arms per budget
    assert p == {"expected": n, "found": n, "missing": 0, "not_ok": 0, "missing_runs": []}
    rows = _rows(out / "partners.csv")
    assert len(rows) == (260 if design.HC_TLS else 130)
    assert {r["partner_status"] for r in rows} == {"ok"}
    assert {r["partner_dir"] for r in rows} == set(r16)
    assert '"partners": {"expected": 130, "found": 130, "missing": 0, "not_ok": 0}' in \
        capsys.readouterr().out


def test_a_contrast_and_an_interaction_by_hand(templates, tmp_path):
    own, r16 = _make(tmp_path / "res", templates)
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    rows = {(r["holm_family"], r["contrast"], r["problem"]): r
            for r in _rows(out / "contrasts.csv")}
    t = templates["gomea"]["test_nmse"], templates["HC-LM"]["test_nmse"]
    # P2: BASE-TLS-100K vs HC-LM-100K, r = log(A / B) per pair
    r = [math.log(_value("BASE-TLS-100K", d, k) * t[0]) - math.log(_value("HC-LM-100K", d, k)
                                                                   * t[1])
         for d in design.DS6 for k in design.SPLITS]
    p2 = rows[("P", design.P2[0], "pooled")]
    assert float(p2["median_log_ratio"]) == pytest.approx(statistics.median(r))
    # P1: d = log(BASE-TLS-100K / HC-LM-100K) - log(BASE-TLS / HC-LM): the round 16 runs of
    # both budgets pair with the HC runs of both budgets (the template values cancel)
    dd = [math.log(_value("BASE-TLS-100K", d, k) / _value("HC-LM-100K", d, k))
          - math.log(_value("BASE-TLS", d, k) / _value("HC-LM", d, k))
          for d in design.DS6 for k in design.SPLITS]
    p1 = rows[("P", design.P1[0], "pooled")]
    assert p1["metric"] == "interaction:test_nmse" and p1["pairs"] == "60"
    assert float(p1["median_log_ratio"]) == pytest.approx(statistics.median(dd))
    assert p1["a2"] == "BASE-TLS" and p1["b2"] == "HC-LM"
    # verdict.json has the primary and secondary rows only
    v = _load(out / "verdict.json")
    assert f"pooled:{design.P1[0]}" in v and not any(k.startswith("Dow:") for k in v)


def test_the_first_half_loads_only_the_partners_of_its_splits(templates, tmp_path):
    own, r16 = _make(tmp_path / "res", templates, splits=design.HALF_SPLITS)
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    by = _by_family(out)
    assert all(r["pairs"] == "30" for r in by["P"]) and all(r["pairs"] == "5" for r in by["DD"])
    assert _load(out / "partners.json")["expected"] == 65
    agg = {(r["problem"], r["arm"]): r for r in _rows(out / "aggregate.csv")}
    assert agg[("Con", "BASE-TLS")]["runs"] == "5"


def test_the_anytime_and_deme_tables_pool_the_6_datasets(templates, tmp_path):
    own, r16 = _make(tmp_path / "res", templates)
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    at = _rows(out / "anytime_ratios.csv")
    last = {(r["problem"], r["arm"]): r for r in at if r["frac"] == "1.0"}
    assert last[("pooled", "HC-LM")]["pairs"] == "60" and last[("Dow", "HC-LM")]["pairs"] == "10"
    assert last[("pooled", "HC-LM-100K")]["pairs"] == "60"
    assert ("Dow", "HC-LM-100K") not in last
    # the HC source has no value before the end of its first deme (250 of 1000 evaluations)
    first = {r["arm"]: r for r in at if r["problem"] == "pooled" and r["frac"] == "0.01"}
    assert first["HC-LM"]["pairs"] == "0" and first["HC-LM"]["missing"] == "60"
    dm = {(r["problem"], r["arm"]): r for r in _rows(out / "demes.csv")}
    assert dm[("pooled", "HC-LM")]["runs"] == "60" and dm[("Dow", "HC-LM")]["runs"] == "10"
    assert dm[("pooled", "BASE-TLS")]["runs"] == "60"  # Dow's 10 partners are not pooled
    assert float(dm[("Con", "HC-TLS" if design.HC_TLS else "HC-LM")]["median_demes"]) == 3


# ---- missing partners


def test_missing_and_failed_partners_are_listed_and_the_report_goes_on(templates, tmp_path,
                                                                       capsys):
    own, r16 = _make(tmp_path / "res", templates)
    (Path(r16[0]) / "runs" / "Con__BASE-TLS__s83" / "summary.json").unlink()
    (Path(r16[2]) / "runs" / "Dow__BASE-TLS__s81" / "summary.json").write_text('{"task": "Do')
    f = Path(r16[1]) / "runs" / "Yac__BASE-TLS-100K__s90" / "summary.json"
    f.write_text(json.dumps({**_load(f), "status": "timeout", "test_nmse": None}))
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    cap = capsys.readouterr()
    p = _load(out / "partners.json")
    assert p == {"expected": 130, "found": 128, "missing": 2, "not_ok": 1,
                 "missing_runs": ["Con/BASE-TLS/83", "Dow/BASE-TLS/81"]}
    assert ("WARNING: 2 of 130 partner runs are missing (not paired): Con/BASE-TLS/83, "
            "Dow/BASE-TLS/81") in cap.err
    rows = {(r["task"], r["arm"], r["seed"]): r for r in _rows(out / "partners.csv")}
    assert rows[("Con", "HC-LM", "83")]["partner_status"] == "missing"
    assert rows[("Con", "HC-LM", "83")]["partner_dir"] == ""
    assert rows[("Yac", "HC-LM-100K", "90")]["partner_status"] == "timeout"
    by = {r["contrast"]: r for r in _by_family(out)["P"]}
    assert by[design.P1[0]]["pairs"] == "59" and by[design.P2[0]]["pairs"] == "60"
    # the failed partner counts as infinitely bad: HC-LM-100K is lower on that pair
    assert int(by[design.P2[0]]["b_lower"]) >= 1
    assert all(r["pairs"] == "9" for r in _by_family(out)["DD"])


def test_no_partner_directory_at_all(templates, tmp_path, capsys):
    own, r16 = _make(tmp_path / "res", templates)
    for d in r16:
        shutil.rmtree(d)
    out = tmp_path / "out"
    assert _report(_spec(tmp_path, r16), out, own) == 0
    p = _load(out / "partners.json")
    assert p["expected"] == 130 and p["missing"] == 130 and p["found"] == 0
    for r in _by_family(out)["P"]:
        assert r["pairs"] == "0" and r["verdict"] == "tie"
    assert "WARNING: 130 of 130 partner runs are missing" in capsys.readouterr().err
    assert _rows(out / "anytime_ratios.csv") == []


@pytest.mark.parametrize("change", [
    {"pooled_problems": ["Air", "Bos"]},
    {"pooled_problems": []},
    {"pooled_problems": ["Air", "Air"]},
    {"partners": {"dirs": ["x"]}},
    {"partners": {"dirs": [], "pairs": {"HC-LM": "BASE-TLS"}}},
    {"partners": {"dirs": ["x"], "pairs": {"HC-LM": "HC-LM"}}},
    {"partners": {"dirs": ["x"], "pairs": {"HC-LM": "BASE-TLS", "BASE-TLS": "Y"}}},
])
def test_a_bad_key_is_refused(change, templates, tmp_path, capsys):
    own, r16 = _make(tmp_path / "res", templates, splits=[81])
    s = {**_load(SPEC), **change}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(s))
    assert _report(p, tmp_path / "out", own) == 2
    assert "bad " in capsys.readouterr().err


def test_a_bad_family_problem_list_is_refused(templates, tmp_path, capsys):
    own, _ = _make(tmp_path / "res", templates, splits=[81])
    s = _load(SPEC)
    s["contrast_families"][4]["problems"] = ["Dow", "Bos"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(s))
    assert _report(p, tmp_path / "out", own) == 2
    assert "bad problems of family DD" in capsys.readouterr().err


# ---- the real spec from the repository root (the paths of the server)


def test_the_real_spec_reads_the_round_16_directories_from_the_repository_root(templates,
                                                                               tmp_path):
    """phase_run.sh runs the report from the repository root with its own directories as
    ../gomea-results/phase1bl/<name>; the spec's partner directories are
    ../gomea-results/phase1bk/<name>, so they resolve on the server as they do here."""
    res = tmp_path / "gomea-results"
    _make(res, templates)
    repo = tmp_path / "pyMoses"
    repo.mkdir()
    own = ",".join(f"../gomea-results/phase1bl/phase1bl-{x}" for x in OWN)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, str(TOOLS / "phase_report.py"), "--stage", "1",
                        str(SPEC), "../gomea-results/phase1bl/report-stage1", own],
                       cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    p = _load(res / "phase1bl" / "report-stage1" / "partners.json")
    assert p["found"] == 130 and p["missing"] == 0


# ---- HC-TLS left out


def test_drop_arms_gives_the_design_without_hc_tls(templates, tmp_path):
    dirs = []
    for q in ("phase1bl", "phase1bl-a", "server-smoke17-1bl", "server-smoke17-1bl-a"):
        shutil.copytree(CFG / q, tmp_path / q)
        dirs.append(str(tmp_path / q))
    assert drop_arms.main([*dirs, "--arms", "HC-TLS", "HC-TLS-100K"]) == 0
    for q in dirs:
        phase = "smoke17-1bl" if "smoke" in q else "1bl"
        pd = design.SMOKE_PARTNER_DIRS if "smoke" in q else design.REAL_PARTNER_DIRS
        assert _load(Path(q) / "spec.json") == design.spec(phase, pd, hc_tls=False)
        for n in design.MANIFESTS:
            m, fm = _load(Path(q) / n), _load(CFG / Path(q).name / n)
            assert list(m["arms"]) == design.arms_of(n, hc_tls=False)
            assert m == {**fm, "arms": {a: fm["arms"][a] for a in design.arms_of(n, hc_tls=False)}}
    # the report of the design without HC-TLS: Holm across the 2 primary contrasts
    own, r16 = _make(tmp_path / "res", templates, hc_tls=False)
    out = tmp_path / "out"
    spec = _spec(tmp_path, r16, base=_load(Path(dirs[0]) / "spec.json"))
    assert _report(spec, out, own) == 0
    by = _by_family(out)
    assert [r["contrast"] for r in by["P"]] == [design.P1[0], design.P2[0]] and "DP" not in by
    # a second run changes nothing; a manifest with no arm left is refused
    assert drop_arms.main([dirs[0], "--arms", "HC-TLS", "HC-TLS-100K"]) == 0
    assert _load(Path(dirs[0]) / "spec.json") == design.spec("1bl", design.REAL_PARTNER_DIRS,
                                                             hc_tls=False)
    before = (Path(dirs[0]) / "sr.json").read_text()
    assert drop_arms.main([dirs[0], "--arms", "HC-LM", "HC-LM-100K"]) == 2
    assert (Path(dirs[0]) / "sr.json").read_text() == before


# ---- byte identity against 88a5e02 (git)


def _old_tools(tmp_path: Path) -> Path:
    d = tmp_path / f"old-{OLD}"
    d.mkdir(exist_ok=True)
    for n in ("phase_report.py", "run_table.py", "linkage_report.py", "budget_report.py"):
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
    for a, b in spec.get("anytime_ratios", {}).items():
        arms += [a, b]
    return list(dict.fromkeys(arms))


BOOL_METRICS = {"cpu_to_target", "evals_to_target"}


def _fixture_runs(root: Path, spec: dict, templates: dict) -> None:
    """Made-up runs for one spec (6 seeds): Boolean summaries for a problem with a Boolean
    metric, else runner summaries (GOMEA with the round 16 fields, and HC with the round 17
    fields for an HC arm), failed runs and "end" records with a trace."""
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
                    t = templates["HC-LM" if arm.startswith("HC") else "gomea"]
                    s = copy.deepcopy(t)
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


def test_every_old_spec_gives_a_byte_identical_report(templates, tmp_path):
    """Every spec.json of configs/gomea/ at 88a5e02 (the rounds before 17, the round 15 and 16
    keys included): the files and the stdout of the report at 88a5e02 and now are the same, at
    stage 1 and stage 2, on runs with the round 16 and round 17 summary fields."""
    specs = _reference_specs()
    assert len(specs) >= 78 and "configs/gomea/phase1bk/spec.json" in specs
    old = _old_tools(tmp_path)
    jobs = {"old": [], "new": []}
    for rel in specs:
        spec = REPO / rel
        name = spec.parent.name
        sj = _load(spec)
        assert not {"partners", "pooled_problems"} & set(sj)
        root = tmp_path / "runs" / name
        _fixture_runs(root, sj, templates)
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
