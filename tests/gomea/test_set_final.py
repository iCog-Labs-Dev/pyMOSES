"""tools/gomea/set_final.py (server round 8): the choice of FINAL from fake 1aw reports, on both
branches of rule 2, with stage 2, failed runs and the refusals; and, after the population round,
FINAL from the POP-applied 1aw of round 7 (`--source-dir`), NOCAP with POP and the control arm
FINAL-DYN (`--population`)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import set_final  # noqa: E402
import set_population as sp  # noqa: E402

D6 = ["WiR", "Con", "EnH", "Air", "Yac", "Tow"]
# POP = DYN = F 0.25, so the control is N = 16
CHOICE = {"pop": sp.population(0.25), "dyn": sp.population(0.25),
          "control": sp.population(None), "pop_arm": "DYN025-LTB", "dyn_arm": "DYN025-LTB"}
SPEC = json.loads((CONFIGS / "phase1aw" / "spec.json").read_text())
FIELDS = ["problem", "contrast", "family", "a", "b", "metric", "pairs", "median_a", "median_b",
          "p", "p_holm", "verdict"]


def _report(d: Path, size_verdicts: dict, nmse: dict, pairs=10, other="tie"):
    """size_verdicts: {dataset: verdict of pen125-size}; nmse: {dataset: (PEN125, LMVAL)}."""
    d.mkdir(parents=True)
    rows, verdict = [], {}
    for fam in ("primary", "secondary"):
        for p, n, a, b, m in SPEC[fam]:
            v = other
            ma, mb = 1.0, 1.0
            if fam == "primary" and m == "program_size" and a == "LMVAL-PEN125":
                v = size_verdicts[p]
            if fam == "secondary" and n == "pen125-nmse":
                ma, mb = nmse[p]
                v = "tie"
            rows.append(dict(problem=p, contrast=n, family=fam, a=a, b=b, metric=m, pairs=pairs,
                             median_a=ma, median_b=mb, p=0.5, p_holm=0.01, verdict=v))
            verdict[f"{p}:{n}"] = v
    with (d / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (d / "verdict.json").write_text(json.dumps(verdict))


def _setup(tmp_path, source="phase1aw", choice=CHOICE):
    """Round 6's population-choice.json and round 7's resolved 1aw (set_population.py apply)."""
    cf = tmp_path / "state6" / "population-choice.json"
    sd = tmp_path / "state7" / f"{source}-config"
    if not cf.exists():
        cf.parent.mkdir(parents=True)
        cf.write_text(json.dumps(choice))
    if not (sd / "sr.json").exists():
        assert sp.main(["apply", "--choice", str(cf), "--target", source, "--out", str(sd)]) == 0
    return cf, sd


def _aw(tmp_path, source="phase1aw"):
    return json.loads((_setup(tmp_path, source)[1] / "sr.json").read_text())


def _run(tmp_path, *extra, source="phase1aw", target="phase1ax", population=True):
    cf, sd = _setup(tmp_path, source)
    out = tmp_path / "state" / f"{target}-config"
    args = ["--results", str(tmp_path / "res"), "--out", str(out), "--source", source,
            "--target", target, "--source-dir", str(sd)]
    args += ["--population", str(cf)] if population else []
    rc = set_final.main([*args, *extra])
    return rc, out


def _choice(out):
    return json.loads((out / "final-choice.json").read_text())


GOOD = {d: (1.05, 1.0) for d in D6}


def test_rule_2_passes_gives_lmval_pen125(tmp_path):
    sv = dict.fromkeys(D6, "tie") | dict.fromkeys(["WiR", "Con", "EnH", "Air"], "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv, GOOD)
    rc, out = _run(tmp_path)
    assert rc == 0
    m = json.loads((out / "sr.json").read_text())
    assert m["arms"]["FINAL"]["argv"] == _aw(tmp_path)["arms"]["LMVAL-PEN125"]["argv"]
    tpl = json.loads((CONFIGS / "phase1ax" / "template.json").read_text())
    assert {k: v for k, v in m.items() if k != "arms"} == {
        k: v for k, v in tpl.items() if k != "arms"}
    assert m["arms"]["HC-LS"] == tpl["arms"]["HC-LS"]
    pop = ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.25"]
    nocap = tpl["arms"]["NOCAP"]["argv"]
    assert m["arms"]["NOCAP"]["argv"] == nocap[:4] + pop + nocap[5:]
    final = m["arms"]["FINAL"]["argv"]
    assert final[4:8] == pop
    assert m["arms"]["FINAL-DYN"]["argv"] == final[:4] + ["--gomea-pop-size", "16"] + final[8:]
    assert m["arms"]["FINAL-DYN"]["budget"] == m["arms"]["FINAL"]["budget"]
    assert (out / "spec.json").read_text() == (CONFIGS / "phase1ax" / "spec.json").read_text()
    c = _choice(out)
    assert c["rule2_passes"] is True and c["size_better"] == 4 and c["chosen_arm"] == "LMVAL-PEN125"
    assert c["report_stage"] == 1 and c["per_dataset"]["Tow"]["nmse_ratio"] == pytest.approx(1.05)
    assert c["final_argv"] == _aw(tmp_path)["arms"]["LMVAL-PEN125"]["argv"]


def test_three_size_wins_fail_rule_2(tmp_path):
    sv = dict.fromkeys(D6, "tie") | dict.fromkeys(["WiR", "Con", "EnH"], "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv, GOOD)
    rc, out = _run(tmp_path)
    assert rc == 0
    m = json.loads((out / "sr.json").read_text())
    assert m["arms"]["FINAL"]["argv"] == _aw(tmp_path)["arms"]["LMVAL"]["argv"]
    c = _choice(out)
    assert c["rule2_passes"] is False and c["size_better"] == 3 and c["chosen_arm"] == "LMVAL"


def test_one_nmse_ratio_above_1_10_fails_rule_2(tmp_path):
    sv = dict.fromkeys(D6, "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv, GOOD | {"Yac": (1.111, 1.0)})
    rc, out = _run(tmp_path)
    assert rc == 0
    c = _choice(out)
    assert c["size_better"] == 6 and c["nmse_ratio_all_ok"] is False
    assert c["per_dataset"]["Yac"]["nmse_ratio_ok"] is False
    assert c["chosen_arm"] == "LMVAL"


def test_ratio_of_exactly_1_10_passes(tmp_path):
    sv = dict.fromkeys(D6, "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv, GOOD | {"Yac": (1.1, 1.0)})
    rc, out = _run(tmp_path)
    assert rc == 0 and _choice(out)["chosen_arm"] == "LMVAL-PEN125"


def test_failed_runs_infinite_medians(tmp_path):
    """inf / inf is not a number: that dataset fails the limit. A finite PEN125 median over an
    infinite LMVAL median is a ratio of 0."""
    sv = dict.fromkeys(D6, "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv,
            GOOD | {"Con": (float("inf"), float("inf"))})
    rc, out = _run(tmp_path)
    assert rc == 0
    c = _choice(out)
    assert c["per_dataset"]["Con"]["nmse_ratio"] == "nan" and c["chosen_arm"] == "LMVAL"
    tmp2 = tmp_path / "b"
    _report(tmp2 / "res" / "phase1aw" / "report-stage1", sv, GOOD | {"Con": (2.0, float("inf"))})
    rc, out = _run(tmp2)
    assert rc == 0 and _choice(out)["chosen_arm"] == "LMVAL-PEN125"


def test_stage_2_verdicts_only_for_extended_contrasts(tmp_path):
    """Stage 1: 3 wins and Con "extend". Stage 2 (extended): Con "A better", but the stage-2
    report also says "tie" for WiR, which was not "extend": WiR keeps its stage-1 win. The NMSE
    medians come from the stage-2 report."""
    s1 = dict.fromkeys(D6, "tie") | dict.fromkeys(["WiR", "EnH", "Air"], "A better")
    s1["Con"] = "extend"
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", s1, GOOD | {"Air": (9.0, 1.0)})
    s2 = dict(s1, Con="A better", WiR="tie")
    _report(tmp_path / "res" / "phase1aw" / "report-stage2", s2, GOOD, pairs=20)
    rc, out = _run(tmp_path)
    assert rc == 0
    c = _choice(out)
    assert c["report_stage"] == 2 and c["extended_at_stage1"] == ["Con:pen125-size"]
    assert c["per_dataset"]["WiR"]["size_verdict"] == "A better"
    assert c["per_dataset"]["Con"]["size_verdict"] == "A better"
    assert c["size_better"] == 4 and c["per_dataset"]["Air"]["nmse_ratio"] == pytest.approx(1.05)
    assert c["chosen_arm"] == "LMVAL-PEN125"


def test_extend_at_stage_1_without_a_stage_2_report_refuses(tmp_path):
    s1 = dict.fromkeys(D6, "tie") | {"Con": "extend"}
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", s1, GOOD)
    rc, out = _run(tmp_path)
    assert rc == 1 and not out.exists()


def test_no_report_refuses_and_writes_nothing(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 1 and not out.exists()


def test_a_choice_is_made_once(tmp_path):
    sv = dict.fromkeys(D6, "A better")
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", sv, GOOD)
    rc, out = _run(tmp_path)
    assert rc == 0
    rc, out = _run(tmp_path)  # the same choice again: fine
    assert rc == 0
    (out / "sr.json").write_text("{}\n")
    rc, out = _run(tmp_path)
    assert rc == 1 and (out / "sr.json").read_text() == "{}\n"
    rc, out = _run(tmp_path, "--force")
    assert rc == 0 and json.loads((out / "sr.json").read_text())["arms"]["FINAL"]["argv"]


def test_dry_run_writes_nothing(tmp_path):
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", dict.fromkeys(D6, "tie"), GOOD)
    rc, out = _run(tmp_path, "--dry-run")
    assert rc == 0 and not out.exists()


def test_the_choice_leaves_the_tracked_tree_unchanged(tmp_path):
    before = subprocess.run(["git", "status", "--porcelain", "--", "configs"], cwd=REPO,
                            capture_output=True, text=True).stdout
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", dict.fromkeys(D6, "A better"), GOOD)
    rc, out = _run(tmp_path)
    assert rc == 0
    after = subprocess.run(["git", "status", "--porcelain", "--", "configs"], cwd=REPO,
                           capture_output=True, text=True).stdout
    assert after == before
    assert json.loads((CONFIGS / "phase1ax" / "template.json").read_text())[
        "arms"]["FINAL"]["argv"] is None
    assert not (CONFIGS / "phase1ax" / "sr.json").exists()


def test_smoke_source_and_target(tmp_path):
    """The smoke plan reads server-smoke7-1aw and writes the smoke 1ax manifest."""
    _report(tmp_path / "res" / "server-smoke7-1aw" / "report-stage1",
            dict.fromkeys(D6, "tie"), GOOD)
    rc, out = _run(tmp_path, source="server-smoke7-1aw", target="server-smoke8-1ax")
    assert rc == 0
    m = json.loads((out / "sr.json").read_text())
    assert m["name"] == "server-smoke8-1ax-sr"
    smoke = _aw(tmp_path, "server-smoke7-1aw")
    assert m["arms"]["FINAL"]["argv"] == smoke["arms"]["LMVAL"]["argv"]
    assert m["arms"]["FINAL"]["budget"] == {"time_s": 15}


# ---- after the population round (SERVER-ROUND6-PREREGISTRATION.md)


def test_the_control_is_dyn_when_pop_is_n16(tmp_path):
    choice = {"pop": sp.population(None), "dyn": sp.population(0.05),
              "control": sp.population(0.05), "pop_arm": None, "dyn_arm": "DYN005-LTB"}
    _setup(tmp_path, choice=choice)
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", dict.fromkeys(D6, "tie"), GOOD)
    rc, out = _run(tmp_path)
    assert rc == 0
    m = json.loads((out / "sr.json").read_text())
    final, dyn = m["arms"]["FINAL"]["argv"], m["arms"]["FINAL-DYN"]["argv"]
    assert final[4:6] == ["--gomea-pop-size", "16"]
    assert final == _aw(tmp_path)["arms"]["LMVAL"]["argv"]
    assert dyn == final[:4] + ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05"] + (
        final[6:])
    assert m["arms"]["NOCAP"]["argv"][4:6] == ["--gomea-pop-size", "16"]
    c = _choice(out)
    assert c["population"]["control"]["name"] == "DYN0.05"
    assert c["population"]["final_dyn_argv"] == dyn


def test_the_template_needs_the_population(tmp_path):
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", dict.fromkeys(D6, "tie"), GOOD)
    rc, out = _run(tmp_path, population=False)
    assert rc == 1 and not out.exists()


def test_an_unresolved_source_is_refused(tmp_path):
    """--source-dir must hold the resolved 1aw: a copy of the template is refused."""
    _report(tmp_path / "res" / "phase1aw" / "report-stage1", dict.fromkeys(D6, "tie"), GOOD)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sr.json").write_text((CONFIGS / "phase1aw" / "template.json").read_text())
    cf, _ = _setup(tmp_path)
    out = tmp_path / "state" / "x"
    rc = set_final.main(["--results", str(tmp_path / "res"), "--out", str(out),
                         "--source-dir", str(raw), "--population", str(cf)])
    assert rc == 1 and not out.exists()
