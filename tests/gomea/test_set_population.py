"""tools/gomea/set_population.py (server round 6): the choice of POP and DYN from fake 1ay
reports (the qualification rule, the median rank, the ties, stage 2), the resolved manifests
of 1az, `apply` and the refusals."""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import set_population as sp  # noqa: E402

SPEC = json.loads((CONFIGS / "phase1ay" / "spec.json").read_text())
P3 = ["mux11", "Con", "EnH"]
DYNS = ["DYN2-LTB", "DYN025-LTB", "DYN005-LTB"]
FIELDS = ["problem", "contrast", "family", "a", "b", "metric", "pairs", "median_a", "median_b",
          "p", "p_holm", "verdict"]


def _report(d: Path, verdicts: dict, medians: dict, other="tie"):
    """verdicts: {(problem, arm): verdict of arm vs N16-LTB}; medians: {(problem, arm): median
    metric}. Missing entries are `other` and 1.0."""
    d.mkdir(parents=True)
    rows, verdict = [], {}
    for fam in ("primary", "secondary"):
        for p, n, a, b in SPEC[fam]:
            v = verdicts.get((p, a), other) if fam == "primary" else "tie"
            rows.append(dict(problem=p, contrast=n, family=fam, a=a, b=b, metric="m",
                             pairs=10, median_a=1, median_b=1, p=0.5, p_holm=0.01, verdict=v))
            verdict[f"{p}:{n}"] = v
    with (d / "contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (d / "aggregate.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["problem", "arm", "runs", "median_metric"])
        w.writeheader()
        for p in P3:
            for a in ["N16-LTB", "N16-RT", *DYNS, "DYN2-RT", "DYN025-RT", "DYN005-RT"]:
                w.writerow(dict(problem=p, arm=a, runs=10, median_metric=medians.get((p, a), 1.0)))
    (d / "verdict.json").write_text(json.dumps(verdict))


def _med(**by_arm):
    """The same median of each arm on every problem: _med(DYN2=3, DYN025=1, DYN005=2)."""
    return {(p, f"{a}-LTB"): v for p in P3 for a, v in by_arm.items()}


def _choose(tmp_path, *extra):
    out = tmp_path / "state" / "phase1az-config"
    rc = sp.main(["choose", "--results", str(tmp_path / "res"), "--out", str(out), *extra])
    return rc, out


def _rec(out):
    return json.loads((out / "population-choice.json").read_text())


R1 = Path("res") / "phase1ay" / "report-stage1"


def test_none_qualifies_pop_is_n16_and_dyn_is_the_best_median_rank(tmp_path):
    _report(tmp_path / R1, {("mux11", "DYN025-LTB"): "A better"}, _med(DYN2=3, DYN025=2, DYN005=1))
    rc, out = _choose(tmp_path)
    assert rc == 0
    c = _rec(out)
    assert c["pop"]["name"] == "N16" and c["pop_arm"] is None
    assert c["dyn"]["argv"] == ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05"]
    assert c["control"] == c["dyn"]  # POP != DYN
    assert c["per_arm"]["DYN005-LTB"]["median_rank"] == 1
    assert c["per_arm"]["DYN025-LTB"]["qualifies"] is False


def test_one_qualifies_and_is_dyn_so_the_control_is_n16(tmp_path):
    v = {("mux11", "DYN025-LTB"): "A better", ("Con", "DYN025-LTB"): "A better"}
    _report(tmp_path / R1, v, _med(DYN2=3, DYN025=1, DYN005=2))
    rc, out = _choose(tmp_path)
    c = _rec(out)
    assert rc == 0 and c["pop_arm"] == c["dyn_arm"] == "DYN025-LTB"
    assert c["pop"]["argv"] == ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.25"]
    assert c["control"]["name"] == "N16"
    assert c["control"]["argv"] == ["--gomea-pop-size", "16"]


def test_pop_and_dyn_can_differ(tmp_path):
    """DYN2 qualifies but ranks worse than DYN005, which does not qualify."""
    v = {("Con", "DYN2-LTB"): "A better", ("EnH", "DYN2-LTB"): "A better"}
    _report(tmp_path / R1, v, _med(DYN2=2, DYN025=3, DYN005=1))
    c = _rec(_choose(tmp_path)[1])
    assert c["pop_arm"] == "DYN2-LTB" and c["dyn_arm"] == "DYN005-LTB"
    assert c["control"] == c["dyn"]


def test_worse_on_one_problem_disqualifies(tmp_path):
    v = {("mux11", "DYN2-LTB"): "A better", ("Con", "DYN2-LTB"): "A better",
         ("EnH", "DYN2-LTB"): "A worse"}
    _report(tmp_path / R1, v, _med(DYN2=1, DYN025=2, DYN005=3))
    c = _rec(_choose(tmp_path)[1])
    assert c["pop"]["name"] == "N16" and c["dyn_arm"] == "DYN2-LTB"


def test_several_qualify_the_best_median_rank_wins(tmp_path):
    v = {(p, a): "A better" for p in ("Con", "EnH") for a in ("DYN2-LTB", "DYN025-LTB")}
    # ranks: DYN2 (1, 3, 3) -> 3; DYN025 (2, 1, 1) -> 1
    med = {("mux11", "DYN2-LTB"): 1, ("mux11", "DYN025-LTB"): 2, ("mux11", "DYN005-LTB"): 3,
           ("Con", "DYN2-LTB"): 3, ("Con", "DYN025-LTB"): 1, ("Con", "DYN005-LTB"): 2,
           ("EnH", "DYN2-LTB"): 3, ("EnH", "DYN025-LTB"): 1, ("EnH", "DYN005-LTB"): 2}
    _report(tmp_path / R1, v, med)
    c = _rec(_choose(tmp_path)[1])
    assert c["per_arm"]["DYN2-LTB"]["median_rank"] == 3
    assert c["per_arm"]["DYN025-LTB"]["median_rank"] == 1
    assert c["pop_arm"] == "DYN025-LTB"


def test_ties_go_to_the_smaller_f(tmp_path):
    """Equal medians rank the smaller F first; equal median ranks choose the smaller F."""
    v = {(p, a): "A better" for p in ("Con", "EnH") for a in DYNS}
    _report(tmp_path / R1, v, {})  # every median 1.0
    c = _rec(_choose(tmp_path)[1])
    assert c["per_arm"]["DYN005-LTB"]["ranks"] == {"mux11": 1, "Con": 1, "EnH": 1}
    assert c["pop_arm"] == c["dyn_arm"] == "DYN005-LTB"
    # ranks DYN2 (1, 3, 2), DYN025 (2, 1, 3), DYN005 (3, 2, 1): all median 2 -> DYN005
    med = {("mux11", "DYN2-LTB"): 1, ("mux11", "DYN025-LTB"): 2, ("mux11", "DYN005-LTB"): 3,
           ("Con", "DYN2-LTB"): 3, ("Con", "DYN025-LTB"): 1, ("Con", "DYN005-LTB"): 2,
           ("EnH", "DYN2-LTB"): 2, ("EnH", "DYN025-LTB"): 3, ("EnH", "DYN005-LTB"): 1}
    (tmp_path / "res").rename(tmp_path / "old")
    _report(tmp_path / R1, {}, med)
    c = _rec(_choose(tmp_path, "--force")[1])
    assert c["per_arm"]["DYN2-LTB"]["median_rank"] == 2
    assert c["per_arm"]["DYN025-LTB"]["median_rank"] == 2
    assert c["per_arm"]["DYN005-LTB"]["median_rank"] == 2
    assert c["dyn_arm"] == "DYN005-LTB"


def test_an_unsolved_median_is_infinite_and_ranks_last(tmp_path):
    med = _med(DYN2=1, DYN025=1, DYN005=1) | {(p, "DYN005-LTB"): "inf" for p in P3}
    _report(tmp_path / R1, {}, med)
    c = _rec(_choose(tmp_path)[1])
    assert c["per_arm"]["DYN005-LTB"]["median_rank"] == 3
    assert c["dyn_arm"] == "DYN025-LTB"


def test_stage_2_verdicts_only_for_extended_contrasts(tmp_path):
    v1 = {("mux11", "DYN2-LTB"): "extend", ("Con", "DYN2-LTB"): "A better"}
    _report(tmp_path / R1, v1, _med(DYN2=1, DYN025=2, DYN005=3))
    # stage 2 says better on mux11 (extended) and worse on Con (not extended: ignored)
    _report(tmp_path / "res" / "phase1ay" / "report-stage2",
            {("mux11", "DYN2-LTB"): "A better", ("Con", "DYN2-LTB"): "A worse"},
            _med(DYN2=1, DYN025=2, DYN005=3))
    c = _rec(_choose(tmp_path)[1])
    assert c["report_stage"] == 2 and c["extended_at_stage1"] == ["mux11:dyn2-vs-n16"]
    assert c["per_arm"]["DYN2-LTB"]["better"] == ["mux11", "Con"]
    assert c["pop_arm"] == "DYN2-LTB"


def test_resolved_1az_manifests(tmp_path):
    _report(tmp_path / R1, {}, _med(DYN2=2, DYN025=1, DYN005=3))
    rc, out = _choose(tmp_path)
    assert rc == 0
    names = sorted(f.name for f in out.iterdir())
    assert names == ["bool-ext-mux11.json", "bool.json", "population-choice.json", "spec.json",
                     "sr-ext-Air.json", "sr-ext-Con.json", "sr-ext-EnH.json", "sr.json"]
    dyn = ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.25"]
    q = CONFIGS / "phase1az"
    for name in names:
        if name in ("population-choice.json", "spec.json"):
            continue
        m = json.loads((out / name).read_text())
        tname = "template.json" if name == "sr.json" else f"template-{name}"
        t = json.loads((q / tname).read_text())
        assert {k: v for k, v in m.items() if k != "arms"} == {
            k: v for k, v in t.items() if k != "arms"}
        for arm, spec in m["arms"].items():
            argv = spec["argv"]
            assert not any("{" in x for x in argv)
            exp = [y for x in t["arms"][arm]["argv"] for y in (dyn if x == "{DYN}" else [x])]
            assert argv == exp, (name, arm)
            if arm.endswith("-DYN"):
                assert argv[4:8] == dyn
            else:
                assert argv[4:6] == ["--gomea-pop-size", "16"]
    assert (out / "spec.json").read_text() == (q / "spec.json").read_text()


def test_apply_resolves_another_phase(tmp_path):
    _report(tmp_path / R1, {("Con", "DYN2-LTB"): "A better", ("EnH", "DYN2-LTB"): "A better"},
            _med(DYN2=1, DYN025=2, DYN005=3))
    rc, out = _choose(tmp_path)
    assert rc == 0
    aw = tmp_path / "state7" / "phase1aw-config"
    rc = sp.main(["apply", "--choice", str(out / "population-choice.json"), "--target",
                  "phase1aw", "--out", str(aw)])
    assert rc == 0
    m = json.loads((aw / "sr.json").read_text())
    pop = ["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "2.0"]
    assert m["arms"]["NOCAP"]["argv"][4:8] == pop
    assert m["arms"]["LMVAL-DYN"]["argv"][4:6] == ["--gomea-pop-size", "16"]  # POP == DYN
    rec = json.loads((aw / "population-applied.json").read_text())
    assert rec["pop"]["name"] == "DYN2" and rec["control"]["name"] == "N16"


def test_missing_report_writes_nothing(tmp_path):
    rc, out = _choose(tmp_path)
    assert rc == 1 and not out.exists()


def test_missing_median_writes_nothing(tmp_path):
    _report(tmp_path / R1, {}, {("Con", "DYN2-LTB"): ""})
    rc, out = _choose(tmp_path)
    assert rc == 1 and not out.exists()


def test_a_second_different_choice_is_refused(tmp_path):
    _report(tmp_path / R1, {}, _med(DYN2=1, DYN025=2, DYN005=3))
    assert _choose(tmp_path)[0] == 0
    assert _choose(tmp_path)[0] == 0  # the same choice again: allowed
    (tmp_path / "res").rename(tmp_path / "old")
    _report(tmp_path / R1, {}, _med(DYN2=3, DYN025=2, DYN005=1))
    rc, out = _choose(tmp_path)
    assert rc == 1 and _rec(out)["dyn_arm"] == "DYN2-LTB"
    assert _choose(tmp_path, "--force")[0] == 0 and _rec(out)["dyn_arm"] == "DYN005-LTB"


def test_dry_run_writes_nothing(tmp_path, capsys):
    _report(tmp_path / R1, {}, _med(DYN2=1, DYN025=2, DYN005=3))
    rc, out = _choose(tmp_path, "--dry-run")
    assert rc == 0 and not out.exists()
    assert "-> POP = N16" in capsys.readouterr().out


@pytest.mark.parametrize("argv, frac", [
    (["-a", "gomea", "--gomea-pop-size", "16"], None),
    (["--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05", "-x"], 0.05),
])
def test_fraction_of(argv, frac):
    assert sp.fraction_of(argv) == frac


def test_swap_population_keeps_the_place():
    a = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
         "--hc-fraction-of-nn", "0.25", "--x", "1"]
    n16 = sp.population(None)
    assert sp.swap_population(a, n16) == ["-a", "gomea", "--gomea-fos", "lt-b",
                                          "--gomea-pop-size", "16", "--x", "1"]
    back = sp.swap_population(sp.swap_population(a, n16), sp.population(0.25))
    assert back == a


def test_the_smoke_plan_reads_the_smoke_phases(tmp_path):
    _report(tmp_path / "res" / "server-smoke6-1ay" / "report-stage1", {},
            _med(DYN2=1, DYN025=2, DYN005=3))
    out = tmp_path / "state" / "cfg"
    rc = sp.main(["choose", "--results", str(tmp_path / "res"), "--source", "server-smoke6-1ay",
                  "--target", "server-smoke6-1az", "--out", str(out)])
    assert rc == 0
    assert json.loads((out / "sr.json").read_text())["name"] == "server-smoke6-1az-sr"
