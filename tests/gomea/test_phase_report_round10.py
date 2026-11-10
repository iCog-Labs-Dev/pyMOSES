"""Server round 10 (SERVER-ROUND10-PREREGISTRATION.md): the spec key
`"pooled_secondary_families"` of tools/gomea/phase_report.py. Each inner list of pooled
secondary contrast names is a Holm family of its own (1bd: the seeding control, 4, and HC
widen search, 1); the other pooled secondary contrasts form one more family (1bd: size and
CPU, 10). Without the key, all pooled secondary contrasts are one family, as before."""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import phase_report  # noqa: E402

from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

PROBLEMS = ["Air", "Bos"]
SPLITS = range(1, 11)


def _reg(task, arm, split, nmse):
    return {"arm": arm, "task": task, "seed": split, "status": "ok", "test_nmse": nmse,
            "train_nmse": nmse, "cpu_s": 10.0, "n_evals": 500, "final_program": "(+ $x1 $x2)"}


def _write(root: Path, summaries) -> str:
    d = root / "r"
    for s in summaries:
        rd = d / "runs" / f"{s['task']}__{s['arm']}__s{s['seed']}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "summary.json").write_text(json.dumps(s))
    return str(d)


def _data(tmp_path):
    """B is the reference. A is 0.5 x B on every pair (20 pairs, A lower on all); C equals B
    times 1 + 0.001 k (B lower on all, ratio inside the band); D is 0.5 x B."""
    out = []
    for p in PROBLEMS:
        for k in SPLITS:
            b = 0.1 + k / 100
            out += [_reg(p, "B", k, b), _reg(p, "A", k, 0.5 * b),
                    _reg(p, "C", k, b * (1 + 0.001 * k)), _reg(p, "D", k, 0.5 * b)]
    return _write(tmp_path, out)


def _spec(**kw):
    return dict({
        "phase": "t", "metrics": dict.fromkeys(PROBLEMS, "test_nmse"), "primary": [],
        "pooled_primary": [["a-vs-b", "A", "B", "test_nmse"]],
        "pooled_secondary": [["d-vs-b", "D", "B", "test_nmse"],
                             ["a-vs-c", "A", "C", "test_nmse"],
                             ["c-vs-b", "C", "B", "test_nmse"],
                             ["a-vs-b-size", "A", "B", "program_size"]],
    }, **kw)


def _run(tmp_path, spec, d):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / "out"
    rc = phase_report.main(["--stage", "1", str(sp), str(out), d])
    if rc != 0:
        return rc, None, None
    rows = {(r["problem"], r["contrast"]): r
            for r in csv.DictReader((out / "contrasts.csv").open())}
    return rc, json.loads((out / "verdict.json").read_text()), rows


P20 = wilcoxon_signed_rank([float(i + 1) for i in range(20)])[1]


def test_default_keeps_one_pooled_secondary_family(tmp_path):
    rc, v, rows = _run(tmp_path, _spec(), _data(tmp_path))
    assert rc == 0
    r = rows[("pooled", "d-vs-b")]
    assert float(r["p"]) == pytest.approx(P20)
    # 4 contrasts in one family; three have p = P20, size has p = 1
    assert float(r["p_holm"]) == pytest.approx(holm([P20, P20, P20, 1.0])[0])
    assert "holm_family" not in r


def test_each_family_has_its_own_holm(tmp_path):
    spec = _spec(pooled_secondary_families=[["d-vs-b"], ["a-vs-c", "c-vs-b"]])
    rc, v, rows = _run(tmp_path, spec, _data(tmp_path))
    assert rc == 0
    fam = {k[1]: r["holm_family"] for k, r in rows.items() if r["family"] == "secondary"}
    assert fam == {"d-vs-b": "pooled_secondary/d-vs-b",
                   "a-vs-c": "pooled_secondary/a-vs-c+c-vs-b",
                   "c-vs-b": "pooled_secondary/a-vs-c+c-vs-b",
                   "a-vs-b-size": "pooled_secondary"}
    assert float(rows[("pooled", "d-vs-b")]["p_holm"]) == pytest.approx(P20)  # alone
    assert float(rows[("pooled", "a-vs-c")]["p_holm"]) == pytest.approx(holm([P20, P20])[0])
    assert float(rows[("pooled", "a-vs-b-size")]["p_holm"]) == 1.0
    # the same verdict rule in each family
    assert v["pooled:d-vs-b"] == "A better"
    assert v["pooled:a-vs-c"] == "A better"
    assert v["pooled:c-vs-b"] == "tie"  # ratio inside the band
    assert v["pooled:a-vs-b-size"] == "tie"


def test_the_primary_family_does_not_change(tmp_path):
    d = _data(tmp_path)
    _, v0, rows0 = _run(tmp_path, _spec(), d)
    _, v1, rows1 = _run(tmp_path, _spec(pooled_secondary_families=[["d-vs-b"]]), d)
    p0, p1 = rows0[("pooled", "a-vs-b")], rows1[("pooled", "a-vs-b")]
    assert p0["p_holm"] == p1["p_holm"] and p0["verdict"] == p1["verdict"] == "A better"
    assert list(v0) == list(v1)  # the verdict keys and their order do not change


def test_per_problem_secondary_rows_keep_holm_and_no_verdict(tmp_path):
    """1bd: the 40 per-dataset contrasts are "secondary" with secondary_verdicts false."""
    sec = [[p, "a-vs-b", "A", "B", "test_nmse"] for p in PROBLEMS]
    spec = _spec(secondary=sec, secondary_verdicts=False,
                 pooled_secondary_families=[["d-vs-b"]])
    rc, v, rows = _run(tmp_path, spec, _data(tmp_path))
    assert rc == 0
    p10 = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    for p in PROBLEMS:
        r = rows[(p, "a-vs-b")]
        assert float(r["p_holm"]) == pytest.approx(holm([p10, p10])[0])
        assert r["verdict"] == "" and v[f"{p}:a-vs-b"] is None
        assert r["holm_family"] == ""
    assert v["pooled:d-vs-b"] == "A better"


@pytest.mark.parametrize("bad", [
    [["nope"]],                      # not a pooled secondary contrast
    [["d-vs-b"], ["d-vs-b"]],        # in two lists
    [[]],                            # an empty list
    ["d-vs-b"],                      # not a list of lists
])
def test_bad_pooled_secondary_families_are_refused(tmp_path, bad):
    rc, _, _ = _run(tmp_path, _spec(pooled_secondary_families=bad), _data(tmp_path))
    assert rc == 2


def test_the_key_needs_pooled_secondary_contrasts(tmp_path):
    spec = _spec(pooled_secondary_families=[["d-vs-b"]])
    del spec["pooled_secondary"]
    rc, _, _ = _run(tmp_path, spec, _data(tmp_path))
    assert rc == 2


def test_round8_failure_rule_holds_in_a_family(tmp_path):
    """A failed run is infinitely bad in an own family too."""
    out = []
    for p in PROBLEMS:
        for k in SPLITS:
            b = 0.1 + k / 100
            out += [_reg(p, "B", k, b), _reg(p, "A", k, 0.5 * b), _reg(p, "C", k, b),
                    dict(_reg(p, "D", k, b), status="error", test_nmse=None)]
    d = _write(tmp_path, out)
    rc, v, rows = _run(tmp_path, _spec(pooled_secondary_families=[["d-vs-b"]]), d)
    r = rows[("pooled", "d-vs-b")]
    assert float(r["median_log_ratio"]) == 1e12 and math.isinf(float(r["ratio"]))
    assert v["pooled:d-vs-b"] == "A worse"
