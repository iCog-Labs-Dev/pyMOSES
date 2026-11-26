"""Backlog 15.4, D1b: ``tools/gomea/d1b_report.py`` on made-up runs (no MOSES run).

* The decision of the rule written before the runs: pass, promising and no.
* The contrasts follow the project rule (``phase_report._pooled_contrast``): a failed run is
  infinitely bad, the ratio is exp(median log(A / B)), the p value is the two-sided exact
  Wilcoxon test (10 wins of 10: p = 2 / 1024).
* The deme statistics: GOMEA from ``events.jsonl`` (a deme with 0 requests is left out), HC
  from ``hc_progress.jsonl``; the size is the node count plus ``n_terms``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import d1b_report as rep  # noqa: E402

SPLITS = range(91, 101)


def _summary(arm, split, test, train, status="ok"):
    return {
        "task": "Yac", "arm": arm, "seed": split, "status": status, "test_nmse": test,
        "test_nmse_clipped": test, "train_nmse": train, "val_nmse": train,
        "final_program": "+(*(2 $1) $2)", "terms": {"n_terms": 2, "fallback": False},
        "cpu_s": 10.0, "wall_s": 11.0, "n_evals": 20000, "ru_maxrss_kb": 102400,
    }  # fmt: skip


def _gomea_events(start: str | None) -> str:
    init = {"type": "init"} | ({"contin_start": start} if start else {})
    evs = [
        {"type": "invocation_start", "deme_pop_size": 100, "dim": 16, "n_contin_genes": 16},
        init, {"type": "generation"}, {"type": "generation"},
        {"type": "end", "generations": 2, "ledger": {"requests": 1500},
         "termination_reason": "stagnation"},
        {"type": "invocation_start", "deme_pop_size": 100, "dim": 30, "n_contin_genes": 30},
        init, {"type": "end", "generations": 0, "ledger": {"requests": 0}},  # left out
        {"type": "invocation_start", "deme_pop_size": 80, "dim": 40, "n_contin_genes": 40},
        init, {"type": "generation"},
        {"type": "end", "generations": 1, "ledger": {"requests": 500},
         "termination_reason": "global_budget"},
    ]  # fmt: skip
    return "".join(json.dumps(e) + "\n" for e in evs)


def _write(root: Path, manifest: str, arm: str, split: int, s: dict, files: dict) -> None:
    d = root / manifest / "runs" / f"Yac__{arm}__s{split}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(s))
    for name, text in files.items():
        (d / name).write_text(text)


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "out"
    for i, split in enumerate(SPLITS):
        base = 10.0 + i
        _write(
            root,
            "d1b-yac",
            "BASE-TLS",
            split,
            _summary("BASE-TLS", split, base, base / 2),
            {"events.jsonl": _gomea_events(None)},
        )
        _write(
            root,
            "d1b-yac",
            "TLS-CW",
            split,
            _summary("TLS-CW", split, 0.8 * base, 0.4 * base),
            {"events.jsonl": _gomea_events("wide")},
        )
        hc = _summary("HC-TLS", split, 2.0 * base, base, status="ok" if i else "error")
        lines = [
            {"expansion": 1, "n_evals": 400},
            {"expansion": 2, "n_evals": 1400},
            {"expansion": 3, "n_evals": 2000},
        ]
        _write(
            root,
            "d1b-yac-hc",
            "HC-TLS",
            split,
            hc,
            {"hc_progress.jsonl": "".join(json.dumps(x) + "\n" for x in lines) + "{cut"},
        )
    return root  # fmt: skip


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def test_the_report_on_made_up_runs(results, tmp_path, capsys):
    out = tmp_path / "report"
    assert rep.main([str(results), "--out", str(out)]) == 0
    text = (out / "summary.txt").read_text()
    assert "Decision (pre-written rule, TLS-CW vs BASE-TLS): PASS" in text
    c = {(r["a"], r["metric"]): r for r in _rows(out / "contrasts.csv")}
    t = c[("TLS-CW", "test_nmse")]
    assert (t["pairs"], t["wins"], t["losses"], t["ties"]) == ("10", "10", "0", "0")
    assert float(t["ratio"]) == pytest.approx(0.8)
    assert float(t["median_ratio"]) == pytest.approx(0.8)
    assert float(t["p"]) == pytest.approx(2 / 1024)
    h = c[("HC-TLS", "test_nmse")]  # one failed HC run: infinitely bad, so a loss
    assert (h["wins"], h["losses"], h["kind"]) == ("0", "10", "descriptive")
    runs = _rows(out / "runs.csv")
    assert len(runs) == 30
    assert {r["size"] for r in runs} == {"9"}  # 2 "(" + 5 atoms + 2 terms (phase_report)
    assert {r["max_rss_mb"] for r in runs} == {"100.0"}
    st = {r["arm"]: r for r in _rows(out / "deme_stats.csv")}

    def num(arm, key):
        return float(st[arm][key])

    assert st["TLS-CW"]["contin_start"] == "wide" and st["BASE-TLS"]["contin_start"] == "step"
    assert num("TLS-CW", "demes_per_run_median") == 2  # the deme with 0 requests is left out
    assert num("TLS-CW", "evals_per_deme_median") == 1000
    assert num("TLS-CW", "evals_per_deme_max") == 1500
    assert num("TLS-CW", "pop_size_min") == 80 and num("TLS-CW", "generations_max") == 2
    assert st["HC-TLS"]["kind"] == "hc" and num("HC-TLS", "demes_per_run_median") == 3
    assert num("HC-TLS", "evals_per_deme_median") == 600
    assert num("HC-TLS", "evals_per_deme_max") == 1000
    assert len(_rows(out / "ratios.csv")) == 40
    assert "PASS" in capsys.readouterr().out


def _c(ratio, p, wins):
    return {"ratio": ratio, "median_ratio": ratio, "p": p, "wins": wins}


def test_the_decision_rule():
    assert rep.decide(_c(0.90, 0.02, 0), _c(1.0, 1.0, 0)) == "pass"
    assert rep.decide(_c(0.90, 0.025, 0), _c(1.0, 1.0, 7)) == "promising"
    assert rep.decide(_c(0.95, 0.5, 0), _c(1.0, 1.0, 7)) == "promising"
    assert rep.decide(_c(0.95, 0.5, 0), _c(1.0, 1.0, 6)) == "no"
    assert rep.decide(_c(0.96, 0.001, 0), _c(1.0, 1.0, 10)) == "no"
