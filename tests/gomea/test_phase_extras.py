"""Tests for tools/gomea/phase_extras.py on synthetic result directories."""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import phase_extras  # noqa: E402

GOMEA_ARGV = ["-a", "gomea", "--gomea-max-evals-per-deme", "250"]


def _reg(arm, split, test, train, prog="+($1 *($2 3))", status="ok", argv=GOMEA_ARGV):
    return {"arm": arm, "task": "Con", "seed": split, "status": status, "argv": argv,
            "test_nmse": test if status == "ok" else None, "train_nmse": train,
            "cpu_s": 100.0 * split, "final_program": prog}


def _bool(arm, seed, argv=GOMEA_ARGV):
    return {"arm": arm, "task": "mux11", "seed": seed, "status": "ok", "success": True,
            "argv": argv, "cpu_s": 10.0, "best_cpx": 30}


def _events(gens, longs=None):
    lines = []
    for i, g in enumerate(gens):
        start = {"type": "invocation_start", "deme_id": str(i)}
        if longs is not None:
            start["long"] = longs[i]
        lines.append(start)
        lines += [{"type": "generation", "generation": k} for k in range(g)]
        lines.append({"type": "end", "generations": g})
    return "\n".join(json.dumps(e) for e in lines) + "\n"


def _write(root, name, runs):
    d = Path(root) / name
    for s, ev in runs:
        rd = d / "runs" / f"{s['task']}__{s['arm']}__s{s['seed']}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "summary.json").write_text(json.dumps(s))
        if ev is not None:
            (rd / "events.jsonl").write_text(ev)
    return str(d)


class TestPhaseExtras(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, rows, arm, group="all"):
        return next(r for r in rows if r["arm"] == arm and r["deme_group"] == group)

    def test_regression_medians(self):
        d = _write(self.root, "r", [
            (_reg("A", 1, 10.0, 8.0, "+($1 $2)"), None),  # size 4 (phase_report count), gap 2
            (_reg("A", 2, 12.0, 6.0, "+($1 *($2 3))"), None),  # size 7, gap 6, ratio 2
            (_reg("A", 3, None, 5.0, status="failed"), None),
        ])
        r = self._row(phase_extras.build([d]), "A")
        self.assertEqual((r["runs"], r["not_ok"], r["kind"]), (3, 1, "regression"))
        self.assertEqual(r["median_test_nmse"], 12.0)  # failed run counts as inf
        self.assertEqual(r["median_train_nmse"], 7.0)
        self.assertEqual(r["median_test_minus_train"], 4.0)
        self.assertAlmostEqual(r["median_test_over_train"], 1.625)
        self.assertEqual(r["median_size"], 5.5)
        self.assertEqual(r["median_cpu_s"], 200.0)
        self.assertIsNone(r["demes"])  # no events.jsonl

    def test_later_dir_overrides(self):
        a = _write(self.root, "a", [(_reg("A", 1, 10.0, 8.0), None)])
        b = _write(self.root, "b", [(_reg("A", 1, 20.0, 8.0), None)])
        self.assertEqual(self._row(phase_extras.build([a, b]), "A")["median_test_nmse"], 20.0)

    def test_generations_per_deme(self):
        d = _write(self.root, "g", [
            (_bool("G", 1), _events([1, 1, 1, 2])),
            (_bool("G", 2), _events([1, 3])),
            (_bool("HC", 1, argv=["-a", "hc"]), None),
        ])
        rows = phase_extras.build([d])
        g = self._row(rows, "G")
        self.assertEqual((g["demes"], g["telemetry_runs"]), (6, 2))
        self.assertEqual(g["median_gens_per_deme"], 1.0)
        self.assertAlmostEqual(g["share_demes_ge2_gens"], 2 / 6)
        self.assertEqual((g["min_run_share_ge2"], g["max_run_share_ge2"]), (0.25, 0.5))
        self.assertEqual([r["deme_group"] for r in rows if r["arm"] == "G"], ["all"])
        self.assertIsNone(self._row(rows, "HC")["demes"])

    def test_generation_count_without_end_field(self):
        p = Path(self.root) / "rd"
        p.mkdir()
        ev = [{"type": "invocation_start"}, {"type": "generation"}, {"type": "generation"},
              {"type": "end"}]
        (p / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\nnot json\n")
        self.assertEqual(phase_extras.demes(p), [{"gens": 2, "long": None}])

    def test_long_flag_groups(self):
        d = _write(self.root, "l", [
            (_bool("L", 1), _events([1, 1, 4], longs=[False, False, True])),
            (_bool("L", 2), _events([1, 1], longs=[False, False])),
        ])
        rows = phase_extras.build([d])
        lg, nm = self._row(rows, "L", "long"), self._row(rows, "L", "normal")
        self.assertEqual((lg["demes"], lg["share_demes_ge2_gens"]), (1, 1.0))
        self.assertEqual((nm["demes"], nm["share_demes_ge2_gens"]), (4, 0.0))
        self.assertEqual(self._row(rows, "L")["demes"], 5)

    def test_cli_writes_csv_and_md(self):
        d = _write(self.root, "c", [(_reg("A", 1, 10.0, 8.0), None),
                                    (_bool("G", 1), _events([1, 2]))])
        out = Path(self.root) / "out"
        self.assertEqual(phase_extras.main([d, "--out", str(out)]), 0)
        with (out / "extras.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({(r["problem"], r["arm"]) for r in rows},
                         {("Con", "A"), ("mux11", "G")})
        md = (out / "extras.md").read_text()
        self.assertIn("| Con | A | 1 | 0 | 10.0 | 8.0 | 2.0 | 1.25 |", md)
        self.assertIn("50.0 %", md)
        self.assertTrue(math.isfinite(float(rows[0]["median_cpu_s"])))

    def test_cli_no_runs(self):
        self.assertEqual(phase_extras.main([self.root, "--out", self.root + "/o"]), 1)


if __name__ == "__main__":
    unittest.main()
