"""Unit tests for tools/gomea/gpgomea_seed.py and seed_lineage.py (backlog item 9:
seeding MOSES with GP-GOMEA elites), and the seeding hooks of the benchmark runner."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import gpgomea_seed as gs  # noqa: E402
import moses_sr_runner  # noqa: E402
import reseed  # noqa: E402
import seed_lineage  # noqa: E402

from benchmarks.gomea import runner  # noqa: E402
from benchmarks.gomea.manifest import _validate, expand_runs  # noqa: E402

# A real `gpgomea-seed` dump line looks like this (tab separated).
_DUMP = (
    " > generation 1 - best fit: 0.25 - evaluations: 400\n"
    "SEED_ELITE\t0\t0.25\t0\t1\t(AND (OR x0 x2) (NOT x1))\n"
    "SEED_POP\t0\t0.375\t1.5\t-2\t(+ x0 (* 2.5 x3))\n"
    "Evaluations:\t401\n"
)


def _bool_eval(tree, row):
    if isinstance(tree, str):
        return row[int(tree[1:])]
    op, args = tree
    vals = [_bool_eval(a, row) for a in args]
    return {
        "AND": lambda: all(vals),
        "OR": lambda: any(vals),
        "NOT": lambda: not vals[0],
        "NAND": lambda: not all(vals),
        "NOR": lambda: not any(vals),
    }[op]()


def _num_eval(tree, row):
    if isinstance(tree, str):
        return row[int(tree[1:])] if tree.startswith("x") else float(tree)
    op, args = tree
    v = [_num_eval(a, row) for a in args]
    if op == "p/":  # GP-GOMEA's protected division
        sign = -1.0 if v[1] < 0 else 1.0
        return sign * v[0] / (1e-6 + abs(v[1]))
    return {"+": v[0] + v[-1], "-": v[0] - v[-1], "*": v[0] * v[-1], "^2": v[0] * v[0]}[op]


class TestParsing(unittest.TestCase):
    def test_dump_and_evaluations(self):
        d = gs.parse_dump(_DUMP)
        self.assertEqual([c["source"] for c in d], ["elite", "population"])
        self.assertEqual(d[1]["a"], 1.5)
        self.assertEqual(d[1]["b"], -2.0)
        self.assertEqual(d[1]["expr"], "(+ x0 (* 2.5 x3))")
        self.assertEqual(gs.parse_evaluations(_DUMP), 401)

    def test_prefix(self):
        t = gs.parse_prefix("(+ x0 (* 2.5 x3))")
        self.assertEqual(t, ("+", ["x0", ("*", ["2.5", "x3"])]))
        self.assertEqual(gs.count_nodes(t), 5)
        self.assertEqual(gs.parse_prefix("x4"), "x4")
        with self.assertRaises(ValueError):
            gs.parse_prefix("(+ x0 x1) x2")


class TestTranslation(unittest.TestCase):
    def test_boolean_forms(self):
        f = gs.bool_to_combo
        self.assertEqual(f(gs.parse_prefix("(AND (OR x0 x2) (NOT x1))")), "and(or($1 $3) !$2)")
        self.assertEqual(f(gs.parse_prefix("(NOT (AND x0 x1))")), "not(and($1 $2))")
        self.assertEqual(f(gs.parse_prefix("(NAND x0 x1)")), "not(and($1 $2))")
        self.assertEqual(f(gs.parse_prefix("(NOR x0 x1)")), "not(or($1 $2))")
        with self.assertRaises(gs.Untranslatable):
            f(gs.parse_prefix("(XOR x0 x1)"))

    def test_regression_forms(self):
        f = gs.regression_to_combo
        self.assertEqual(f(gs.parse_prefix("(- x0 x1)")), "+($1 *(-1.0 $2))")
        self.assertEqual(f(gs.parse_prefix("(p/ x0 2.5)")), "/($1 2.5)")
        self.assertEqual(f(gs.parse_prefix("(^2 x2)")), "*($3 $3)")
        with self.assertRaises(gs.Untranslatable):
            f(gs.parse_prefix("(aq x0 x1)"))
        with self.assertRaises(gs.Untranslatable):
            f(gs.parse_prefix("(sqrt x0)"))
        t = gs.parse_prefix("(* x0 x1)")
        self.assertEqual(gs.scaled_regression_to_combo(t, 1.5, -2.0), "+(1.5 *(-2.0 *($1 $2)))")
        self.assertEqual(gs.scaled_regression_to_combo(t, 3.25, 0.0), "3.25")
        # full precision constants
        self.assertEqual(gs._num(0.1 + 0.2), "0.30000000000000004")

    def test_boolean_round_trip_score(self):
        """GP-GOMEA's error rate of a program equals pyMoses's raw score of its
        translation, on mux6 (`-H mux -k 2`), for several programs."""
        flags = ["-H", "mux", "-k", "2"]
        xs, ys = gs.bool_rows(flags)
        exprs = [
            "(AND (OR x0 x2) (NOT x1))",
            "(OR (AND (NOT x0) (AND (NOT x1) x2)) (AND x0 (AND x1 x5)))",
            "(NAND (NOR x3 x4) x0)",
            "(OR (AND (OR (AND x2 (NOT x0)) (AND x3 x0)) (NOT x1))"
            " (AND (OR (AND x4 (NOT x0)) (AND x5 x0)) x1))",  # mux6 itself
        ]
        combos, expected = [], []
        for e in exprs:
            t = gs.parse_prefix(e)
            wrong = sum(_bool_eval(t, x) != y for x, y in zip(xs, ys, strict=True))
            fit = wrong / len(xs)  # GP-GOMEA's accuracy fitness: error rate
            combos.append(gs.bool_to_combo(t))
            expected.append(gs.expected_raw_score("bool", fit, len(xs)))
        scored = gs.score_programs(flags, combos)
        self.assertEqual([s["raw"] for s in scored], expected)
        self.assertEqual(expected[-1], 0.0)

    def test_regression_round_trip_score(self):
        """Linear scaling written into the tree: pyMoses's raw score (-SSE) equals
        -rows * GP-GOMEA's linearly scaled MSE."""
        rows = [
            [1.0, 2.0, 0.5, 3.0],
            [2.0, -1.0, 1.5, 2.0],
            [3.0, 0.5, -2.0, 1.0],
            [4.0, 1.5, 0.25, -1.0],
            [5.0, -0.5, 1.0, 0.5],
            [6.0, 2.5, -1.5, 2.5],
        ]
        ys = [3.0 * r[0] - r[1] * r[2] + 0.1 * i for i, r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            # target in the middle: MOSES's $k is the k-th non-target column
            lines = ["a,b,y,c,d"] + [
                f"{r[0]!r},{r[1]!r},{y!r},{r[2]!r},{r[3]!r}" for r, y in zip(rows, ys, strict=True)
            ]
            path.write_text("\n".join(lines) + "\n")
            dat = Path(tmp) / "train.dat"
            self.assertEqual(gs.write_regression_dat(path, dat), len(rows))
            first = dat.read_text().splitlines()[0].split()
            self.assertEqual([float(v) for v in first], rows[0] + [ys[0]])
            exprs = [
                "(- (* x0 2.5) (* x1 x2))",
                "(+ (p/ x0 x3) (^2 x1))",
                "(* x3 -1.25)",
            ]
            combos, expected = [], []
            for e in exprs:
                t = gs.parse_prefix(e)
                p = [_num_eval(t, r) for r in rows]
                mp, my = sum(p) / len(p), sum(ys) / len(ys)
                den = sum((v - mp) ** 2 for v in p)
                b = sum((v - mp) * (y - my) for v, y in zip(p, ys, strict=True)) / den
                a = my - b * mp
                mse = sum((y - a - b * v) ** 2 for v, y in zip(p, ys, strict=True)) / len(ys)
                combos.append(gs.scaled_regression_to_combo(t, a, b))
                expected.append(gs.expected_raw_score("regression", mse, len(ys)))
            scored = gs.score_programs(["-i", str(path), "-u", "y"], combos)
        for s, e in zip(scored, expected, strict=True):
            self.assertTrue(gs._matches(s["raw"], e, 1e-5), (s, e))


class TestBudget(unittest.TestCase):
    def test_shared_budget_and_argv(self):
        self.assertEqual(gs.shared_budget(40000, 8336), 31664)
        with self.assertRaises(ValueError):
            gs.shared_budget(1000, 1000)
        argv = ["-H", "mux", "-m", "40000", "-r", "1"]
        self.assertEqual(gs.replace_max_evals(argv, 31664)[3], "31664")
        self.assertEqual(argv[3], "40000")  # not modified in place
        with self.assertRaises(ValueError):
            gs.replace_max_evals(["-H", "mux"], 5)
        self.assertEqual(
            gs.exemplar_argv("bool", ["and($1 $2)"]), ["-e", "and($1 $2)", "-e", "and"]
        )
        self.assertEqual(gs.exemplar_argv("regression", [], with_empty=True), ["-e", "+"])
        self.assertEqual(gs.exemplar_argv("bool", ["$1"], with_empty=False), ["-e", "$1"])

    def test_empty_exemplars_parse_to_the_defaults(self):
        from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
        from pymoses.comboreduct.type_checker.type_signature import type_node
        from pymoses.moses.main.demo_problems import _type_to_exemplar

        for kind, t in (("bool", type_node.boolean_type), ("regression", type_node.contin_type)):
            self.assertEqual(
                combo_tree_to_str(str_to_combo_tree(gs.EMPTY_EXEMPLAR[kind])),
                combo_tree_to_str(_type_to_exemplar(t)),
            )

    def _seeded(self, tmp, seeds):
        run_dir = Path(tmp)
        rec = {
            "evaluations_used": 1000,
            "gpgomea": {"cpu_s": 2.0, "wall_s": 2.5, "best_fit": 0.25},
            "seeds": [{"pymoses_raw": s} for s in seeds],
            "seed_budget": 800,
            "moses_budget": 9000,
            "total_budget": 10000,
        }
        (run_dir / "seeds.json").write_text(json.dumps(rec))
        return {"seed_spec": {"binary": "x", "share": 0.08}, "target_raw": 0}, run_dir

    def test_runner_charges_gpgomea(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, run_dir = self._seeded(tmp, [-12.0, -14.0])
            s = {
                "status": "ok",
                "n_calls": 5000,
                "cpu_s": 10.0,
                "evals_to_target": 4000,
                "cpu_to_target": 8.0,
                "time_to_target": 9.0,
                "best_raw": 0.0,
                "success": True,
            }
            s = runner._add_seed_cost(s, run, run_dir)
            self.assertEqual(s["n_calls"], 6000)
            self.assertEqual(s["moses_n_calls"], 5000)
            self.assertEqual(s["evals_to_target"], 5000)
            self.assertEqual(s["moses_evals_to_target"], 4000)
            self.assertEqual(s["cpu_s"], 12.0)
            self.assertEqual(s["cpu_to_target"], 10.0)
            self.assertEqual(s["seeding"]["seed_best_raw"], -12.0)
            # an unsolved run keeps its own best unless a seed is better
            s = {
                "status": "ok",
                "n_calls": 9000,
                "cpu_s": 1.0,
                "evals_to_target": None,
                "best_raw": -20.0,
                "success": False,
            }
            s = runner._add_seed_cost(s, run, run_dir)
            self.assertFalse(s["success"])
            self.assertEqual(s["best_raw"], -12.0)
            self.assertIsNone(s["evals_to_target"])

    def test_runner_perfect_seed_is_a_success_at_gpgomea_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, run_dir = self._seeded(tmp, [0.0])
            s = {
                "status": "ok",
                "n_calls": 0,
                "cpu_s": 0.5,
                "evals_to_target": None,
                "best_raw": None,
                "success": False,
            }
            s = runner._add_seed_cost(s, run, run_dir)
            self.assertTrue(s["success"])
            self.assertEqual(s["evals_to_target"], 1000)
            self.assertEqual(s["best_raw"], 0.0)

    def test_unseeded_summary_untouched(self):
        s = {"status": "ok", "n_calls": 5}
        self.assertEqual(runner._add_seed_cost(dict(s), {"target_raw": 0}, Path(".")), s)


class TestManifest(unittest.TestCase):
    def _m(self, seed):
        return {
            "schema_version": 1,
            "name": "t",
            "phase": 1,
            "commit": "x",
            "common_argv": [],
            "tasks": {"mux6": {"argv": ["-H", "mux", "-k", "2"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}, "HC-S": {"argv": ["-a", "hc"], "seed": seed}},
            "seeds": [1],
            "budget": {"argv": ["-m", "1000"], "kind": "evaluations"},
            "wall_cap_s": 10,
            "workers": 1,
            "output_root": "out",
        }

    def test_seed_block(self):
        good = {"binary": "/b", "share": 0.2, "height": 3, "k": 5}
        self.assertEqual(_validate(self._m(good)), [])
        for bad in (
            {"share": 0.2},
            {"binary": "/b"},
            {"binary": "/b", "share": 0.2, "budget": 100},
            {"binary": "/b", "share": 1.5},
            {"binary": "/b", "share": 0.2, "k": 0},
            {"binary": "/b", "share": 0.2, "bogus": 1},
        ):
            self.assertTrue(_validate(self._m(bad)), bad)
        with tempfile.TemporaryDirectory() as tmp:
            runs = {r["arm"]: r for r in expand_runs(self._m(good), base=tmp)}
        self.assertNotIn("seed_spec", runs["HC"])
        self.assertEqual(runs["HC-S"]["seed_spec"], good)
        self.assertEqual(runs["HC-S"]["task_argv"], ["-H", "mux", "-k", "2"])


class TestLineage(unittest.TestCase):
    def test_lineage_of_a_seeded_run(self):
        """A short seeded MOSES run on mux6: every expansion starts from a tracked
        lineage and the best program's lineage is known."""
        import contextlib
        import io

        from pymoses.moses.main.moses_exec import moses_exec

        tracker = seed_lineage.install(seed_lineage.labels_for(2))
        self.addCleanup(seed_lineage.uninstall)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            moses_exec(
                [
                    "moses",
                    "-H",
                    "mux",
                    "-k",
                    "2",
                    "-m",
                    "300",
                    "-r",
                    "1",
                    "-l",
                    "WARN",
                    "-f",
                    str(Path(tmp) / "m.log"),
                    "-e",
                    "or(and($3 !$1) $4)",
                    "-e",
                    "and($1 $2 $6)",
                    "-e",
                    "and",
                ]
            )
            out = Path(tmp) / "lineage.json"
            tracker.write(out)
            rep = json.loads(out.read_text())
        self.assertEqual(set(rep["initial"]), {"seed1", "seed2", "empty"})
        self.assertTrue(rep["expansions_by_lineage"])
        self.assertNotIn("unknown", rep["expansions_by_lineage"])
        self.assertTrue(set(rep["best_lineages"]) <= {"seed1", "seed2", "empty"})
        self.assertTrue(rep["pool"])
        k = seed_lineage.key_of("or(and($3 !$1) $4)")
        self.assertIsInstance(k, str)
        seed_lineage.uninstall()
        from pymoses.moses.metapopulation.metapopulation import metapopulation

        self.assertNotEqual(metapopulation.init.__qualname__, "install.<locals>.init")


class _Stats:
    def __init__(self, n_evals=0):
        self.n_evals, self.n_expansions = n_evals, 0


class _Pa:
    max_evals = 10000


class _Mp:
    def __init__(self):
        self.score, self.injected = -10.0, []

    def best_score(self):
        return self.score

    def get_cscorer(self):
        return "scorer"

    def inject_candidates(self, trees, label):
        self.injected.append((label, len(trees)))
        return len(trees)


class _Dex:
    @staticmethod
    def simplify_candidate(tree):
        pass


def _fake_seeds(calls, used=120, combo="and($1 $2)"):
    def make_seeds(kind, binary, budget, height, seed, k, workdir, **kw):
        calls.append({"budget": budget, "seed": seed, "scorer": kw.get("scorer")})
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return {
            "evaluations_used": used,
            "gpgomea": {"cpu_s": 0.5},
            "seeds": [{"combo": combo, "pymoses_raw": -3.0}],
        }

    return make_seeds


class TestReseed(unittest.TestCase):
    def test_stagnation_budget_and_limit(self):
        calls: list = []
        orig = gs.make_seeds
        gs.make_seeds = _fake_seeds(calls)
        self.addCleanup(setattr, gs, "make_seeds", orig)
        with tempfile.TemporaryDirectory() as tmp:
            r = reseed.Reseeder(
                {"reseed_after": 2, "reseed_share": 0.25, "max_reseeds": 2},
                "gp",
                tmp,
                7,
                kind="bool",
                bool_flags=["-H", "mux", "-k", "2"],
            )
            mp, dex, pa, st = _Mp(), _Dex(), _Pa(), _Stats(2000)
            r(mp, dex, pa, st)  # first best seen
            r(mp, dex, pa, st)  # 1 without improvement
            self.assertEqual(calls, [])
            mp.score = -5.0
            r(mp, dex, pa, st)  # improved: counter reset
            r(mp, dex, pa, st)
            self.assertEqual(calls, [])
            r(mp, dex, pa, st)  # 2 without improvement: re-seed
            self.assertEqual(calls[0]["budget"], 2000)  # 0.25 * (10000 - 2000)
            self.assertEqual(calls[0]["seed"], 1007)
            self.assertEqual(calls[0]["scorer"][0], "scorer")
            self.assertEqual(st.n_evals, 2120)  # GP-GOMEA's evaluations count against -m
            self.assertEqual(mp.injected, [("reseed1", 1)])
            for _ in range(2):
                r(mp, dex, pa, st)
            self.assertEqual(calls[1]["seed"], 2007)
            for _ in range(6):  # max_reseeds reached
                r(mp, dex, pa, st)
            self.assertEqual(len(calls), 2)
            r.write(Path(tmp) / "reseed.json")
            rep = json.loads((Path(tmp) / "reseed.json").read_text())
        self.assertEqual(rep["n_reseeds"], 2)
        self.assertEqual(rep["reseed_evals"], 240)
        self.assertEqual(st.n_evals, 2240)

    def test_small_budget_is_skipped(self):
        calls: list = []
        orig = gs.make_seeds
        gs.make_seeds = _fake_seeds(calls)
        self.addCleanup(setattr, gs, "make_seeds", orig)
        with tempfile.TemporaryDirectory() as tmp:
            r = reseed.Reseeder({"reseed_after": 1}, "gp", tmp, 1, kind="bool")
            mp, st = _Mp(), _Stats(9900)
            r(mp, _Dex(), _Pa(), st)
            r(mp, _Dex(), _Pa(), st)
            self.assertEqual(calls, [])
            self.assertEqual(r.events[0]["status"], "skipped")
            self.assertEqual(r.report()["n_reseeds"], 0)
            self.assertEqual(st.n_evals, 9900)
            # a skip does not use up max_reseeds (1): with budget left, the next one runs
            st.n_evals = 2000
            r(mp, _Dex(), _Pa(), st)
            self.assertEqual(len(calls), 1)
            self.assertEqual(r.report()["n_reseeds"], 1)
            self.assertEqual(r.events[1]["reseed"], 1)

    def test_bad_spec(self):
        for spec in ({"reseed_after": 0}, {"reseed_after": 2, "reseed_share": 1.0}):
            with self.assertRaises(ValueError):
                reseed.Reseeder(spec, "gp", ".", 1)

    def test_hook_in_a_moses_run(self):
        """pa4: the installed hook re-seeds after one expansion without improvement,
        its programs join the metapopulation, and its evaluations are in n_evals."""
        import contextlib
        import io
        import re

        from pymoses.moses.main.moses_exec import moses_exec
        from pymoses.moses.moses import local_moses

        calls: list = []
        orig = gs.make_seeds
        prog = "or(and($1 !$2 $3) and($2 $4 !$3))"
        gs.make_seeds = _fake_seeds(calls, used=100, combo=prog)
        self.addCleanup(setattr, gs, "make_seeds", orig)
        tracker = seed_lineage.install(seed_lineage.labels_for(0))
        self.addCleanup(seed_lineage.uninstall)
        with tempfile.TemporaryDirectory() as tmp:
            spec = {"reseed_after": 1, "max_reseeds": 1, "reseed_share": 0.5}
            r = reseed.install({**spec, "reseed_min_budget": 10}, "gp", tmp, 3, kind="bool")
            self.addCleanup(reseed.uninstall)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                argv = ["-H", "pa", "-k", "4", "-m", "600", "-r", "1", "-l", "WARN", "-V1"]
                moses_exec(["moses", *argv, "-f", str(Path(tmp) / "m.log")])
        rep = r.report()
        self.assertEqual(rep["n_reseeds"], 1)
        ev = rep["events"][0]
        self.assertEqual(calls[0]["budget"], ev["evals_left"] // 2)
        self.assertEqual(ev["n_evals_after"], ev["n_evals_before"] + 100)
        n_evals = int(re.search(r"n_evals:\s*(\d+)", buf.getvalue()).group(1))
        self.assertGreaterEqual(n_evals, 600)
        # the injected program (reduced as MOSES reduces exemplars) has lineage reseed1
        self.assertEqual(list(tracker.lineage.values()).count("reseed1"), 1)
        reseed.uninstall()
        self.assertEqual(local_moses.expansion_hooks, [])


class TestRunnerSeedKeys(unittest.TestCase):
    def _job(self, binary, **spec):
        return {
            "binary": binary,
            "budget": {"evals": 2000},
            "seed": 1,
            "seed_spec": {"binary": "/b", "share": 0.2, **spec},
        }

    def test_reserve_flag_and_cpp_refusal(self):
        orig = gs.make_seeds

        def fake(*a, **kw):
            return {"evaluations_used": 400, "seeds": [{"combo": "$1"}], "gpgomea": {"cpu_s": 0}}

        gs.make_seeds = fake
        self.addCleanup(setattr, gs, "make_seeds", orig)
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job("/venv/bin/moses", reserve_empty_every=4)
            rec, argv = moses_sr_runner._apply_seed(job, ["moses", "-m", "2000"], Path(tmp))
            self.assertEqual(argv[:3], ["moses", "-m", "1600"])
            self.assertEqual(argv[-2:], ["--reserve-empty-every", "4"])
            self.assertIn("-e", argv)
            _, argv = moses_sr_runner._apply_seed(
                self._job("/venv/bin/moses"), ["moses", "-m", "2000"], Path(tmp)
            )
            self.assertNotIn("--reserve-empty-every", argv)
            for key in ({"reserve_empty_every": 2}, {"reseed_after": 3}):
                with self.assertRaises(ValueError):
                    moses_sr_runner._apply_seed(
                        self._job("/opt/moses-cpp/moses", **key),
                        ["moses", "-m", "2000"],
                        Path(tmp),
                    )


_BINARY = os.environ.get("GPGOMEA_SEED_BINARY")


@unittest.skipUnless(
    _BINARY and Path(_BINARY).exists(),
    "set GPGOMEA_SEED_BINARY to build_gpgomea_seed.sh's gpgomea-seed",
)
class TestEndToEnd(unittest.TestCase):
    def test_mux6(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = gs.make_seeds(
                "bool", _BINARY, 1000, 2, 7, 3, Path(tmp), bool_flags=["-H", "mux", "-k", "2"]
            )
        self.assertEqual(rec["n_score_matches"], rec["gpgomea"]["n_checked"])
        self.assertGreaterEqual(rec["evaluations_used"], 1000)
        self.assertTrue(1 <= len(rec["seeds"]) <= 3)
        for s in rec["seeds"]:
            self.assertLessEqual(s["gpgomea_nodes"], 7)
        self.assertEqual(rec["exemplar_argv"][-2:], ["-e", "and"])
        self.assertEqual(rec["exemplar_argv"][0], "-e")


if __name__ == "__main__":
    unittest.main()
