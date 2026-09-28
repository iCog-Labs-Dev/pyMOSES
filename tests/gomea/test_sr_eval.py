"""Unit tests for tools/gomea/sr_eval.py (MOSES program scoring on SR splits)."""

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
import sr_eval  # noqa: E402

# Header order puts the target in the middle to exercise MOSES's column mapping:
# $1 = a, $2 = b, $3 = c (target `y` removed).
_TRAIN = "a,y,b,c\n1,5,2,0\n2,9,1,1\n3,13,0,2\n4,17,3,3\n"
_TEST = "a,y,b,c\n0,1,1,1\n5,21,2,0\n"


class TestParsing(unittest.TestCase):
    def test_candidate_line_forms(self):
        p = sr_eval.parse_candidate_line
        self.assertEqual(p("-7988.1103515625 *(2 $3) "), (-7988.1103515625, "*(2 $3)"))
        self.assertEqual(p("-16666.3 0 "), (-16666.3, "0"))
        self.assertEqual(p("*(2 $3)"), (None, "*(2 $3)"))
        line = "-1.5 +($1 -2.5)  [score=-1.5, penalized score=-2, complexity=3]"
        self.assertEqual(p(line), (-1.5, "+($1 -2.5)"))
        self.assertIsNone(p("n_evals: 696"))
        self.assertIsNone(p(""))

    def test_moses_output(self):
        out = "-2 *(2 $3) \n-5 0 \nn_evals: 20000\n"
        self.assertEqual(sr_eval.parse_moses_output(out), [(-2.0, "*(2 $3)"), (-5.0, "0")])

    def test_relabel(self):
        self.assertEqual(
            sr_eval.relabel("+($x2 *($x10 $1))", ["x1", "x2", "x10"]), "+($2 *($3 $1))"
        )
        with self.assertRaises(ValueError):
            sr_eval.relabel("$nope", ["x1"])


class TestSemantics(unittest.TestCase):
    def test_moses_division(self):
        # 0/0 == 0 (numerator short-circuit); x/0 overflows -> nan, as MOSES's interpreter.
        X = [[0.0, 0.0], [1.0, 0.0], [3.0, 2.0]]
        p = sr_eval.predict("/($1 $2)", X)
        self.assertEqual(p[0], 0.0)
        self.assertTrue(math.isnan(p[1]))
        self.assertEqual(p[2], 1.5)

    def test_linear_scaling_and_nmse(self):
        y = [1.0, 3.0, 5.0, 7.0]
        p = [0.0, 1.0, 2.0, 3.0]
        a, b = sr_eval.linear_scaling(y, p)
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(b, 2.0)
        self.assertAlmostEqual(sr_eval.nmse(y, p, a, b), 0.0)
        # Constant prediction: b = 0, a = mean(y), NMSE = 100.
        a, b = sr_eval.linear_scaling(y, [2.0] * 4)
        self.assertEqual((a, b), (4.0, 0.0))
        self.assertAlmostEqual(sr_eval.nmse(y, [2.0] * 4, a, b), 100.0)
        self.assertEqual(sr_eval.nmse(y, [math.nan, 0, 0, 0]), math.inf)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.train, self.test = d / "train.csv", d / "test.csv"
        self.train.write_text(_TRAIN)
        self.test.write_text(_TEST)

    def tearDown(self):
        self.dir.cleanup()

    def test_column_mapping_and_scaling_fitted_on_train(self):
        tr = sr_eval.read_table(str(self.train))
        self.assertEqual(tr.labels, ["a", "b", "c"])
        self.assertEqual(tr.y, [5.0, 9.0, 13.0, 17.0])
        # y = 1 + 4a (train); program 2a -> scaling a=1, b=2; test row (0,..) -> 1, (5,..) -> 21.
        r = sr_eval.evaluate("*(2 $1)", tr, sr_eval.read_table(str(self.test)))
        self.assertAlmostEqual(r["a"], 1.0)
        self.assertAlmostEqual(r["b"], 2.0)
        self.assertAlmostEqual(r["train"]["nmse_scaled"], 0.0)
        self.assertAlmostEqual(r["test"]["nmse_scaled"], 0.0)
        self.assertGreater(r["train"]["nmse_raw"], 0.0)

    def test_cli_moses_output_with_labels(self):
        out = Path(self.dir.name) / "moses.out"
        out.write_text("-3.0 *(2 $a)  [score=-3]\n-9 0 \nn_evals: 100\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sr_eval.main(
                ["--train", str(self.train), "--test", str(self.test), "--moses-output", str(out)]
            )
        self.assertEqual(rc, 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(res["moses_score"], -3.0)
        self.assertAlmostEqual(res["test"]["nmse_scaled"], 0.0)

    def test_score_is_minus_sse(self):
        # MOSES's contin score for -H it is -sum((y - p)^2) over training rows.
        tr = sr_eval.read_table(str(self.train))
        p = sr_eval.predict("+(1 *(4 $1) $3)", tr.X)
        sse = sum((t - q) ** 2 for t, q in zip(tr.y, p, strict=True))
        self.assertAlmostEqual(sse, 0 + 1 + 4 + 9)


if __name__ == "__main__":
    unittest.main()
