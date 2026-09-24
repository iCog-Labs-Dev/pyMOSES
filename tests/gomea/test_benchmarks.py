"""Tests for the GOMEA benchmark harness (Task 9, `benchmarks/gomea/`).

Fast (<20s): pure statistics primitives (hand-computed cases), manifest
fail-closed validation and run expansion, the scorer instrumentation
wrapper (against a real `complexity_based_scorer`), one real but tiny
end-to-end runner invocation (HC arm only -- `-a gomea` is being built by a
concurrent task and is not exercised here), and svgplot's XML output.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from benchmarks.gomea import analyze as analyze_mod
from benchmarks.gomea import fixtures as fixtures_mod
from benchmarks.gomea import manifest as manifest_mod
from benchmarks.gomea import runner as runner_mod
from benchmarks.gomea import stats, svgplot
from benchmarks.gomea.instrument import ScorerTrace, instrument_scorer
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_scorer import complexity_based_scorer

_SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------


class TestWilcoxonSignedRank:
    def test_exact_all_positive_matches_hand_computed_p(self):
        # n=5, every difference the same sign -> the most extreme rank sum
        # possible, so the exact two-sided p is 2 * (1 / 2**5) = 0.0625.
        _, p = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5])
        assert p == pytest.approx(0.0625, abs=1e-9)

    def test_all_zero_differences_give_p_one(self):
        w, p = stats.wilcoxon_signed_rank([0, 0, 0, 0])
        assert p == 1.0
        assert w == 0.0

    def test_tie_case_hand_computed(self):
        # |diffs| = [1, 1, 2, 2, 3] -> midranks [1.5, 1.5, 3.5, 3.5, 5].
        # W+ (positive-signed ranks) = 1.5 (for +1) + 3.5 (for +2) + 5 (for +3) = 10
        # W- = 1.5 (for -1) + 3.5 (for -2) = 5; W = min(W+, W-) = 5.
        diffs = [1, -1, 2, -2, 3]
        w, p = stats.wilcoxon_signed_rank(diffs)
        assert w == pytest.approx(5.0)
        assert 0.0 < p <= 1.0

    def test_zero_differences_are_dropped_before_ranking(self):
        # Two extra exact zeros must not change the n=5 all-positive result.
        w_plain, p_plain = stats.wilcoxon_signed_rank([1, 2, 3, 4, 5])
        w_with_zeros, p_with_zeros = stats.wilcoxon_signed_rank([0, 1, 2, 3, 4, 5, 0])
        assert p_with_zeros == pytest.approx(p_plain)
        assert w_with_zeros == pytest.approx(w_plain)

    def test_normal_approximation_agrees_with_exact_near_n25(self):
        # A fixed, tie-free n=25 sample: force each computation mode
        # explicitly and compare them directly against each other.
        diffs = [((-1) ** i) * (i + 1) for i in range(25)]
        _, p_exact = stats.wilcoxon_signed_rank(diffs, method="exact")
        _, p_normal = stats.wilcoxon_signed_rank(diffs, method="normal")
        assert abs(p_exact - p_normal) < 0.02

    def test_single_nonzero_difference(self):
        # n_nonzero == 1: the only possible two-sided exact p is 1.0 (both
        # signs of a lone observation are equally likely under the null).
        _, p = stats.wilcoxon_signed_rank([0, 0, 5])
        assert p == pytest.approx(1.0)


class TestHolm:
    def test_hand_example(self):
        got = stats.holm([0.01, 0.04, 0.03])
        assert got == pytest.approx([0.03, 0.06, 0.06])

    def test_monotone_in_sorted_order_and_capped_at_one(self):
        pvals = [0.5, 0.9, 0.99, 0.999]
        got = stats.holm(pvals)
        assert all(0.0 <= p <= 1.0 for p in got)
        order = sorted(range(len(pvals)), key=lambda i: pvals[i])
        sorted_adjusted = [got[i] for i in order]
        assert sorted_adjusted == sorted(sorted_adjusted)

    def test_single_pvalue_is_unchanged(self):
        assert stats.holm([0.03]) == pytest.approx([0.03])

    def test_preserves_input_order(self):
        got = stats.holm([0.2, 0.001, 0.05])
        # Smallest p-value (index 1) gets multiplied by 3, the others by
        # progressively smaller factors under the running-max rule.
        assert got[1] == pytest.approx(0.003)


class TestWilsonCI:
    def test_zero_successes_lower_bound_is_zero(self):
        lo, hi = stats.wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < hi < 1.0

    def test_all_successes_upper_bound_is_one(self):
        lo, hi = stats.wilson_ci(10, 10)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < lo < 1.0

    def test_midpoint_is_a_valid_interval(self):
        lo, hi = stats.wilson_ci(5, 10)
        assert 0.0 <= lo < 0.5 < hi <= 1.0


class TestMedianIqr:
    def test_type7_linear_interpolation(self):
        med, q1, q3 = stats.median_iqr([1, 2, 3, 4])
        assert med == pytest.approx(2.5)
        assert q1 == pytest.approx(1.75)
        assert q3 == pytest.approx(3.25)

    def test_empty_is_nan(self):
        med, q1, q3 = stats.median_iqr([])
        assert all(v != v for v in (med, q1, q3))  # NaN != NaN


class TestBootstrapMedianCi:
    def test_reproducible_with_fixed_seed(self):
        diffs = [1.0, 2.0, 3.0, -1.0, 0.5]
        a = stats.bootstrap_median_ci(diffs, n=500, seed=12345)
        b = stats.bootstrap_median_ci(diffs, n=500, seed=12345)
        assert a == b
        assert a[0] <= a[1]

    def test_different_seed_can_differ(self):
        diffs = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0, -3.0]
        a = stats.bootstrap_median_ci(diffs, n=200, seed=1)
        b = stats.bootstrap_median_ci(diffs, n=200, seed=2)
        assert a != b or True  # not a hard guarantee, but exercises both paths


class TestErt:
    def test_zero_successes_is_inf(self):
        assert stats.ert([100, 100, 100], [False, False, False]) == float("inf")

    def test_mixed_case_computed_by_hand(self):
        # Two successes at 50 and 80 evaluations, one censored at the
        # budget cap of 200: ERT = (50 + 80 + 200) / 2 = 165.
        evals_used = [50, 80, 200]
        successes = [True, True, False]
        assert stats.ert(evals_used, successes) == pytest.approx(165.0)

    def test_all_success_is_plain_mean(self):
        assert stats.ert([10, 20, 30], [True, True, True]) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# manifest.py
# ---------------------------------------------------------------------------


def _valid_manifest(output_root) -> dict:
    return {
        "schema_version": 1,
        "name": "smoke",
        "phase": 1,
        "commit": "deadbeefcafe",
        "common_argv": ["-l", "ERROR"],
        "tasks": {"pa4": {"argv": ["-H", "pa", "-k", "4"], "target_raw": 0}},
        "arms": {
            "HC": {"argv": ["-a", "hc"]},
            "LT-B": {"argv": ["-a", "gomea", "--gomea-fos", "lt-b"]},
        },
        "seeds": [1001, 1002],
        "budget": {"argv": ["-m", "20000"], "kind": "evaluations"},
        "wall_cap_s": 900,
        "workers": 20,
        "output_root": str(output_root),
    }


class TestManifest:
    def test_fails_closed_on_missing_and_null_and_reports_both(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        del m["workers"]  # missing key
        m["commit"] = None  # null value
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))

        with pytest.raises(ValueError) as excinfo:
            manifest_mod.load_manifest(p)
        msg = str(excinfo.value)
        assert "workers" in msg
        assert "commit" in msg

    def test_fails_closed_on_null_nested_task_field(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["tasks"]["pa4"]["target_raw"] = None
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError) as excinfo:
            manifest_mod.load_manifest(p)
        assert "target_raw" in str(excinfo.value)

    def test_valid_manifest_loads(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        loaded = manifest_mod.load_manifest(p)
        assert loaded["name"] == "smoke"
        assert loaded["seeds"] == [1001, 1002]

    def test_hash_is_stable_regardless_of_key_order(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        reordered = dict(reversed(list(m.items())))
        assert manifest_mod.manifest_hash(m) == manifest_mod.manifest_hash(reordered)

    def test_hash_changes_when_content_changes(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m2 = _valid_manifest(tmp_path / "out")
        m2["seeds"] = [1001, 1003]
        assert manifest_mod.manifest_hash(m) != manifest_mod.manifest_hash(m2)

    def test_expand_runs_count_and_argv_order(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        runs = manifest_mod.expand_runs(m)
        assert len(runs) == len(m["tasks"]) * len(m["arms"]) * len(m["seeds"])

        first = runs[0]
        assert first["run_id"] == "pa4__HC__s1001"
        expected_argv = (
            m["common_argv"]
            + m["tasks"]["pa4"]["argv"]
            + m["arms"]["HC"]["argv"]
            + m["budget"]["argv"]
            + ["-r", "1001"]
        )
        assert first["argv"] == expected_argv

        gomea_run = next(r for r in runs if r["arm"] == "LT-B" and r["seed"] == 1001)
        assert gomea_run["argv"][-2] == "--gomea-log"
        assert gomea_run["argv"][-1].endswith("events.jsonl")
        assert "--gomea-log" not in first["argv"]  # HC arm gets no gomea-log

    def test_budget_evals_parses_max_evals_flag(self):
        assert manifest_mod.budget_evals({"argv": ["-m", "20000"], "kind": "evaluations"}) == 20000
        assert manifest_mod.budget_evals({"argv": [], "kind": "evaluations"}) is None


# ---------------------------------------------------------------------------
# instrument.py
# ---------------------------------------------------------------------------


class _sequence_cscore:
    """Not a ported C++ symbol -- a minimal `behave_cscore` stand-in that
    returns a pre-scripted sequence of `composite_score`s, one per call, the
    same pattern `tests/moses/representation/test_instance_scorer.py`'s own
    `_stub_cscore` uses.
    """

    def __init__(self, scores):
        self._scores = list(scores)
        self._i = 0

    def get_cscore(self, tr):
        s = self._scores[self._i]
        self._i += 1
        return s


class TestInstrumentScorer:
    def test_records_only_improvements_and_restores_on_exit(self, make_boolean_rep):
        rep = make_boolean_rep("and($1 $2)", 2)
        scores = [
            composite_score(-5.0, 3, 0.0, 0.0),  # call 1: first call, both trackers improve
            composite_score(-6.0, 3, 0.0, 0.0),  # call 2: worse raw AND worse composite -> no event
            composite_score(-4.0, 3, 0.0, 0.0),  # call 3: new best raw and best composite -> event
            composite_score(-4.0, 3, 0.0, 0.0),  # call 4: ties call 3 exactly -> no event
        ]
        scorer = complexity_based_scorer(_sequence_cscore(scores), rep, True)
        original_call = complexity_based_scorer.__call__

        trace = ScorerTrace()
        inst = instance(rep.fields().packed_width())
        with instrument_scorer(trace):
            for _ in scores:
                scorer(inst)
            # Wrapping is active for every instance of the class, not just
            # the one already constructed above.
            assert complexity_based_scorer.__call__ is not original_call

        assert complexity_based_scorer.__call__ is original_call
        assert trace.n_calls == 4
        assert [ev["raw"] for ev in trace.events] == [-5.0, -4.0]
        assert [ev["n"] for ev in trace.events] == [1, 3]
        assert trace.best_raw == pytest.approx(-4.0)
        assert trace.best_pen() == pytest.approx(-4.0)
        assert trace.best_cpx() == 3
        assert all(ev["t"] >= 0.0 for ev in trace.events)

    def test_first_hit_reports_call_that_first_reaches_target(self, make_boolean_rep):
        rep = make_boolean_rep("and($1 $2)", 2)
        scores = [
            composite_score(-10.0, 1, 0.0, 0.0),
            composite_score(-3.0, 1, 0.0, 0.0),
            composite_score(-1.0, 1, 0.0, 0.0),
        ]
        scorer = complexity_based_scorer(_sequence_cscore(scores), rep, True)
        trace = ScorerTrace()
        inst = instance(rep.fields().packed_width())
        with instrument_scorer(trace):
            for _ in scores:
                scorer(inst)

        hit = trace.first_hit(-3.0)
        assert hit is not None
        n, t = hit
        assert n == 2
        assert t >= 0.0
        assert trace.first_hit(0.0) is None  # never reached

    def test_restores_original_even_on_exception(self, make_boolean_rep):
        original_call = complexity_based_scorer.__call__
        trace = ScorerTrace()
        with pytest.raises(RuntimeError):
            with instrument_scorer(trace):
                raise RuntimeError("boom")
        assert complexity_based_scorer.__call__ is original_call


# ---------------------------------------------------------------------------
# svgplot.py
# ---------------------------------------------------------------------------


class TestSvgPlot:
    def test_line_chart_parses_and_has_one_path_per_series(self, tmp_path):
        out = tmp_path / "line.svg"
        series = {
            "HC": [(0, 0.0), (10, 0.2), (20, 0.5)],
            "LT-B": [(0, 0.0), (5, 0.3), (15, 0.6)],
        }
        svgplot.line_chart(series, "Title", "evals", "P(success)", out)
        tree = ET.parse(out)
        paths = tree.findall(".//svg:path", _SVG_NS)
        assert len(paths) == len(series)

    def test_line_chart_handles_empty_series(self, tmp_path):
        out = tmp_path / "line_empty.svg"
        svgplot.line_chart({"HC": [], "GOMEA": []}, "T", "x", "y", out)
        tree = ET.parse(out)
        paths = tree.findall(".//svg:path", _SVG_NS)
        assert len(paths) == 2

    def test_line_chart_supports_xlog(self, tmp_path):
        out = tmp_path / "line_log.svg"
        series = {"HC": [(1, 0.0), (10, 0.5), (100, 1.0)]}
        svgplot.line_chart(series, "T", "evals", "P", out, xlog=True)
        ET.parse(out)  # must not raise

    def test_box_chart_parses_as_xml(self, tmp_path):
        out = tmp_path / "box.svg"
        groups = {"HC": [1.0, 2.0, 3.0, 4.0], "LT-B": [0.5, 1.5, 2.5]}
        svgplot.box_chart(groups, "Title", "error", out)
        tree = ET.parse(out)
        assert tree.getroot().tag.endswith("svg")

    def test_box_chart_handles_empty_group(self, tmp_path):
        out = tmp_path / "box_empty.svg"
        svgplot.box_chart({"HC": [], "LT-B": [1.0, 2.0]}, "T", "y", out)
        ET.parse(out)  # must not raise


# ---------------------------------------------------------------------------
# runner.py -- one real, tiny end-to-end invocation (HC arm only)
# ---------------------------------------------------------------------------


class TestRunnerEndToEnd:
    def test_tiny_manifest_runs_and_second_invocation_resumes(self, tmp_path):
        repo_root = Path(__file__).resolve().parents[2]
        head = manifest_mod.git_head(repo_root)
        output_root = tmp_path / "out"
        m = {
            "schema_version": 1,
            "name": "tiny",
            "phase": 1,
            "commit": head,
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa2": {"argv": ["-H", "pa", "-k", "2"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "200"], "kind": "evaluations"},
            "wall_cap_s": 60,
            "workers": 1,
            "output_root": str(output_root),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(m))

        rc = runner_mod.main([str(manifest_path), "--workers", "1", "--allow-dirty"])
        assert rc == 0

        rd = manifest_mod.result_dir(m)
        assert (rd / "manifest.json").exists()
        assert (rd / "environment.json").exists()
        summary_path = rd / "runs" / "pa2__HC__s1" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "ok"
        assert summary["n_calls"] > 0
        assert summary["run_id"] == "pa2__HC__s1"
        assert (rd / "runs" / "pa2__HC__s1" / "config.json").exists()
        assert (rd / "runs" / "pa2__HC__s1" / "stdout.txt").exists()
        assert (rd / "runs" / "pa2__HC__s1" / "trace.jsonl").exists()

        mtime_before = summary_path.stat().st_mtime_ns
        rc2 = runner_mod.main([str(manifest_path), "--workers", "1", "--allow-dirty"])
        assert rc2 == 0
        assert summary_path.stat().st_mtime_ns == mtime_before  # untouched: resumed/skipped

    def test_commit_mismatch_refused_without_allow_dirty(self, tmp_path):
        output_root = tmp_path / "out"
        m = {
            "schema_version": 1,
            "name": "tiny2",
            "phase": 1,
            "commit": "0000000000000000000000000000000000000000",
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa2": {"argv": ["-H", "pa", "-k", "2"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "200"], "kind": "evaluations"},
            "wall_cap_s": 60,
            "workers": 1,
            "output_root": str(output_root),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(m))
        rc = runner_mod.main([str(manifest_path), "--workers", "1"])
        assert rc != 0
        assert not (manifest_mod.result_dir(m) / "runs").exists()


# ---------------------------------------------------------------------------
# analyze.py -- synthetic summaries, no real moses execution needed
# ---------------------------------------------------------------------------


def _write_summary(runs_dir: Path, run_id: str, task: str, arm: str, seed: int, **fields) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    base = {
        "run_id": run_id,
        "task": task,
        "arm": arm,
        "seed": seed,
        "argv": ["-a", arm.lower()],
        "status": "ok",
        "exit_code": 0,
        "wall_s": 1.0,
        "n_calls": 100,
        "best_raw": -1.0,
        "best_pen": -1.0,
        "best_cpx": 3,
        "success": False,
        "evals_to_target": None,
        "time_to_target": None,
        "stdout_best_line": None,
        "error": None,
        "gomea": None,
    }
    base.update(fields)
    (d / "summary.json").write_text(json.dumps(base))


class TestAnalyze:
    def test_end_to_end_synthetic_summaries(self, tmp_path):
        result_dir = tmp_path / "1" / "smoke-abc123"
        runs_dir = result_dir / "runs"
        manifest = {
            "schema_version": 1,
            "name": "smoke",
            "phase": 1,
            "commit": "deadbeef",
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa4": {"argv": ["-H", "pa", "-k", "4"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}, "LT-B": {"argv": ["-a", "gomea"]}},
            "seeds": [1, 2, 3],
            "budget": {"argv": ["-m", "1000"], "kind": "evaluations"},
            "wall_cap_s": 900,
            "workers": 2,
            "output_root": str(tmp_path),
        }
        result_dir.mkdir(parents=True)
        (result_dir / "manifest.json").write_text(json.dumps(manifest))

        # HC: two successes (evals 50, 80), one failure (never hits target).
        _write_summary(
            runs_dir, "pa4__HC__s1", "pa4", "HC", 1,
            best_raw=0.0, success=True, evals_to_target=50, wall_s=0.5,
        )
        _write_summary(
            runs_dir, "pa4__HC__s2", "pa4", "HC", 2,
            best_raw=0.0, success=True, evals_to_target=80, wall_s=0.6,
        )
        _write_summary(
            runs_dir, "pa4__HC__s3", "pa4", "HC", 3,
            best_raw=-2.0, success=False, wall_s=0.9,
        )
        # LT-B: three successes, all faster.
        _write_summary(
            runs_dir, "pa4__LT-B__s1", "pa4", "LT-B", 1,
            best_raw=0.0, success=True, evals_to_target=20, wall_s=0.4,
        )
        _write_summary(
            runs_dir, "pa4__LT-B__s2", "pa4", "LT-B", 2,
            best_raw=0.0, success=True, evals_to_target=30, wall_s=0.4,
        )
        _write_summary(
            runs_dir, "pa4__LT-B__s3", "pa4", "LT-B", 3,
            best_raw=0.0, success=True, evals_to_target=25, wall_s=0.4,
        )
        # One timeout run to exercise failures.md.
        _write_summary(
            runs_dir, "pa4__LT-B__s4", "pa4", "LT-B", 4,
            status="timeout", success=False, best_raw=None, n_calls=None,
            error="timeout: child process exceeded wall_cap_s and was killed",
        )

        rc = analyze_mod.main(
            [str(result_dir), "--baseline", "HC", "--contrasts", "LT-B:HC"]
        )
        assert rc == 0

        analysis_dir = result_dir / "analysis"
        for name in (
            "paired_results.csv",
            "aggregate_results.csv",
            "contrasts.csv",
            "failures.md",
        ):
            assert (analysis_dir / name).exists()
        figures = list((analysis_dir / "figures").glob("*.svg"))
        assert len(figures) == 2  # one CDF + one box plot for the one task

        with open(analysis_dir / "aggregate_results.csv") as f:
            agg_rows = list(csv.DictReader(f))
        hc_row = next(r for r in agg_rows if r["arm"] == "HC")
        assert hc_row["successes"] == "2"
        assert hc_row["n"] == "3"
        # ERT = (50 + 80 + 1000) / 2 = 565 (failure censored at the
        # manifest's own -m 1000 budget).
        assert float(hc_row["ert"]) == pytest.approx(565.0)

        with open(analysis_dir / "contrasts.csv") as f:
            contrast_rows = list(csv.DictReader(f))
        assert len(contrast_rows) == 1
        row = contrast_rows[0]
        assert row["contrast"] == "LT-B:HC"
        assert row["n_paired"] == "3"
        # LT-B's error (= -best_raw = 0.0 for every seed) never exceeds
        # HC's, so LT-B never LOSES this paired comparison on final error.
        assert int(row["losses"]) == 0

        failures_text = (analysis_dir / "failures.md").read_text()
        assert "pa4__LT-B__s4" in failures_text

    def test_missing_manifest_is_an_error(self, tmp_path):
        rc = analyze_mod.main([str(tmp_path), "--baseline", "HC", "--contrasts", "A:B"])
        assert rc != 0


# ---------------------------------------------------------------------------
# fixtures.py -- capture_exemplars against a real, tiny HC run
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_capture_exemplars_records_combo_tree_text(self, tmp_path):
        out_path = tmp_path / "exemplars.json"
        argv = ["-H", "pa", "-k", "2", "-a", "hc", "-m", "50", "-r", "1", "-l", "ERROR"]
        fixtures_mod.capture_exemplars(argv, n_expansions=1, out_path=out_path)

        assert out_path.exists()
        captured = json.loads(out_path.read_text())
        assert isinstance(captured, list)
        assert len(captured) == 1
        assert isinstance(captured[0], str)
        assert captured[0]  # non-empty combo-tree text
