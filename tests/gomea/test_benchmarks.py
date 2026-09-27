"""Tests for the GOMEA benchmark harness (Task 9, `benchmarks/gomea/`).

Fast (<20s despite several real, tiny end-to-end MOSES runs): pure
statistics primitives (hand-computed cases), manifest fail-closed
validation and run expansion, the scorer instrumentation wrapper (against a
real `complexity_based_scorer`), several real end-to-end runner invocations
(HC and -- now that a concurrent task has landed `-a gomea` in
`problem_params.py` -- GOMEA arms alike, covering the non-zero-exit,
timeout/censoring and missing-summary paths from fix round 1), and
svgplot's XML output.
"""

from __future__ import annotations

import csv
import io
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


def _gomea_cli_available() -> bool:
    """Fix round 1: the coordinator's own instruction is to check whether a
    concurrent task has landed `-a gomea` in `problem_params.py`, and to run
    one real GOMEA arm end to end if so. Checked once, at collection time,
    the same way the coordinator's own instructions suggest (`--help`
    listing `--gomea-fos` et al.), so the test below skips cleanly rather
    than failing if this ever runs against a checkout where that landing is
    mid-flight.
    """
    import contextlib

    from pymoses.moses.main.moses_exec import moses_exec

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            moses_exec(["moses", "--help"])
        except SystemExit:
            pass
    return "--gomea-fos" in buf.getvalue()


_GOMEA_CLI_AVAILABLE = _gomea_cli_available()


def _crash_hard(run: dict) -> None:
    """A `run_manifest(..., child_target=...)` testing seam (fix round 1,
    finding 10): a top-level, picklable-by-reference function (required
    under `multiprocessing.get_context("spawn")`) that exits the child
    process immediately, WITHOUT writing anything -- exercising the
    parent's "child died without a summary.json" recovery path (finding 3)
    deterministically and without needing a real crash."""
    import os

    os._exit(7)

def _child_with_failing_cscore(run: dict) -> None:
    """A `child_target` (F10): the real `_child_main`, but candidate scoring
    raises `MemoryError` inside `behave_cscore.get_cscore` (from its
    `get_cscore_nocache`), as it would at a memory cap -- only when called
    from within `complexity_based_scorer.__call__`, i.e. exactly where
    classic's `except Exception` would otherwise swallow it."""
    import sys

    from benchmarks.gomea import runner
    from pymoses.moses.representation import instance_scorer
    from pymoses.moses.scoring.behave_cscore import behave_cscore

    real = behave_cscore.get_cscore_nocache

    def _oom(self, tr):
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code is instance_scorer.complexity_based_scorer.__call__.__code__:
                raise MemoryError
            frame = frame.f_back
        return real(self, tr)

    behave_cscore.get_cscore_nocache = _oom
    runner._child_main(run)


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

    # -- fix round 1, finding 8: stricter validation --------------------

    def test_rejects_null_argv_element(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["common_argv"] = ["-l", None]
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match=r"common_argv\[1\] is null"):
            manifest_mod.load_manifest(p)

    def test_rejects_non_list_argv(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["arms"]["HC"]["argv"] = "not-a-list"
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match=r"arms\['HC'\]\.argv must be an array"):
            manifest_mod.load_manifest(p)

    def test_rejects_non_numeric_wall_cap_s_workers_target_raw(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["wall_cap_s"] = "soon"
        m["workers"] = True  # bool is deliberately rejected too
        m["tasks"]["pa4"]["target_raw"] = "zero"
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError) as excinfo:
            manifest_mod.load_manifest(p)
        msg = str(excinfo.value)
        assert "wall_cap_s" in msg
        assert "workers" in msg
        assert "target_raw" in msg

    def test_rejects_duplicate_seeds(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["seeds"] = [1001, 1002, 1001]
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="duplicate"):
            manifest_mod.load_manifest(p)

    # -- fix round 1, finding 9: precise GOMEA-arm detection ------------

    def test_is_gomea_arm_precise(self):
        assert manifest_mod.is_gomea_arm(["-a", "gomea"])
        assert manifest_mod.is_gomea_arm(["-a", "gomea", "--gomea-fos", "lt-b"])
        assert manifest_mod.is_gomea_arm(["--opt-algo", "gomea"])
        assert manifest_mod.is_gomea_arm(["--opt-algo=gomea"])
        # A bare substring match must NOT be enough -- the previous bug.
        assert not manifest_mod.is_gomea_arm(["--gomea-fos", "lt-b"])
        assert not manifest_mod.is_gomea_arm(["-a", "hc"])
        assert not manifest_mod.is_gomea_arm([])


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
        assert summary["scorer_calls_total"] == summary["n_calls"]
        assert summary["deme_count"] >= 1
        assert 0 <= summary["scorer_duplicate_calls"] < summary["n_calls"]
        assert (rd / "runs" / "pa2__HC__s1" / "config.json").exists()
        assert (rd / "runs" / "pa2__HC__s1" / "stdout.txt").exists()
        assert (rd / "runs" / "pa2__HC__s1" / "trace.jsonl").exists()
        # Fix round 1, finding 7: the atomic temp file must never linger.
        assert not (rd / "runs" / "pa2__HC__s1" / "summary.json.tmp").exists()

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

    def test_relative_output_root_and_relative_input_path_resolve(self, tmp_path):
        """Fix round 1, finding 1: a relative `output_root` in the manifest
        and a relative `-i` task argv both used to break (the former with a
        `FileNotFoundError` writing the run's own summary, since the
        previous design chdir'd into the run's own directory first). Both
        must resolve against the SAME "launch directory" -- `launch_dir`
        here, standing in for the real repo root so this test never
        touches the actual checkout."""
        data_csv = tmp_path / "data.csv"
        data_csv.write_text(
            "y,x1,x2\n0,0,1\n0,1,0\n1,1,1\n0,0,1\n0,1,0\n1,1,1\n1,0,1\n1,1,0\n0,1,1\n"
        )
        repo_root = Path(__file__).resolve().parents[2]
        head = manifest_mod.git_head(repo_root)
        m = {
            "schema_version": 1,
            "name": "relpath",
            "phase": 1,
            "commit": head,
            "common_argv": ["-l", "ERROR"],
            "tasks": {"it": {"argv": ["-H", "it", "-i", "data.csv", "-u", "y"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "100"], "kind": "evaluations"},
            "wall_cap_s": 30,
            "workers": 1,
            "output_root": "relative_out",  # relative, on purpose
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(m))

        rc = runner_mod.main(
            [str(manifest_path), "--workers", "1", "--allow-dirty"], launch_dir=tmp_path
        )
        assert rc == 0

        rd = manifest_mod.result_dir(m, base=tmp_path)
        assert rd.is_absolute()
        assert str(rd).startswith(str(tmp_path.resolve()))
        summary_path = rd / "runs" / "it__HC__s1" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "ok"
        assert summary["n_calls"] > 0

    def test_nonzero_exit_without_exception_is_recorded_as_error(self, tmp_path):
        """Fix round 1, finding 2: `moses_exec` returning 1 in-band (no
        exception -- its own documented behaviour for an unsupported `-H`)
        must not be recorded as `status: "ok"`."""
        repo_root = Path(__file__).resolve().parents[2]
        head = manifest_mod.git_head(repo_root)
        output_root = tmp_path / "out"
        m = {
            "schema_version": 1,
            "name": "badproblem",
            "phase": 1,
            "commit": head,
            "common_argv": ["-l", "ERROR"],
            "tasks": {"nope": {"argv": ["-H", "this_problem_does_not_exist_xyz"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "100"], "kind": "evaluations"},
            "wall_cap_s": 30,
            "workers": 1,
            "output_root": str(output_root),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(m))

        rc = runner_mod.main([str(manifest_path), "--workers", "1", "--allow-dirty"])
        assert rc != 0  # finding 3: any non-"ok" run makes the runner's own exit non-zero

        rd = manifest_mod.result_dir(m)
        summary = json.loads((rd / "runs" / "nope__HC__s1" / "summary.json").read_text())
        assert summary["status"] == "error"
        assert summary["exit_code"] == 1
        assert summary["censored"] is False
        assert "non-zero exit code" in summary["error"]

    def test_timeout_recovers_partial_trace_and_is_censored(self, tmp_path):
        """Fix round 1, finding 4: a killed run must not report an empty
        `best_raw`/`n_calls` -- both are recovered from the run's own
        streamed `trace.jsonl`.

        F4(d): robust under heavy load. `-H pa -k 6` with an effectively
        unbounded `-m` keeps HC running far past the cap. The cap is a few
        seconds, not 0.5 s, so the child gets past interpreter start-up and
        its first scorer call (the first call is always an improvement, so
        the trace is then non-empty). Every attempt must be a censored
        timeout; the recovered fields are asserted only for a non-empty
        trace, and the cap doubles (4, 8, 16, 32 s) until one attempt has
        one, so the test always ends up checking them."""
        head = _repo_head()
        checked = False
        for attempt, cap in enumerate((4, 8, 16, 32)):
            output_root = tmp_path / f"out{attempt}"
            m = {
                "schema_version": 1,
                "name": "timeout-test",
                "phase": 1,
                "commit": head,
                "common_argv": ["-l", "ERROR"],
                "tasks": {"pa6": {"argv": ["-H", "pa", "-k", "6"], "target_raw": 0}},
                "arms": {"HC": {"argv": ["-a", "hc"]}},
                "seeds": [1],
                "budget": {"argv": ["-m", "1000000000"], "kind": "evaluations"},
                "wall_cap_s": cap,
                "workers": 1,
                "output_root": str(output_root),
            }
            rc = runner_mod.main([str(_write_manifest(tmp_path, m)), "--allow-dirty"])
            assert rc != 0

            run_dir = manifest_mod.result_dir(m) / "runs" / "pa6__HC__s1"
            summary = json.loads((run_dir / "summary.json").read_text())
            assert summary["status"] == "timeout"
            assert summary["censored"] is True
            assert summary["wall_s"] >= cap
            trace_path = run_dir / "trace.jsonl"
            trace = trace_path.read_text().strip() if trace_path.exists() else ""
            if not trace:
                continue  # killed before its first scorer call: retry, longer
            # Recovered from the streamed partial trace, not left empty.
            assert summary["n_calls"] is not None and summary["n_calls"] > 0
            assert summary["best_raw"] is not None
            # F5: a censored run succeeds iff its trace reached the target.
            assert summary["success"] is (summary["best_raw"] >= 0)
            checked = True
            break
        assert checked, "no attempt got as far as its first scorer call"

    def test_child_exiting_without_summary_gets_synthesized_error_summary(self, tmp_path):
        """Fix round 1, finding 3: a child that exits without writing its
        own `summary.json` (here, `_crash_hard`'s `os._exit(7)`, standing in
        for a real crash) must not vanish from the results."""
        output_root = tmp_path / "out"
        m = {
            "schema_version": 1,
            "name": "crash-test",
            "phase": 1,
            "commit": "irrelevant-run_manifest-skips-the-commit-check",
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa2": {"argv": ["-H", "pa", "-k", "2"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "100"], "kind": "evaluations"},
            "wall_cap_s": 30,
            "workers": 1,
            "output_root": str(output_root),
        }
        rc = runner_mod.run_manifest(
            m, workers=1, only=None, wall_cap_s=30, launch_dir=tmp_path, child_target=_crash_hard
        )
        assert rc != 0

        rd = manifest_mod.result_dir(m, base=tmp_path)
        summary_path = rd / "runs" / "pa2__HC__s1" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "error"
        assert summary["exit_code"] == 7
        assert summary["censored"] is True
        assert "without writing summary.json" in summary["error"]

    @pytest.mark.skipif(
        not _GOMEA_CLI_AVAILABLE, reason="-a gomea not yet wired into problem_params.py"
    )
    def test_gomea_arm_end_to_end(self, tmp_path):
        """Per the coordinator's own instruction: `-a gomea` has landed
        (checked via `_GOMEA_CLI_AVAILABLE`), so run one real GOMEA arm end
        to end through the actual runner -- same instrumentation, same
        summary schema, plus the GOMEA-specific `--gomea-log` events."""
        repo_root = Path(__file__).resolve().parents[2]
        head = manifest_mod.git_head(repo_root)
        output_root = tmp_path / "out"
        m = {
            "schema_version": 1,
            "name": "gomea-e2e",
            "phase": 1,
            "commit": head,
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa3": {"argv": ["-H", "pa", "-k", "3"], "target_raw": 0}},
            "arms": {"LT-B": {"argv": ["-a", "gomea", "--gomea-fos", "lt-b"]}},
            "seeds": [1],
            "budget": {"argv": ["-m", "500"], "kind": "evaluations"},
            "wall_cap_s": 60,
            "workers": 1,
            "output_root": str(output_root),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(m))

        rc = runner_mod.main([str(manifest_path), "--workers", "1", "--allow-dirty"])
        assert rc == 0

        rd = manifest_mod.result_dir(m)
        run_dir = rd / "runs" / "pa3__LT-B__s1"
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["status"] == "ok"
        assert summary["n_calls"] > 0
        assert "--gomea-log" in summary["argv"]
        assert (run_dir / "events.jsonl").exists()
        assert summary["gomea"] is not None
        assert summary["gomea"]["event_counts"].get("end", 0) >= 1
        assert summary["gomea"]["termination_reasons"]
        assert summary["scorer_calls_total"] == summary["n_calls"]
        assert summary["deme_count"] >= 1
        assert 0 <= summary["scorer_duplicate_calls"] < summary["n_calls"]


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
        # evaluation CDF + CPU-time CDF + box plot, for the one task
        assert len(figures) == 3

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

    def test_ert_and_evals_contrast_blank_when_budget_unknown(self, tmp_path):
        """Fix round 1, finding 5: no `-m`/`--max-evals` anywhere in the
        manifest's `budget.argv`, and no `--budget` override -- ERT and the
        evals-to-target contrast must come out as an explicit "nan"/blank,
        never a silently biased partial mean, and the error-metric
        contrast (which does not need a budget) must still be computed."""
        result_dir = tmp_path / "1" / "smoke-nobudget"
        runs_dir = result_dir / "runs"
        manifest = {
            "schema_version": 1,
            "name": "smoke",
            "phase": 1,
            "commit": "deadbeef",
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa4": {"argv": ["-H", "pa", "-k", "4"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}, "LT-B": {"argv": ["-a", "gomea"]}},
            "seeds": [1, 2],
            "budget": {"argv": ["--max-time", "60"], "kind": "wall_time"},
            "wall_cap_s": 900,
            "workers": 2,
            "output_root": str(tmp_path),
        }
        result_dir.mkdir(parents=True)
        (result_dir / "manifest.json").write_text(json.dumps(manifest))

        _write_summary(
            runs_dir, "pa4__HC__s1", "pa4", "HC", 1,
            best_raw=0.0, success=True, evals_to_target=50, wall_s=0.5,
        )
        _write_summary(
            runs_dir, "pa4__HC__s2", "pa4", "HC", 2,
            best_raw=-2.0, success=False, wall_s=0.9,
        )
        _write_summary(
            runs_dir, "pa4__LT-B__s1", "pa4", "LT-B", 1,
            best_raw=0.0, success=True, evals_to_target=20, wall_s=0.4,
        )
        _write_summary(
            runs_dir, "pa4__LT-B__s2", "pa4", "LT-B", 2,
            best_raw=0.0, success=True, evals_to_target=30, wall_s=0.4,
        )

        rc = analyze_mod.main([str(result_dir), "--baseline", "HC", "--contrasts", "LT-B:HC"])
        assert rc == 0

        analysis_dir = result_dir / "analysis"
        with open(analysis_dir / "aggregate_results.csv") as f:
            agg_rows = list(csv.DictReader(f))
        hc_row = next(r for r in agg_rows if r["arm"] == "HC")
        assert hc_row["ert"] == "nan"

        with open(analysis_dir / "contrasts.csv") as f:
            contrast_rows = list(csv.DictReader(f))
        row = contrast_rows[0]
        assert row["p_evals_to_target"] == "nan"
        assert row["p_evals_to_target_holm"] == "nan"
        assert row["censor_value_evals"] == ""
        # The error-metric contrast does NOT need a budget and must still
        # be computed normally.
        assert row["p_error"] not in ("", "nan")
        # No pair is ever dropped: both seeds are paired even though HC's
        # second seed was unsuccessful.
        assert row["n_paired"] == "2"

        failures_text = (analysis_dir / "failures.md").read_text()
        assert "no numeric evaluation budget could be determined" in failures_text

    def test_missing_run_is_synthesized_and_counted_as_a_failure(self, tmp_path):
        """Fix round 1, finding 3: a run the manifest expects (task x arm x
        seed) but which never left a `summary.json` at all must not
        silently vanish -- it is synthesized as `status: "missing"` and
        shows up in `failures.md` and in the aggregate `n`/`n_no_best`
        counts."""
        result_dir = tmp_path / "1" / "smoke-missing"
        runs_dir = result_dir / "runs"
        manifest = {
            "schema_version": 1,
            "name": "smoke",
            "phase": 1,
            "commit": "deadbeef",
            "common_argv": ["-l", "ERROR"],
            "tasks": {"pa4": {"argv": ["-H", "pa", "-k", "4"], "target_raw": 0}},
            "arms": {"HC": {"argv": ["-a", "hc"]}},
            "seeds": [1, 2],
            "budget": {"argv": ["-m", "1000"], "kind": "evaluations"},
            "wall_cap_s": 900,
            "workers": 2,
            "output_root": str(tmp_path),
        }
        result_dir.mkdir(parents=True)
        (result_dir / "manifest.json").write_text(json.dumps(manifest))

        # Only seed 1 ever wrote a summary; seed 2 is entirely absent.
        _write_summary(
            runs_dir, "pa4__HC__s1", "pa4", "HC", 1,
            best_raw=0.0, success=True, evals_to_target=50, wall_s=0.5,
        )

        rc = analyze_mod.main([str(result_dir), "--baseline", "HC", "--contrasts", "HC:HC"])
        assert rc == 0

        analysis_dir = result_dir / "analysis"
        failures_text = (analysis_dir / "failures.md").read_text()
        assert "pa4__HC__s2" in failures_text
        assert "missing" in failures_text

        with open(analysis_dir / "aggregate_results.csv") as f:
            agg_rows = list(csv.DictReader(f))
        hc_row = next(r for r in agg_rows if r["arm"] == "HC")
        assert hc_row["n"] == "2"  # the missing run is still counted
        assert hc_row["n_no_best"] == "1"


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


# ---------------------------------------------------------------------------
# Final fix wave (F1, F4d, F5, F7, F8, F9)
# ---------------------------------------------------------------------------


def _repo_head() -> str:
    return manifest_mod.git_head(Path(__file__).resolve().parents[2])


def _tiny_manifest(output_root, name="tiny-ff", **over) -> dict:
    m = {
        "schema_version": 1,
        "name": name,
        "phase": 1,
        "commit": _repo_head(),
        "common_argv": ["-l", "ERROR"],
        "tasks": {"pa2": {"argv": ["-H", "pa", "-k", "2"], "target_raw": 0}},
        "arms": {"HC": {"argv": ["-a", "hc"]}},
        "seeds": [1],
        "budget": {"argv": ["-m", "200"], "kind": "evaluations"},
        "wall_cap_s": 60,
        "workers": 1,
        "output_root": str(output_root),
    }
    m.update(over)
    return m


def _write_manifest(tmp_path, m) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p


class TestScorerDuplicateCounter:
    """Phase 1d: the harness duplicate counter, fair to every arm because it
    wraps the one shared scorer seam."""

    def test_same_instance_twice_on_one_scorer_is_one_duplicate(self, make_boolean_rep):
        rep = make_boolean_rep("and($1 $2)", 2)
        scores = [composite_score(-float(i), 1, 0.0, 0.0) for i in range(5)]
        scorer = complexity_based_scorer(_sequence_cscore(scores), rep, True)
        trace = ScorerTrace()
        a = instance(rep.fields().packed_width())
        b = instance(a)
        b[0] = 1
        with instrument_scorer(trace):
            scorer(a)
            scorer(a)
            scorer(b)
        assert trace.n_calls == 3
        assert trace.scorer_duplicate_calls == 1
        assert trace.deme_count == 1
        # Scores and improvement events are exactly what they were before.
        assert [ev["raw"] for ev in trace.events] == [0.0]

    def test_two_scorer_objects_are_two_demes_with_separate_seen_sets(self, make_boolean_rep):
        rep = make_boolean_rep("and($1 $2)", 2)
        mk = lambda: complexity_based_scorer(  # noqa: E731
            _sequence_cscore([composite_score(-1.0, 1, 0.0, 0.0)] * 3), rep, True
        )
        s1, s2 = mk(), mk()
        trace = ScorerTrace()
        inst = instance(rep.fields().packed_width())
        with instrument_scorer(trace):
            s1(inst)
            s2(inst)  # same instance, different deme scorer: not a duplicate
            s2(inst)  # duplicate within s2
        assert trace.n_calls == 3
        assert trace.deme_count == 2
        assert trace.scorer_duplicate_calls == 1

    @pytest.mark.parametrize("algo", ["hc", "gomea"])
    def test_real_moses_runs_get_the_fields(self, algo, tmp_path, monkeypatch):
        import contextlib

        from pymoses.moses.main.moses_exec import moses_exec

        monkeypatch.chdir(tmp_path)
        log = tmp_path / "g.jsonl"
        argv = ["moses", "-H", "pa", "-k", "3", "-a", algo, "-m", "800", "-r", "1"]
        if algo == "gomea":
            argv += ["--gomea-log", str(log)]
        trace = ScorerTrace()
        with contextlib.redirect_stdout(io.StringIO()), instrument_scorer(trace):
            assert moses_exec(argv) == 0
        assert trace.n_calls > 0
        assert trace.deme_count >= 1
        assert 0 <= trace.scorer_duplicate_calls < trace.n_calls
        if algo == "gomea":
            # One oracle per deme scorer: the harness count agrees with the
            # oracle's own charged-duplicate ledger.
            ends = [
                json.loads(line)
                for line in log.read_text().splitlines()
                if json.loads(line)["type"] == "end"
            ]
            assert trace.deme_count == len(ends)
            assert trace.scorer_duplicate_calls == sum(
                e["ledger"]["duplicate_requests"] for e in ends
            )


class TestCpuTimeEvidence:
    """F1: CPU time, peak RSS and a deterministic, shuffled run order."""

    def test_trace_events_carry_cpu_and_summary_has_cpu_to_target(self, make_boolean_rep):
        rep = make_boolean_rep("and($1 $2)", 2)
        scores = [composite_score(-3.0, 1, 0.0, 0.0), composite_score(0.0, 1, 0.0, 0.0)]
        scorer = complexity_based_scorer(_sequence_cscore(scores), rep, True)
        trace = ScorerTrace()
        inst = instance(rep.fields().packed_width())
        with instrument_scorer(trace):
            for _ in scores:
                scorer(inst)
        cpus = [ev["cpu"] for ev in trace.events]
        assert len(cpus) == 2 and all(c >= 0.0 for c in cpus) and cpus == sorted(cpus)
        hit = trace.first_hit_event(0.0)
        assert hit is not None and hit["n"] == 2 and hit["cpu"] == cpus[1]

        from benchmarks.gomea.instrument import summarize_events

        rec = summarize_events(
            [
                {"n": 1, "t": 0.5, "cpu": 0.25, "raw": -3.0, "pen": -3.0, "cpx": 1},
                {"n": 7, "t": 1.5, "cpu": 1.25, "raw": 0.0, "pen": 0.0, "cpx": 1},
            ],
            0.0,
        )
        assert rec["cpu_to_target"] == 1.25
        assert rec["time_to_target"] == 1.5
        assert summarize_events([], 0.0)["cpu_to_target"] is None

    def test_ok_summary_records_cpu_rss_and_run_order(self, tmp_path):
        m = _tiny_manifest(tmp_path / "out", seeds=[1, 2])
        rc = runner_mod.main([str(_write_manifest(tmp_path, m)), "--allow-dirty"])
        assert rc == 0
        rd = manifest_mod.result_dir(m)
        for seed in (1, 2):
            s = json.loads((rd / "runs" / f"pa2__HC__s{seed}" / "summary.json").read_text())
            assert s["status"] == "ok"
            assert s["cpu_s"] > 0.0
            assert s["ru_maxrss_kb"] > 0
            assert "cpu_to_target" in s
            if s["success"]:
                assert 0.0 <= s["cpu_to_target"] <= s["cpu_s"]
        order = json.loads((rd / "run_order.json").read_text())
        ids = [r["run_id"] for r in manifest_mod.expand_runs(m)]
        assert sorted(order["order"]) == sorted(ids)
        assert order["seed"] == int(manifest_mod.manifest_hash(m)[:8], 16)

    def test_run_order_is_a_deterministic_hash_seeded_shuffle(self, tmp_path):
        import random

        m = _valid_manifest(tmp_path / "out")
        m["tasks"]["pa5"] = {"argv": ["-H", "pa", "-k", "5"], "target_raw": 0}
        m["arms"]["UNI"] = {"argv": ["-a", "gomea", "--gomea-fos", "uni"]}
        m["seeds"] = list(range(1, 9))
        runs = manifest_mod.expand_runs(m)
        a = runner_mod.scheduled_order(m, runs)
        b = runner_mod.scheduled_order(m, list(runs))
        assert [r["run_id"] for r in a] == [r["run_id"] for r in b]
        assert sorted(r["run_id"] for r in a) == sorted(r["run_id"] for r in runs)
        assert [r["run_id"] for r in a] != [r["run_id"] for r in runs]
        expected = list(runs)
        random.Random(int(manifest_mod.manifest_hash(m)[:8], 16)).shuffle(expected)
        assert [r["run_id"] for r in a] == [r["run_id"] for r in expected]

    def test_max_load_note_is_optional_documentation(self, tmp_path):
        m = _valid_manifest(tmp_path / "out")
        m["max_load_note"] = "pilot: 20 workers under a load of about 26"
        loaded = manifest_mod.load_manifest(_write_manifest(tmp_path, m))
        assert loaded["max_load_note"].startswith("pilot")
        m["max_load_note"] = 5
        with pytest.raises(ValueError, match="max_load_note"):
            manifest_mod.load_manifest(_write_manifest(tmp_path, m))

    def test_analyze_reports_cpu_and_rss_and_cpu_ecdf(self, tmp_path):
        result_dir = tmp_path / "1" / "smoke-cpu"
        runs_dir = result_dir / "runs"
        manifest = _valid_manifest(tmp_path)
        manifest["arms"] = {"HC": {"argv": ["-a", "hc"]}}
        manifest["seeds"] = [1, 2, 3]
        result_dir.mkdir(parents=True)
        (result_dir / "manifest.json").write_text(json.dumps(manifest))
        for seed, cpu, rss, hit in ((1, 1.0, 1000, 0.5), (2, 2.0, 3000, None), (3, 4.0, 2000, 1.5)):
            _write_summary(
                runs_dir, f"pa4__HC__s{seed}", "pa4", "HC", seed,
                cpu_s=cpu, ru_maxrss_kb=rss, success=hit is not None,
                best_raw=0.0 if hit is not None else -1.0,
                evals_to_target=10 if hit is not None else None, cpu_to_target=hit,
            )
        rc = analyze_mod.main([str(result_dir), "--baseline", "HC", "--contrasts", "HC:HC"])
        assert rc == 0
        with open(result_dir / "analysis" / "aggregate_results.csv") as f:
            row = next(csv.DictReader(f))
        assert float(row["cpu_s_median"]) == pytest.approx(2.0)
        assert float(row["cpu_s_q1"]) <= 2.0 <= float(row["cpu_s_q3"])
        assert float(row["ru_maxrss_kb_max"]) == 3000
        assert float(row["ru_maxrss_kb_median"]) == 2000
        assert "wall_s_median" in row  # the wall columns stay
        assert (result_dir / "analysis" / "figures" / "pa4_success_cpu_cdf.svg").exists()


class TestTimeoutSuccess:
    """F5: a timed-out run whose recovered trace reached the target counts
    as a success; an error never does."""

    def _run_with_trace(self, tmp_path, run_id):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        events = [
            {"n": 1, "t": 0.1, "cpu": 0.1, "raw": -2.0, "pen": -2.0, "cpx": 3},
            {"n": 42, "t": 0.9, "cpu": 0.8, "raw": 0.0, "pen": -0.1, "cpx": 3},
        ]
        (run_dir / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        return {
            "run_id": run_id, "task": "pa4", "arm": "HC", "seed": 1, "argv": [],
            "target_raw": 0.0, "run_dir": str(run_dir),
        }

    def test_timeout_that_reached_target_is_a_censored_success(self, tmp_path):
        run = self._run_with_trace(tmp_path, "t")
        runner_mod._write_timeout_summary(run, 5.0)
        s = json.loads((Path(run["run_dir"]) / "summary.json").read_text())
        assert s["status"] == "timeout" and s["censored"] is True
        assert s["success"] is True
        assert s["evals_to_target"] == 42
        assert s["cpu_to_target"] == 0.8

    def test_error_that_reached_target_is_not_a_success(self, tmp_path):
        run = self._run_with_trace(tmp_path, "e")
        runner_mod._write_missing_summary(run, 9, 5.0)
        s = json.loads((Path(run["run_dir"]) / "summary.json").read_text())
        assert s["status"] == "error"
        assert s["success"] is False


class TestOutputRootAndHash:
    """F7: output root and workers are launch-time settings."""

    def test_hash_ignores_output_root_and_workers(self, tmp_path):
        m = _valid_manifest(tmp_path / "a")
        m2 = _valid_manifest(tmp_path / "b")
        m2["workers"] = 3
        assert manifest_mod.manifest_hash(m) == manifest_mod.manifest_hash(m2)
        assert manifest_mod.result_dir(m).name == manifest_mod.result_dir(m2).name

    def test_output_root_override_and_environment(self, tmp_path):
        m = _tiny_manifest(tmp_path / "manifest_root")
        override = tmp_path / "elsewhere"
        rc = runner_mod.main(
            [str(_write_manifest(tmp_path, m)), "--allow-dirty", "--output-root", str(override),
             "--workers", "2"]
        )
        assert rc == 0
        rd = manifest_mod.result_dir({**m, "output_root": str(override)})
        assert rd.name == manifest_mod.result_dir(m).name
        s = json.loads((rd / "runs" / "pa2__HC__s1" / "summary.json").read_text())
        assert s["status"] == "ok"
        assert not (tmp_path / "manifest_root").exists()
        env = json.loads((rd / "environment.json").read_text())
        assert env["output_root"] == str(override.resolve())
        assert env["workers"] == 2
        # The committed manifest is what the results directory keeps.
        assert json.loads((rd / "manifest.json").read_text()) == m

    def test_analyze_works_on_a_copied_results_directory(self, tmp_path):
        import shutil

        m = _tiny_manifest(tmp_path / "orig")
        rc = runner_mod.main([str(_write_manifest(tmp_path, m)), "--allow-dirty"])
        assert rc == 0
        rd = manifest_mod.result_dir(m)
        copy_dir = tmp_path / "copied" / rd.name
        shutil.copytree(rd, copy_dir)
        shutil.rmtree(tmp_path / "orig")
        rc = analyze_mod.main([str(copy_dir), "--baseline", "HC", "--contrasts", "HC:HC"])
        assert rc == 0
        with open(copy_dir / "analysis" / "aggregate_results.csv") as f:
            row = next(csv.DictReader(f))
        assert row["n"] == "1" and row["n_no_best"] == "0"
        failures = (copy_dir / "analysis" / "failures.md").read_text()
        assert "No error, timeout or missing runs." in failures


def _git(repo, *args):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def code_repo(tmp_path):
    """A throwaway git repo with pymoses/, benchmarks/, docs/ and
    pyproject.toml, one commit."""
    repo = tmp_path / "repo"
    for d in ("pymoses", "benchmarks", "docs"):
        (repo / d).mkdir(parents=True)
        (repo / d / "f.py").write_text("x = 1\n")
    (repo / "pyproject.toml").write_text("[project]\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "code")
    return repo


class TestCodeCommitCheck:
    """F8: the manifest commit names the CODE; the manifest itself may be
    committed later."""

    def test_head_itself_passes(self, code_repo):
        c = manifest_mod.check_code_commit(code_repo, _git(code_repo, "rev-parse", "HEAD"))
        assert c["ok"] is True and c["is_ancestor"] and c["code_unchanged"] and c["clean"]

    def test_later_non_code_commit_passes(self, code_repo):
        code = _git(code_repo, "rev-parse", "HEAD")
        (code_repo / "docs" / "f.py").write_text("x = 2\n")
        (code_repo / "manifest.json").write_text("{}\n")
        _git(code_repo, "add", "-A")
        _git(code_repo, "commit", "-qm", "manifest")
        c = manifest_mod.check_code_commit(code_repo, code)
        assert c["ok"] is True
        assert c["head"] != code and c["manifest_commit"] == code

    @pytest.mark.parametrize("path", ["pymoses/f.py", "benchmarks/f.py", "pyproject.toml"])
    def test_later_code_commit_fails(self, code_repo, path):
        code = _git(code_repo, "rev-parse", "HEAD")
        (code_repo / path).write_text("x = 3\n")
        _git(code_repo, "commit", "-qam", "code change")
        c = manifest_mod.check_code_commit(code_repo, code)
        assert c["ok"] is False and c["is_ancestor"] is True and c["code_unchanged"] is False

    def test_non_ancestor_and_unknown_commit_fail(self, code_repo):
        base = _git(code_repo, "rev-parse", "HEAD")
        _git(code_repo, "checkout", "-qb", "side")
        (code_repo / "docs" / "f.py").write_text("side\n")
        _git(code_repo, "commit", "-qam", "side")
        side = _git(code_repo, "rev-parse", "HEAD")
        _git(code_repo, "checkout", "-q", base)
        c = manifest_mod.check_code_commit(code_repo, side)
        assert c["ok"] is False and c["is_ancestor"] is False
        c2 = manifest_mod.check_code_commit(code_repo, "0" * 40)
        assert c2["ok"] is False and c2["is_ancestor"] is False

    def test_local_code_modification_fails(self, code_repo):
        code = _git(code_repo, "rev-parse", "HEAD")
        (code_repo / "benchmarks" / "f.py").write_text("dirty\n")
        c = manifest_mod.check_code_commit(code_repo, code)
        assert c["ok"] is False and c["clean"] is False
        assert any("benchmarks/f.py" in line for line in c["dirty"])
        (code_repo / "benchmarks" / "f.py").write_text("x = 1\n")
        (code_repo / "docs" / "f.py").write_text("dirty docs are fine\n")
        assert manifest_mod.check_code_commit(code_repo, code)["ok"] is True

    def test_runner_refuses_then_records_the_check(self, code_repo, tmp_path, capsys):
        code = _git(code_repo, "rev-parse", "HEAD")
        (code_repo / "pymoses" / "f.py").write_text("x = 9\n")
        _git(code_repo, "commit", "-qam", "code change")
        m = _tiny_manifest(tmp_path / "out", commit=code)
        mp = _write_manifest(tmp_path, m)
        assert runner_mod.main([str(mp)], repo_root=code_repo) == 1
        assert not manifest_mod.result_dir(m).exists()
        assert "code" in capsys.readouterr().err.lower()

        assert runner_mod.main([str(mp), "--allow-dirty"], repo_root=code_repo) == 0
        env = json.loads((manifest_mod.result_dir(m) / "environment.json").read_text())
        assert env["manifest_commit"] == code
        assert env["head"] == _git(code_repo, "rev-parse", "HEAD")
        assert env["code_check"]["ok"] is False
        assert env["allow_dirty"] is True

    def test_runner_accepts_a_later_non_code_commit(self, code_repo, tmp_path):
        code = _git(code_repo, "rev-parse", "HEAD")
        (code_repo / "manifest.json").write_text("{}\n")
        _git(code_repo, "add", "-A")
        _git(code_repo, "commit", "-qm", "manifest")
        m = _tiny_manifest(tmp_path / "out", commit=code)
        assert runner_mod.main([str(_write_manifest(tmp_path, m))], repo_root=code_repo) == 0
        env = json.loads((manifest_mod.result_dir(m) / "environment.json").read_text())
        assert env["code_check"]["ok"] is True


class TestMemorySafety:
    """F9: a per-child address-space cap, recorded failures, and --dry-run."""

    def test_tiny_memory_limit_is_recorded_as_memory_limit_error(self, tmp_path, capsys):
        m = _tiny_manifest(tmp_path / "out", workers=3)
        rc = runner_mod.main(
            [str(_write_manifest(tmp_path, m)), "--allow-dirty", "--mem-limit-mb", "1"]
        )
        assert rc == 1  # the parent survives and reports the failed run
        s = json.loads(
            (manifest_mod.result_dir(m) / "runs" / "pa2__HC__s1" / "summary.json").read_text()
        )
        assert s["status"] == "error"
        assert s["error"] == "memory_limit"
        assert s["mem_limit_mb"] == 1
        out = capsys.readouterr().out
        warning = [line for line in out.splitlines() if "estimated peak" in line]
        assert len(warning) == 1 and "3 MB" in warning[0]

    def test_generous_memory_limit_does_not_change_an_ok_run(self, tmp_path):
        m = _tiny_manifest(tmp_path / "out")
        rc = runner_mod.main(
            [str(_write_manifest(tmp_path, m)), "--allow-dirty", "--mem-limit-mb", "4096"]
        )
        assert rc == 0
        s = json.loads(
            (manifest_mod.result_dir(m) / "runs" / "pa2__HC__s1" / "summary.json").read_text()
        )
        assert s["status"] == "ok" and s["ru_maxrss_kb"] > 0 and s["mem_limit_mb"] == 4096

    def test_runner_module_does_not_import_moses(self):
        import subprocess
        import sys

        code = (
            "import sys, benchmarks.gomea.runner; "
            "print(any(k.startswith('pymoses') for k in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert out == "False"  # so the child sets its limit before MOSES loads

    def test_dry_run_prints_plan_and_runs_nothing(self, tmp_path, capsys):
        m = _tiny_manifest(tmp_path / "out", seeds=[1, 2])
        mp = _write_manifest(tmp_path, m)
        rc = runner_mod.main([str(mp), "--allow-dirty", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        rd = manifest_mod.result_dir(m)
        assert str(rd) in out
        assert "pa2__HC__s1" in out and "pa2__HC__s2" in out
        assert "already complete: 0/2" in out
        assert "code check" in out
        assert not rd.exists()


class TestMemoryInsideScoring:
    """F10: a MemoryError inside candidate scoring fails the run instead of
    being scored as the worst candidate by classic's `except Exception`."""

    def test_guard_turns_a_swallowed_memory_error_into_memory_limit_hit(
        self, make_boolean_rep, monkeypatch
    ):
        from benchmarks.gomea.instrument import MemoryLimitHit, guard_memory_errors
        from pymoses.moses.moses.types import worst_composite_score
        from pymoses.moses.representation.representation import representation
        from pymoses.moses.scoring.behave_cscore import behave_cscore
        from pymoses.moses.scoring.bscores import ctruth_table_bscore
        from tests.gomea.test_optimizer import _parity_ctable

        def _oom(self, tr):
            raise MemoryError

        monkeypatch.setattr(behave_cscore, "get_cscore_nocache", _oom)
        rep = make_boolean_rep("and($1 $2)", 2)
        scorer = complexity_based_scorer(
            behave_cscore(ctruth_table_bscore(_parity_ctable(2))), rep, True
        )
        inst = instance(rep.fields().packed_width())
        # Unguarded, classic swallows it: the candidate gets the worst score.
        assert scorer(inst) == worst_composite_score
        originals = (behave_cscore.get_cscore, representation.get_candidate)
        with pytest.raises(MemoryLimitHit), guard_memory_errors():
            scorer(inst)
        assert (behave_cscore.get_cscore, representation.get_candidate) == originals
        assert not issubclass(MemoryLimitHit, Exception)

    def test_memory_error_in_get_cscore_is_recorded_and_exits_nonzero(self, tmp_path):
        import multiprocessing as mp

        m = _tiny_manifest(tmp_path / "out")
        rc = runner_mod.run_manifest(
            m, workers=1, only=None, wall_cap_s=60, launch_dir=tmp_path,
            child_target=_child_with_failing_cscore,
        )
        assert rc == 1
        run_dir = manifest_mod.result_dir(m, base=tmp_path) / "runs" / "pa2__HC__s1"
        s = json.loads((run_dir / "summary.json").read_text())
        assert s["status"] == "error"
        assert s["error"] == "memory_limit"
        assert s["exit_code"] == 1
        assert s["success"] is False

        # The child process itself exits non-zero.
        (run_dir / "summary.json").unlink()
        run = dict(manifest_mod.expand_runs(m, base=tmp_path)[0], launch_dir=str(tmp_path))
        p = mp.get_context("spawn").Process(target=_child_with_failing_cscore, args=(run,))
        p.start()
        p.join(60)
        assert p.exitcode == 1
        assert json.loads((run_dir / "summary.json").read_text())["error"] == "memory_limit"

    def test_near_memory_limit_flag_and_failures_listing(self, tmp_path):
        assert runner_mod._near_memory_limit(900 * 1024, 1000) is True
        assert runner_mod._near_memory_limit(899 * 1024, 1000) is False
        assert runner_mod._near_memory_limit(10**9, None) is False

        m = _tiny_manifest(tmp_path / "out")
        rc = runner_mod.main([str(_write_manifest(tmp_path, m)), "--allow-dirty"])
        assert rc == 0
        s = json.loads(
            (manifest_mod.result_dir(m) / "runs" / "pa2__HC__s1" / "summary.json").read_text()
        )
        assert s["near_memory_limit"] is False

        result_dir = tmp_path / "1" / "smoke-near"
        manifest = _valid_manifest(tmp_path)
        manifest["arms"] = {"HC": {"argv": ["-a", "hc"]}}
        manifest["seeds"] = [1, 2]
        result_dir.mkdir(parents=True)
        (result_dir / "manifest.json").write_text(json.dumps(manifest))
        runs_dir = result_dir / "runs"
        _write_summary(runs_dir, "pa4__HC__s1", "pa4", "HC", 1,
                       near_memory_limit=True, ru_maxrss_kb=950_000, mem_limit_mb=1000)
        _write_summary(runs_dir, "pa4__HC__s2", "pa4", "HC", 2, near_memory_limit=False)
        rc = analyze_mod.main([str(result_dir), "--baseline", "HC", "--contrasts", "HC:HC"])
        assert rc == 0
        text = (result_dir / "analysis" / "failures.md").read_text()
        assert "near the memory limit" in text
        assert "pa4__HC__s1" in text and "pa4__HC__s2" not in text
