"""Server round 17: the HC-TLS check (``SERVER-ROUND17-PREREGISTRATION.md``, "Arms").

HC-TLS is hill climbing with the constant tuning of BASE-TLS::

    -a hc --knob-probe-baseline reduced --linear-scaling terms --const-opt none

``--linear-scaling terms`` sets the per-term fit on the one scorer that all optimizers share
(``table_problems._regression``, ``contin_bscore.set_linear_scaling_terms``). No part of the
per-term fit is GOMEA-only: the runner refits the terms of each printed candidate with
``sr_eval.py`` for any arm with ``--linear-scaling terms`` in its argv. These tests check
the pass criteria of the pre-registration:

* the command line accepts the flags with ``-a hc``, and the HC search scores its
  candidates with the per-term fit (the counters ``contin_bscore._terms_stats``);
* the score that MOSES prints for each candidate is the per-term-fit score, computed again
  outside MOSES with ``sr_eval.term_columns`` and ``least_squares.fit_terms``, and not the
  score of the plain a + b*f fit;
* end to end (``moses_sr_runner``, Yac old split 1 built from ``data/gomea-sr``, 2,000
  evaluations): the run ends ok, the final model uses the per-term fit (``terms.n_terms``
  >= 2, no fallback), and the printed score of each candidate gives its training NMSE of
  ``sr_eval.py`` (``raw_to_train_nmse``) and its per-term score on ``train_scaled.csv``.

Two earlier tests ran ``-a hc`` with ``--linear-scaling terms``, but did not check the scores:
``tests/moses/scoring/test_linear_scaling_terms.py::test_cli_terms_runs_and_changes_scores``
(the HC-TLS flags, 400 evaluations: the run ends and its output differs from
``--linear-scaling 1``) and ``tests/gomea/test_const_opt_terms.py`` (with ``--const-opt lm``).

The scores are float32 (``score_t``), so a score and its value computed again agree to a
relative difference of at most ``REL`` (at most 6e-8 on the HC-TLS probe runs on Con and
Yac, old splits 1 to 3).
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from pymoses.moses.scoring.bscores import TERMS_STATS_KEYS, contin_bscore
from pymoses.moses.scoring.least_squares import fit_terms

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402
import sr_datasets  # noqa: E402
import sr_eval  # noqa: E402

HC_TLS = [
    "-a", "hc", "--knob-probe-baseline", "reduced",
    "--linear-scaling", "terms", "--const-opt", "none",
]  # fmt: skip
REL = 1e-6
N_ROWS = 48


def _table(path: Path) -> Path:
    """48 rows, y = 4 x1 + 0.6 x2 x3 - 1.3 x3 + a fixed noise (so no program fits exactly)."""
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x1", "x2", "x3", "y"])
        for i in range(N_ROWS):
            x1, x2, x3 = -3 + 6 * i / (N_ROWS - 1), ((i * 7) % 11) / 5.0, ((i * 5) % 13) / 4.0
            y = 4 * x1 + 0.6 * x2 * x3 - 1.3 * x3 + 0.4 * math.sin(12.9898 * i)
            wr.writerow([repr(x1), repr(x2), repr(x3), repr(y)])
    return path


def _reg(train: Path, log: Path, m: str = "500") -> list[str]:
    """The fixed regression setup of tools/gomea/moses_sr_runner.py (``-v`` 60 * n_train)."""
    return [
        "-i", str(train), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", str(60 * N_ROWS), "-c", "10", "-V1", "-l", "WARN", "-f", str(log), "-m", m,
    ]  # fmt: skip


def _run(argv: list[str], cwd: Path) -> str:
    """The stdout of one in-process MOSES run."""
    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(cwd)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue()


def _recompute(prog: str, table: sr_eval.Table) -> tuple[int, str | None, float, float]:
    """``(n_terms, fallback reason, per-term score, plain score)`` of ``prog`` on ``table``,
    computed outside MOSES. A score is minus the sum of the squared errors on the rows, as
    MOSES scores a regression candidate without the complexity penalty. The per-term score
    is the plain score for one term or a fallback (the rule of the scorer)."""
    p = sr_eval.predict(prog, table.X, table.labels, table.bool_cols)
    a, b = sr_eval.linear_scaling(table.y, p)
    plain = -math.fsum((a + b * x - t) ** 2 for x, t in zip(p, table.y, strict=True))
    k, cols = sr_eval.term_columns(prog, [table], table.bool_cols)
    if k < 2:
        return k, None, plain, plain
    if cols is None:
        return k, "constant", plain, plain
    fit = fit_terms(table.y, cols[0])
    if fit.weights is None:
        return k, fit.reason, plain, plain
    terms = -math.fsum((f - t) ** 2 for f, t in zip(fit.fitted, table.y, strict=True))
    return k, None, terms, plain


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---- the flags


def test_hc_tls_has_the_constant_flags_of_base_tls():
    """HC-TLS takes `--linear-scaling`, `--const-opt` and `--knob-probe-baseline` from the
    round 16 arm BASE-TLS (configs/gomea/phase1bk/sr.json); only `-a` differs."""
    arm = json.loads((REPO / "configs" / "gomea" / "phase1bk" / "sr.json").read_text())
    base = arm["arms"]["BASE-TLS"]["argv"]

    def value(argv, flag):
        return argv[argv.index(flag) + 1]

    for flag in ("--linear-scaling", "--const-opt", "--knob-probe-baseline"):
        assert value(HC_TLS, flag) == value(base, flag), flag
    assert value(base, "-a") == "gomea"
    assert value(HC_TLS, "-a") == "hc"
    assert runner._opt_algo(HC_TLS) == "hc"
    assert runner._linear_scaling(HC_TLS) == "terms"
    assert runner._const_opt(HC_TLS) == "none"


# ---- the search (in process, a small table)


def test_the_hc_search_scores_with_the_per_term_fit(tmp_path, monkeypatch):
    """With the HC-TLS flags, the scorer of the run has the per-term fit on (and the plain
    a + b*f fit off), and the HC search scores candidates through it: the per-term fit is
    used for some of them, and the counters add up."""
    seen: list[contin_bscore] = []
    orig = contin_bscore.set_linear_scaling_terms

    def spy(self, on=True):
        seen.append(self)
        return orig(self, on)

    monkeypatch.setattr(contin_bscore, "set_linear_scaling_terms", spy)
    out = _run(_reg(_table(tmp_path / "train.csv"), tmp_path / "moses.log") + HC_TLS, tmp_path)
    assert int(out.rsplit("n_evals:", 1)[1]) == 500
    assert len(seen) == 1
    b = seen[0]
    assert b._linear_scaling_terms is True
    assert b._linear_scaling is False
    stats = b._terms_stats
    assert set(stats) == set(TERMS_STATS_KEYS)
    assert stats["calls"] > 0
    assert stats["fitted"] > 0
    outcomes = ("one_term", "fitted", "fallback_constant", "fallback_singular",
                "fallback_nonfinite", "fallback_error")  # fmt: skip
    assert stats["calls"] == sum(stats[k] for k in outcomes)


def test_hc_tls_prints_the_per_term_fit_score_of_each_candidate(tmp_path):
    """The score that MOSES prints for each candidate equals its per-term-fit score
    computed again outside MOSES. At least one candidate uses the per-term fit (2 terms or
    more, no fallback), and for it the plain a + b*f score is clearly worse: MOSES prints
    the per-term score, not the plain one."""
    train = _table(tmp_path / "train.csv")
    out = _run(_reg(train, tmp_path / "moses.log") + HC_TLS, tmp_path)
    cands = sr_eval.parse_moses_output(out)
    assert len(cands) == 10
    table = sr_eval.read_table(str(train))
    fitted = 0
    for score, prog in cands:
        assert score is not None and math.isfinite(score), prog
        k, reason, terms, plain = _recompute(prog, table)
        assert _rel(score, terms) <= REL, (prog, score, terms)
        if k >= 2 and reason is None:
            fitted += 1
            assert _rel(terms, plain) > 1e-3, prog  # the plain score is clearly different
            assert terms > plain, prog
    assert fitted >= 1


# ---- end to end: the runner on Yac, old split 1


def _yac_split_1(datasets: Path) -> Path:
    """``datasets/Yac/splits/s1/{train,val,test}.csv`` built from the committed
    ``data/gomea-sr/Yac/full.csv`` as ``sr_datasets.build`` builds them."""
    rows = sr_datasets.load_full(REPO / "data" / "gomea-sr", "Yac")[0]
    nfeat = sr_datasets.DATASETS["Yac"][3]
    tr, va, te = sr_datasets.split_indices("Yac", 1, len(rows))
    dst = datasets / "Yac" / "splits" / "s1"
    dst.mkdir(parents=True)
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        (dst / f"{name}.csv").write_text(sr_datasets.csv_text([rows[i] for i in ix], nfeat))
    return dst


def _pymoses_bin(tmp_path: Path) -> Path:
    """A ``bin/moses`` that the runner takes for pyMoses: it runs the ``python`` beside it
    (here this interpreter) with ``PYTHONPATH`` this repository."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    moses = bindir / "moses"
    moses.write_text("#!/bin/sh\nexit 1\n")  # never run: the runner starts `python -c`
    moses.chmod(0o755)
    return moses


def test_hc_tls_end_to_end_on_yac_old_split_1(tmp_path):
    """The HC-TLS check run: the HC-TLS arm through the runner (`expand`, `_run_one`) with
    2,000 evaluations. The run ends ok, the reported model uses the per-term fit, and the
    printed score of each candidate is its per-term-fit score (on the scaled training target
    that MOSES reads, and as the training NMSE of sr_eval.py on the original split)."""
    _yac_split_1(tmp_path / "datasets")
    manifest = {
        "name": "hctls-check", "datasets_dir": str(tmp_path / "datasets"),
        "datasets": ["Yac"], "splits": [1], "wall_cap_s": 3600, "mem_limit_mb": 5120,
        "clip_test_nmse": True, "plain_nmse": True,
        "arms": {"HC-TLS": {"binary": str(_pymoses_bin(tmp_path)),
                            "budget": {"evals": 2000}, "argv": HC_TLS}},
    }  # fmt: skip
    (job,) = runner.expand(manifest, tmp_path / "out", REPO)
    s = runner._run_one(job)
    run = Path(job["run_dir"])

    # (a) the run ends ok and uses its budget
    assert s["status"] == "ok", s.get("error")
    assert s["n_evals"] == 2000
    assert s["n_candidates"] == 10
    assert s["n_candidates_unscorable"] == 0

    # (b) the reported model uses the per-term fit
    assert s["terms"]["n_terms"] >= 2
    assert s["terms"]["fallback"] is False
    assert s["terms"]["reason"] is None
    assert len(s["terms"]["weights"]) == s["terms"]["n_terms"] + 1

    # (c) the score of each candidate (the reported one among them) is its per-term-fit
    # score, computed again outside MOSES
    cands = sr_eval.parse_moses_output((run / "stdout.txt").read_text())
    scored = json.loads((run / "candidates.json").read_text())
    assert [p for _s, p in cands] == [c["program"] for c in scored]
    assert s["final_program"] in [p for _s, p in cands]
    scaled = sr_eval.read_table(str(run / "train_scaled.csv"))
    var = runner._scaled_target_var(run / "train_scaled.csv")
    for (score, prog), c in zip(cands, scored, strict=True):
        assert score is not None and math.isfinite(score), prog
        k, reason, terms, _plain = _recompute(prog, scaled)
        assert k == c["terms"]["n_terms"], prog
        assert reason == c["terms"]["reason"], prog
        assert _rel(score, terms) <= REL, (prog, score, terms)
        nmse = runner.raw_to_train_nmse(score, s["n_train"], var)
        assert _rel(nmse, c["train_nmse"]) <= REL, prog
