"""Backlog 12.1 (the validation table of ``--const-opt-val``): the rows that the LM guard
reads, and the output of LM-guard runs, are the same as at commit 0f6f2eb (before the fix).

The reference files in ``data/val_rows_0f6f2eb/`` were captured at commit 0f6f2eb by running
this module as a script there::

    .venv/bin/python tests/gomea/test_val_rows_identity.py DATASETS_DIR OUT_DIR

DATASETS_DIR is a Phase 1f datasets directory (``tools/gomea/sr_datasets.py``) with splits
1 to 60 of the 10 datasets. For each (dataset, split), the script scales ``train.csv`` and
``val.csv`` as ``tools/gomea/moses_sr_runner.py`` does (``_scale_train``,
``_scale_like_train``), loads the scaled training table as MOSES does (``loadTable``, then
``contin_bscore``), and asks the guard for its validation rows (``_guard_rows``, the method
that GOMEA and HC use). ``digests.json`` has the SHA-256 of those rows (``rows_digest``:
each value with its Python type, and each target) for each pair where the old code loads
the file, and the error for each pair where it fails (Dow splits 11 and 56). It has no data
rows.

The LM-guard runs use Bos split 1 (a committed dataset; ``x4`` is a boolean column), scaled
as the runner does, with the fixed regression setup of the runner: the round 13 HC-LM
flags, the round 12 BASE flags (GOMEA with LM and the guard; with a deme cap of 400
evaluations, so that LM and the guard run) and ``--const-opt polish``. Each run must read the
validation rows (``guard_calls``).
Compared: the full MOSES stdout and the ``--gomea-log`` file without its wall-clock fields
(``CLOCK_KEYS``).

``--check DATASETS_DIR`` compares the rows of every (dataset, split) in DATASETS_DIR with
``digests.json`` and prints the counts (Dow and Tow are not committed, so the tests below
use the 8 committed datasets only).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
REF = Path(__file__).resolve().parent / "data" / "val_rows_0f6f2eb"
SRC = REPO / "data" / "gomea-sr"  # the committed full.csv files
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402
import sr_datasets  # noqa: E402

DATASETS = ("Air", "Bos", "Con", "Dow", "EnC", "EnH", "Tow", "WiR", "WiW", "Yac")
COMMITTED = ("Air", "Bos", "Con", "EnC", "EnH", "WiR", "WiW", "Yac")  # no Dow, no Tow
SPLITS = range(1, 61)
CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def scale_split(train: Path, val: Path, dst: Path) -> int:
    """``dst/train_scaled.csv`` and ``dst/val_scaled.csv`` as the runner writes them; returns
    the number of training rows."""
    dst.mkdir(parents=True, exist_ok=True)
    n, mean, sd = runner._scale_train(train, dst / "train_scaled.csv")
    runner._scale_like_train(val, dst / "val_scaled.csv", mean, sd)
    return n


@functools.cache
def _full_rows(abbr: str) -> list:
    return sr_datasets.load_full(SRC, abbr)[0]


def committed_split(abbr: str, k: int, dst: Path) -> tuple[Path, Path]:
    """``train.csv`` and ``val.csv`` of split ``k`` of a committed dataset, built from
    ``data/gomea-sr/<abbr>/full.csv`` as ``sr_datasets.build`` builds them."""
    rows = _full_rows(abbr)
    nfeat = sr_datasets.DATASETS[abbr][3]
    tr, va, _te = sr_datasets.split_indices(abbr, k, len(rows))
    dst.mkdir(parents=True, exist_ok=True)
    for name, ix in (("train", tr), ("val", va)):
        (dst / f"{name}.csv").write_text(sr_datasets.csv_text([rows[i] for i in ix], nfeat))
    return dst / "train.csv", dst / "val.csv"


def guard_rows(train_scaled: Path, val_scaled: Path):
    """The validation rows that the LM guard reads for this training table."""
    from pymoses.comboreduct.table.table_io import loadTable
    from pymoses.gomea.const_opt import RegressionRows
    from pymoses.gomea.optimizer import gomea_optimization
    from pymoses.moses.scoring.bscores import contin_bscore

    b = contin_bscore(loadTable(str(train_scaled), "y"))
    cscorer = SimpleNamespace(get_bscorer=lambda: b)
    me = SimpleNamespace(
        _val_rows=None, gomea_params=SimpleNamespace(const_opt_val=str(val_scaled)))
    return gomea_optimization._guard_rows(me, cscorer, RegressionRows.of_bscore(b))


def rows_digest(rows) -> str:
    """SHA-256 of the rows: ``scaled``, then per row each input value with its Python type,
    and the target."""
    h = hashlib.sha256(f"scaled={rows.scaled!r}\n".encode())
    for r, y in zip(rows.inputs, rows.targets, strict=True):
        cells = ",".join(f"{type(v).__name__}:{v!r}" for v in r)
        h.update(f"{cells}|{y!r}\n".encode())
    return h.hexdigest()


def split_digest(train: Path, val: Path, work: Path) -> str | dict:
    """The digest of the guard rows of one split, or ``{"error": ...}`` when they fail to
    load."""
    scale_split(train, val, work)
    try:
        return rows_digest(guard_rows(work / "train_scaled.csv", work / "val_scaled.csv"))
    except Exception as exc:  # the old code fails on Dow splits 11 and 56
        return {"error": f"{type(exc).__name__}: {exc}"}


def _reference() -> dict:
    return json.loads((REF / "digests.json").read_text())


@pytest.mark.parametrize("abbr", COMMITTED)
def test_the_guard_rows_of_every_split_are_those_of_0f6f2eb(abbr, tmp_path):
    ref = _reference()
    for k in SPLITS:
        train, val = committed_split(abbr, k, tmp_path / f"s{k}")
        got = split_digest(train, val, tmp_path / f"s{k}" / "scaled")
        assert got == ref[f"{abbr}/s{k}"], (abbr, k)


def test_the_reference_has_every_split_and_only_the_two_known_failures():
    ref = _reference()
    assert set(ref) == {f"{a}/s{k}" for a in DATASETS for k in SPLITS}
    failed = sorted(key for key, v in ref.items() if isinstance(v, dict))
    assert failed == ["Dow/s11", "Dow/s56"]
    for key in failed:
        assert ref[key]["error"].startswith("ValueError: Expecting boolean value, got ")


# ---- LM-guard runs on Bos split 1


def _reg(n_train: int, m: str) -> list[str]:
    return ["-i", "train_scaled.csv", "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n",
            "log", "-v", str(60 * n_train), "-c", "10", "-V1", "-l", "WARN", "-m", m]


_VAL = ["--const-opt-val", "val_scaled.csv"]
# configs/gomea/phase1bh/sr.json, HC-LM and BASE, flag for flag
_HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
          "--const-opt", "lm", "--const-opt-guard", "val"]
_BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
         "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
         "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
         "--knob-probe-baseline", "reduced", "--linear-scaling", "1", "--const-opt", "lm",
         "--const-opt-guard", "val", "--gomea-accept-complexity-ratio", "0.125",
         "--gomea-deme-store", "accepted"]  # fmt: skip
_POLISH = ["-a", "gomea", "--gomea-pop-size", "16", "--gomea-skip-inactive", "0",
           "--gomea-max-evals-per-deme", "250", "--gomea-genotype-memo", "1",
           "--gomea-contin", "copy", "--linear-scaling", "1", "--const-opt", "polish"]
CASES = {
    "bos-s1-hc-lm": ("1500", _HC_LM + _VAL),
    # a deme cap, so that the demes end with budget left and LM and the guard run
    "bos-s1-base": ("1500", _BASE + ["--gomea-max-evals-per-deme", "400"] + _VAL),
    "bos-s1-polish": ("700", _POLISH + _VAL),
}
_MAIN = ("import sys; sys.setrecursionlimit(100000); "
         "from pymoses.moses.main.moses_exec import moses_exec; "
         "sys.exit(moses_exec(sys.argv[1:]))")


def canonical_events(text: str) -> str:
    """The ``--gomea-log`` events without their wall-clock fields."""
    lines = []
    for line in text.splitlines():
        ev = {k: v for k, v in json.loads(line).items() if k not in CLOCK_KEYS}
        lines.append(json.dumps(ev, sort_keys=True))
    return "\n".join(lines) + "\n" if lines else ""


def bos_split1(work: Path) -> int:
    """Bos split 1 in ``work``, scaled as the runner does; returns the training row count."""
    train, val = committed_split("Bos", 1, work / "split")
    return scale_split(train, val, work)


def run_case(case: str, work: Path) -> tuple[str, str]:
    """(stdout, canonical --gomea-log) of one LM-guard run on Bos split 1 in ``work``."""
    n_train = bos_split1(work)
    m, flags = CASES[case]
    (work / "events.jsonl").unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, "-c", _MAIN, "moses", *_reg(n_train, m), *flags,
         "--gomea-log", "events.jsonl"],
        cwd=work, capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONHASHSEED": "0",
             "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout, canonical_events((work / "events.jsonl").read_text())


def guard_calls(events: str) -> int:
    """Guard checks (deme ends) and polish calls in a canonical --gomea-log: each one reads the
    validation rows."""
    n = 0
    for line in events.splitlines():
        ev = json.loads(line)
        if ev["type"] == "end":
            n += (ev.get("const_opt") or {}).get("lm_guard_checked", 0)
        elif ev["type"] == "polish":
            n += 1
    return n


@pytest.mark.parametrize("case", sorted(CASES))
def test_lm_guard_runs_on_a_normal_split_are_byte_identical_to_0f6f2eb(case, tmp_path):
    stdout, events = run_case(case, tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()
    assert guard_calls(events) > 0  # the run read the validation rows


def _check(ddir: Path) -> int:
    """``--check``: the rows of every (dataset, split) in ``ddir`` against digests.json."""
    ref = _reference()
    counts = {"same": 0, "differ": 0, "old_error_now_loads": 0, "error": 0}
    with tempfile.TemporaryDirectory() as td:
        for a in DATASETS:
            for k in SPLITS:
                s = ddir / a / "splits" / f"s{k}"
                got = split_digest(s / "train.csv", s / "val.csv", Path(td) / f"{a}-{k}")
                want = ref[f"{a}/s{k}"]
                if isinstance(got, dict):
                    counts["error"] += 1
                    print(f"{a}/s{k}: error {got['error']} (reference {want})")
                elif isinstance(want, dict):
                    counts["old_error_now_loads"] += 1
                    print(f"{a}/s{k}: loads now; at 0f6f2eb: {want['error']}")
                elif got == want:
                    counts["same"] += 1
                else:
                    counts["differ"] += 1
                    print(f"{a}/s{k}: DIFFERS from 0f6f2eb")
    print(json.dumps(counts))
    return 0 if counts["differ"] == 0 and counts["error"] == 0 else 1


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(_check(Path(sys.argv[2])))
    ddir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    with tempfile.TemporaryDirectory() as td:
        for a in DATASETS:
            for k in SPLITS:
                s = ddir / a / "splits" / f"s{k}"
                digests[f"{a}/s{k}"] = split_digest(
                    s / "train.csv", s / "val.csv", Path(td) / f"{a}-{k}")
    (out / "digests.json").write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(sum(isinstance(v, str) for v in digests.values()), "digests,",
          sorted(k for k, v in digests.items() if isinstance(v, dict)), "failed")
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, e = run_case(name, work)
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.events.jsonl").write_text(e)
        print(name, len(s.splitlines()), "stdout lines,", len(e.splitlines()), "events,",
              guard_calls(e), "guard calls")
