"""Backlog 12.1: the validation table of ``--const-opt-val`` takes the input types of the
training table (``const_opt.load_rows``), not types inferred from its own first 20 lines.

Round 13, Dow HC-LM split 56: the first 19 validation values of ``x13`` were 0, so the loader
typed ``x13`` boolean, and a later value (801.904) raised ``ValueError``. The tables here have
the same shape, with no Dow rows: column ``z`` is numeric in the training table, and its
first 19 validation values are 0. Column ``b`` is boolean (0 and 1 only) in both tables, as
Bos ``x4`` is. The identity of every other split is ``test_val_rows_identity.py``.
"""

from __future__ import annotations

import csv
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.comboreduct.type_checker.type_signature import type_node
from pymoses.gomea.const_opt import RegressionRows, load_rows
from pymoses.gomea.optimizer import gomea_optimization
from pymoses.moses.scoring.bscores import contin_bscore

C = 1.37  # y = x1 (x2 + C), as in test_hc_lm.py
LABELS = ["x1", "x2", "z", "b"]
# validation z: 19 zeros, then values that are not 0 or 1 (801.904 as on Dow split 56)
VAL_Z = ["0"] * 19 + ["801.904", "0", "2.5", "0", "1", "0.25", "0", "0", "3", "0", "0"]


def _write(path: Path, rows, header=("x1", "x2", "z", "b", "y")) -> Path:
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    return path


def _train(path: Path) -> Path:
    """40 rows; z is numeric in the first 20 rows (contin), b is 0 or 1 (boolean)."""
    rows = []
    for i in range(40):
        x1, x2 = -3 + 6 * i / 39, ((i * 7) % 11) / 5.0
        rows.append([repr(x1), repr(x2), str(((i * 3) % 7) / 2), str(i % 2),
                     repr(x1 * (x2 + C))])
    return _write(path, rows)


def _val_rows(z=VAL_Z, b=None, y=None):
    rows = []
    for i, zi in enumerate(z):
        x1, x2 = -2.9 + 5.8 * i / (len(z) - 1), ((i * 5) % 9) / 4.0
        bi = (i // 3) % 2 if b is None else b[i]
        yi = repr(x1 * (x2 + C)) if y is None else y[i]
        rows.append([repr(x1), repr(x2), zi, str(bi), yi])
    return rows


def _val(path: Path, **kw) -> Path:
    return _write(path, _val_rows(**kw))


def _types(train: Path):
    return loadTable(str(train), "y").itable.get_types()


def test_the_loader_alone_types_the_column_boolean_and_fails(tmp_path):
    """The cause: types inferred from the first 20 lines of the validation file."""
    val = _val(tmp_path / "val.csv")
    with pytest.raises(ValueError, match="Expecting boolean value, got 801.904"):
        loadTable(str(val), "y")
    assert _types(_train(tmp_path / "train.csv")) == [type_node.contin_type] * 3 + [
        type_node.boolean_type]


def test_a_numeric_column_with_19_zeros_first_loads_with_numeric_values(tmp_path):
    train, val = _train(tmp_path / "train.csv"), _val(tmp_path / "val.csv")
    rows = load_rows(str(val), LABELS, True, _types(train))
    assert len(rows.inputs) == len(VAL_Z) and rows.scaled
    z = [r[2] for r in rows.inputs]
    assert z == [float(t) for t in VAL_Z]
    assert all(type(v) is float for v in z)
    assert z[19] == 801.904
    want_y = [float(r[4]) for r in _val_rows()]
    assert rows.targets == want_y


def test_a_boolean_column_still_loads_as_boolean(tmp_path):
    train, val = _train(tmp_path / "train.csv"), _val(tmp_path / "val.csv")
    rows = load_rows(str(val), LABELS, False, _types(train))
    b = [r[3] for r in rows.inputs]
    assert set(b) == {builtin.logical_true, builtin.logical_false}
    assert b == [builtin.logical_true if (i // 3) % 2 else builtin.logical_false
                 for i in range(len(VAL_Z))]
    # the same values as the loader gives when it types the file itself (no z problem)
    ok = _val(tmp_path / "ok.csv", z=["2.5"] + VAL_Z[1:])
    assert [r[3] for r in loadTable(str(ok), "y").itable] == b


def test_the_guard_reads_the_rows_with_the_training_types(tmp_path):
    """``_guard_rows`` (GOMEA, HC, and LM during the search) passes the training types."""
    train, val = _train(tmp_path / "train.csv"), _val(tmp_path / "val.csv")
    bs = contin_bscore(loadTable(str(train), "y"))
    me = SimpleNamespace(_val_rows=None, gomea_params=SimpleNamespace(const_opt_val=str(val)))
    rows = gomea_optimization._guard_rows(me, SimpleNamespace(get_bscorer=lambda: bs),
                                          RegressionRows.of_bscore(bs))
    assert [r[2] for r in rows.inputs] == [float(t) for t in VAL_Z]
    assert me._val_rows is rows  # read once


def test_a_value_the_training_type_cannot_take_raises_a_clear_error(tmp_path):
    train = _train(tmp_path / "train.csv")
    b = [(i // 3) % 2 for i in range(len(VAL_Z))]
    b[6] = "0.5"  # row 7: b is boolean in the training table
    val = _val(tmp_path / "val.csv", b=b)
    with pytest.raises(ValueError) as exc:
        load_rows(str(val), LABELS, True, _types(train))
    assert str(exc.value) == (
        f"--const-opt-val {val}: row 7, column b: Expecting boolean value, got 0.5 "
        "(the training table types b as boolean)")


def test_a_target_that_is_not_a_number_raises_a_clear_error(tmp_path):
    train = _train(tmp_path / "train.csv")
    y = [repr(float(i)) for i in range(len(VAL_Z))]
    y[11] = "1.5.2"
    val = _val(tmp_path / "val.csv", y=y)
    with pytest.raises(ValueError, match=r": row 12, column y: Could not cast 1\.5\.2 to "
                                         r"contin \(the target is contin\)$"):
        load_rows(str(val), LABELS, True, _types(train))


def test_the_target_is_contin_even_if_its_first_values_are_0_and_1(tmp_path):
    train = _train(tmp_path / "train.csv")
    y = [str(i % 2) for i in range(len(VAL_Z))]
    y[25] = "4.75"
    val = _val(tmp_path / "val.csv", y=y)
    rows = load_rows(str(val), LABELS, True, _types(train))
    assert rows.targets == [float(t) for t in y]


@pytest.mark.parametrize("header", [("x1", "z", "x2", "b", "y"), ("x1", "x2", "z", "y"),
                                    ("x1", "x2", "z", "b", "w", "y")])
def test_the_label_check_stays(header, tmp_path):
    train = _train(tmp_path / "train.csv")
    cols = ("x1", "x2", "z", "b", "y")
    rows = [[dict(zip(cols, r, strict=True)).get(h, "0") for h in header] for r in _val_rows()]
    val = _write(tmp_path / "val.csv", rows, header=header)
    labels = [h for h in header if h != "y"]
    with pytest.raises(ValueError, match=r"input columns \[.*\] differ from the training "
                                         r"columns \['x1', 'x2', 'z', 'b'\]") as exc:
        load_rows(str(val), LABELS, True, _types(train))
    assert str(labels) in str(exc.value)


def test_a_row_with_a_missing_value_raises_a_clear_error(tmp_path):
    train = _train(tmp_path / "train.csv")
    rows = _val_rows()
    rows[25] = rows[25][:2] + rows[25][3:]  # no z on row 26 (after the 20 lines of the
    val = _write(tmp_path / "val.csv", rows)  # type inference, which checks its own lines)
    with pytest.raises(ValueError, match=r": row 26 has 4 values, the header 5$"):
        load_rows(str(val), LABELS, True, _types(train))


# ---- runs: the guard and polish read the table with the training types


def _reg(train: Path, m: str) -> list[str]:
    """The fixed regression setup of tools/gomea/moses_sr_runner.py."""
    return ["-i", str(train), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


def _run(argv, tmp_path):
    """(stdout, events of --gomea-log) of one in-process run."""
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    log.unlink(missing_ok=True)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


_GOMEA = ["-a", "gomea", "--gomea-pop-size", "16", "--gomea-skip-inactive", "0",
          "--gomea-max-evals-per-deme", "250", "--gomea-genotype-memo", "1",
          "--gomea-contin", "copy", "--linear-scaling", "1"]
RUNS = {
    # the round 13 HC-LM flags (the arm that failed on Dow split 56)
    "hc-lm": ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
              "--const-opt", "lm", "--const-opt-guard", "val"],
    "gomea-lm": _GOMEA + ["--const-opt", "lm", "--const-opt-guard", "val"],
    "polish": _GOMEA + ["--const-opt", "polish"],
}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_run_with_the_guard_or_polish_reads_the_table(name, tmp_path):
    train, val = _train(tmp_path / "train.csv"), _val(tmp_path / "val.csv")
    out, events = _run(_reg(train, "800") + RUNS[name] + ["--const-opt-val", str(val)],
                       tmp_path)
    assert int(out.rsplit("n_evals:", 1)[1]) <= 800
    checked = sum((e.get("const_opt") or {}).get("lm_guard_checked", 0)
                  for e in events if e["type"] == "end")
    polish = [e for e in events if e["type"] == "polish"]
    if name == "polish":
        assert len(polish) == 1 and "val_nmse_new" in polish[0]  # LM ran, the guard compared
    else:
        assert checked >= 1  # past the first guard call


@pytest.mark.parametrize("b_train, ok", [("0.5", True), ("1", False)])
def test_a_sparse_table_keeps_its_types_and_they_must_be_the_training_types(
        b_train, ok, tmp_path):
    """A sparse file (``key : value``) gets one type for all its keyed columns, from all of
    its rows. Training b: contin with a 0.5 in its first 20 rows, else boolean."""
    rows = [[repr(0.1 * i), b_train if i == 3 else str(i % 2), repr(0.5 * i), repr(1.0 * i)]
            for i in range(30)]
    train = _write(tmp_path / "train.csv", rows, header=("x1", "b", "z", "y"))
    sparse = tmp_path / "val.csv"
    sparse.write_text("x1,y\n" + "".join(
        f"{0.2 * i!r}, {2.0 * i!r}, b : {i % 2}, z : {0.25 * i!r}\n" for i in range(30)))
    types = _types(train)
    if ok:
        rows = load_rows(str(sparse), ["x1", "b", "z"], True, types)
        t = loadTable(str(sparse), "y")
        assert rows.inputs == [list(r) for r in t.itable]
        assert rows.targets == [2.0 * i for i in range(30)]
    else:
        with pytest.raises(ValueError, match=r"input types \['contin', 'contin', 'contin'\] "
                                             r"differ from the training types \['contin', "
                                             r"'boolean', 'contin'\]"):
            load_rows(str(sparse), ["x1", "b", "z"], True, types)
