"""Shared helpers for the tests of the stand-alone GOMEA harness (NOT A PORT: the IT linkage
study, BACKLOG 13.9). All data is synthetic."""

from __future__ import annotations

import random
from pathlib import Path

HEADER = ("x1", "x2", "x3", "x4", "x5", "y")


def synthetic_rows(n: int, seed: int) -> list[tuple]:
    """Rows (x1 .. x5, y): x1 > 0; x2 in [-2, 2]; x3 an integer 0 .. 4 (zeros); x4 one of two
    values (3.5 or 7, so x4, x4^2 and 1/x4 are aliased with the intercept, as EnC/EnH x5); x5 a
    0/1 column (boolean in sr_eval); y = 3 x1 x2 + 5 x3 + 2 x5 + 0.5 x4 + noise."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        x1 = round(r.uniform(0.5, 3.0), 4)
        x2 = round(r.uniform(-2.0, 2.0), 4)
        x3 = r.randint(0, 4)
        x4 = r.choice((3.5, 7))
        x5 = r.randint(0, 1)
        y = 3 * x1 * x2 + 5 * x3 + 2 * x5 + 0.5 * x4 + r.gauss(0.0, 0.3)
        out.append((x1, x2, x3, x4, x5, round(y, 6)))
    return out


def write_split(d: Path, n_train: int = 60, n_val: int = 20, n_test: int = 20,
                seed: int = 1) -> Path:
    """train.csv, val.csv and test.csv of synthetic rows in ``d``."""
    d.mkdir(parents=True, exist_ok=True)
    rows = synthetic_rows(n_train + n_val + n_test, seed)
    parts = {"train": rows[:n_train], "val": rows[n_train:n_train + n_val],
             "test": rows[n_train + n_val:]}
    for name, body in parts.items():
        lines = [",".join(HEADER)] + [",".join(str(v) for v in row) for row in body]
        (d / f"{name}.csv").write_text("\n".join(lines) + "\n")
    return d

