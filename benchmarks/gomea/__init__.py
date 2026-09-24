"""Task 9: the benchmark harness for comparing MOSES inner optimizers.

Runs many seeded end-to-end MOSES CLI invocations (`pymoses.moses.main.
moses_exec.moses_exec`) in parallel processes, with IDENTICAL scorer-level
instrumentation for every arm (`instrument.py`), and analyses the results
with honest, right-censoring-aware statistics (`stats.py`, `analyze.py`).
Stdlib only.

Modules:
    instrument  -- ScorerTrace + instrument_scorer: the one instrumentation
                   seam every arm goes through identically.
    manifest    -- manifest schema validation (fails closed), hashing, run
                   expansion, environment capture.
    runner      -- the CLI that executes a manifest's runs.
    analyze     -- the CLI that turns a result directory into CSVs/figures.
    stats       -- pure statistics functions (Wilson CI, Wilcoxon, Holm,
                   bootstrap CI, median/IQR, ERT).
    svgplot     -- a minimal, dependency-free SVG chart writer.
    fixtures    -- frozen-exemplar capture for Task 10's replay harness.
"""
