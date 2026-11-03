"""Write the reference of tests/gomea/test_const_opt_guard.py. Run it at commit ed7e383, the
commit before `--const-opt-guard`, from the repository root::

    .venv/bin/python tests/gomea/data/const_opt_lm_ed7e383/make_reference.py OUT_DIR
"""

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from pymoses.moses.main.moses_exec import moses_exec

DATA = Path("tests/gomea/data/const_opt_none").resolve()
ARGV = [
    "-i", str(DATA / "train.csv"), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "3", "-V1", "-l", "WARN", "-m", "700", "-a", "gomea",
    "--gomea-pop-size", "16", "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "1",
]  # fmt: skip
CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})
CASES = {"lm": ["--const-opt", "lm"],
         "polish": ["--const-opt", "polish", "--const-opt-val", str(DATA / "train.csv")]}

out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
for name, extra in CASES.items():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "events.jsonl"
        old = os.getcwd()
        os.chdir(td)
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert moses_exec(["moses", *ARGV, *extra, "--gomea-log", str(log)]) == 0
        os.chdir(old)
        lines = []
        for line in log.read_text().splitlines():
            ev = {k: v for k, v in json.loads(line).items() if k not in CLOCK_KEYS}
            if "telemetry_path" in ev.get("params", {}):
                ev["params"]["telemetry_path"] = "LOG"
            lines.append(json.dumps(ev, sort_keys=True))
    (out_dir / f"{name}.out").write_text(buf.getvalue())
    (out_dir / f"{name}.events.jsonl").write_text(
        ("\n".join(lines) + "\n").replace(str(DATA), "DATA"))
    print(name, len(buf.getvalue().splitlines()), "stdout lines,", len(lines), "events")
