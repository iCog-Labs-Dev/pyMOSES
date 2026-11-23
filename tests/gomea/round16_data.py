"""Server round 16: made-up GOMEA logs and a fake pyMoses for the round 16 runner and report
tests (``test_round16_runner.py``, ``test_round16_report.py``). No data rows."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"
PROGRAMS = ["+(sin(*(1.3 $1)) *(0.5 $2))", "+(sin($1) $2 *($1 $2))", "*(2.5 sin(*(1.3 $1)))",
            "$2"]
TLS = ["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "none"]


# ---- made-up GOMEA logs


def rv_record(evals, **kw) -> dict:
    return {"mode": "rvg", "budget": "balance", "rv_evals": evals, "evals": evals, **kw}


def gomea_log(demes) -> str:
    """A GOMEA log: `demes` is a list of (generations, end, rv) with generations a list of
    (requests, best_raw, rv record or None), end (requests, best raw) and rv the "rv" record
    of the "end" event (or None)."""
    lines = []
    for i, (gens, end, rv) in enumerate(demes):
        lines.append({"type": "invocation_start", "deme_id": str(i + 1)})
        lines.append({"type": "init", "n": 10})
        for g, (req, raw, grv) in enumerate(gens):
            e = {"type": "generation", "generation": g, "best_raw": raw,
                 "ledger": {"requests": req}}
            if grv is not None:
                e["rv"] = grv
            lines.append(e)
        e = {"type": "end", "generations": len(gens), "termination_reason": "converged",
             "best": {"raw": end[1], "penalized": end[1] - 1.0}, "ledger": {"requests": end[0]}}
        if rv is not None:
            e["rv"] = rv
        lines.append(e)
    return "".join(json.dumps(x) + "\n" for x in lines)


TWO_DEMES = [
    # deme 1: generations end at 30, 120 and 300 evaluations; the end record has a better raw
    ([(30, -800.0, {"ran": True, "ams1_evals": 5, "ams2_evals": 0}),
      (120, -600.0, {"ran": True, "ams1_evals": 0, "ams2_evals": 0}),
      (300, -610.0, {"ran": False})], (300, -590.0),
     rv_record(40, generations=2, improving=10, kept_p=3, rejected=27, ams1_evals=5,
               ams1_improving=2, fi_triggers=1, fi_evals=4, skip_struct=7, skip_dup=2,
               skipped_p_rv=1, converged_by="structure", nis_max_seen=3, eval_s=0.5,
               step_s=0.75)),
    # deme 2 (starts at 300): generations end at 400 and 1000 (run evaluations)
    ([(100, -700.0, {"ran": True, "ams1_evals": 4, "ams2_evals": 2}),
      (700, -500.0, None)], (700, -500.0),
     rv_record(60, generations=1, improving=20, kept_p=4, rejected=36, ams1_evals=4,
               ams2_evals=2, skip_struct=1, skip_cache=5, converged_by=None, nis_max_seen=5,
               eval_s=0.25, step_s=0.5)),
]


# ---- a fake pyMoses


def splits(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    lines = TRAIN.read_text().splitlines()
    head, rows = lines[0], lines[1:]
    for name, keep in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
        body = [r for i, r in enumerate(rows) if i % 5 in keep]
        (dst / f"{name}.csv").write_text("\n".join([head, *body]) + "\n")
    return dst


def fake_pymoses(work: Path, events: str, n_evals: int = 1000) -> Path:
    """A fake pyMoses (`bin/moses`, run by the runner as `bin/python -c BOOT ...`): it prints
    PROGRAMS as candidates and writes `events` to the path after `--gomea-log` (or to the run
    directory, its cwd, when it has no such argument)."""
    b = work / "bin"
    b.mkdir(parents=True, exist_ok=True)
    (b / "moses").write_text("#!/bin/sh\nexit 1\n")
    (b / "moses").chmod(0o755)
    (work / "events.src").write_text(events)
    out = "".join(f"-{i + 10} {p} \n" for i, p in enumerate(PROGRAMS)) + f"n_evals: {n_evals}\n"
    script = ("#!/bin/sh\nlog=events.jsonl\nprev=\nfor a in \"$@\"; do\n"
              "  [ \"$prev\" = --gomea-log ] && log=\"$a\"\n  prev=\"$a\"\ndone\n"
              f"cp '{work / 'events.src'}' \"$log\"\n"
              "cat <<'EOF'\n" + out + "EOF\n")
    (b / "python").write_text(script)
    (b / "python").chmod(0o755)
    return b / "moses"


def job(work: Path, events: str, argv=TLS, **kw) -> dict:
    return {
        "run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1,
        "binary": str(fake_pymoses(work, events)), "argv": argv,
        "budget": {"evals": 1000}, "seed_spec": None,
        "split_dir": str(splits(work / "splits" / "s1")), "wall_cap_s": 60,
        "mem_limit_mb": None, "run_dir": str(work / "run"), **kw,
    }  # fmt: skip
