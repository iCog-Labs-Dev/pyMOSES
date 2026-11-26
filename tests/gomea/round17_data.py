"""Server round 17: made-up HC data and a fake pyMoses for an HC arm, for the round 17 runner
and report tests (``test_round17_runner.py``, ``test_round17_report.py``). No data rows."""

from __future__ import annotations

import json
from pathlib import Path

from round16_data import PROGRAMS, splits

HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
         "--const-opt", "lm", "--const-opt-guard", "val"]
HC_TLS = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
          "--const-opt", "none"]


def hc_lines(*pts) -> str:
    """Hook lines from (n_evals, best_raw) pairs."""
    return "".join(json.dumps({"expansion": i + 1, "n_evals": n, "best_raw": r}) + "\n"
                   for i, (n, r) in enumerate(pts))


def hc_end(max_evals, hc_evals, tuning, lm) -> dict:
    return {"type": "end", "optimizer": "hc", "max_evals": max_evals, "hc_evals": hc_evals,
            "ledger": {"requests": hc_evals + tuning, "tuning_requests": tuning},
            "const_opt": {"mode": "lm", "tuning_evals": tuning, "lm": lm}}


HC_ENDS = [  # a budget of 1000: demes of 250 (LM kept), 270 (guard rejected), 480 (budget)
    hc_end(1000, 230, 20, {"status": "improved", "mse0": 9.5, "mse": 9.0, "written_back": True}),
    hc_end(750, 260, 10, {"status": "improved", "mse0": 7.0, "mse": 6.5, "written_back": False,
                           "guard": "rejected"}),
    hc_end(480, 480, 0, {"status": "budget", "mse0": float("inf"), "mse": float("inf")}),
]


def fake_hc(work: Path, lines: str, events: str | None, n_evals: int = 1000) -> Path:
    """A fake pyMoses for an HC arm: it prints PROGRAMS, writes `events` (if any) to the path
    after `--gomea-log`, and writes `lines` to the path in the boot when the boot installs the
    hook."""
    b = work / "bin"
    b.mkdir(parents=True, exist_ok=True)
    (b / "moses").write_text("#!/bin/sh\nexit 1\n")
    (work / "lines.src").write_text(lines)
    (work / "events.src").write_text(events or "")
    out = "".join(f"-{i + 10} {p} \n" for i, p in enumerate(PROGRAMS)) + f"n_evals: {n_evals}\n"
    script = (
        "#!/bin/sh\nboot=\"$2\"\nprev=\nfor a in \"$@\"; do\n"
        f"  [ \"$prev\" = --gomea-log ] && cp '{work / 'events.src'}' \"$a\"\n"
        "  prev=\"$a\"\ndone\n"
        "case \"$boot\" in *hc_progress.install*)\n"
        "  p=$(printf '%s' \"$boot\" | sed -n \"s/.*hc_progress.install('\\([^']*\\)').*/\\1/p\")\n"
        f"  cp '{work / 'lines.src'}' \"$p\" ;;\nesac\n"
        "cat <<'EOF'\n" + out + "EOF\n")
    (b / "python").write_text(script)
    (b / "python").chmod(0o755)
    return b / "moses"


def hc_job(work: Path, lines: str, events: str | None = None, argv=HC_LM, **kw) -> dict:
    return {"run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1,
            "binary": str(fake_hc(work, lines, events)), "argv": argv,
            "budget": {"evals": 1000}, "seed_spec": None,
            "split_dir": str(splits(work / "splits" / "s1")), "wall_cap_s": 60,
            "mem_limit_mb": None, "run_dir": str(work / "run"), **kw}


LINES = hc_lines((250, -900.0), (520, -700.0), (1000, -10.0))
EVENTS = "".join(json.dumps(e) + "\n" for e in HC_ENDS)
