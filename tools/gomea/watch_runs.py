"""Watch benchmark runner logs and exit as soon as something goes wrong.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/watch_runs.py LOG[:PID] [LOG[:PID] ...] [--interval S]

Meant to run as a background task, so that its exit is the alert. It polls every
`--interval` seconds (default 60) and exits:

* with code 1, printing the offending lines, when a log gains a line that reports a
  crash or a failed run: a Python `Traceback`, a line with `Error`/`Exception`, or a
  runner progress line whose status is not `ok` (`[k/N] run_id: error ...`);
* with code 1 when a runner PID given as `LOG:PID` has exited before its log reached
  `[N/N]`;
* with code 0 when every log with a PID has reached `[N/N]` and its runner has exited.

Lines already present when the watch starts are reported once too, so a crash that
happened before the watch began is not missed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

PROGRESS = re.compile(r"^\[(\d+)/(\d+)\] (\S+): (\S+)")
BAD = re.compile(r"Traceback|Error|Exception|Killed|MemoryError")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def scan(text: str) -> tuple[list[str], bool]:
    """Problem lines in a log, and whether it reached [N/N]."""
    problems, complete = [], False
    for line in text.splitlines():
        m = PROGRESS.match(line)
        if m:
            done, total, _run, status = m.groups()
            if status != "ok":
                problems.append(line)
            if done == total:
                complete = True
        elif BAD.search(line):
            problems.append(line)
    return problems, complete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="LOG or LOG:PID")
    ap.add_argument("--interval", type=float, default=60.0)
    args = ap.parse_args(argv)
    watched = []
    for t in args.targets:
        path, _, pid = t.partition(":")
        watched.append((Path(path), int(pid) if pid else None))
    while True:
        finished = True
        for path, pid in watched:
            text = path.read_text(errors="replace") if path.exists() else ""
            problems, complete = scan(text)
            if problems:
                print(f"ALERT {path}: {len(problems)} problem line(s):")
                print("\n".join(problems[:40]))
                return 1
            if pid is not None:
                if not _alive(pid) and not complete:
                    print(f"ALERT {path}: runner PID {pid} exited before the log reached [N/N]")
                    print("\n".join(text.splitlines()[-15:]))
                    return 1
                finished = finished and complete and not _alive(pid)
        if finished and any(pid is not None for _, pid in watched):
            print("all watched runners finished cleanly")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
