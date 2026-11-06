"""Chains of server rounds: a plan's optional `"then": "<plan name>"` names the plan that runs
after it (docs/gomea/SERVER-RUN.md, "Round 5"). Standard library only.

    python3 tools/server/chain.py COMMAND PLAN_FILE

COMMAND is one of (the shell scripts use them):

* `lock`: the path of the chain's lock file. Each orchestrator of the chain holds it for its
  whole life, so two orchestrators of one chain cannot run at the same time;
* `stop-marker`: the path of the file that stop.sh writes. A chained start reads it;
* `live`: one line "PID PGID PLAN_FILE" for each live orchestrator of the chain;
* `target`: the first plan from PLAN_FILE on that is not DONE (nothing if all are DONE);
* `forward`: PLAN_FILE and each plan after it, one per line;
* `state-dirs`: the state directories of `forward`, one per line.

A chain is known from the plan files only. Its root is the plan that no named plan points to.
The lock and the marker live in the root's state directory, so all plans of a chain use the
same ones. A plan without `then` and without a predecessor is a chain of one plan.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO.parent / "gomea-results"
SERVER = REPO / "tools" / "server"

# The plan names of _common.sh (tests/server/test_chain.py checks that they agree).
PLAN_FILES = {
    "round": "plan.json", "smoke": "plan-smoke.json",
    "round2": "plan2.json", "smoke2": "plan2-smoke.json",
    "round3": "plan3.json", "smoke3": "plan3-smoke.json",
    "round4": "plan4.json", "smoke4": "plan4-smoke.json",
    "round5": "plan5.json", "smoke5": "plan5-smoke.json",
    "round7": "plan7.json", "smoke7": "plan7-smoke.json",
    "round8": "plan8.json", "smoke8": "plan8-smoke.json",
}


def _load(plan: Path) -> dict:
    return json.loads(Path(plan).read_text())


def resolve(name: str) -> Path | None:
    """The plan file of a plan name, `None` when the name or the file does not exist."""
    f = PLAN_FILES.get(name)
    if f is None or not (SERVER / f).is_file():
        return None
    return SERVER / f


def name_of(plan: Path) -> str | None:
    real = Path(plan).resolve()
    for name, f in PLAN_FILES.items():
        if (SERVER / f).resolve() == real:
            return name
    return None


def next_plan(plan: Path) -> Path | None:
    then = _load(plan).get("then")
    return resolve(then) if then else None


def forward(plan: Path) -> list[Path]:
    """`plan` and each plan after it (a cycle ends at the first repeated plan)."""
    out, seen = [], set()
    cur: Path | None = Path(plan)
    while cur is not None and cur.resolve() not in seen:
        out.append(cur)
        seen.add(cur.resolve())
        cur = next_plan(cur)
    return out


def _previous(plan: Path) -> Path | None:
    real = Path(plan).resolve()
    for name in PLAN_FILES:
        p = resolve(name)
        if p is not None and p.resolve() != real:
            n = next_plan(p)
            if n is not None and n.resolve() == real:
                return p
    return None


def root(plan: Path) -> Path:
    cur, seen = Path(plan), {Path(plan).resolve()}
    while True:
        p = _previous(cur)
        if p is None or p.resolve() in seen:
            return cur
        cur = p
        seen.add(p.resolve())


def members(plan: Path) -> list[Path]:
    return forward(root(plan))


def state_dir(plan: Path) -> Path:
    return RES / _load(plan).get("state_dir", "server-run")


def lock_path(plan: Path) -> Path:
    return state_dir(root(plan)) / "chain.lock"


def stop_marker(plan: Path) -> Path:
    return state_dir(root(plan)) / "chain.stop"


def is_done(plan: Path) -> bool:
    return (state_dir(plan) / "DONE").exists()


def target(plan: Path) -> Path | None:
    """The first plan from `plan` on that is not DONE."""
    return next((p for p in forward(plan) if not is_done(p)), None)


def live_pid(plan: Path) -> dict | None:
    """The record of `orchestrator.pid` when that orchestrator is alive."""
    try:
        d = json.loads((state_dir(plan) / "orchestrator.pid").read_text())
        pid = int(d["pid"])
        os.kill(pid, 0)
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return d if "orchestrate.py" in cmd else None


def live(plan: Path) -> list[tuple[Path, dict]]:
    """The live orchestrators of the chain of `plan`."""
    out = []
    for p in members(plan):
        d = live_pid(p)
        if d is not None:
            out.append((p, d))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, plan = argv[0], Path(argv[1])
    if cmd == "lock":
        print(lock_path(plan))
    elif cmd == "stop-marker":
        print(stop_marker(plan))
    elif cmd == "live":
        for p, d in live(plan):
            print(d["pid"], d["pgid"], p)
    elif cmd == "target":
        t = target(plan)
        if t is not None:
            print(t.resolve())
    elif cmd == "forward":
        for p in forward(plan):
            print(p)
    elif cmd == "state-dirs":
        for p in forward(plan):
            print(state_dir(p))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
