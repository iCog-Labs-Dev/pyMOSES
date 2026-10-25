"""Unattended runner of the server round (SERVER-ROUND-PREREGISTRATION.md; how to use it:
docs/gomea/SERVER-RUN.md). Normally started by tools/server/start.sh, which detaches it.

    .venv/bin/python tools/server/orchestrate.py [--plan PLAN.json]     # run / resume
    .venv/bin/python tools/server/orchestrate.py [--plan ..] --status   # what status.sh prints
    .venv/bin/python tools/server/orchestrate.py [--plan ..] --bundle   # write bundle-latest

The plan (tools/server/plan.json) lists the steps in order and names the state directory
(`state_dir`, under ../gomea-results). Everything the round records goes there:

* `run.log`: human-readable, timestamped;
* `events.jsonl`: one JSON object per line (`time`, `event`, ...), appended at once:
  round_start / round_resume (environment and resources), step_start, step_end, step_skip
  (reason), workers_decision, phase_attempt, run_end (every finished run with its full
  summary, deduplicated across restarts), report and verdict (every row of
  report-stage{1,2}/contrasts.csv), pilot_p, pytest, repro, resources (every 5 min),
  scoreboard (every 10 min), bundle, error, stopped, round_end;
* `state.json`: per step its status (running, done, failed, skipped) and details, so a
  restart skips finished steps and resumes the current one (the runners skip runs that
  already have a summary.json); RETRY_FAILED=1 reruns failed and skipped steps;
* `scores-latest.txt`, `scores-history.txt`: tools/gomea/scores_now.py every 10 min;
* `bundle-latest.tar.gz` every hour and at the end, plus `bundle-final-<UTC>.tar.gz`,
  `round-summary.md` and `DONE` at the end.

Step types: `check` (S0: pytest and the reproducibility check; failures are flagged, never
fatal) and `phase` (optionally `requires: ["gpgomea"]`, and a `pilot` phase plus an
`after_pilot` command that must succeed before the phase). A phase runs as
`WORKERS=W tools/gomea/phase_run.sh PHASE >> ../gomea-results/PHASE.log`, W from
resources.py at the phase's start; a non-zero exit is retried once, then the step is
marked failed and the round goes on. A step whose requirement or configuration is missing
is skipped with the reason logged.

Single instance (flock on `orchestrator.lock`); `orchestrator.pid` records the PID and the
process group, which stop.sh signals. The orchestrator never commits or pushes.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resources  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO.parent / "gomea-results"
PY = REPO / ".venv" / "bin" / "python"
DEFAULT_PLAN = REPO / "tools" / "server" / "plan.json"

POLL_S = int(os.environ.get("ORCH_POLL_S", 60))
RESOURCES_S = int(os.environ.get("ORCH_RESOURCES_S", 300))
SCORES_S = int(os.environ.get("ORCH_SCORES_S", 600))
BUNDLE_S = int(os.environ.get("ORCH_BUNDLE_S", 3600))
BUNDLE_SKIP = {"moses.log", "train_scaled.csv"}
RUN_FILES = {"summary.json", "manifest.json", "events.jsonl", "environment.json",
             "run_order.json", "seeds.json", "lineage.json", "reseed.json",
             "longrand-p.json"}


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def since(stamp: str) -> int:
    """Seconds from a `utc()` stamp to now (wall clock, so it includes any downtime)."""
    t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return round((datetime.now(UTC) - t).total_seconds())


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Stopped(Exception):
    pass


class Round:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self.plan = json.loads(plan_path.read_text())
        self.sd = RES / self.plan.get("state_dir", "server-run")
        self.sd.mkdir(parents=True, exist_ok=True)
        self.events_path = self.sd / "events.jsonl"
        self.log_path = self.sd / "run.log"
        self.state_path = self.sd / "state.json"
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else {"plan": str(plan_path), "steps": {}})
        self.phases: list[str] = []
        for s in self.plan["steps"]:
            for key in ("pilot", "phase"):
                if s.get(key) and s[key] not in self.phases:
                    self.phases.append(s[key])
        self.seen: set[str] = set()
        self.current: dict = {}

    # ---- logging
    def log(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        with self.lock:
            with self.log_path.open("a") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def event(self, name: str, **kw) -> None:
        rec = {"time": utc(), "event": name, **kw}
        text = json.dumps(rec, default=str)
        with self.lock:
            with self.events_path.open("a") as f:
                f.write(text + "\n")
                f.flush()
                os.fsync(f.fileno())

    def save_state(self) -> None:
        with self.lock:
            _atomic_write(self.state_path, json.dumps(self.state, indent=1, default=str) + "\n")

    def step_state(self, sid: str) -> dict:
        return self.state["steps"].setdefault(sid, {})

    # ---- dedupe keys from earlier sessions
    def load_seen(self) -> None:
        if not self.events_path.exists():
            return
        with self.events_path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("key"):
                    self.seen.add(e["key"])

    # ---- runs and reports
    def phase_of_bool_dir(self, d: Path) -> str | None:
        best = None
        for p in self.phases:
            if d.name.startswith(p + "-") and (best is None or len(p) > len(best)):
                best = p
        return best

    def summary_files(self, phase: str | None = None) -> list[tuple[str, Path]]:
        out = []
        for p in self.phases if phase is None else [phase]:
            for f in glob.glob(str(RES / p / "*" / "runs" / "*" / "summary.json")):
                out.append((p, Path(f)))
        for d in glob.glob(str(RES / "1" / "*")):
            p = self.phase_of_bool_dir(Path(d))
            if p is not None and (phase is None or p == phase):
                for f in glob.glob(str(Path(d) / "runs" / "*" / "summary.json")):
                    out.append((p, Path(f)))
        return sorted(out)

    def scan(self) -> None:
        with self.lock:
            self._scan()

    def _scan(self) -> None:
        for phase, f in self.summary_files():
            key = "run:" + str(f.relative_to(RES))
            if key in self.seen:
                continue
            try:
                s = json.loads(f.read_text())
            except (OSError, ValueError):
                continue  # half-written; next poll
            self.event("run_end", key=key, phase=phase, path=str(f.relative_to(RES)), summary=s)
            self.seen.add(key)
        for phase in self.phases:
            for stage in (1, 2):
                rd = RES / phase / f"report-stage{stage}"
                c = rd / "contrasts.csv"
                if not c.exists():
                    continue
                try:
                    text = c.read_text()
                    rows = list(csv.DictReader(io.StringIO(text)))
                except (OSError, ValueError):
                    continue
                rkey = f"report:{c.relative_to(RES)}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
                if rkey not in self.seen:
                    verdict = None
                    try:
                        verdict = json.loads((rd / "verdict.json").read_text())
                    except (OSError, ValueError):
                        pass
                    self.event("report", key=rkey, phase=phase, stage=stage,
                               path=str(c.relative_to(RES)), verdicts=verdict, n_rows=len(rows))
                    self.seen.add(rkey)
                    self.log(f"report {phase} stage {stage}: {verdict}")
                for row in rows:
                    vkey = (f"verdict:{c.relative_to(RES)}:"
                            + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16])
                    if vkey not in self.seen:
                        self.event("verdict", key=vkey, phase=phase, stage=stage, **row)
                        self.seen.add(vkey)

    # ---- snapshots
    def resource_snapshot(self, pgid: int | None = None) -> dict:
        snap: dict = {}
        try:
            snap["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
        except OSError:
            pass
        mi = resources.meminfo(Path("/proc"))
        snap["mem_available_mb"] = mi.get("MemAvailable", 0) // 2**20 or None
        cur = resources.cgroup_mem_current()
        snap["cgroup_mem_current_mb"] = cur // 2**20 if cur is not None else None
        du = shutil.disk_usage(RES)
        snap["disk_free_gb"] = round(du.free / 2**30, 1)
        snap.update(self.process_counts(pgid))
        return snap

    @staticmethod
    def process_counts(pgid: int | None = None) -> dict:
        """Processes of the round's process group (default: ours) by kind."""
        me = os.getpgrp() if pgid is None else pgid
        group = moses = gpg = runners = 0
        for d in glob.glob("/proc/[0-9]*"):
            try:
                pid = int(d.rsplit("/", 1)[1])
                if os.getpgid(pid) != me:
                    continue
                cmd = Path(d, "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, ValueError):
                continue
            group += 1
            if "moses_exec" in cmd or "multiprocessing.spawn" in cmd:
                moses += 1
            if "gpgomea" in cmd and "python" not in cmd:
                gpg += 1
            if re.search(r"benchmarks\.gomea\.runner|moses_sr_runner\.py|gpgomea_sr_runner\.py", cmd) \
                    and "multiprocessing" not in cmd:
                runners += 1
        return {"procs_in_group": group, "moses_procs": moses, "gpgomea_procs": gpg,
                "runner_procs": runners}

    def scoreboard(self) -> None:
        try:
            r = subprocess.run([str(PY), "tools/gomea/scores_now.py", *self.phases], cwd=REPO,
                               capture_output=True, text=True, timeout=600)
            text = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.returncode else "")
        except (OSError, subprocess.TimeoutExpired) as exc:
            text = f"scores_now.py failed: {exc}\n"
        stamp = utc()
        _atomic_write(self.sd / "scores-latest.txt", f"# {stamp}\n{text}")
        with self.lock:
            with (self.sd / "scores-history.txt").open("a") as f:
                f.write(f"\n===== {stamp} =====\n{text}")
        self.event("scoreboard", n_lines=text.count("\n"))

    def bundle_async(self) -> None:
        """Hourly bundle in a niced child, so it never delays the main loop."""
        try:
            subprocess.Popen(["nice", "-n", "10", str(PY), str(Path(__file__)), "--plan",
                              str(self.plan_path), "--bundle"], cwd=REPO,
                             stdout=subprocess.DEVNULL, stderr=open(self.sd / "bundle.err", "a"))
        except OSError as exc:
            self.event("error", where="bundle", error=str(exc))

    def monitor(self) -> None:
        last = {"res": 0.0, "scores": time.time() - SCORES_S + 120, "bundle": time.time()}
        while not self.stop.is_set():
            try:
                self.scan()
                t = time.time()
                if t - last["res"] >= RESOURCES_S:
                    last["res"] = t
                    self.event("resources", current=self.current, **self.resource_snapshot())
                if t - last["scores"] >= SCORES_S:
                    last["scores"] = t
                    self.scoreboard()
                if t - last["bundle"] >= BUNDLE_S:
                    last["bundle"] = t
                    self.bundle_async()
            except Exception as exc:  # the monitor must never die
                self.event("error", where="monitor", error=f"{type(exc).__name__}: {exc}")
            self.stop.wait(POLL_S)

    # ---- steps
    def workers(self, phase: str) -> int:
        try:
            r = subprocess.run([str(PY), "tools/server/resources.py"], cwd=REPO,
                               capture_output=True, text=True, timeout=60)
            d = json.loads(r.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            d = {"workers": 1, "error": str(exc)}
        self.event("workers_decision", phase=phase, **d)
        self.log(f"{phase}: workers W={d['workers']} (limited by {d.get('limited_by')}; "
                 f"usable CPUs {d.get('usable_cpus')}, memory {d.get('mem_mb')} MB)")
        return int(d["workers"])

    def run_counts(self, phase: str) -> dict:
        ok = bad = 0
        for _, f in self.summary_files(phase):
            try:
                s = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if s.get("status") == "ok":
                ok += 1
            else:
                bad += 1
        return {"runs_ok": ok, "runs_failed": bad}

    def run_phase(self, phase: str, st: dict) -> int:
        """Run one phase through phase_run.sh, retrying once on a non-zero exit."""
        w = self.workers(phase)
        st.setdefault("workers", []).append(w)
        rc = None
        for attempt in (1, 2):
            st["attempts"] = st.get("attempts", 0) + 1
            self.save_state()
            logp = RES / f"{phase}.log"
            with logp.open("a") as lf:
                lf.write(f"\n=== orchestrator: {phase} attempt {attempt}, WORKERS={w}, {utc()}\n")
            self.event("phase_attempt", phase=phase, attempt=attempt, workers=w, log=str(logp))
            self.log(f"{phase}: attempt {attempt} (log {logp})")
            self.current = {"phase": phase, "attempt": attempt, "workers": w}
            with logp.open("a") as lf:
                p = subprocess.Popen(["bash", "tools/gomea/phase_run.sh", phase], cwd=REPO,
                                     stdout=lf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "WORKERS": str(w)})
                rc = p.wait()
            self.scan()
            self.log(f"{phase}: attempt {attempt} exited rc={rc}")
            if rc == 0:
                break
            self.event("error", where=phase, error=f"phase_run.sh exited {rc} (attempt {attempt})")
        self.current = {}
        return rc

    def set_commit(self, step: dict) -> None:
        """`set_commit_in`: Boolean manifests whose `commit` is set to the current HEAD before
        the phase runs (the smoke plan's, committed as "SET_AT_RUNTIME"), so the runner's
        code check passes on whatever commit runs them. Rewrites them in the working tree."""
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
        for rel in step.get("set_commit_in", []):
            p = REPO / rel
            m = json.loads(p.read_text())
            if m.get("commit") != head:
                old, m["commit"] = m.get("commit"), head
                p.write_text(json.dumps(m, indent=2) + "\n")
                self.event("set_commit", path=rel, old=old, new=head)
                self.log(f"{step['id']}: set commit of {rel} to HEAD {head}")

    def requirement(self, req: str) -> str | None:
        """None when met, else the reason it is not."""
        if req == "gpgomea":
            j = RES / "server-run" / "gpgomea-build.json"
            try:
                d = json.loads(j.read_text())
            except (OSError, ValueError):
                return f"GP-GOMEA not built ({j} missing or unreadable; run setup.sh)"
            if not d.get("ok"):
                return f"GP-GOMEA build failed ({j}: {d.get('error')})"
            for b in (RES / "phase1b/gpgomea-build/gpgomea", RES / "gpgomea-seed-build/gpgomea-seed"):
                if not os.access(b, os.X_OK):
                    return f"GP-GOMEA binary {b} missing"
            return None
        return f"unknown requirement {req!r}"

    def skip(self, sid: str, st: dict, reason: str) -> None:
        st.update(status="skipped", reason=reason, end=utc())
        self.save_state()
        self.event("step_skip", step=sid, reason=reason)
        self.log(f"SKIP {sid}: {reason}")

    def do_phase(self, step: dict, st: dict) -> None:
        sid, phase = step["id"], step["phase"]
        for req in step.get("requires", []):
            why = self.requirement(req)
            if why:
                return self.skip(sid, st, why)
        for p in [step.get("pilot"), phase]:
            if p and not (REPO / "configs" / "gomea" / p / "spec.json").exists():
                return self.skip(sid, st, f"no configs/gomea/{p}/spec.json in this checkout")
        if step.get("pilot"):
            pst = st.setdefault("pilot", {})
            if pst.get("status") not in ("done", "failed"):
                self.event("step_start", step=f"{sid}:pilot", phase=step["pilot"])
                pst.setdefault("start", utc())
                rc = self.run_phase(step["pilot"], pst)
                pst.update(status="done" if rc == 0 else "failed", rc=rc,
                           duration_s=since(pst["start"]), **self.run_counts(step["pilot"]))
                self.save_state()
                self.event("step_end", step=f"{sid}:pilot", phase=step["pilot"], **pst)
            if pst["status"] != "done":
                return self.skip(sid, st, f"pilot {step['pilot']} failed (rc={pst.get('rc')})")
            if step.get("after_pilot"):
                ast = st.setdefault("after_pilot", {})
                if ast.get("status") != "done":
                    argv = [a.format(repo=REPO, results=RES) for a in step["after_pilot"]]
                    try:
                        r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=3600)
                        rc, out, err = r.returncode, r.stdout, r.stderr
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        rc, out, err = -1, "", f"{type(exc).__name__}: {exc}"
                    ast.update(status="done" if rc == 0 else "failed", rc=rc, argv=argv,
                               stdout=out[-20000:], stderr=err[-5000:])
                    self.save_state()
                    self.event("pilot_p", step=sid, argv=argv, rc=rc, stdout=out[-20000:],
                               stderr=err[-5000:])
                    self.log(f"{sid}: after-pilot command rc={rc}: {out.strip()[-2000:]} {err.strip()[-500:]}")
                if ast["status"] != "done":
                    return self.skip(sid, st, f"after-pilot command failed (rc={ast.get('rc')}): "
                                     f"{(ast.get('stderr') or '').strip()[-300:]}")
        self.set_commit(step)
        rc = self.run_phase(phase, st)
        st.update(status="done" if rc == 0 else "failed", rc=rc, end=utc(),
                  **self.run_counts(phase))
        self.save_state()

    def do_check(self, step: dict, st: dict) -> None:
        sid = step["id"]
        flags = []
        w = self.workers(sid)
        procs = {}
        pcfg = step.get("pytest", {})
        if "pytest" not in st and not pcfg.get("skip"):
            argv = [str(PY), "-m", "pytest", "-p", "no:cacheprovider", *pcfg.get("args", [])]
            logf = (self.sd / "pytest.log").open("w")
            procs["pytest"] = (subprocess.Popen(argv, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT),
                               logf, time.time(), argv)
            self.log(f"S0: pytest started: {' '.join(argv[1:])}")
        elif pcfg.get("skip"):
            st["pytest"] = {"skipped": True}
            flags.append("pytest skipped by the plan")
        rcfg = step.get("repro", {})
        if "repro" not in st and not rcfg.get("skip"):
            only = os.environ.get("REPRO_ONLY") or ",".join(rcfg.get("only") or [])
            argv = [str(PY), "tools/server/repro_check.py", "--out", str(self.sd / "repro"),
                    "--report", str(self.sd / "repro.json"), "--workers", str(min(w, 4))]
            argv += ["--only", only] if only else []
            logf = (self.sd / "repro.log").open("a")
            procs["repro"] = (subprocess.Popen(argv, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT),
                              logf, time.time(), argv)
            self.log(f"S0: reproducibility check started ({only or 'all four runs'})")
        elif rcfg.get("skip"):
            st["repro"] = {"skipped": True}
            flags.append("repro check skipped by the plan")
        self.current = {"step": sid, "running": list(procs)}
        for name, (p, logf, t0, argv) in procs.items():
            rc = p.wait()
            logf.close()
            dur = round(time.time() - t0)
            if name == "pytest":
                st["pytest"] = res = {"rc": rc, "duration_s": dur, "args": argv[3:],
                                      **self.pytest_counts(self.sd / "pytest.log")}
                res["ok"] = rc == 0
                self.event("pytest", **res)
                self.log(f"S0: pytest rc={rc} {res}")
            else:
                try:
                    rep = json.loads((self.sd / "repro.json").read_text())
                except (OSError, ValueError):
                    rep = {"ok": False, "error": "no repro.json"}
                st["repro"] = {"rc": rc, "duration_s": dur, "ok": rep.get("ok"),
                               "n_match": rep.get("n_match"), "n_selected": rep.get("n_selected"),
                               "match": {k: v.get("match") for k, v in rep.get("runs", {}).items()}}
                self.event("repro", rc=rc, duration_s=dur, report=rep)
                self.log(f"S0: reproducibility check rc={rc}: {st['repro']['match']}")
            self.save_state()
        pt, rp = st.get("pytest", {}), st.get("repro", {})
        if not pt.get("skipped") and not pt.get("ok"):
            flags.append(f"pytest failed ({pt.get('failed')} failed, {pt.get('errors')} errors)")
        if pcfg.get("args"):
            flags.append(f"pytest limited by the plan to {pcfg['args']}")
        if not rp.get("skipped") and not rp.get("ok"):
            flags.append("reproducibility MISMATCH (cross-machine comparisons invalid)")
        if (os.environ.get("REPRO_ONLY") or rcfg.get("only")) and not rp.get("skipped"):
            flags.append("repro check limited to a subset")
        self.current = {}
        st.update(status="done", flags=flags, end=utc())
        self.save_state()

    @staticmethod
    def pytest_counts(logp: Path) -> dict:
        try:
            tail = logp.read_text()[-5000:]
        except OSError:
            return {}
        line = next((ln for ln in reversed(tail.splitlines())
                     if re.search(r"\d+ (passed|failed|error)", ln)), "")
        out = {"summary_line": line.strip("= ").strip()}
        for k in ("passed", "failed", "skipped", "errors", "error", "xfailed", "xpassed", "deselected"):
            m = re.search(rf"(\d+) {k}\b", line)
            if m:
                out["errors" if k == "error" else k] = int(m.group(1))
        return out

    def run(self) -> int:
        fresh = not self.state["steps"]
        self.load_seen()
        try:
            env = __import__("envinfo").collect()
        except Exception as exc:
            env = {"error": f"{type(exc).__name__}: {exc}"}
        self.state.setdefault("round_started", utc())
        self.state["sessions"] = self.state.get("sessions", 0) + 1
        self.save_state()
        self.event("round_start" if fresh else "round_resume", plan=self.plan,
                   session=self.state["sessions"], environment=env,
                   pid=os.getpid(), pgid=os.getpgrp())
        self.log(f"{'start' if fresh else 'resume'} round {self.plan.get('name')} "
                 f"(session {self.state['sessions']}, plan {self.plan_path}, state {self.sd})")
        mon = threading.Thread(target=self.monitor, daemon=True)
        mon.start()
        retry = os.environ.get("RETRY_FAILED") == "1"
        for step in self.plan["steps"]:
            sid = step["id"]
            st = self.step_state(sid)
            if st.get("status") == "done" or (st.get("status") in ("failed", "skipped") and not retry):
                self.log(f"{sid}: already {st['status']}, not rerun")
                continue
            if st.get("status") == "running":
                self.log(f"{sid}: resuming (was running when the last session ended)")
            st.update(status="running", start=st.get("start") or utc())
            st.pop("reason", None)
            self.save_state()
            self.event("step_start", step=sid, type=step["type"], resumed=st.get("attempts", 0) > 0)
            t0 = time.time()
            try:
                if step["type"] == "check":
                    self.do_check(step, st)
                elif step["type"] == "phase":
                    self.do_phase(step, st)
                else:
                    self.skip(sid, st, f"unknown step type {step['type']!r}")
            except Stopped:
                raise
            except Exception as exc:
                st.update(status="failed", error=f"{type(exc).__name__}: {exc}", end=utc())
                self.save_state()
                self.event("error", where=sid, error=st["error"])
                self.log(f"{sid}: ERROR {st['error']}")
            if st["status"] != "skipped":
                st["duration_s"] = since(st["start"])
                self.save_state()
                self.event("step_end", step=sid, rc=st.get("rc"), status=st["status"],
                           duration_s=st["duration_s"], session_s=round(time.time() - t0),
                           workers=st.get("workers"), runs_ok=st.get("runs_ok"),
                           runs_failed=st.get("runs_failed"), flags=st.get("flags"))
                self.log(f"{sid}: {st['status']} (rc={st.get('rc')}, {st['duration_s']} s since "
                         f"the step first started)")
        self.finish()
        return 0

    def finish(self) -> None:
        self.scan()
        self.stop.set()
        self.scoreboard()
        self.state["round_ended"] = utc()
        self.save_state()
        self.write_summary()
        self.event("round_end", steps={k: v.get("status") for k, v in self.state["steps"].items()})
        self.log("round finished; writing the final bundle")
        b = make_bundle(self)
        final = self.sd / f"bundle-final-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
        shutil.copyfile(b, final)
        self.event("bundle", path=str(final), bytes=final.stat().st_size, final=True)
        (self.sd / "DONE").write_text(utc() + "\n")
        self.log(f"DONE. Bring back {final}")

    def write_summary(self) -> None:
        events = []
        if self.events_path.exists():
            for line in self.events_path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
        env = next((e.get("environment") for e in events if e["event"] == "round_start"), None) or {}
        L = [f"# Server round summary: {self.plan.get('name')}", "",
             f"Started {self.state.get('round_started')}, ended {self.state.get('round_ended')}, "
             f"{self.state.get('sessions')} orchestrator session(s). Plan `{self.plan_path.name}`, "
             f"state `{self.sd}`.", "", "## Environment", ""]
        res = env.get("resources", {})
        L += [f"* host `{env.get('hostname')}`, {env.get('os_release', {}).get('PRETTY_NAME')}, "
              f"kernel {env.get('kernel')}",
              f"* CPU {env.get('cpu_model')}: {env.get('logical_cpus')} logical, "
              f"{env.get('physical_cores_in_affinity')} physical in affinity, cgroup quota "
              f"{env.get('cgroup_cpu_quota')}",
              f"* memory {env.get('memtotal_mb')} MB total, cgroup limit "
              f"{env.get('cgroup_mem_limit_mb')} MB; workers decision {res.get('workers')} "
              f"(limited by {res.get('limited_by')})",
              f"* Python {(env.get('python') or '').split()[0] if env.get('python') else None}, "
              f"git HEAD `{env.get('git_head')}`, dirty: "
              f"{bool((env.get('git_status_porcelain') or '').strip())}", ""]
        L += ["## Steps", "", "| step | status | duration | workers | runs ok | runs failed | notes |",
              "|---|---|---|---|---|---|---|"]
        for step in self.plan["steps"]:
            st = self.state["steps"].get(step["id"], {})
            notes = st.get("reason") or "; ".join(st.get("flags") or []) or st.get("error") or ""
            if st.get("pilot"):
                p = st["pilot"]
                notes = (f"pilot {p.get('status')} ({p.get('runs_ok')} ok, {p.get('runs_failed')} failed); "
                         + notes)
            dur = st.get("duration_s")
            L.append(f"| {step['id']} | {st.get('status', 'not run')} | "
                     f"{f'{dur / 3600:.2f} h' if dur is not None else ''} | {st.get('workers', '')} | "
                     f"{st.get('runs_ok', '')} | {st.get('runs_failed', '')} | {notes} |")
        s0 = next((self.state["steps"].get(s["id"], {}) for s in self.plan["steps"]
                   if s["type"] == "check"), {})
        L += ["", "## S0", "", f"* pytest: {s0.get('pytest')}", f"* reproducibility: {s0.get('repro')}"]
        pp = [e for e in events if e["event"] == "pilot_p"]
        if pp:
            L += ["", "## Pilot", ""] + [f"* `{' '.join(e['argv'][1:])}` rc={e['rc']}:\n\n```\n"
                                           f"{e['stdout'].strip()[-3000:]}\n```" for e in pp]
        L += ["", "## Verdicts (contrasts.csv)", ""]
        for phase in self.phases:
            for stage in (1, 2):
                c = RES / phase / f"report-stage{stage}" / "contrasts.csv"
                if not c.exists():
                    continue
                rows = list(csv.DictReader(io.StringIO(c.read_text())))
                L += [f"### {phase} stage {stage}", "",
                      "| problem | contrast | family | a lower | b lower | Holm p | verdict |",
                      "|---|---|---|---|---|---|---|"]
                for r in rows:
                    L.append(f"| {r.get('problem')} | {r.get('a')} vs {r.get('b')} | {r.get('family')} | "
                             f"{r.get('a_lower')} | {r.get('b_lower')} | {r.get('p_holm')} | "
                             f"{r.get('verdict')} |")
                L.append("")
        skips = [e for e in events if e["event"] == "step_skip"]
        if skips:
            L += ["## Skips", ""] + [f"* {e['step']}: {e['reason']}" for e in skips] + [""]
        errs = [e for e in events if e["event"] == "error"]
        if errs:
            L += ["## Errors", ""] + [f"* {e['time']} {e.get('where')}: {e.get('error')}"
                                      for e in errs[-50:]] + [""]
        n_runs = sum(e["event"] == "run_end" for e in events)
        L += [f"{n_runs} run_end events, {len(events)} events in total."]
        _atomic_write(self.sd / "round-summary.md", "\n".join(L) + "\n")


def make_bundle(r: Round) -> Path:
    """bundle-latest.tar.gz: server-run/ (no bundles), every summary/manifest/events file
    of the round's phases, the report dirs, the phase logs and the git HEAD."""
    out = r.sd / "bundle-latest.tar.gz"
    tmp = r.sd / f".bundle-{os.getpid()}.tar.gz.tmp"
    added: set[str] = set()

    def add(tf: tarfile.TarFile, p: Path) -> None:
        arc = "gomea-results/" + str(p.relative_to(RES))
        if arc in added or not p.is_file():
            return
        added.add(arc)
        try:
            tf.add(p, arcname=arc, recursive=False)
        except OSError:
            pass

    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for dirpath, dirnames, files in os.walk(r.sd):
            for f in files:
                if f.startswith(("bundle-", ".bundle-")) or f.endswith(".tmp") or f in BUNDLE_SKIP:
                    continue
                add(tf, Path(dirpath) / f)
        for p in [RES / "server-run" / n for n in ("environment.json", "setup.log", "pip-freeze.txt",
                                                   "gpgomea-build.json", "gpgomea-smoke.json")]:
            add(tf, p)
        dirs = [RES / ph for ph in r.phases]
        dirs += [Path(d) for d in glob.glob(str(RES / "1" / "*")) if r.phase_of_bool_dir(Path(d))]
        for d in dirs:
            for dirpath, dirnames, files in os.walk(d):
                rep = "/report-stage" in dirpath + "/"
                for f in files:
                    if f in BUNDLE_SKIP:
                        continue
                    if rep or f in RUN_FILES:
                        add(tf, Path(dirpath) / f)
        for ph in r.phases:
            for f in glob.glob(str(RES / f"{ph}*.log")):
                add(tf, Path(f))
        git = "\n".join(
            subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout
            for a in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"],
                      ["git", "diff", "--stat"]))
        info = tarfile.TarInfo("gomea-results/git-head.txt")
        data = f"repo {REPO}\nbundled {utc()}\n{git}".encode()
        info.size, info.mtime = len(data), int(time.time())
        tf.addfile(info, io.BytesIO(data))
    os.replace(tmp, out)
    r.event("bundle", path=str(out), bytes=out.stat().st_size, files=len(added) + 1)
    return out


def status(r: Round) -> int:
    pidf = r.sd / "orchestrator.pid"
    alive = False
    info = {}
    if pidf.exists():
        try:
            info = json.loads(pidf.read_text())
            os.kill(int(info["pid"]), 0)
            alive = "orchestrate.py" in Path(f"/proc/{info['pid']}/cmdline").read_bytes().decode(errors="replace")
        except (OSError, ValueError, KeyError):
            alive = False
    done = (r.sd / "DONE").exists()
    print(f"round {r.plan.get('name')} (state {r.sd})")
    print(f"orchestrator: {'RUNNING pid ' + str(info.get('pid')) + ' pgid ' + str(info.get('pgid')) if alive else 'not running'}"
          f"{'; DONE ' + (r.sd / 'DONE').read_text().strip() if done else ''}")
    print("\nsteps:")
    for step in r.plan["steps"]:
        st = r.state["steps"].get(step["id"], {})
        extra = st.get("reason") or "; ".join(st.get("flags") or []) or ""
        w = st.get("workers")
        print(f"  {step['id']:16} {st.get('status', '-'):8} "
              f"{'W=' + str(w[-1]) if w else '':6} "
              f"{'ok ' + str(st.get('runs_ok')) + ' failed ' + str(st.get('runs_failed')) if 'runs_ok' in st else ''} {extra}")
    events = []
    if r.events_path.exists():
        with r.events_path.open() as f:
            events = [json.loads(x) for x in f.readlines()[-400:] if x.strip()]
    cur = [e for e in events if e["event"] in ("phase_attempt", "workers_decision")]
    if alive and cur:
        e = cur[-1]
        print(f"\ncurrent: {e.get('phase')} attempt {e.get('attempt', '?')} with W={e.get('workers')}")
    n_run_end = 0
    if r.events_path.exists():
        with r.events_path.open() as f:
            n_run_end = sum('"event": "run_end"' in x for x in f)
    print(f"\nrun_end events so far: {n_run_end}")
    print("\nlast events:")
    for e in [e for e in events if e["event"] not in ("resources",)][-15:]:
        brief = {k: v for k, v in e.items() if k not in ("time", "event", "summary", "environment",
                                                         "plan", "report", "key", "stdout")}
        if e["event"] == "run_end":
            s = e.get("summary", {})
            brief = {"phase": e.get("phase"), "run": s.get("run_id"), "status": s.get("status")}
        print(f"  {e['time']} {e['event']:16} {json.dumps(brief, default=str)[:150]}")
    print("\nresources now:", json.dumps(r.resource_snapshot(int(info["pgid"]) if alive else None)))
    d = resources.decide()
    print(f"workers decision now: {d['workers']} (limited by {d['limited_by']})")
    sl = r.sd / "scores-latest.txt"
    if sl.exists():
        print(f"\nscoreboard ({sl}):")
        print(sl.read_text()[-6000:])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    a = ap.parse_args()
    plan = Path(a.plan)
    if not plan.is_absolute():
        plan = (Path.cwd() / plan).resolve()
    r = Round(plan)
    if a.status:
        return status(r)
    if a.bundle:
        make_bundle(r)
        print(r.sd / "bundle-latest.tar.gz")
        return 0
    lockf = open(r.sd / "orchestrator.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another orchestrator holds {r.sd / 'orchestrator.lock'}; not starting", file=sys.stderr)
        return 3
    _atomic_write(r.sd / "orchestrator.pid", json.dumps(
        {"pid": os.getpid(), "pgid": os.getpgrp(), "started": utc(), "plan": str(plan)}) + "\n")

    def on_signal(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise Stopped(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        return r.run()
    except Stopped as exc:
        r.stop.set()
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)  # our children, if the signal was ours only
        except OSError:
            pass
        r.save_state()
        r.event("stopped", signal=str(exc), current=r.current)
        r.log(f"stopped by {exc}; rerun tools/server/start.sh to resume")
        return 143
    except Exception as exc:
        r.event("error", where="orchestrator", error=f"{type(exc).__name__}: {exc}")
        r.log(f"orchestrator crashed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
