"""The chain of server rounds (`"then"` in a plan; tools/server/chain.py), and the orchestrator's
active time. Each test runs the real start.sh, stop.sh, status.sh, orchestrate.py and autostart
block on a small copy of tools/server in a temporary directory, with a fake phase_run.sh that
writes a trace and waits for a file. Nothing touches ../gomea-results."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "tools" / "server"
TIMEOUT = 120

FAKE_PHASE = r"""#!/usr/bin/env bash
# args: PHASE. Writes the trace, waits for ../gomea-results/release-PHASE, exits with the
# content of ../gomea-results/rc-PHASE (default 0).
cd "$(dirname "$0")/../.."
R=../gomea-results
E="RETRY_FAILED=${RETRY_FAILED:-unset} REPRO_ONLY=${REPRO_ONLY:-unset}"
echo "begin $1 $(date +%s.%N) $E session=$(ps -o sid= $$ | tr -d ' ')" >> "$R/trace.log"
while [ ! -e "$R/release-$1" ]; do sleep 0.1; done
echo "end $1 $(date +%s.%N)" >> "$R/trace.log"
exit "$(cat "$R/rc-$1" 2>/dev/null || echo 0)"
"""


def _plan(state_dir, phases, then=None):
    p = {"name": state_dir, "state_dir": state_dir,
         "steps": [{"id": x, "type": "phase", "phase": x} for x in phases]}
    if then:
        p["then"] = then
    return p


class Rig:
    def __init__(self, root: Path):
        self.repo = root / "pyMoses"
        self.res = root / "gomea-results"
        self.res.mkdir(parents=True)
        tools = self.repo / "tools" / "server"
        tools.mkdir(parents=True)
        for f in SERVER.iterdir():
            if f.suffix in (".py", ".sh"):
                shutil.copy(f, tools / f.name)
        (self.repo / "tools" / "gomea").mkdir()
        (self.repo / "tools" / "gomea" / "phase_run.sh").write_text(FAKE_PHASE)
        (self.repo / ".venv" / "bin").mkdir(parents=True)
        (self.repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
        self.jupyter = root / "jupyter"
        self.plans = {"plan4.json": _plan("server-run4", ["p4"], "round5"),
                      "plan5.json": _plan("server-run5", ["p5"])}
        for name, plan in self.plans.items():
            self.write_plan(name, plan)
        for ph in ("p4", "p5"):
            d = self.repo / "configs" / "gomea" / ph
            d.mkdir(parents=True)
            (d / "spec.json").write_text("{}")
        self.sd4 = self.res / "server-run4"
        self.sd5 = self.res / "server-run5"

    def write_plan(self, name, plan):
        (self.repo / "tools" / "server" / name).write_text(json.dumps(plan))

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("RETRY_FAILED", "REPRO_ONLY", "PLAN", "ORCH_CHAINED", "ORCH_NO_CHAIN",
                            "MAX_WORKERS", "PYTEST_CURRENT_TEST")}
        return {**env, "ORCH_POLL_S": "1", "ORCH_SCORES_S": "100000", "ORCH_BUNDLE_S": "100000",
                "GRACE": "20", "JUPYTER_CONFIG_DIR": str(self.jupyter), **extra}

    def sh(self, script, *args, **env):
        return subprocess.run(["bash", str(self.repo / "tools" / "server" / script), *args],
                              cwd=self.repo, env=self.env(**env), capture_output=True, text=True,
                              timeout=TIMEOUT)

    def start(self, plan="round4", *args, **env):
        return self.sh("start.sh", "--plan", plan, *args, **env)

    def release(self, phase):
        (self.res / f"release-{phase}").write_text("")

    def trace(self):
        f = self.res / "trace.log"
        return f.read_text().splitlines() if f.exists() else []

    def begun(self, phase):
        return sum(x.startswith(f"begin {phase} ") for x in self.trace())

    def wait(self, what, cond):
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if cond():
                return
            time.sleep(0.1)
        raise AssertionError(f"timeout waiting for {what}\n{self.dump()}")

    def wait_begun(self, phase, n=1):
        self.wait(f"{phase} begun {n}", lambda: self.begun(phase) >= n)

    def wait_done(self, sd):
        self.wait(f"{sd}/DONE", lambda: (sd / "DONE").exists())

    def wait_gone(self, sd):
        self.wait(f"{sd} orchestrator gone", lambda: not self.live(sd))

    def dump(self):
        out = []
        for sd in (self.sd4, self.sd5):
            for n in ("run.log", "orchestrator.out", "chain-start.log"):
                if (sd / n).exists():
                    out.append(f"--- {sd.name}/{n}\n{(sd / n).read_text()[-2000:]}")
        return "\n".join(out + self.trace())

    def pid(self, sd):
        try:
            return json.loads((sd / "orchestrator.pid").read_text())
        except (OSError, ValueError):
            return None

    def live(self, sd):
        d = self.pid(sd)
        if d is None:
            return False
        try:
            cmd = Path(f"/proc/{d['pid']}/cmdline").read_bytes()
            state = Path(f"/proc/{d['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return False
        return b"orchestrate.py" in cmd and state != "Z"

    def events(self, sd):
        f = sd / "events.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def names(self, sd):
        return [e["event"] for e in self.events(sd)]

    def state(self, sd):
        return json.loads((sd / "state.json").read_text())

    def bundles(self, sd):
        return sorted(sd.glob("bundle-final-*.tar.gz"))

    def pod_restart(self):
        """All processes of the rig die, as in a pod restart, and nothing cleans up."""
        for sd in (self.sd4, self.sd5):
            d = self.pid(sd)
            if d is not None and self.live(sd):
                os.killpg(d["pgid"], signal.SIGKILL)
                self.wait("the kill", lambda s=sd: not self.live(s))

    def autostart_block(self):
        cfg = self.jupyter / "jupyter_server_config.py"
        exec(compile(cfg.read_text(), str(cfg), "exec"), {})  # what Jupyter does at its start

    def cleanup(self):
        for ph in ("p4", "p5"):
            self.release(ph)
        for sd in (self.sd4, self.sd5):
            d = self.pid(sd)
            if d is not None:
                try:
                    os.killpg(d["pgid"], signal.SIGKILL)
                except OSError:
                    pass


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.cleanup()


# ---- the normal chain


def test_the_rounds_run_one_after_the_other_with_separate_results(rig):
    assert rig.start().returncode == 0
    rig.wait_begun("p4")
    assert not rig.sd5.exists() or not (rig.sd5 / "run.log").exists()
    rig.release("p4")
    rig.wait_begun("p5")
    # round 4 is complete before round 5 begins
    assert (rig.sd4 / "DONE").exists() and (rig.sd4 / "round-summary.md").exists()
    assert len(rig.bundles(rig.sd4)) == 1
    done4 = datetime.fromisoformat((rig.sd4 / "DONE").read_text().strip().replace("Z", "+00:00"))
    rig.release("p5")
    rig.wait_done(rig.sd5)
    assert len(rig.bundles(rig.sd5)) == 1 and (rig.sd5 / "round-summary.md").exists()
    # the times do not overlap
    lines = [x.split() for x in rig.trace()]
    assert [(x[0], x[1]) for x in lines] == [
        ("begin", "p4"), ("end", "p4"), ("begin", "p5"), ("end", "p5")]
    assert float(lines[1][2]) < float(lines[2][2])
    start5 = next(e for e in rig.events(rig.sd5) if e["event"] == "round_start")
    assert start5["chained"] is True
    assert datetime.fromisoformat(start5["time"].replace("Z", "+00:00")) >= done4
    # each round's own bundle holds its own state only
    for sd, other in ((rig.sd4, "server-run5"), (rig.sd5, "server-run4")):
        with tarfile.open(rig.bundles(sd)[0]) as tf:
            names = tf.getnames()
        assert any(f"gomea-results/{sd.name}/state.json" == n for n in names)
        assert not [n for n in names if other in n]
    # the event is in the log of the plan that ended, after its DONE
    rig.wait("the last event of round 4", lambda: "chain_started" in rig.names(rig.sd4))
    ev = {e["event"]: e for e in rig.events(rig.sd4)}
    assert ev["chain_start"]["next"] == "round5"
    assert ev["chain_started"]["start_rc"] == 0
    names = rig.names(rig.sd4)
    assert names.index("round_end") < names.index("chain_start")
    assert "chain: started round5" in (rig.sd4 / "run.log").read_text()


def test_the_next_round_runs_in_a_new_session(rig):
    rig.start()
    rig.wait_begun("p4")
    rig.release("p4")
    rig.wait_begun("p5")
    sessions = [re.search(r"session=(\d+)", x).group(1)
                for x in rig.trace() if x.startswith("begin")]
    assert sessions[0] != sessions[1]
    rig.wait_gone(rig.sd4)
    assert rig.live(rig.sd5)


def test_a_plan_without_then_starts_nothing(rig):
    rig.write_plan("plan4.json", _plan("server-run4", ["p4"]))
    rig.release("p4")
    assert rig.start().returncode == 0
    rig.wait_done(rig.sd4)
    time.sleep(1)
    assert "chain_start" not in rig.names(rig.sd4) and rig.begun("p5") == 0
    assert not (rig.sd4 / "chain.stop").exists()


# ---- a step that fails or is skipped


def test_a_failed_step_goes_on_and_the_log_says_so(rig):
    (rig.res / "rc-p4").write_text("1")
    rig.release("p4")
    rig.start()
    rig.wait_begun("p5")
    assert rig.state(rig.sd4)["steps"]["p4"]["status"] == "failed"
    log = (rig.sd4 / "run.log").read_text()
    assert "steps not done {'p4': 'failed'}" in log and "the chain goes on to round5" in log


def test_a_skipped_step_goes_on_and_the_log_says_so(rig):
    plan = _plan("server-run4", ["p4"], "round5")
    plan["steps"][0]["requires"] = ["knob-cap:nothing"]
    rig.write_plan("plan4.json", plan)
    rig.start()
    rig.wait_begun("p5")
    assert rig.begun("p4") == 0
    assert "steps not done {'p4': 'skipped'}" in (rig.sd4 / "run.log").read_text()


def test_a_flagged_s0_goes_on_and_the_log_says_so(rig):
    plan = _plan("server-run4", ["p4"], "round5")
    plan["steps"].insert(0, {"id": "S0", "type": "check", "pytest": {"skip": True},
                             "repro": {"skip": True}})
    rig.write_plan("plan4.json", plan)
    rig.release("p4")
    rig.start()
    rig.wait_begun("p5")
    assert "pytest skipped by the plan" in (rig.sd4 / "run.log").read_text().split("chain:")[1]


# ---- stop


def test_a_stop_in_the_first_round_does_not_start_the_next(rig):
    rig.start()
    rig.wait_begun("p4")
    r = rig.sh("stop.sh", "--plan", "round4")
    assert r.returncode == 0 and "stopping orchestrator" in r.stdout
    rig.wait_gone(rig.sd4)
    time.sleep(2)
    assert rig.begun("p5") == 0 and not (rig.sd5 / "run.log").exists()
    assert not (rig.sd4 / "DONE").exists()
    names = rig.names(rig.sd4)
    assert "stopped" in names and "chain_start" not in names


def test_a_start_after_a_stop_resumes_the_round_and_the_chain_goes_on(rig):
    rig.start()
    rig.wait_begun("p4")
    rig.sh("stop.sh", "--plan", "round4")
    rig.wait_gone(rig.sd4)
    assert (rig.sd4 / "chain.stop").exists()
    assert rig.start().returncode == 0
    rig.wait_begun("p4", 2)
    assert not (rig.sd4 / "chain.stop").exists()
    rig.release("p4")
    rig.wait_begun("p5")


def test_stop_with_the_first_plan_name_stops_the_second_round(rig):
    rig.start()
    rig.wait_begun("p4")
    rig.release("p4")
    rig.wait_begun("p5")
    rig.wait_gone(rig.sd4)
    r = rig.sh("stop.sh", "--plan", "round4")
    assert "stopping orchestrator" in r.stdout and "plan5.json" in r.stdout
    rig.wait_gone(rig.sd5)
    assert "stopped" in rig.names(rig.sd5) and not (rig.sd5 / "DONE").exists()


def test_stop_with_the_second_plan_name_stops_the_first_round(rig):
    rig.start()
    rig.wait_begun("p4")
    r = rig.sh("stop.sh", "--plan", "round5")
    assert "plan4.json" in r.stdout
    rig.wait_gone(rig.sd4)
    assert rig.begun("p5") == 0


def test_a_stop_just_before_the_handover_blocks_the_next_start(rig):
    """stop.sh wrote its marker, but the round ended before the signal came."""
    rig.start()
    rig.wait_begun("p4")
    (rig.sd4 / "chain.stop").write_text("now\n")
    rig.release("p4")
    rig.wait_done(rig.sd4)
    rig.wait_gone(rig.sd4)
    time.sleep(1)
    assert rig.begun("p5") == 0
    ev = [e for e in rig.events(rig.sd4) if e["event"] == "chain_not_started"]
    assert ev and ev[0]["reason"] == "stop.sh ran"


def test_a_chained_start_on_its_way_reads_the_marker(rig):
    (rig.sd4).mkdir()
    (rig.sd4 / "chain.stop").write_text("now\n")
    r = rig.start("round5", "--chained")
    assert r.returncode == 0 and "stop.sh ran" in r.stdout
    assert not (rig.sd5 / "run.log").exists()


def test_an_orchestrator_started_as_chained_reads_the_marker_itself(rig):
    (rig.sd4).mkdir()
    (rig.sd4 / "chain.stop").write_text("now\n")
    r = subprocess.run([str(rig.repo / ".venv/bin/python"), "tools/server/orchestrate.py", "--plan",
                        "tools/server/plan5.json"], cwd=rig.repo, capture_output=True, text=True,
                       env=rig.env(ORCH_CHAINED="1"), timeout=TIMEOUT)
    assert r.returncode == 0
    assert rig.begun("p5") == 0 and not rig.live(rig.sd5)


# ---- never two orchestrators of the chain


def test_never_two_orchestrators_while_the_first_round_runs(rig):
    assert rig.start().returncode == 0
    rig.wait_begun("p4")
    for plan in ("round4", "round5"):
        r = rig.start(plan)
        assert r.returncode == 1 and "already running" in r.stderr, plan
    for plan in ("plan4.json", "plan5.json"):
        r = subprocess.run([str(rig.repo / ".venv/bin/python"), "tools/server/orchestrate.py",
                            "--plan", f"tools/server/{plan}"], cwd=rig.repo, capture_output=True,
                           text=True, env=rig.env(), timeout=TIMEOUT)
        assert r.returncode == 3 and "chain" in r.stderr, plan
    assert rig.begun("p5") == 0 and rig.begun("p4") == 1


def test_never_two_orchestrators_while_the_second_round_runs(rig):
    rig.start()
    rig.wait_begun("p4")
    rig.release("p4")
    rig.wait_begun("p5")
    rig.wait_gone(rig.sd4)
    for plan in ("round4", "round5"):
        r = rig.start(plan)
        assert r.returncode == 1 and "already running" in r.stderr, plan
    r = subprocess.run([str(rig.repo / ".venv/bin/python"), "tools/server/orchestrate.py", "--plan",
                        "tools/server/plan4.json", ], cwd=rig.repo, capture_output=True, text=True,
                       env=rig.env(), timeout=TIMEOUT)
    assert r.returncode == 3
    assert rig.begun("p5") == 1


def test_the_lock_alone_refuses_a_second_orchestrator(rig):
    """A lost pid file does not open the door: the lock decides."""
    rig.start()
    rig.wait_begun("p4")
    keep = (rig.sd4 / "orchestrator.pid").read_text()
    (rig.sd4 / "orchestrator.pid").unlink()
    r = rig.start("round5")
    assert r.returncode == 1 and "held" in r.stderr
    (rig.sd4 / "orchestrator.pid").write_text(keep)


def test_two_starts_at_the_same_time_give_one_orchestrator(rig):
    ps = [subprocess.Popen(["bash", str(rig.repo / "tools/server/start.sh"), "--plan", "round4"],
                           cwd=rig.repo, env=rig.env(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True) for _ in range(3)]
    for p in ps:
        p.communicate(timeout=TIMEOUT)
    rig.wait_begun("p4")
    time.sleep(2)
    assert rig.begun("p4") == 1


# ---- resume


def test_start_with_the_first_plan_name_resumes_the_second_round(rig):
    rig.start()
    rig.wait_begun("p4")
    rig.release("p4")
    rig.wait_begun("p5")
    rig.wait_gone(rig.sd4)
    rig.pod_restart()
    r = rig.start("round4")
    assert r.returncode == 0 and "the next plan of the chain" in r.stdout
    rig.wait_begun("p5", 2)
    assert rig.begun("p4") == 1
    assert rig.state(rig.sd5)["sessions"] == 2 and rig.state(rig.sd4)["sessions"] == 1
    rig.release("p5")
    rig.wait_done(rig.sd5)
    assert len(rig.bundles(rig.sd4)) == 1


def test_start_with_all_done_starts_nothing(rig):
    rig.release("p4")
    rig.release("p5")
    rig.start()
    rig.wait_done(rig.sd5)
    rig.wait_gone(rig.sd5)
    n = len(rig.trace())
    r = rig.start("round4")
    assert r.returncode == 0
    assert r.stdout.count("The round is DONE") == 2
    for sd in (rig.sd4, rig.sd5):
        assert rig.bundles(sd)[0].name in r.stdout
    time.sleep(1)
    assert len(rig.trace()) == n


def test_a_crash_between_done_and_the_handover_is_repaired_by_a_start(rig):
    """DONE exists, and round 5 has not started: the pod died in that moment."""
    rig.release("p4")
    (rig.sd4).mkdir()
    (rig.sd4 / "DONE").write_text("2026-01-01T00:00:00Z\n")
    r = rig.start("round4")
    assert r.returncode == 0
    rig.wait_begun("p5")
    assert rig.begun("p4") == 0


def test_force_reruns_the_plan_and_does_not_start_the_next(rig):
    (rig.sd4).mkdir()
    (rig.sd4 / "DONE").write_text("2026-01-01T00:00:00Z\n")
    rig.release("p4")
    assert rig.start("round4", "--force").returncode == 0
    rig.wait_done(rig.sd4)
    rig.wait_gone(rig.sd4)
    time.sleep(1)
    assert rig.begun("p5") == 0
    ev = [e for e in rig.events(rig.sd4) if e["event"] == "chain_not_started"]
    assert "--force" in ev[0]["reason"]


# ---- RETRY_FAILED


def _failed_round5(rig):
    """Round 4 still to run; round 5 has one failed step."""
    rig.sd5.mkdir()
    (rig.sd5 / "state.json").write_text(json.dumps(
        {"plan": "x", "steps": {"p5": {"status": "failed", "start": "2026-01-01T00:00:00Z"}}}))


def test_retry_failed_does_not_pass_through_the_chain(rig):
    _failed_round5(rig)
    rig.release("p5")
    rig.start("round4", RETRY_FAILED="1", REPRO_ONLY="x")
    rig.wait_begun("p4")
    assert "RETRY_FAILED=1 REPRO_ONLY=x" in rig.trace()[0]
    rig.release("p4")
    rig.wait_done(rig.sd5)
    assert rig.begun("p5") == 0  # the failed step stays failed
    assert rig.state(rig.sd5)["steps"]["p5"]["status"] == "failed"


def test_retry_failed_does_not_pass_when_start_goes_on_to_the_next_plan(rig):
    _failed_round5(rig)
    rig.sd4.mkdir()
    (rig.sd4 / "DONE").write_text("2026-01-01T00:00:00Z\n")
    rig.release("p5")
    rig.start("round4", RETRY_FAILED="1")
    rig.wait_done(rig.sd5)
    assert rig.begun("p5") == 0


def test_retry_failed_works_on_the_plan_it_names(rig):
    _failed_round5(rig)
    rig.release("p5")
    rig.start("round5", RETRY_FAILED="1")
    rig.wait_begun("p5")
    assert "RETRY_FAILED=1" in rig.trace()[0]


# ---- autostart


def test_autostart_installs_for_the_whole_chain(rig):
    r = rig.sh("install_autostart.sh", "--plan", "round4")
    assert r.returncode == 0
    text = (rig.jupyter / "jupyter_server_config.py").read_text()
    assert "server-run4" in text and "server-run5" in text
    assert rig.sh("uninstall_autostart.sh").returncode == 0
    assert "pyMOSES" not in (rig.jupyter / "jupyter_server_config.py").read_text()


def test_pod_restart_during_the_first_round(rig):
    rig.sh("install_autostart.sh", "--plan", "round4")
    rig.start()
    rig.wait_begun("p4")
    rig.pod_restart()
    rig.autostart_block()
    rig.wait_begun("p4", 2)
    assert rig.state(rig.sd4)["sessions"] == 2 and rig.begun("p5") == 0
    rig.wait("autostart log", lambda: "started" in (rig.sd4 / "autostart.log").read_text())
    rig.release("p4")
    rig.wait_begun("p5")  # the chain goes on from the resumed session


def test_pod_restart_during_the_second_round(rig):
    rig.sh("install_autostart.sh", "--plan", "round4")
    rig.start()
    rig.wait_begun("p4")
    rig.release("p4")
    rig.wait_begun("p5")
    rig.wait_gone(rig.sd4)
    rig.pod_restart()
    rig.autostart_block()
    rig.wait_begun("p5", 2)
    assert rig.begun("p4") == 1
    assert rig.state(rig.sd5)["sessions"] == 2
    rig.release("p5")
    rig.wait_done(rig.sd5)


def test_pod_restart_after_both_rounds_starts_nothing(rig):
    rig.sh("install_autostart.sh", "--plan", "round4")
    rig.release("p4")
    rig.release("p5")
    rig.start()
    rig.wait_done(rig.sd5)
    rig.wait_gone(rig.sd5)
    n = len(rig.trace())
    size = (rig.sd4 / "autostart.log").stat().st_size if (rig.sd4 / "autostart.log").exists() else 0
    rig.autostart_block()
    time.sleep(2)
    assert len(rig.trace()) == n
    assert ((rig.sd4 / "autostart.log").stat().st_size
            if (rig.sd4 / "autostart.log").exists() else 0) == size


def test_a_second_restart_while_running_does_not_start_a_second_orchestrator(rig):
    rig.sh("install_autostart.sh", "--plan", "round4")
    rig.start()
    rig.wait_begun("p4")
    rig.autostart_block()  # the Jupyter server restarts, the round does not stop
    rig.wait("the block's start.sh", lambda: (rig.sd4 / "autostart.log").exists()
             and "already running" in (rig.sd4 / "autostart.log").read_text())
    assert rig.begun("p4") == 1


# ---- status


def test_status_says_the_next_plan_and_each_bundle(rig):
    rig.start()
    rig.wait_begun("p4")
    r = rig.sh("status.sh", "--plan", "round4")
    assert r.returncode == 0, r.stderr
    chain = r.stdout.split("chain:")[1].split("steps:")[0]
    assert re.search(r"round4 \(state .*server-run4\): RUNNING pid", chain)
    assert re.search(r"round5 \(state .*server-run5\): not started", chain)
    assert chain.count("no bundle yet") == 2
    rig.release("p4")
    rig.wait_begun("p5")
    rig.wait_gone(rig.sd4)
    r = rig.sh("status.sh", "--plan", "round4")
    chain = r.stdout.split("chain:")[1].split("steps:")[0]
    assert re.search(r"round4 .*: DONE", chain)
    assert re.search(r"round5 .*: RUNNING pid", chain)
    assert str(rig.bundles(rig.sd4)[0]) in chain
    rig.sh("stop.sh", "--plan", "round4")
    rig.wait_gone(rig.sd5)
    r = rig.sh("status.sh", "--plan", "round5")
    chain = r.stdout.split("chain:")[1].split("steps:")[0]
    assert re.search(r"round5 .*: stopped or interrupted", chain)
    assert "bundle-latest.tar.gz" in chain


def test_status_of_a_plan_without_a_chain_has_no_chain_section(rig):
    rig.write_plan("plan4.json", _plan("server-run4", ["p4"]))
    assert "chain:" not in rig.sh("status.sh", "--plan", "round4").stdout


# ---- the plan names and the real plans


def _chain_module():
    sys.path.insert(0, str(SERVER))
    import chain

    return chain


def test_plan_names_of_chain_py_and_common_sh_agree():
    chain = _chain_module()
    for name in chain.PLAN_FILES:
        r = subprocess.run(
            ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
            cwd=REPO, capture_output=True, text=True, check=True)
        assert r.stdout.strip() == str(SERVER / chain.PLAN_FILES[name]), name


@pytest.mark.parametrize("first, second",
                         [("plan4.json", "round5"), ("plan4-smoke.json", "smoke5")])
def test_real_plans_chain(first, second):
    chain = _chain_module()
    assert json.loads((SERVER / first).read_text())["then"] == second
    assert chain.resolve(second) is not None
    assert chain.next_plan(SERVER / first) == chain.resolve(second)
    assert chain.root(chain.resolve(second)) == SERVER / first
    assert chain.lock_path(SERVER / first) == chain.lock_path(chain.resolve(second))
    assert chain.members(chain.resolve(second)) == [SERVER / first, chain.resolve(second)]


@pytest.mark.parametrize("name", ["plan.json", "plan2.json", "plan3.json", "plan-smoke.json",
                                  "plan2-smoke.json", "plan3-smoke.json"])
def test_earlier_plans_are_not_in_a_chain(name):
    chain = _chain_module()
    assert chain.members(SERVER / name) == [SERVER / name]


# ---- the active time of a step


def test_duration_excludes_the_idle_time_between_sessions(rig):
    """A step that first started long ago, with 100 s active before this session."""
    rig.write_plan("plan4.json", _plan("server-run4", ["p4"]))
    rig.sd4.mkdir()
    (rig.sd4 / "state.json").write_text(json.dumps(
        {"plan": "x", "sessions": 1, "round_started": "2020-01-01T00:00:00Z",
         "steps": {"p4": {"status": "running", "start": "2020-01-01T00:00:00Z",
                          "duration_s": 100}}}))
    rig.release("p4")
    rig.start()
    rig.wait_done(rig.sd4)
    st = rig.state(rig.sd4)["steps"]["p4"]
    assert 100 <= st["duration_s"] < 200
    assert st["elapsed_s"] > 86400 * 300
    end = next(e for e in rig.events(rig.sd4) if e["event"] == "step_end")
    assert end["duration_s"] == st["duration_s"] and end["elapsed_s"] == st["elapsed_s"]
    summary = (rig.sd4 / "round-summary.md").read_text()
    assert "active time" in summary
    row = next(x for x in summary.splitlines() if x.startswith("| p4 |"))
    assert " 0.03 h " in row or " 0.04 h " in row or " 0.05 h " in row


def test_duration_over_a_real_kill_and_an_idle_gap(rig):
    rig.write_plan("plan4.json", _plan("server-run4", ["p4"]))
    rig.start()
    rig.wait_begun("p4")
    time.sleep(3)  # the monitor saves the active time at each poll (1 s here)
    rig.pod_restart()
    before = rig.state(rig.sd4)["steps"]["p4"]["duration_s"]
    assert before >= 2
    time.sleep(12)  # idle: no orchestrator
    rig.start()
    rig.wait_begun("p4", 2)
    rig.release("p4")
    rig.wait_done(rig.sd4)
    st = rig.state(rig.sd4)["steps"]["p4"]
    assert st["duration_s"] >= before
    assert st["elapsed_s"] - st["duration_s"] >= 10


def test_a_pilot_has_its_active_time_too(rig):
    plan = _plan("server-run4", ["p4"])
    plan["steps"][0]["pilot"] = "p5"
    rig.write_plan("plan4.json", plan)
    rig.release("p4")
    rig.release("p5")
    rig.start()
    rig.wait_done(rig.sd4)
    pst = rig.state(rig.sd4)["steps"]["p4"]["pilot"]
    assert pst["status"] == "done" and pst["duration_s"] >= 0 and "elapsed_s" in pst


# ---- the bundle


def test_the_bundle_holds_the_const_opt_choice(rig):
    sys.path.insert(0, str(SERVER))
    import orchestrate

    assert "const-opt-choice.json" in orchestrate.RUN_FILES
    rig.write_plan("plan4.json", _plan("server-run4", ["p4"]))
    (rig.res / "p4").mkdir()
    (rig.res / "p4" / "const-opt-choice.json").write_text('{"choice": "lm"}')
    rig.release("p4")
    rig.start()
    rig.wait_done(rig.sd4)
    with tarfile.open(rig.bundles(rig.sd4)[0]) as tf:
        assert "gomea-results/p4/const-opt-choice.json" in tf.getnames()
