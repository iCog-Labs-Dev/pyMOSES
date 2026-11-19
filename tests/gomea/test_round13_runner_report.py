"""Server round 13: the runner and report changes (``SERVER-ROUND13-PREREGISTRATION.md``,
"Rules", "Arms" and "TLS-G").

* ``sr_eval.evaluate(..., plain=True, clip=True)``: the plain a + b*f values and the test NMSE
  with clipped predictions; without the keywords the result does not change.
* ``moses_sr_runner.py``: ``test_nmse_clipped`` for every arm of a manifest with
  ``"clip_test_nmse": true`` (absent without the key), the plain-scaling fields of a
  ``--linear-scaling terms`` arm, and the TLS-G guard (``"terms_guard": "val"``): for each
  candidate the lower validation NMSE of the per-term model and the plain a + b*f (a tie keeps
  plain), ``terms.guard`` in the summary and in each candidate.
* ``phase_report.py`` with ``run_table.py``: the program size of a TLS arm (``"size_terms"``),
  the per-run table and the runs with a test NMSE above 100 (``"run_table"``), and the
  same-search check (``"same_search"``).
* Byte identity: the output of the runner and of the report at cb1ccfe (the commit before
  these changes, from git) is the same as now for an arm or a spec without the new keys,
  apart from the new fields.

Backlog 12.2 (the RV16 review of the round 13 build):

* the plain-scaling fields need the manifest key ``"plain_nmse": true``; without it a
  ``--linear-scaling terms`` arm (also the BASE-TLS arm of the round 12 manifest) writes the
  output of cb1ccfe, and TLS-G still compares the plain values;
* a missing reference commit fails the byte-identity tests (``_old_tool``); it never skips;
* the spec key ``"pooled_descriptive"``: named pooled secondary contrasts keep their p, Holm p
  and ratio and get the verdict "descriptive"; without the key the report is that of 0f6f2eb
  (the commit before backlog 12.2).
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "gomea"
sys.path.insert(0, str(TOOLS))
import moses_sr_runner as runner  # noqa: E402
import phase_report  # noqa: E402
import run_table  # noqa: E402
import sr_eval  # noqa: E402

OLD = "cb1ccfe"  # the commit before the round 13 runner and report changes
TRAIN = REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"
PROGRAMS = [
    "+(sin(*(1.3 $1)) *(0.5 $2))",  # per-term fit, 2 terms
    "+(sin($1) $2 *($1 $2))",  # per-term fit, 3 terms
    "+(/(1 +(4 $1)) sin($2) exp(*(0.1 $1)))",  # per-term fit, 3 terms
    "+($1 *(2 $1))",  # collinear: fallback
    "+(sin($1) $2 0.25)",  # a numeric leaf: fallback
    "*(2.5 sin(*(1.3 $1)))",  # one term
    "$2",  # one term
]
TLS = ["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "none"]
PLAIN = ["-a", "gomea", "--linear-scaling", "1"]
VOLATILE = ("argv", "wall_s", "cpu_s", "ru_maxrss_kb")


def _splits(dst: Path, f=None) -> Path:
    """train/val/test CSVs from the test table (row i: train if i % 5 < 3, val if 3, test if
    4); `f(x1, x2)` replaces the target when given."""
    dst.mkdir(parents=True, exist_ok=True)
    lines = TRAIN.read_text().splitlines()
    head, rows = lines[0], lines[1:]
    if f is not None:
        rows = [f"{a},{b},{f(float(a), float(b))!r}" for a, b, _ in (r.split(",") for r in rows)]
    for name, keep in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
        body = [r for i, r in enumerate(rows) if i % 5 in keep]
        (dst / f"{name}.csv").write_text("\n".join([head, *body]) + "\n")
    return dst


def _fake_moses(work: Path, programs, events: str = "") -> Path:
    """A fake MOSES that prints `programs` as candidates (and writes `events` as the run's
    events.jsonl, in its cwd, the run directory)."""
    work.mkdir(parents=True, exist_ok=True)
    out = "".join(f"-{i + 10} {p} \n" for i, p in enumerate(programs)) + "n_evals: 99\n"
    script = "#!/bin/sh\n"
    if events:
        script += "cat > events.jsonl <<'EOF'\n" + events + "EOF\n"
    script += "cat <<'EOF'\n" + out + "EOF\n"
    fake = work / "fake_moses.sh"
    fake.write_text(script)
    fake.chmod(0o755)
    return fake


def _job(work: Path, argv, programs=PROGRAMS, events: str = "", **kw) -> dict:
    return {
        "run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1,
        "binary": str(_fake_moses(work, programs, events)), "argv": argv,
        "budget": {"evals": 99}, "seed_spec": None,
        "split_dir": str(_splits(work / "splits" / "s1")), "wall_cap_s": 60,
        "mem_limit_mb": None, "run_dir": str(work / "run"), **kw,
    }  # fmt: skip


def _tables(work: Path):
    d = work / "splits" / "s1"
    return [sr_eval.read_table(str(d / f"{s}.csv")) for s in ("train", "val", "test")]


def _cands(work: Path) -> list[dict]:
    return json.loads((work / "run" / "candidates.json").read_text())


# ---- sr_eval: plain and clip


@pytest.mark.parametrize("prog", PROGRAMS)
@pytest.mark.parametrize("terms", [False, True])
def test_evaluate_without_the_new_keywords_does_not_change(prog, terms, tmp_path):
    _splits(tmp_path / "splits" / "s1")
    train, val, _ = _tables(tmp_path)
    base = sr_eval.evaluate(prog, train, val, terms=terms)
    new = sr_eval.evaluate(prog, train, val, terms=terms, plain=True, clip=True)
    extra = {"plain"} if terms else set()
    assert set(new) == set(base) | extra
    assert set(new["test"]) == set(base["test"]) | {"nmse_clipped"}
    for k in base:
        if k != "test":
            assert new[k] == base[k], k
    assert {k: new["test"][k] for k in base["test"]} == base["test"]


@pytest.mark.parametrize("prog", PROGRAMS)
def test_plain_values_are_those_of_the_plain_fit(prog, tmp_path):
    _splits(tmp_path / "splits" / "s1")
    train, val, test = _tables(tmp_path)
    for t in (val, test):
        plain = sr_eval.evaluate(prog, train, t)
        new = sr_eval.evaluate(prog, train, t, terms=True, plain=True, clip=True)
        assert new["plain"]["train"] == plain["train"]["nmse_scaled"]
        assert new["plain"]["test"] == plain["test"]["nmse_scaled"]
        assert new["plain"]["test_clipped"] == sr_eval.evaluate(
            prog, train, t, clip=True)["test"]["nmse_clipped"]
        if new["terms"]["n_terms"] < 2 or new["terms"]["fallback"]:
            assert new["test"]["nmse_scaled"] == new["plain"]["test"]
            assert new["test"]["nmse_clipped"] == new["plain"]["test_clipped"]


def test_clip_range_and_the_clip_rule():
    lo, hi = sr_eval.clip_range([2.0, 12.0, 7.0])
    assert (lo, hi) == (1.0, 13.0)
    y = [1.0, 2.0, 3.0, 4.0]
    assert sr_eval.nmse_clipped(y, y, 0.0, 10.0) == 0.0
    # +inf and -inf become the bounds; a value inside the range does not change
    q = [math.inf, 2.0, -math.inf, 4.0]
    assert sr_eval.nmse_clipped(y, q, 1.0, 4.0) == sr_eval.nmse(y, [4.0, 2.0, 1.0, 4.0])
    assert sr_eval.nmse_clipped(y, [100.0, 2.0, 3.0, -50.0], 0.0, 5.0) == sr_eval.nmse(
        y, [5.0, 2.0, 3.0, 0.0])
    # a nan prediction stays nan: the NMSE is infinite
    assert sr_eval.nmse_clipped(y, [math.nan, 2.0, 3.0, 4.0], 0.0, 5.0) == math.inf


def test_the_clipped_nmse_clips_the_scaled_predictions_to_the_train_range(tmp_path):
    """A pole on the test rows: the scaled prediction far outside the training range."""
    _splits(tmp_path / "splits" / "s1", f=lambda a, b: a + b)
    train, _, test = _tables(tmp_path)
    prog = "/(1 +(-0.3 $1))"  # 1 / (x1 - 0.3)
    res = sr_eval.evaluate(prog, train, test, clip=True)
    lo, hi = sr_eval.clip_range(train.y)
    p = sr_eval.predict(prog, test.X)
    q = [res["a"] + res["b"] * x for x in p]
    want = sr_eval.nmse(test.y, [x if math.isnan(x) else min(max(x, lo), hi) for x in q])
    assert res["test"]["nmse_clipped"] == want
    assert res["test"]["nmse_clipped"] <= res["test"]["nmse_scaled"]


# ---- the runner: new fields


def test_every_arm_gets_the_clipped_test_nmse_of_the_reported_candidate(tmp_path):
    s = runner._run_one(_job(tmp_path, PLAIN, clip_test_nmse=True))
    train, _, test = _tables(tmp_path)
    want = sr_eval.evaluate(s["final_program"], train, test, clip=True)["test"]
    assert s["test_nmse"] == want["nmse_scaled"]
    assert s["test_nmse_clipped"] == want["nmse_clipped"]
    assert "terms" not in s and "plain_val_nmse" not in s
    # candidates.json has no new field for an arm without --linear-scaling terms
    assert all(set(c) == {"program", "train_nmse", "val_nmse", "test_nmse"}
               for c in _cands(tmp_path))


@pytest.mark.parametrize("argv, guard", [(PLAIN, None), (TLS, None), (TLS, "val")])
def test_without_the_manifest_key_there_is_no_clipped_test_nmse(argv, guard, tmp_path):
    kw = {"terms_guard": guard} if guard else {}
    off = runner._run_one(_job(tmp_path / "off", argv, **kw))
    on = runner._run_one(_job(tmp_path / "on", argv, clip_test_nmse=True, **kw))
    assert "test_nmse_clipped" not in off and "test_nmse_clipped" in on
    assert json.loads((tmp_path / "off" / "run" / "summary.json").read_text()).keys() == (
        set(on) - {"test_nmse_clipped"})
    for k in ("final_program", "train_nmse", "val_nmse", "test_nmse"):
        assert off[k] == on[k]
    assert _cands(tmp_path / "off") == _cands(tmp_path / "on")


def test_a_tls_arm_gets_the_plain_fields(tmp_path):
    s = runner._run_one(_job(tmp_path, TLS, clip_test_nmse=True, plain_nmse=True))
    train, val, test = _tables(tmp_path)
    for c in _cands(tmp_path):
        assert c["plain_val_nmse"] == sr_eval.evaluate(c["program"], train, val)["test"][
            "nmse_scaled"]
        assert c["plain_test_nmse"] == sr_eval.evaluate(c["program"], train, test)["test"][
            "nmse_scaled"]
        assert "guard" not in c["terms"] and "terms_val_nmse" not in c
    best = min(_cands(tmp_path), key=lambda c: c["val_nmse"])
    assert s["final_program"] == best["program"]
    assert (s["plain_val_nmse"], s["plain_test_nmse"]) == (best["plain_val_nmse"],
                                                           best["plain_test_nmse"])
    assert set(s["terms"]) == {"n_terms", "fallback", "reason", "weights"}
    want = sr_eval.evaluate(s["final_program"], train, test, terms=True, clip=True)
    assert s["test_nmse_clipped"] == want["test"]["nmse_clipped"]


def test_the_guard_keeps_the_lower_validation_nmse_and_plain_on_a_tie(tmp_path):
    ref = runner._run_one(_job(tmp_path / "tls", TLS, plain_nmse=True))
    s = runner._run_one(_job(tmp_path / "g", TLS, terms_guard="val", clip_test_nmse=True,
                             plain_nmse=True))
    train, val, test = _tables(tmp_path / "g")
    seen = set()
    for c, r in zip(_cands(tmp_path / "g"), _cands(tmp_path / "tls"), strict=True):
        assert c["program"] == r["program"]
        assert (c["terms_val_nmse"], c["terms_test_nmse"]) == (r["val_nmse"], r["test_nmse"])
        assert (c["plain_val_nmse"], c["plain_test_nmse"]) == (r["plain_val_nmse"],
                                                               r["plain_test_nmse"])
        g = c["terms"]["guard"]
        seen.add(g)
        assert {k: v for k, v in c["terms"].items() if k != "guard"} == r["terms"]
        if r["plain_val_nmse"] <= r["val_nmse"]:  # lower or tie: plain
            assert g == "plain"
            plain = sr_eval.evaluate(c["program"], train, val)
            assert c["train_nmse"] == plain["train"]["nmse_scaled"]
            assert (c["val_nmse"], c["test_nmse"]) == (r["plain_val_nmse"], r["plain_test_nmse"])
        else:
            assert g == "terms"
            assert (c["train_nmse"], c["val_nmse"], c["test_nmse"]) == (
                r["train_nmse"], r["val_nmse"], r["test_nmse"])
        if r["terms"]["n_terms"] < 2 or r["terms"]["fallback"]:
            assert r["plain_val_nmse"] == r["val_nmse"] and g == "plain"  # a tie
    assert seen == {"plain", "terms"}
    best = min(_cands(tmp_path / "g"), key=lambda c: c["val_nmse"])
    assert s["final_program"] == best["program"] and s["terms"] == best["terms"]
    assert s["terms"]["guard"] in ("terms", "plain")
    assert (s["train_nmse"], s["val_nmse"], s["test_nmse"]) == (
        best["train_nmse"], best["val_nmse"], best["test_nmse"])
    kept = s["terms"]["guard"] == "terms"
    want = sr_eval.evaluate(s["final_program"], train, test, terms=kept, clip=True)
    assert s["test_nmse_clipped"] == want["test"]["nmse_clipped"]
    assert ref["final_program"] in [c["program"] for c in _cands(tmp_path / "tls")]


def test_the_guard_chooses_plain_for_the_final_model_when_terms_overfit(tmp_path):
    """Only the plain fit and the per-term fit of one 2-term program: when the plain fit has
    the lower validation NMSE, the summary reports the plain values and guard "plain"."""
    prog = "+(sin(*(1.3 $1)) *(0.5 $2))"
    work = tmp_path / "w"
    _splits(work / "splits" / "s1", f=lambda a, b: math.sin(1.3 * a) + 0.5 * b)
    job = _job(work, TLS, programs=[prog], terms_guard="val")
    job["split_dir"] = str(work / "splits" / "s1")
    s = runner._run_one(job)
    train, val, test = _tables(work)
    t = sr_eval.evaluate(prog, train, val, terms=True, plain=True)
    want = "plain" if t["plain"]["test"] <= t["test"]["nmse_scaled"] else "terms"
    assert s["terms"]["guard"] == want


def test_the_guard_on_an_unscorable_candidate(tmp_path):
    s = runner._run_one(_job(tmp_path, TLS, programs=["+($1 $9)", PROGRAMS[0]],
                             terms_guard="val", plain_nmse=True))
    c0, c1 = _cands(tmp_path)
    assert "eval_error" in c0 and c0["plain_val_nmse"] == math.inf and "terms" not in c0
    assert c1["terms"]["guard"] in ("terms", "plain")
    assert s["final_program"] == PROGRAMS[0] and s["status"] == "ok"


def test_expand_carries_the_guard_key_and_refuses_a_wrong_one(tmp_path):
    arm = {"binary": "/x/.venv/bin/moses", "argv": TLS, "budget": {"evals": 10}}
    m = {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con"], "splits": [1],
         "wall_cap_s": 10,
         "arms": {"BASE-TLS": arm, "TLS-G": {**arm, "terms_guard": "val"}}}
    jobs = {j["arm"]: j for j in runner.expand(m, tmp_path, tmp_path)}
    assert "terms_guard" not in jobs["BASE-TLS"] and jobs["TLS-G"]["terms_guard"] == "val"
    assert jobs["TLS-G"]["argv"] == jobs["BASE-TLS"]["argv"]
    assert runner.moses_argv(jobs["TLS-G"], tmp_path, 100) == runner.moses_argv(
        jobs["BASE-TLS"], tmp_path, 100)
    assert all("clip_test_nmse" not in j for j in jobs.values())
    on = runner.expand({**m, "clip_test_nmse": True}, tmp_path, tmp_path)
    assert all(j["clip_test_nmse"] is True for j in on)
    off = runner.expand({**m, "clip_test_nmse": False}, tmp_path, tmp_path)
    assert all("clip_test_nmse" not in j for j in off)
    for v in ("yes", 1, None):
        with pytest.raises(ValueError, match="clip_test_nmse must be true or false"):
            runner.expand({**m, "clip_test_nmse": v}, tmp_path, tmp_path)
    for bad, why in [({**arm, "argv": PLAIN, "terms_guard": "val"}, "--linear-scaling terms"),
                     ({**arm, "argv": ["-a", "gomea"], "terms_guard": "val"},
                      "--linear-scaling terms"),
                     ({**arm, "terms_guard": "test"}, "must be"),
                     ({**arm, "terms_guard": True}, "must be")]:
        with pytest.raises(ValueError, match=why):
            runner.expand({**m, "arms": {"X": bad}}, tmp_path, tmp_path)


def test_the_pole_counters_and_the_lexicase_block_reach_the_summary(tmp_path):
    co = {"mode": "lm", "lm_calls": 3, "lm_improved": 2, "tuning_evals": 30,
          "lm_guard_checked": 2, "lm_guard_rejected": 1, "guard": "val",
          "lm_pole_checked": 2, "lm_pole_rejected": 1}
    lex = {"n_cases": 30, "selections": 4, "cases_used_mean": 2.5, "cases_used_max": 4,
           "pool_size_mean": 7.0, "bscore_recomputes": 12, "bscore_recompute_cpu_s": 0.5}
    events = "".join(json.dumps(e) + "\n" for e in [
        {"type": "end", "dim": 20, "ledger": {"requests": 50}, "const_opt": co},
        {"type": "end", "dim": 30, "ledger": {"requests": 49}, "const_opt": co},
        {"type": "lexicase", **lex},
    ])
    s = runner._run_one(_job(tmp_path, PLAIN, events=events))
    disk = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert disk["const_opt"]["lm_pole_checked"] == 4 and disk["const_opt"]["lm_pole_rejected"] == 2
    assert disk["const_opt"]["lm_pole_rejected_share"] == 0.5
    assert disk["lexicase"] == lex and s["lexicase"] == lex


# ---- byte identity against the code before the change (git)


def _old_tool(tmp_path: Path, names, commit: str | None = None) -> Path:
    """The files `names` of tools/gomea/ at commit OLD (or `commit`), in a directory of their
    own."""
    commit = commit or OLD
    d = tmp_path / f"old-{commit}"
    d.mkdir()
    for n in names:
        r = subprocess.run(["git", "show", f"{commit}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:
            # backlog 12.2: a missing reference must never let the test pass (no skip)
            pytest.fail(f"git show {commit}:tools/gomea/{n} failed: "
                        f"{r.stderr.decode().strip()!r}. The byte-identity tests need the "
                        f"reference commit {commit}; use a clone with the full history.")
        (d / n).write_bytes(r.stdout)
    return d


RUN_ONE = ("import json, sys; sys.path.insert(0, sys.argv[1]); import moses_sr_runner as r; "
           "r._run_one(json.loads(sys.argv[2]))")


def _run_in(tool_dir: Path, job: dict) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run([sys.executable, "-c", RUN_ONE, str(tool_dir), json.dumps(job)], env=env,
                   check=True, cwd=job["run_dir"].rsplit("/", 1)[0])


@pytest.mark.parametrize("argv", [PLAIN, TLS, ["-a", "hc", "--linear-scaling", "1"]])
def test_the_runner_output_is_byte_identical_apart_from_the_new_fields(argv, tmp_path):
    old = _old_tool(tmp_path, ["moses_sr_runner.py", "sr_eval.py"])
    progs = PROGRAMS + ["+($1 $9)"]  # with an unscorable candidate
    # backlog 12.2: the new job asks for the plain fields, which round 13 always wrote
    jo = _job(tmp_path / "o", argv, programs=progs)
    jn = _job(tmp_path / "n", argv, programs=progs, plain_nmse=True)
    _run_in(old, jo)
    _run_in(TOOLS, jn)
    co, cn = (Path(j["run_dir"]) / "candidates.json" for j in (jo, jn))
    so, sn = (json.loads((Path(j["run_dir"]) / "summary.json").read_text()) for j in (jo, jn))
    new_s = set()  # no "clip_test_nmse" in the job: no test_nmse_clipped
    if argv is TLS:
        new_c = ("plain_val_nmse", "plain_test_nmse")
        strip = [{k: v for k, v in c.items() if k not in new_c} for c in json.loads(cn.read_text())]
        assert json.dumps(strip, indent=2) == co.read_text()
        assert all(all(k in c for k in new_c) for c in json.loads(cn.read_text()))
        new_s |= {"plain_val_nmse", "plain_test_nmse"}
    else:
        assert cn.read_bytes() == co.read_bytes()
    assert set(sn) - set(so) == new_s and set(so) <= set(sn)

    def text(s):
        return json.dumps({k: v for k, v in s.items() if k not in VOLATILE and k not in new_s},
                          indent=2, sort_keys=True)
    assert text(sn) == text(so)


def _made_up_runs(root: Path, arms, datasets, splits, size_terms_record=True):
    """Made-up summaries (and events.jsonl and candidates.json) of a regression phase."""
    for d in datasets:
        for k in splits:
            for i, arm in enumerate(arms):
                rd = root / "runs" / f"{d}__{arm}__s{k}"
                rd.mkdir(parents=True)
                nmse = (1 + i / 10) * (0.1 + k / 1000 + datasets.index(d) / 1000)
                if arm == arms[-1] and k == splits[0]:
                    nmse = 250.0 + k  # a run above 100
                s = {"arm": arm, "task": d, "seed": k, "status": "ok", "test_nmse": nmse,
                     "train_nmse": nmse / 2, "val_nmse": nmse * 0.9, "cpu_s": 10.0 + i,
                     "n_evals": 20000, "final_program": "+(*($1 $2) sin($3))",
                     "const_opt": {"mode": "lm", "lm_calls": 2, "tuning_evals": 100,
                                   "tuning_share": 0.005}}
                if size_terms_record and "TLS" in arm:
                    s["terms"] = {"n_terms": 2, "fallback": False, "reason": None,
                                  "weights": [0.0, 1.0, 2.0]}
                if arm == arms[0] and k == splits[-1]:
                    s = {**s, "status": "error", "test_nmse": None}
                (rd / "summary.json").write_text(json.dumps(s))
                end = {"type": "end", "generations": 4, "termination_reason": "stagnation",
                       "ledger": {"requests": 10}, "dim": 40 + i}
                (rd / "events.jsonl").write_text((json.dumps(end) + "\n") * (2 + i % 2))
                (rd / "candidates.json").write_text(json.dumps([{"program": "$1"}]))


def test_an_old_spec_gives_a_byte_identical_report(tmp_path):
    """The 1bf spec of round 12 (no round 13 key), with summaries that have a per-term fit:
    contrasts.csv, aggregate.csv and verdict.json are those of the report at cb1ccfe, and the
    report writes no new file."""
    old = _old_tool(tmp_path, ["phase_report.py"])
    spec = REPO / "configs" / "gomea" / "phase1bf" / "spec.json"
    arms = list(json.loads((REPO / "configs" / "gomea" / "phase1bf" / "sr.json").read_text())[
        "arms"])
    datasets = list(json.loads(spec.read_text())["metrics"])
    root = tmp_path / "r"
    _made_up_runs(root, arms, datasets, list(range(41, 46)))
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    outs = {}
    for name, tool in (("old", old / "phase_report.py"), ("new", TOOLS / "phase_report.py")):
        outs[name] = tmp_path / f"out-{name}"
        r = subprocess.run([sys.executable, str(tool), "--stage", "1", str(spec),
                            str(outs[name]), str(root)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs[name + "-stdout"] = r.stdout
    assert sorted(p.name for p in outs["new"].iterdir()) == sorted(
        p.name for p in outs["old"].iterdir()) == ["aggregate.csv", "contrasts.csv",
                                                  "verdict.json"]
    for f in ("aggregate.csv", "contrasts.csv", "verdict.json"):
        assert (outs["new"] / f).read_bytes() == (outs["old"] / f).read_bytes(), f
    assert outs["new-stdout"] == outs["old-stdout"]


# ---- the report: size, runs.csv, the list above 100, the same-search check


@pytest.mark.parametrize("terms, want", [
    (None, 0),
    ({"n_terms": 1, "fallback": False}, 0),
    ({"n_terms": 3, "fallback": True}, 0),
    ({"n_terms": 3, "fallback": False}, 3),
    ({"n_terms": 2, "fallback": False, "guard": "terms"}, 2),
    ({"n_terms": 2, "fallback": False, "guard": "plain"}, 0),
])  # fmt: skip
def test_the_size_of_a_tls_arm(terms, want):
    s = {"status": "ok", "final_program": "+(*($1 $2) sin($3))"}  # 3 "(" + 6 atoms = 9
    if terms is not None:
        s["terms"] = terms
    assert run_table.terms_size(s) == want
    assert phase_report._nodes(s) == 9
    assert phase_report._size(s) == 9  # without the spec key
    assert phase_report._size({**s, "_size_terms": True}) == 9 + want
    assert phase_report._value("program_size", {**s, "_size_terms": True}) == 9 + want


def _spec13(tmp_path: Path, arms) -> Path:
    spec = {"phase": "t13", "metrics": {"Con": "test_nmse", "Dow": "test_nmse"}, "primary": [],
            "pooled_primary": [["g-vs-tls", "TLS-G", "BASE-TLS", "test_nmse"]],
            "pooled_secondary": [["g-vs-tls-size", "TLS-G", "BASE-TLS", "program_size"],
                                 ["g-vs-tls-clipped", "TLS-G", "BASE-TLS", "test_nmse_clipped"]],
            "extend": False, "size_terms": True, "run_table": True,
            "same_search": [["TLS-G", "BASE-TLS"]]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


def test_the_round13_tables(tmp_path):
    arms = ["BASE", "BASE-TLS", "TLS-G"]
    root = tmp_path / "r"
    _made_up_runs(root, arms, ["Con", "Dow"], [51, 52, 53])
    # TLS-G: guard "plain" on one run; a different candidate list on one (dataset, split)
    g = root / "runs" / "Con__TLS-G__s52"
    s = json.loads((g / "summary.json").read_text())
    s["terms"]["guard"] = "plain"
    s["test_nmse_clipped"] = 0.05
    s["lexicase"] = {"n_cases": 10, "selections": 3}
    (g / "summary.json").write_text(json.dumps(s))
    (g / "candidates.json").write_text(json.dumps(
        [{"program": "$1", "terms": {"guard": "plain"}}, {"program": "$2", "terms": {}}]))
    (root / "runs" / "Dow__BASE-TLS__s53" / "candidates.json").unlink()
    out = tmp_path / "out"
    assert phase_report.main(["--stage", "1", str(_spec13(tmp_path, arms)), str(out),
                              str(root)]) == 0
    rows = list(csv.DictReader((out / "runs.csv").open()))
    assert list(rows[0]) == list(run_table.RUN_FIELDS) and len(rows) == 18
    r = {(x["dataset"], x["arm"], x["split"]): x for x in rows}
    gr = r[("Con", "TLS-G", "52")]
    assert (gr["size"], gr["nodes"], gr["n_terms"], gr["terms_guard"]) == ("9", "9", "2", "plain")
    assert gr["guard_plain_candidates"] == "1" and gr["test_nmse_clipped"] == "0.05"
    assert gr["lexicase_n_cases"] == "10" and gr["lexicase_pool_size_mean"] == ""
    tr = r[("Con", "BASE-TLS", "52")]
    assert (tr["size"], tr["nodes"], tr["n_terms"], tr["terms_guard"]) == ("11", "9", "2", "")
    b = r[("Con", "BASE", "51")]
    assert (b["size"], b["n_terms"], b["demes"], b["knobs_per_deme"], b["total_knobs"]) == (
        "9", "", "2", "40.0", "80")  # the median of two demes of 40 knobs
    assert r[("Con", "BASE-TLS", "51")]["demes"] == "3"
    assert r[("Con", "BASE-TLS", "51")]["total_knobs"] == "123"
    assert (b["lm_calls"], b["lm_share"]) == ("2", "0.005")
    assert r[("Con", "BASE", "53")]["status"] == "error"
    above = list(csv.DictReader((out / "nmse_above_100.csv").open()))
    assert [(x["dataset"], x["arm"], x["split"]) for x in above] == [
        ("Con", "TLS-G", "51"), ("Dow", "TLS-G", "51")]
    assert list(above[0]) == list(run_table.ABOVE_100_FIELDS)
    assert above[0]["test_nmse"] == "301.0" and above[0]["train_nmse"] == "150.5"
    ss = json.loads((out / "same_search.json").read_text())
    assert ss == [{"a": "TLS-G", "b": "BASE-TLS", "pairs": 6, "same": 4, "differ": 1,
                   "missing": 1}]
    sr = {(x["dataset"], x["split"]): x for x in csv.DictReader((out / "same_search.csv").open())}
    assert sr[("Con", "52")]["result"] == "differ" and sr[("Con", "52")]["first_difference"] == "1"
    assert sr[("Dow", "53")]["result"] == "missing"
    # the size contrast counts the term weights of the TLS arms; the clipped metric is known
    con = {x["contrast"]: x for x in csv.DictReader((out / "contrasts.csv").open())}
    assert con["g-vs-tls-size"]["pairs"] == "6"
    assert float(con["g-vs-tls-size"]["median_a"]) == 11.0
    assert con["g-vs-tls-clipped"]["metric"] == "log_ratio:test_nmse_clipped"
    agg = list(csv.DictReader((out / "aggregate.csv").open()))
    a = {(x["problem"], x["arm"]): x for x in agg}
    assert a[("Con", "BASE-TLS")]["median_size"] == "11" and a[("Con", "BASE-TLS")][
        "median_nodes"] == "9"
    assert a[("Con", "TLS-G")]["median_size"] == "11"  # 11, 9 (guard plain), 11


@pytest.mark.parametrize("key, value", [("size_terms", "yes"), ("run_table", 1),
                                        ("same_search", [["TLS-G"]]),
                                        ("same_search", "TLS-G")])
def test_a_malformed_round13_key_is_refused(key, value, tmp_path):
    root = tmp_path / "r"
    _made_up_runs(root, ["BASE", "BASE-TLS", "TLS-G"], ["Con", "Dow"], [51])
    spec = json.loads(_spec13(tmp_path, []).read_text())
    spec[key] = value
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(spec))
    assert phase_report.main(["--stage", "1", str(p), str(tmp_path / "o"), str(root)]) == 2


# ---- backlog 12.2: the plain fields are opt-in, no skip, descriptive pooled contrasts

PLAIN_FIELDS = ("plain_val_nmse", "plain_test_nmse")
BEFORE_12_2 = "0f6f2eb"  # the commit before the backlog 12.2 changes


@pytest.mark.parametrize("guard", [None, "val"])
def test_without_the_plain_key_a_tls_arm_writes_no_plain_fields(guard, tmp_path):
    kw = {"terms_guard": guard} if guard else {}
    progs = PROGRAMS + ["+($1 $9)"]  # with an unscorable candidate
    off = runner._run_one(_job(tmp_path / "off", TLS, programs=progs, **kw))
    on = runner._run_one(_job(tmp_path / "on", TLS, programs=progs, plain_nmse=True, **kw))
    assert set(on) - set(off) == set(PLAIN_FIELDS) and set(off) <= set(on)
    for k in set(off) - set(VOLATILE):
        assert off[k] == on[k], k
    c_off, c_on = _cands(tmp_path / "off"), _cands(tmp_path / "on")
    assert all(k in c for c in c_on for k in PLAIN_FIELDS)
    assert c_off == [{k: v for k, v in c.items() if k not in PLAIN_FIELDS} for c in c_on]
    if guard:  # TLS-G still compares the plain values: both choices occur
        assert {c["terms"]["guard"] for c in c_off if "terms" in c} == {"plain", "terms"}


@pytest.mark.parametrize("argv", [PLAIN, ["-a", "hc", "--linear-scaling", "1"]])
def test_the_plain_key_changes_nothing_for_an_arm_without_terms(argv, tmp_path):
    off = runner._run_one(_job(tmp_path / "off", argv))
    on = runner._run_one(_job(tmp_path / "on", argv, plain_nmse=True))
    assert set(off) == set(on) and all(off[k] == on[k] for k in set(off) - set(VOLATILE))
    assert (tmp_path / "off" / "run" / "candidates.json").read_bytes() == (
        tmp_path / "on" / "run" / "candidates.json").read_bytes()


def test_a_round12_tls_job_writes_no_new_keys(tmp_path):
    """The BASE-TLS job of the round 12 manifest (phase1bf; no "plain_nmse"), with a fake
    MOSES: candidates.json is byte-identical to that of the runner at cb1ccfe, and the summary
    has the same keys and values (apart from the clock and the paths)."""
    m = json.loads((REPO / "configs" / "gomea" / "phase1bf" / "sr.json").read_text())
    assert "plain_nmse" not in m and "clip_test_nmse" not in m
    job = next(j for j in runner.expand(m, tmp_path / "out", REPO) if j["arm"] == "BASE-TLS")
    assert runner._linear_scaling(job["argv"]) == "terms"
    old = _old_tool(tmp_path, ["moses_sr_runner.py", "sr_eval.py"])
    progs = PROGRAMS + ["+($1 $9)"]  # with an unscorable candidate
    dirs = {}
    for name, tool in (("o", old), ("n", TOOLS)):
        fake = _job(tmp_path / name, job["argv"], programs=progs)
        j = {**job, **{k: fake[k] for k in ("binary", "split_dir", "run_dir")}}
        _run_in(tool, j)
        dirs[name] = Path(j["run_dir"])
    assert (dirs["n"] / "candidates.json").read_bytes() == (
        dirs["o"] / "candidates.json").read_bytes()
    so, sn = (json.loads((dirs[n] / "summary.json").read_text()) for n in ("o", "n"))
    assert set(sn) == set(so)
    assert not set(sn) & {"plain_val_nmse", "plain_test_nmse", "test_nmse_clipped"}
    assert {k: v for k, v in sn.items() if k not in VOLATILE} == {
        k: v for k, v in so.items() if k not in VOLATILE}


def test_expand_carries_the_plain_key_and_refuses_a_wrong_one(tmp_path):
    arm = {"binary": "/x/.venv/bin/moses", "argv": TLS, "budget": {"evals": 10}}
    m = {"name": "t", "datasets_dir": str(tmp_path), "datasets": ["Con"], "splits": [1],
         "wall_cap_s": 10, "arms": {"BASE-TLS": arm, "BASE": {**arm, "argv": PLAIN}}}
    assert all("plain_nmse" not in j for j in runner.expand(m, tmp_path, tmp_path))
    on = runner.expand({**m, "plain_nmse": True}, tmp_path, tmp_path)
    assert len(on) == 2 and all(j["plain_nmse"] is True for j in on)
    off = runner.expand({**m, "plain_nmse": False}, tmp_path, tmp_path)
    assert all("plain_nmse" not in j for j in off)
    for v in ("yes", 1, None):
        with pytest.raises(ValueError, match="plain_nmse must be true or false"):
            runner.expand({**m, "plain_nmse": v}, tmp_path, tmp_path)


def test_a_missing_reference_commit_fails_the_test_and_does_not_skip(monkeypatch, tmp_path):
    monkeypatch.setitem(globals(), "OLD", "0" * 40)
    with pytest.raises(pytest.fail.Exception,
                       match=r"git show 0{40}:tools/gomea/sr_eval.py failed: .*full history"):
        _old_tool(tmp_path, ["sr_eval.py"])


def _spec_descriptive(tmp_path: Path, descr=None) -> Path:
    spec = {"phase": "t", "metrics": {"Con": "test_nmse", "Dow": "test_nmse"}, "primary": [],
            "pooled_primary": [["tls-vs-base", "BASE-TLS", "BASE", "test_nmse"]],
            "pooled_secondary": [["g-vs-base", "TLS-G", "BASE", "test_nmse"],
                                 ["g-vs-tls", "TLS-G", "BASE-TLS", "test_nmse"],
                                 ["g-vs-base-cpu", "TLS-G", "BASE", "cpu_s"]],
            "pooled_secondary_families": [["g-vs-base", "g-vs-tls"]], "extend": False}
    if descr is not None:
        spec["pooled_descriptive"] = descr
    p = tmp_path / ("spec.json" if descr is None else "spec-descr.json")
    p.write_text(json.dumps(spec))
    return p


def test_descriptive_pooled_contrasts_keep_p_holm_and_ratio_but_get_no_verdict(tmp_path):
    root = tmp_path / "r"
    _made_up_runs(root, ["BASE", "BASE-TLS", "TLS-G"], ["Con", "Dow"], list(range(51, 61)))
    out = {}
    for name, descr in (("plain", None), ("descr", ["g-vs-base", "g-vs-base-cpu"])):
        out[name] = tmp_path / name
        assert phase_report.main(["--stage", "1", str(_spec_descriptive(tmp_path, descr)),
                                  str(out[name]), str(root)]) == 0
    named = {"g-vs-base", "g-vs-base-cpu"}
    a = {r["contrast"]: r for r in csv.DictReader((out["plain"] / "contrasts.csv").open())}
    b = {r["contrast"]: r for r in csv.DictReader((out["descr"] / "contrasts.csv").open())}
    assert a.keys() == b.keys()
    for name in a:
        if name in named:
            assert a[name]["verdict"] == "A worse"  # a verdict without the key
            assert b[name]["verdict"] == "descriptive"
            assert {k: v for k, v in b[name].items() if k != "verdict"} == {
                k: v for k, v in a[name].items() if k != "verdict"}  # p, p_holm, ratio
            assert b[name]["p_holm"] and b[name]["ratio"]
        else:
            assert b[name] == a[name]
    va = json.loads((out["plain"] / "verdict.json").read_text())
    vb = json.loads((out["descr"] / "verdict.json").read_text())
    assert vb == {k: ("descriptive" if k.split(":", 1)[1] in named else v)
                  for k, v in va.items()}
    assert (out["descr"] / "aggregate.csv").read_bytes() == (
        out["plain"] / "aggregate.csv").read_bytes()


@pytest.mark.parametrize("descr", [["g-vs-base", "g-vs-base"], ["nope"], ["tls-vs-base"],
                                   "g-vs-base", [["g-vs-base"]], [1]])
def test_a_malformed_pooled_descriptive_key_is_refused(descr, tmp_path, capsys):
    root = tmp_path / "r"
    _made_up_runs(root, ["BASE", "BASE-TLS", "TLS-G"], ["Con", "Dow"], [51, 52])
    assert phase_report.main(["--stage", "1", str(_spec_descriptive(tmp_path, descr)),
                              str(tmp_path / "o"), str(root)]) == 2
    assert "bad pooled_descriptive" in capsys.readouterr().err


def test_the_round13_spec_gives_the_report_of_0f6f2eb(tmp_path):
    """The 1bh spec of round 13 has no "pooled_descriptive" (it records what ran): every file
    and the stdout of the report are those of the report at 0f6f2eb."""
    spec = REPO / "configs" / "gomea" / "phase1bh" / "spec.json"
    sj = json.loads(spec.read_text())
    assert "pooled_descriptive" not in sj
    arms = list(json.loads((REPO / "configs" / "gomea" / "phase1bh" / "sr.json").read_text())[
        "arms"])
    arms += [a for a in json.loads((REPO / "configs" / "gomea" / "phase1bh" /
                                    "sr-Dow.json").read_text())["arms"] if a not in arms]
    root = tmp_path / "r"
    _made_up_runs(root, arms, list(sj["metrics"]), list(range(51, 56)))
    old = _old_tool(tmp_path, ["phase_report.py", "run_table.py"], commit=BEFORE_12_2)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    outs, stdout = {}, {}
    for name, tool in (("old", old / "phase_report.py"), ("new", TOOLS / "phase_report.py")):
        outs[name] = tmp_path / f"out-{name}"
        r = subprocess.run([sys.executable, str(tool), "--stage", "1", str(spec),
                            str(outs[name]), str(root)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        stdout[name] = r.stdout
    files = sorted(p.name for p in outs["old"].iterdir())
    assert sorted(p.name for p in outs["new"].iterdir()) == files
    assert {"contrasts.csv", "verdict.json", "runs.csv", "same_search.csv"} <= set(files)
    for f in files:
        assert (outs["new"] / f).read_bytes() == (outs["old"] / f).read_bytes(), f
    assert stdout["new"] == stdout["old"]
