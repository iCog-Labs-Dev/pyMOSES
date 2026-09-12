#!/usr/bin/env python3
"""Differential CLI sweep: run classic MOSES and the port on the same argv and diff.

This replaces the first-generation sweep, whose hole is documented at length in
``tools/flag_sweep_spec.py``.  The short version: it guessed values from a fixed
ladder (``1 2 0.5 3 0 tanimoto hc``), kept whichever classic exited 0 on, and
printed a quiet ``classic-rejects-all-candidates`` line for the six options where
no ladder value was legal.  Those six were never compared -- and one of them,
``--output-format=scheme``, was a real divergence.  It also only ever looked at
long options with no short form, so 49 further options were never in the sweep.

Four properties this version guarantees, each aimed at one way the old one lied:

* **Value-aware.**  Every option carries its own list of values (legal *and*
  deliberately illegal), read out of ``moses --help`` and
  ``moses/moses/main/problem-params.cc``.  Nothing is guessed from a ladder.
* **Loud on a dead flag.**  If classic rejects *every* value listed for an
  option, that is a ``NO-LEGAL-VALUE`` finding printed in a banner and the run
  exits non-zero.  It is never a quiet bucket.
* **Empty output is not agreement.**  Two empty stdouts compare equal; that has
  produced a false clean bill in this project before.  A pair where both sides
  exit 0 with no output is classified ``BOTH-EMPTY`` and reported separately from
  ``IDENTICAL``.
* **Gated options are run inside their gate.**  ``--fs-algo`` means nothing
  without ``--enable-fs=1``; the old sweep scored four such options ``IDENTICAL``
  while exercising none of them.  ``FlagSpec.needs`` carries the context.

Every classic invocation is bounded -- literally
``bash -c 'ulimit -v 2097152; ulimit -c 0; exec timeout 300 "$@"'`` -- because
unbounded classic runs have OOM'd this machine.  Argument vectors are Python
lists handed straight to ``subprocess`` and then to bash's ``"$@"``, so there is
no word-splitting step that can silently drop an argument.

Usage
-----
    python tools/flag_sweep.py --classic /path/to/moses --out /tmp/sweep
    python tools/flag_sweep.py --classic ... --out ... --mode combos
    python tools/flag_sweep.py --classic ... --out ... --only 'output-format'

``--port-root`` should point at a clean ``git archive HEAD | tar -x`` export, not
at the working tree; a dirty tree has produced a screenful of false "port fails"
here before.  It defaults to the repository this file lives in.

Exit status is 1 when there is anything to look at (a finding or a
NO-LEGAL-VALUE option), 0 only when the sweep ran cases and all of them agreed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flag_sweep_spec import COMBOS, SPECS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSIC_AS_KB = 2097152  # `ulimit -v 2097152` == 2 GiB
PORT_AS_KB = 6291456  # CPython's own arenas need headroom
CLASSIC_TIMEOUT = 300
PORT_TIMEOUT = 300

MAX_EVALS = "200"
SEED = "1"

# Small, fixed tables: a pair of runs stays under a second on both sides, and
# the boolean one is the same shape as classic's own tests/moses/dataset.csv.
DATASETS = {
    "bool": ("res,arg1,arg2\n0,0,1\n0,1,0\n1,1,1\n0,0,1\n0,1,1\n1,1,1\n0,0,0\n1,1,0\n", "res"),
    "ts": (
        "i1,i2,when,w,y\n"
        "0,0,2012-01-03,1,0\n"
        "0,1,2012-01-11,2,0\n"
        "1,0,2012-01-19,1,0\n"
        "1,1,2012-02-03,3,1\n"
        "0,0,2012-02-11,1,0\n"
        "0,1,2012-02-19,2,1\n"
        "1,0,2012-03-03,1,0\n"
        "1,1,2012-03-11,2,1\n"
        "0,1,2012-03-19,1,0\n"
        "1,1,2012-04-03,4,1\n",
        "y",
    ),
    "wide": (
        "a1,a2,a3,a4,a5,a6,res\n"
        + "".join(
            ",".join(str((i >> k) & 1) for k in range(5, -1, -1))
            + f",{1 if (((i >> 5) & 1) and ((i >> 4) & 1)) or ((i >> 1) & 1) else 0}\n"
            for i in range(64)
        ),
        "res",
    ),
    "ts_wide": (
        "a1,a2,a3,a4,when,res\n"
        + "".join(
            ",".join(str((i >> k) & 1) for k in range(3, -1, -1))
            + f",2012-{1 + i % 12:02d}-{1 + i % 28:02d}"
            + f",{1 if (((i >> 3) & 1) and ((i >> 2) & 1)) or (i & 1) else 0}\n"
            for i in range(16)
        ),
        "res",
    ),
    # -Hip demands a contin output with boolean inputs (table-problems.cc:236).
    "ip": (
        "a1,a2,a3,y\n"
        + "".join(
            ",".join(str((i >> k) & 1) for k in range(2, -1, -1))
            + f",{((i >> 2) & 1) * 2.5 + ((i >> 1) & 1) - (i & 1) * 0.5:.2f}\n"
            for i in range(8)
        ),
        "y",
    ),
    "contin": (
        "x1,x2,y\n"
        "0.0,1.0,1.0\n"
        "1.0,0.0,1.0\n"
        "1.0,1.0,2.0\n"
        "2.0,1.0,3.0\n"
        "0.5,0.5,1.0\n"
        "2.0,2.0,4.0\n"
        "3.0,1.0,4.0\n"
        "1.0,3.0,4.0\n",
        "y",
    ),
}

# Base argv, keyed by the long option name so a case that sweeps the same option
# can drop it.  `--input-file` is repeatable in classic, so it is never dropped.
BASE_TOKENS = (
    ("--input-file", "-i{data}"),
    ("--target-feature", "-u{target}"),
    ("--max-evals", f"-m{MAX_EVALS}"),
    ("--random-seed", f"-r{SEED}"),
    ("--log-file", "-f{log}"),
    ("--output-with-labels", "-W1"),
)
NEVER_DROP = {"--input-file"}

FINDINGS = ("NO-LEGAL-VALUE", "BOTH-OK-DIFFER", "CLASSIC-OK-PORT-FAILS", "PORT-OK-CLASSIC-FAILS")
BENIGN = ("IDENTICAL", "BOTH-EMPTY", "BOTH-FAIL-SAME-RC", "BOTH-FAIL-DIFF-RC", "SKIPPED")


@dataclass
class RunResult:
    rc: int | str
    stdout: str
    stderr: str


@dataclass
class Case:
    flag: str
    value: str
    context: str
    dataset: str
    argv_tail: tuple[str, ...]
    combo: str = ""


@dataclass
class Verdict:
    flag: str
    value: str
    context: str
    kind: str
    classic_rc: int | str
    port_rc: int | str
    detail: str
    reproducer: str
    combo: str = ""
    classic_fingerprint: str = ""


def bounded(argv: list[str], *, as_kb: int, timeout: int) -> list[str]:
    """Wrap argv in the mandated `( ulimit -v N; timeout T cmd )` bound.

    A bash `"$@"` array is used rather than an inline expansion: the interactive
    shell here is zsh, which does not word-split unquoted expansions, and an
    `cmd $args` loop has silently passed a single argument -- and reported false
    agreement -- three times in this project.
    """
    script = f'ulimit -v {as_kb}; ulimit -c 0; exec timeout {timeout} "$@"'
    return ["/bin/bash", "-c", script, "_", *argv]


def _run(argv: list[str], *, cwd: Path, as_kb: int, timeout: int) -> RunResult:
    wrapped = bounded(argv, as_kb=as_kb, timeout=timeout)
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, no shell interpolation
            wrapped,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return RunResult("TIMEOUT", "", "")
    except OSError as exc:
        return RunResult("SPAWN-ERROR", "", str(exc))
    rc: int | str = "TIMEOUT" if proc.returncode == 124 else proc.returncode
    return RunResult(rc, proc.stdout, proc.stderr)


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:240] if lines else ""


def _first_diff(a: str, b: str) -> str:
    al, bl = a.splitlines(), b.splitlines()
    for i in range(max(len(al), len(bl))):
        x = al[i] if i < len(al) else "<eof>"
        y = bl[i] if i < len(bl) else "<eof>"
        if x != y:
            return f"line {i + 1}: classic={x[:110]!r} port={y[:110]!r}"
    return "(same lines, differing trailing bytes)"


def _tail_option_names(tail: tuple[str, ...]) -> set[str]:
    names = set()
    for tok in tail:
        if tok.startswith("--"):
            names.add(tok.split("=", 1)[0])
    return names


def build_cases(only: re.Pattern | None) -> list[Case]:
    """One Case per (option, value); every value of every option, no ladder."""
    cases: list[Case] = []
    for spec in SPECS:
        if spec.skip:
            continue
        if only and not only.search(spec.flag):
            continue
        for value in spec.values:
            # A `needs` token naming the option under test would pass it twice,
            # and boost rejects a repeated scalar option with multiple_occurrences
            # -> OC_ASSERT -> SIGABRT.  That would be a fabricated finding, so drop
            # the collision -- by long name (--fs-target-size lives in FS) and by
            # short name (PRE carries -q0.1 and -q is itself swept).
            tail = [
                n
                for n in spec.needs
                if n.split("=", 1)[0] != spec.flag and not (spec.short and n.startswith(spec.short))
            ]
            tail.append(spec.flag if spec.kind == "novalue" else f"{spec.flag}={value}")
            cases.append(
                Case(
                    flag=spec.flag,
                    value=value,
                    context=" ".join(spec.needs),
                    dataset=spec.dataset,
                    argv_tail=tuple(tail),
                )
            )
    return cases


def build_combo_cases(only: re.Pattern | None) -> list[Case]:
    cases: list[Case] = []
    for combo in COMBOS:
        if only and not only.search(combo.name):
            continue
        names = [axis[0] for axis in combo.axes]
        needs = [n for n in combo.needs if n.split("=", 1)[0] not in names]
        for values in itertools.product(*[axis[1] for axis in combo.axes]):
            tail = needs + [f"{n}={v}" for n, v in zip(names, values, strict=True)]
            cases.append(
                Case(
                    flag="+".join(names),
                    value=",".join(values),
                    context=" ".join(combo.needs),
                    dataset=combo.dataset,
                    argv_tail=tuple(tail),
                    combo=combo.name,
                )
            )
    return cases


class Sweeper:
    def __init__(self, classic: Path, port_root: Path, out: Path, python: Path) -> None:
        self.classic = classic
        self.port_root = port_root
        self.out = out
        self.python = python
        self.out.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.out / "data"
        self.data_dir.mkdir(exist_ok=True)
        for name, (body, _target) in DATASETS.items():
            (self.data_dir / f"{name}.csv").write_text(body)
        (self.data_dir / "features.txt").write_text("arg1\n")
        self.runner = self.out / "_run_port.py"
        self.runner.write_text(
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from pymetta.moses.main.moses_exec import moses_exec\n"
            "sys.exit(moses_exec(sys.argv[1:]))\n"
        )

    def _expand(self, value: str, dataset: str, workdir: Path) -> str:
        return (
            value.replace("@DATA@", str(self.data_dir / f"{dataset}.csv"))
            .replace("@MISSING@", str(self.data_dir / "definitely-not-here.csv"))
            .replace("@OUTFILE@", str(workdir / "out.txt"))
            .replace("@LOGFILE@", str(workdir / "explicit.log"))
            .replace("@FEATFILE@", str(self.data_dir / "features.txt"))
            .replace("@TARGET@", DATASETS[dataset][1])
        )

    def _base(self, case: Case, workdir: Path) -> list[str]:
        swept = _tail_option_names(case.argv_tail) | set(case.flag.split("+"))
        subs = {
            "data": str(self.data_dir / f"{case.dataset}.csv"),
            "target": DATASETS[case.dataset][1],
            "log": str(workdir / "moses.log"),
        }
        return [
            tok.format(**subs)
            for name, tok in BASE_TOKENS
            if name in NEVER_DROP or name not in swept
        ]

    def _one(self, case: Case) -> Verdict:
        with tempfile.TemporaryDirectory(dir=self.out) as tmp:
            workdir = Path(tmp)
            tail = [self._expand(t, case.dataset, workdir) for t in case.argv_tail]
            argv = self._base(case, workdir) + tail
            to_file = "@OUTFILE@" in " ".join(case.argv_tail)
            outfile = workdir / "out.txt"

            c = _run(
                [str(self.classic), *argv],
                cwd=workdir,
                as_kb=CLASSIC_AS_KB,
                timeout=CLASSIC_TIMEOUT,
            )
            cout = c.stdout
            if to_file:
                cout = outfile.read_text() if outfile.exists() else ""
                outfile.unlink(missing_ok=True)

            p = _run(
                [str(self.python), "-W", "ignore", str(self.runner), str(self.port_root), *argv],
                cwd=workdir,
                as_kb=PORT_AS_KB,
                timeout=PORT_TIMEOUT,
            )
            pout = p.stdout
            if to_file:
                pout = outfile.read_text() if outfile.exists() else ""

        repro = " ".join(["moses", *argv])
        cok, pok = c.rc == 0, p.rc == 0
        if cok and pok:
            if cout == pout:
                kind = "IDENTICAL" if cout.strip() else "BOTH-EMPTY"
                detail = "" if cout.strip() else "both exited 0 with empty output -- proves nothing"
            else:
                kind, detail = "BOTH-OK-DIFFER", _first_diff(cout, pout)
        elif cok and not pok:
            kind, detail = "CLASSIC-OK-PORT-FAILS", _last_line(p.stderr)
        elif pok and not cok:
            kind, detail = "PORT-OK-CLASSIC-FAILS", _last_line(c.stderr)
        elif c.rc == p.rc:
            kind, detail = "BOTH-FAIL-SAME-RC", _last_line(p.stderr)
        else:
            kind = "BOTH-FAIL-DIFF-RC"
            detail = f"classic: {_last_line(c.stderr)} || port: {_last_line(p.stderr)}"
        return Verdict(
            flag=case.flag,
            value=case.value,
            context=case.context,
            kind=kind,
            classic_rc=c.rc,
            port_rc=p.rc,
            detail=detail,
            reproducer=repro,
            combo=case.combo,
            classic_fingerprint=hashlib.sha1(cout.encode()).hexdigest()[:12],
        )

    def run(self, cases: list[Case], jobs: int) -> list[Verdict]:
        verdicts: list[Verdict] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for i, v in enumerate(pool.map(self._one, cases), 1):
                verdicts.append(v)
                print(f"[{i:4d}/{len(cases)}] {v.kind:22s} {v.flag}={v.value}", flush=True)
        return verdicts


def skipped_verdicts(only: re.Pattern | None) -> list[Verdict]:
    return [
        Verdict(s.flag, "", "", "SKIPPED", "-", "-", s.skip, "", "")
        for s in SPECS
        if s.skip and (not only or only.search(s.flag))
    ]


def no_legal_value(verdicts: list[Verdict]) -> list[str]:
    """Options where classic rejected every single value the spec lists.

    The old sweep printed this as a quiet ``classic-rejects-all-candidates``
    line and moved on, which is how ``--output-format=scheme`` stayed hidden.
    Here it is a finding: either the value list is wrong or the option cannot be
    driven, and a human has to decide which.
    """
    by_flag: dict[str, list[Verdict]] = {}
    for v in verdicts:
        if v.combo or v.kind == "SKIPPED":
            continue
        by_flag.setdefault(v.flag, []).append(v)
    return sorted(f for f, vs in by_flag.items() if all(v.classic_rc != 0 for v in vs))


ACKNOWLEDGED_DEAD = {s.flag: s.known_dead for s in SPECS if s.known_dead}


def inert_flags(verdicts: list[Verdict]) -> list[str]:
    """Options whose value made no difference to *classic* at all.

    Every verdict for such an option is an ``IDENTICAL`` that proves nothing:
    the two implementations agreed on an output the option never influenced.
    This is the "a sweep that finds nothing should be checked for whether it
    could have found anything" guard.  It is reported, not failed -- plenty of
    options genuinely cannot move a 200-evaluation boolean run -- but it tells a
    reader which rows of the table are evidence and which are filler.
    """
    prints: dict[str, set[str]] = {}
    any_ok: dict[str, bool] = {}
    n: dict[str, int] = {}
    for v in verdicts:
        if v.combo or v.kind == "SKIPPED":
            continue
        prints.setdefault(v.flag, set()).add(v.classic_fingerprint)
        any_ok[v.flag] = any_ok.get(v.flag, False) or v.classic_rc == 0
        n[v.flag] = n.get(v.flag, 0) + 1
    return sorted(f for f, h in prints.items() if len(h) == 1 and any_ok[f] and n[f] > 1)


def report(verdicts: list[Verdict], dead: list[str], out: Path) -> int:
    (out / "flag_sweep.json").write_text(json.dumps([asdict(v) for v in verdicts], indent=2))

    lines: list[str] = []
    add = lines.append
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.kind] = counts.get(v.kind, 0) + 1

    add("=" * 78)
    add("FLAG SWEEP SUMMARY")
    add("=" * 78)
    for kind in (*FINDINGS, *BENIGN, *sorted(set(counts) - set(FINDINGS) - set(BENIGN))):
        if kind in counts:
            add(f"  {kind:24s} {counts[kind]:5d}")
    add(f"  {'TOTAL':24s} {len(verdicts):5d}")
    add("")

    acked = [f for f in dead if f in ACKNOWLEDGED_DEAD]
    unacked = [f for f in dead if f not in ACKNOWLEDGED_DEAD]
    if unacked:
        add("!" * 78)
        add("!! NO-LEGAL-VALUE -- classic rejected EVERY value this sweep knows for:")
        for flag in unacked:
            add(f"!!   {flag}")
        add("!! Either flag_sweep_spec.py's value list is wrong, the context in `needs`")
        add("!! is wrong, or the option cannot be driven at all.  This is the exact")
        add("!! failure mode the old sweep reported silently as")
        add("!! 'classic-rejects-all-candidates', and it hid a real divergence.")
        add("!! Fix it, or write the reason into that option's `known_dead`.")
        add("!" * 78)
        add("")
    if acked:
        add("-" * 78)
        add(f"NO-LEGAL-VALUE, acknowledged ({len(acked)}) -- classic cannot be made to")
        add("accept any value of these, and somebody has written down why:")
        for flag in acked:
            add(f"  {flag}: {ACKNOWLEDGED_DEAD[flag]}")
        add("")

    for kind in FINDINGS:
        hits = [v for v in verdicts if v.kind == kind]
        if not hits:
            continue
        add("-" * 78)
        add(f"{kind}  ({len(hits)})")
        add("-" * 78)
        for v in hits:
            label = f"{v.combo}: " if v.combo else ""
            add(f"  {label}{v.flag}={v.value}   (classic rc={v.classic_rc} port rc={v.port_rc})")
            if v.context:
                add(f"      context: {v.context}")
            if v.detail:
                add(f"      {v.detail}")
            add(f"      repro: {v.reproducer}")
            add("")

    inert = inert_flags(verdicts)
    if inert:
        add("-" * 78)
        add(f"INERT ({len(inert)}) -- classic produced the SAME output for every value of")
        add("these options, so their IDENTICAL verdicts record agreement on an output the")
        add("option never influenced.  Not a failure; a note on what the table proves.")
        for flag in inert:
            add(f"  {flag}")
        add("")

    empties = [v for v in verdicts if v.kind == "BOTH-EMPTY"]
    if empties:
        add("-" * 78)
        add(f"BOTH-EMPTY ({len(empties)}) -- agreement on nothing is not agreement")
        for v in empties:
            add(f"  {v.flag}={v.value}")
        add("")

    text = "\n".join(lines)
    (out / "flag_sweep.txt").write_text(text)
    print(text)
    findings = sum(counts.get(k, 0) for k in FINDINGS)
    return 1 if (unacked or findings) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="value-aware differential CLI sweep of classic MOSES vs the port"
    )
    ap.add_argument("--classic", required=True, type=Path, help="path to the classic moses binary")
    ap.add_argument(
        "--port-root",
        type=Path,
        default=REPO_ROOT,
        help="importable root of the port; use a clean `git archive` export, not the work tree",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--mode", choices=("values", "combos", "all"), default="all")
    ap.add_argument("--only", default="", help="regex filter on option (or combo) name")
    ap.add_argument("--jobs", type=int, default=4)
    ns = ap.parse_args(argv)

    if not ns.classic.exists():
        print(f"classic binary not found: {ns.classic}", file=sys.stderr)
        return 2

    only = re.compile(ns.only) if ns.only else None
    sw = Sweeper(ns.classic, ns.port_root.resolve(), ns.out.resolve(), ns.python)

    cases: list[Case] = []
    if ns.mode in ("values", "all"):
        cases += build_cases(only)
    if ns.mode in ("combos", "all"):
        cases += build_combo_cases(only)
    if not cases:
        print("no cases selected -- a sweep that runs nothing finds nothing", file=sys.stderr)
        return 2

    print(f"{len(cases)} flag/value pairs to run ({ns.jobs} at a time)", flush=True)
    verdicts = sw.run(cases, ns.jobs) + skipped_verdicts(only)
    return report(verdicts, no_legal_value(verdicts), ns.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
