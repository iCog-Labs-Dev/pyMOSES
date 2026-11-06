"""Choose the GOMEA population of rounds 7 and 8 (POP) and of 1az (DYN) from the final 1ay
report, and write population-resolved manifests (server round 6).

Usage (from the repository root; the plans run it as `choice` steps)::

    .venv/bin/python tools/gomea/set_population.py choose --out DIR [--results DIR]
        [--configs DIR] [--source PHASE] [--target PHASE] [--dry-run] [--force]
    .venv/bin/python tools/gomea/set_population.py apply --choice FILE --target PHASE
        --out DIR [--configs DIR] [--force]

`SERVER-ROUND6-PREREGISTRATION.md`, 1ay:

* **POP (rounds 7 and 8):** a dynamic fraction qualifies if its LT-B arm is better than N16-LTB
  on at least 2 of the 3 problems and worse on none. If several qualify, POP is the one with the
  best median rank. If none qualifies, POP is N = 16.
* **DYN (1az and the control arms):** the dynamic fraction whose LT-B arm has the best median
  rank, whether or not it qualifies.
* **Median rank:** on each problem, rank the 3 dynamic LT-B arms by their median metric (1 =
  best); then take the median of the 3 ranks. Ties go to the smaller F.
* **Control arms (rounds 7 and 8):** the population DYN, or N = 16 if POP equals DYN.

`choose`

1. reads the final report of SOURCE (`RESULTS/SOURCE/report-stage{1,2}/`), with the rule of
   `set_final.final_report`: stage 2 if a primary contrast was "extend" at stage 1, and only an
   "extend" contrast takes its stage-2 verdict;
2. takes the dynamic LT-B arms from the primary contrasts of SOURCE (arm A of each contrast, arm
   B being the N = 16 reference), and the fraction F of each arm from its argv in
   `CONFIGS/SOURCE/sr.json` (`--hc-fraction-of-nn`; `bool.json` must agree);
3. counts per arm the problems with the final verdict "A better" and "A worse";
4. ranks the arms on each problem by `median_metric` of the final report's `aggregate.csv` (a
   failed or unsolved run counts as infinite there; equal medians rank the smaller F first),
   and takes the median of the ranks;
5. applies the two rules (equal median ranks go to the smaller F);
6. writes DIR/population-choice.json (the rules, the numbers, POP, DYN and the control) and the
   resolved manifests of TARGET (see "Templates").

`apply` resolves the templates of TARGET with the POP, DYN and control of an earlier
population-choice.json, and writes DIR/population-applied.json (the record of that step).

**Templates.** A phase whose argv depends on the choice has `template.json` (resolved to
`sr.json`) and `template-<name>.json` (resolved to `<name>.json`, for example
`template-sr-ext-Con.json` to `sr-ext-Con.json` and `template-bool.json` to `bool.json`) and a
`spec.json` (copied). In an argv, the token `{POP}`, `{DYN}` or `{CONTROL}` becomes the flags
of that population: `--gomea-pop-size 16`, or `--gomea-pop-size dynamic --hc-fraction-of-nn F`.
An arm with `"argv": null` stays null (set_final.py fills FINAL and FINAL-DYN of 1ax).

DIR is in a round's state directory, and phase_run.sh reads the phase from there
(PHASE_CONFIG_DIR). So the choice never changes a tracked file. The script exits 1 and writes
nothing if a report or manifest is missing or unreadable, if an arm has no median on a problem,
if a resolved argv still has a `{...}` token, or if DIR/sr.json exists with a different content
(`--force` replaces it). RESULTS defaults to `../gomea-results`, CONFIGS to `configs/gomea`,
SOURCE to `phase1ay`, TARGET to `phase1az`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from set_final import ChoiceError, _json_safe, _read_json, _read_rows, final_report  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

N16 = {"name": "N16", "fraction": None, "argv": ["--gomea-pop-size", "16"]}
TOKENS = ("{POP}", "{DYN}", "{CONTROL}")
POP_FLAG, FRAC_FLAG = "--gomea-pop-size", "--hc-fraction-of-nn"
MIN_BETTER = 2
RULES = {
    "pop": "a dynamic fraction qualifies if its LT-B arm is 'A better' than N16-LTB on at least "
    f"{MIN_BETTER} of the 3 problems and 'A worse' on none; of several, the best median rank; "
    "if none qualifies, POP = N16",
    "dyn": "the dynamic fraction whose LT-B arm has the best median rank, qualified or not",
    "median_rank": "on each problem, rank the 3 dynamic LT-B arms by median metric (1 = best; "
    "equal medians: the smaller F first); the median of the 3 ranks; ties go to the smaller F",
    "control": "the population DYN, or N16 if POP equals DYN",
}


def population(fraction: float | None) -> dict:
    """The population setting of a fraction (None: N = 16)."""
    if fraction is None:
        return dict(N16)
    return {"name": f"DYN{fraction:g}", "fraction": fraction,
            "argv": [POP_FLAG, "dynamic", FRAC_FLAG, repr(float(fraction))]}


def fraction_of(argv: list[str]) -> float | None:
    """F of an argv with `--gomea-pop-size dynamic`; None for a fixed N."""
    if POP_FLAG not in argv:
        raise ChoiceError(f"no {POP_FLAG} in {argv}")
    if argv[argv.index(POP_FLAG) + 1] != "dynamic":
        return None
    if FRAC_FLAG not in argv:
        raise ChoiceError(f"{POP_FLAG} dynamic without {FRAC_FLAG} in {argv}")
    return float(argv[argv.index(FRAC_FLAG) + 1])


def swap_population(argv: list[str], pop: dict) -> list[str]:
    """`argv` with its population flags replaced by those of `pop`, at the same place."""
    if POP_FLAG not in argv:
        raise ChoiceError(f"no {POP_FLAG} in {argv}")
    i = argv.index(POP_FLAG)
    j = i + 2
    if argv[i + 1] == "dynamic":
        if argv[j:j + 1] != [FRAC_FLAG]:
            raise ChoiceError(f"want {FRAC_FLAG} right after {POP_FLAG} dynamic in {argv}")
        j += 2
    out = argv[:i] + pop["argv"] + argv[j:]
    if out.count(POP_FLAG) != 1 or out.count(FRAC_FLAG) > 1:
        raise ChoiceError(f"population flags repeated in {argv}")
    return out


def _median_metric(rows: list[dict], problem: str, arm: str, report: str) -> float:
    for r in rows:
        if r.get("problem") == problem and r.get("arm") == arm:
            v = r.get("median_metric", "")
            if v in ("", None):
                break
            return float(v)
    raise ChoiceError(f"no median_metric for {problem} {arm} in {report}/aggregate.csv")


def evaluate(rep: dict, src_arms: dict, bool_arms: dict | None) -> dict:
    """The numbers of the two rules and the choice."""
    spec, verdicts = rep["spec"], rep["verdicts"]
    problems = list(dict.fromkeys(c[0] for c in spec["primary"]))
    arms = list(dict.fromkeys(c[2] for c in spec["primary"]))
    refs = {c[3] for c in spec["primary"]}
    if len(refs) != 1 or len(arms) != 3 or len(problems) != 3:
        raise ChoiceError(f"spec: want 3 dynamic arms against one reference on 3 problems, got "
                          f"{arms} against {sorted(refs)} on {problems}")
    ref = refs.pop()
    if fraction_of(src_arms[ref]["argv"]) is not None:
        raise ChoiceError(f"the reference {ref} has a dynamic population")
    frac = {}
    for a in arms:
        f = fraction_of(src_arms[a]["argv"])
        if f is None:
            raise ChoiceError(f"{a} has no dynamic population")
        if bool_arms is not None and fraction_of(bool_arms[a]["argv"]) != f:
            raise ChoiceError(f"{a}: sr.json and bool.json give different fractions")
        frac[a] = f
    agg = _read_rows(Path(rep["report"]) / "aggregate.csv")
    per_arm = {a: {"fraction": frac[a], "better": [], "worse": [], "ranks": {}, "medians": {}}
               for a in arms}
    for p in problems:
        for c in spec["primary"]:
            if c[0] != p:
                continue
            v = verdicts.get(f"{p}:{c[1]}")
            if v is None:
                raise ChoiceError(f"no verdict for {p}:{c[1]} in {rep['report']}")
            per_arm[c[2]].setdefault("verdicts", {})[p] = v
            if v == "A better":
                per_arm[c[2]]["better"].append(p)
            elif v == "A worse":
                per_arm[c[2]]["worse"].append(p)
        med = {a: _median_metric(agg, p, a, rep["report"]) for a in arms}
        med_ref = _median_metric(agg, p, ref, rep["report"])
        for k, a in enumerate(sorted(arms, key=lambda a: (med[a], frac[a])), start=1):
            per_arm[a]["ranks"][p] = k
            per_arm[a]["medians"][p] = med[a]
        for a in arms:
            per_arm[a].setdefault("reference_medians", {})[p] = med_ref
    for a in arms:
        d = per_arm[a]
        d["median_rank"] = statistics.median(d["ranks"].values())
        d["qualifies"] = len(d["better"]) >= MIN_BETTER and not d["worse"]
    order = sorted(arms, key=lambda a: (per_arm[a]["median_rank"], frac[a]))
    dyn_arm = order[0]
    qualified = [a for a in order if per_arm[a]["qualifies"]]
    pop_arm = qualified[0] if qualified else None
    pop = population(frac[pop_arm]) if pop_arm else population(None)
    dyn = population(frac[dyn_arm])
    control = population(None) if pop == dyn else dyn
    return {"reference": ref, "problems": problems, "per_arm": per_arm,
            "pop_arm": pop_arm, "dyn_arm": dyn_arm, "pop": pop, "dyn": dyn, "control": control}


def resolve_argv(argv: list[str] | None, choice: dict) -> list[str] | None:
    if argv is None:
        return None
    pops = {"{POP}": choice["pop"], "{DYN}": choice["dyn"], "{CONTROL}": choice["control"]}
    out: list[str] = []
    for t in argv:
        out += pops[t]["argv"] if t in pops else [t]
    left = [t for t in out if "{" in t or "}" in t]
    if left:
        raise ChoiceError(f"unresolved token(s) {left} in {argv}")
    return out


def resolve_dir(configs: Path, target: str, choice: dict) -> dict[str, str]:
    """{file name: text} of the resolved manifests and the spec of TARGET."""
    q = configs / target
    templates = sorted(q.glob("template*.json"))
    if not templates:
        raise ChoiceError(f"no template*.json in {q}")
    out = {"spec.json": (q / "spec.json").read_text()}
    for t in templates:
        name = "sr.json" if t.name == "template.json" else t.name[len("template-"):]
        m = _read_json(t)
        for spec in m["arms"].values():
            spec["argv"] = resolve_argv(spec.get("argv"), choice)
        out[name] = json.dumps(m, indent=2) + "\n"
    if "sr.json" not in out:
        raise ChoiceError(f"{q} has no template.json (the resolved sr.json)")
    return out


def write_dir(out: Path, files: dict[str, str], record_name: str, record: dict,
              force: bool) -> None:
    sr = out / "sr.json"
    if sr.exists() and sr.read_text() != files["sr.json"] and not force:
        raise ChoiceError(f"{sr} exists with another content; nothing written "
                          "(--force replaces it)")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    (out / record_name).write_text(json.dumps(_json_safe(record), indent=2) + "\n")


def _print_choice(ev: dict) -> None:
    for a, d in ev["per_arm"].items():
        meds = ", ".join(f"{p} {d['medians'][p]:.6g} (rank {d['ranks'][p]}, N16 "
                         f"{d['reference_medians'][p]:.6g}, {d['verdicts'][p]})"
                         for p in ev["problems"])
        print(f"{a} (F = {d['fraction']:g}): {meds}; median rank {d['median_rank']:g}; "
              f"better on {len(d['better'])}, worse on {len(d['worse'])}; "
              f"{'qualifies' if d['qualifies'] else 'does not qualify'}")
    print(f"-> POP = {ev['pop']['name']} ({' '.join(ev['pop']['argv'])}); "
          f"DYN = {ev['dyn']['name']} ({' '.join(ev['dyn']['argv'])}); "
          f"control = {ev['control']['name']}")


def choose(a) -> int:
    results, configs, out = Path(a.results).resolve(), Path(a.configs).resolve(), Path(a.out)
    try:
        rep = final_report(results, configs, a.source)
        src = _read_json(configs / a.source / "sr.json")["arms"]
        bpath = configs / a.source / "bool.json"
        barms = _read_json(bpath)["arms"] if bpath.exists() else None
        ev = evaluate(rep, src, barms)
        files = resolve_dir(configs, a.target, ev)
    except (ChoiceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"set_population: {exc}", file=sys.stderr)
        print("set_population: nothing written", file=sys.stderr)
        return 1
    print(f"report stage {rep['stage']} ({rep['report']}); extended {rep['extended'] or 'none'}")
    _print_choice(ev)
    record = {"rules": RULES, "source": a.source, "target": a.target,
              "report_stage": rep["stage"], "report": rep["report"],
              "extended_at_stage1": rep["extended"], **ev,
              "manifests": sorted(str(out / n) for n in files)}
    if a.dry_run:
        return 0
    try:
        write_dir(out, files, "population-choice.json", record, a.force)
    except (ChoiceError, OSError) as exc:
        print(f"set_population: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}: {', '.join(sorted(files))} and population-choice.json")
    return 0


def apply(a) -> int:
    configs, out = Path(a.configs).resolve(), Path(a.out)
    try:
        choice = _read_json(Path(a.choice))
        for k in ("pop", "dyn", "control"):
            if not isinstance(choice.get(k), dict) or not choice[k].get("argv"):
                raise ChoiceError(f"{a.choice} has no {k}")
        files = resolve_dir(configs, a.target, choice)
        record = {"choice_file": str(Path(a.choice).resolve()), "target": a.target,
                  **{k: choice[k] for k in ("pop", "dyn", "control", "pop_arm", "dyn_arm")},
                  "manifests": sorted(str(out / n) for n in files)}
        write_dir(out, files, "population-applied.json", record, a.force)
    except (ChoiceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"set_population: {exc}", file=sys.stderr)
        print("set_population: nothing written", file=sys.stderr)
        return 1
    print(f"POP = {choice['pop']['name']}, DYN = {choice['dyn']['name']}, control = "
          f"{choice['control']['name']}; wrote {out}: {', '.join(sorted(files))} and "
          "population-applied.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("choose")
    c.add_argument("--out", required=True)
    c.add_argument("--results", default=str(REPO.parent / "gomea-results"))
    c.add_argument("--configs", default=str(REPO / "configs" / "gomea"))
    c.add_argument("--source", default="phase1ay")
    c.add_argument("--target", default="phase1az")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--force", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("--choice", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--configs", default=str(REPO / "configs" / "gomea"))
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return choose(a) if a.cmd == "choose" else apply(a)


if __name__ == "__main__":
    raise SystemExit(main())
