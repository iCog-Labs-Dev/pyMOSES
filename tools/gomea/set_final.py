"""Choose the FINAL arm of 1ax from the final 1aw report (server round 8).

Usage (from the repository root, after `tools/gomea/phase_run.sh phase1aw`; the round-8 plan
runs it as its `choice` step)::

    .venv/bin/python tools/gomea/set_final.py --out DIR [--results DIR] [--configs DIR]
        [--source PHASE] [--target PHASE] [--dry-run] [--force]

`SERVER-ROUND8-PREREGISTRATION.md`, "The choice of FINAL": if rule 2 of 1aw passes (LMVAL-PEN125
is better than LMVAL on size on at least 4 of 6 datasets, and its median test NMSE is at most
10 % higher on each), FINAL is the LMVAL-PEN125 arm of 1aw; otherwise it is the LMVAL arm.
Rules 1 and 3 of 1aw do not change FINAL. This script

1. reads the final report of SOURCE (`RESULTS/SOURCE/report-stage{1,2}/`): stage 2 if a primary
   contrast of SOURCE was "extend" at stage 1 (the rule of phase_run.sh), else stage 1. A
   contrast takes its stage-2 verdict only if it was "extend" at stage 1; all others keep
   their stage-1 verdict (`docs/gomea/QUEUE.md`, the standing verdict rule);
2. counts the datasets whose primary contrast LMVAL-PEN125 vs LMVAL on `program_size` has the
   final verdict "A better";
3. per dataset takes `median_a / median_b` of the secondary contrast LMVAL-PEN125 vs LMVAL on
   `test_nmse` of the final report (a failed run counts as an infinite NMSE, as in the
   report; inf / inf is not a number, so that dataset fails the 1.10 limit);
4. rule 2 passes if at least 4 datasets are "A better" and every ratio is at most 1.10;
5. writes DIR/sr.json: `CONFIGS/TARGET/template.json` with the FINAL arm's `argv` set to the
   exact argv of the chosen arm of `CONFIGS/SOURCE/sr.json`; DIR/spec.json: a copy of
   `CONFIGS/TARGET/spec.json`; and DIR/final-choice.json: the rule, the numbers and the choice.

DIR is in the round's state directory (the plan passes `{state}/phase1ax-config`), and
phase_run.sh reads the phase from there (PHASE_CONFIG_DIR). So the choice never changes a
tracked file, and the working tree stays clean for every later S0.

It exits 1 and writes nothing if a report file is missing or unreadable, if a dataset has no
size or NMSE contrast in the report, if the chosen argv has `--gomea-max-evals-per-deme`
(the pre-registration says that both arms have no cap), or if DIR/sr.json exists with a
different content (a choice is made once; `--force` replaces it). `--dry-run` prints the choice
and writes nothing. RESULTS defaults to `../gomea-results`, CONFIGS to `configs/gomea`, SOURCE to
`phase1aw`, TARGET to `phase1ax`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PEN, LMVAL, FINAL = "LMVAL-PEN125", "LMVAL", "FINAL"
MIN_BETTER = 4
MAX_RATIO = 1.10
CAP = "--gomea-max-evals-per-deme"
RULE = ("rule 2 of 1aw: LMVAL-PEN125 vs LMVAL on program_size has the final verdict "
        f"'A better' on at least {MIN_BETTER} of 6 datasets, and on each dataset the median "
        f"test NMSE of LMVAL-PEN125 / LMVAL <= {MAX_RATIO}; passes -> FINAL = LMVAL-PEN125, "
        "else FINAL = LMVAL")


class ChoiceError(Exception):
    pass


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise ChoiceError(f"cannot read {p}: {type(exc).__name__}: {exc}") from exc


def _read_rows(p: Path) -> list[dict]:
    try:
        with p.open(newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        raise ChoiceError(f"cannot read {p}: {type(exc).__name__}: {exc}") from exc


def final_report(results: Path, configs: Path, source: str) -> dict:
    """The final verdicts and rows of SOURCE, and which stage they come from."""
    spec = _read_json(configs / source / "spec.json")
    r1 = results / source / "report-stage1"
    v1 = _read_json(r1 / "verdict.json")
    extended = sorted({f"{p}:{n}" for p, n, *_ in spec["primary"]
                       if v1.get(f"{p}:{n}") == "extend"})
    if not extended:
        return {"stage": 1, "extended": [], "verdicts": v1,
                "rows": _read_rows(r1 / "contrasts.csv"), "spec": spec,
                "report": str(r1)}
    r2 = results / source / "report-stage2"
    v2 = _read_json(r2 / "verdict.json")
    verdicts = {k: (v2.get(k) if k in extended else v) for k, v in v1.items()}
    return {"stage": 2, "extended": extended, "verdicts": verdicts,
            "rows": _read_rows(r2 / "contrasts.csv"), "spec": spec, "report": str(r2)}


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return math.inf if a > 0 else (1.0 if a == 0 else math.nan)
    return a / b  # inf / inf -> nan


def evaluate(rep: dict) -> dict:
    """The numbers of rule 2 and the choice."""
    spec, verdicts = rep["spec"], rep["verdicts"]
    size = [c for c in spec["primary"]
            if c[2] == PEN and c[3] == LMVAL and len(c) == 5 and c[4] == "program_size"]
    nmse = [c for c in spec["secondary"]
            if c[2] == PEN and c[3] == LMVAL and len(c) == 5 and c[4] == "test_nmse"]
    datasets = [c[0] for c in size]
    if not datasets or sorted(datasets) != sorted(c[0] for c in nmse):
        raise ChoiceError(f"spec: size contrasts on {datasets}, NMSE contrasts on "
                          f"{[c[0] for c in nmse]}: want one of each per dataset")
    rows = {(r["problem"], r["contrast"]): r for r in rep["rows"]}
    per_ds = {}
    for (d, sname, *_), (_, nname, *_) in zip(
            sorted(size), sorted(nmse), strict=True):
        if f"{d}:{sname}" not in verdicts:
            raise ChoiceError(f"no verdict for {d}:{sname} in {rep['report']}")
        row = rows.get((d, nname))
        if row is None:
            raise ChoiceError(f"no row {d}:{nname} in {rep['report']}/contrasts.csv")
        ma, mb = float(row["median_a"]), float(row["median_b"])
        ratio = _ratio(ma, mb)
        per_ds[d] = {
            "size_contrast": sname, "size_verdict": verdicts[f"{d}:{sname}"],
            "size_pairs": int(rows[(d, sname)]["pairs"]) if (d, sname) in rows else None,
            "size_p_holm": float(rows[(d, sname)]["p_holm"]) if (d, sname) in rows else None,
            "nmse_contrast": nname, "nmse_pairs": int(row["pairs"]),
            "median_test_nmse_pen125": ma, "median_test_nmse_lmval": mb,
            "nmse_ratio": ratio, "nmse_ratio_ok": ratio <= MAX_RATIO,
        }
    order = [c[0] for c in size]
    per_ds = {d: per_ds[d] for d in order}
    n_better = sum(v["size_verdict"] == "A better" for v in per_ds.values())
    all_ok = all(v["nmse_ratio_ok"] for v in per_ds.values())
    passes = n_better >= MIN_BETTER and all_ok
    return {"size_better": n_better, "nmse_ratio_all_ok": all_ok, "rule2_passes": passes,
            "per_dataset": per_ds, "chosen_arm": PEN if passes else LMVAL}


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(REPO.parent / "gomea-results"))
    ap.add_argument("--configs", default=str(REPO / "configs" / "gomea"))
    ap.add_argument("--source", default="phase1aw")
    ap.add_argument("--target", default="phase1ax")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    results, configs, out = Path(a.results).resolve(), Path(a.configs).resolve(), Path(a.out)
    try:
        rep = final_report(results, configs, a.source)
        ev = evaluate(rep)
        src = _read_json(configs / a.source / "sr.json")
        chosen = src["arms"].get(ev["chosen_arm"])
        if chosen is None:
            raise ChoiceError(f"{configs / a.source / 'sr.json'} has no arm {ev['chosen_arm']}")
        final_argv = list(chosen["argv"])
        if CAP in final_argv or any(t.startswith(CAP + "=") for t in final_argv):
            raise ChoiceError(f"the argv of {ev['chosen_arm']} has {CAP}: {final_argv}")
        template = _read_json(configs / a.target / "template.json")
        if FINAL not in template["arms"] or template["arms"][FINAL].get("argv") is not None:
            raise ChoiceError(f"{configs / a.target / 'template.json'}: want arm {FINAL} "
                              "with argv null")
        spec_text = (configs / a.target / "spec.json").read_text()
    except (ChoiceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"set_final: {exc}", file=sys.stderr)
        print("set_final: nothing written", file=sys.stderr)
        return 1
    manifest = json.loads(json.dumps(template))
    manifest["arms"][FINAL] = {k: v for k, v in template["arms"][FINAL].items()
                               if k in ("binary", "budget")} | {"argv": final_argv}
    text = json.dumps(manifest, indent=2) + "\n"
    for d, v in ev["per_dataset"].items():
        print(f"{d}: size {v['size_verdict']} (Holm p {v['size_p_holm']}); median test NMSE "
              f"PEN125 {v['median_test_nmse_pen125']:.6g} / LMVAL "
              f"{v['median_test_nmse_lmval']:.6g} = {v['nmse_ratio']:.4g}"
              f"{'' if v['nmse_ratio_ok'] else ' > 1.10'}")
    print(f"report stage {rep['stage']} ({rep['report']}); extended {rep['extended'] or 'none'}")
    print(f"size better on {ev['size_better']} of {len(ev['per_dataset'])}; all NMSE ratios "
          f"<= {MAX_RATIO}: {ev['nmse_ratio_all_ok']}; rule 2 "
          f"{'passes' if ev['rule2_passes'] else 'fails'}")
    print(f"-> FINAL = {a.source} {ev['chosen_arm']}: {' '.join(final_argv)}")
    record = {
        "rule": RULE, "source": a.source, "target": a.target,
        "report_stage": rep["stage"], "report": rep["report"],
        "extended_at_stage1": rep["extended"], **ev, "final_argv": final_argv,
        "manifest": str(out / "sr.json"),
    }
    if a.dry_run:
        return 0
    if (out / "sr.json").exists() and (out / "sr.json").read_text() != text and not a.force:
        print(f"set_final: {out / 'sr.json'} exists with another content; nothing written "
              "(--force replaces it)", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(spec_text)
    (out / "sr.json").write_text(text)
    (out / "final-choice.json").write_text(json.dumps(_json_safe(record), indent=2) + "\n")
    print(f"wrote {out / 'sr.json'}, {out / 'spec.json'} and {out / 'final-choice.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
