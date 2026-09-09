#!/usr/bin/env python3
"""Normalise a MOSES log so classic's and the port's can be compared line by line.

A byte-identical comparison is impossible: every record carries a wall-clock
timestamp, and the run itself embeds a hostname, the argv it was invoked with,
absolute file paths, microsecond timings and a RAM reading. None of those are
behaviour. Everything else is: the LEVEL of each record, its MESSAGE text, and
the ORDER the records appear in.

So this tool reduces a log to exactly that -- one `LEVEL<TAB>MESSAGE` line per
record -- with the volatile fields replaced by fixed placeholders.

Usage::

    python3 tools/normalize_log.py <log>                 # print normalised records
    python3 tools/normalize_log.py --compare <a> <b>     # diff two logs
    python3 tools/normalize_log.py --rules               # print the rules applied

## Records, not lines

A record is `[<timestamp>] [<LEVEL>] <message>`, and a message may contain
newlines -- `Logger::log` appends `std::endl` to whatever it was given, and
several MOSES messages are multi-line (`"New best score: ...\\n\\tfor tree:
..."`, the whole `Stats: #` legend, the MPI warning's trailing `\\n`). A
physical line that does NOT start with a timestamp is therefore a continuation
of the record above it, and is folded back into it; embedded newlines come out
as a literal `\\n` so that one record is always one output line.

That folding is why the counts this tool reports differ from `wc -l`: classic's
193-line log for `-i dataset.csv -m300 -W1 -l DEBUG` is 173 records.

## The volatile fields, and why each is volatile

* **Timestamp** -- wall clock. Dropped entirely.
* **Hostname** (`hostname: archlinux`) -- machine identity.
* **Paths** -- classic is usually invoked with a relative input path and the
  port with an absolute one, and the two runs must write to different log
  files. Any whitespace-delimited token containing `/` is reduced to its
  basename, and any token ending in `.log` becomes `<LOG>`.
* **Program name** -- `Command line:` records argv[0], which is
  `./build/moses/moses/main/moses` for classic and whatever driver ran the
  port. The basename rule above collapses both to `moses`; a bare `moses` or
  `moses.py` token is normalised to `<PROG>`.
* **`DemesHill:` columns 5, 7 and 10** -- two microsecond readings and a RAM
  fraction (hill-climbing.cc:385-396 streams `usec` twice and `ram_usage`
  once). Note the legend at hill-climbing.cc:826-842 names only 15 columns
  while the data row has 17 values; that mismatch is classic's, and this tool
  indexes the DATA row, not the legend.
* **`Stats:` column 3** -- `((int) stats.elapsed_secs)` (local_moses.cc:161).
  Zero on classic's sub-second runs and not zero on the port's.

Nothing else is touched. In particular scores, complexities, tree shapes,
instance bit patterns, evaluation counts and deme sizes all pass through
verbatim -- those are the things a divergence would show up in.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["apply_allow_list", "normalize_line", "normalize_log", "parse_records"]

# `[2026-09-09 18:24:03:734] [INFO] ` -- note the COLON before the
# milliseconds (Logger.cc:512-513's `"[%s:%03ld] "`), not a dot.
_RECORD_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:\d{3})\] "
    r"\[(?P<level>[A-Za-z ]+)\] ?(?P<msg>.*)$"
)

_HOSTNAME_RE = re.compile(r"^hostname: .*$")

# A `DemesHill:`/`Hill:` data row -- the legend row is the one whose first
# field after the prefix is `#`, and it is left alone.
_HILL_RE = re.compile(r"^(?P<prefix>\S*Hill: )(?P<body>(?!#).*)$")

_STATS_RE = re.compile(r"^Stats: (?P<body>(?!#).*)$")

VOLATILE = "<VOL>"


def _normalize_token(tok: str) -> str:
    """Reduce one whitespace-delimited token to its path- and name-invariant form."""
    if "/" in tok:
        tok = tok.rsplit("/", 1)[-1]
    if tok.endswith(".log"):
        return "<LOG>"
    if tok in ("moses", "moses.py", "moses-exec"):
        return "<PROG>"
    return tok


def normalize_line(level: str, msg: str) -> str:
    """Return the `LEVEL<TAB>MESSAGE` form of one record, volatiles replaced.

    Order matters: the `DemesHill:`/`Stats:` column rules run BEFORE the
    per-token path rule, because a column value is not a path and must not be
    mistaken for one.
    """
    if _HOSTNAME_RE.match(msg):
        msg = "hostname: <HOST>"

    hill = _HILL_RE.match(msg)
    if hill:
        cols = hill.group("body").split("\t")
        # hill-climbing.cc:385-396 streams, in order: demeID, iteration,
        # total_steps, total_evals, usec, max_nn_evals, usec,
        # number_of_new_instances, current_number_of_instances, ram_usage,
        # current_number_of_evals, has_improved, best_score, delta,
        # best_raw_score, delta_raw, complexity. Indices 4, 6 and 9 (0-based)
        # are the two `usec` readings and `ram_usage`.
        for i in (4, 6, 9):
            if i < len(cols):
                cols[i] = VOLATILE
        msg = hill.group("prefix") + "\t".join(cols)

    stats = _STATS_RE.match(msg)
    if stats:
        cols = stats.group("body").split("\t")
        # local_moses.cc:158-161: n_expansions, n_evals, (int) elapsed_secs, ...
        if len(cols) > 2:
            cols[2] = VOLATILE
        msg = "Stats: " + "\t".join(cols)

    # Split on spaces only, so the tab-separated column rows above survive.
    msg = " ".join(_normalize_token(t) for t in msg.split(" "))
    return f"{level}\t{msg}"


def parse_records(text: str) -> list[tuple[str, str]]:
    """Split a raw log into `(level, message)` records, folding continuations.

    A physical line without a `[timestamp] [LEVEL] ` prefix continues the
    record above it. A file that begins with such a line -- which should not
    happen, but would silently drop content if ignored -- raises.
    """
    records: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        m = _RECORD_RE.match(line)
        if m:
            records.append((m.group("level"), [m.group("msg")]))
        elif records:
            records[-1][1].append(line)
        elif line:
            raise ValueError(f"log begins with a continuation line: {line!r}")
    # `Logger::log` ends every record with a newline, so the last physical
    # line of the file is always empty and belongs to no record's content.
    out = []
    for level, parts in records:
        while parts and parts[-1] == "":
            parts.pop()
        out.append((level, "\n".join(parts)))
    return out


def normalize_log(text: str) -> list[str]:
    """Return one normalised `LEVEL<TAB>MESSAGE` line per record."""
    return [normalize_line(lvl, msg).replace("\n", "\\n") for lvl, msg in parse_records(text)]


def apply_allow_list(records: list[str], allowed: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove the allow-listed records from `records`, in order.

    `allowed` is a list of normalised record strings that classic emits and
    the port legitimately does not -- one entry per OCCURRENCE, not per
    distinct text, so a record classic emits twice needs two entries. The
    entries are consumed in the order they are matched, and an entry that
    never matches is returned in the second element of the pair. A stale
    allow-list is therefore visible rather than silently inert: the list can
    only shrink as more call sites land, never quietly cover new gaps.
    """
    remaining = list(allowed)
    kept = []
    for rec in records:
        if rec in remaining:
            remaining.remove(rec)
            continue
        kept.append(rec)
    return kept, remaining


def _compare(a: Path, b: Path, allow: Path | None, full: bool) -> int:
    """Print a divergence report for two logs; return 0 when they agree."""
    la = normalize_log(a.read_text(errors="replace"))
    lb = normalize_log(b.read_text(errors="replace"))
    print(f"{a}: {len(la)} records")
    print(f"{b}: {len(lb)} records")

    unmatched_allow: list[str] = []
    if allow is not None:
        entries = [ln for ln in allow.read_text().split("\n") if ln and not ln.startswith("#")]
        la, unmatched_allow = apply_allow_list(la, entries)
        print(
            f"{a}: {len(la)} records after removing {len(entries) - len(unmatched_allow)} allowed"
        )
        for entry in unmatched_allow:
            print(f"  STALE allow-list entry (never matched): {entry}")

    if la == lb:
        print(f"\nidentical: {len(la)} records")
        return 0 if not unmatched_allow else 1

    if full:
        for line in difflib.unified_diff(la, lb, fromfile=str(a), tofile=str(b), lineterm="", n=1):
            print(line)
        return 1

    matched = 0
    for i in range(min(len(la), len(lb))):
        if la[i] != lb[i]:
            print(f"\nfirst divergence at record {i + 1}:")
            print(f"  A: {la[i]}")
            print(f"  B: {lb[i]}")
            break
        matched += 1
    else:
        print(f"\ncommon prefix of {matched} records; lengths differ")
        longer, name = (la, a) if len(la) > len(lb) else (lb, b)
        for extra in longer[matched : matched + 20]:
            print(f"  only in {name}: {extra}")
    return 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--compare", action="store_true", help="diff two logs instead of printing one")
    ap.add_argument("--rules", action="store_true", help="print the normalisation rules and exit")
    ap.add_argument("--allow", type=Path, help="file of records to drop from the FIRST log")
    ap.add_argument("--full", action="store_true", help="print a unified diff, not just the first")
    ns = ap.parse_args(argv)

    if ns.rules:
        print(__doc__)
        return 0
    if ns.compare:
        if len(ns.logs) != 2:
            ap.error("--compare takes exactly two logs")
        return _compare(*ns.logs, allow=ns.allow, full=ns.full)
    if len(ns.logs) != 1:
        ap.error("give exactly one log, or --compare with two")
    for line in normalize_log(ns.logs[0].read_text(errors="replace")):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
