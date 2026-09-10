"""C++: moses/feature-selection/main/eval-features.{h,cc}

The standalone ``eval-features`` binary: "Program to output the feature
quality of a feature set given a data set. One can provide a list of
feature sets instead of one feature set and output the results for each
data set" (eval-features.cc:39-43's own comment).

Both C++ files port into this one module, because eval-features.h is not
a header in the interface sense -- it defines every function inline and
is included by exactly one translation unit, eval-features.cc, which adds
only ``main()``. Splitting them in Python would create two modules where
C++ has one compilation unit.

Structure follows ``pymetta.moses.main.moses_exec`` and
``pymetta.feature_selection.main.fs_main``: ``boost::program_options``
becomes one ``argparse.ArgumentParser``, ``-h``/``--help`` is argparse's,
and ``int main(int, char**)`` becomes ``eval_features_main(argv)`` with
``argv[0]`` the program name.

THREE UPSTREAM DEFECTS ARE REPRODUCED HERE, NOT FIXED. All three were
measured against ``$CLASSIC/build/moses/feature-selection/main/eval-features``
on 2026-09-10, every run bounded by ``( ulimit -v 2097152; timeout 60 ... )``.

1. **``-u``/``--target-feature`` and ``-Y``/``--ignore-feature`` are
   parsed and then thrown away.** eval-features.cc:74-81 binds them to
   ``main``'s own locals ``target_feature_str`` / ``ignore_features_str``,
   which nothing ever reads; ``read_eval_output_results``
   (eval-features.h:145-162) declares its OWN empty ``target_feature``,
   ``timestamp_feature`` and ``ignore_features`` and passes those to
   ``loadTable``. So the target column is ALWAYS the first column of the
   file, whatever ``-u`` says. Measured on ``trainingBoolean.dat``
   (columns ``x1,x2,x3,y``): ``-u y -f x1`` and ``-f x1`` both print
   ``0``, and the target really is ``x1``.

2. **Every feature name resolves to the wrong column, by exactly one.**
   ``read_eval_output_results`` looks names up in ``table.get_labels()``
   (table.cc:491), which is ``[output_label] + input_labels`` -- but the
   index it produces indexes the ITABLE, whose columns are the INPUTS
   alone. So a name's index is one too large, and every lookup lands on
   the column to its RIGHT in the file. With defect 1 making ``x1`` the
   target, the labels are ``[x1, x2, x3, y]`` and the inputs are
   ``[x2, x3, y]``, so:

       -f x1  -> index 0 -> input column x2  -> 0
       -f x2  -> index 1 -> input column x3  -> 0
       -f x3  -> index 2 -> input column y   -> 0.0433733
       -f y   -> index 3 -> OUT OF RANGE

   Confirmed arithmetically, not merely observed: ``MI(y; x1) = 0.0488``
   for that table and ``0.0488 * 8/9 == 0.0433733`` exactly, where
   ``8/9`` is ``MICScorer``'s own confidence factor at ``-d 0`` for a
   single feature over 8 rows; ``MI(x2; x1) = MI(x3; x1) = 0``.

3. **The last label indexes past the end of every row.** That is the
   ``-f y`` case above. ``seq_filtered``
   ($COGUTIL/opencog/util/algorithm.h:352) is ``res.push_back(seq[idx])``
   with a bare ``operator[]`` and no bounds check, so classic reads past
   the end of the row -- undefined behaviour, which happened to print
   ``0``. A port cannot faithfully reproduce undefined behaviour
   (docs/CARRY-FORWARD.md section 3 settles this for the analogous
   out-of-range ``$99`` argument, where reproducing the observed value
   would have been strictly worse than failing), so
   ``eval_output_results`` raises ``IndexError`` naming the index, the
   arity and this note.

``-H``/``--scorer`` is a fourth dead option: it is parsed into
``pa.scorer`` and never read, because ``eval_output_results``
(eval-features.h:132-143) hard-codes ``MICScorer<set<arity_t>>``. It is
registered here so the CLI surface matches, and ignored, as classic does.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from pymetta.comboreduct.table.table import ITable, OTable
from pymetta.comboreduct.table.table_io import loadTable, tokenize_row
from pymetta.feature_selection.main.feature_selection_params import MI_SCORER, feature_set
from pymetta.feature_selection.scorers.mutual_info import MICScorer
from pymetta.util.logger import ostream_double
from pymetta.util.rng import randGen

__all__ = [
    "eval_features_main",
    "eval_features_parameters",
    "eval_output_results",
    "feature_sets",
    "get_features_idx",
    "output_results",
    "read_eval_output_results",
]


@dataclass
class eval_features_parameters:
    """C++: moses/feature-selection/main/eval-features.h:66 eval_features_parameters

    "structure containing the options for the eval-table program" (its own
    comment -- which names the wrong program; this is eval-FEATURES).
    A bare aggregate with no constructor in C++, so every member is
    default-initialized by ``main``'s own
    ``po::value<...>()->default_value(...)`` before use; the defaults
    pinned here are those, and ``scorer``/``confidence_penalty_intensity``
    are the only two that carry one (``mi`` and ``0``, eval-features.cc:63
    and :93).
    """

    input_file: str = ""
    scorer: str = MI_SCORER
    features: list[str] = field(default_factory=list)
    features_file: str = ""
    output_file: str = ""
    confidence_penalty_intensity: float = 0.0


def output_results(pa: eval_features_parameters, qs: Sequence[float]) -> None:
    """C++: moses/feature-selection/main/eval-features.h:76 output_results

    ``ostream_container(out, qs, " ", "", "\\n")`` -- space-separated, no
    brackets, and a trailing newline that is the container's ``right``
    argument, not a separate ``endl``. Since ``ostream_container``'s
    ``empty_lr`` defaults to true ($COGUTIL/opencog/util/iostreamContainer.h:66),
    an EMPTY ``qs`` still prints ``left`` and ``right``, i.e. a bare
    newline -- measured: ``eval-features -i <file>`` with no ``-f`` at all
    prints exactly one ``\\n`` and exits 0.

    The file branch opens with ``ios_base::app``, i.e. APPEND, unlike
    ``saveTable``'s truncating open -- so repeated runs with the same
    ``-o`` accumulate one line each. Reproduced with mode ``"a"``.

    Each ``double`` is streamed at an ``ostream``'s default precision of
    6 (``ostream_double``), the same formatting ``log_selected_features``
    uses.
    """
    line = " ".join(ostream_double(q) for q in qs) + "\n"
    if not pa.output_file:
        sys.stdout.write(line)
    else:
        with open(pa.output_file, "a") as of:
            of.write(line)


def get_features_idx(
    features: Sequence[str], labels: Sequence[str], pa: eval_features_parameters
) -> feature_set:
    """C++: moses/feature-selection/main/eval-features.h:86 get_features_idx

    ``OC_ASSERT((size_t)idx != labels.size(), "No such a feature %s in
    file %s", ...)`` is active in release builds, so it is a raise, not a
    bare ``assert``. Its message is classic's verbatim, since a caller
    comparing against the real binary's stderr sees exactly that text.

    Returns a ``std::set<arity_t>``, i.e. ASCENDING and de-duplicated --
    ``feature_set`` is a Python ``set`` here and every consumer sorts it
    (``mutual_information_it_ot`` does ``sorted(set(fs))``), so the
    ordering C++ gets from ``std::set`` is preserved where it matters.

    NOTE the off-by-one this function is the second half of: ``labels`` is
    ``table.get_labels()``, output label included, while the index is used
    against the ITable. See this module's docstring, defect 2. DO NOT FIX.
    """
    res: feature_set = set()
    for f in features:
        # C++: `distance(labels.begin(), boost::find(labels, f))` -- the
        # not-found value is labels.size(), which is what the assert tests.
        idx = labels.index(f) if f in labels else len(labels)
        if idx == len(labels):
            raise AssertionError(f"No such a feature {f} in file {pa.input_file}")
        res.add(idx)
    return res


def feature_sets(pa: eval_features_parameters, labels: Sequence[str]) -> list[feature_set]:
    """C++: moses/feature-selection/main/eval-features.h:100 feature_sets

    At most two entries: one from ``--feature``/``-f`` (all of them
    together form ONE set, not one set each), and one per non-empty line
    of ``--features-file``/``-F``.

    ``while (in.good()) { getline(in, line); if (line.empty()) continue;
    ... }`` -- the C++ loop tests ``good()`` BEFORE the read, so the final
    ``getline`` that hits EOF still runs and yields an empty string, which
    the ``empty()`` guard then skips. Reading the file's lines directly
    here is equivalent because that trailing pass contributes nothing.

    ``tokenizeRow<string>(line)`` is ``tokenize_row``
    (``pymetta.comboreduct.table.table_io``), the same DSV tokenizer the
    loaders use -- so a features-file line is comma/space/tab separated,
    not comma-only, whatever the ``--features-file`` help text says.
    """
    res: list[feature_set] = []
    if pa.features:
        res.append(get_features_idx(pa.features, labels, pa))
    if pa.features_file:
        with open(pa.features_file) as in_:
            for raw in in_:
                line = raw.rstrip("\n")
                if not line:
                    continue
                res.append(get_features_idx(tokenize_row(line), labels, pa))
    return res


def eval_output_results(
    pa: eval_features_parameters, fss: Sequence[feature_set], it: ITable, ot: OTable
) -> None:
    """C++: moses/feature-selection/main/eval-features.h:132 eval_output_results

    The non-template overload (eval-features.h:132-143), which builds the
    scorer and then calls the TEMPLATE overload (eval-features.h:120-130)
    to score each feature set and print. Both collapse into this one
    function: the template exists in C++ only so the scorer type can vary,
    and it never does -- eval-features.h:138 hard-codes
    ``typedef MICScorer<set<arity_t>> FSScorer``, ignoring ``pa.scorer``
    entirely (see this module's docstring).

    The out-of-range guard is this port's, not classic's: see the module
    docstring, defect 3.
    """
    fs_sc = MICScorer(it, ot, pa.confidence_penalty_intensity)
    qs: list[float] = []
    for fs in fss:
        for idx in fs:
            if idx >= it.get_arity():
                raise IndexError(
                    f"eval-features: feature index {idx} is past the end of a "
                    f"{it.get_arity()}-column input table. classic indexes the row "
                    "with a bare operator[] here (seq_filtered, "
                    "opencog/util/algorithm.h:352) and reads past the end -- "
                    "undefined behaviour, reproduced as a raise rather than "
                    "guessed at; see eval_features.py's module docstring, "
                    "defects 2 and 3."
                )
        qs.append(fs_sc(fs))
    output_results(pa, qs)


def read_eval_output_results(pa: eval_features_parameters) -> None:
    """C++: moses/feature-selection/main/eval-features.h:145 read_eval_output_results

    ``loadTable(pa.input_file, target_feature, timestamp_feature,
    ignore_features)`` with all three arguments freshly declared and
    EMPTY (eval-features.h:147-151) -- this is defect 1 in this module's
    docstring, and the reason ``-u``/``-Y`` do nothing. DO NOT FIX.
    """
    # C++: eval-features.h:147-151. Written out rather than inlined as
    # literals, so the three dead locals stay visible.
    target_feature = ""
    timestamp_feature = ""
    ignore_features: list[str] = []

    table = loadTable(pa.input_file, target_feature, timestamp_feature, ignore_features)

    # C++: eval-features.h:155 -- output label FIRST, then the inputs.
    labels = table.get_labels()

    # C++: eval-features.h:158
    fss = feature_sets(pa, labels)

    # C++: eval-features.h:161
    eval_output_results(pa, fss, table.itable, table.otable)


def _add_options(parser: argparse.ArgumentParser) -> None:
    """C++: moses/feature-selection/main/eval-features.cc:45 main (its
    ``desc.add_options()`` block, eval-features.cc:55-95)

    The breadcrumb cites ``main`` because the option block is a chained
    call inside ``main``'s body, not a symbol of its own. No ``help=``
    text, matching ``fs_main._add_options`` and
    ``problem_params.add_options``.

    Note the SHORT NAMES differ from ``feature-selection``'s for the same
    letters: here ``-F`` is ``--features-file`` and ``-f`` is
    ``--feature`` (eval-features.h:50-51), whereas in ``fs-main.cc`` ``-F``
    is ``--log-file`` and ``-f`` is ``--initial-feature``. ``eval-features``
    has no logging options at all.
    """
    add = parser.add_argument
    add("-r", "--random-seed", dest="rand_seed", type=int, default=1)
    add("-H", "--scorer", dest="scorer", type=str, default=MI_SCORER)
    add("-i", "--input-file", dest="input_file", type=str, default="")
    add("-u", "--target-feature", dest="target_feature_str", type=str, default="")
    add("-Y", "--ignore-feature", dest="ignore_features_str", action="append", default=[])
    add("-o", "--output-file", dest="output_file", type=str, default="")
    add("-f", "--feature", dest="features", action="append", default=[])
    add("-F", "--features-file", dest="features_file", type=str, default="")
    add(
        "-d",
        "--confidence-penalty-intensity",
        dest="confidence_penalty_intensity",
        type=float,
        default=0.0,
    )


def eval_features_main(argv: Sequence[str] | str) -> int:
    """C++: moses/feature-selection/main/eval-features.cc:45 main

    ``argv[0]`` is the program name, so only ``argv[1:]`` reaches the
    parser; a bare ``str`` is split on whitespace first -- the convention
    ``moses_exec`` and ``fs_main`` both keep.

    Returns ``1`` for ``--help`` / no arguments (eval-features.cc:101-104,
    which really does ``return 1`` for a successful ``--help``) and ``0``
    otherwise: C++'s ``main`` falls off the end after
    ``read_eval_output_results(pa)``.

    ``randGen().seed(rand_seed)`` runs even though nothing downstream
    draws from the RNG (``MICScorer`` and ``mutualInformation`` are both
    deterministic) -- ported anyway, because it is observable through the
    process-wide generator if a caller drives this in-process.
    """
    if isinstance(argv, str):
        return eval_features_main(argv.split())

    argv_list = list(argv)
    parser = argparse.ArgumentParser(prog="eval-features", add_help=True)
    _add_options(parser)

    # C++: eval-features.cc:101 `if(vm.count("help") || argc == 1)`.
    # argparse handles `--help` itself; `argc == 1` is reproduced here.
    if len(argv_list) <= 1:
        parser.print_help()
        return 1

    args = parser.parse_args(argv_list[1:])

    pa = eval_features_parameters(
        input_file=args.input_file,
        scorer=args.scorer,
        features=list(args.features),
        features_file=args.features_file,
        output_file=args.output_file,
        confidence_penalty_intensity=args.confidence_penalty_intensity,
    )

    # C++: eval-features.cc:107 -- init random generator
    randGen().seed(args.rand_seed)
    # C++: eval-features.cc:108
    read_eval_output_results(pa)
    return 0
