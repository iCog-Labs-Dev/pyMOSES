"""C++: moses/moses/scoring/precision_bscore.h and precision_bscore.cc

Ports `precision_bscore` (the `-H pre` problem's scorer) and
`precision_conj_bscore` (the `-H pre-conj` problem's scorer -- the
registered name has a HYPHEN, table-problems.h:134; `-Hpre_conj` names
nothing in classic's registry and is refused there and here alike).

## Scope: the boolean path -- exactly what the registered `-H pre`
## problem actually constructs

`table-problems.cc`'s `pre_table_problem::run()` calls `common_type_setup
(pms, id::boolean_type)` UNCONDITIONALLY before constructing
`precision_bscore` -- despite this class's own header doc comment
extensively describing contin-table support ("This scorer works for both
boolean regression and contin regression..."), the constructor's own
`OC_ASSERT(output_type == id::boolean_type, ...)` and the registered
problem's forced boolean setup together mean the contin path the header
describes is UNREACHABLE from the one place this class is actually wired
up in classic. This port therefore targets exactly the boolean,
non-boosting path -- the reachable one -- and raises a clear, EARLY gate
for the rest:

* `time_bscore=True` (`--time-bscore`) and `dispersion_pressure > 0.0`
  (`--time-dispersion-pressure`) ARE ported: both read the
  per-observation timestamp (`time2res[tcv.first.timestamp]` /
  `get_timestamp_class(tcv.first.timestamp)`, precision_bscore.cc:242-263)
  that this port's `CTable` now carries -- see table.py's `TimedCounter`
  and time_dispersion.py's module docstring. The dispersion branch builds
  a `ctable_res` whose single cell is keyed by `TimedValue`, so it inherits
  classic's `TimedValue::operator<` UB in full; see `do_score`'s own
  DIVERGENCE marker for what a user observes.
* Ensemble/boosting scoring IS ported now: `call_ensemble` (the
  `operator()(const scored_combo_tree_set&)` overload,
  precision_bscore.cc:368), `exact_selection` (:391), `bias_selection`
  (:453), `get_error(const combo_tree&)` (:501) and the
  `reset_weights`/`update_weights` overrides (:698, :707). `table-
  problems.cc:385` calls `bscore.use_weighted_scores()` whenever
  `--boost=1`, so `_return_weighted_score` IS `True` for every boosted
  `-H pre` run and `do_score`'s own `if (_return_weighted_score)` branch
  (precision_bscore.cc:283-289) is on the hot path, not a gate.
* `bias_selection` (the `exact_experts == false` arm of `call_ensemble`)
  is ported but is UNREACHABLE from classic's own CLI: `ensemble::
  add_expert` (ensemble.cc:184) calls `_bscorer.get_error(tree)` BEFORE
  it branches on `exact_experts`, and `precision_bscore::get_error`
  opens with `OC_ASSERT(exact_experts, "Precision scorer: inexact
  experts not yet supported!")` (precision_bscore.cc:503). So
  `--boost=1 --boost-exact=0` aborts in classic (SIGABRT) and raises
  `AssertionError` here, in `get_error`, before any ensemble scorer
  runs. `--boost-bias` therefore only ever changes `bias_scale`, which
  only `bias_selection` reads. Ported for structural fidelity and
  covered by a direct unit test rather than by the CLI.
"""

import math
from functools import cmp_to_key

from pymetta.comboreduct.combo.vertex import (
    argument,
    bool_to_vertex,
    builtin,
    combo_tree,
    negate_vertex,
    vertex_equal,
    vertex_to_bool,
)
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import (
    CTable,
    TimedCounter,
    timed_items_of,
    timestamp_less,
    type_node,
)
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t, very_worst_score
from pymetta.moses.scoring.scoring_base import _ieee_div, bscore_base, discrete_complexity_coef
from pymetta.moses.scoring.time_dispersion import TemporalGranularity, bscore_ctable_time_dispersion


class precision_bscore(bscore_ctable_time_dispersion):
    """C++: moses/moses/scoring/precision_bscore.h:107 precision_bscore"""

    def __init__(
        self,
        ctable: CTable,
        activation_pressure: float = 1.0,
        min_activation: float = 0.5,
        max_activation: float = 1.0,
        positive: bool = True,
        dispersion_pressure: float = 0.0,
        dispersion_exponent: float = 1.0,
        exact_experts: bool = True,
        bias_scale: float = 1.0,
        time_bscore: bool = False,
        granularity: TemporalGranularity = TemporalGranularity.day,
        disable_debug_log: bool = False,
    ) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:114 precision_bscore

        `min_activation`/`max_activation` are CLAMPED to `[0.0, 1.0]`
        (`clamp(x, 0.0f, 1.0f)`) in the member-initializer list, BEFORE
        `reset_weights()` and the threshold sanity check -- ported in
        that exact order.

        `activation_pressure`/`min_activation`/`max_activation` are C++
        `float` PARAMETERS -- narrowed at the call boundary via
        `_to_score_t`, same convention as `discriminating_bscore`'s own
        constructor.
        """
        super().__init__(ctable, dispersion_pressure, dispersion_exponent, granularity)
        self.min_activation: score_t = _to_score_t(max(0.0, min(1.0, min_activation)))
        self.max_activation: score_t = _to_score_t(max(0.0, min(1.0, max_activation)))
        self.activation_pressure: score_t = _to_score_t(activation_pressure)
        self.positive = positive
        self.bias_scale = bias_scale
        self.exact_experts = exact_experts
        self.time_bscore = time_bscore
        self._disable_debug_log = disable_debug_log

        self.reset_weights()

        self._target = bool_to_vertex(positive)
        self._neg_target = negate_vertex(self._target)

        self.output_type = self._wrk_ctable.get_output_type()
        if self.output_type != type_node.boolean_type:
            raise AssertionError("Error: Precision scorer: output type must be boolean!")

        if not (0.0 <= self.activation_pressure):
            raise AssertionError("activation pressure must be non-negative")
        if 0.0 < self.activation_pressure:
            if not (0.0 < self.min_activation <= self.max_activation):
                raise AssertionError(
                    "Precision scorer, invalid activation bounds.  "
                    "The minimum activation must be greater than zero, "
                    "and the maximum activation must be greater "
                    "than or equal to the minimum activation.\n"
                )

    def sum_outputs(self, counter) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:175 precision_bscore::sum_outputs

        "For boolean tables, sum the total number of 'T' values in the
        output. This sum represents the best possible score i.e. we
        found all of the true values correcty. The 'F's are false
        positives." (precision_bscore.cc:171-174's own doc comment.)
        """
        pos = counter.get(self._target, 0.0)
        neg = counter.get(self._neg_target, 0.0)
        return _to_score_t(0.5 * (pos - neg))

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:180 set_complexity_coef(unsigned, float)
        and precision_bscore.cc:197 set_complexity_coef(score_t)

        Divides by `self._ctable_weight` (a `count_t`/double, never
        truncated) where `discriminating_bscore`'s own two overloads
        divide by `self._ctable_usize` (truncated) instead -- these are
        genuinely different denominators in C++ too, not a copy-paste
        variant of the same method.
        """
        if len(args) == 1:
            (ratio,) = args
            ratio = _to_score_t(ratio)
            self._complexity_coef = 0.0
            if ratio > 0:
                self._complexity_coef = _to_score_t(_ieee_div(1.0, ratio))
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                self._complexity_coef = _to_score_t(
                    _ieee_div(discrete_complexity_coef(alphabet_size, p), self._ctable_weight)
                )
        else:
            raise TypeError(
                "set_complexity_coef expects either (ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )

    def do_score(self, select) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:207 precision_bscore::do_score

        `select` is a `Callable[[tuple], bool]` over a ctable ROW KEY
        (mirroring `std::function<bool(const multi_type_seq&)>`).

        `score_t acto = orow.total_count();` (precision_bscore.cc:238)
        narrows a `count_t` (double) total down to float32 AT THAT
        DECLARATION -- unlike `discriminator.count`'s own `totalc`
        (discriminating_bscore.py), this one is NOT further truncated to
        an integer (`_to_score_t`, not `_to_usize`: `acto` stays a
        `score_t`, matching C++'s own `score_t acto`, not an `unsigned`).
        `sao`/`active` are `score_t` ACCUMULATORS narrowed on every `+=`,
        same "real float32 member" pattern as `discriminator.count`'s own
        docstring.

        ## The two timestamp branches

        `time_bscore` (precision_bscore.cc:242-248, :266-275) replaces the
        per-ROW bscore with a per-TIMESTAMP one: `time2res` is a
        `std::map<TTable::value_type, Counter<bool, count_t>>` keyed on the
        raw observation date (NOT the granularity class -- the two branches
        differ there, and only the dispersion branch calls
        `get_timestamp_class`). Its key is a plain
        `boost::gregorian::date`, whose `operator<` IS a total order, so
        that map's iteration order is well defined and OBSERVABLE: it
        decides the order of the emitted bscore components. Reproduced by
        sorting on `timestamp_less` (table.py) on the way out.

        Note `count_t count = selected ? tcv.second : 0.0;`
        (precision_bscore.cc:246) still WRITES the zero, so an unselected
        row's timestamp gets a bucket -- the bscore has one component per
        distinct timestamp in the table, not per SELECTED timestamp. The
        `+= 0.0` is load-bearing, not a no-op.

        `_pressure > 0.0` (precision_bscore.cc:255-263, :340-345) builds a
        one-cell `CTable` of results, keyed on `TimedValue(logical_true or
        logical_false, get_timestamp_class(timestamp))`, and feeds
        `ordered_by_time()` of it to `get_time_dispersion_penalty`
        (time_dispersion.py). `CTable::key_type dummy_input;` is a
        default-constructed `multi_type_seq` (`()` here), so EVERY row of
        the real table accumulates into that single cell.

        # DIVERGENCE(C++ moses/moses/scoring/precision_bscore.cc:261): that
        # single cell is a `TimedCounter`, so `ctable_res` is a second live
        # instance of classic's `TimedValue::operator<` UB -- see
        # `TimedCounter`'s class docstring (table.py) for the measured
        # behaviour of the comparator and this port's decision to compute
        # the INTENDED result instead. Here the observed pool is exactly the
        # one that probe measures: two vertices (`logical_true` for a
        # selected row, `logical_false` for an unselected one) crossed with
        # the table's timestamp classes. Because `selected` varies per
        # candidate, whether the value order and the date order conflict
        # varies per candidate too, so the corruption is not a property of
        # the dataset alone.
        #
        # What a user observes, MEASURED 2026-09-10 with
        # `-Hpre -q0.5 -w1.0 --timestamp-feature=when
        # --time-dispersion-pressure=0.5 -m300 -W1 -r1`:
        #   * tests/fixtures/table_data/timestamp_bool_clean.csv -- classic
        #     SIGSEGVs (rc 139, core dumped, no diagnostic), at every one of
        #     -m100/-m300/-m1000 crossed with -r1/-r7/-r42. gdb puts the
        #     fault in `representation::get_candidate_rec`'s tree-node
        #     allocation, NOT in any timestamp code: the corrupted
        #     red-black tree damages the heap and an unrelated later
        #     allocation is what dies. This port completes, printing a full
        #     10-candidate listing headed
        #     `0.884651064872741699 !$i1`.
        #   * timestamp_bool_dup.csv -- classic completes, printing the one
        #     line `0.632471561431884766 true`; this port prints a
        #     10-candidate listing headed `0.981214404106140137 !$i2`,
        #     differing for the SAME reason the UN-flagged run on that file
        #     already differs (its CTable cell is corrupted before scoring
        #     ever starts, so this is not a new divergence).
        # Adding `--time-bscore-granularity=month` to either case collapses
        # every date into one class, which removes the conflict and makes
        # classic well defined again: both then print `nan true`
        # (Hmax = log2(1) == 0, so the penalty divides by zero -- and see
        # `_powf`, time_dispersion.py, for why that NaN prints POSITIVE).
        #
        # An UNtimestamped table is the other conflict-free family, and
        # there this port is byte-identical to classic: measured over 7
        # datasets x {-m300, -m1000} x {-r1, -r5} x 7 flag combinations,
        # 168/168 identical (tests/moses/main/test_time_bscore_cli.py).
        """
        bs: behavioral_score = []
        ac: behavioral_score = []
        active: score_t = 0.0
        sao: score_t = 0.0

        # C++: precision_bscore.cc:225-226. `time2res` is only read when
        # `time_bscore`, `ctable_res` only when `_pressure > 0.0`, but both
        # are declared unconditionally there and cost nothing empty here.
        time2res: dict = {}
        ctable_res = CTable()

        for row, counter in self._wrk_ctable.items():
            sumo: score_t = 0.0
            acto: score_t = 0.0
            selected = select(row)
            if selected:
                sumo = self.sum_outputs(counter)
                sao = _to_score_t(sao + sumo)
                acto = _to_score_t(sum(counter.values()))
                active = _to_score_t(active + acto)

            if self.time_bscore:
                # C++: precision_bscore.cc:242-248
                for tv, count in timed_items_of(counter):
                    target = vertex_to_bool(tv.value)
                    if not self.positive:
                        target = not target
                    count = count if selected else 0.0
                    bucket = time2res.setdefault(tv.timestamp, {})
                    bucket[target] = bucket.get(target, 0.0) + count
            else:
                bs.append(sumo)
                ac.append(acto)

            # C++: precision_bscore.cc:254-263, "Build CTable of results"
            if self._pressure > 0.0:
                dummy_input: tuple = ()
                for tv, count in timed_items_of(counter):
                    tclass = self.get_timestamp_class(tv.timestamp)
                    result = builtin.logical_true if selected else builtin.logical_false
                    cell = ctable_res.get(dummy_input)
                    if cell is None:
                        cell = TimedCounter()
                        ctable_res[dummy_input] = cell
                    cell.add(result, tclass, count)

        # C++: precision_bscore.cc:266-275, "Fill un-normalized bscore
        # spread over the temporal axis". `std::map` order, so ascending
        # date with the not-a-date sentinel last (`timestamp_less`).
        if self.time_bscore:
            ordered = sorted(
                time2res, key=cmp_to_key(lambda a, b: -1 if timestamp_less(a, b) else 1)
            )
            for timestamp in ordered:
                counts = time2res[timestamp]
                tp = counts.get(True, 0.0)
                fp = counts.get(False, 0.0)
                bs.append(_to_score_t(0.5 * (tp - fp)))
                ac.append(_to_score_t(tp + fp))

        if 0.0 < active:
            # normalization is the inverse of activity
            iac = _to_score_t(_ieee_div(1.0, active))

            # C++: precision_bscore.cc:283-289. "For the boosted scorer,
            # the active rows are weighted too.  Its still tp+fp, except
            # that the bscore for fp are negative..." (:281-282's own
            # comment.)  `wactive` is a **double** accumulator
            # (`double wactive = 0.0;`), and `_weights[i]` is a double, so
            # the product is computed at full precision and NOTHING is
            # narrowed inside the loop -- unlike `active`/`sao` above,
            # which are real `score_t` members and narrow on every `+=`.
            # Only the final `iac = 1.0 / wactive;` narrows, once, because
            # `iac` is the `score_t` declared two lines up.
            if self._return_weighted_score:
                # DIVERGENCE(C++ moses/moses/scoring/precision_bscore.cc:285):
                # this loop runs to `_size`, the COMPRESSED ROW count, but
                # under `time_bscore` `ac` holds one entry per distinct
                # TIMESTAMP instead (precision_bscore.cc:273) -- so
                # `--time-bscore=1 --boost=1` indexes a `std::vector<score_t>`
                # past its end with a bare `operator[]`, which is undefined
                # behaviour, not a wrong-but-defined answer.
                #
                # MEASURED on tests/fixtures/table_data/timestamp_bool_clean
                # .csv (9 rows, 4 compressed, 3 distinct dates) with
                # `-Hpre -q0.1 -w0.9 --timestamp-feature=when --time-bscore=1
                # --boost=1 -m200 -W1`: classic exits 0 and prints
                # `0.5 or($1 $2)`, reproducibly across seeds 1/2/3 and repeat
                # runs -- reproducible here, but computed from whatever
                # follows a 3-element heap vector, so it is a property of this
                # build's allocator rather than of MOSES. A port cannot
                # reproduce that, and reproducing a *plausible* number would
                # be worse than failing: it would look like agreement. This
                # raises instead, naming both sizes -- the same choice
                # `interpreter.py` makes for `interpreter.cc:54`'s unchecked
                # argument index (docs/CARRY-FORWARD.md Sec 3). DO NOT "fix"
                # by clamping or by padding `ac`.
                #
                # The IN-BOUNDS case (at least as many timestamps as rows) is
                # NOT undefined in C++, merely odd -- it pairs row weights
                # with timestamp activities -- so it is reproduced faithfully
                # rather than rejected.
                if len(ac) < self._size:
                    raise IndexError(
                        "precision_bscore.do_score: the boosted normalisation "
                        f"reads {self._size} activity entries but time_bscore "
                        f"produced only {len(ac)} (one per timestamp). C++ "
                        "reads past the end of the vector here "
                        "(precision_bscore.cc:285); that is undefined "
                        "behaviour and cannot be ported."
                    )
                wactive = 0.0
                for i in range(self._size):
                    wactive += self._weights[i] * ac[i]
                iac = _to_score_t(_ieee_div(1.0, wactive))

            # Normalize all components by active, where active = tp+fp
            bs = [_to_score_t(v * iac) for v in bs]
            bs.append(0.5)
        else:
            bs.append(0.0)

        if not self._return_weighted_score:
            activation = _to_score_t(_ieee_div(active, self._ctable_weight))
            activation_penalty = self.get_activation_penalty(activation)
            bs.append(activation_penalty)

            # C++: precision_bscore.cc:339-345, "Add time dispersion
            # penalty" -- appended AFTER the activation penalty, and only
            # on the non-boosting path.
            if self._pressure > 0.0:
                bs.append(self.get_time_dispersion_penalty(ctable_res.ordered_by_time()))

        return bs

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:352 precision_bscore::operator()"""
        iv = interpreter_visitor(tr)

        def selector(row):
            return vertex_equal(iv(list(row)), builtin.logical_true)

        return self.do_score(selector)

    def call_ensemble(self, ensemble) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:368 precision_bscore
        (the `operator()(const scored_combo_tree_set&)` overload)

        "The ensemble score must not use weights, but must be give the
        result that the ensemble would give on regular data.  Thus,
        disable weighting while we compute the ensemble score."
        (precision_bscore.cc:370-372's own comment.)

        `_return_weighted_score` is `mutable` in C++ (scoring_base.h:230)
        precisely so this `const` method can toggle it off and back on
        around the call.  The toggle is NOT exception-safe in C++ either
        (no RAII guard, no try/catch), so a raise out of
        `exact_selection`/`bias_selection` leaves the flag `False` in both
        languages -- reproduced rather than "fixed" with `try/finally`.

        `OC_ASSERT(_return_weighted_score, "Unexpected call to ensemble
        scorer!")` (precision_bscore.cc:373) -- this method is only ever
        reachable when boosting, because `table-problems.cc:385` is the
        only thing that sets the flag on a `precision_bscore`.
        """
        if not self._return_weighted_score:
            raise AssertionError("Unexpected call to ensemble scorer!")
        if self.exact_experts:
            self._return_weighted_score = False
            bs = self.exact_selection(ensemble)
            self._return_weighted_score = True
            return bs
        else:
            self._return_weighted_score = False
            bs = self.bias_selection(ensemble)
            self._return_weighted_score = True
            return bs

    def exact_selection(self, ensemble) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:391 precision_bscore::exact_selection

        "scorer, suitable for use with boosting, where the members of the
        boosted ensemble are always correct in their selections. That is,
        they are 'exact' scorers." (precision_bscore.cc:386-389's own doc
        comment.)

        Only the `#else` arm is ported: `SCORE_MANY_TREES`
        (precision_bscore.cc:398-427) is `#define`d out in classic's own
        source (the `// #define SCORE_MANY_TREES` at :397 is commented
        out), so it never compiles -- the same "port only the branch the
        preprocessor actually builds" convention `bscore_base.update_
        weights` already applies to its `#else`.  C++'s own comment says
        "either way, the two scorers here should be functionally
        equivalent".

        The `or`-of-all-members tree is built in whatever order
        `scored_combo_tree_set` iterates, which neither language
        specifies -- but unlike `ensemble.add_adaboost`'s `min_element`,
        that order is NOT observable here: `logical_or` is commutative
        and the tree is only ever interpreted, never printed, so every
        permutation selects exactly the same rows and yields the same
        bscore.
        """
        # For large tables, it is much more efficient to convert the
        # ensemble into a single tree, so that we iterate over the table
        # just once, instead of once-per-tree.
        if len(ensemble) == 1:
            return self(next(iter(ensemble)).get_tree())

        tr = combo_tree()
        head = tr.set_head(builtin.logical_or)

        for sct in ensemble:
            tr.append_child(head, sct.get_tree().begin())

        return self(tr)

    def bias_selection(self, ensemble) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:453 precision_bscore::bias_selection

        "scorer, suitable for use with boosting, where the members of the
        ensemble sometimes make mistakes. The mistakes are avoided by
        requiring the ensemble to cast a minimum vote, before a row is
        truely considered to be selected.  The minimum vote is called the
        'bias' in the code below." (precision_bscore.cc:447-451's own doc
        comment, typo in the original not corrected.)

        UNREACHABLE from classic's CLI -- see this module's docstring for
        the `get_error` assertion that fires first.  Ported and unit-
        tested directly.

        `score_t trweight = sct.get_weight();` (precision_bscore.cc:460)
        NARROWS the `double` ensemble weight to binary32 before it is
        accumulated -- `hypoth` itself is a `std::vector<double>`, so the
        accumulation is in double but each addend has already been
        rounded to float.  `double sumo = sum_outputs(orow);`
        (precision_bscore.cc:485) is the mirror image: `sum_outputs`
        returns a `score_t` which is then widened, so the comparison
        `0.0 > sumo` is made on the float32 value.

        The selector is stateful: `hypoth[i++]` (precision_bscore.cc:496)
        walks the table in the SAME order `do_score` will, which is what
        makes the index line up.  That is `_wrk_ctable`'s own iteration
        order in both languages.

        The `logger().info() << "Precision ensemble bias: "` line
        (precision_bscore.cc:489) is not reproduced: this module emits no
        log output at all (it imports no logger), unlike select_bscore.py.
        """
        # Step 1: If any tree in the ensemble picks a row, that row is
        # picked with appropriate weight.
        hypoth = [0.0] * self._size
        for sct in ensemble:
            tr = sct.get_tree()
            trweight = _to_score_t(sct.get_weight())

            iv = interpreter_visitor(tr)

            i = 0
            for row in self._wrk_ctable:
                if vertex_equal(iv(list(row)), builtin.logical_true):
                    hypoth[i] += trweight
                i += 1

        # Step 2: find the worst wrong answer; that is our bias.
        i = 0
        bias = 0.0
        for _row, orow in self._wrk_ctable.items():
            # sumo will be negative if it should not be selected.
            # We look at the negative ones, because they are wrong...
            sumo = self.sum_outputs(orow)
            if 0.0 > sumo and bias < hypoth[i]:
                bias = hypoth[i]
            i += 1

        # Step 3: tot up the active rows.
        state = {"i": 0}

        def selector(_row):
            idx = state["i"]
            state["i"] = idx + 1
            return self.bias_scale * bias < hypoth[idx]

        return self.do_score(selector)

    def get_error(self, tr: combo_tree) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:501 precision_bscore::get_error

        `OC_ASSERT(exact_experts, "Precision scorer: inexact experts not
        yet supported!");` (precision_bscore.cc:503-504) -- ported
        verbatim, and it is what makes `--boost-exact=0` abort before
        `bias_selection` can ever run (see this module's docstring).

        The `_return_weighted_score` toggle around the `operator()` call
        (precision_bscore.cc:507-509) is load-bearing, not decorative:
        `ensemble::add_expert` (ensemble.cc:184) calls this on every
        candidate while the flag is `True`, and it is the toggle that
        makes `do_score` normalise by the FLAT activity (`1.0 / active`)
        rather than the weighted one. Without it the "perfect expert"
        test `err <= _tolerance` would drift with the row weights.
        """
        if not self.exact_experts:
            raise AssertionError("Precision scorer: inexact experts not yet supported!")

        self._return_weighted_score = False
        bs = self(tr)
        self._return_weighted_score = True

        accum = 0.0
        for i in range(self._size):
            accum += bs[i]
        return _to_score_t(0.5 - accum)

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:520 precision_bscore::best_possible_bscore

        `max_precisions` (a `std::multimap<contin_t, pair<contin_t,
        count_t>>`, ascending by `contin_t`/double precision) is a plain
        Python `list` of `(precision, sumo, total)` tuples, sorted
        ascending, then walked in reverse -- same convention
        `discriminating_bscore.best_possible_bscore` uses (see its own
        docstring for the tie-order note, which applies identically
        here: ties in `precision` are UNTESTED on the current fixture).
        """
        max_precisions: list[tuple[float, float, float]] = []
        for counter in self._wrk_ctable.values():
            # count_t sumo = sum_outputs(c); -- widening score_t -> double, no narrow
            sumo = self.sum_outputs(counter)
            total = sum(counter.values())
            precision = sumo / total  # double precision = sumo / total;
            max_precisions.append((precision, sumo, total))
        max_precisions.sort(key=lambda t: t[0])

        active = 0.0  # count_t active = 0.0;
        sao: score_t = 0.0
        best_sc: score_t = very_worst_score
        best_precision: score_t = 0.0
        best_activation: score_t = 0.0
        best_activation_penalty: score_t = 0.0

        for _precision, sumo, total in reversed(max_precisions):
            sao = _to_score_t(sao + sumo)
            active += total  # count_t accumulator, no narrow

            precision = _to_score_t(_ieee_div(sao, active) + 0.5)
            activation = _to_score_t(_ieee_div(active, self._ctable_weight))
            activation_penalty = self.get_activation_penalty(activation)
            sc = _to_score_t(precision + activation_penalty)

            if sc > best_sc:
                best_sc = sc
                best_precision = precision
                best_activation = activation
                best_activation_penalty = activation_penalty

            if self._ctable_weight * self.min_activation <= active:
                break

        del best_precision, best_activation, best_activation_penalty  # diagnostics only in C++
        return [best_sc]

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:614 precision_bscore::worst_possible_bscore

        "The worst possible score is to fail to identify any of the
        positive rows. This renders TP = 0." (precision_bscore.cc:
        611-613's own comment.)
        """
        return [0.0]

    def get_activation_penalty(self, activation: score_t) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:622 precision_bscore::get_activation_penalty

        "Note that it returns -inf if activation is 0.0 -- but that's OK,
        it won't hurt anything." (precision_bscore.cc:621's own comment.)
        Same two-step, double-rounding-aware narrowing as
        `discriminating_bscore.get_threshold_penalty` -- see that
        method's own docstring -- but with NO `0.99999999 < dst`
        saturation guard: this class's own C++ source does not have one,
        so `log(1.0 - dst)` for `dst >= 1.0` IS reachable here (unlike
        the discriminating-scorer family). C's `log(0.0)` returns
        `-HUGE_VAL` (`-inf`, no exception) and `log(negative)` returns
        `NaN` (a domain error, also no exception) -- Python's `math.log`
        RAISES `ValueError` for both, so both are handled explicitly
        rather than narrowing precision alone via `_to_score_t`.
        """
        if self.activation_pressure > 0:
            dst: score_t = 0.0
            if activation < self.min_activation:
                ratio = _to_score_t(activation / self.min_activation)
                dst = _to_score_t(1.0 - ratio)
            if self.max_activation < activation:
                numer = _to_score_t(activation - self.max_activation)
                dst = _to_score_t(_ieee_div(numer, 1.0 - self.max_activation))
            log_arg = 1.0 - dst
            if log_arg == 0.0:
                log_val = -math.inf
            elif log_arg < 0.0:
                log_val = math.nan
            else:
                log_val = math.log(log_arg)
            return _to_score_t(self.activation_pressure * log_val)
        return 0.0

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:639 precision_bscore::min_improv"""
        return _to_score_t(_ieee_div(1.0, self._ctable_weight))

    def gen_canonical_best_candidate(self) -> combo_tree:
        """C++: moses/moses/scoring/precision_bscore.cc:644 gen_canonical_best_candidate

        "we generate a massive combo tree that is supposed to maximize
        the precision" (precision_bscore.h:141-143's own doc comment.)
        `ptc` is sorted the same way `best_possible_bscore` sorts
        `max_precisions` -- ascending by (double) precision, walked in
        reverse.
        """
        ptc: list[tuple[float, tuple, float]] = []
        for row, counter in self._wrk_ctable.items():
            total = sum(counter.values())
            precision = self.sum_outputs(counter) / total
            ptc.append((precision, row, total))
        ptc.sort(key=lambda t: t[0])

        active = 0.0
        tr = combo_tree(builtin.logical_or)
        head = tr.begin()
        for _precision, row, total in reversed(ptc):
            active += total
            dch = tr.append_child(head, builtin.logical_and)
            idx = 1
            for v in row:
                is_true = vertex_equal(v, builtin.logical_true)
                tr.append_child(dch, argument(idx if is_true else -idx))
                idx += 1
            if self._ctable_weight * self.min_activation <= active:
                break
        return tr

    def reset_weights(self) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:698 precision_bscore::reset_weights"""
        self.wnorm = 1.0
        if self._return_weighted_score:
            self._weights = [1.0] * self._size
        else:
            self._weights = []

    def update_weights(self, rew: list[float]) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:707 precision_bscore::update_weights

        Boosting-only -- never called anywhere in this port.
        """
        if not self._return_weighted_score:
            raise AssertionError("Unexpected use of weights in the bscorer!")
        if len(rew) != self._size:
            raise AssertionError("Unexpected size of weight array!")

        znorm = 0.0
        for i in range(self._size):
            self._weights[i] *= self.wnorm
            if rew[i] < 1.0:
                self._weights[i] *= rew[i]
            znorm += self._weights[i]

        znorm = _ieee_div(float(self._size), znorm)
        self.wnorm = _ieee_div(1.0, znorm)
        for i in range(self._size):
            self._weights[i] *= znorm


class precision_conj_bscore(bscore_base):
    """C++: moses/moses/scoring/precision_bscore.h:202 precision_conj_bscore

    "A bit like precision but tries to maximize the number conjunctions
    after precision. It blatantly assumes that the domain is boolean."
    (precision_bscore.h:198-201's own doc comment.) No time/boosting
    dependency at all -- fully portable.
    """

    def __init__(self, ctable: CTable, hardness: float, positive: bool = True) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:739 precision_conj_bscore"""
        super().__init__()
        self.ctable = ctable
        self.ctable_usize = ctable.uncompressed_size()
        self.hardness = hardness
        self.positive = positive
        self._target = bool_to_vertex(positive)

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:750 set_complexity_coef(unsigned, float)
        and precision_bscore.cc:767 set_complexity_coef(score_t)
        """
        if len(args) == 1:
            (ratio,) = args
            ratio = _to_score_t(ratio)
            self._complexity_coef = 0.0
            if ratio > 0:
                self._complexity_coef = _to_score_t(_ieee_div(1.0, ratio))
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                self._complexity_coef = _to_score_t(
                    _ieee_div(discrete_complexity_coef(alphabet_size, p), self.ctable_usize)
                )
        else:
            raise TypeError(
                "set_complexity_coef expects either (ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:776 precision_conj_bscore::operator()"""
        active = 0.0  # count_t
        sao: score_t = 0.0
        iv = interpreter_visitor(tr)
        for row, counter in self.ctable.items():
            if vertex_equal(iv(list(row)), builtin.logical_true):
                sumo = _to_score_t(counter.get(self._target, 0.0))
                totalc = sum(counter.values())
                sao = _to_score_t(sao + sumo)
                active += totalc

        precision: score_t = 1.0
        if 0 < active:
            precision = _to_score_t(_ieee_div(sao, active))
        bs: behavioral_score = [precision]

        conj_n = 0
        head = tr.begin()
        if vertex_equal(head.data, builtin.logical_and):
            conj_n += 1
        sib = head.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.logical_and):
                conj_n += 1
            sib = sib.next_sibling
        conj_n_penalty = _to_score_t(self.hardness * (-1.0 / (1.0 + conj_n)))
        bs.append(conj_n_penalty)
        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:826 best_possible_bscore"""
        return [1.0, 0.0]

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:831 precision_conj_bscore::min_improv"""
        return 0.0
