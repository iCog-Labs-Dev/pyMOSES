"""Unit tests for pymoses.moses.moses.optimization.optimization.

No C++ probe/fixture backs this module (it is pure arithmetic over an
already-oracle-tested field_set, plus one abstract-base contract) -- every
assertion here is hand-computed and documented, not re-typed from a JSON
fixture. `information_theoretic_bits`'s mixed-field-set case specifically
targets a bug this task's own review found by hand-tracing (not by any
probe corpus: every hillclimbing.json case happens to have zero bit
fields, so this exact regression would not show up there) -- see the
mutation list in the task report.
"""

import math

import pytest

from pymoses.moses.moses.optimization.optimization import (
    deme_inst_t,
    deme_t,
    information_theoretic_bits,
    optim_parameters,
    optim_stats,
    optimizer_base,
)
from pymoses.moses.moses.types import demeID_t
from pymoses.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymoses.moses.representation.scored_instance import scored_instance


def test_deme_aliases_are_the_bridge_containers():
    assert deme_t is __import__(
        "pymoses.moses.representation.instance_set", fromlist=["instance_set"]
    ).instance_set
    assert deme_inst_t is scored_instance


# --------------------------------------------------------- information_theoretic_bits


def test_information_theoretic_bits_all_bit_fields():
    # 5 single-bit knobs: log2(2) == 1 each, no discrete-multi/contin terms.
    fs = field_set(disc_spec(2), 5)
    assert information_theoretic_bits(fs) == pytest.approx(5.0)


def test_information_theoretic_bits_mixed_disc_and_bit_fields():
    # 2 genuine multi-valued disc(3) knobs + 3 bit(2) knobs. Hand-computed:
    # 2*log2(3) + 3*log2(2) = 2*log2(3) + 3.
    fs = field_set([disc_spec(3), disc_spec(3), disc_spec(2), disc_spec(2), disc_spec(2)])
    assert fs.n_disc_fields() == 2
    assert fs.n_bits() == 3
    expected = 2 * math.log2(3) + 3
    assert information_theoretic_bits(fs) == pytest.approx(expected)


def test_information_theoretic_bits_includes_contin_term():
    # 1 bit field + 1 contin field: 1 + log2(5).
    fs = field_set([disc_spec(2), contin_spec(0.0, 1.0, 2.0, 3)])
    assert information_theoretic_bits(fs) == pytest.approx(1.0 + math.log2(5.0))


# ------------------------------------------------------------------- optim_parameters


def test_optim_parameters_defaults_match_cpp_constructor():
    op = optim_parameters()
    assert op.opt_algo == "hc"
    assert op.pop_size_ratio == 20
    assert op.terminate_if_gte == 0
    assert op.max_dist == 4
    assert op.min_score_improv() == pytest.approx(0.5)


def test_max_distance_clamps_to_field_set_dimension():
    fs = field_set(disc_spec(2), 2)  # dim_size() == 2
    op = optim_parameters(max_dist=4)
    assert op.max_distance(fs) == 2  # min(4, 2)


def test_score_improved_nonnegative_branch():
    # imp >= 0: big_step iff best_score > prev_hi + imp.
    op = optim_parameters(min_score_improv=0.5)
    assert op.score_improved(10.6, 10.0) is True  # 10.6 > 10.5
    assert op.score_improved(10.5, 10.0) is False  # not strictly greater
    assert op.score_improved(10.4, 10.0) is False


def test_score_improved_negative_branch_is_a_fraction():
    # imp < 0: big_step iff best_score > prev_hi - imp*|prev_hi|
    # (an extra minus sign because imp is negative, per the header comment).
    op = optim_parameters(min_score_improv=-0.1)
    prev_hi = 100.0
    threshold = prev_hi - (-0.1) * abs(prev_hi)  # 100 + 10 = 110
    assert op.score_improved(threshold + 1, prev_hi) is True
    assert op.score_improved(threshold, prev_hi) is False


# ----------------------------------------------------------------------- optim_stats


def test_optim_stats_defaults():
    stats = optim_stats()
    assert stats.nsteps == 0
    assert stats.demeID == demeID_t(0)
    assert stats.total_steps == 0
    assert stats.total_evals == 0
    assert stats.field_set_size == 0
    assert stats.over_budget is False


# -------------------------------------------------------------------- optimizer_base


def test_optimizer_base_call_is_not_implemented():
    base = optimizer_base()
    with pytest.raises(NotImplementedError):
        base(None, None, 0, 0)


def test_optimizer_base_inherits_optim_stats_and_default_params():
    base = optimizer_base()
    assert isinstance(base, optim_stats)
    assert isinstance(base.opt_params, optim_parameters)
    assert base.opt_params.opt_algo == "hc"


# ---------------------------------------------------------------------------- guards


def test_information_theoretic_bits_bug_would_double_count_bits():
    """Not a guard on the shipped code -- documents, by direct computation,
    that the naive "iterate the whole disc_and_bit() sequence" version this
    task's own review caught and fixed would have doubled every bit field's
    contribution. Mirrors the mutation performed and reverted for this fix
    (see the task report's mutation list): temporarily iterating the full
    sequence instead of `[:fs.n_disc_fields()]` reproduces exactly this
    wrong total.
    """
    fs = field_set([disc_spec(3), disc_spec(3), disc_spec(2), disc_spec(2), disc_spec(2)])
    correct = 2 * math.log2(3) + 3
    buggy = sum(math.log2(s.multy) for s in fs.disc_and_bit()) + fs.n_bits()
    assert buggy == pytest.approx(correct + 3)  # double-counts the 3 bit fields
    assert information_theoretic_bits(fs) == pytest.approx(correct)
