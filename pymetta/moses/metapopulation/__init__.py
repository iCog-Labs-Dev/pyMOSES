"""Metapopulation: the pool of scored candidates and its deme factory.

`metapop_params.py` (parameters + gates) and `ensemble.py` (a minimal,
non-boosting-capable stand-in) are ported here first (S10 Task 1) because
`metapopulation.h` unconditionally includes `ensemble.h` -- see
`docs/MODULE-COVERAGE.md`'s ORDERING HAZARD note for
`moses/moses/metapopulation/ensemble.*`. `metapopulation.py` and
`merging.py` (S10 Tasks 2-3) are not yet present.
"""
