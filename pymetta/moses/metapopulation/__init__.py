"""Metapopulation: the pool of scored candidates and its deme factory.

`metapop_params.py` (parameters + gates) and `ensemble.py` (a minimal,
non-boosting-capable stand-in) are ported here first (S10 Task 1) because
`metapopulation.h` unconditionally includes `ensemble.h` -- see
`docs/MODULE-COVERAGE.md`'s ORDERING HAZARD note for
`moses/moses/metapopulation/ensemble.*`. `metapopulation.py` and `merging.py`
(S10 Tasks 2-3) are present alongside them.

WHY THIS MODULE IMPORTS `merging` FOR ITS SIDE EFFECT
-----------------------------------------------------
Classic splits `metapopulation`'s methods across two translation units:
`metapopulation.cc` and `merging.cc` both define `metapopulation::` members,
and the linker guarantees a program sees all of them. This port mirrors that
file split, and `merging.py` attaches its half onto the class with `setattr`
at import time.

Python has no linker, so that attachment happens ONLY if something imports
`merging` -- which made `metapopulation.merge_demes` exist or not depending on
import order. Measured: importing `metapopulation` alone left `merge_demes`
absent while `select_exemplar` (defined in `metapopulation.py` itself) was
present, and an S11 agent blocked on exactly that. Importing `merging` here
restores C++'s guarantee: obtaining the class by any route yields the complete
method set.

Do not remove this import because it looks unused. It is the port's stand-in
for the linker, and `tests/moses/metapopulation/test_package_binding.py` pins
that `merge_demes` is reachable without importing `merging` explicitly.
"""

from pymetta.moses.metapopulation import merging as _merging  # noqa: F401
