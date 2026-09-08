"""The deme factory: builds a representation + instance_set ("deme") from an
exemplar combo_tree, and drives an injected optimizer over it.

``deme_params.py`` (``deme_parameters``) and ``deme_expander.py``
(``deme_expander``) port ``moses/moses/deme/deme_params.h`` and
``moses/moses/deme/deme_expander.{h,cc}``, single-deme path only --
see ``deme_expander.py``'s own module docstring for the two P2 gates
(on-the-fly feature selection, subsample-deme filtering) this excludes.
"""

from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters

__all__ = ["deme_expander", "deme_parameters"]
