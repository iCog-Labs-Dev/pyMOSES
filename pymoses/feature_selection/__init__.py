"""C++: moses/feature-selection/

Package for the standalone feature-selection subsystem: the scorers
(``scorers/``, mutual information by default) and algorithms (``algo/``)
behind both the standalone ``feature-selection`` binary
(``moses/feature-selection/main/``) and the in-loop ``--enable-fs`` path
(``moses/moses/deme/feature_selector.{h,cc}``, ported at
``pymoses.moses.deme.feature_selector``).

Only the ``"simple"`` algorithm (``algo/simple.py``) and the ``"mi"``
(mutual information) scorer (``scorers/mutual_info.py``) are ported -- see
``main/feature_selection.py``'s module docstring for the dispatcher and
exactly which algorithms/scorers raise ``NotImplementedError``.
"""
