"""`python -m pymoses.feature_selection.main` -- the `feature-selection` entry point.

C++: moses/feature-selection/main/fs-main.cc:224 main

The CMake target `fs-main` installs as the binary `feature-selection`;
`[project.scripts]` gives this module that same name. `sys.argv` is passed
whole, `argv[0]` included, for the reason `pymoses/moses/main/__main__.py`
records.
"""

import sys

from pymoses.feature_selection.main.fs_main import fs_main

if __name__ == "__main__":
    sys.exit(fs_main(sys.argv))
