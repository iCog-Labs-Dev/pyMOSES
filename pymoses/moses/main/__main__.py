"""`python -m pymoses.moses.main` -- the port's `moses` entry point.

C++: moses/moses/main/moses-exec.cc:31 main

Classic's whole `main()` is

    int main(int argc, char** argv) {
        return moses_exec(argc, argv);
    }

and the CMake target `moses-exec` installs as the binary `moses`. This
module is that three-line `main()`, and `[project.scripts]` in
pyproject.toml gives it the same installed name.

`sys.argv` is passed WHOLE, `argv[0]` included: `moses_exec` reads the
program name from it, exactly as every C++ overload does -- and classic's
`-F`/`determine_log_name` names its log after the program too. Dropping
`argv[0]` here would silently shift every option by one.
"""

import sys

from pymoses.moses.main.moses_exec import moses_exec

if __name__ == "__main__":
    sys.exit(moses_exec(sys.argv))
