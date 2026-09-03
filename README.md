# pyMeTTa

Python port of classic OpenCog MOSES (a C++ evolutionary program learner),
aligned line-for-line to the C++ source it is ported from.

## Why

The port's entire value rests on being auditable against the C++: every
ported symbol carries a `C++: <path>:<line> <symbol>` docstring breadcrumb,
and `tools/check_alignment.py` (run in CI) checks that every breadcrumb still
points at a real, nearby line in the classic checkout.

## Layout

- `pymetta/` — the port itself, mirroring the classic C++ source tree
  (`comboreduct/...`, `moses/...`) file-for-file.
- `tools/` — the alignment checker, the C++ oracle probes, and the golden
  fixture capture script.
- `tests/` — the pytest suite, including `oracle` tests that diff against
  committed C++ golden fixtures and `live` tests that require rebuilding the
  C++ probes.

## Requirements

Python 3.14, standard library only at runtime. Dev tools: `pytest`, `ruff`.

## Development

```bash
bash tools/ci.sh   # ruff + pytest + alignment checker
```

`tools/ci.sh` is self-bootstrapping: on first run it creates a local `.venv`
and installs the pinned dev tools (`pytest`, `ruff`) into it; later runs
reuse that `.venv` and skip installation entirely. No manual venv creation
or activation, and no `PATH`/`pip install` step, is required first.

See `docs/superpowers/specs/2026-09-02-pymetta-foundation-design.md` for the
full design and `docs/superpowers/plans/2026-09-03-pymetta-s0-s6-foundation.md`
for the task-by-task implementation plan.
