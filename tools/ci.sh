#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Self-bootstrapping: dev deps are pytest + ruff only (see pyproject.toml
# [project.optional-dependencies].dev). This never touches system/user
# site-packages -- everything lands in a local, gitignored .venv -- and it
# is idempotent: once .venv/bin/{ruff,pytest} exist, no install step runs.
VENV=.venv

if [ ! -x "$VENV/bin/python3" ]; then
  python3 -m venv "$VENV"
fi

if [ ! -x "$VENV/bin/ruff" ] || [ ! -x "$VENV/bin/pytest" ]; then
  "$VENV/bin/python3" -m pip install --quiet "pytest>=9.0" "ruff>=0.6"
fi

PY="$VENV/bin/python3"
"$PY" -m ruff check pymetta tools tests
"$PY" -m pytest
"$PY" tools/check_alignment.py
"$PY" tools/coverage_totals.py --check
"$PY" tools/check_fixture_manifests.py
