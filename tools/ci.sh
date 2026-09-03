#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m ruff check pymetta tools tests
python3 -m pytest
python3 tools/check_alignment.py
