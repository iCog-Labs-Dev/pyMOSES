"""Fixtures for the tests of the stand-alone GOMEA harness (NOT A PORT: the IT linkage study,
BACKLOG 13.9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from standalone_data import write_split


@pytest.fixture
def split_dir(tmp_path) -> Path:
    return write_split(tmp_path / "splits" / "s1")
