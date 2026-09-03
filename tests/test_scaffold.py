"""Smoke tests for the package skeleton: importability and metadata."""

import tomllib
from pathlib import Path

import pymetta

REPO = Path(__file__).resolve().parent.parent


def test_pymetta_package_imports():
    assert pymetta is not None


def test_pyproject_declares_pymetta():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["name"] == "pymetta"
    assert data["project"]["requires-python"] == ">=3.14"


def test_readme_exists():
    assert (REPO / "README.md").is_file()
