"""Smoke tests for the package skeleton: importability and metadata."""

import tomllib
from pathlib import Path

import pymoses

REPO = Path(__file__).resolve().parent.parent


def test_pymoses_package_imports():
    assert pymoses is not None


def test_pyproject_declares_pymoses():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["name"] == "pymoses"
    assert data["project"]["requires-python"] == ">=3.14"


def test_readme_exists():
    assert (REPO / "README.md").is_file()
