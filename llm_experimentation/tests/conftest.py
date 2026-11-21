"""Fixtures shared by tests/unit and tests/integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))      # so tests can `import helpers`

import helpers  # noqa: E402


@pytest.fixture
def toy_sel(tmp_path_factory):
    """A fresh 30-item synthetic Selection (see helpers.make_toy_sel)."""
    return helpers.make_toy_sel(tmp_path_factory.getbasetemp())


@pytest.fixture
def needs_real_data():
    """Skip the test when the Pokémon data folder is missing."""
    if not (helpers.REPO_ROOT / "data" / "pokemon_db").exists():
        pytest.skip("Pokémon data folder not present")
