import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASSIC = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses")
COGUTIL = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/CogUtil")

@pytest.fixture(scope="session")
def classic_root() -> Path:
    if not CLASSIC.is_dir():
        pytest.skip(f"classic MOSES checkout not found at {CLASSIC}")
    return CLASSIC

@pytest.fixture(scope="session")
def cogutil_root() -> Path:
    if not COGUTIL.is_dir():
        pytest.skip(f"CogUtil checkout not found at {COGUTIL}")
    return COGUTIL

@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
