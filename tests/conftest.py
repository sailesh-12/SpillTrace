"""Test configuration.

The application itself runs on REAL data only. Automated tests use the deterministic synthetic
fixtures in tests/fixtures/data (a labelled twin experiment) so they run offline and reproducibly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "data"
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_config  # noqa: E402

FIXTURE_SCENE = FIX / "sar" / "demo" / "demo_scene.png"


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    return load_config(overrides={
        "mode": "test",
        "storage": {"output_dir": str(out), "backend": "file"},
        "environment": {"provider": "netcdf", "current_files": [str(FIX / "forcing" / "demo_currents.nc")],
                        "wind_files": [str(FIX / "forcing" / "demo_wind.nc")]},
        "ais": {"mode": "real", "real_provider": "local", "files": [str(FIX / "ais" / "demo_ais.csv")],
                "column_map": "default"},
        "lookalike": {"fetch_wind": False},
        "drift": {"particles": 60, "ensemble_runs": 2, "forward": {"particles": 40}},
    })


@pytest.fixture(scope="session")
def demo_ready():
    needed = [FIX / "forcing" / "demo_currents.nc", FIX / "ais" / "demo_ais.csv", FIXTURE_SCENE]
    if not all(p.exists() for p in needed):
        pytest.skip("fixtures missing: run python tests/fixtures/make_synthetic_fixtures.py")
    return True


@pytest.fixture(scope="session")
def model(cfg):
    p = cfg.path(cfg.segmentation.model_path)
    if not p.exists():
        pytest.skip("model checkpoint missing")
    from app.segmentation.model_loader import load_model
    return load_model(p, cfg.segmentation.encoder_name, "auto")
