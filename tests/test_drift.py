"""7-9: particle initialisation, backward simulation, source probability generation."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from shapely.geometry import Point, box, mapping, shape

from app.core.errors import ForcingUnavailable
from app.drift import backtrack as bt
from app.drift import source_probability as sp
from app.environmental.forcing import NetCDFForcingProvider, build_provider

OBS = datetime(2025, 6, 10, 13, 4, tzinfo=timezone.utc)
SPILL = mapping(box(71.78, 18.68, 71.80, 18.70))


def test_particles_sampled_throughout_polygon():
    lon, lat = bt.sample_points_in_polygon(SPILL, 500, np.random.default_rng(0))
    poly = shape(SPILL)
    assert all(poly.contains(Point(x, y)) for x, y in zip(lon, lat))
    # spread across the polygon, not clustered at the centroid
    assert np.ptp(lon) > 0.015 and np.ptp(lat) > 0.015


def test_missing_forcing_file_is_actionable(tmp_path):
    with pytest.raises(ForcingUnavailable) as e:
        NetCDFForcingProvider([tmp_path / "c.nc"], [tmp_path / "w.nc"])
    assert "open_meteo" in e.value.hint


def test_forcing_coverage_check(cfg, demo_ready):
    p = build_provider(cfg)
    p.check_coverage([71.7, 18.6, 71.9, 18.8], OBS - timedelta(hours=48), OBS)
    with pytest.raises(ForcingUnavailable):
        p.check_coverage([71.7, 18.6, 71.9, 18.8], OBS - timedelta(days=30), OBS)   # outside time range
    with pytest.raises(ForcingUnavailable):
        p.check_coverage([60, 10, 61, 11], OBS - timedelta(hours=6), OBS)          # outside domain
    assert p.provenance == "SYNTHETIC_DEMO"


@pytest.fixture(scope="module")
def backward(cfg, demo_ready):
    p = build_provider(cfg)
    return bt.run_ensemble(p, SPILL, OBS, 6, cfg.drift, "backward", n_particles=50, members=2)


def test_backward_simulation_moves_back_in_time(backward):
    assert backward.times[0] == OBS and backward.times[-1] == OBS - timedelta(hours=6)
    assert backward.lon.shape == (2, 50, 7)
    lon0, lat0 = np.nanmean(backward.lon[:, :, 0]), np.nanmean(backward.lat[:, :, 0])
    lon6, lat6 = np.nanmean(backward.lon[:, :, -1]), np.nanmean(backward.lat[:, :, -1])
    assert np.hypot(lon6 - lon0, lat6 - lat0) > 0.01                  # particles moved
    assert -180 <= np.nanmin(backward.lon) and np.nanmax(backward.lat) <= 90   # valid coordinates
    # demo forcing drifts oil E-SE, so the backtracked source lies to the W
    assert lon6 < lon0
    assert "OpenOil" in backward.model and backward.config.get("processes:evaporation") is False


def test_source_probability_normalised_with_nested_regions(backward):
    grid = sp.build_grid(backward, {"lat": 18.69, "lon": 71.79}, [3, 6], OBS, 500, 800)
    for label in ("T-3h", "T-6h", "combined"):
        assert grid.maps[label].sum() == pytest.approx(1.0, abs=1e-6)
    regs = sp.regions(grid, {"high": 0.5, "medium": 0.8, "low": 0.95})
    assert regs["high"]["area_km2"] <= regs["medium"]["area_km2"] <= regs["low"]["area_km2"]
    assert regs["low"]["_shape"].contains(regs["high"]["_shape"].representative_point())
    gj = sp.to_geojson(grid, regs)
    cell = next(f for f in gj["features"] if f["properties"]["kind"] == "probability_cell")
    assert {"probability", "lat", "lon", "release_time"} <= set(cell["properties"])
