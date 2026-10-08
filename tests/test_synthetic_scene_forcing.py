"""SYNTHETIC SAR scene, dB point-target detection and SYNTHETIC forcing fallback (Indian waters)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import rasterio

from app.ais.track_processing import haversine_km
from app.environmental.synthetic_forcing import synthetic_provider
from app.sar.synthetic_scene import generate_scene
from app.sar_ship_detection.detector import DbPointTargetDetector, geolocate

T = datetime(2026, 9, 27, 0, 31, tzinfo=timezone.utc)
AOI = [71.6, 18.7, 71.9, 18.95]          # open Arabian Sea off Mumbai (small for test speed)


@pytest.fixture(scope="module")
def scene(tmp_path_factory, cfg):
    return generate_scene(AOI, T, tmp_path_factory.mktemp("syn"), 20, seed=4, norm=dict(cfg.sentinel1.normalization))


def test_synthetic_scene_is_labelled(scene):
    assert scene["provenance"] == "SYNTHETIC_DEMO" and "not a satellite observation" in scene["warning"]
    with rasterio.open(scene["path"]) as ds:
        assert ds.tags()["PROVENANCE"] == "SYNTHETIC_DEMO" and ds.crs.to_string() == "EPSG:4326"
    assert scene["synthetic_scene"]["discharging_vessel"]["time"] == T.isoformat()


def test_model_detects_synthetic_trail_but_not_lookalike(scene, model):
    from app.segmentation.inference import run_inference
    from app.segmentation.postprocess import clean_mask, components
    g = rasterio.open(scene["path"]).read(1)
    sea = np.load(scene["sea_mask"])
    b = run_inference(model, np.repeat(g[..., None], 3, 2))["binary_mask"] & sea
    comps = components(clean_mask(b, 50, 3))
    assert comps and b[sea].mean() < 0.02                      # the trail, not the whole sea
    dv = scene["synthetic_scene"]["discharging_vessel"]
    with rasterio.open(scene["path"]) as ds:
        r, c = ds.index(dv["lon"], dv["lat"])
    assert any(abs(k["bbox_px"][1] - r) < 800 and abs(k["bbox_px"][0] - c) < 800 for k in comps)


def test_db_point_targets_find_the_discharging_ship(scene):
    from app.geospatial.georeference import read_scene
    anom = np.load(scene["path"].replace(".tif", "_anom.npy"))
    dets = geolocate(DbPointTargetDetector(anom, 10.0, 2).detect(), read_scene(scene["path"]).georef)
    dv = scene["synthetic_scene"]["discharging_vessel"]
    assert any(haversine_km(d["lat"], d["lon"], dv["lat"], dv["lon"]) < 0.3 for d in dets)
    assert len(dets) <= 15                                      # speckle does not create false targets


def test_synthetic_forcing_is_labelled_and_seasonal(tmp_path):
    p = synthetic_provider(AOI, T - timedelta(hours=50), T + timedelta(hours=24), tmp_path)
    assert p.provenance == "SYNTHETIC_DEMO"
    s = p.sample(18.8, 71.75, T)
    assert 170 <= s["wind_from_deg"] <= 320 and 3 < s["wind_speed"] < 14      # SW-monsoon-season wind (September)
    p.check_coverage(AOI, T - timedelta(hours=49), T + timedelta(hours=24))


def test_forcing_auto_falls_back_to_synthetic(cfg, model, demo_ready):
    from app.core.config import load_config
    from app.services import pipeline as P
    from conftest import FIXTURE_SCENE
    c2 = load_config(overrides={**cfg, "environment": {**cfg["environment"], "mode": "auto",
                                                       "current_files": ["tests/none.nc"]}})
    an = P.run_full(c2, str(FIXTURE_SCENE), particles=40, members=2)
    assert an.state["status"] == "COMPLETED", an.state.get("error")
    assert an.load("drift/summary.json")["forcing"]["provenance"] == "SYNTHETIC_DEMO"
    assert any("SYNTHETIC" in w and "forcing" in w.lower() for w in an.state["warnings"])
