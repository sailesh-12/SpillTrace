"""5-6: geospatial conversion, area calculation, georeferencing failure modes."""
import json

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from app.core.errors import InputImageError
from app.geospatial.georeference import georef_from_dict, read_scene
from app.geospatial.polygon import build_spill_record, geodesic_metrics, pixel_to_wgs84


def test_png_without_georef_is_reported_not_invented(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(p)
    sc = read_scene(p)
    assert sc.georef is None and sc.georef_status == "GEOREFERENCING_UNAVAILABLE"


def test_sidecar_bbox_and_timestamp(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(p)
    (tmp_path / "x.png.geo.json").write_text(json.dumps(
        {"bbox": [70.0, 18.0, 70.2, 18.1], "timestamp": "2025-01-01T00:00:00Z", "provenance": "DEMO_ASSUMED"}))
    sc = read_scene(p)
    assert sc.georef.bounds_wgs84 == pytest.approx([70.0, 18.0, 70.2, 18.1])
    assert sc.timestamp.isoformat() == "2025-01-01T00:00:00+00:00"
    assert sc.georef.provenance == "DEMO_ASSUMED"


def test_invalid_bbox_rejected():
    with pytest.raises(InputImageError):
        georef_from_dict({"bbox": [70.2, 18.0, 70.0, 18.1]}, 10, 10, "t")


def test_unsupported_format(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("no")
    with pytest.raises(InputImageError):
        read_scene(p)


def test_geotiff_transform_and_tag_timestamp(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds
    p = tmp_path / "s1.tif"
    data = np.random.default_rng(0).random((1, 64, 64)).astype("float32") * 0.1   # linear sigma0
    with rasterio.open(p, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(72, 18, 72.1, 18.1, 64, 64)) as dst:
        dst.write(data)
        dst.update_tags(ACQUISITION_START_TIME="2025-06-10T13:04:00Z")
    sc = read_scene(p)
    assert sc.georef.source == "geotiff_transform" and sc.timestamp.hour == 13
    assert sc.rgb_u8.dtype == np.uint8 and sc.rgb_u8.shape == (64, 64, 3)


def test_pixel_to_wgs84_and_geodesic_area():
    g = georef_from_dict({"bbox": [72.0, 18.0, 72.01, 18.01]}, 100, 100, "t")
    poly = pixel_to_wgs84(box(0, 0, 100, 100), g)
    assert poly.bounds == pytest.approx((72.0, 18.0, 72.01, 18.01))
    m = geodesic_metrics(poly)
    # 0.01 deg lat ~ 1106 m, 0.01 deg lon at 18N ~ 1058 m
    assert m["area_m2"] == pytest.approx(1106 * 1058, rel=0.01)
    assert m["centroid"]["lat"] == pytest.approx(18.005, abs=1e-4)


def test_spill_record_multipolygon_preserves_components():
    g = georef_from_dict({"bbox": [72.0, 18.0, 72.01, 18.01]}, 100, 100, "t")
    a, b = pixel_to_wgs84(box(0, 0, 10, 10), g), pixel_to_wgs84(box(50, 50, 70, 70), g)
    rec = build_spill_record("SPILL_T", [a, b], [{"area_px": 100}, {"area_px": 400}], None, 0.9)
    assert rec["polygon"]["type"] == "MultiPolygon" and rec["n_components"] == 2
    assert rec["area_m2"] == pytest.approx(sum(c["area_m2"] for c in rec["components"]), rel=1e-6)
