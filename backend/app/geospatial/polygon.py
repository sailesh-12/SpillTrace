"""Pixel polygons -> WGS84 geometry + geodesic spill characteristics."""
from __future__ import annotations

from datetime import datetime

from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import transform as shp_transform

from app.geospatial.georeference import Georef

_GEOD = None


def geod():
    global _GEOD
    if _GEOD is None:
        from pyproj import Geod
        _GEOD = Geod(ellps="WGS84")
    return _GEOD


def pixel_to_wgs84(poly: Polygon, georef: Georef) -> Polygon:
    a, b, c, d, e, f = georef.transform

    def fwd(x, y, z=None):
        return a * x + b * y + c, d * x + e * y + f

    out = shp_transform(fwd, poly)
    if georef.crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        from pyproj import Transformer
        tr = Transformer.from_crs(georef.crs, "EPSG:4326", always_xy=True)
        out = shp_transform(tr.transform, out)
    return out


def geodesic_metrics(geom) -> dict:
    """Area (m^2) and perimeter (m) on the WGS84 ellipsoid; centroid via local equal-area proj."""
    area, perim = geod().geometry_area_perimeter(geom)
    c = centroid_lonlat(geom)
    minx, miny, maxx, maxy = geom.bounds
    return {"area_m2": abs(float(area)), "perimeter_m": float(perim),
            "centroid": {"lat": c[1], "lon": c[0]},
            "bbox": [float(minx), float(miny), float(maxx), float(maxy)]}


def local_projection(lon: float, lat: float):
    """Azimuthal equidistant projection centred on (lon, lat): metres, accurate for ~100 km."""
    from pyproj import CRS, Transformer
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    fwd = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    inv = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    return fwd, inv


def centroid_lonlat(geom) -> tuple[float, float]:
    rc = geom.representative_point()
    fwd, inv = local_projection(rc.x, rc.y)
    cen = shp_transform(fwd.transform, geom).centroid
    lon, lat = inv.transform(cen.x, cen.y)
    return float(lon), float(lat)


def shape_descriptors(geom) -> dict:
    """Shape features used by look-alike analysis (computed in metres)."""
    import math
    rc = geom.representative_point()
    fwd, _ = local_projection(rc.x, rc.y)
    g = shp_transform(fwd.transform, geom)
    mrr = g.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    edges = sorted({math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(4)})
    length, width = (edges[-1], edges[0]) if len(edges) > 1 else (edges[0], edges[0])
    return {
        "elongation": float(length / max(width, 1e-6)),
        "compactness": float(4 * math.pi * g.area / max(g.length ** 2, 1e-9)),   # 1 = circle
        "solidity": float(g.area / max(g.convex_hull.area, 1e-9)),
        "length_m": float(length), "width_m": float(width),
    }


def build_spill_record(spill_id: str, wgs_polys: list[Polygon], comp_stats: list[dict],
                       timestamp: datetime | None, detection_confidence: float) -> dict:
    """Spill representation contract (spec §11). Components are kept separate."""
    comps = []
    for i, (p, st) in enumerate(zip(wgs_polys, comp_stats)):
        m = geodesic_metrics(p)
        comps.append({"component_id": f"{spill_id}_C{i + 1:02d}", "geometry": mapping(p),
                      "area_px": st["area_px"], "mean_probability": st.get("mean_probability"), **m})
    mp = MultiPolygon(wgs_polys) if len(wgs_polys) > 1 else wgs_polys[0]
    total = geodesic_metrics(mp)
    return {
        "spill_id": spill_id,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "centroid": total["centroid"],
        "area_m2": total["area_m2"],
        "perimeter_m": total["perimeter_m"],
        "bbox": total["bbox"],
        "polygon": mapping(mp),
        "n_components": len(comps),
        "components": comps,
        "detection_confidence": detection_confidence,
        "shape": shape_descriptors(mp),
    }
