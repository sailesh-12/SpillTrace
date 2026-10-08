"""Multi-temporal spill evolution + drift validation.

Two observations are only treated as the same spill after an explicit,
configurable association test (max centroid distance per hour elapsed, or
polygon overlap). If a forward-drift prediction from the earlier scene is
available, the observed displacement is compared with the modelled one.
"""
from __future__ import annotations

from datetime import datetime

from shapely.geometry import shape

from app.ais.track_processing import bearing_deg, haversine_km
from app.geospatial.polygon import geod


def associate(a: dict, b: dict, max_speed_kmh: float = 3.0, min_hours: float = 0.0) -> dict:
    ta, tb = datetime.fromisoformat(a["timestamp"]), datetime.fromisoformat(b["timestamp"])
    dt_h = (tb - ta).total_seconds() / 3600
    d = float(haversine_km(a["centroid"]["lat"], a["centroid"]["lon"], b["centroid"]["lat"], b["centroid"]["lon"]))
    overlap = shape(a["polygon"]).intersects(shape(b["polygon"]))
    ok = dt_h > min_hours and (overlap or d <= max_speed_kmh * max(dt_h, 1e-6))
    return {"associated": bool(ok), "hours_between": dt_h, "centroid_distance_km": d, "polygons_overlap": bool(overlap),
            "criterion": f"overlap OR centroid displacement <= {max_speed_kmh} km/h × Δt"}


def compare(a: dict, b: dict, predicted_centroid: dict | None = None, **assoc_kw) -> dict:
    asc = associate(a, b, **assoc_kw)
    out = {"association": asc}
    if not asc["associated"]:
        out["note"] = "Observations were not associated as the same spill; evolution metrics not computed."
        return out
    pa, pb = shape(a["polygon"]), shape(b["polygon"])
    inter, union = pa.intersection(pb), pa.union(pb)
    ia = abs(geod().geometry_area_perimeter(inter)[0]) if not inter.is_empty else 0.0
    ua = abs(geod().geometry_area_perimeter(union)[0])
    dt_h = asc["hours_between"]
    out.update({
        "centroid_movement_km": asc["centroid_distance_km"],
        "movement_direction_deg": float(bearing_deg(a["centroid"]["lat"], a["centroid"]["lon"],
                                                    b["centroid"]["lat"], b["centroid"]["lon"])),
        "observed_speed_kmh": asc["centroid_distance_km"] / dt_h if dt_h else None,
        "area_change_m2": b["area_m2"] - a["area_m2"],
        "spreading_rate_m2_per_h": (b["area_m2"] - a["area_m2"]) / dt_h if dt_h else None,
        "polygon_iou": ia / ua if ua else 0.0,
    })
    if predicted_centroid:
        err = float(haversine_km(predicted_centroid["lat"], predicted_centroid["lon"], b["centroid"]["lat"], b["centroid"]["lon"]))
        pd_ = float(haversine_km(a["centroid"]["lat"], a["centroid"]["lon"], predicted_centroid["lat"], predicted_centroid["lon"]))
        out["drift_validation"] = {"predicted_centroid": predicted_centroid, "modelled_displacement_km": pd_,
                                   "observed_displacement_km": asc["centroid_distance_km"], "drift_validation_error_km": err}
    return out
