"""Per-slick triage: rank detected dark features by oil-vs-look-alike indicators.

Real open-sea SAR scenes contain many oil look-alikes (low-wind areas, biogenic films,
current fronts, rain cells). The segmentation model was trained on the SOS dataset,
which has no look-alike negatives, so on real scenes it flags many of them. Operational
services (e.g. EMSA CleanSeaNet) handle this with analyst verification; we support the
analyst with transparent indicators per detected component:

  ship_attached   a SAR bright point target within `ship_link_km` of the slick  (strong oil cue:
                  classic signature of an operational discharge from a moving ship)
  elongation      long, narrow shapes are typical of ship discharges
  contrast_db     damping contrast inside vs. a surrounding sea ring (dB, from the scene stretch)
  wind            10 m wind at acquisition time (ERA5): < ~3 m/s -> natural look-alikes common;
                  > ~12 m/s -> slicks are usually mixed down
  filament_net    low solidity + low elongation network patterns typical of natural films

Nothing here is a trained classifier; labels are OIL_LIKELY / LOOKALIKE_LIKELY / UNCERTAIN.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def component_contrast_db(gray: np.ndarray, comp: dict, sea: np.ndarray, db_per_level: float, pad: int = 12) -> float | None:
    """Damping contrast inside vs. a 3-10 px surrounding sea ring, computed on a padded crop."""
    r, c = comp.get("offset", (0, 0))
    m = comp["mask"]
    h, w = m.shape
    r0, c0 = max(r - pad, 0), max(c - pad, 0)
    r1, c1 = min(r + h + pad, gray.shape[0]), min(c + w + pad, gray.shape[1])
    local = np.zeros((r1 - r0, c1 - c0), bool)
    local[r - r0:r - r0 + h, c - c0:c - c0 + w] = m
    g, s = gray[r0:r1, c0:c1], sea[r0:r1, c0:c1]
    ring = ndimage.binary_dilation(local, iterations=10) & ~ndimage.binary_dilation(local, iterations=3) & s
    if ring.sum() < 20 or local.sum() < 5:
        return None
    return float((g[ring].mean() - g[local].mean()) * db_per_level)


def trail_end_distance_km(geom, lon: float, lat: float) -> tuple[float, float]:
    """(distance from point to the nearest END of the slick's major axis, distance to the slick) in km."""
    from shapely.geometry import LineString, Point
    from shapely.ops import transform as tr
    from app.geospatial.polygon import local_projection
    fwd, _ = local_projection(lon, lat)
    g = tr(fwd.transform, geom)
    p = Point(0.0, 0.0)
    xs, ys = g.minimum_rotated_rectangle.exterior.coords.xy
    corners = list(zip(xs[:4], ys[:4]))
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    short = sorted(edges, key=lambda e: LineString(e).length)[:2]
    ends = [Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in short]
    return min(p.distance(e) for e in ends) / 1000, g.distance(p) / 1000


def triage_components(spill: dict, comp_masks: list[dict], gray: np.ndarray, sea: np.ndarray,
                      db_per_level: float, ships: list[dict], wind: dict | None, cfg) -> list[dict]:
    from app.ais.track_processing import haversine_km
    from app.geospatial.polygon import shape_descriptors
    from shapely.geometry import Point, shape

    t = cfg.lookalike
    link_km = t.get("ship_link_km", 1.0)
    ws = wind.get("wind_speed_ms") if wind else None
    out = []
    for comp, mask in zip(spill["components"], comp_masks):
        geom = shape(comp["geometry"])
        sd = shape_descriptors(geom)
        contrast = component_contrast_db(gray, mask, sea, db_per_level)
        # nearest SAR point target to the slick, and to the ends of its major axis
        near_ship, near_km, end_km = None, None, None
        minx, miny, maxx, maxy = geom.bounds
        pad_deg = 3 * link_km / 111.32 / max(np.cos(np.radians(miny)), 0.2)
        for s in ships:
            if "lat" not in s or not (minx - pad_deg <= s["lon"] <= maxx + pad_deg and miny - pad_deg <= s["lat"] <= maxy + pad_deg):
                continue
            e_km, d_km = trail_end_distance_km(geom, s["lon"], s["lat"])
            if near_km is None or d_km < near_km:
                near_ship, near_km, end_km = s, float(d_km), float(e_km)
        oil, look, reasons = 0.0, 0.0, []
        trail = sd["elongation"] >= t.get("elongated", 4.0)
        if near_km is not None and near_km <= link_km:
            if trail and end_km is not None and end_km <= max(link_km, 0.15 * sd["length_m"] / 1000):
                oil += 2.0
                reasons.append(f"Ship-trail pattern: SAR point target (possible vessel) at the end of a "
                               f"{sd['length_m'] / 1000:.1f} km narrow trail — classic operational-discharge signature "
                               "(a vessel's turbulent wake can look similar at low wind).")
            else:
                reasons.append(f"SAR point target {near_km:.2f} km away, but the dark feature is not a trail ending at "
                               "the target — more consistent with a vessel inside a low-backscatter area than a discharge.")
        if sd["elongation"] >= t.get("elongated", 4.0):
            oil += 1.0
            reasons.append(f"Elongated (length/width {sd['elongation']:.1f}, length {sd['length_m'] / 1000:.1f} km).")
        if contrast is not None:
            if contrast >= t.get("strong_contrast_db", 2.0):
                oil += 1.0
                reasons.append(f"Strong damping contrast {contrast:.1f} dB vs. surrounding sea.")
            elif contrast < t.get("weak_contrast_db", 1.0):
                look += 1.0
                reasons.append(f"Weak damping contrast {contrast:.1f} dB.")
        if sd["solidity"] < 0.3 and sd["elongation"] < 3:
            look += 1.0
            reasons.append(f"Filament-network shape (solidity {sd['solidity']:.2f}) typical of natural films/fronts.")
        if comp["area_m2"] > 50e6 and sd["elongation"] < 3:
            look += 1.0
            reasons.append(f"Very large, compact dark area ({comp['area_m2'] / 1e6:.0f} km²) — typical of low-wind zones.")
        if ws is not None:
            if ws < t.low_wind_ms:
                look += 1.5
                reasons.append(f"Low wind at acquisition ({ws:.1f} m/s): natural look-alikes are common.")
            elif ws > t.high_wind_ms:
                look += 0.5
                reasons.append(f"High wind ({ws:.1f} m/s): oil slicks are usually mixed down.")
            else:
                oil += 0.5
                reasons.append(f"Wind {ws:.1f} m/s is within the range where oil slicks are detectable.")
        p = comp.get("mean_probability") or 0.0
        label = ("OIL_LIKELY" if oil - look >= 1.5 else "LOOKALIKE_LIKELY" if look - oil >= 1.5 else "UNCERTAIN")
        out.append({"component_id": comp["component_id"], "label": label, "triage_score": round(oil - look + p, 3),
                    "oil_points": oil, "lookalike_points": look, "mean_probability": p,
                    "area_km2": comp["area_m2"] / 1e6, "centroid": comp["centroid"], "elongation": sd["elongation"],
                    "length_km": sd["length_m"] / 1000, "solidity": sd["solidity"], "contrast_db": contrast,
                    "nearest_sar_target_km": near_km, "ship_trail": any("Ship-trail pattern" in x for x in reasons),
                    "reasons": reasons})
    out.sort(key=lambda r: r["triage_score"], reverse=True)
    for i, r in enumerate(out, 1):
        r["triage_rank"] = i
    return out
