"""Threat analysis and the Spill Severity & Impact Index (SSII, 0-100).

The SSII is an explainable, transparent PRIORITISATION heuristic for responders — not a damage estimate. It
combines five components, each normalised to 0-1 and reported with the evidence behind it:

  size         observed slick area (log scale, 0.05 km² -> 0, 100 km² -> 1)
  coast        proximity of the slick to the shoreline (exp(-d / 25 km))
  beaching     threat of oil reaching the shore: forward-model stranding within the forecast window, else an
               extrapolation of the forecast drift vector to the first shoreline it meets
  sensitivity  most sensitive receptor threatened (gazetteer weight 1-10), discounted by time to impact
  persistence  oil-type persistence (heavy fuel / crude > diesel) and wind-driven natural dispersion

Volume is NOT observable from SAR (thickness is unknown). A planning range is given from area x assumed
thickness (Bonn Agreement Oil Appearance Code: sheen/rainbow ~0.3-5 µm, metallic 5-50 µm) and mapped to the
NOS-DCP response tiers (Tier 1 < 700 t, Tier 2 700-10 000 t, Tier 3 > 10 000 t).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Point, shape

from app.impact.coast import KM, Coast, nearest_place
from app.impact.gazetteer import SITE_TYPE_LABEL, SITES

DEFAULTS = {
    "weights": {"size": 0.20, "coast": 0.15, "beaching": 0.25, "sensitivity": 0.25, "persistence": 0.15},
    "thickness_um": {"low": 0.3, "central": 5.0, "high": 50.0},
    "oil_density_t_m3": 0.9,
    "beach_buffer_km": 0.5,              # particle within this distance of the shore counts as reaching it
    "site_path_buffer_km": 15.0,         # site within this distance of the extrapolated drift path is on the path
    "max_extrapolation_hours": 240.0,
}


def _level(score: float) -> str:
    return "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MODERATE" if score >= 25 else "LOW"


def _tier(tonnes: float) -> str:
    return "Tier 1" if tonnes < 700 else "Tier 2" if tonnes <= 10000 else "Tier 3"


def _persistence(oil_type: str | None, wind_ms: float | None) -> tuple[float, str]:
    t = (oil_type or "").lower()
    if any(k in t for k in ("heavy", "hfo", "ifo", "bunker", "fuel oil", "crude", "sludge")):
        p, why = 1.0, "persistent oil (heavy fuel / crude)"
    elif any(k in t for k in ("diesel", "gas oil", "mgo", "marine gas", "kerosene", "light", "gasoline")):
        p, why = 0.35, "light, non-persistent oil (evaporates / disperses)"
    else:
        p, why = 0.7, "oil type unknown — assumed moderately persistent"
    if wind_ms is not None and wind_ms > 10:
        p *= 0.8
        why += f"; strong wind {wind_ms:.1f} m/s aids natural dispersion"
    elif wind_ms is not None and wind_ms < 3:
        why += f"; calm wind {wind_ms:.1f} m/s — slick stays coherent"
    return p, why


def _centroid(frame) -> tuple[float, float] | None:
    a = np.asarray(frame, float)
    if a.size == 0:
        return None
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def threat_analysis(spill: dict, forward: dict | None, fparticles: dict | None, coast: Coast, cfg: dict) -> dict:
    """Distance to coast, beaching ETA (forward model or extrapolated) and threatened sensitive sites."""
    poly = shape(spill["polygon"])
    obs = datetime.fromisoformat(spill["timestamp"])
    d_coast, near_pt = coast.distance_km(poly)
    out: dict = {"distance_to_coast_km": d_coast,
                 "nearest_shore_point": {"lon": near_pt[0], "lat": near_pt[1]} if near_pt else None,
                 "nearest_shore_place": None, "beaching": {"method": None}, "drift_vector": None, "sites": []}
    if near_pt:
        nm, dk = nearest_place(*near_pt)
        out["nearest_shore_place"] = {"name": nm, "offset_km": dk}

    # ---- beaching from the forward (OpenOil) forecast -------------------------------------------
    times = [datetime.fromisoformat(t) for t in (fparticles or {}).get("times", [])]
    frames = (fparticles or {}).get("frames", [])
    beach = None
    if frames and not coast.empty:
        buf = coast.land.buffer(cfg["beach_buffer_km"])
        shapely.prepare(buf)
        n0 = max(len(frames[0]), 1)
        for t, fr in zip(times, frames):
            a = np.asarray(fr, float)
            if not a.size:
                continue
            x, y = coast.to_km(a[:, 0], a[:, 1])
            hit = shapely.contains_xy(buf, x, y)
            frac = float(hit.sum()) / n0
            if frac >= 0.01:
                lo, la = float(a[hit, 0].mean()), float(a[hit, 1].mean())
                nm, _ = nearest_place(lo, la)
                beach = {"method": "forward_model", "eta_hours": (t - obs).total_seconds() / 3600, "eta_time": t.isoformat(),
                         "fraction_at_first_contact": frac, "landing_point": {"lon": lo, "lat": la}, "landing_place": nm,
                         "note": "OpenOil forward forecast: first time >= 1 % of particles reach the shoreline"}
                break
        # final beached fraction
        a = np.asarray(frames[-1], float)
        if a.size:
            x, y = coast.to_km(a[:, 0], a[:, 1])
            out["beached_fraction_end"] = float(shapely.contains_xy(buf, x, y).sum()) / n0

    # ---- drift vector (forecast centre motion) -> extrapolation ------------------------------------
    if len(frames) >= 2:
        c0, c1 = _centroid(frames[0]), _centroid(frames[-1])
        hrs = max((times[-1] - times[0]).total_seconds() / 3600, 1e-6)
        if c0 and c1:
            kx = KM * math.cos(math.radians(c0[1]))
            dx, dy = (c1[0] - c0[0]) * kx, (c1[1] - c0[1]) * KM
            spd = math.hypot(dx, dy) / hrs
            brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            out["drift_vector"] = {"speed_kmh": spd, "bearing_deg": brg, "from": {"lon": c0[0], "lat": c0[1]},
                                   "to": {"lon": c1[0], "lat": c1[1]}, "hours": hrs}
    dv = out["drift_vector"]
    if beach is None and dv and dv["speed_kmh"] > 0.05:
        start = dv["to"]
        max_km = dv["speed_kmh"] * cfg["max_extrapolation_hours"]
        dist, hitpt = coast.ray_to_shore(start["lon"], start["lat"], dv["bearing_deg"], max_km=min(max_km, 800))
        if dist is not None:
            eta = dv["hours"] + dist / dv["speed_kmh"]
            nm, _ = nearest_place(*hitpt)
            beach = {"method": "extrapolated", "eta_hours": eta, "eta_time": (obs + timedelta(hours=eta)).isoformat(),
                     "landing_point": {"lon": hitpt[0], "lat": hitpt[1]}, "landing_place": nm,
                     "note": "extrapolated: assumes the forecast drift direction and speed persist beyond the forecast window"}
    out["beaching"] = beach or {"method": "none", "eta_hours": None,
                                "note": "no shoreline contact in the forecast and none on the extrapolated drift path"}

    # ---- sensitive sites -------------------------------------------------------------------------
    fc_end = np.asarray(frames[-1], float) if frames else np.empty((0, 2))
    cloud = MultiPoint([tuple(p) for p in fc_end]) if len(fc_end) else None
    path = None
    if dv and dv["speed_kmh"] > 0.05:
        L = dv["speed_kmh"] * cfg["max_extrapolation_hours"]
        if beach and beach.get("method") == "extrapolated":
            L = (beach["eta_hours"] - dv["hours"]) * dv["speed_kmh"] + 20
        x0, y0 = coast.to_km(dv["from"]["lon"], dv["from"]["lat"])
        b = math.radians(dv["bearing_deg"])
        path = shapely.LineString([(float(x0), float(y0)), (float(x0) + (L + dv["speed_kmh"] * dv["hours"]) * math.sin(b),
                                                           float(y0) + (L + dv["speed_kmh"] * dv["hours"]) * math.cos(b))])
    poly_km = coast.geom_km(poly)
    cloud_km = coast.geom_km(cloud) if cloud is not None else None
    for sid, name, typ, lo, la, rad, w, note in SITES:
        sx, sy = coast.to_km(lo, la)
        sp = Point(float(sx), float(sy))
        d_now = max(poly_km.distance(sp) - rad, 0.0)
        if d_now > 400:
            continue
        d_fc = max(cloud_km.distance(sp) - rad, 0.0) if cloud_km is not None else None
        eta, reason = None, None
        if d_now <= 5:
            eta, reason = 0.0, "slick already at / inside the site"
        elif d_fc is not None and d_fc <= 5 and times:
            eta, reason = (times[-1] - obs).total_seconds() / 3600, "forecast oil reaches the site within the forecast window"
        elif path is not None:
            perp = max(path.distance(sp) - rad, 0.0)
            if perp <= cfg["site_path_buffer_km"]:
                along = path.project(sp)
                eta = along / dv["speed_kmh"]
                reason = f"site {perp:.0f} km from the extrapolated drift path"
        if eta is None and d_now > 60:
            continue
        threatened = eta is not None and eta <= cfg["max_extrapolation_hours"]
        tf = math.exp(-eta / 96) if eta is not None else 0.0
        out["sites"].append({"id": sid, "name": name, "type": typ, "type_label": SITE_TYPE_LABEL.get(typ, typ),
                             "lon": lo, "lat": la, "weight": w, "receptor": note, "distance_now_km": d_now,
                             "distance_forecast_km": d_fc, "eta_hours": eta, "threatened": threatened,
                             "reason": reason or f"{d_now:.0f} km from the slick; not on the forecast path",
                             "exposure": w / 10 * tf})
    out["sites"].sort(key=lambda s: (-s["exposure"], s["distance_now_km"]))
    out["sites"] = out["sites"][:12]
    return out


def severity_index(spill: dict, drift: dict | None, threat: dict, cfg: dict) -> dict:
    area_km2 = spill["area_m2"] / 1e6
    comp, why = {}, {}
    comp["size"] = float(np.clip(math.log10(max(area_km2, 1e-6) / 0.05) / math.log10(100 / 0.05), 0, 1))
    why["size"] = f"observed slick area {area_km2:.2f} km²"
    d = threat.get("distance_to_coast_km")
    comp["coast"] = float(math.exp(-d / 25)) if d is not None else 0.0
    why["coast"] = (f"{d:.1f} km to the nearest shoreline" + (f" (near {threat['nearest_shore_place']['name']})"
                    if threat.get("nearest_shore_place") else "")) if d is not None else "no shoreline within ~330 km"
    b = threat["beaching"]
    if b.get("eta_hours") is not None:
        base = 1.0 if b["method"] == "forward_model" else 0.85
        comp["beaching"] = float(base * math.exp(-max(b["eta_hours"] - 24, 0) / 72))
        why["beaching"] = (f"oil forecast to reach {b['landing_place']} in ~{b['eta_hours']:.0f} h "
                           f"({'forward model' if b['method'] == 'forward_model' else 'extrapolated drift'})")
    else:
        comp["beaching"], why["beaching"] = 0.0, "no shoreline contact expected on current drift"
    thr = [s for s in threat["sites"] if s["threatened"]]
    if thr:
        top = max(thr, key=lambda s: s["exposure"])
        comp["sensitivity"] = float(min(1.0, top["exposure"] + 0.05 * (len(thr) - 1)))
        why["sensitivity"] = (f"{len(thr)} sensitive receptor(s) threatened; most exposed: {top['name']} "
                              f"({top['type_label']}, weight {top['weight']}/10, ETA ~{top['eta_hours']:.0f} h)")
    else:
        comp["sensitivity"], why["sensitivity"] = 0.0, "no gazetteer receptor on the forecast / extrapolated path"
    oil = (drift or {}).get("oil", {}).get("oil_type")
    wind = ((drift or {}).get("forcing_at_spill") or {}).get("wind_speed")
    comp["persistence"], why["persistence"] = _persistence(oil, wind)
    w = cfg["weights"]
    score = 100 * sum(w[k] * comp[k] for k in w) / sum(w.values())
    th = cfg["thickness_um"]
    rho = cfg["oil_density_t_m3"]
    vol = {k: spill["area_m2"] * v * 1e-6 for k, v in th.items()}                 # m³
    ton = {k: v * rho for k, v in vol.items()}
    return {
        "name": "Spill Severity & Impact Index (SSII)",
        "score": score, "level": _level(score),
        "components": {k: {"value": comp[k], "weight": w[k], "points": 100 * w[k] * comp[k] / sum(w.values()),
                           "evidence": why[k]} for k in w},
        "volume_estimate": {"m3": vol, "tonnes": ton, "thickness_um": th,
                            "basis": "area × assumed thickness (Bonn Agreement appearance codes); SAR does not measure thickness"},
        "response_tier": {"central": _tier(ton["central"]), "range": [_tier(ton["low"]), _tier(ton["high"])],
                          "basis": "NOS-DCP tiers: Tier 1 < 700 t, Tier 2 700–10 000 t, Tier 3 > 10 000 t (planning estimate)"},
        "disclaimer": "Prioritisation heuristic for response planning; not a damage assessment. Confirm on scene.",
    }
