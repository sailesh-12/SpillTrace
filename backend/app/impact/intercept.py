"""Next-port intercept estimate for candidate vessels.

From the last AIS fixes, the vessel's course and speed are estimated and the Indian port that lies ahead on that
course (within +-40 deg) is chosen; the ETA assumes a straight-line passage at the last observed speed. This is a
heuristic to help authorities plan an inspection (Port State Control / MARPOL Annex I Oil Record Book check),
not a voyage prediction. An AIS silence at the end of the query window is flagged."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.impact.gazetteer import PORTS

KM = 111.32


def _dist_brg(lon1, lat1, lon2, lat2):
    kx = KM * math.cos(math.radians((lat1 + lat2) / 2))
    dx, dy = (lon2 - lon1) * kx, (lat2 - lat1) * KM
    return math.hypot(dx, dy), (math.degrees(math.atan2(dx, dy)) + 360) % 360


def _basin(lon: float, lat: float) -> str:
    """Sea basin, so a straight-line 'next port' never crosses the peninsula."""
    if lon >= 91.0:
        return "AN"                      # Andaman Sea / islands
    if lat < 8.0 and lon < 82.5:
        return "S"                       # south of the Cape / Sri Lanka — both coasts reachable
    return "W" if lon < 77.8 else "E"


def next_port(feature: dict, query_end: datetime | None, bbox: list | None = None) -> dict | None:
    coords = feature["geometry"]["coordinates"]
    ts = feature["properties"].get("t") or []
    if len(coords) < 2 or len(ts) != len(coords):
        return None
    lon, lat, t_last = coords[-1][0], coords[-1][1], ts[-1]
    # course/speed over the last ~1 h (at least 1 km of motion)
    j = len(coords) - 2
    while j > 0 and (t_last - ts[j] < 3600 or _dist_brg(coords[j][0], coords[j][1], lon, lat)[0] < 1.0):
        j -= 1
    d, brg = _dist_brg(coords[j][0], coords[j][1], lon, lat)
    hrs = max((t_last - ts[j]) / 3600, 1e-6)
    kn = d / hrs / 1.852
    last_time = datetime.fromtimestamp(t_last, tz=timezone.utc)
    silent = None
    inside = True
    if bbox:                                       # a vessel that sailed out of the search area is not "silent"
        w, s_, e, n = bbox
        m = 5 / KM
        inside = (w + m < lon < e - m) and (s_ + m < lat < n - m)
    if query_end is not None and inside and (query_end - last_time).total_seconds() > 3 * 3600:
        silent = f"No AIS after {last_time:%d %b %H:%M} UTC ({(query_end - last_time).total_seconds() / 3600:.0f} h before the end of the query window)"
    basin = _basin(lon, lat)
    ok = {"W": {"W", "S"}, "E": {"E", "S"}, "AN": {"AN", "E"}, "S": {"W", "E", "S"}}[basin]
    best = None
    for name, plon, plat in PORTS:
        if _basin(plon, plat) not in ok:
            continue
        dist, pb = _dist_brg(lon, lat, plon, plat)
        off = abs((pb - brg + 540) % 360 - 180)
        if off <= 40 and dist <= 2000 and (best is None or dist < best[1]):
            best = (name, dist, off, plon, plat)
    nearest = min(PORTS, key=lambda p: _dist_brg(lon, lat, p[1], p[2])[0])
    out = {"last_fix": {"lon": lon, "lat": lat, "time": last_time.isoformat()}, "course_deg": brg, "speed_kn": kn,
           "ais_silence": silent, "nearest_port": nearest[0]}
    if best and kn > 0.5:
        eta_h = best[1] / (kn * 1.852)
        out.update({"port": best[0], "distance_km": best[1], "bearing_offset_deg": best[2], "eta_hours_from_last_fix": eta_h,
                    "eta_time": (last_time + timedelta(hours=eta_h)).isoformat(),
                    "port_position": {"lon": best[3], "lat": best[4]},
                    "note": "straight-line estimate at last observed speed; confirm with port call data / VTS"})
    else:
        out.update({"port": None, "note": "no Indian port ahead on the current course (vessel may be transiting); "
                                          f"nearest Indian port: {nearest[0]}"})
    return out
