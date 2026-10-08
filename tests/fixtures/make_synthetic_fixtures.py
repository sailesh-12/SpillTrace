"""TEST FIXTURES ONLY — generate the synthetic twin-experiment dataset used by the automated tests.
The application never reads these files; it runs on real Sentinel-1, Open-Meteo and DMA AIS data.

Produces, all explicitly labelled SYNTHETIC_DEMO / DEMO_ASSUMED:
  data/sar/demo/demo_scene.png(.geo.json)   one real SOS SAR tile + an ASSUMED georeference
  data/forcing/demo_currents.nc             analytic current field (CF-1.8, OpenDrift-readable)
  data/forcing/demo_wind.nc                 analytic 10 m wind field
  data/ais/demo_ais.csv                     synthetic AIS for fictitious "DEMO ..." vessels
  data/ais/demo_scenario_truth.json         the planted scenario (NOT read by the pipeline)

Why a twin experiment: no real AIS exists for the SOS tiles (they are unlocated
crops). We plant a scenario whose "truth" is known, let the pipeline analyse it
blind (it only sees the image, forcing files and AIS), and check that the planted
vessel is recovered. This validates the method; it says nothing about real ships.

Usage:  python tests/fixtures/make_synthetic_fixtures.py [--image data/sar/samples/1.png]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent   # tests/fixtures -> writes tests/fixtures/data/...
OBS_TIME = datetime(2025, 6, 10, 13, 4, tzinfo=timezone.utc)       # ASSUMED acquisition time
SCENE_CENTER = (18.70, 71.80)                                       # ASSUMED (lat, lon), Arabian Sea W of Mumbai
PIXEL_M = 40.0                                                      # ASSUMED ground pixel spacing
T0, T1 = OBS_TIME - timedelta(hours=72), OBS_TIME + timedelta(hours=40)
LON = np.round(np.arange(69.5, 74.0001, 0.05), 4)
LAT = np.round(np.arange(16.5, 21.0001, 0.05), 4)
M2 = 12.42 * 3600
KM_PER_DEG_LAT = 111.32


def current_uv(lon, lat, t_sec):
    """Analytic surface current: SSE mean flow + weak meander + M2 tidal ellipse (m/s)."""
    u = 0.08 + 0.04 * np.sin(2 * np.pi * (lat - 16.5) / 2.0)
    v = -0.18 + 0.03 * np.cos(2 * np.pi * (lon - 69.5) / 3.0)
    u = u + 0.12 * np.cos(2 * np.pi * t_sec / M2)
    v = v + 0.05 * np.sin(2 * np.pi * t_sec / M2)
    return u, v


def wind_uv(lon, lat, t_sec):
    """Analytic SW-monsoon-like 10 m wind from ~240 deg, 7 +/- 2 m/s diurnal (m/s)."""
    spd = 7.0 + 2.0 * np.sin(2 * np.pi * t_sec / 86400.0) + 0.3 * (lat - 18.7)
    frm = np.radians(240 + 10 * np.sin(2 * np.pi * t_sec / (3 * 86400)))
    return -spd * np.sin(frm) + 0 * lon, -spd * np.cos(frm) + 0 * lon


def write_forcing():
    times = pd.date_range(T0.replace(tzinfo=None), T1.replace(tzinfo=None), freq="1h")
    tsec = (times - pd.Timestamp(OBS_TIME.replace(tzinfo=None))).total_seconds().values
    lon2, lat2 = np.meshgrid(LON, LAT)
    cu = np.stack([current_uv(lon2, lat2, t)[0] for t in tsec]).astype(np.float32)
    cv = np.stack([current_uv(lon2, lat2, t)[1] for t in tsec]).astype(np.float32)
    wu = np.stack([wind_uv(lon2, lat2, t)[0] for t in tsec]).astype(np.float32)
    wv = np.stack([wind_uv(lon2, lat2, t)[1] for t in tsec]).astype(np.float32)
    coords = {"time": times.values,
              "lat": ("lat", LAT, {"standard_name": "latitude", "units": "degrees_north"}),
              "lon": ("lon", LON, {"standard_name": "longitude", "units": "degrees_east"})}
    label = {"provenance": "SYNTHETIC_DEMO", "Conventions": "CF-1.8",
             "warning": "SYNTHETIC ANALYTIC FIELD FOR DEMONSTRATION ONLY - NOT REAL OBSERVATIONS OR MODEL OUTPUT"}
    out = ROOT / "data" / "forcing"
    out.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"x_sea_water_velocity": (("time", "lat", "lon"), cu, {"standard_name": "x_sea_water_velocity", "units": "m s-1"}),
                "y_sea_water_velocity": (("time", "lat", "lon"), cv, {"standard_name": "y_sea_water_velocity", "units": "m s-1"})},
               coords=coords, attrs={**label, "source": "SIH26143 synthetic demo currents"}).to_netcdf(out / "demo_currents.nc")
    xr.Dataset({"x_wind": (("time", "lat", "lon"), wu, {"standard_name": "x_wind", "units": "m s-1"}),
                "y_wind": (("time", "lat", "lon"), wv, {"standard_name": "y_wind", "units": "m s-1"})},
               coords=coords, attrs={**label, "source": "SIH26143 synthetic demo 10 m wind"}).to_netcdf(out / "demo_wind.nc")


def km_offset(lat, lon, dx_km, dy_km):
    return lat + dy_km / KM_PER_DEG_LAT, lon + dx_km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))


def backtrack_point(lat, lon, hours, wdf=0.03):
    """Simple Euler backward integration of current + wdf*wind (used ONLY to plant the scenario)."""
    dt = -300.0
    t = 0.0
    for _ in range(int(hours * 3600 / -dt)):
        u, v = current_uv(lon, lat, t)
        wu, wv = wind_uv(lon, lat, t)
        uu, vv = u + wdf * wu, v + wdf * wv
        lat, lon = km_offset(lat, lon, uu * dt / 1000, vv * dt / 1000)
        t += dt
    return lat, lon


def make_track(rng, waypoints, t_start, fix_interval_s=(120, 360), gaps=(), speed_profile=None):
    """Great-circle-ish piecewise-linear track through (lat, lon, knots) waypoints."""
    rows, t = [], t_start
    for (la0, lo0, kn0), (la1, lo1, _) in zip(waypoints[:-1], waypoints[1:]):
        dy = (la1 - la0) * KM_PER_DEG_LAT
        dx = (lo1 - lo0) * KM_PER_DEG_LAT * math.cos(math.radians((la0 + la1) / 2))
        dist_km = math.hypot(dx, dy)
        dur = dist_km / (kn0 * 1.852) * 3600
        cog = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        n = max(int(dur / np.mean(fix_interval_s)), 1)
        ts = np.sort(rng.uniform(0, dur, n))
        for s in ts:
            f = s / dur
            tt = t + timedelta(seconds=float(s))
            if any(g0 <= tt <= g1 for g0, g1 in gaps):
                continue
            rows.append((tt, la0 + f * (la1 - la0) + rng.normal(0, 0.00005),
                         lo0 + f * (lo1 - lo0) + rng.normal(0, 0.00005),
                         max(kn0 + rng.normal(0, 0.3), 0), (cog + rng.normal(0, 1.5)) % 360))
        t += timedelta(seconds=dur)
    return rows


def write_ais(scene_lat, scene_lon):
    rng = np.random.default_rng(26143)
    # Planted release: oil observed at the scene centre was, per the analytic field, here 20 h earlier.
    rel_h = 20.0
    src_lat, src_lon = backtrack_point(scene_lat, scene_lon, rel_h)
    rel_t = OBS_TIME - timedelta(hours=rel_h)

    vessels = []

    def add(mmsi, name, vtype, imo, rows, role):
        vessels.append({"mmsi": mmsi, "name": name, "type": vtype, "imo": imo, "rows": rows, "role": role})

    # 1. Planted scenario tanker: NW->SE transit, slows to ~4 kn for ~1.5 h at the planted source, 45 min AIS gap later.
    a_lat, a_lon = km_offset(src_lat, src_lon, -120, 60)
    b_lat, b_lon = km_offset(src_lat, src_lon, -4, 2)
    c_lat, c_lon = km_offset(src_lat, src_lon, 4, -2)
    d_lat, d_lon = km_offset(src_lat, src_lon, 110, -70)
    leg1_h = math.hypot(116, 58) / (13 * 1.852)
    start = rel_t - timedelta(hours=leg1_h) - timedelta(minutes=35)
    gap0 = rel_t + timedelta(hours=2.2)
    add("419000101", "DEMO TANKER ALPHA", "tanker", "9000101",
        make_track(rng, [(a_lat, a_lon, 13), (b_lat, b_lon, 4), (c_lat, c_lon, 12.5), (d_lat, d_lon, 12.5)], start,
                   gaps=[(gap0, gap0 + timedelta(minutes=45))]), "planted_source")

    # 2. Bulk carrier crossing ~6 km from the planted source ~7 h after release (plausible, weaker).
    e_lat, e_lon = km_offset(src_lat, src_lon, -80, -90)
    f_lat, f_lon = km_offset(src_lat, src_lon, 60, 70)
    t_cross = rel_t + timedelta(hours=7)
    dist_to_mid = math.hypot(80, 90) - 6
    add("419000202", "DEMO BULK BRAVO", "bulk_carrier", "9000202",
        make_track(rng, [(e_lat, e_lon, 11), km_offset(src_lat, src_lon, 4, -4) + (11,), (f_lat, f_lon, 11)],
                   t_cross - timedelta(hours=dist_to_mid / (11 * 1.852))), "decoy_near")

    # 3. "Nearest vessel trap": container ship passing right over the observed slick at observation time.
    g_lat, g_lon = km_offset(scene_lat, scene_lon, 0, -70)
    h_lat, h_lon = km_offset(scene_lat, scene_lon, 0, 70)
    add("419000303", "DEMO CONTAINER CHARLIE", "container", "9000303",
        make_track(rng, [(g_lat, g_lon, 16), (h_lat, h_lon, 16)], OBS_TIME - timedelta(hours=70 / (16 * 1.852))),
        "nearest_vessel_trap")

    # 4. Cargo ship crossing the planted source location 70 h before observation (outside release window).
    add("419000404", "DEMO CARGO DELTA", "cargo", "9000404",
        make_track(rng, [km_offset(src_lat, src_lon, -60, -30) + (12,), (src_lat, src_lon, 12),
                         km_offset(src_lat, src_lon, 60, 30) + (12,)],
                   OBS_TIME - timedelta(hours=70) - timedelta(hours=math.hypot(60, 30) / (12 * 1.852))),
        "temporal_mismatch")

    # 5-6. Fishing vessels loitering ~30 km north of the source region.
    for k, (dx, dy) in enumerate([(5, 30), (-10, 35)]):
        base = km_offset(src_lat, src_lon, dx, dy)
        wps = [(*km_offset(*base, rng.normal(0, 3), rng.normal(0, 3)), 3.5) for _ in range(14)]
        add(f"41900050{k + 1}", f"DEMO FISHING ECHO-{k + 1}", "fishing", None,
            make_track(rng, wps, OBS_TIME - timedelta(hours=60)), "background")

    # 7. Tug near the coast, far away.
    add("419000601", "DEMO TUG FOXTROT", "tug", None,
        make_track(rng, [(18.95, 72.75, 7), (18.80, 72.80, 7), (18.95, 72.78, 7)], OBS_TIME - timedelta(hours=40)),
        "background")

    # 8. Passenger ship on a distant coastal route.
    add("419000701", "DEMO PASSENGER GOLF", "passenger", "9000701",
        make_track(rng, [(20.5, 72.6, 18), (17.2, 73.1, 18)], OBS_TIME - timedelta(hours=30)), "background")

    rows = []
    for v in vessels:
        for (t, la, lo, sog, cog) in v["rows"]:
            rows.append({"mmsi": v["mmsi"], "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "lat": round(la, 6),
                         "lon": round(lo, 6), "sog": round(sog, 1), "cog": round(cog, 1),
                         "vessel_name": v["name"], "imo": v["imo"] or "", "vessel_type": v["type"]})
    df = pd.DataFrame(rows).sort_values(["mmsi", "timestamp"])
    # realistic data problems: duplicates, one invalid coordinate, one impossible jump
    df = pd.concat([df, df.sample(15, random_state=1)])
    bad = df.iloc[[40]].copy(); bad["lat"] = 123.0
    jump = df.iloc[[300]].copy(); jump["lon"] = jump["lon"] + 1.5
    df = pd.concat([df, bad, jump]).reset_index(drop=True)
    out = ROOT / "data" / "ais"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "demo_ais.csv", "w", encoding="utf-8", newline="") as f:
        f.write("# PROVENANCE: SYNTHETIC_DEMO - fictitious vessels for the SIH26143 twin experiment. NOT REAL AIS.\n")
        df.to_csv(f, index=False)
    truth = {"warning": "Planted scenario for validation. The pipeline never reads this file.",
             "planted_release_time": rel_t.isoformat(), "planted_source": {"lat": src_lat, "lon": src_lon},
             "vessels": [{k: v[k] for k in ("mmsi", "name", "type", "role")} for v in vessels]}
    (out / "demo_scenario_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def write_scene(image: Path):
    out = ROOT / "data" / "sar" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    w, h = Image.open(image).size
    lat, lon = SCENE_CENTER
    dlat = h * PIXEL_M / 1000 / KM_PER_DEG_LAT
    dlon = w * PIXEL_M / 1000 / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    shutil.copy(image, out / "demo_scene.png")
    geo = {"bbox": [lon - dlon / 2, lat - dlat / 2, lon + dlon / 2, lat + dlat / 2], "crs": "EPSG:4326",
           "timestamp": OBS_TIME.strftime("%Y-%m-%dT%H:%M:%SZ"), "provenance": "DEMO_ASSUMED",
           "sensor": "Sentinel-1 (SOS dataset tile)",
           "note": f"Image pixels are a real SOS SAR tile ({image.name}); its location, {PIXEL_M:.0f} m pixel "
                   "spacing and timestamp are DEMO ASSUMPTIONS (SOS tiles are distributed without georeferencing)."}
    (out / "demo_scene.png.geo.json").write_text(json.dumps(geo, indent=2))
    return geo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT.parents[1] / "data" / "sar" / "samples" / "1.png"))
    args = ap.parse_args()
    geo = write_scene(Path(args.image))
    write_forcing()
    truth = write_ais(*SCENE_CENTER)
    print("Scene bbox (DEMO_ASSUMED):", [round(v, 4) for v in geo["bbox"]])
    print("Planted release:", truth["planted_release_time"], truth["planted_source"])
    print("Wrote data/forcing/demo_*.nc, data/ais/demo_ais.csv, data/sar/demo/demo_scene.png")
