"""SYNTHETIC Sentinel-1-like SAR scene for an area of interest (used when no usable real scene exists).

Clearly labelled SYNTHETIC_DEMO everywhere (GeoTIFF tag PROVENANCE, acquisition.json, UI badge, report).
It exists so the full investigation chain can be demonstrated for any Indian-waters AOI; it is NOT an
observation. Physics-inspired construction (all in linear intensity, then the SAME normalisation as real scenes):
  * sea clutter: gamma speckle (ENL 4.4, like Sentinel-1 GRDH) x smooth wind modulation (+-1.5 dB, streaked along
    the wind direction) x a gentle incidence-angle ramp;
  * a ship-attached operational-discharge trail: a vessel at the head (bright point target with range/azimuth
    sidelobes), trail extending behind it along its course, width growing 80 -> 450 m, damping 6-8 dB, slight
    meander and drift;
  * a natural look-alike: a broad low-wind patch (-3 to -4 dB, compact, no ship) for the triage to discriminate;
  * 4-8 other ships as bright point targets.
The scenario (discharging vessel position/course/speed at image time) is saved so the synthetic AIS generator can
plant that vessel's track — keeping SAR, AIS and evidence consistent in demonstration mode.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.errors import InputImageError
from app.sar.sentinel1 import _grid, sea_mask, sos_normalize

KM_PER_DEG = 111.32


def _smooth_noise(rng, h, w, scale_px, aniso_deg=None):
    """Unit-variance smooth random field. With aniso_deg, streaks are elongated along that direction
    (Gaussian filter in the Fourier domain: no rotation seams, periodic without edge artefacts)."""
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    ky = np.fft.fftfreq(h)[:, None].astype(np.float32)
    kx = np.fft.rfftfreq(w)[None, :].astype(np.float32)
    if aniso_deg is None:
        sa = sc = scale_px
        ka, kc = kx, ky
    else:
        t = math.radians(aniso_deg)
        ka = kx * math.sin(t) + ky * math.cos(t)          # along the streak direction
        kc = kx * math.cos(t) - ky * math.sin(t)
        sa, sc = scale_px * 2.5, scale_px * 0.4
    g = np.exp(-2 * math.pi ** 2 * ((ka * sa) ** 2 + (kc * sc) ** 2)).astype(np.float32)
    out = np.fft.irfft2(np.fft.rfft2(n) * g, s=(h, w)).astype(np.float32)
    return out / (out.std() + 1e-9)


def generate_scene(aoi: list[float], timestamp: datetime, out_dir: Path, res_m: float = 20.0, seed: int = 0,
                   norm: dict | None = None, progress=None) -> dict:
    import rasterio
    from rasterio.transform import from_bounds
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    w, h = _grid(aoi, res_m)
    if progress:
        progress(f"Generating SYNTHETIC SAR scene {w}x{h} px at {res_m:g} m (no usable real Sentinel-1 coverage)")
    sea = sea_mask(aoi, w, h)
    if sea.mean() < 0.2:
        raise InputImageError("The area is mostly land — a synthetic SAR scene needs open water.", "Draw the box over sea.")
    lat_c = (aoi[1] + aoi[3]) / 2
    px_x_km = (aoi[2] - aoi[0]) * KM_PER_DEG * math.cos(math.radians(lat_c)) / w
    px_y_km = (aoi[3] - aoi[1]) * KM_PER_DEG / h
    from app.ais.synthetic_provider import PRESETS, region_preset
    preset = region_preset(aoi)
    wind_from = {"arabian_sea_west_coast": 240, "bay_of_bengal_east_coast": 220,
                 "sri_lanka_east_west_route": 250}.get(preset, 230) + rng.normal(0, 15)

    # ---- sea clutter ---------------------------------------------------------------------------------
    enl = 4.4
    speckle = rng.gamma(enl, 1 / enl, (h, w)).astype(np.float32)
    wn = _smooth_noise(rng, h, w, max(4, 3000 / res_m), aniso_deg=wind_from)
    wn2 = _smooth_noise(rng, h, w, max(3, 800 / res_m), aniso_deg=wind_from)      # finer wind rows
    wind_db = np.clip(1.0 * wn + 0.4 * wn2, -2.5, 2.5)            # +-1 dB wind cells, +-0.4 dB wind rows
    ramp_db = np.linspace(1.5, -1.5, w)[None, :]                       # incidence-angle trend across range
    base_db = -17.0 + wind_db + ramp_db

    # ---- discharge trail behind a vessel ---------------------------------------------------------------
    sea_idx = np.argwhere(ndimage.binary_erosion(sea, iterations=int(3000 / res_m)))
    if not len(sea_idx):
        raise InputImageError("Not enough open water away from the coast for a synthetic slick.", "Enlarge the box.")
    # ship somewhere in the central part of the open water, so the whole trail stays inside the scene
    d = np.hypot((sea_idx[:, 0] - h / 2) / h, (sea_idx[:, 1] - w / 2) / w)
    central = sea_idx[d < 0.22] if (d < 0.22).any() else sea_idx
    cy, cx = central[rng.integers(len(central))]
    course = (rng.choice(PRESETS[preset]["bearings"]) + 180 * rng.integers(0, 2) + rng.normal(0, 8)) % 360
    L_km = rng.uniform(7, 14)
    yy, xx = np.mgrid[0:h, 0:w]
    dx_km, dy_km = (xx - cx) * px_x_km, (cy - yy) * px_y_km             # east, north (km) from the ship
    ux, uy = math.sin(math.radians(course)), math.cos(math.radians(course))
    along = -(dx_km * ux + dy_km * uy)                                  # distance BEHIND the ship (km)
    cross = dx_km * uy - dy_km * ux
    meander = 0.15 * np.sin(along / 2.2 + rng.uniform(0, 6)) + 0.02 * along     # slight drift sideways
    width = 0.08 + 0.37 * np.clip(along / L_km, 0, 1)                   # 80 m at the ship -> 450 m at the tail
    core = np.exp(-0.5 * ((cross - meander) / (width / 2)) ** 2) * (along > 0.05) * (along < L_km)
    core *= np.clip(1.2 - along / L_km, 0, 1) ** 0.6                     # fades toward the tail
    damp_db = rng.uniform(6, 8) * core
    # ---- natural look-alike: compact low-wind patch without a ship ------------------------------------
    ly, lx = sea_idx[rng.integers(len(sea_idx))]
    r_km = rng.uniform(2, 4)
    look = np.exp(-(((xx - lx) * px_x_km) ** 2 + ((yy - ly) * px_y_km) ** 2) / (2 * r_km ** 2))
    look_db = 3.5 * look * (np.hypot((xx - lx) * px_x_km, (yy - ly) * px_y_km) > 0)
    inten = speckle * 10 ** ((base_db - damp_db - look_db) / 10)

    # ---- ships: point targets with sidelobes ---------------------------------------------------------
    def ship(r, c, db_peak=18):
        amp = 10 ** ((-17 + db_peak) / 10)
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, h), max(c - 1, 0), min(c + 2, w)
        inten[r0:r1, c0:c1] += amp
        for k in range(2, 9):                                           # range/azimuth sidelobes
            for (rr, cc) in ((r, c + k), (r, c - k), (r + k, c), (r - k, c)):
                if 0 <= rr < h and 0 <= cc < w:
                    inten[rr, cc] += amp * 0.08 / k
    ship(int(cy), int(cx), 20)
    others = []
    for _ in range(rng.integers(4, 9)):
        r, c = sea_idx[rng.integers(len(sea_idx))]
        ship(int(r), int(c), rng.uniform(12, 18))
        others.append((int(r), int(c)))

    db = (10 * np.log10(np.maximum(inten, 1e-6))).astype(np.float32)
    u8, radiometry = sos_normalize(db, sea, res_m, **(norm or {}))
    transform = from_bounds(*aoi, w, h)
    lon_of = lambda c: aoi[0] + (c + 0.5) * (aoi[2] - aoi[0]) / w
    lat_of = lambda r: aoi[3] - (r + 0.5) * (aoi[3] - aoi[1]) / h
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"SYNTHETIC_S1_{timestamp:%Y%m%dT%H%M%S}_{res_m:g}m.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=transform, compress="deflate") as ds:
        ds.write(u8, 1)
        ds.update_tags(ACQUISITION_START_TIME=timestamp.isoformat(), SOURCE_ITEM="SYNTHETIC", PLATFORM="SYNTHETIC",
                       POLARIZATION="VV", PROVENANCE="SYNTHETIC_DEMO",
                       SCALING=f"sos_matched gray_per_db={radiometry['gray_per_db']:.3f} "
                               f"speckle_std_db={radiometry['speckle_std_db']:.2f}",
                       PIXEL_SPACING_M=str(res_m),
                       WARNING="SYNTHETIC SAR scene generated for demonstration - not a satellite observation")
    np.save(out_dir / (tif.stem + "_sea.npy"), sea)
    from app.sar.sentinel1 import sos_normalize as _n
    np.save(out_dir / (tif.stem + "_anom.npy"), _n.last_anomaly)
    scenario = {"discharging_vessel": {"lat": lat_of(cy), "lon": lon_of(cx), "course_deg": float(course),
                                       "speed_kn": float(rng.uniform(10, 13)), "time": timestamp.isoformat()},
                "trail_length_km": float(L_km), "wind_from_deg": float(wind_from % 360),
                "lookalike_center": {"lat": lat_of(ly), "lon": lon_of(lx), "radius_km": float(r_km)},
                "other_ships": [{"lat": lat_of(r), "lon": lon_of(c)} for r, c in others], "preset": preset}
    return {"path": str(tif), "sea_mask": str(out_dir / (tif.stem + "_sea.npy")), "item_id": "SYNTHETIC",
            "acquired": timestamp.isoformat(), "platform": "SYNTHETIC", "orbit_state": None, "aoi": aoi,
            "resolution_m": res_m, "size": [w, h], "sea_fraction": float(sea.mean()), "radiometry": radiometry,
            "provenance": "SYNTHETIC_DEMO", "synthetic_scene": scenario,
            "warning": "SYNTHETIC SAR scene generated for demonstration — not a satellite observation."}
