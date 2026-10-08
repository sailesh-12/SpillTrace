"""SYNTHETIC environmental forcing for Indian waters — fallback when real forcing (Open-Meteo) is unavailable.

Labelled SYNTHETIC_DEMO (NetCDF global attribute `provenance`, analysis warning, report). It is a
climatology-INSPIRED analytic field, not an observation or model output:
  wind      SW monsoon (Jun-Sep) from ~240 deg 8-10 m/s; NE monsoon (Dec-Feb) from ~40 deg 5-7 m/s;
            transition months westerly/sea-breeze-like ~5 m/s; +-1.5 m/s diurnal cycle.
  currents  West India Coastal Current: equatorward (to ~170 deg) Jun-Sep, poleward (to ~350 deg) Nov-Feb;
            East India Coastal Current: poleward (to ~20 deg) Feb-Apr, equatorward (to ~200 deg) Oct-Dec;
            south of Sri Lanka: Southwest Monsoon Current eastward Jun-Sep, Northeast Monsoon Current westward
            Dec-Mar; plus a weak meander and an M2 tidal ellipse (0.10 x 0.05 m/s).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.core.provenance import Provenance

M2 = 12.42 * 3600


def _regime(bbox, month: int):
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    if month in (6, 7, 8, 9):
        wind = (240, 9.0)
    elif month in (12, 1, 2):
        wind = (40, 6.0)
    else:
        wind = (280, 5.0)
    if lat < 9.5 and 74 <= lon <= 90:                                   # south of Sri Lanka
        cur = (90, 0.45) if month in (6, 7, 8, 9) else (270, 0.35) if month in (12, 1, 2, 3) else (90, 0.15)
    elif lon < 78:                                                     # west coast
        cur = (170, 0.30) if month in (6, 7, 8, 9) else (350, 0.25) if month in (11, 12, 1, 2) else (170, 0.10)
    else:                                                              # east coast
        cur = (20, 0.40) if month in (2, 3, 4) else (200, 0.35) if month in (10, 11, 12) else (20, 0.15)
    return wind, cur


def write_synthetic_forcing(bbox, start: datetime, end: datetime, out_dir: Path, step: float = 0.1) -> tuple[Path, Path, dict]:
    import pandas as pd
    import xarray as xr
    out_dir.mkdir(parents=True, exist_ok=True)
    start = start.replace(minute=0, second=0, microsecond=0) - timedelta(hours=6)
    end = end + timedelta(hours=6)
    lats = np.round(np.arange(bbox[1] - 0.2, bbox[3] + 0.2 + 1e-9, step), 4)
    lons = np.round(np.arange(bbox[0] - 0.2, bbox[2] + 0.2 + 1e-9, step), 4)
    times = pd.date_range(start.astimezone(timezone.utc).replace(tzinfo=None), end.astimezone(timezone.utc).replace(tzinfo=None), freq="1h")
    (w_from, w_spd), (c_to, c_spd) = _regime(bbox, start.month)
    LO, LA = np.meshgrid(lons, lats)
    t_s = (times - times[0]).total_seconds().values
    cu = np.empty((len(times), len(lats), len(lons)), np.float32); cv = np.empty_like(cu)
    wu = np.empty_like(cu); wv = np.empty_like(cu)
    for k, ts in enumerate(t_s):
        meander = 0.05 * np.sin(2 * np.pi * (LA - lats[0]) / 1.5) + 0.03 * np.cos(2 * np.pi * (LO - lons[0]) / 2.0)
        cu[k] = c_spd * math.sin(math.radians(c_to)) + meander + 0.10 * math.cos(2 * math.pi * ts / M2)
        cv[k] = c_spd * math.cos(math.radians(c_to)) + 0.6 * meander + 0.05 * math.sin(2 * math.pi * ts / M2)
        spd = w_spd + 1.5 * math.sin(2 * math.pi * ts / 86400) + 0.3 * (LA - lats.mean())
        frm = math.radians(w_from + 10 * math.sin(2 * math.pi * ts / (3 * 86400)))
        wu[k], wv[k] = -spd * math.sin(frm), -spd * math.cos(frm)
    coords = {"time": times.values, "lat": ("lat", lats, {"standard_name": "latitude", "units": "degrees_north"}),
              "lon": ("lon", lons, {"standard_name": "longitude", "units": "degrees_east"})}
    label = {"provenance": Provenance.DEMO_SYNTHETIC.value, "Conventions": "CF-1.8",
             "warning": "SYNTHETIC climatology-inspired analytic field for demonstration - NOT observations"}
    key = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{start:%Y%m%d%H}_{end:%Y%m%d%H}"
    cp, wp = out_dir / f"syn_currents_{key}.nc", out_dir / f"syn_wind_{key}.nc"
    xr.Dataset({"x_sea_water_velocity": (("time", "lat", "lon"), cu, {"standard_name": "x_sea_water_velocity", "units": "m s-1"}),
                "y_sea_water_velocity": (("time", "lat", "lon"), cv, {"standard_name": "y_sea_water_velocity", "units": "m s-1"})},
               coords=coords, attrs={**label, "source": "SIH26143 synthetic Indian-waters currents"}).to_netcdf(cp)
    xr.Dataset({"x_wind": (("time", "lat", "lon"), wu, {"standard_name": "x_wind", "units": "m s-1"}),
                "y_wind": (("time", "lat", "lon"), wv, {"standard_name": "y_wind", "units": "m s-1"})},
               coords=coords, attrs={**label, "source": "SIH26143 synthetic Indian-waters 10 m wind"}).to_netcdf(wp)
    return cp, wp, {"wind_from_deg": w_from, "wind_ms": w_spd, "current_to_deg": c_to, "current_ms": c_spd}


def synthetic_provider(bbox, start, end, cache_dir: Path):
    from app.environmental.forcing import NetCDFForcingProvider
    cp, wp, regime = write_synthetic_forcing(bbox, start, end, cache_dir)
    p = NetCDFForcingProvider([cp], [wp])
    p.name = "synthetic"
    p.regime = regime
    return p
