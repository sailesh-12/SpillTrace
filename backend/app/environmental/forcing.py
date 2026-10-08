"""Environmental forcing providers (currents, wind, optional waves).

All providers share one interface so the drift engine never knows where data
came from. Every provider reports its provenance (LIVE / LOCAL / SYNTHETIC_DEMO).

  NetCDFForcingProvider   CF-convention NetCDF files (CMEMS, HYCOM, ERA5 exports,
                          or the synthetic demo files). Provenance is read from the
                          file's global attribute `provenance` if present.
  OpenMeteoForcingProvider  fetches hourly currents (marine API) and 10 m wind
                          (archive/forecast API) on a coarse grid, no API key,
                          caches to NetCDF and then behaves like NetCDF provider.

Nothing here substitutes default values when data are missing: coverage is
checked explicitly and ForcingUnavailable is raised with an actionable hint.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.core.errors import ForcingUnavailable
from app.core.provenance import Provenance

CURRENT_VARS = (("x_sea_water_velocity", "y_sea_water_velocity"), ("uo", "vo"), ("u", "v"), ("water_u", "water_v"))
WIND_VARS = (("x_wind", "y_wind"), ("u10", "v10"), ("eastward_wind", "northward_wind"))


def _np_utc(t: datetime) -> np.datetime64:
    return np.datetime64(t.astimezone(timezone.utc).replace(tzinfo=None), "s")


class EnvironmentalForcingProvider(ABC):
    name = "abstract"

    @abstractmethod
    def describe(self) -> dict: ...

    @abstractmethod
    def check_coverage(self, bbox: list[float], start: datetime, end: datetime) -> dict: ...

    @abstractmethod
    def opendrift_readers(self) -> list: ...

    @abstractmethod
    def sample(self, lat: float, lon: float, time: datetime) -> dict: ...


_READERS: dict = {}
_PACE: list = []            # (time, n_points) of recent Open-Meteo requests in this process


def _pace(n_points: int, per_minute: int = 550):
    """Client-side pacing below Open-Meteo's 600 locations/minute limit."""
    import threading
    import time
    global _PACE_LOCK
    try:
        _PACE_LOCK
    except NameError:
        _PACE_LOCK = threading.Lock()
    while True:
        with _PACE_LOCK:
            now = time.time()
            _PACE[:] = [(t, k) for t, k in _PACE if now - t < 60]
            used = sum(k for _, k in _PACE)
            if used + n_points <= per_minute or not _PACE:
                _PACE.append((now, n_points))
                return
            wait = 60 - (now - _PACE[0][0]) + 0.5
        time.sleep(max(wait, 0.5))


def probe_drift_speed(lat: float, lon: float, start: datetime, end: datetime, wind_factor: float = 0.04) -> dict:
    """Two point requests: upper bound of surface drift speed |current| + wdf*|wind| over the window.
    Used to size the forcing domain instead of assuming a fixed worst case."""
    d0, d1 = start.date().isoformat(), end.date().isoformat()
    m = OpenMeteoForcingProvider._get("https://marine-api.open-meteo.com/v1/marine", {
        "latitude": lat, "longitude": lon, "hourly": "ocean_current_velocity", "start_date": d0, "end_date": d1,
        "timezone": "GMT"})
    w = OpenMeteoForcingProvider._get(OpenMeteoForcingProvider.wind_url(end), {
        "latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "wind_speed_unit": "ms",
        "start_date": d0, "end_date": d1, "timezone": "GMT"})
    unit = m["hourly_units"]["ocean_current_velocity"]
    cur = np.array(m["hourly"]["ocean_current_velocity"], float) / (3.6 if "km" in unit else 1.0)
    wnd = np.array(w["hourly"]["wind_speed_10m"], float)
    k = min(len(cur), len(wnd))
    spd = np.nan_to_num(cur[:k]) + wind_factor * np.nan_to_num(wnd[:k])
    return {"p95_ms": float(np.percentile(spd, 95)) if k else None, "max_ms": float(spd.max()) if k else None}


def _cached_reader(path: str):
    from opendrift.readers import reader_netCDF_CF_generic
    key = (path, Path(path).stat().st_mtime)
    if key not in _READERS:
        _READERS[key] = reader_netCDF_CF_generic.Reader(path)
    return _READERS[key]


class _Field:
    """One NetCDF file holding a (u, v) vector field on a lat/lon grid."""

    def __init__(self, path: Path, kind: str):
        import xarray as xr
        self.path, self.kind = Path(path), kind
        self.ds = xr.open_dataset(self.path)
        cands = CURRENT_VARS if kind == "current" else WIND_VARS
        by_std = {self.ds[v].attrs.get("standard_name"): v for v in self.ds.data_vars}
        self.u = self.v = None
        for u, v in cands:
            if u in by_std and v in by_std:
                self.u, self.v = by_std[u], by_std[v]
                break
            if u in self.ds and v in self.ds:
                self.u, self.v = u, v
                break
        if self.u is None:
            raise ForcingUnavailable(f"{self.path.name}: no {kind} vector variables found.",
                                     f"Expected CF standard names {cands[0]}.")
        self.lat = next(c for c in ("lat", "latitude", "y") if c in self.ds.coords)
        self.lon = next(c for c in ("lon", "longitude", "x") if c in self.ds.coords)
        self.time = next(c for c in ("time", "valid_time") if c in self.ds.coords)
        self.provenance = self.ds.attrs.get("provenance", Provenance.LOCAL.value)
        self.source = self.ds.attrs.get("source", self.path.name)

    def coverage(self) -> dict:
        t = self.ds[self.time].values
        return {"file": self.path.name, "kind": self.kind, "provenance": self.provenance, "source": self.source,
                "lon": [float(self.ds[self.lon].min()), float(self.ds[self.lon].max())],
                "lat": [float(self.ds[self.lat].min()), float(self.ds[self.lat].max())],
                "time": [str(t.min())[:19] + "Z", str(t.max())[:19] + "Z"]}

    def covers(self, bbox, start, end) -> tuple[bool, str]:
        c = self.coverage()
        w, s, e, n = bbox
        if w < c["lon"][0] or e > c["lon"][1] or s < c["lat"][0] or n > c["lat"][1]:
            return False, f"{self.path.name} spatial extent {c['lon']}x{c['lat']} does not cover bbox {bbox}"
        t = self.ds[self.time].values
        if _np_utc(start) < t.min() or _np_utc(end) > t.max():
            return False, f"{self.path.name} time range {c['time']} does not cover {start:%Y-%m-%dT%H:%MZ}..{end:%Y-%m-%dT%H:%MZ}"
        return True, "ok"

    def sample(self, lat, lon, time) -> tuple[float, float]:
        sel = {self.lat: lat, self.lon: lon, self.time: _np_utc(time)}
        pt = self.ds[[self.u, self.v]].interp(sel)
        return float(pt[self.u]), float(pt[self.v])


class NetCDFForcingProvider(EnvironmentalForcingProvider):
    name = "netcdf"

    def __init__(self, current_files: list, wind_files: list, wave_files: list | None = None):
        missing = [str(p) for p in [*current_files, *wind_files] if not Path(p).exists()]
        if missing:
            raise ForcingUnavailable(
                f"Forcing file(s) not found: {missing}",
                "Download CMEMS/HYCOM currents and ERA5/GFS wind as NetCDF into data/forcing/ and list them "
                "in configs/config.yaml (environment.current_files / wind_files), or set "
                "environment.provider: open_meteo (live, no API key).")
        self.currents = [_Field(p, "current") for p in current_files]
        self.winds = [_Field(p, "wind") for p in wind_files]
        if not self.currents or not self.winds:
            raise ForcingUnavailable("Both current and wind forcing are required for drift.",
                                     "Configure environment.current_files and environment.wind_files.")
        self.wave_files = [Path(p) for p in (wave_files or []) if Path(p).exists()]

    @property
    def provenance(self) -> str:
        provs = {f.provenance for f in [*self.currents, *self.winds]}
        return provs.pop() if len(provs) == 1 else "MIXED:" + ",".join(sorted(provs))

    def describe(self) -> dict:
        return {"provider": self.name, "provenance": self.provenance,
                "currents": [f.coverage() for f in self.currents],
                "wind": [f.coverage() for f in self.winds],
                "waves": [p.name for p in self.wave_files] or "UNAVAILABLE (Stokes drift not modelled explicitly)"}

    def check_coverage(self, bbox, start, end) -> dict:
        report = {}
        for kind, fields in (("current", self.currents), ("wind", self.winds)):
            results = [f.covers(bbox, start, end) for f in fields]
            if not any(ok for ok, _ in results):
                raise ForcingUnavailable(
                    f"Backward drift unavailable because {kind} forcing does not cover the requested "
                    f"region/time. " + "; ".join(msg for _, msg in results),
                    "Obtain forcing covering the spill area plus the full backtracking window.")
            report[kind] = "covered"
        return report

    def opendrift_readers(self) -> list:
        # Reader construction costs seconds per file; readers are reusable across simulations, so cache them.
        return [_cached_reader(str(f.path)) for f in [*self.currents, *self.winds]] + \
               [_cached_reader(str(p)) for p in self.wave_files]

    def sample(self, lat, lon, time) -> dict:
        cu, cv = self.currents[0].sample(lat, lon, time)
        wu, wv = self.winds[0].sample(lat, lon, time)
        return {"current_u": cu, "current_v": cv, "current_speed": float(np.hypot(cu, cv)),
                "current_to_deg": float(np.degrees(np.arctan2(cu, cv)) % 360),
                "wind_u": wu, "wind_v": wv, "wind_speed": float(np.hypot(wu, wv)),
                "wind_from_deg": float((np.degrees(np.arctan2(-wu, -wv))) % 360),
                "provenance": self.provenance}


class OpenMeteoForcingProvider(NetCDFForcingProvider):
    """Fetch real forcing from Open-Meteo (no credentials) and cache it as CF NetCDF.

    Currents: marine-api.open-meteo.com (ocean_current_velocity/direction, ~0.08 deg global
    ocean model, direction = towards). Wind: archive-api (ERA5, ~0.25 deg) for the past, the
    forecast API for the last few days / future. Currents and wind use separate grids.
    Requests are paced and retried because the free tier is rate limited.
    """
    name = "open_meteo"

    def __init__(self, bbox, start, end, cache_dir: Path, current_step=0.1, wind_step=0.25, margin=0.2,
                 progress=None):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        # Snap the domain outward (0.25 deg, whole UTC days) so nearby investigations share one download,
        # and reuse ANY cached file that already covers the request (Open-Meteo counts every grid point).
        import math
        bbox = [math.floor(bbox[0] * 4) / 4, math.floor(bbox[1] * 4) / 4, math.ceil(bbox[2] * 4) / 4, math.ceil(bbox[3] * 4) / 4]
        start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
        key = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{start:%Y%m%d%H}_{end:%Y%m%d%H}"
        cur_p = self._covering(cache_dir, "currents", bbox, start, end) or cache_dir / f"om_currents_{key}.nc"
        wind_p = self._covering(cache_dir, "wind", bbox, start, end) or cache_dir / f"om_wind_{key}.nc"
        # currents and wind come from different Open-Meteo endpoints: fetch them concurrently
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(current_step, cur_p, "current"), (wind_step, wind_p, "wind")]
        jobs = [j for j in jobs if not j[1].exists()]
        if self.progress and not jobs:
            self.progress("Open-Meteo forcing reused from cache (0 API calls)")
        if jobs:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda j: self._fetch_field(bbox, start, end, j[0], margin, j[1], j[2]), jobs))
        super().__init__([cur_p], [wind_p])

    @staticmethod
    def _covering(cache_dir: Path, kind: str, bbox, start, end) -> Path | None:
        for f in cache_dir.glob(f"om_{kind}_*.nc"):
            try:
                w, s_, e, n, t0, t1 = f.stem.split("_")[2:]
                t0 = datetime.strptime(t0, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                t1 = datetime.strptime(t1, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                if float(w) <= bbox[0] and float(s_) <= bbox[1] and float(e) >= bbox[2] and float(n) >= bbox[3] \
                        and t0 <= start and t1 >= end:
                    return f
            except ValueError:
                continue
        return None

    @staticmethod
    def _get(url, params, retries: int = 3):
        """GET with Open-Meteo-aware rate-limit handling.

        The free tier counts every requested LOCATION as a call (600/min, 5,000/hour, 10,000/day).
        Minutely limit -> wait once for the window to reset; hourly/daily limit -> fail immediately with an
        actionable message (previously the client backed off for minutes before failing)."""
        import time
        import httpx
        n_pts = len(str(params.get("latitude", "")).split(","))
        _pace(n_pts)
        for attempt in range(retries):
            try:
                r = httpx.get(url, params=params, timeout=90)
                if r.status_code == 429:
                    reason = ""
                    try:
                        reason = r.json().get("reason", "")
                    except ValueError:
                        pass
                    if "Minutely" in reason and attempt < retries - 1:
                        time.sleep(61)
                        continue
                    raise ForcingUnavailable(
                        f"Open-Meteo free-tier limit reached: {reason or 'HTTP 429'}",
                        "The free tier allows ~5,000 grid-point requests per hour and 10,000 per day. Wait for the "
                        "next hour, re-use an area already cached, or supply CMEMS/ERA5 NetCDF files "
                        "(environment.provider: netcdf).")
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as exc:
                if attempt == retries - 1:
                    raise ForcingUnavailable(f"Open-Meteo request failed: {exc}",
                                             "Check internet access and retry.") from exc
                time.sleep(3)
        raise ForcingUnavailable("Open-Meteo request failed after retries.", "Retry later.")

    @staticmethod
    def wind_url(end: datetime) -> str:
        recent = end > datetime.now(timezone.utc) - timedelta(days=6)
        return "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"

    def _fetch_field(self, bbox, start, end, step, margin, out: Path, kind: str):
        import time
        import xarray as xr
        lats = np.round(np.arange(bbox[1] - margin, bbox[3] + margin + 1e-9, step), 4)
        lons = np.round(np.arange(bbox[0] - margin, bbox[2] + margin + 1e-9, step), 4)
        pts = [(la, lo) for la in lats for lo in lons]
        d0, d1 = (start - timedelta(days=1)).date().isoformat(), (end + timedelta(days=1)).date().isoformat()
        if kind == "current":
            url, hourly = "https://marine-api.open-meteo.com/v1/marine", "ocean_current_velocity,ocean_current_direction"
        else:
            url, hourly = self.wind_url(end), "wind_speed_10m,wind_direction_10m"
        chunk_n = 100
        chunks = [pts[i:i + chunk_n] for i in range(0, len(pts), chunk_n)]
        done = [0]

        def fetch(chunk):
            params = {"latitude": ",".join(str(p[0]) for p in chunk), "longitude": ",".join(str(p[1]) for p in chunk),
                      "start_date": d0, "end_date": d1, "timezone": "GMT", "hourly": hourly}
            if kind == "wind":
                params["wind_speed_unit"] = "ms"
            res = self._get(url, params)
            done[0] += len(chunk)
            if self.progress:
                self.progress(f"Open-Meteo {kind}: {done[0]}/{len(pts)} grid points")
            return res if isinstance(res, list) else [res]

        # 4 concurrent requests: latency (~2 s/request), not bandwidth, dominated the sequential loop.
        # Order is preserved by map(); 429 responses are backed off inside _get().
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(fetch, chunks))
        us, vs, times = [], [], None
        for res in results:
            for r in res:
                h = r["hourly"]
                if kind == "current":
                    unit = r["hourly_units"]["ocean_current_velocity"]
                    spd = np.array(h["ocean_current_velocity"], float) / (3.6 if "km" in unit else 1.0)
                    d = np.radians(np.array(h["ocean_current_direction"], float))      # towards
                    us.append(spd * np.sin(d)); vs.append(spd * np.cos(d))
                else:
                    spd = np.array(h["wind_speed_10m"], float)
                    d = np.radians(np.array(h["wind_direction_10m"], float))          # from
                    us.append(-spd * np.sin(d)); vs.append(-spd * np.cos(d))
                times = np.array(h["time"], dtype="datetime64[s]")
        shape = (len(lats), len(lons), len(times))
        grid = lambda arrs: np.nan_to_num(np.transpose(np.array(arrs).reshape(shape), (2, 0, 1))).astype(np.float32)
        un, vn = ("x_sea_water_velocity", "y_sea_water_velocity") if kind == "current" else ("x_wind", "y_wind")
        src = ("Open-Meteo Marine API (global ocean model currents)" if kind == "current"
               else f"Open-Meteo {'ERA5 archive' if 'archive' in url else 'forecast'} API (10 m wind)")
        xr.Dataset({un: (("time", "lat", "lon"), grid(us), {"standard_name": un, "units": "m s-1"}),
                    vn: (("time", "lat", "lon"), grid(vs), {"standard_name": vn, "units": "m s-1"})},
                   coords={"time": times, "lat": ("lat", lats, {"standard_name": "latitude", "units": "degrees_north"}),
                           "lon": ("lon", lons, {"standard_name": "longitude", "units": "degrees_east"})},
                   attrs={"provenance": Provenance.LIVE.value, "Conventions": "CF-1.8", "source": src,
                          "grid_step_deg": step}).to_netcdf(out)


def point_wind(lat: float, lon: float, t: datetime) -> dict:
    """10 m wind at one point/time from Open-Meteo (ERA5 archive or forecast). No key."""
    res = OpenMeteoForcingProvider._get(OpenMeteoForcingProvider.wind_url(t), {
        "latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms",
        "start_date": t.date().isoformat(), "end_date": t.date().isoformat(), "timezone": "GMT"})
    h = res["hourly"]
    i = min(range(len(h["time"])), key=lambda k: abs(datetime.fromisoformat(h["time"][k]).replace(tzinfo=timezone.utc) - t))
    return {"wind_speed_ms": h["wind_speed_10m"][i], "wind_from_deg": h["wind_direction_10m"][i], "time": h["time"][i],
            "source": "Open-Meteo " + ("ERA5 archive" if "archive" in OpenMeteoForcingProvider.wind_url(t) else "forecast"),
            "provenance": Provenance.LIVE.value}


def build_provider(cfg, bbox=None, start=None, end=None, progress=None) -> EnvironmentalForcingProvider:
    env = cfg.environment
    if env.provider == "netcdf":
        return NetCDFForcingProvider([cfg.path(p) for p in env.current_files],
                                     [cfg.path(p) for p in env.wind_files],
                                     [cfg.path(p) for p in env.get("wave_files", [])])
    if env.provider == "open_meteo":
        if bbox is None:
            raise ForcingUnavailable("Open-Meteo provider needs the spill bbox and time window.")
        om = env.open_meteo
        return OpenMeteoForcingProvider(bbox, start, end, cfg.path("data/forcing/cache"),
                                        om.get("current_step_deg", 0.1), om.get("wind_step_deg", 0.25),
                                        om.get("margin_deg", 0.2), progress)
    raise ForcingUnavailable(f"Unknown environment.provider '{env.provider}'", "Use netcdf or open_meteo.")
