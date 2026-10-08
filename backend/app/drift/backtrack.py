"""Backward (hindcast) and forward drift ensembles for a detected spill.

Particles are sampled uniformly throughout the spill polygon(s) — not just the
centroid — because every part of the slick carries information about where it
has been. Each ensemble member perturbs the wind-drift factor, the seed
positions and OpenDrift's stochastic wind/current uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform

from app.drift import opendrift_runner
from app.geospatial.polygon import local_projection


def sample_points_in_polygon(geom_geojson: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform (area-weighted) random points inside a (Multi)Polygon, sampled in metres."""
    geom = shape(geom_geojson)
    rc = geom.representative_point()
    fwd, inv = local_projection(rc.x, rc.y)
    g = shp_transform(fwd.transform, geom)
    from shapely.prepared import prep
    pg = prep(g)
    minx, miny, maxx, maxy = g.bounds
    xs, ys = [], []
    while len(xs) < n:
        cx = rng.uniform(minx, maxx, n * 4)
        cy = rng.uniform(miny, maxy, n * 4)
        for x, y in zip(cx, cy):
            if pg.contains(Point(x, y)):
                xs.append(x); ys.append(y)
                if len(xs) == n:
                    break
    lon, lat = inv.transform(np.array(xs), np.array(ys))
    return np.asarray(lon), np.asarray(lat)


@dataclass
class EnsembleResult:
    direction: str                         # backward | forward
    start_time: datetime
    times: list[datetime]                  # common hourly output times
    lon: np.ndarray                        # (members, particles, times)
    lat: np.ndarray
    status: list                           # per member array of final status
    members: list[dict] = field(default_factory=list)
    model: str = ""
    config: dict = field(default_factory=dict)

    def positions_at(self, t: datetime) -> tuple[np.ndarray, np.ndarray]:
        """All-member particle positions at the output time closest to t (flattened)."""
        idx = int(np.argmin([abs((tt - t).total_seconds()) for tt in self.times]))
        return self.lon[:, :, idx].ravel(), self.lat[:, :, idx].ravel()


def run_ensemble(provider, spill_geojson: dict, start: datetime, hours: float, dcfg, direction: str = "backward",
                 n_particles: int | None = None, members: int | None = None, seed: int = 26143,
                 progress=None) -> EnsembleResult:
    rng = np.random.default_rng(seed)
    pert = dict(dcfg.perturbation)
    n = int(n_particles or dcfg.particles)
    m = int(members or dcfg.ensemble_runs)
    step = int(dcfg.timestep_seconds)
    step = -abs(step) if direction == "backward" else abs(step)
    wlo, whi = pert.get("wind_drift_factor_range", [0.03, 0.03])
    oil = dcfg.oil_model_assumption if str(dcfg.oil_type).upper() == "UNKNOWN" else dcfg.oil_type
    # All members run in ONE OpenDrift simulation (per-particle wind drift factor): model, reader, landmask
    # and oil-database initialisation dominated the cost, so separate runs per member were ~6x slower.
    lons, lats, wdfs, meta = [], [], [], []
    for k in range(m):
        wdf = wlo + (whi - wlo) * (k / max(m - 1, 1))       # member-level systematic wind drift factor
        lon, lat = sample_points_in_polygon(spill_geojson, n, rng)
        lons.append(lon); lats.append(lat); wdfs.append(np.full(n, wdf))
        meta.append({"member": k, "wind_drift_factor": round(wdf, 4), "particles": n})
    if progress:
        progress(0, m)
    r = opendrift_runner.run(dcfg.model, provider.opendrift_readers(), np.concatenate(lons), np.concatenate(lats),
                             start, hours, step, int(dcfg.output_timestep_seconds), np.concatenate(wdfs), oil, pert,
                             weathering=(direction == "forward"))
    for mm in meta:
        mm["model"] = r.model
    if progress:
        progress(m, m)
    nt = len(r.times)
    return EnsembleResult(
        direction=direction, start_time=start, times=r.times,
        lon=r.lon[:, :nt].reshape(m, n, nt), lat=r.lat[:, :nt].reshape(m, n, nt),
        status=list(r.status.reshape(m, n)), members=meta, model=r.model, config=r.config)


def fill_stranded(ens: EnsembleResult) -> EnsembleResult:
    """Carry the last valid position forward for deactivated (e.g. stranded) particles."""
    for arr in (ens.lon, ens.lat):
        for mi in range(arr.shape[0]):
            a = arr[mi]
            for j in range(1, a.shape[1]):
                bad = np.isnan(a[:, j])
                a[bad, j] = a[bad, j - 1]
    return ens


def backward_times(obs: datetime, offsets_h: list[float]) -> list[datetime]:
    return [obs - timedelta(hours=h) for h in offsets_h]
