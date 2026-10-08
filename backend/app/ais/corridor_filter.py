"""Spatio-temporal corridor matching of AIS tracks against the backtracked oil.

The backward ensemble tells us where the oil was at every hour before the
observation. A vessel is only relevant if it was close to that oil *at the same
time*. We evaluate each track on a 10-minute grid over the release window:

  d_st(t)      distance from vessel(t) to the nearest backtracked particle at t
               (particle positions linearly interpolated between hourly outputs)
  d_region     distance from the track (within the window) to the high-probability
               source region polygon (0 if inside)

Candidate criterion (configurable):
  track inside release window AND (min d_region <= max_source_distance_km
                                    OR min d_st <= corridor_buffer_km)
Everything else is filtered out with an explicit reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform


@dataclass
class Corridor:
    """Time-resolved backtracked particle cloud in local metric coordinates."""
    t_epoch: np.ndarray            # hourly output times (ascending)
    x: np.ndarray                  # (n_particles_total, n_times)
    y: np.ndarray
    fwd: object
    inv: object
    window: tuple                  # (start, end) datetimes of the release window
    regions_m: dict = field(default_factory=dict)   # level -> shapely geometry in metres
    _trees: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_ensemble(cls, ens, fwd, inv, window, regions: dict):
        order = np.argsort([t.timestamp() for t in ens.times])
        t = np.array([ens.times[i].timestamp() for i in order])
        lon = ens.lon[:, :, order].reshape(-1, len(order))
        lat = ens.lat[:, :, order].reshape(-1, len(order))
        x, y = fwd.transform(lon, lat)
        regs = {k: shp_transform(fwd.transform, v["_shape"]) for k, v in regions.items()}
        return cls(t, np.asarray(x), np.asarray(y), fwd, inv, window, regs)

    def cloud_at(self, te: float) -> np.ndarray:
        j = int(np.clip(np.searchsorted(self.t_epoch, te), 1, len(self.t_epoch) - 1))
        t0, t1 = self.t_epoch[j - 1], self.t_epoch[j]
        f = float(np.clip((te - t0) / (t1 - t0), 0, 1))
        x = self.x[:, j - 1] * (1 - f) + self.x[:, j] * f
        y = self.y[:, j - 1] * (1 - f) + self.y[:, j] * f
        ok = np.isfinite(x) & np.isfinite(y)
        return np.column_stack([x[ok], y[ok]])

    def tree_at(self, te: float):
        """KD-tree of the cloud at time te, cached (shared by all vessels evaluated on the same grid)."""
        key = round(float(te))
        if key not in self._trees:
            c = self.cloud_at(te)
            self._trees[key] = (cKDTree(c), len(c)) if len(c) else (None, 0)
        return self._trees[key]


@dataclass
class TrackMatch:
    mmsi: str
    is_candidate: bool
    reason: str
    query_t: np.ndarray = None       # epoch seconds
    vx: np.ndarray = None            # vessel metric coords at query times (NaN in gaps)
    vy: np.ndarray = None
    d_st_km: np.ndarray = None       # spatio-temporal distance to particle cloud
    near_frac: np.ndarray = None     # fraction of particles within corridor_buffer at t
    min_d_st_km: float = np.inf
    t_best_st: float | None = None
    min_d_region_km: float = np.inf
    t_best_region: float | None = None
    p_best_region: tuple | None = None


def match_track(track, cor: Corridor, cfg_ais, step_s: int = 600) -> TrackMatch:
    return match_tracks([track], cor, cfg_ais, step_s)[track.mmsi]


def match_tracks(tracks: list, cor: Corridor, cfg_ais, step_s: int = 600) -> dict:
    """Vectorised over vessels: for every 10-min step ONE KD-tree query answers all vessels at once
    (per-vessel point loops took ~42 s for 1,600 real vessels)."""
    import shapely
    w0, w1 = cor.window
    tq = np.arange(w0.timestamp(), w1.timestamp() + 1, step_s)
    max_gap = cfg_ais.gap_threshold_minutes * 60 * 4
    buf_m = cfg_ais.corridor_buffer_km * 1000
    n_v, n_t = len(tracks), len(tq)
    VX = np.full((n_v, n_t), np.nan)
    VY = np.full((n_v, n_t), np.nan)
    out: dict = {}
    for i, tr in enumerate(tracks):
        lat, lon, _ = tr.interpolate(tq, max_gap_s=max_gap)
        ok = np.isfinite(lat)
        if ok.any():
            x, y = cor.fwd.transform(lon[ok], lat[ok])
            VX[i, ok], VY[i, ok] = x, y
    D = np.full((n_v, n_t), np.nan)
    F = np.zeros((n_v, n_t))
    for j in range(n_t):
        rows = np.flatnonzero(np.isfinite(VX[:, j]))
        if not len(rows):
            continue
        tree, n = cor.tree_at(tq[j])
        if tree is None:
            continue
        pts = np.column_stack([VX[rows, j], VY[rows, j]])
        d, _ = tree.query(pts)
        D[rows, j] = d / 1000
        near = d <= buf_m
        if near.any():
            F[rows[near], j] = tree.query_ball_point(pts[near], buf_m, return_length=True) / n
    region = cor.regions_m.get("high")
    if region is not None:
        shapely.prepare(region)
    for i, tr in enumerate(tracks):
        valid = np.isfinite(VX[i])
        if not valid.any():
            te = tr.t_epoch
            when = ("after" if te.min() > w1.timestamp() else "before" if te.max() < w0.timestamp()
                    else "only across AIS gaps in")
            out[tr.mmsi] = TrackMatch(tr.mmsi, False,
                                      f"No AIS positions inside the release window (track is {when} window "
                                      f"{w0:%d %b %H:%M}–{w1:%d %b %H:%M} UTC)")
            continue
        m = TrackMatch(tr.mmsi, False, "", tq, VX[i], VY[i], D[i], F[i])
        if np.isfinite(D[i]).any():
            k = int(np.nanargmin(D[i]))
            m.min_d_st_km, m.t_best_st = float(D[i, k]), float(tq[k])
        if region is not None:
            idx = np.flatnonzero(valid)
            dists = shapely.distance(region, shapely.points(VX[i, idx], VY[i, idx])) / 1000
            k = int(np.argmin(dists))
            m.min_d_region_km, m.t_best_region = float(dists[k]), float(tq[idx[k]])
            lon_b, lat_b = cor.inv.transform(VX[i, idx[k]], VY[i, idx[k]])
            m.p_best_region = (float(lat_b), float(lon_b))
        m.is_candidate = bool(m.min_d_region_km <= cfg_ais.max_source_distance_km or
                              m.min_d_st_km <= cfg_ais.corridor_buffer_km)
        m.reason = ("Track approaches the backtracked source region/corridor inside the release window"
                    if m.is_candidate else
                    f"Closest approach {m.min_d_region_km:.1f} km to high-probability region and "
                    f"{m.min_d_st_km:.1f} km to time-matched drift corridor exceed thresholds "
                    f"({cfg_ais.max_source_distance_km} / {cfg_ais.corridor_buffer_km} km)")
        out[tr.mmsi] = m
    return out


def epoch_to_dt(te: float) -> datetime:
    return datetime.fromtimestamp(te, tz=timezone.utc)
