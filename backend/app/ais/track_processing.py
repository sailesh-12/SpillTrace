"""AIS cleaning and trajectory construction.

Cleaning is reported, never silent:
  duplicates, invalid coordinates, missing timestamps, impossible implied speeds.
Gaps are measured, never filled with invented positions (interpolation is only
used transiently to evaluate distance at a query time and is flagged as such).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPE_CODES = [((80, 89), "tanker"), ((70, 79), "cargo"), ((60, 69), "passenger"), ((30, 30), "fishing"),
              ((31, 32), "tug"), ((52, 52), "tug")]
TEXT_TYPES = {"tanker": "tanker", "oil": "tanker", "chemical": "tanker", "cargo": "cargo", "bulk": "bulk_carrier",
              "container": "container", "tug": "tug", "passenger": "passenger", "ferry": "passenger",
              "fishing": "fishing", "towing": "tug", "hsc": "passenger", "high speed": "passenger"}


def normalize_type(raw) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() in ("", "nan", "None"):
        return "unknown"
    s = str(raw).strip().lower()
    try:
        code = int(float(s))
        for (lo, hi), name in TYPE_CODES:
            if lo <= code <= hi:
                return name
        return "other"
    except ValueError:
        pass
    for key, name in TEXT_TYPES.items():
        if key in s:
            return name
    return "other"


def to_epoch_s(ts: pd.Series) -> np.ndarray:
    """UTC datetimes -> float epoch seconds, independent of pandas datetime resolution (ns/us/s)."""
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def rdp_indices(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: indices of vertices kept (display simplification; timestamps stay aligned)."""
    n = len(x)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        seg = np.hypot(dx, dy) or 1e-12
        d = np.abs(dy * (x[a + 1:b] - x[a]) - dx * (y[a + 1:b] - y[a])) / seg
        i = int(np.argmax(d))
        if d[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack += [(a, k), (k, b)]
    return np.nonzero(keep)[0]


def track_feature(t, tol_deg: float, **props) -> dict:
    """Compact GeoJSON for the map: simplified geometry + integer epoch-second timestamps of kept vertices.
    Also splits nothing and invents nothing; gaps remain visible as long straight segments with large dt."""
    g = t.df
    lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
    ts = t.t_epoch.astype(np.int64)
    # also keep vertices bounding AIS gaps (> 30 min) so the map never interpolates across them silently
    idx = set(rdp_indices(lon, lat * 1.0, tol_deg).tolist())
    gaps = np.nonzero(np.diff(ts) > 1800)[0]
    idx.update(gaps.tolist()); idx.update((gaps + 1).tolist())
    idx = np.array(sorted(idx))
    return {"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": np.round(np.column_stack([lon[idx], lat[idx]]), 5).tolist()},
            "properties": {"mmsi": t.mmsi, "name": t.name, "vessel_type": t.vessel_type,
                           "t": ts[idx].tolist(), "n_fixes": int(len(g)), **props}}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


@dataclass
class Track:
    mmsi: str
    name: str | None
    imo: str | None
    vessel_type: str
    df: pd.DataFrame                       # timestamp, lat, lon, sog, cog, speed_kn, dist_km, dt_s
    cleaning: dict = field(default_factory=dict)

    @property
    def t_epoch(self) -> np.ndarray:
        # cached: this was recomputed thousands of times per analysis (~12 s on real traffic)
        if getattr(self, "_te", None) is None or len(self._te) != len(self.df):
            self._te = to_epoch_s(self.df["timestamp"])
        return self._te

    def interpolate(self, t_epoch: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear position at query times; NaN outside track or inside gaps > max_gap_s."""
        te = self.t_epoch
        lat = np.interp(t_epoch, te, self.df["lat"].to_numpy(), left=np.nan, right=np.nan)
        lon = np.interp(t_epoch, te, self.df["lon"].to_numpy(), left=np.nan, right=np.nan)
        j = np.clip(np.searchsorted(te, t_epoch), 1, len(te) - 1)
        in_gap = (te[j] - te[j - 1]) > max_gap_s
        lat[in_gap] = np.nan
        lon[in_gap] = np.nan
        return lat, lon, in_gap


def _drop_speed_spikes(g: pd.DataFrame, max_speed_kn: float, max_passes: int = 6) -> tuple[pd.DataFrame, int]:
    """Vectorised removal of fixes whose implied speed from the previous AND to the next fix is
    impossible (position spikes). First/last fixes only need their single neighbour to be impossible."""
    dropped = 0
    for _ in range(max_passes):
        if len(g) < 3:
            break
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        t = to_epoch_s(g["timestamp"])
        d = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = np.diff(t)
        v = np.where(dt > 0, d / 1.852 / np.maximum(dt, 1e-9) * 3600, 0.0)
        bad_seg = v > max_speed_kn
        spike = np.r_[False, bad_seg[:-1] & bad_seg[1:], False]      # interior: both neighbours impossible
        if not spike.any():
            # endpoints only once the interior is clean (otherwise a spike next to an end would take it too)
            spike = np.zeros(len(g), bool)
            spike[0], spike[-1] = bad_seg[0], bad_seg[-1]
            if not spike.any():
                break
        dropped += int(spike.sum())
        g = g[~spike].reset_index(drop=True)
    return g, dropped


def clean_and_build(df: pd.DataFrame, max_speed_kn: float = 45.0) -> tuple[list[Track], dict]:
    report = {"input_rows": int(len(df))}
    df = df.copy()
    # providers may hand over strings/objects/naive datetimes: normalise to tz-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["timestamp"])
    report["missing_timestamp"] = n0 - len(df)
    n0 = len(df)
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180) & df["lat"].notna() & df["lon"].notna()]
    df = df[~((df["lat"] == 0) & (df["lon"] == 0))]
    report["invalid_coordinates"] = n0 - len(df)
    n0 = len(df)
    df = df.drop_duplicates(subset=["mmsi", "timestamp", "lat", "lon"])
    df = df.sort_values(["mmsi", "timestamp"]).drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
    report["duplicates"] = n0 - len(df)

    # ---- vectorised over ALL vessels at once (a per-vessel pandas loop took ~34 s for 1,600 vessels) ----
    df = df.reset_index(drop=True)
    mm = df["mmsi"].astype(str).to_numpy()
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    t = to_epoch_s(df["timestamp"])
    keep = np.ones(len(df), bool)
    impossible = 0
    for _ in range(8):                                   # iterative spike removal (see _drop_speed_spikes)
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            break
        same = mm[idx[1:]] == mm[idx[:-1]]
        d = haversine_km(lat[idx[:-1]], lon[idx[:-1]], lat[idx[1:]], lon[idx[1:]])
        dt = np.diff(t[idx])
        bad = same & (dt > 0) & (d / 1.852 / np.maximum(dt, 1e-9) * 3600 > max_speed_kn)
        bad_in, bad_out = np.r_[False, bad], np.r_[bad, False]
        spike = bad_in & bad_out                         # interior spikes first
        if not spike.any():
            first_of = np.r_[True, ~same]               # endpoints only once interiors are clean
            last_of = np.r_[~same, True]
            spike = (first_of & bad_out) | (last_of & bad_in)
            if not spike.any():
                break
        impossible += int(spike.sum())
        keep[idx[spike]] = False
    df = df[keep].reset_index(drop=True)
    mm, lat, lon, t = mm[keep], lat[keep], lon[keep], t[keep]
    start = np.r_[True, mm[1:] != mm[:-1]]
    dt = np.r_[np.nan, np.diff(t)]
    dist = np.r_[np.nan, haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    course = np.r_[np.nan, bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    dt[start], dist[start], course[start] = np.nan, np.nan, np.nan
    df["dt_s"], df["dist_km"], df["course_deg"] = dt, dist, course
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_kn"] = np.where(dt > 0, dist / 1.852 / (dt / 3600), np.nan)   # derived from positions
    static = {}
    for c in ("vessel_name", "imo", "vessel_type"):
        col = df[c]
        txt = col.astype("string").str.strip()
        col = txt.where(col.notna() & (txt != "") & (txt != "nan"))
        first = col.groupby(mm).first()
        static[c] = first.astype(object).where(first.notna(), None)      # pd.NA -> None for downstream code
    bounds = np.r_[np.flatnonzero(start), len(df)]
    tracks = []
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        if b_ - a_ < 2:
            continue
        m = mm[a_]
        tracks.append(Track(mmsi=m, name=static["vessel_name"].get(m), imo=static["imo"].get(m),
                            vessel_type=normalize_type(static["vessel_type"].get(m)),
                            df=df.iloc[a_:b_].reset_index(drop=True)))
    report["impossible_speed"] = impossible
    report["vessels"] = len(tracks)
    report["positions_used"] = int(sum(len(t.df) for t in tracks))
    return tracks, report
