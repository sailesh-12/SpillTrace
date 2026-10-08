"""Candidate evidence features (all transparent, all in physical units)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from app.ais.corridor_filter import Corridor, TrackMatch, epoch_to_dt
from app.ais.gaps import find_gaps
from app.ais.track_processing import Track, haversine_km


def _angle_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def temporal_feature(m: TrackMatch, cor: Corridor, radius_m: float = 1500.0) -> dict:
    """Time difference between the vessel's presence at its best-matching location and the
    time at which the backtracked oil occupied that location.

    Anchored at the best spatio-temporal match (not at region entry: the region is large, so
    many track points are 'inside' it while the oil was elsewhere at those times)."""
    t_anchor = m.t_best_st if m.t_best_st is not None else m.t_best_region
    if t_anchor is None:
        return {"time_difference_hours": None}
    i = int(np.nanargmin(np.abs(m.query_t - t_anchor)))
    P = np.array([m.vx[i], m.vy[i]])
    w0, w1 = cor.window[0].timestamp(), cor.window[1].timestamp()
    ts = np.arange(w0, w1 + 1, 600.0)
    # oil density at P over time: fraction of ensemble particles within radius_m of P
    counts = np.zeros(len(ts))
    for k, t in enumerate(ts):
        tree, n = cor.tree_at(t)
        if tree is not None:
            counts[k] = len(tree.query_ball_point(P, radius_m)) / n
    if counts.sum() <= 0:
        return {"time_difference_hours": None, "vessel_time": epoch_to_dt(t_anchor).isoformat(),
                "note": f"no backtracked oil within {radius_m / 1000:.1f} km of the vessel's best-match point"}
    t_oil = float((ts * counts).sum() / counts.sum())
    t_sd = float(np.sqrt(((ts - t_oil) ** 2 * counts).sum() / counts.sum()))
    return {"time_difference_hours": abs(t_anchor - t_oil) / 3600,
            "vessel_time": epoch_to_dt(t_anchor).isoformat(),
            "oil_time_at_location": epoch_to_dt(t_oil).isoformat(),
            "oil_time_spread_hours": t_sd / 3600,
            "peak_particle_fraction": float(counts.max()),
            "method": f"density-weighted time of backtracked oil within {radius_m / 1000:.1f} km of vessel position"}


def behavior_features(track: Track, t_center: float, center_latlon: tuple, near_km: float) -> dict:
    df = track.df
    te = track.t_epoch
    d_center = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(), center_latlon[0], center_latlon[1])
    near = (d_center <= near_km) & (np.abs(te - t_center) <= 6 * 3600)
    if not near.any():
        near = np.abs(te - t_center) <= 1800
    if not near.any():
        return {"available": False}
    t_in, t_out = te[near].min(), te[near].max()
    before = (te < t_in) & (te >= t_in - 3 * 3600)
    after = (te > t_out) & (te <= t_out + 3 * 3600)
    spd = df["speed_kn"].to_numpy()
    crs = df["course_deg"].to_numpy()
    mean = lambda mask: float(np.nanmean(spd[mask])) if mask.any() and np.isfinite(spd[mask]).any() else None
    s_b, s_n, s_a = mean(before), mean(near), mean(after)
    ref = np.nanmean([v for v in (s_b, s_a) if v is not None]) if (s_b or s_a) else None
    slow_ratio = (s_n / ref) if (ref and s_n is not None and ref > 0.5) else None
    cn = crs[near | (np.abs(te - t_center) <= 3600)]
    cn = cn[np.isfinite(cn)]
    course_change = float(np.max(_angle_diff(cn[1:], cn[:-1]))) if len(cn) > 1 else 0.0
    net_course_change = None
    cb, ca = crs[before & np.isfinite(crs)], crs[after & np.isfinite(crs)]
    if len(cb) and len(ca):
        net_course_change = float(_angle_diff(np.median(ca), np.median(cb)))
    slow = near & (spd < 2.0)
    stop_min = float(np.nansum(df["dt_s"].to_numpy()[slow]) / 60) if slow.any() else 0.0
    return {"available": True, "speed_before_kn": s_b, "speed_near_kn": s_n, "speed_after_kn": s_a,
            "slowdown_ratio": slow_ratio, "max_course_change_deg": course_change,
            "net_course_change_deg": net_course_change, "stop_duration_min": stop_min,
            "near_window": [epoch_to_dt(t_in).isoformat(), epoch_to_dt(t_out).isoformat()]}


def continuity_features(track: Track, t_center: float, threshold_min: float, window_h: float = 6.0) -> dict:
    gaps = find_gaps(track, threshold_min)
    rel = [g for g in gaps if g["_t1"] >= t_center - window_h * 3600 and g["_t0"] <= t_center + window_h * 3600]
    return {"all_gaps": [{k: v for k, v in g.items() if not k.startswith("_")} for g in gaps],
            "relevant_gaps": [{k: v for k, v in g.items() if not k.startswith("_")} for g in rel],
            "max_relevant_gap_min": max((g["duration_min"] for g in rel), default=0.0)}


def compute_features(track: Track, m: TrackMatch, cor: Corridor, cfg) -> dict:
    a = cfg.ais
    t_center = m.t_best_st if m.t_best_st is not None else m.t_best_region
    if t_center is None:                         # e.g. vessel kept only because it is SAR-attached
        t_center = float(cor.window[1].timestamp())
    # nearest query time with a valid (non-gap) vessel position
    i = int(np.argmin(np.where(np.isfinite(m.vx), np.abs(m.query_t - t_center), np.inf)))
    lon_c, lat_c = cor.inv.transform(m.vx[i], m.vy[i])
    return {
        "source_distance_km": m.min_d_region_km,
        "closest_approach_time": epoch_to_dt(m.t_best_region).isoformat() if m.t_best_region else None,
        "spatiotemporal_distance_km": m.min_d_st_km,
        "best_match_time": epoch_to_dt(m.t_best_st).isoformat() if m.t_best_st else None,
        "best_match_position": {"lat": float(lat_c), "lon": float(lon_c)},
        "particle_fraction_within_buffer": float(np.nanmax(m.near_frac)) if m.near_frac is not None else 0.0,
        "temporal": temporal_feature(m, cor, cfg.drift.kde_bandwidth_m),
        "vessel_type": track.vessel_type,
        "behavior": behavior_features(track, t_center, (float(lat_c), float(lon_c)), a.near_region_km),
        "continuity": continuity_features(track, t_center, a.gap_threshold_minutes),
    }
