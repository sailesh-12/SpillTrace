"""AIS continuity anomaly analysis.

A gap is a period longer than the configured threshold without a position
report. It is reported as an "AIS continuity anomaly" — NEVER as evasion,
because gaps have many benign causes (coverage, receiver load, equipment,
data availability, transmission problems).
"""
from __future__ import annotations

import numpy as np

from app.ais.track_processing import haversine_km

POSSIBLE_CAUSES = ["terrestrial/satellite AIS coverage limitations", "receiver capacity or slot collisions",
                   "shipboard equipment problems", "gaps in the data provider's archive",
                   "transmission problems", "intentional switch-off (cannot be distinguished from the above)"]


def find_gaps(track, threshold_min: float) -> list[dict]:
    df = track.df
    out = []
    dts = df["dt_s"].to_numpy(float)
    for i in np.flatnonzero(np.nan_to_num(dts) / 60 > threshold_min):     # only rows that start a gap
        if i == 0:
            continue
        dt = dts[i]
        a, b = df.iloc[i - 1], df.iloc[i]
        dist = float(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]))
        out.append({
            "label": "AIS continuity anomaly",
            "gap_start": a["timestamp"].isoformat(), "gap_end": b["timestamp"].isoformat(),
            "duration_min": round(dt / 60, 1),
            "position_before": {"lat": float(a["lat"]), "lon": float(a["lon"])},
            "position_after": {"lat": float(b["lat"]), "lon": float(b["lon"])},
            "distance_across_gap_km": round(dist, 2),
            "implied_speed_kn": round(dist / 1.852 / (dt / 3600), 1),
            "_t0": a["timestamp"].timestamp(), "_t1": b["timestamp"].timestamp(),
        })
    return out


def data_quality(track) -> dict:
    dt = track.df["dt_s"].dropna()
    return {"n_positions": int(len(track.df)), "median_report_interval_s": float(dt.median()) if len(dt) else None,
            "first": track.df["timestamp"].iloc[0].isoformat(), "last": track.df["timestamp"].iloc[-1].isoformat()}
