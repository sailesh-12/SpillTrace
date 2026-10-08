"""Evidence Correlation Score (0-100).

A transparent weighted sum of feature scores in [0, 1]. The weights are a
PROTOTYPE RANKING HEURISTIC — they are not calibrated and the score is NOT a
probability that the vessel caused the spill.
"""
from __future__ import annotations

import math

SCORE_NAME = "Evidence Correlation Score"
DISCLAIMER = ("The Evidence Correlation Score ranks candidates by how well their AIS tracks correlate with the "
              "backtracked drift in space and time. It is a prototype heuristic, not a calibrated probability, "
              "and does not establish that any vessel caused the spill.")


def feature_scores(f: dict, cfg) -> dict:
    s = cfg.scoring
    spatial = math.exp(-max(f["source_distance_km"], 0) / s.spatial_scale_km)
    dt = f["temporal"].get("time_difference_hours")
    temporal = math.exp(-dt / s.temporal_scale_hours) if dt is not None else 0.0
    trajectory = math.exp(-max(f["spatiotemporal_distance_km"], 0) / s.trajectory_scale_km)
    vtype = float(s.vessel_type_relevance.get(f["vessel_type"], s.vessel_type_relevance.get("unknown", 0.5)))
    b = f["behavior"]
    if b.get("available"):
        slow = 0.0 if b["slowdown_ratio"] is None else min(max(1 - b["slowdown_ratio"], 0), 1)
        turn = min((b["net_course_change_deg"] or 0) / 90.0, 1.0)
        stop = min(b["stop_duration_min"] / 120.0, 1.0)
        behavior = min(0.6 * slow + 0.25 * turn + 0.15 * stop, 1.0)
    else:
        behavior = 0.0
    continuity = min(f["continuity"]["max_relevant_gap_min"] / 120.0, 1.0)
    sa = f.get("sar_attached") or {}
    sar_attached = 1.0 if sa.get("matched") else 0.0
    return {"spatial": spatial, "temporal": temporal, "trajectory": trajectory, "sar_attached": sar_attached,
            "vessel_type": vtype, "behavior": behavior, "ais_continuity": continuity}


def total_score(fs: dict, weights: dict) -> float:
    wsum = sum(weights.values())
    return 100.0 * sum(weights[k] * fs.get(k, 0.0) for k in weights) / wsum
