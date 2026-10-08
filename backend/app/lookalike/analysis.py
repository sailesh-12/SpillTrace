"""Look-alike assessment (second stage after segmentation).

Dataset inspection (Phase 0): the Refined Deep-SAR Oil Spill (SOS) dataset has
BINARY labels (oil / background) only — there are no look-alike labels (low-wind
areas, biogenic films, rain cells...). We therefore do NOT train or claim a
look-alike classifier. Instead we expose transparent physical/shape indicators
from the SAR-oil literature and return OIL_LIKELY / LOOKALIKE_LIKELY / UNCERTAIN
with the reasons. A validated classifier can later implement `assess()`.
"""
from __future__ import annotations

import numpy as np


def assess(rgb_u8: np.ndarray, mask: np.ndarray, spill: dict | None, seg_conf: float,
           wind_speed_ms: float | None, cfg) -> dict:
    gray = rgb_u8[..., 0].astype(float)
    reasons, oil_pts, look_pts = [], 0.0, 0.0
    frac = float(mask.mean())
    if frac > 0.95:
        look_pts += 2
        reasons.append(f"Detection covers {100 * frac:.0f}% of the scene with no visible slick boundary: "
                       "cannot distinguish a slick interior from a low-backscatter (low-wind) area.")
    inside, outside = gray[mask], gray[~mask]
    contrast_db = None
    if inside.size and outside.size and outside.mean() > 0:
        contrast_db = float(10 * np.log10(max(outside.mean(), 1e-3) / max(inside.mean(), 1e-3)))
        if contrast_db > 3:
            oil_pts += 1; reasons.append(f"Strong damping contrast ({contrast_db:.1f} dB, 8-bit proxy).")
        elif contrast_db < 1:
            look_pts += 1; reasons.append(f"Weak damping contrast ({contrast_db:.1f} dB, 8-bit proxy).")
    if spill:
        sh = spill["shape"]
        if sh["elongation"] > 3:
            oil_pts += 1; reasons.append(f"Elongated shape (elongation {sh['elongation']:.1f}) typical of ship-discharge tracks.")
        if sh["compactness"] < 0.1:
            reasons.append(f"Complex boundary (compactness {sh['compactness']:.2f}) — seen in both weathered oil and natural films.")
    if wind_speed_ms is not None:
        if wind_speed_ms < cfg.lookalike.low_wind_ms:
            look_pts += 1.5; reasons.append(f"Low wind ({wind_speed_ms:.1f} m/s): natural low-backscatter look-alikes are common.")
        elif wind_speed_ms > cfg.lookalike.high_wind_ms:
            look_pts += 0.5; reasons.append(f"High wind ({wind_speed_ms:.1f} m/s): slicks are usually mixed down and harder to see.")
        else:
            oil_pts += 0.5; reasons.append(f"Moderate wind ({wind_speed_ms:.1f} m/s) is favourable for oil-slick detection.")
    else:
        reasons.append("Wind speed unavailable: wind-based look-alike check not performed.")
    if oil_pts - look_pts >= 1.5 and seg_conf > 0.7:
        label = "OIL_LIKELY"
    elif look_pts - oil_pts >= 1.5:
        label = "LOOKALIKE_LIKELY"
    else:
        label = "UNCERTAIN"
    return {"label": label, "segmentation_confidence": seg_conf, "oil_indicator_points": oil_pts,
            "lookalike_indicator_points": look_pts, "contrast_db_proxy": contrast_db, "reasons": reasons,
            "method": "rule-based indicators (no look-alike labels in SOS dataset; not a trained classifier)",
            "uncertainty": "Heuristic; not validated against labelled look-alikes. Treat as a triage aid."}
